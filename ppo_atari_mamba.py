import argparse
import random
import time
from collections import deque

import gymnasium as gym
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from gae import compute_advantages
from env_utils import make_atari_env
from exp_utils import add_common_args, setup_logging, finish_logging

# Import Mamba
from mamba_ssm import Mamba

def parse_args():
    parser = argparse.ArgumentParser()
    add_common_args(parser)

    parser.add_argument("--seq-len", type=int, default=4,
        help="sequence length for Mamba model")
    parser.add_argument("--hidden-dim", type=int, default=128,
        help="Size of the hidden dimension for the Mamba model")
    parser.add_argument("--reconstruction-coef", type=float, default=0.0,
        help="Coefficient for optional observation reconstruction loss (0 disables it)")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, args):
        super(Agent, self).__init__()

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )

        if args.reconstruction_coef > 0:
            self.transposed_cnn = nn.Sequential(
                layer_init(nn.Linear(args.hidden_dim, 64 * 7 * 7)),
                nn.ReLU(),
                nn.Unflatten(1, (64, 7, 7)),
                layer_init(nn.ConvTranspose2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                layer_init(nn.ConvTranspose2d(64, 32, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.ConvTranspose2d(32, 1, 8, stride=4)),
                nn.Sigmoid(),
            )
        
        self.input_proj = layer_init(nn.Linear(512, args.hidden_dim))
        self.mamba = Mamba(
            d_model=args.hidden_dim,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        self.actor = layer_init(nn.Linear(args.hidden_dim, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(args.hidden_dim, 1), std=1)

    def get_states(self, x):
        # x has shape (batch_size, seq_len, channels, height, width)
        batch_size, seq_len = x.shape[0], x.shape[1]
        x = x.contiguous().view(batch_size * seq_len, *x.shape[2:])  # Flatten sequence dimension
        hidden = self.network(x / 255.0)  # Normalize pixel values
        hidden = self.input_proj(hidden)
        hidden = hidden.view(batch_size, seq_len, -1)
        output = self.mamba(hidden)
        return output

    def get_value(self, x):
        hidden = self.get_states(x)
        last_hidden = hidden[:, -1, :]
        return self.critic(last_hidden)

    def get_action_and_value(self, x, action=None):
        hidden = self.get_states(x)
        last_hidden = hidden[:, -1, :]
        self.x = last_hidden
        logits = self.actor(last_hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        else:
            action = action.view(-1)
        return action, probs.log_prob(action), probs.entropy(), self.critic(last_hidden)
    def reconstruct_observation(self):
        x = self.transposed_cnn(self.x)
        return x


if __name__ == "__main__":
    args = parse_args()
    writer, run_name = setup_logging(args)

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available on this system.")
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    torch.set_default_device(device)

    # Environment setup
    envs = gym.vector.SyncVectorEnv(
        [make_atari_env(args.gym_id, args.seed + i, i, args.capture_video, run_name, frame_stack=1) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    bce_loss = nn.BCELoss()

    if args.track:
        total_params = sum(p.numel() for p in agent.parameters())
        trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
        wandb.config.update({
            "total_parameters": total_params,
            "trainable_parameters": trainable_params
        }, allow_val_change=True)

    # Initialize observation buffers
    sequence_length = args.seq_len
    obs_shape = (sequence_length,) + envs.single_observation_space.shape
    obs_buffers = [deque(maxlen=sequence_length) for _ in range(args.num_envs)]
    obs_dim = envs.single_observation_space.shape  # (channels, height, width)

    # Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=[args.seed + i for i in range(args.num_envs)])
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    # Initialize buffers
    for i in range(args.num_envs):
        for _ in range(sequence_length):
            obs_buffers[i].append(next_obs[i].cpu().numpy())

    for update in range(1, num_updates + 1):
        # Annealing the learning rate
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            dones[step] = next_done

            # Update buffers
            for i in range(args.num_envs):
                obs_buffers[i].append(next_obs[i].cpu().numpy())

            # Prepare sequences
            obs_sequences = np.array([list(obs_buffers[i]) for i in range(args.num_envs)])
            obs_sequences = torch.tensor(obs_sequences, dtype=torch.float32).to(device)
            obs[step] = obs_sequences

            # Action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs_sequences)
                values[step] = value.view(-1)
            actions[step] = action
            logprobs[step] = logprob

            # Execute the game and log data
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(done).to(device)

            # Reset buffers for environments that are done
            for i, d in enumerate(done):
                if d:
                    obs_buffers[i] = deque([next_obs[i].cpu().numpy()] * sequence_length, maxlen=sequence_length)

            final_info = info.get('final_info')
            if final_info is not None and len(final_info) > 0:
                valid_entries = [entry for entry in final_info if entry is not None and 'episode' in entry]
                if valid_entries:
                    episodic_returns = [entry['episode']['r'] for entry in valid_entries]
                    episodic_lengths = [entry['episode']['l'] for entry in valid_entries]
                    avg_return = float(f'{np.mean(episodic_returns):.3f}')
                    avg_length = float(f'{np.mean(episodic_lengths):.3f}')
                    print(f"global_step={global_step}, avg_return={avg_return}, avg_length={avg_length}")
                    writer.add_scalar("charts/episodic_return", avg_return, global_step)
                    writer.add_scalar("charts/episodic_length", avg_length, global_step)

        # bootstrap value if not done
        with torch.no_grad():
            obs_sequences = np.array([list(obs_buffers[i]) for i in range(args.num_envs)])
            obs_sequences = torch.tensor(obs_sequences, dtype=torch.float32).to(device)
            next_value = agent.get_value(obs_sequences).view(1, -1)
            advantages, returns = compute_advantages(
                rewards, values, dones, next_value, next_done,
                args.gamma, args.gae_lambda, args.gae, args.num_steps, device
            )

        # Flatten the batch but keep the sequence dimension
        b_obs = obs.transpose(1, 0)  # Shape: (num_envs, num_steps, seq_len, channels, height, width)
        b_logprobs = logprobs.transpose(1, 0)
        b_actions = actions.transpose(1, 0)
        b_advantages = advantages.transpose(1, 0)
        b_returns = returns.transpose(1, 0)
        b_values = values.transpose(1, 0)

        # Optimizing the policy and value network
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        clipfracs = []

        # Initialize accumulators for metrics
        total_loss_list = []
        pg_loss_list = []
        v_loss_list = []
        entropy_list = []
        grad_norm_list = []
        approx_kl_list = []
        old_approx_kl_list = []
        grad_norm_list = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                # Get sequences for the minibatch environments
                mb_obs = b_obs[mbenvinds]
                mb_actions = b_actions[mbenvinds]
                mb_logprobs = b_logprobs[mbenvinds]
                mb_advantages = b_advantages[mbenvinds]
                mb_returns = b_returns[mbenvinds]
                mb_values = b_values[mbenvinds]

                # Flatten batch and steps
                mb_obs = mb_obs.reshape(-1, sequence_length, *obs_dim)
                mb_actions = mb_actions.reshape(-1)
                mb_logprobs = mb_logprobs.reshape(-1)
                mb_advantages = mb_advantages.reshape(-1)
                mb_returns = mb_returns.reshape(-1)
                mb_values = mb_values.reshape(-1)

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions.long())
                newlogprob = newlogprob.view(-1)
                entropy = entropy.view(-1)

                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    # Calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - mb_returns) ** 2
                    v_clipped = mb_values + torch.clamp(
                        newvalue - mb_values,
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                reconstruction_loss = torch.tensor(0.0, device=device)
                if args.reconstruction_coef > 0.0:
                    predicted_obs = agent.reconstruct_observation()
                    target_obs = mb_obs.float() / 255.0
                    assert predicted_obs.shape == target_obs.shape, (
                        f"Shape mismatch: predicted_obs {predicted_obs.shape} vs target_obs {target_obs.shape}"
                    )
                    reconstruction_loss = bce_loss(predicted_obs, target_obs)
                    loss += args.reconstruction_coef * reconstruction_loss

                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                # Append metrics for this minibatch
                total_loss_list.append(loss.item())
                pg_loss_list.append(pg_loss.item())
                v_loss_list.append(v_loss.item())
                entropy_list.append(entropy_loss.item())
                grad_norm_list.append(grad_norm.item())
                approx_kl_list.append(approx_kl.item())
                old_approx_kl_list.append(old_approx_kl.item())
                grad_norm_list.append(grad_norm.item())

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        # Compute means
        avg_total_loss = np.mean(total_loss_list)
        avg_pg_loss = np.mean(pg_loss_list)
        avg_v_loss = np.mean(v_loss_list)
        avg_entropy = np.mean(entropy_list)
        avg_grad_norm = np.mean(grad_norm_list)
        avg_approx_kl = np.mean(approx_kl_list)
        avg_old_approx_kl = np.mean(old_approx_kl_list)

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/total_loss", avg_total_loss, global_step)
        writer.add_scalar("losses/value_loss", avg_v_loss, global_step)
        writer.add_scalar("losses/policy_loss", avg_pg_loss, global_step)
        writer.add_scalar("losses/entropy", avg_entropy, global_step)
        writer.add_scalar("losses/grad_norm", avg_grad_norm, global_step)
        writer.add_scalar("losses/old_approx_kl", avg_old_approx_kl, global_step)
        writer.add_scalar("losses/approx_kl", avg_approx_kl, global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        sps = int(global_step / (time.time() - start_time))
        print("SPS:", sps)
        writer.add_scalar("charts/SPS", sps, global_step)

    finish_logging(args, writer, run_name, envs)
