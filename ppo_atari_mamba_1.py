import argparse
import random
import time
from collections import deque
from distutils.util import strtobool

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
from layers import layer_init
from mamba_ssm import Mamba

def parse_args():
    parser = argparse.ArgumentParser()
    add_common_args(parser)


    parser.add_argument("--memory-length", type=int, default=16,
        help="Number of past hidden states to store in the memory buffer")
    parser.add_argument("--hidden-dim", type=int, default=512,
        help="Size of the hidden dimension for the Mamba model")
    parser.add_argument("--d-state", type=int, default=16,
        help="State-space size for Mamba.")
    parser.add_argument("--d-conv", type=int, default=4,
        help="Convolutional projection size for Mamba.")
    parser.add_argument("--expand", type=int, default=2,
        help="Expansion factor in the Mamba state-space model.")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

class Agent(nn.Module):
    def __init__(self, envs, args):
        super(Agent, self).__init__()
        self.memory_length = args.memory_length

        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, args.hidden_dim)),
            nn.ReLU(),
        )
        
        self.mamba = Mamba(
            d_model=args.hidden_dim,
            d_state=args.d_state,
            d_conv=args.d_conv,
            expand=args.expand,
        )
        self.actor = layer_init(nn.Linear(args.hidden_dim, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(args.hidden_dim, 1), std=1)

        self.register_buffer("init_memory", torch.zeros(1, self.memory_length, args.hidden_dim))

    def forward_encoder(self, x):
        return self.encoder(x / 255.0)

    def get_action_and_value(self, x, prev_memory=None, action=None):
        """
        x: current observation, shape (batch, channels, H, W)
        prev_memory: previous memory buffer, shape (batch, memory_length, hidden_dim)
                     If None, initialize it with zeros.
        Returns: action, log_prob, entropy, value, new_memory
        """
        batch = x.shape[0]
        current_hidden = self.forward_encoder(x)  # (batch, hidden_dim)
        current_hidden = current_hidden.unsqueeze(1)  # (batch, 1, hidden_dim)

        if prev_memory is None:
            prev_memory = self.init_memory.expand(batch, -1, -1)  # (batch, memory_length, hidden_dim)

        # Concatenate the memory with the current hidden state.
        sequence = torch.cat([prev_memory, current_hidden], dim=1)  # (batch, memory_length + 1, hidden_dim)

        processed_seq = self.mamba(sequence)  # (batch, memory_length + 1, hidden_dim)
        context = processed_seq[:, -1, :]  # Use the last element as context.

        logits = self.actor(context)
        value = self.critic(context).squeeze(-1)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()

        # Update memory: drop the oldest and append the new context.
        new_memory = torch.cat([prev_memory[:, 1:, :], context.unsqueeze(1)], dim=1)
        return action, log_prob, entropy, value, new_memory

    def get_value(self, x, prev_memory=None):
        _, _, _, value, _ = self.get_action_and_value(x, prev_memory)
        return value


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

    # Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Initialize per-environment memory (shape: [num_envs, memory_length, hidden_dim])
    memory = agent.init_memory.expand(args.num_envs, -1, -1).clone()
    # Create a buffer to store the memory state for each time step.
    rollout_memory = torch.zeros((args.num_steps, args.num_envs,
                                    memory.shape[1], memory.shape[2]), device=device)


    # Start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=[args.seed + i for i in range(args.num_envs)])
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # Annealing the learning rate
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Store the current memory state for later use in PPO updates
            rollout_memory[step] = memory

            # Action logic with memory.
            with torch.no_grad():
                action, logprob, _, value, new_memory = agent.get_action_and_value(next_obs, prev_memory=memory)
                memory = new_memory  # Update memory for next step.
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Execute the game and log data
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(done).to(device)

            # Reset memory for finished episodes
            done_mask = torch.tensor(done, device=device, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
            memory = memory * (1 - done_mask) + agent.init_memory.expand_as(memory) * done_mask

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
            next_value = agent.get_value(next_obs, prev_memory=memory).reshape(1, -1)
            advantages, returns = compute_advantages(
                rewards, values, dones, next_value, next_done,
                args.gamma, args.gae_lambda, args.gae, args.num_steps, device
            )

        # Flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        # Flatten rollout memory: now each sample gets its associated memory state.
        b_memory = rollout_memory.reshape((-1, memory.shape[1], memory.shape[2]))

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
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
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Use the stored memory for each minibatch sample.
                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds],
                    prev_memory=b_memory[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # Calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

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

