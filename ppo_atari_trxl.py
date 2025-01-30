import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from layers import Transformer
from gae import compute_advantages
from env_utils import make_atari_env
from exp_utils import add_common_args, setup_logging, finish_logging

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="ALE/Breakout-v5",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=2.75e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--final-lr", type=float, default=1.0e-5,
        help="the final learning rate after annealing")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-mamba",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=32,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=512,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.995,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=8,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=3,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggle advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle whether or not to use a clipped loss for the value function")
    parser.add_argument("--init-ent-coef", type=float, default=0.0001,
        help="initial entropy coefficient")
    parser.add_argument("--final-ent-coef", type=float, default=0.000001,
        help="final entropy coefficient after annealing")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.25,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")

    # Additional arguments for transformer
    parser.add_argument("--trxl-num-layers", type=int, default=3,
        help="the number of transformer layers")
    parser.add_argument("--trxl-num-heads", type=int, default=4,
        help="the number of heads used in multi-head attention")
    parser.add_argument("--trxl-dim", type=int, default=384,
        help="the dimension of the transformer")
    parser.add_argument("--trxl-memory-length", type=int, default=119,
        help="the length of TrXL's sliding memory window")
    parser.add_argument("--trxl-positional-encoding", type=str, default="absolute",
        help='the positional encoding type: "", "absolute", "learned"')
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

def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


class Agent(nn.Module):
    def __init__(self, envs, args, max_episode_steps):
        super(Agent, self).__init__()
        self.max_episode_steps = max_episode_steps
        
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, args.trxl_dim)),
            nn.ReLU(),
        )

        if args.reconstruction_coef > 0:
            self.transposed_cnn = nn.Sequential(
                layer_init(nn.Linear(args.trxl_dim, 64 * 7 * 7)),
                nn.ReLU(),
                nn.Unflatten(1, (64, 7, 7)),
                layer_init(nn.ConvTranspose2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                layer_init(nn.ConvTranspose2d(64, 32, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.ConvTranspose2d(32, 1, 8, stride=4)),
                nn.Sigmoid(),
            )
            
        self.transformer = Transformer(
            args.trxl_num_layers, args.trxl_dim, args.trxl_num_heads, self.max_episode_steps, args.trxl_positional_encoding
        )

        self.hidden_post_trxl = nn.Sequential(
            layer_init(nn.Linear(args.trxl_dim, args.trxl_dim)),
            nn.ReLU(),
        )

        self.actor = layer_init(nn.Linear(args.trxl_dim, envs.single_action_space.n), std=np.sqrt(0.01))
        self.critic = layer_init(nn.Linear(args.trxl_dim, 1), std=1.0)

    def get_value(self, x, memory, memory_mask, memory_indices):
        x = self.network(x / 255.0)
        x, _ = self.transformer(x, memory, memory_mask, memory_indices)
        x = self.hidden_post_trxl(x)
        return self.critic(x).flatten()

    def get_action_and_value(self, x, memory, memory_mask, memory_indices, action=None):
        x = self.network(x / 255.0)
        x, memory = self.transformer(x, memory, memory_mask, memory_indices)
        x = self.hidden_post_trxl(x)
        self.x = x
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x).squeeze(-1), memory

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

    env_current_episode_step = torch.zeros((args.num_envs,), dtype=torch.long)
    max_episode_steps = envs.envs[0].spec.max_episode_steps
    if not max_episode_steps:
        envs.envs[0].reset()
        max_episode_steps = envs.envs[0].max_episode_steps
    if max_episode_steps <= 0:
        max_episode_steps = 1024
    args.trxl_memory_length = min(args.trxl_memory_length, max_episode_steps)

    agent = Agent(envs, args, max_episode_steps).to(device)
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
    stored_memories = []
    stored_memory_masks = torch.zeros((args.num_steps, args.num_envs, args.trxl_memory_length), dtype=torch.bool)
    stored_memory_index = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long)
    stored_memory_indices = torch.zeros((args.num_steps, args.num_envs, args.trxl_memory_length), dtype=torch.long)

    # Start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=[args.seed + i for i in range(args.num_envs)])
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    next_memory = torch.zeros((args.num_envs, max_episode_steps, args.trxl_num_layers, args.trxl_dim), dtype=torch.float32)
    memory_mask = torch.tril(torch.ones((args.trxl_memory_length, args.trxl_memory_length)), diagonal=-1)

    # Indices for memory
    from_indices = torch.repeat_interleave(
        torch.arange(0, args.trxl_memory_length).unsqueeze(0), args.trxl_memory_length - 1, dim=0
    ).long()
    to_indices = torch.stack(
        [torch.arange(i, i + args.trxl_memory_length) for i in range(max_episode_steps - args.trxl_memory_length + 1)]
    ).long()
    memory_indices = torch.cat((from_indices, to_indices))

    for update in range(1, num_updates + 1):
        sampled_episode_infos = []
        do_anneal = args.anneal_lr and (args.total_timesteps > 0)
        steps_to_anneal = num_updates * args.batch_size if not hasattr(args, "anneal_steps") else args.anneal_steps
        if steps_to_anneal <= 0:
            steps_to_anneal = num_updates * args.batch_size
        frac = 1 - global_step / steps_to_anneal if do_anneal and global_step < steps_to_anneal else 0
        lr = (args.learning_rate - args.final_lr) * frac + args.final_lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        ent_coef = (args.init_ent_coef - args.final_ent_coef) * frac + args.final_ent_coef

        # Prepare current environment memory references
        stored_memories = [next_memory[e] for e in range(args.num_envs)]
        for e in range(args.num_envs):
            stored_memory_index[:, e] = e

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            stored_memory_masks[step] = memory_mask[torch.clip(env_current_episode_step, 0, args.trxl_memory_length - 1)]
            stored_memory_indices[step] = memory_indices[env_current_episode_step]

            # Action logic
            with torch.no_grad():
                memory_window = batched_index_select(next_memory, 1, stored_memory_indices[step])
                action, logprob, _, value, new_memory = agent.get_action_and_value(
                    next_obs, memory_window, stored_memory_masks[step], stored_memory_indices[step]
                )
                next_memory[torch.arange(args.num_envs), env_current_episode_step] = new_memory
                actions[step], logprobs[step], values[step] = action, logprob, value

            # Execute the game and log data
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(done).to(device)

            # If done, reset environment memory
            for idx, d in enumerate(next_done):
                if d:
                    env_current_episode_step[idx] = 0
                    mem_index = stored_memory_index[step, idx]
                    stored_memories[mem_index] = stored_memories[mem_index].clone()
                    next_memory[idx] = torch.zeros(
                        (max_episode_steps, args.trxl_num_layers, args.trxl_dim), dtype=torch.float32
                    )
                    if step < args.num_steps - 1:
                        stored_memories.append(next_memory[idx])
                        stored_memory_index[step + 1:, idx] = len(stored_memories) - 1
                else:
                    env_current_episode_step[idx] += 1

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
            start_idx = torch.clip(env_current_episode_step - args.trxl_memory_length, 0)
            end_idx = torch.clip(env_current_episode_step, args.trxl_memory_length)
            indices = torch.stack([torch.arange(start_idx[b], end_idx[b]) for b in range(args.num_envs)]).long()
            memory_window = batched_index_select(next_memory, 1, indices)
            next_value = agent.get_value(
                next_obs, memory_window,
                memory_mask[torch.clip(env_current_episode_step, 0, args.trxl_memory_length - 1)],
                stored_memory_indices[-1],
            )
            advantages, returns = compute_advantages(
                rewards, values, dones, next_value, next_done,
                args.gamma, args.gae_lambda, args.gae, args.num_steps, device
            )

        # Flatten the batch
        b_obs = obs.reshape(-1, *obs.shape[2:])
        b_logprobs = logprobs.reshape(-1, *logprobs.shape[2:])
        b_actions = actions.reshape(-1, *actions.shape[2:])
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_memory_index = stored_memory_index.reshape(-1)
        b_memory_indices = stored_memory_indices.reshape(-1, *stored_memory_indices.shape[2:])
        b_memory_mask = stored_memory_masks.reshape(-1, *stored_memory_masks.shape[2:])
        stored_memories = torch.stack(stored_memories, dim=0)

        actual_max_episode_steps = (stored_memory_indices * stored_memory_masks).max().item() + 1
        if actual_max_episode_steps < args.trxl_memory_length:
            b_memory_indices = b_memory_indices[:, :actual_max_episode_steps]
            b_memory_mask = b_memory_mask[:, :actual_max_episode_steps]
            stored_memories = stored_memories[:, :actual_max_episode_steps]

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
            b_inds = torch.randperm(args.batch_size)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                mb_memories = stored_memories[b_memory_index[mb_inds]]
                mb_memory_windows = batched_index_select(mb_memories, 1, b_memory_indices[mb_inds])

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds], mb_memory_windows, b_memory_mask[mb_inds], b_memory_indices[mb_inds], b_actions[mb_inds]
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
                loss = pg_loss - ent_coef * entropy_loss + v_loss * args.vf_coef

                reconstruction_loss = torch.tensor(0.0, device=device)
                if args.reconstruction_coef > 0.0:
                    predicted_obs = agent.reconstruct_observation()
                    target_obs = b_obs[mb_inds].float() / 255.0
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
