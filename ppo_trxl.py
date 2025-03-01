import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from collections import deque

from gae import compute_advantages
from exp_utils import add_common_args, setup_logging, finish_logging
from env_utils import make_atari_env, make_minigrid_env, make_poc_env, make_classic_env, make_memory_gym_env
from layers import Transformer, batched_index_select, layer_init

def parse_args():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
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
        help="the coefficient of the observation reconstruction loss")
    parser.add_argument("--final-lr", type=float, default=1.0e-5,
        help="the final learning rate after annealing")
    parser.add_argument("--init-ent-coef", type=float, default=0.0001,
        help="initial entropy coefficient")
    parser.add_argument("--final-ent-coef", type=float, default=0.000001,
        help="final entropy coefficient after annealing")
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    return args


class Agent(nn.Module):
    def __init__(self, envs, args, action_space_shape, max_episode_steps):
        super().__init__()
        self.obs_space = envs.single_observation_space
        self.max_episode_steps = max_episode_steps
        self.args = args
        if len(self.obs_space.shape) == 3:  # image observation
            if self.obs_space.shape[0] in [1, 3]:
                in_channels = self.obs_space.shape[0]  # channels-first (e.g., ALE/Breakout-v5)
            else:
                in_channels = self.obs_space.shape[2]
            self.encoder = nn.Sequential(
                layer_init(nn.Conv2d(in_channels, 32, 8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * 7 * 7, self.args.trxl_dim)),
                nn.ReLU(),
            )
        else:  # vector observation
            input_dim = np.prod(self.obs_space.shape)
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, self.args.trxl_dim),
                nn.ReLU(),
            )

        # Transformer model
        self.transformer = Transformer(
            args.trxl_num_layers, args.trxl_dim, args.trxl_num_heads, 
            self.max_episode_steps, args.trxl_positional_encoding
        )

        self.hidden_post_trxl = nn.Sequential(
            layer_init(nn.Linear(args.trxl_dim, args.trxl_dim)),
            nn.ReLU(),
        )

        self.actor_branches = nn.ModuleList(
            [
                layer_init(nn.Linear(args.trxl_dim, out_features=num_actions), std=0.01)
                for num_actions in action_space_shape
            ]
        )
        self.critic = layer_init(nn.Linear(args.trxl_dim, 1), std=1.0)

    def get_states(self, x):
        if "minigrid" in self.args.gym_id.lower() or "mortar" in self.args.gym_id.lower():
            x = x.permute(0, 3, 1, 2) / 255.0
        if "ale/" in self.args.gym_id.lower():
            x = x / 255.0
        hidden = self.encoder(x)
        return hidden

    def get_value(self, x, memory, memory_mask, memory_indices):
        x = self.get_states(x)
        x, _ = self.transformer(x, memory, memory_mask, memory_indices)
        x = self.hidden_post_trxl(x)
        return self.critic(x).flatten()

    def get_action_and_value(self, x, memory, memory_mask, memory_indices, action=None):
        x = self.get_states(x)
        x, memory = self.transformer(x, memory, memory_mask, memory_indices)
        x = self.hidden_post_trxl(x)
        
        probs = [Categorical(logits=branch(x)) for branch in self.actor_branches]
        if action is None:
            action = torch.stack([dist.sample() for dist in probs], dim=1)
        logprobs = [dist.log_prob(action[:, i]) for i, dist in enumerate(probs)]
        entropies = torch.stack([dist.entropy() for dist in probs], dim=1).sum(1).reshape(-1)
        return action, torch.stack(logprobs, dim=1), entropies, self.critic(x).flatten(), memory

if __name__ == "__main__":
    args = parse_args()
    writer, run_name = setup_logging(args)

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.backends.cudnn.benchmark = False

    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available on this system.")
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    torch.set_default_device(device)

    # Environment setup
    if "ale" in args.gym_id.lower():
        envs_lst = [make_atari_env(args.gym_id, args.seed + i, i, args.capture_video, 
                                   run_name, frame_stack=1) for i in range(args.num_envs)]
    elif "minigrid" in args.gym_id.lower():
        envs_lst = [make_minigrid_env(args.gym_id, args.seed + i, i, args.capture_video, 
                                      run_name, agent_view_size=3, tile_size=28, max_episode_steps=96) for i in range(args.num_envs)]
    elif "poc" in args.gym_id.lower():
        envs_lst = [make_poc_env(args.gym_id, args.seed + i, i, args.capture_video,
                                 run_name, step_size=0.2, glob=False, freeze=False, max_episode_steps=96) for i in range(args.num_envs)]
    elif args.gym_id == "MortarMayhem-Grid-v0":
        envs_lst = [make_memory_gym_env(args.gym_id, args.seed + i, i, args.capture_video,
                                        run_name) for i in range(args.num_envs)]
    else:
        envs_lst = [make_classic_env(args.gym_id, args.seed + i, i, args.capture_video, 
                                     run_name) for i in range(args.num_envs)]
    envs = gym.vector.SyncVectorEnv(envs_lst)
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    env_current_episode_step = torch.zeros((args.num_envs,), dtype=torch.long)
    max_episode_steps = getattr(envs.envs[0], "max_episode_steps", 1024)
    if not max_episode_steps:
        envs.envs[0].reset()
        max_episode_steps = getattr(envs.envs[0], "max_episode_steps", 1024)
    if max_episode_steps <= 0:
        max_episode_steps = 1024
    args.trxl_memory_length = min(args.trxl_memory_length, max_episode_steps)

    # Define action space
    observation_space = envs.single_observation_space
    action_space_shape = ((envs.single_action_space.n,) if isinstance(envs.single_action_space, gym.spaces.Discrete)
                          else tuple(envs.single_action_space.nvec))

    agent = Agent(envs, args, action_space_shape, max_episode_steps).to(device)
    optimizer = optim.AdamW(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    total_params = sum(p.numel() for p in agent.parameters())
    trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    if args.track:
        wandb.config.update({
            "total_parameters": total_params,
            "trainable_parameters": trainable_params
        }, allow_val_change=True)
    print(f"Total parameters: {total_params / 10e6:.4f}M, trainable parameters: {trainable_params / 10e6:.4f}M")

    # Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs, len(action_space_shape)), dtype=torch.long).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs, len(action_space_shape))).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    stored_memories = []
    stored_memory_masks = torch.zeros((args.num_steps, args.num_envs, args.trxl_memory_length), dtype=torch.bool).to(device)
    stored_memory_index = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long).to(device)
    stored_memory_indices = torch.zeros((args.num_steps, args.num_envs, args.trxl_memory_length), dtype=torch.long).to(device)

    # Start the game
    global_step = 0
    start_time = time.time()
    episode_infos = deque(maxlen=100)
    next_obs, _ = envs.reset(seed=[args.seed + i for i in range(args.num_envs)])
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_memory = torch.zeros((args.num_envs, max_episode_steps, args.trxl_num_layers, args.trxl_dim), dtype=torch.float32).to(device)
    memory_mask = torch.tril(torch.ones((args.trxl_memory_length, args.trxl_memory_length)), diagonal=-1).to(device)

    # Indices for memory
    from_indices = torch.repeat_interleave(
        torch.arange(0, args.trxl_memory_length).unsqueeze(0), args.trxl_memory_length - 1, dim=0
    ).long().to(device)
    to_indices = torch.stack(
        [torch.arange(i, i + args.trxl_memory_length) for i in range(max_episode_steps - args.trxl_memory_length + 1)]
    ).long().to(device)
    memory_indices = torch.cat((from_indices, to_indices))

    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        update_start_time = time.time()
        # Annealing the learning rate
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * (args.learning_rate - args.final_lr) + args.final_lr
            optimizer.param_groups[0]["lr"] = lrnow
        
        # Entropy coefficient annealing
        ent_coef = (args.init_ent_coef - args.final_ent_coef) * frac + args.final_ent_coef
        # Prepare current environment memory references
        stored_memories = [next_memory[e] for e in range(args.num_envs)]
        for e in range(args.num_envs):
            stored_memory_index[:, e] = e

        inference_time_total = 0.0
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            stored_memory_masks[step] = memory_mask[torch.clip(env_current_episode_step, 0, args.trxl_memory_length - 1)]
            stored_memory_indices[step] = memory_indices[env_current_episode_step]

            # Action logic
            inf_start = time.time()
            with torch.no_grad():
                memory_window = batched_index_select(next_memory, 1, stored_memory_indices[step])
                action, logprob, _, value, new_memory = agent.get_action_and_value(
                    next_obs, memory_window, stored_memory_masks[step], stored_memory_indices[step]
                )
                next_memory[torch.arange(args.num_envs), env_current_episode_step] = new_memory
                actions[step], logprobs[step], values[step] = action, logprob, value
            inference_time_total += (time.time() - inf_start)

            # Execute the game and log data
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy().squeeze(1))
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
                        (max_episode_steps, args.trxl_num_layers, args.trxl_dim), dtype=torch.float32, device=device
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
                    episode_infos.append({'r': avg_return, 'l': avg_length})
                    writer.add_scalar("charts/episode_return", avg_return, global_step)
                    writer.add_scalar("charts/episode_length", avg_length, global_step)

        avg_inference_latency = inference_time_total / args.num_steps
        writer.add_scalar("metrics/inference_latency", avg_inference_latency, global_step)

        # bootstrap value if not done
        with torch.no_grad():
            start_idx = torch.clip(env_current_episode_step - args.trxl_memory_length, 0)
            end_idx = torch.clip(env_current_episode_step, args.trxl_memory_length)
            indices = torch.stack([torch.arange(start_idx[b], end_idx[b], device=device) for b in range(args.num_envs)])
            memory_window = batched_index_select(next_memory, 1, indices)
            next_value = agent.get_value(
                next_obs, memory_window,
                memory_mask[torch.clip(env_current_episode_step, 0, args.trxl_memory_length - 1)],
                stored_memory_indices[-1],
            ).reshape(1, -1)
            advantages, returns = compute_advantages(
                rewards, values, dones, next_value, next_done,
                args.gamma, args.gae_lambda, args.gae, args.num_steps, device
            )

        # Flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1, len(action_space_shape))
        b_actions = actions.reshape(-1, len(action_space_shape))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_memory_index = stored_memory_index.reshape(-1)
        b_memory_indices = stored_memory_indices.reshape(-1, args.trxl_memory_length)
        b_memory_mask = stored_memory_masks.reshape(-1, args.trxl_memory_length)
        stored_memories = torch.stack(stored_memories, dim=0)

        # Actual maximum episode steps might be smaller than the allocated size
        actual_max_episode_steps = (b_memory_indices * b_memory_mask.long()).max().item() + 1
        if actual_max_episode_steps < args.trxl_memory_length:
            b_memory_indices = b_memory_indices[:, :actual_max_episode_steps]
            b_memory_mask = b_memory_mask[:, :actual_max_episode_steps]
            stored_memories = stored_memories[:, :actual_max_episode_steps]

        # Initialize accumulators for metrics
        clipfracs = []
        total_loss_list = []
        pg_loss_list = []
        v_loss_list = []
        entropy_list = []
        grad_norm_list = []
        approx_kl_list = []
        old_approx_kl_list = []

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

                # Advantage processing
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                mb_advantages = mb_advantages.unsqueeze(1).repeat(1, len(action_space_shape))

                # Policy loss calculation
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # Calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

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

        sps = int(global_step / (time.time() - start_time))
        current_return = np.mean([ep['r'] for ep in episode_infos]) if episode_infos else 0.0
        print(f"Update {update}: SPS={sps}, Return={current_return:.2f}, "
              f"pi_loss={pg_loss.item():.6f}, v_loss={v_loss.item():.6f}, entropy={entropy_loss.item():.6f}, "
              f"explained_var={explained_var:.6f}")
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
        writer.add_scalar("charts/SPS", sps, global_step)

        # Log average episode return
        if episode_infos:
            avg_episode_return = np.mean([ep['r'] for ep in episode_infos])
            writer.add_scalar("charts/avg_episode_return", avg_episode_return, global_step)

        # Log training update duration (wall-clock time per update)
        update_time = time.time() - update_start_time
        writer.add_scalar("metrics/training_time_per_update", update_time, global_step)
        
        # Log GPU memory usage
        gpu_memory_allocated = torch.cuda.memory_allocated(device)  
        gpu_memory_reserved = torch.cuda.memory_reserved(device)
        total_gpu_memory = torch.cuda.get_device_properties(device).total_memory

        gpu_memory_allocated_gb = gpu_memory_allocated / (1024**3)
        gpu_memory_reserved_gb = gpu_memory_reserved / (1024**3)
        gpu_memory_allocated_percent = (gpu_memory_allocated / total_gpu_memory) * 100
        gpu_memory_reserved_percent = (gpu_memory_reserved / total_gpu_memory) * 100

        writer.add_scalar("metrics/GPU_memory_allocated_GB", gpu_memory_allocated_gb, global_step)
        writer.add_scalar("metrics/GPU_memory_reserved_GB", gpu_memory_reserved_gb, global_step)
        writer.add_scalar("metrics/GPU_memory_allocated_percent", gpu_memory_allocated_percent, global_step)
        writer.add_scalar("metrics/GPU_memory_reserved_percent", gpu_memory_reserved_percent, global_step)
        
        # Save model checkpoint every save_interval updates
        if args.save_model and update % args.save_interval == 0:
            model_path = f"runs/{run_name}/{args.exp_name}_update_{update}.cleanrl_model"
            model_data = {
                "model_weights": agent.state_dict(),
                "args": vars(args),
            }
            torch.save(model_data, model_path)
            print(f"Model saved to {model_path}")
        
    finish_logging(args, writer, run_name, envs)
