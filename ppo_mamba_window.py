import argparse
import os
import random
import time
from distutils.util import strtobool
from collections import deque

import gymnasium as gym
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from mamba_ssm import Mamba

from gae import compute_advantages
from env_utils import make_minigrid_env, make_atari_env, make_poc_env, make_classic_env, make_memory_gym_env
from exp_utils import setup_logging, finish_logging, add_common_args

# Add a new argument for episodic memory length
def parse_args():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--hidden-dim", type=int, default=512,
        help="Size of the hidden dimension for CNN")
    parser.add_argument("--memory-length", type=int, default=4,
        help="Number of past hidden states to use as episodic memory")
    parser.add_argument("--d-state", type=int, default=64,
        help="State-space size for Mamba")
    parser.add_argument("--d-conv", type=int, default=4,
        help="Convolutional projection size for Mamba")
    parser.add_argument("--expand", type=int, default=2,
        help="Expansion factor in the Mamba state-space model")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# --- Agent using Mamba-based episodic memory ---
class Agent(nn.Module):
    def __init__(self, envs, args):
        super(Agent, self).__init__()
        self.obs_space = envs.single_observation_space
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.memory_length = args.memory_length  # episodic memory window length

        # Determine the observation space type and create appropriate encoder
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
                layer_init(nn.Linear(64 * 7 * 7, self.args.hidden_dim)),
                nn.ReLU(),
            )
        else:  # vector observation
            input_dim = np.prod(self.obs_space.shape)
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, self.args.hidden_dim),
                nn.ReLU(),
            )

        # Mamba block: it processes a sequence of tokens.
        self.mamba = Mamba(
            d_model=args.hidden_dim,
            d_state=args.d_state,
            d_conv=args.d_conv,
            expand=args.expand,
        )

        # Actor and critic heads.
        self.actor = layer_init(nn.Linear(args.hidden_dim, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(args.hidden_dim, 1), std=1)

    def get_states(self, x):
        """Process observations to get encoded states."""
        if "minigrid" in self.args.gym_id.lower() or "mortar" in self.args.gym_id.lower():
            x = x.permute(0, 3, 1, 2) / 255.0
        if "ale/" in self.args.gym_id.lower():
            x = x / 255.0
        hidden = self.encoder(x)
        return hidden

    def forward_with_memory(self, x, memory_window):
        """
        x: current observation (B, obs_dim)
        memory_window: (B, memory_length, hidden_dim)
        """
        B = x.shape[0]
        x_enc = self.get_states(x)
        x_token = x_enc.unsqueeze(1)  # (B, 1, hidden_dim)

        # If no memory is provided, initialize it to zeros.
        if memory_window is None:
            memory_window = torch.zeros(B, self.memory_length, self.hidden_dim, device=x.device)

        # Concatenate memory window with current token.
        seq = torch.cat([memory_window, x_token], dim=1)  # (B, memory_length+1, hidden_dim)

        # Process the sequence through Mamba.
        out_seq = self.mamba(seq)  # (B, memory_length+1, hidden_dim)
        current_repr = out_seq[:, -1, :]  # Use the last token as the current representation.

        # Compute policy logits and value.
        logits = self.actor(current_repr)
        value = self.critic(current_repr).flatten()

        new_token = current_repr.detach()  # Detach the current token for the memory window.

        # Update memory: shift the memory window and append the new token.
        new_memory = torch.cat([memory_window[:, 1:], new_token.unsqueeze(1)], dim=1)
        return logits, value, new_memory, current_repr

    def get_action_and_value(self, x, memory_window, action=None):
        logits, value, new_memory, hidden = self.forward_with_memory(x, memory_window)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value, new_memory

    def get_value(self, x, memory_window):
        _, value, new_memory, _ = self.forward_with_memory(x, memory_window)
        return value, new_memory

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

    # Environment setup - enhanced with support for multiple env types
    if "ale" in args.gym_id.lower():
        envs_lst = [make_atari_env(args.gym_id, args.seed + i, i, args.capture_video, 
                                  run_name, frame_stack=1) for i in range(args.num_envs)]
    elif "minigrid" in args.gym_id.lower():
        envs_lst = [make_minigrid_env(args.gym_id, args.seed + i, i, args.capture_video, 
                                     run_name, agent_view_size=3, tile_size=28, max_episode_steps=96) for i in range(args.num_envs)]
    elif "poc" in args.gym_id.lower():
        envs_lst = [make_poc_env(args.gym_id, args.seed + i, i, args.capture_video,
                                run_name, step_size=0.02, glob=False, freeze=True, max_episode_steps=96) for i in range(args.num_envs)]
    elif args.gym_id == "MortarMayhem-Grid-v0":
        envs_lst = [make_memory_gym_env(args.gym_id, args.seed + i, i, args.capture_video,
                                       run_name) for i in range(args.num_envs)]
    else:
        envs_lst = [make_classic_env(args.gym_id, args.seed + i, i, args.capture_video, 
                                    run_name) for i in range(args.num_envs)]
    
    envs = gym.vector.SyncVectorEnv(envs_lst)
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Log parameter counts
    total_params = sum(p.numel() for p in agent.parameters())
    trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    if args.track:
        wandb.config.update({
            "total_parameters": total_params,
            "trainable_parameters": trainable_params
        }, allow_val_change=True)
    print(f"Total parameters: {total_params / 1e6:.4f}M, trainable parameters: {trainable_params / 1e6:.4f}M")

    # --- Storage setup ---
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # New buffer for storing the memory window used at each step.
    stored_memories = torch.zeros((args.num_steps, args.num_envs, agent.memory_length, args.hidden_dim), device=device)

    # --- Start the game ---
    global_step = 0
    start_time = time.time()
    episode_infos = deque(maxlen=100)
    next_obs, _ = envs.reset(seed=[args.seed + i for i in range(args.num_envs)])
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    # Initialize per-env episodic memory
    next_memory = torch.zeros((args.num_envs, agent.memory_length, args.hidden_dim), device=device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        update_start_time = time.time()
        
        # Annealing the learning rate
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        inference_time_total = 0.0
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            # Save the current memory window for this step.
            stored_memories[step] = next_memory

            # Action logic with timing
            inf_start = time.time()
            with torch.no_grad():
                action, logprob, entropy, value, new_memory = agent.get_action_and_value(next_obs, next_memory)
                values[step] = value.flatten()
            inference_time_total += (time.time() - inf_start)
            actions[step] = action
            logprobs[step] = logprob
            next_memory = new_memory  # Update memory for next step

            # Execute the game and log data
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(done).to(device)

            # For any environment where an episode ended, reset its memory.
            for i, d in enumerate(next_done):
                if d.item():
                    next_memory[i] = torch.zeros(agent.memory_length, args.hidden_dim, device=device)

            # Process episode information
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

        # Log inference performance
        avg_inference_latency = inference_time_total / args.num_steps
        writer.add_scalar("metrics/inference_latency", avg_inference_latency, global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value, next_memory = agent.get_value(next_obs, next_memory)
            next_value = next_value.reshape(1, -1)
            advantages, returns = compute_advantages(
                rewards, values, dones, next_value, next_done,
                args.gamma, args.gae_lambda, args.gae, args.num_steps, device
            )

        # Flatten the batch.
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        # Flatten the stored memory windows.
        b_memory = stored_memories.reshape((args.batch_size, agent.memory_length, args.hidden_dim))

        # --- PPO update ---
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        total_loss_list = []
        pg_loss_list = []
        v_loss_list = []
        entropy_list = []
        grad_norm_list = []
        approx_kl_list = []
        old_approx_kl_list = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                # Pass both the observation and the corresponding stored memory window.
                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds], b_memory[mb_inds], action=b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss (PPO clipped objective)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds],
                                                               -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                
                # Calculate gradient norms for logging
                total_grad_norm = 0.0
                for p in agent.parameters():
                    if p.grad is not None:
                        total_grad_norm += p.grad.data.norm(2).item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                grad_norm_list.append(total_grad_norm)
                
                # Clip gradients
                grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                # Append metrics for this minibatch
                total_loss_list.append(loss.item())
                pg_loss_list.append(pg_loss.item())
                v_loss_list.append(v_loss.item())
                entropy_list.append(entropy_loss.item())
                approx_kl_list.append(approx_kl.item())
                old_approx_kl_list.append(old_approx_kl.item())

            # Early stopping based on KL divergence
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        # Compute metrics means
        avg_total_loss = np.mean(total_loss_list)
        avg_pg_loss = np.mean(pg_loss_list)
        avg_v_loss = np.mean(v_loss_list)
        avg_entropy = np.mean(entropy_list)
        avg_grad_norm = np.mean(grad_norm_list)
        avg_approx_kl = np.mean(approx_kl_list)
        avg_old_approx_kl = np.mean(old_approx_kl_list)

        # Calculate explained variance
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Log performance metrics
        sps = int(global_step / (time.time() - start_time))
        current_return = np.mean([ep['r'] for ep in episode_infos]) if episode_infos else 0.0
        
        print(f"Update {update}: SPS={sps}, Return={current_return:.2f}, "
              f"pi_loss={avg_pg_loss:.6f}, v_loss={avg_v_loss:.6f}, entropy={avg_entropy:.6f}, "
              f"explained_var={explained_var:.6f}")
              
        # Log to the writer
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

        # Log training update duration
        update_time = time.time() - update_start_time
        writer.add_scalar("metrics/training_time_per_update", update_time, global_step)
        
        # Log GPU memory usage
        if torch.cuda.is_available():
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
        
        # Save model checkpoint
        if args.save_model and update % args.save_interval == 0:
            model_path = f"runs/{run_name}/{args.exp_name}_update_{update}.cleanrl_model"
            model_data = {
                "model_weights": agent.state_dict(),
                "args": vars(args),
            }
            torch.save(model_data, model_path)
            print(f"Model saved to {model_path}")
        
    # Cleanup
    finish_logging(args, writer, run_name, envs)