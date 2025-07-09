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
from torch.distributions.normal import Normal
from collections import deque

from gae import compute_advantages
from exp_utils import add_common_args, setup_logging, finish_logging
from env_utils import make_atari_env, make_minigrid_env, make_poc_env, make_classic_env, make_memory_gym_env, make_continuous_env
from layers import layer_init

def parse_args():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--hidden-dim", type=int, default=512,
        help="the hidden dimension of the model")
    parser.add_argument("--obs-stack", type=int, default=1,
        help="the number of frames to stack for the observation")
    parser.add_argument("--masked-indices", type=str, default="1,3",
        help="indices of the observations to mask")
    parser.add_argument("--frame-stack", type=int, default=1,
        help="frame stack for the environment")
    args = parser.parse_args()
    args.masked_indices = [int(x) for x in args.masked_indices.split(',')]
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

class Agent(nn.Module):
    def __init__(self, envs, args):
        super(Agent, self).__init__()
        self.obs_space = envs.single_observation_space
        self.args = args
        mujoco_envs = ["HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"]
        if args.gym_id in mujoco_envs:
            input_dim = np.prod(self.obs_space.shape)
            self.encoder = nn.Sequential(
                nn.Flatten(),
                layer_init(nn.Linear(input_dim, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, self.args.hidden_dim)),
                nn.Tanh(),
            )
        else:
             # For image-based environments (e.g., Atari, Minigrid), use a conv encoder.
            obs_shape = self.obs_space.shape
            conv_input = False
            in_channels = None

            # Handle both non-stacked (3D) and stacked (4D) observations.
            if isinstance(self.obs_space, gym.spaces.Box) and len(obs_shape) in [3, 4]:
                if len(obs_shape) == 3:
                    # e.g. (channels, height, width) or (height, width, channels)
                    if obs_shape[0] in [1, 3, 4]:
                        in_channels = obs_shape[0]
                    else:
                        in_channels = obs_shape[2]
                    conv_input = True
                elif len(obs_shape) == 4:
                    # Shape is (frame_stack, height, width, channels)
                    if obs_shape[-1] in [1, 3, 4]:
                        # Combine frame stacking with channels.
                        in_channels = obs_shape[0] * obs_shape[-1]
                        conv_input = True

            if conv_input:
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
            else:
                # Fallback to a vector encoder if not an image.
                input_dim = np.prod(obs_shape)
                self.encoder = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(input_dim, self.args.hidden_dim),
                    nn.ReLU(),
                )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(args.hidden_dim, args.hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(args.hidden_dim, args.hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(args.hidden_dim, 1), std=1.0),
        )
        
        if isinstance(envs.single_action_space, gym.spaces.Discrete):
            self.is_continuous = False
            self.actor = nn.Sequential(
                layer_init(nn.Linear(args.hidden_dim, args.hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(args.hidden_dim, args.hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(args.hidden_dim, envs.single_action_space.n), std=0.01),
            )
        elif isinstance(envs.single_action_space, gym.spaces.Box):
            self.is_continuous = True
            action_dim = np.prod(envs.single_action_space.shape)
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(args.hidden_dim, args.hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(args.hidden_dim, args.hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(args.hidden_dim, action_dim), std=0.01),
            )
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
    
    def get_states(self, x):
        if "minigrid" in self.args.gym_id.lower() or "mortar" in self.args.gym_id.lower():
            if x.ndim == 5:
                # First, permute to (batch, frame_stack, channels, height, width)
                x = x.permute(0, 1, 4, 2, 3)
                # Then flatten the frame_stack and channel dimensions:
                batch, fs, C, H, W = x.shape
                x = x.reshape(batch, fs * C, H, W) / 255.0
            else:
                # If no frame stacking is applied, shape is (batch, height, width, channels)
                x = x.permute(0, 3, 1, 2) / 255.0
        if "ale/" in self.args.gym_id.lower():
            x = x / 255.0
        hidden = self.encoder(x)
        return hidden

    def get_value(self, x):
        hidden = self.get_states(x)
        return self.critic(hidden)

    def get_action_and_value(self, x, action=None):
        hidden = self.get_states(x)
        if self.is_continuous:
            action_mean = self.actor_mean(hidden)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            dist = Normal(action_mean, action_std)
            if action is None:
                action = dist.sample()
            logprob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)
        else:
            logits = self.actor(hidden)
            dist = Categorical(logits=logits)
            if action is None:
                action = dist.sample()
            logprob = dist.log_prob(action)
            entropy = dist.entropy()
        value = self.critic(hidden)
        return action, logprob, entropy, value

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

    print(f"Config Args: {args}")

    # Environment setup
    if "ale" in args.gym_id.lower():
        import ale_py
        gym.register_envs(ale_py)
        envs_lst = [make_atari_env(args.gym_id, args.seed + i, i, args.capture_video, 
                                   run_name, frame_stack=args.frame_stack) for i in range(args.num_envs)]
    elif "minigrid" in args.gym_id.lower():
        envs_lst = [make_minigrid_env(args.gym_id, args.seed + i, i, args.capture_video, 
                                      run_name, agent_view_size=3, tile_size=28, max_episode_steps=96, frame_stack=args.frame_stack) for i in range(args.num_envs)]
    elif "poc" in args.gym_id.lower():
        envs_lst = [make_poc_env(args.gym_id, args.seed + i, i, args.capture_video,
                                 run_name, step_size=0.02, glob=False, freeze=True, max_episode_steps=96) for i in range(args.num_envs)]
    elif args.gym_id == "MortarMayhem-Grid-v0":
        envs_lst = [make_memory_gym_env(args.gym_id, args.seed + i, i, args.capture_video,
                                        run_name) for i in range(args.num_envs)]
    elif args.gym_id in ["HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"]:
        envs_lst = [make_continuous_env(args.gym_id, args.seed + i, i, args.capture_video,
                                        run_name, obs_stack=args.obs_stack) for i in range(args.num_envs)]
    else:
        envs_lst = [make_classic_env(args.gym_id, args.seed + i, i, args.capture_video, 
                                     run_name, masked_indices=args.masked_indices, obs_stack=args.obs_stack) for i in range(args.num_envs)]
    envs = gym.vector.SyncVectorEnv(envs_lst, autoreset_mode="SameStep")

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

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
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Start the game
    global_step = 0
    start_time = time.time()
    episode_infos = deque(maxlen=100)
    next_obs, _ = envs.reset(seed=[args.seed + i for i in range(args.num_envs)])
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
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

            # Action logic
            inf_start = time.time()
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
            inference_time_total += (time.time() - inf_start)
            values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Execute the game and log data
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(done).to(device)


            final_info = info.get('final_info')
            # print(f"Info: {info}")
            # print(f"FInal Info: {final_info}")
            if "ale" in args.gym_id.lower():
                if final_info is not None and len(final_info) > 0:

                    # ① 取布尔掩码；若你的 wrapper 没写 _r/_l，就默认全 True
                    mask = final_info.get("_r")
                    if mask is None:
                        mask = np.ones(final_info["r"].shape[0], dtype=bool)

                    # ② 把 True 的那几个 episode 捞出来
                    episodic_returns = final_info["r"][mask].flatten()    # 1-D array
                    episodic_lengths = final_info["l"][mask].flatten()

                    # ③ 若确实有 episode 结束，就计算平均并记录
                    if episodic_returns.size > 0:
                        avg_return = float(f"{np.mean(episodic_returns):.3f}")
                        avg_length = float(f"{np.mean(episodic_lengths):.3f}")

                        episode_infos.append({"r": avg_return, "l": avg_length})
                        writer.add_scalar("charts/episode_return",  avg_return,  global_step)
                        writer.add_scalar("charts/episode_length",  avg_length,  global_step)
                        avg_inference_latency = inference_time_total / args.num_steps
                        writer.add_scalar("metrics/inference_latency", avg_inference_latency, global_step)
            elif args.gym_id in ["HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"]:
                if final_info is not None and len(final_info) > 0:
                    # valid_entries = [entry for entry in final_info if entry is not None and 'episode' in entry]
                    # print(f"Valid entries: {valid_entries}")
                    # if valid_entries:
                    episode_data = final_info['episode']
                    episodic_returns = episode_data['r']
                    episodic_lengths = episode_data['l']
                    # episodic_returns = [entry['episode']['r'] for entry in valid_entries]
                    # episodic_lengths = [entry['episode']['l'] for entry in valid_entries]
                    avg_return = float(f'{np.mean(episodic_returns):.3f}')
                    avg_length = float(f'{np.mean(episodic_lengths):.3f}')
                    episode_infos.append({'r': avg_return, 'l': avg_length})
                    writer.add_scalar("charts/episode_return", avg_return, global_step)
                    writer.add_scalar("charts/episode_length", avg_length, global_step)


        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
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

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)

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
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds] if not agent.is_continuous else b_actions[mb_inds])
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
