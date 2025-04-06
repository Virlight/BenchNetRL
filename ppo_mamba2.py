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
from types import SimpleNamespace

from gae import compute_advantages
from exp_utils import add_common_args, setup_logging, finish_logging
from env_utils import make_atari_env, make_minigrid_env, make_poc_env, make_classic_env, make_memory_gym_env, make_continuous_env
from layers import layer_init
from mamba_ssm import Mamba

def parse_args():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--hidden-dim", type=int, default=512,
        help="the hidden dimension of the model")
    parser.add_argument("--d-state", type=int, default=16,
        help="SSM state expansion factor for Mamba")
    parser.add_argument("--d-conv", type=int, default=4,
        help="local convolution width for Mamba")
    parser.add_argument("--expand", type=int, default=2,
        help="expansion factor for the Mamba block")
    parser.add_argument("--mamba-lr", type=float, default=1e-4,
        help="learning rate for Mamba parameters (lower than base LR)")
    parser.add_argument("--dt-init", type=str, default="random", choices=["constant", "random"],
        help="Initialization method for dt projection weights")
    parser.add_argument("--dt-scale", type=float, default=1.0,
        help="Scaling factor for dt initialization")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

class Agent(nn.Module):
    def __init__(self, envs, args):
        super(Agent, self).__init__()
        self.obs_space = envs.single_observation_space
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
        self.mamba = Mamba(
            d_model=args.hidden_dim,
            d_state=args.d_state,
            d_conv=args.d_conv,
            expand=args.expand,
            dt_scale=args.dt_scale,
            dt_init=args.dt_init,
        )
        self.mamba.layer_idx = 0
        self.norm = nn.LayerNorm(self.args.hidden_dim)

        self.post_mamba_mlp = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
        )   
        
        self.actor = layer_init(nn.Linear(args.hidden_dim, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(args.hidden_dim, 1), std=1.0)

    def forward_sequence_with_mask(self, x, dones, init_mamba_state=None):
        """
        x: [T, B, H, W, C] – rollout observations.
        dones: [T, B] – 1 indicates termination at that time step.
        init_mamba_state: a tuple (conv_state, ssm_state) of initial states.
        """
        T, B = x.shape[:2]
        # Flatten time and batch for encoding.
        x_flat = x.reshape(-1, *x.shape[2:])  # [T*B, H, W, C]
        features = self.get_states(x_flat)      # [T*B, hidden_dim]
        features = features.reshape(T, B, -1)    # [T, B, hidden_dim]

        outputs = []
        current_state = init_mamba_state  # current_state = (conv_state, ssm_state)
        for t in range(T):
            # Get features for time step t: shape [B, 1, hidden_dim]
            current_feature = features[t].unsqueeze(1)
            # Compute one recurrent step.
            out, new_conv_state, new_ssm_state = self.mamba.step(
                current_feature, current_state[0], current_state[1]
            )
            # Create a mask for environments that are done.
            # (Assumes dones[t] is 1 for done, 0 for not done.)
            done_mask = (1 - dones[t]).unsqueeze(-1)  # shape [B, 1]
            # Reset states for environments where done_mask==0.
            new_conv_state = new_conv_state * done_mask.unsqueeze(-1)
            new_ssm_state = new_ssm_state * done_mask.unsqueeze(-1)

            outputs.append(out.squeeze(1))  # remove the time dim added by unsqueeze(1)
            current_state = (new_conv_state, new_ssm_state)
        
        # Stack outputs to shape [T, B, hidden_dim].
        out_seq = torch.stack(outputs, dim=0)
        # Add the residual connection from the original features.
        out_seq = self.post_mamba_mlp(out_seq + features)
        out_seq = self.norm(out_seq)
        return out_seq

    def get_states(self, x):
        if "minigrid" in self.args.gym_id.lower() or "mortar" in self.args.gym_id.lower():
            x = x.permute(0, 3, 1, 2) / 255.0
        if "ale/" in self.args.gym_id.lower():
            x = x / 255.0
        hidden = self.encoder(x)
        return hidden

    def get_value(self, x, mamba_state):
        encoded = self.get_states(x)
        current = encoded.unsqueeze(1)  # (B, 1, hidden_dim)
        out, new_conv_state, new_ssm_state = self.mamba.step(current, mamba_state[0], mamba_state[1])
        out = self.post_mamba_mlp(out) + current
        out = self.norm(out)
        hidden = out.squeeze(1)  # (B, hidden_dim)
        value = self.critic(hidden).flatten()
        return value, (new_conv_state, new_ssm_state)

    def get_action_and_value(self, x, mamba_state, action=None):
        encoded = self.get_states(x)
        current = encoded.unsqueeze(1)
        out, new_conv_state, new_ssm_state = self.mamba.step(current, mamba_state[0], mamba_state[1])
        out = self.post_mamba_mlp(out) + current
        out = self.norm(out)
        hidden = out.squeeze(1)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), (new_conv_state, new_ssm_state)


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
                                 run_name, step_size=0.02, glob=False, freeze=True, max_episode_steps=96) for i in range(args.num_envs)]
    elif args.gym_id == "MortarMayhem-Grid-v0":
        envs_lst = [make_memory_gym_env(args.gym_id, args.seed + i, i, args.capture_video,
                                        run_name) for i in range(args.num_envs)]
    elif args.gym_id in ["HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"]:
        envs_lst = [make_continuous_env(args.gym_id, args.seed + i, i, args.capture_video,
                                        run_name) for i in range(args.num_envs)]
    else:
        envs_lst = [make_classic_env(args.gym_id, args.seed + i, i, args.capture_video, 
                                     run_name) for i in range(args.num_envs)]
    envs = gym.vector.SyncVectorEnv(envs_lst)
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam([
        {"params": agent.encoder.parameters()},
        {"params": agent.norm.parameters()},
        {"params": agent.actor.parameters()},
        {"params": agent.critic.parameters()},
        {"params": agent.mamba.parameters(), "lr": args.mamba_lr}
    ], lr=args.learning_rate, eps=1e-5)

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
    conv_state, ssm_state = agent.mamba.allocate_inference_cache(args.num_envs, max_seqlen=1)
    next_mamba_state = (conv_state, ssm_state)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        update_start_time = time.time()
        initial_mamba_state = (next_mamba_state[0].clone(), next_mamba_state[1].clone())
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
                action, logprob, _, value, next_mamba_state = agent.get_action_and_value(
                    next_obs, next_mamba_state
                )
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

            for env_id, d in enumerate(done):
                if d:
                    next_mamba_state[0][env_id].zero_()
                    next_mamba_state[1][env_id].zero_()

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
            next_value, _ = agent.get_value(next_obs, next_mamba_state)
            next_value = next_value.reshape(1, -1)
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
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)

        # Initialize accumulators for metrics
        clipfracs = []
        total_loss_list = []
        pg_loss_list = []
        v_loss_list = []
        entropy_list = []
        grad_norm_list = []
        mamba_grad_norm_list = []
        approx_kl_list = []
        old_approx_kl_list = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

                # Get the initial Mamba state for this minibatch from the rollout.
                init_state = (initial_mamba_state[0][mbenvinds].clone(),
                              initial_mamba_state[1][mbenvinds].clone())

                # Slice the rollout
                mb_obs = obs[:, mbenvinds] # shape: [T, minibatch_size, H, W, C]
                mb_dones = dones[:, mbenvinds]  # Slice dones to match minibatch.
                full_seq_output = agent.forward_sequence_with_mask(mb_obs, mb_dones, init_state)  # shape: [T, minibatch_size, hidden_dim]
                T, B, hidden_dim = full_seq_output.shape
                flat_features = full_seq_output.reshape(-1, hidden_dim)  # shape: [T*B, hidden_dim]

                # Get logits and values:
                logits = agent.actor(flat_features)                      # shape: [T*B, num_actions]
                probs = Categorical(logits=logits)
                new_logprobs = probs.log_prob(b_actions[mb_inds])
                new_entropies = probs.entropy()
                new_values = agent.critic(flat_features).reshape(-1)

                # Compute the ratio and approximate KL divergence:
                logratio = new_logprobs - b_logprobs[mb_inds]
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
                if args.clip_vloss:
                    mb_values_flat = b_values[mb_inds]
                    v_loss_unclipped = (new_values - b_returns[mb_inds]) ** 2
                    v_clipped = mb_values_flat + torch.clamp(new_values - mb_values_flat,
                                                            -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_values - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = new_entropies.mean()

                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                total_grad_norm = 0.0
                for p in agent.parameters():
                    if p.grad is not None:
                        total_grad_norm += p.grad.data.norm(2).item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                grad_norm_list.append(total_grad_norm)

                mamba_grad_norm = 0.0
                for p in agent.mamba.parameters():
                    if p.grad is not None:
                        mamba_grad_norm += p.grad.data.norm(2).item() ** 2
                mamba_grad_norm = mamba_grad_norm ** 0.5
                mamba_grad_norm_list.append(mamba_grad_norm)
                grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                # Append metrics for this minibatch
                total_loss_list.append(loss.item())
                pg_loss_list.append(pg_loss.item())
                v_loss_list.append(v_loss.item())
                entropy_list.append(entropy_loss.item())
                #grad_norm_list.append(grad_norm.item())
                #mamba_grad_norm_list.append(mamba_grad_norm.item())
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
        avg_mamba_grad_norm = np.mean(mamba_grad_norm_list)
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
        writer.add_scalar("losses/mamba_grad_norm", avg_mamba_grad_norm, global_step)
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
