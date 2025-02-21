import argparse
import random
import time
from collections import deque
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
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
    # We no longer use --seq-len since we use Transformer-XL style memory.
    parser.add_argument("--memory-length", type=int, default=20,
        help="Length of episodic memory (number of past tokens to retain)")
    parser.add_argument("--hidden-dim", type=int, default=256,
        help="Size of the hidden dimension for the Mamba model")
    parser.add_argument("--use-mean-hidden", type=lambda x: bool(strtobool(x)), default=False,
        help="If toggled, use the mean of all hidden states instead of the last token")
    parser.add_argument("--reconstruction-coef", type=float, default=0.0,
        help="Coefficient for optional observation reconstruction loss (0 disables it)")
    parser.add_argument("--d-state", type=int, default=16,
        help="State-space size for Mamba")
    parser.add_argument("--d-conv", type=int, default=4,
        help="Convolutional projection size for Mamba")
    parser.add_argument("--expand", type=int, default=2,
        help="Expansion factor in the Mamba state-space model")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.num_envs // args.num_minibatches)
    return args

# -------------------------------
# Agent using CNN encoder + episodic memory (Mamba SSM block)
#
# Instead of using Mamba's output as memory (which causes recurrent feedback),
# we now store the CNN–encoded token as memory.
# -------------------------------
class Agent(nn.Module):
    def __init__(self, args, observation_space, action_space, max_episode_steps):
        super(Agent, self).__init__()
        self.obs_shape = observation_space.shape
        self.max_episode_steps = max_episode_steps
        self.hidden_dim = args.hidden_dim
        self.memory_length = args.memory_length
        self.use_mean_hidden = args.use_mean_hidden

        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(observation_space.shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, args.hidden_dim)),
            nn.ReLU(),
        )

        # Mamba block is applied on a sequence of encoded tokens.
        self.mamba = Mamba(
            d_model=args.hidden_dim,
            d_state=args.d_state,
            d_conv=args.d_conv,
            expand=args.expand,
        )

        # Actor and Critic heads.
        self.actor = layer_init(nn.Linear(args.hidden_dim, action_space.n), std=np.sqrt(0.01))
        self.critic = layer_init(nn.Linear(args.hidden_dim, 1), std=1)

        # Optional observation reconstruction (for auxiliary loss)
        if args.reconstruction_coef > 0.0:
            self.transposed_cnn = nn.Sequential(
                layer_init(nn.Linear(args.hidden_dim, 64 * 7 * 7)),
                nn.ReLU(),
                nn.Unflatten(1, (64, 7, 7)),
                layer_init(nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1)),
                nn.ReLU(),
                layer_init(nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2)),
                nn.ReLU(),
                layer_init(nn.ConvTranspose2d(32, observation_space.shape[0], kernel_size=8, stride=4)),
                nn.Sigmoid(),
            )

    def get_value(self, x, memory):
        # x: (B, C, H, W)
        # memory: (B, T, hidden_dim) where T <= memory_length (stored CNN encodings)
        if len(self.obs_shape) > 1:
            encoded = self.encoder(x / 255.0)
        else:
            encoded = self.encoder(x)
        current = encoded.unsqueeze(1)  # (B, 1, hidden_dim)
        if memory is None or memory.size(1) == 0:
            seq = current
        else:
            seq = torch.cat([memory, current], dim=1)
        out = self.mamba(seq)  # (B, T+1, hidden_dim)
        if self.use_mean_hidden:
            hidden = out.mean(dim=1)
        else:
            hidden = out[:, -1, :]
        return self.critic(hidden).flatten()

    def get_action_and_value(self, x, memory, action=None):
        # Compute the CNN encoding.
        if len(self.obs_shape) > 1:
            encoded = self.encoder(x / 255.0)
        else:
            encoded = self.encoder(x)
        current = encoded.unsqueeze(1)  # (B, 1, hidden_dim)
        if memory is None or memory.size(1) == 0:
            seq = current
        else:
            seq = torch.cat([memory, current], dim=1)
        out = self.mamba(seq)  # (B, T+1, hidden_dim)
        if self.use_mean_hidden:
            hidden = out.mean(dim=1)
        else:
            hidden = out[:, -1, :]
        # Save detached hidden state for reconstruction loss.
        self.hidden = hidden.detach()
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden).flatten(), encoded

    def reconstruct_observation(self):
        x = self.transposed_cnn(self.hidden)
        return x


def main():
    args = parse_args()
    writer, run_name = setup_logging(args)

    # Seeding.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available on this system.")
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    torch.set_default_device(device)

    envs = gym.vector.SyncVectorEnv(
        [make_atari_env(args.gym_id, args.seed + i, i, args.capture_video, run_name, frame_stack=1)
         for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    observation_space = envs.single_observation_space
    action_space = envs.single_action_space

    # Use a fixed max_episode_steps (Atari environments typically have a fixed limit).
    max_episode_steps = 1024

    # Initialize episodic memory buffer per environment.
    # The memory now stores the CNN–encoded features.
    next_memory = torch.zeros((args.num_envs, args.memory_length, args.hidden_dim), device=device)
    # Also, record the memory used at each rollout time step for PPO update.
    stored_memory = torch.zeros((args.num_steps, args.num_envs, args.memory_length, args.hidden_dim), device=device)

    # Storage for rollout.
    obs = torch.zeros((args.num_steps, args.num_envs) + observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long, device=device)
    log_probs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    agent = Agent(args, observation_space, action_space, max_episode_steps).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    bce_loss = nn.BCELoss()

    # Reset environments.
    next_obs, _ = envs.reset(seed=[args.seed + i for i in range(args.num_envs)])
    next_obs = torch.tensor(next_obs, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    global_step = 0
    start_time = time.time()
    episode_infos = deque(maxlen=100)

    for update in range(1, args.total_timesteps // args.batch_size + 1):
        # Anneal learning rate if desired.
        if args.anneal_lr:
            frac = 1.0 - (update - 1) / (args.total_timesteps // args.batch_size)
            lrnow = frac * args.learning_rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lrnow

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            # Store the memory window used for this time step.
            stored_memory[step] = next_memory.clone()

            # Get action and value, using current episodic memory.
            with torch.no_grad():
                action, logprob, entropy, value, encoded = agent.get_action_and_value(next_obs, next_memory)
                values[step] = value
                actions[step] = action
                log_probs[step] = logprob

                # Update episodic memory:
                # Instead of using Mamba's output, store the CNN-encoded token.
                next_memory = torch.roll(next_memory, shifts=-1, dims=1)
                next_memory[:, -1, :] = encoded.detach()

            # Step the environments.
            next_obs_np, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done_np = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward, device=device)
            next_obs = torch.tensor(next_obs_np, device=device)
            next_done = torch.tensor(done_np, device=device, dtype=torch.float32)

            # If an environment is done, reset its episodic memory.
            for env_id, d in enumerate(done_np):
                if d:
                    next_memory[env_id].zero_()
                    if "final_info" in infos and infos["final_info"][env_id] is not None and "episode" in infos["final_info"][env_id]:
                        episode_infos.append(infos["final_info"][env_id]["episode"])
            
            final_info = infos.get('final_info')
            if final_info is not None and len(final_info) > 0:
                valid_entries = [entry for entry in final_info if entry is not None and 'episode' in entry]
                if valid_entries:
                    episodic_returns = [entry['episode']['r'] for entry in valid_entries]
                    episodic_lengths = [entry['episode']['l'] for entry in valid_entries]
                    avg_return = float(f'{np.mean(episodic_returns):.3f}')
                    avg_length = float(f'{np.mean(episodic_lengths):.3f}')
                    #print(f"global_step={global_step}, avg_return={avg_return}, avg_length={avg_length}")
                    writer.add_scalar("charts/episodic_return", avg_return, global_step)
                    writer.add_scalar("charts/episodic_length", avg_length, global_step)

        # Bootstrap value.
        with torch.no_grad():
            next_value = agent.get_value(next_obs, next_memory)
        advantages, returns = compute_advantages(
            rewards, values, dones, next_value, next_done,
            args.gamma, args.gae_lambda, args.gae, args.num_steps, device
        )

        # Flatten the rollout.
        b_obs = obs.reshape(-1, *obs.shape[2:])
        b_logprobs = log_probs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        # Flatten the stored memory: shape becomes (batch_size, memory_length, hidden_dim)
        b_stored_memory = stored_memory.reshape(-1, args.memory_length, args.hidden_dim)

        # PPO update.
        clipfracs = []
        for epoch in range(args.update_epochs):
            # For PPO update, we split by environment index.
            env_inds = np.arange(args.num_envs)
            np.random.shuffle(env_inds)
            for start in range(0, args.num_envs, args.minibatch_size):
                end = start + args.minibatch_size
                mb_env_inds = env_inds[start:end]
                mb_inds = []
                for t in range(args.num_steps):
                    for e in mb_env_inds:
                        mb_inds.append(t * args.num_envs + e)
                mb_inds = torch.tensor(mb_inds, device=device)
                
                mb_obs = b_obs[mb_inds]
                mb_actions = b_actions[mb_inds]
                mb_memory = b_stored_memory[mb_inds]  # Retrieve the stored memory window for these samples.

                # Pass the stored memory window to get_action_and_value.
                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(mb_obs, mb_memory, mb_actions)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = torch.exp(logratio)
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                if args.clip_vloss:
                    v_loss_clipped = b_values[mb_inds] + (newvalue - b_values[mb_inds]).clamp(-args.clip_coef, args.clip_coef)
                    v_loss = torch.max(v_loss_unclipped, (v_loss_clipped - b_returns[mb_inds]) ** 2).mean()
                else:
                    v_loss = v_loss_unclipped.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                if args.reconstruction_coef > 0.0:
                    recon = agent.reconstruct_observation()
                    target = mb_obs.float() / 255.0
                    r_loss = bce_loss(recon, target)
                    loss += args.reconstruction_coef * r_loss
                else:
                    r_loss = torch.tensor(0.0, device=device)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

        explained_var = 1 - np.var(b_returns.cpu().numpy() - b_values.cpu().numpy()) / (np.var(b_returns.cpu().numpy()) + 1e-8)
        sps = int(global_step / (time.time() - start_time))
        print(f"Update {update}: SPS={sps}, Return={np.mean([ep['r'] for ep in episode_infos]) if episode_infos else 0:.2f}, "
              f"pi_loss={pg_loss.item():.6f}, v_loss={v_loss.item():.6f}, entropy={entropy_loss.item():.6f}, "
              f"explained_var={explained_var:.6f}")
        
        # with open(f"log.txt", "a") as f:
        #     f.write(f"Update {update}: SPS={sps}, Return={np.mean([ep['r'] for ep in episode_infos]) if episode_infos else 0:.2f}, "
        #       f"pi_loss={pg_loss.item():.3f}, v_loss={v_loss.item():.3f}, entropy={entropy_loss.item():.3f}, "
        #       f"explained_var={explained_var:.3f}\n")

        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/loss", loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/reconstruction_loss", r_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

    finish_logging(args, writer, run_name, envs)
