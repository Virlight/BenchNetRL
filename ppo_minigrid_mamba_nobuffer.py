import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from mamba_ssm import Mamba
from envs.poc_memory_env import PocMemoryEnv

from gae import compute_advantages
from exp_utils import setup_logging, finish_logging

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def make_env_poc(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = PocMemoryEnv(step_size=0.2, glob=False, freeze=False, max_episode_steps=32)
        return env
    return thunk

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="PocMemoryEnv",
                        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="the learning rate")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed for the experiment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
                        help="total timesteps of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, torch.backends.cudnn.deterministic=True")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, CUDA will be enabled")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, the experiment will be tracked with wandb")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-poc-memory",
                        help="the wandb project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, videos will be captured")

    # PPO arguments
    parser.add_argument("--num-envs", type=int, default=8,
                        help="the number of parallel environments")
    parser.add_argument("--num-steps", type=int, default=128,
                        help="the number of steps per rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="toggle learning rate annealing")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="lambda for GAE")
    parser.add_argument("--num-minibatches", type=int, default=4,
                        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
                        help="the number of epochs for PPO update")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="normalize advantages")
    parser.add_argument("--clip-coef", type=float, default=0.1,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="toggle clipped value loss")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="value loss coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="max gradient norm for clipping")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="target KL divergence threshold")
    
    # Mamba-specific arguments (for our recurrent cell)
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="hidden dimension for the encoder and Mamba")
    parser.add_argument("--use-mean-hidden", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, use mean hidden state instead of last token")
    parser.add_argument("--d-state", type=int, default=16,
                        help="SSM state expansion factor for Mamba")
    parser.add_argument("--d-conv", type=int, default=4,
                        help="local convolution width for Mamba")
    parser.add_argument("--expand", type=int, default=2,
                        help="expansion factor for the Mamba block")
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

class Agent(nn.Module):
    def __init__(self, args, observation_space, action_space):
        super(Agent, self).__init__()
        self.hidden_dim = args.hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(observation_space.shape[0], args.hidden_dim),
            nn.ReLU(),
        )
        
        self.mamba = Mamba(
            d_model=args.hidden_dim,
            d_state=args.d_state,
            d_conv=args.d_conv,
            expand=args.expand,
        )
        
        self.actor = layer_init(nn.Linear(args.hidden_dim, action_space.n), std=np.sqrt(0.01))
        self.critic = layer_init(nn.Linear(args.hidden_dim, 1), std=1.0)

    def get_value(self, x, mamba_state):
        encoded = self.encoder(x)  # (B, hidden_dim)
        current = encoded.unsqueeze(1)  # (B, 1, hidden_dim)
        out, new_conv_state, new_ssm_state = self.mamba.step(current, mamba_state[0], mamba_state[1])
        hidden = out.squeeze(1)  # (B, hidden_dim)
        value = self.critic(hidden).flatten()
        return value, (new_conv_state, new_ssm_state)

    def get_action_and_value(self, x, mamba_state, action=None):
        encoded = self.encoder(x)
        current = encoded.unsqueeze(1)
        out, new_conv_state, new_ssm_state = self.mamba.step(current, mamba_state[0], mamba_state[1])
        hidden = out.squeeze(1)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        logprob = probs.log_prob(action)
        entropy = probs.entropy()
        value = self.critic(hidden).flatten()
        return action, logprob, entropy, value, (new_conv_state, new_ssm_state)

def main():
    args = parse_args()
    writer, run_name = setup_logging(args)

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    torch.set_default_device(device)

    # Create PocMemoryEnv (Proof of Concept Memory Environment)
    envs = gym.vector.SyncVectorEnv(
        [make_env_poc(args.gym_id, args.seed + i, i, args.capture_video, run_name)
         for i in range(args.num_envs)]
    )
    observation_space = envs.single_observation_space
    action_space = envs.single_action_space

    # Create the agent
    agent = Agent(args, observation_space, action_space).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # The Mamba block provides an allocate_inference_cache() method that returns a tuple of states
    conv_state, ssm_state = agent.mamba.allocate_inference_cache(args.num_envs, max_seqlen=1)
    next_mamba_state = (conv_state, ssm_state)

    # Rollout storage
    obs = torch.zeros((args.num_steps, args.num_envs) + observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long, device=device)
    log_probs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    next_obs, _ = envs.reset(seed=[args.seed + i for i in range(args.num_envs)])
    next_obs = torch.tensor(next_obs, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    global_step = 0
    start_time = time.time()
    from collections import deque
    episode_infos = deque(maxlen=100)

    for update in range(1, args.total_timesteps // args.batch_size + 1):
        # Optionally anneal the learning rate
        if args.anneal_lr:
            frac = 1.0 - (update - 1) / (args.total_timesteps // args.batch_size)
            lrnow = frac * args.learning_rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lrnow

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, entropy, value, new_mamba_state = agent.get_action_and_value(next_obs, next_mamba_state)
                values[step] = value
                actions[step] = action
                log_probs[step] = logprob
                next_mamba_state = new_mamba_state

            next_obs_np, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done_np = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward, device=device)
            next_obs = torch.tensor(next_obs_np, device=device)
            next_done = torch.tensor(done_np, device=device, dtype=torch.float32)

            # If an environment is done, reset its Mamba state
            for env_id, d in enumerate(done_np):
                if d:
                    next_mamba_state[0][env_id].zero_()
                    next_mamba_state[1][env_id].zero_()
                    if "final_info" in infos and infos["final_info"][env_id] is not None:
                        episode_infos.append(infos["final_info"][env_id])

        # Bootstrap value
        with torch.no_grad():
            next_value, _ = agent.get_value(next_obs, next_mamba_state)
            next_value = next_value.reshape(1, -1)
            advantages, returns = compute_advantages(
                rewards, values, dones, next_value, next_done,
                args.gamma, args.gae_lambda, args.gae, args.num_steps, device
            )

        # Flatten the rollout
        b_obs = obs.reshape(-1, *obs.shape[2:])
        b_logprobs = log_probs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # PPO update
        clipfracs = []
        for epoch in range(args.update_epochs):
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

                # For the update pass we do not backpropagate through time,
                # we use a freshly zeroed Mamba state for the minibatch
                mb_size = mb_obs.shape[0]
                batch_conv_state = torch.zeros(mb_size, agent.mamba.d_model * agent.mamba.expand,
                    agent.mamba.d_conv, device=device)
                batch_ssm_state = torch.zeros(mb_size, agent.mamba.d_model * agent.mamba.expand,
                    agent.mamba.d_state, device=device)
                batch_mamba_state = (batch_conv_state, batch_ssm_state)

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(mb_obs, batch_mamba_state, mb_actions)
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

        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/loss", loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

    finish_logging(args, writer, run_name, envs)

if __name__ == "__main__":
    main()
