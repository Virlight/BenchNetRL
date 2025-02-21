import argparse
import os
import random
import time
from distutils.util import strtobool

from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from gae import compute_advantages
from exp_utils import setup_logging, finish_logging
from layers import layer_init
from mamba_ssm import Mamba

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="MountainCarContinuous-v0",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=8000000,
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
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle whether or not to use a clipped loss for the value function")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--seq-len", type=int, default=4,
        help="sequence length for Mamba model")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id, render_mode="rgb_array") if capture_video else gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        input_dim = np.array(envs.single_observation_space.shape).prod()
        self.base_network = nn.Sequential(
            layer_init(nn.Linear(input_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.input_proj = layer_init(nn.Linear(64, 128))
        self.mamba = Mamba(
            d_model=128,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        self.critic = layer_init(nn.Linear(128, 1), std=1.0)
        self.actor_mean = layer_init(nn.Linear(128, np.prod(envs.single_action_space.shape)), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_states(self, x):
        # x has shape (batch_size, seq_len, obs_dim)
        batch_size, seq_len, obs_dim = x.shape
        x = x.view(batch_size * seq_len, obs_dim)
        base_out = self.base_network(x)          # Shape: (batch_size * seq_len, 64)
        proj_out = self.input_proj(base_out)     # Shape: (batch_size * seq_len, 128)
        proj_out = proj_out.view(batch_size, seq_len, -1)
        mamba_out = self.mamba(proj_out)         # Shape: (batch_size, seq_len, 128)
        return mamba_out

    def get_value(self, x):
        hidden = self.get_states(x)
        last_hidden = hidden[:, -1, :]
        return self.critic(last_hidden)

    def get_action_and_value(self, x, action=None):
        hidden = self.get_states(x)
        last_hidden = hidden[:, -1, :]
        action_mean = self.actor_mean(last_hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(last_hidden)

if __name__ == "__main__":
    args = parse_args()
    writer, run_name = setup_logging(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available on this system.")
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    torch.set_default_device(device)

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Count and log parameters after agent initialization
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
    obs_dim = envs.single_observation_space.shape[0]

    # Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset(seed=[args.seed + i for i in range(args.num_envs)])[0]).to(device)
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
            obs_sequences = np.array([list(obs_buffers[i]) for i in range(args.num_envs)])  # Shape: (num_envs, seq_len, obs_dim)
            obs_sequences = torch.tensor(obs_sequences, dtype=torch.float32).to(device)
            obs[step] = obs_sequences

            # Action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs_sequences)
                values[step] = value.flatten()
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

            if 'final_info' in info:
                final_info_array = np.array(info['final_info'])
                valid_indices = np.where(final_info_array != None)[0]
                valid_final_infos = final_info_array[valid_indices]
                episodic_returns = np.array([entry['episode']['r'] for entry in valid_final_infos if 'episode' in entry])
                episodic_lengths = np.array([entry['episode']['l'] for entry in valid_final_infos if 'episode' in entry])
                avg_return = float(f'{np.round(np.mean(episodic_returns), 3):.3f}')
                avg_length = float(f'{np.round(np.mean(episodic_lengths), 3):.3f}')
                print(f"global_step={global_step}, avg_return={avg_return}, avg_length={avg_length}")
                writer.add_scalar("charts/episodic_return", avg_return, global_step)
                writer.add_scalar("charts/episodic_length", avg_length, global_step)

        # Bootstrap value if not done
        with torch.no_grad():
            obs_sequences = np.array([list(obs_buffers[i]) for i in range(args.num_envs)])
            obs_sequences = torch.tensor(obs_sequences, dtype=torch.float32).to(device)
            next_value = agent.get_value(obs_sequences).view(1, -1)
            advantages, returns = compute_advantages(
                rewards, values, dones, next_value, next_done,
                args.gamma, args.gae_lambda, args.gae, args.num_steps, device
            )

        # Flatten the batch but keep the sequence dimension
        b_obs = obs.transpose(1, 0)  # Shape: (num_envs, num_steps, seq_len, obs_dim)
        b_logprobs = logprobs.transpose(1, 0)
        b_actions = actions.transpose(1, 0)
        b_advantages = advantages.transpose(1, 0)
        b_returns = returns.transpose(1, 0)
        b_values = values.transpose(1, 0)

        # Optimizing the policy and value network
        assert args.num_envs * args.num_steps % args.minibatch_size == 0
        num_minibatches = args.num_envs * args.num_steps // args.minibatch_size
        mb_size = args.minibatch_size

        inds = np.arange(args.num_envs * args.num_steps)
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
            np.random.shuffle(inds)
            for start in range(0, args.num_envs * args.num_steps, mb_size):
                end = start + mb_size
                mb_inds = inds[start:end]
                env_inds = mb_inds // args.num_steps
                step_inds = mb_inds % args.num_steps

                mb_obs = b_obs[env_inds, step_inds].reshape(-1, sequence_length, obs_dim)
                mb_actions = b_actions[env_inds, step_inds].reshape(-1, *envs.single_action_space.shape)
                mb_logprobs = b_logprobs[env_inds, step_inds].reshape(-1)
                mb_advantages = b_advantages[env_inds, step_inds].reshape(-1)
                mb_returns = b_returns[env_inds, step_inds].reshape(-1)
                mb_values = b_values[env_inds, step_inds].reshape(-1)

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)
                newlogprob = newlogprob.view(-1)
                entropy = entropy.view(-1)
                newvalue = newvalue.view(-1)

                # Compute ratios
                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    # Calculate approx_kl
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
        print(f"SPS: {int(global_step / (time.time() - start_time))}")
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    finish_logging(args, writer, run_name, envs)
