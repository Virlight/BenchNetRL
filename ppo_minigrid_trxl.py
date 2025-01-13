import argparse
import os
import random
import time
from collections import deque
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="MiniGrid-DoorKey-16x16-v0",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=2.75e-4,
        help="the initial learning rate of the optimizer")
    parser.add_argument("--final-lr", type=float, default=1.0e-5,
        help="the final learning rate after annealing")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=100000000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-minigrid",
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
        help="use a clipped loss for the value function")
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
        help="the coefficient of the observation reconstruction loss")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    return args

def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(
            gym_id,
            agent_view_size=3,
            tile_size=28,
            render_mode="rgb_array" if capture_video else None,
        )
        env = ImgObsWrapper(RGBImgPartialObsWrapper(env, tile_size=28))
        env = gym.wrappers.TimeLimit(env, 96)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
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

class PositionalEncoding(nn.Module):
    def __init__(self, dim, min_timescale=2.0, max_timescale=1e4):
        super().__init__()
        freqs = torch.arange(0, dim, min_timescale)
        inv_freqs = max_timescale ** (-freqs / dim)
        self.register_buffer("inv_freqs", inv_freqs)

    def forward(self, seq_len):
        seq = torch.arange(seq_len - 1, -1, -1.0)
        sinusoidal_inp = rearrange(seq, "n -> n ()") * rearrange(self.inv_freqs, "d -> () d")
        pos_emb = torch.cat((sinusoidal_inp.sin(), sinusoidal_inp.cos()), dim=-1)
        return pos_emb

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads

        assert self.head_size * num_heads == embed_dim, "Embedding dimension must be divisible by num_heads"

        self.values = nn.Linear(self.head_size, self.head_size, bias=False)
        self.keys = nn.Linear(self.head_size, self.head_size, bias=False)
        self.queries = nn.Linear(self.head_size, self.head_size, bias=False)
        self.fc_out = nn.Linear(self.num_heads * self.head_size, embed_dim)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.num_heads, self.head_size)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_size)
        query = query.reshape(N, query_len, self.num_heads, self.head_size)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_dim ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.num_heads * self.head_size)
        return self.fc_out(out), attention

class TransformerLayer(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(dim, num_heads)
        self.layer_norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.layer_norm_attn = nn.LayerNorm(dim)
        self.fc_projection = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())

    def forward(self, value, key, query, mask):
        query_ = self.layer_norm_q(query)
        value = self.norm_kv(value)
        key = value
        attention, attention_weights = self.attention(value, key, query_, mask)
        x = attention + query
        x_ = self.layer_norm_attn(x)
        forward = self.fc_projection(x_)
        out = forward + x
        return out, attention_weights

class Transformer(nn.Module):
    def __init__(self, num_layers, dim, num_heads, max_episode_steps, positional_encoding):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.positional_encoding = positional_encoding
        if positional_encoding == "absolute":
            self.pos_embedding = PositionalEncoding(dim)
        elif positional_encoding == "learned":
            self.pos_embedding = nn.Parameter(torch.randn(max_episode_steps, dim))
        self.transformer_layers = nn.ModuleList([TransformerLayer(dim, num_heads) for _ in range(num_layers)])

    def forward(self, x, memories, mask, memory_indices):
        if self.positional_encoding == "absolute":
            pos_embedding = self.pos_embedding(self.max_episode_steps)[memory_indices]
            memories = memories + pos_embedding.unsqueeze(2)
        elif self.positional_encoding == "learned":
            memories = memories + self.pos_embedding[memory_indices].unsqueeze(2)

        out_memories = []
        for i, layer in enumerate(self.transformer_layers):
            out_memories.append(x.detach())
            x, attention_weights = layer(
                memories[:, :, i], memories[:, :, i], x.unsqueeze(1), mask
            )
            x = x.squeeze()
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
        return x, torch.stack(out_memories, dim=1)

class Agent(nn.Module):
    def __init__(self, args, observation_space, action_space_shape, max_episode_steps):
        super().__init__()
        self.obs_shape = observation_space.shape
        self.max_episode_steps = max_episode_steps

        if len(self.obs_shape) > 1:
            self.encoder = nn.Sequential(
                layer_init(nn.Conv2d(3, 32, 8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * 7 * 7, args.trxl_dim)),
                nn.ReLU(),
            )
        else:
            self.encoder = layer_init(nn.Linear(observation_space.shape[0], args.trxl_dim))

        self.transformer = Transformer(
            args.trxl_num_layers, args.trxl_dim, args.trxl_num_heads, self.max_episode_steps, args.trxl_positional_encoding
        )

        self.hidden_post_trxl = nn.Sequential(
            layer_init(nn.Linear(args.trxl_dim, args.trxl_dim)),
            nn.ReLU(),
        )

        self.actor_branches = nn.ModuleList(
            [
                layer_init(nn.Linear(args.trxl_dim, out_features=num_actions), np.sqrt(0.01))
                for num_actions in action_space_shape
            ]
        )
        self.critic = layer_init(nn.Linear(args.trxl_dim, 1), 1)

        self.reconstruction_coef = args.reconstruction_coef
        if self.reconstruction_coef > 0.0:
            self.transposed_cnn = nn.Sequential(
                layer_init(nn.Linear(args.trxl_dim, 64 * 7 * 7)),
                nn.ReLU(),
                nn.Unflatten(1, (64, 7, 7)),
                layer_init(nn.ConvTranspose2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                layer_init(nn.ConvTranspose2d(64, 32, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.ConvTranspose2d(32, 3, 8, stride=4)),
                nn.Sigmoid(),
            )

    def get_value(self, x, memory, memory_mask, memory_indices):
        if len(self.obs_shape) > 1:
            x = self.encoder(x.permute((0, 3, 1, 2)) / 255.0)
        else:
            x = self.encoder(x)
        x, _ = self.transformer(x, memory, memory_mask, memory_indices)
        x = self.hidden_post_trxl(x)
        return self.critic(x).flatten()

    def get_action_and_value(self, x, memory, memory_mask, memory_indices, action=None):
        if len(self.obs_shape) > 1:
            x = self.encoder(x.permute((0, 3, 1, 2)) / 255.0)
        else:
            x = self.encoder(x)
        x, memory = self.transformer(x, memory, memory_mask, memory_indices)
        x = self.hidden_post_trxl(x)
        self.x = x
        probs = [Categorical(logits=branch(x)) for branch in self.actor_branches]
        if action is None:
            action = torch.stack([dist.sample() for dist in probs], dim=1)
        log_probs = [dist.log_prob(action[:, i]) for i, dist in enumerate(probs)]
        entropies = torch.stack([dist.entropy() for dist in probs], dim=1).sum(1).reshape(-1)
        return action, torch.stack(log_probs, dim=1), entropies, self.critic(x).flatten(), memory

    def reconstruct_observation(self):
        x = self.transposed_cnn(self.x)
        return x.permute((0, 2, 3, 1))

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available on this system.")

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    torch.set_default_device(device)

    # Env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    observation_space = envs.single_observation_space
    action_space_shape = ((envs.single_action_space.n,) if isinstance(envs.single_action_space, gym.spaces.Discrete)
                          else tuple(envs.single_action_space.nvec))

    env_current_episode_step = torch.zeros((args.num_envs,), dtype=torch.long)
    max_episode_steps = envs.envs[0].spec.max_episode_steps
    if not max_episode_steps:
        envs.envs[0].reset()
        max_episode_steps = envs.envs[0].max_episode_steps
    if max_episode_steps <= 0:
        max_episode_steps = 1024
    args.trxl_memory_length = min(args.trxl_memory_length, max_episode_steps)

    agent = Agent(args, observation_space, action_space_shape, max_episode_steps).to(device)
    optimizer = optim.AdamW(agent.parameters(), lr=args.learning_rate)
    bce_loss = nn.BCELoss()

    # Count and log parameters after agent initialization
    if args.track:
        total_params = sum(p.numel() for p in agent.parameters())
        trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
        import wandb
        wandb.config.update({
            "total_parameters": total_params,
            "trainable_parameters": trainable_params
        }, allow_val_change=True)

    # Storage setup
    rewards = torch.zeros((args.num_steps, args.num_envs))
    actions = torch.zeros((args.num_steps, args.num_envs, len(action_space_shape)), dtype=torch.long)
    dones = torch.zeros((args.num_steps, args.num_envs))
    obs = torch.zeros((args.num_steps, args.num_envs) + observation_space.shape)
    log_probs = torch.zeros((args.num_steps, args.num_envs, len(action_space_shape)))
    values = torch.zeros((args.num_steps, args.num_envs))
    stored_memories = []
    stored_memory_masks = torch.zeros((args.num_steps, args.num_envs, args.trxl_memory_length), dtype=torch.bool)
    stored_memory_index = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long)
    stored_memory_indices = torch.zeros((args.num_steps, args.num_envs, args.trxl_memory_length), dtype=torch.long)

    global_step = 0
    start_time = time.time()
    episode_infos = deque(maxlen=100)
    next_obs, _ = envs.reset(seed=[args.seed + i for i in range(args.num_envs)])
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs)
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

    for iteration in range(1, args.num_iterations + 1):
        sampled_episode_infos = []
        do_anneal = args.anneal_lr and (args.total_timesteps > 0)  # or your own logic
        steps_to_anneal = args.num_iterations * args.batch_size if not hasattr(args, "anneal_steps") else args.anneal_steps
        if steps_to_anneal <= 0:
            steps_to_anneal = args.num_iterations * args.batch_size
        frac = 1 - global_step / steps_to_anneal if do_anneal and global_step < steps_to_anneal else 0
        lr = (args.learning_rate - args.final_lr) * frac + args.final_lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        ent_coef = (args.init_ent_coef - args.final_ent_coef) * frac + args.final_ent_coef

        # Prepare current environment memory references
        stored_memories = [next_memory[e] for e in range(args.num_envs)]
        for e in range(args.num_envs):
            stored_memory_index[:, e] = e

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            stored_memory_masks[step] = memory_mask[torch.clip(env_current_episode_step, 0, args.trxl_memory_length - 1)]
            stored_memory_indices[step] = memory_indices[env_current_episode_step]

            with torch.no_grad():
                memory_window = batched_index_select(next_memory, 1, stored_memory_indices[step])
                action, logprob, _, value, new_memory = agent.get_action_and_value(
                    next_obs, memory_window, stored_memory_masks[step], stored_memory_indices[step]
                )
                next_memory[torch.arange(args.num_envs), env_current_episode_step] = new_memory
                actions[step], log_probs[step], values[step] = action, logprob, value

            # Step environment
            next_obs_, reward, terminations, truncations, info = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs_).to(device)
            next_done = torch.Tensor(next_done).to(device)

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
            start_idx = torch.clip(env_current_episode_step - args.trxl_memory_length, 0)
            end_idx = torch.clip(env_current_episode_step, args.trxl_memory_length)
            indices = torch.stack([torch.arange(start_idx[b], end_idx[b]) for b in range(args.num_envs)]).long()
            memory_window = batched_index_select(next_memory, 1, indices)
            next_value = agent.get_value(
                next_obs, memory_window,
                memory_mask[torch.clip(env_current_episode_step, 0, args.trxl_memory_length - 1)],
                stored_memory_indices[-1],
            )
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # Flatten the batch
        b_obs = obs.reshape(-1, *obs.shape[2:])
        b_logprobs = log_probs.reshape(-1, *log_probs.shape[2:])
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

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                mb_advantages = mb_advantages.unsqueeze(1).repeat(1, len(action_space_shape))

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = torch.exp(logratio)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + (newvalue - b_values[mb_inds]).clamp(
                        -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * args.vf_coef

                # Reconstruction
                r_loss = torch.tensor(0.0)
                if agent.reconstruction_coef > 0.0:
                    r_loss = bce_loss(agent.reconstruct_observation(), b_obs[mb_inds] / 255.0)
                    loss += agent.reconstruction_coef * r_loss

                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

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

    if args.track and args.capture_video:
        import wandb
        wandb.save(f"videos/{run_name}/*.mp4")
        wandb.save(f"videos/{run_name}/*.json")
        video_path = f"videos/{run_name}"
        video_files = [f for f in os.listdir(video_path) if f.endswith(('.mp4', '.gif'))]
        for video_file in video_files:
            wandb.log({"video": wandb.Video(os.path.join(video_path, video_file), fps=4, format="mp4")})

    writer.close()
    envs.close()