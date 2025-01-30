import os
import time
import csv
import subprocess
import concurrent.futures
import torch.nn as nn
from mamba_ssm import Mamba

from carbs import CARBS
from carbs import CARBSParams, Param, LogSpace, LinearSpace
from carbs import ObservationInParam

def compute_model_size(env_action_dim, hidden_dim, d_state, d_conv, expand):
    class DummyAgent(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv2d(1, 32, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 512),
                nn.ReLU(),
            )
            self.input_proj = nn.Linear(512, hidden_dim)
            self.mamba = Mamba(
                d_model=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.actor = nn.Linear(hidden_dim, env_action_dim)
            self.critic = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            pass
    agent = DummyAgent()
    total_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    return total_params

def run_experiment(params):
    """
    1) Launch `ppo_atari_mamba.py` multiple times in parallel (4 seeds).
    2) Parse the final returns from `training_results.txt`.
    3) Take an average of them as a performance metric.
    4) Return (performance_metric, cost_metric).
    """
    results_file = "training_results.txt"
    if os.path.exists(results_file):
        os.remove(results_file)

    game_seed_pairs = [
        ("ALE/Pong-v5", 1),
        ("ALE/Pong-v5", 2),
        # ("ALE/Pong-v5", 3),
        # ("ALE/Pong-v5", 4),
    ]

    learning_rate = params["learning_rate"]
    ent_coef      = params["ent_coef"]
    gamma         = params["gamma"]
    reconstruction_coef = params["reconstruction_coef"]
    hidden_dim    = int(params["hidden_dim"])
    seq_len       = int(params["seq_len"])
    d_state       = int(params["d_state"])
    d_conv        = int(params["d_conv"])
    expand        = int(params["expand"])

    env_action_dim = 6
    model_size = compute_model_size(
        env_action_dim=env_action_dim,
        hidden_dim=hidden_dim,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand
    )
    
    total_timesteps = 4000000
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for (game, seed) in game_seed_pairs:
            cmd = [
                "python",
                "ppo_atari_mamba.py",
                f"--gym-id={game}",
                f"--seed={seed}",
                f"--learning-rate={learning_rate}",
                f"--reconstruction-coef={reconstruction_coef}",
                f"--ent-coef={ent_coef}",
                f"--gamma={gamma}",
                f"--hidden-dim={hidden_dim}",
                f"--seq-len={seq_len}",
                f"--d-state={d_state}",
                f"--d-conv={d_conv}",
                f"--expand={expand}",
                f"--total-timesteps={total_timesteps}",
                "--num-steps=128",
                "--num-envs=8",
            ]
            futures.append(executor.submit(subprocess.run, cmd, check=True))
        concurrent.futures.wait(futures)

    end_time = time.time()
    total_time = end_time - start_time
    num_runs = len(game_seed_pairs)
    total_return = 0.0
    total_solved_steps = 0.0
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            for line in f:
                tokens = line.strip().split()
                try:
                    token = float(tokens[-1])
                    total_return += token
                except ValueError:
                    pass
    
    avg_return = total_return / num_runs
    avg_solved_steps = total_solved_steps / num_runs if num_runs > 0 else float(total_timesteps)

    csv_file = "carbs_runs_1.csv"
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "learning_rate", "reconstruction_coef", "ent_coef", "gamma",
                "hidden_dim", "seq_len", "d_state", "d_conv", "expand",
                "avg_return", "model_size", "total_time_sec"
            ])
        writer.writerow([
            learning_rate,
            reconstruction_coef,
            ent_coef,
            gamma,
            hidden_dim,
            seq_len,
            d_state,
            d_conv,
            expand,
            avg_return,
            model_size,
            total_time
        ])
    return avg_return, float(model_size)


if __name__ == "__main__":
    param_spaces = [
        Param(
            name="learning_rate",
            space=LogSpace(min=2e-4, max=3e-4, is_integer=False),
            search_center=2.5e-4
        ),
        Param(
            name="reconstruction_coef",
            space=LinearSpace(min=0.0, max=0.05, scale=0.005, is_integer=False),
            search_center=0.0
        ),
        Param(
            name="ent_coef",
            space=LogSpace(min=1e-4, max=2e-2, is_integer=False),
            search_center=1e-3
        ),
        Param(
            name="gamma",
            space=LinearSpace(min=0.90, max=0.99, scale=0.01, is_integer=False),
            search_center=0.95
        ),
        Param(
            name="hidden_dim",
            space=LogSpace(is_integer=True, min=64, max=384, scale=0.3),
            search_center=256
        ),
        Param(
            name="seq_len",
            space=LogSpace(is_integer=True, min=2, max=16, scale=0.3),
            search_center=8
        ),
        Param(
            name="d_state",
            space=LogSpace(is_integer=True, min=8, max=32, scale=0.3),
            search_center=16
        ),
        Param(
            name="d_conv",
            space=LogSpace(is_integer=True, min=2, max=4, scale=0.3),
            search_center=3
        ),
        Param(
            name="expand",
            space=LogSpace(is_integer=True, min=1, max=4, scale=0.3),
            search_center=2
        ),
    ]

    carbs_params = CARBSParams(
        better_direction_sign=1, 
        resample_frequency=0,
        is_wandb_logging_enabled=False,
    )
    carbs = CARBS(carbs_params, param_spaces)

    n_trials = 20
    for _ in range(n_trials):
        suggestion_obj = carbs.suggest()
        suggestion = suggestion_obj.suggestion
        performance_metric, cost = run_experiment(suggestion)
        obs = ObservationInParam(
            input=suggestion,
            output=performance_metric,
            cost=cost
        )
        carbs.observe(obs)
