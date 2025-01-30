import os
import csv
import time
import subprocess
import concurrent.futures
import pickle

from carbs import CARBS
from carbs import CARBSParams, Param, LogSpace, LinearSpace
from carbs import ObservationInParam

def run_experiment(params):
    results_file = "training_results.txt"
    if os.path.exists(results_file):
        os.remove(results_file)

    game_seed_pairs = [
        ("Breakout-v5", 1),
    ]

    learning_rate = params["learning_rate"]
    ent_coef      = params["ent_coef"]
    gamma         = params["gamma"]
    hidden_dim    = int(params["hidden_dim"])
    seq_len       = int(params["seq_len"])
    d_state       = int(params["d_state"])
    d_conv        = int(params["d_conv"])
    expand        = int(params["expand"])

    total_timesteps = 100_000
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        for (game, seed) in game_seed_pairs:
            cmd = [
                "python",
                "ppo_atari_mamba_envpool.py",
                f"--gym-id={game}",
                f"--seed={seed}",
                f"--learning-rate={learning_rate}",
                f"--ent-coef={ent_coef}",
                f"--gamma={gamma}",
                f"--hidden-dim={hidden_dim}",
                f"--seq-len={seq_len}",
                f"--d-state={d_state}",
                f"--d-conv={d_conv}",
                f"--expand={expand}",
                f"--total-timesteps={total_timesteps}",
                "--num-steps=128",
                f"--num-envs=16",
            ]
            futures.append(executor.submit(subprocess.run, cmd, check=True))
        concurrent.futures.wait(futures)

    num_runs = len(game_seed_pairs)
    total_return = 0.0
    model_size = 0.0
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            for line in f:
                tokens = line.strip().split()
                for t in tokens:
                    if t.startswith("model_size="):
                        model_size_str = t.split("=")[1]
                        model_size = float(model_size_str)
                try:
                    final_return = float(tokens[-1])
                    total_return += final_return
                except ValueError:
                    pass

    avg_return = total_return / num_runs if num_runs else 0.0

    csv_file = "carbs_runs_2.csv"
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "learning_rate",
                "ent_coef",
                "gamma",
                "hidden_dim",
                "seq_len",
                "d_state",
                "d_conv",
                "expand",
                "avg_return",
                "model_size",
            ])
        writer.writerow([
            learning_rate,
            ent_coef,
            gamma,
            hidden_dim,
            seq_len,
            d_state,
            d_conv,
            expand,
            avg_return,
            model_size,
        ])

    return avg_return, model_size

if __name__ == "__main__":
    param_spaces = [
        Param(name="learning_rate", space=LogSpace(min=2e-4, max=3e-4, is_integer=False), search_center=2.5e-4),
        Param(name="ent_coef", space=LogSpace(min=1e-4, max=2e-2, is_integer=False), search_center=1e-3),
        Param(name="gamma", space=LinearSpace(min=0.90, max=0.99, scale=0.01, is_integer=False), search_center=0.95),
        Param(name="hidden_dim", space=LogSpace(is_integer=True, min=64, max=384, scale=0.3), search_center=256),
        Param(name="seq_len", space=LogSpace(is_integer=True, min=2, max=16, scale=0.3), search_center=8),
        Param(name="d_state", space=LogSpace(is_integer=True, min=8, max=32, scale=0.3), search_center=16),
        Param(name="d_conv", space=LogSpace(is_integer=True, min=2, max=4, scale=0.3), search_center=3),
        Param(name="expand", space=LogSpace(is_integer=True, min=1, max=4, scale=0.3), search_center=2),
    ]

    carbs_params = CARBSParams(
        better_direction_sign=1,
        resample_frequency=0,
        is_wandb_logging_enabled=False,
    )

    load = True
    if load:
        with open("carbs_state.pkl", "rb") as f:
            carbs = pickle.load(f)
    else:
        carbs = CARBS(carbs_params, param_spaces)

    n_trials = 3
    for _ in range(n_trials):
        suggestion_obj = carbs.suggest()
        suggestion = suggestion_obj.suggestion
        performance_metric, cost = run_experiment(suggestion)
        obs = ObservationInParam(input=suggestion, output=performance_metric, cost=cost)
        carbs.observe(obs)

    with open("carbs_state.pkl", "wb") as f:
        pickle.dump(carbs, f)