import os
import csv
import time
import subprocess
import pickle

from carbs import CARBS, CARBSParams, Param, LogSpace, LinearSpace, LogitSpace
from carbs import ObservationInParam

def run_experiment(params):
    results_file = "training_results_atari.txt"
    if os.path.exists(results_file):
        os.remove(results_file)

    game_seed_pairs = [("Breakout-v5", 1)]

    learning_rate       = params["learning_rate"]
    ent_coef            = params["ent_coef"]
    gamma               = params["gamma"]
    # num_steps           = int(params["num_steps"])
    reconstruction_coef = params["reconstruction_coef"]
    update_epochs   = int(params["update_epochs"])
    clip_coef       = params["clip_coef"]
    # clip_vloss      = bool(int(params["clip_vloss"]))
    vf_coef         = params["vf_coef"]
    # max_grad_norm   = params["max_grad_norm"]
    # target_kl       = params["target_kl"]

    hidden_dim      = 512
    total_timesteps = 10_000_000
    num_envs        = 16
    num_minibatches = 8

    for (game, seed) in game_seed_pairs:
        cmd = [
            "python",
            "ppo_atari_envpool_carbs.py",
            f"--gym-id={game}",
            f"--seed={seed}",
            f"--learning-rate={learning_rate}",
            f"--ent-coef={ent_coef}",
            f"--gamma={gamma}",
            # f"--num-steps={num_steps}",
            f"--reconstruction-coef={reconstruction_coef}",
            f"--hidden-dim={hidden_dim}",
            f"--total-timesteps={total_timesteps}",
            f"--num-envs={num_envs}",
            f"--num-minibatches={num_minibatches}",
            f"--update-epochs={update_epochs}",
            f"--clip-coef={clip_coef}",
            # f"--clip-vloss={clip_vloss}",
            f"--vf-coef={vf_coef}",
            # f"--max-grad-norm={max_grad_norm}",
            # f"--target-kl={target_kl}",
        ]
        subprocess.run(cmd, check=True)

    # ALE/Breakout-v5 seed=1 model_size=123 total_time=45.67 avg_return_last_x=12.34
    total_return = 0.0
    total_time = 0.0
    total_model_size = 0.0
    num_runs = 0

    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            for line in f:
                tokens = line.strip().split()
                run_total_time = None
                run_avg_return = None
                run_model_size = None
                for token in tokens:
                    if token.startswith("total_time="):
                        try:
                            run_total_time = float(token.split("=")[1])
                        except ValueError:
                            run_total_time = 0.0
                    elif token.startswith("avg_return_last_x="):
                        try:
                            run_avg_return = float(token.split("=")[1])
                        except ValueError:
                            run_avg_return = 0.0
                    elif token.startswith("model_size="):
                        try:
                            run_model_size = float(token.split("=")[1])
                        except ValueError:
                            run_model_size = 0.0
                if (run_avg_return is not None and 
                    run_total_time is not None and 
                    run_model_size is not None):
                    total_return    += run_avg_return
                    total_time      += run_total_time
                    total_model_size += run_model_size
                    num_runs       += 1

    avg_return     = total_return / num_runs if num_runs > 0 else 0.0
    avg_time       = total_time   / num_runs if num_runs > 0 else 0.0
    avg_model_size = total_model_size / num_runs if num_runs > 0 else 0.0

    csv_file = "carbs_runs_atari.csv"
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "learning_rate",
                "ent_coef",
                "gamma",
                # "num_steps",
                "reconstruction_coef",
                "update_epochs",
                "clip_coef",
                # "clip_vloss",
                "vf_coef",
                # "max_grad_norm",
                # "target_kl",
                "avg_return",
                "avg_time",
                "model_size",
            ])
        writer.writerow([
            learning_rate,
            ent_coef,
            gamma,
            # num_steps,
            reconstruction_coef,
            update_epochs,
            clip_coef,
            # clip_vloss,
            vf_coef,
            # max_grad_norm,
            # target_kl,
            avg_return,
            avg_time,
            avg_model_size,
        ])

    return avg_return, avg_time, avg_model_size

if __name__ == "__main__":
    param_spaces = [
        Param(
            name="learning_rate",
            space=LogSpace(min=1e-5, max=1e-3, is_integer=False, scale=0.5),
            search_center=2.5e-4,
        ),
        Param(
            name="ent_coef",
            space=LogSpace(min=1e-3, max=1e-1, is_integer=False, scale=0.5),
            search_center=0.01,
        ),
        Param(
            name="gamma",
            space=LinearSpace(min=0.90, max=0.99, is_integer=False, scale=0.2),
            search_center=0.95,
        ),
        # Param(
        #     name="num_steps",
        #     space=LogSpace(is_integer=True, min=64, max=512),
        #     search_center=128,
        # ),
        Param(
            name="reconstruction_coef",
            space=LinearSpace(min=0.0, max=0.1, is_integer=False, scale=0.1),
            search_center=0.001,
        ),
        Param(
            name="update_epochs",
            space=LogSpace(min=1, max=10, is_integer=True),
            search_center=4,
        ),
        Param(
            name="clip_coef",
            space=LogSpace(min=0.05, max=0.3, is_integer=False),
            search_center=0.1,
        ),
        # Param(
        #     name="clip_vloss",
        #     space=LinearSpace(min=0, max=1, is_integer=True),
        #     search_center=1,
        # ),
        Param(
            name="vf_coef",
            space=LinearSpace(min=0.1, max=1.0, is_integer=False),
            search_center=0.5,
        ),
        # Param(
        #     name="max_grad_norm",
        #     space=LinearSpace(min=0.3, max=1.0, is_integer=False),
        #     search_center=0.5,
        # ),
        # Param(
        #     name="target_kl",
        #     space=LinearSpace(min=0.01, max=0.2, is_integer=False),
        #     search_center=0.05,
        # ),
    ]

    carbs_params = CARBSParams(
        better_direction_sign=1,
        resample_frequency=0,
        is_wandb_logging_enabled=False,
    )

    state_filename = "carbs_atari_state.pkl"
    if os.path.exists(state_filename):
        with open(state_filename, "rb") as f:
            carbs = pickle.load(f)
    else:
        carbs = CARBS(carbs_params, param_spaces)

    n_trials = 45
    for _ in range(n_trials):
        suggestion_obj = carbs.suggest()
        suggestion = suggestion_obj.suggestion
        print(f"Running trial with suggestion: {suggestion}")
        avg_return, avg_time, avg_model_size = run_experiment(suggestion)
        print(f"Result: avg_return = {avg_return:.3f}, cost = {avg_time:.2f} sec, model_size = {avg_model_size:.1f}")
        obs = ObservationInParam(input=suggestion, output=avg_return, cost=avg_time)
        carbs.observe(obs)

        if _ % 3 == 0:
            with open(state_filename, "wb") as f:
                pickle.dump(carbs, f)
