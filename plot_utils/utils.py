import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def read_csv_2_dict(csv_str, step=1):
    """
    Reads a CSV file and returns data as a list of lists. Skips rows based on step.
    """
    csv_reader = csv.reader(open(csv_str))
    i = 0
    list_all = []
    for row in csv_reader:
        if i > 0 and i % step == 0:  # Skip rows based on the step
            list_all.append([float(row[1]), float(row[2])])
        i += 1
    return list_all


def smoothen(data, window_size):
    """
    Smoothens the data using a moving average with a given window size.
    """
    res = np.zeros_like(data)
    for i in range(len(data)):
        if i > window_size:
            res[i] = np.mean(data[i - window_size:i])
        elif i > 0:
            res[i] = np.mean(data[:i])
        else:  # i == 0
            res[i] = data[i]
    return res


def draw(data_dict, smooth_window=5, title="", xlabel="Training Steps", ylabel="Average Reward", save_path=None, colors=None):
    if colors is None:
        colors = ['orange', 'hotpink', 'dodgerblue', 'mediumpurple', 'c', 
                  'cadetblue', 'steelblue', 'mediumslateblue', 'mediumturquoise']

    plt.figure(figsize=(10, 6))
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    print("Labels:", data_dict.keys())

    for i, (label, runs_data) in enumerate(data_dict.items()):
        if not runs_data:
            print(f"No data available for label {label}; skipping.")
            continue

        # Convert each run's data to a NumPy array if not empty
        arrays = [np.array(run) for run in runs_data if run and len(run) > 0]
        if not arrays:
            print(f"No valid arrays for label {label}; skipping.")
            continue

        # Collect all unique time steps across all runs
        all_steps = sorted(set().union(*(arr[:, 0] for arr in arrays)))

        # Create a list of pandas Series for each run, indexed by time steps
        run_series = []
        for arr in arrays:
            s = pd.Series(data=arr[:, 1], index=arr[:, 0])
            # Remove duplicate indices by keeping the last occurrence
            s = s[~s.index.duplicated(keep='last')]
            # Reindex to include all_steps, forward-fill missing values
            s = s.reindex(all_steps, method='ffill')
            run_series.append(s)

        # Concatenate all series into a DataFrame, aligning on the index (time steps)
        df = pd.concat(run_series, axis=1)

        # Compute mean and standard deviation across runs for each time step
        mean_reward = df.mean(axis=1).values
        std_reward = df.std(axis=1, ddof=1).values
        timestep = np.array(all_steps)

        # Smooth the mean and std rewards if a smoothing window is specified
        mean_reward = smoothen(mean_reward, smooth_window)
        std_reward = smoothen(std_reward, smooth_window)

        plt.plot(timestep, mean_reward, color=colors[i % len(colors)], label=label, linewidth=1.5)
        plt.fill_between(timestep, mean_reward - std_reward, mean_reward + std_reward, 
                         alpha=0.2, color=colors[i % len(colors)])

    plt.title(title, fontsize=16, pad=12)
    plt.legend(loc="best", fontsize=12)
    plt.grid()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='png')
    plt.show()


def visualize_rl_training(data_paths, labels, smooth_window=5, title="RL Training Visualization", xlabel="Training Steps",
                          ylabel="Average Reward", save_path=None):
    """
    Visualizes RL training performance based on provided data.

    Args:
        data_paths (list): List of lists, where each sublist contains paths to CSV files for a specific label.
        labels (list): List of labels corresponding to each group of data paths.
        smooth_window (int): Window size for smoothing.
        title (str): Title of the plot.
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
        save_path (str): File path to save the plot. If None, the plot will not be saved.
    """
    assert len(data_paths) == len(labels), "Each label must correspond to a set of data paths."

    data_dict = {}

    for label, paths in zip(labels, data_paths):
        episode_rewards = []
        for path in paths:
            list_data = read_csv_2_dict(path)
            episode_rewards.append(list_data)
        data_dict[label] = episode_rewards

    draw(data_dict, smooth_window, title, xlabel, ylabel, save_path)


if __name__ == "__main__":
    data_paths = [
        [
            "/path/to/data1.csv",
            "/path/to/data2.csv"
        ]
    ]
    labels = ["Algorithm A", "Algorithm B"]

    visualize_rl_training(
        data_paths=data_paths,
        labels=labels,
        smooth_window=5,
        title="RL Training Visualization",
        xlabel="Training Steps",
        ylabel="Average Reward",
        save_path="rl_training_plot.pdf"
    )
