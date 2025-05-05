import tensorflow as tf
import os
import glob
import utils

def find_event_file_in_folder(folder_path):
    """Finds the (assumed single) TensorBoard event file in a given folder."""
    pattern = os.path.join(folder_path, "events.out.tfevents.*")
    files = glob.glob(pattern)
    return files[0] if files else None

def read_tfevents(file_path, tag_name="reward", step=4):
    """
    Reads a TensorBoard event file and extracts scalars for a specific tag,
    taking every `step` points.
    """
    scalar_data = []
    for i, event in enumerate(tf.compat.v1.train.summary_iterator(file_path)):
        for value in event.summary.value:
            if value.tag == tag_name and i % step == 0:
                scalar_data.append([event.step, value.simple_value])
    return scalar_data

def aggregate_data_from_folders(folders, tag_name, step):
    """
    For a list of folders, finds the event file in each folder, reads, and collects their data.
    Returns a list of arrays, each corresponding to one run's data.
    """
    all_runs_data = []
    for folder in folders:
        event_file = find_event_file_in_folder(folder)
        if event_file:
            data = read_tfevents(event_file, tag_name, step)
            if data:
                all_runs_data.append(data)
    return all_runs_data

def visualize_from_tfevents(folder_groups, labels, tag_name="reward", smooth_window=5, 
                           title="RL Training Visualization", save_path=None, step=4):
    """
    Visualizes RL training performance using TensorBoard event files from folders.
    
    Args:
        folder_groups (list): List of lists, each containing folder paths (runs) for a label.
        labels (list): List of labels corresponding to each group of folders.
        tag_name (str): Scalar tag to extract and visualize.
        smooth_window (int): Window size for smoothing.
        title (str): Title of the plot.
        save_path (str): Path to save the plot.
        step (int): Interval for downsampling points.
    """
    assert len(folder_groups) == len(labels), "Each label must correspond to a set of folders."

    data_dict = {}
    for label, folders in zip(labels, folder_groups):
        # Aggregate data from all event files in the given folders
        runs_data = aggregate_data_from_folders(folders, tag_name, step)
        data_dict[label] = runs_data

    utils.draw(data_dict, smooth_window, title, "Training Steps", "Average Reward", save_path)

def print_tfevents(file_path):
    """
    Opens a TensorBoard event file and prints all available tags and their corresponding values.
    
    Args:
        file_path (str): Path to the .tfevents file.
    """
    print(f"Reading TensorBoard file: {file_path}")
    for event in tf.compat.v1.train.summary_iterator(file_path):
        for value in event.summary.value:
            print(f"Step: {event.step}, Tag: {value.tag}, Value: {value.simple_value}")

if __name__ == "__main__":
    folder_groups = [
        [
            "runs\CartPole-v1__ppo__1__1736687709",
            "runs\CartPole-v1__ppo__2__1736687660"
        ],
        [
            "runs\CartPole-v1__ppo__1__1736687709",
            "runs\CartPole-v1__ppo__2__1736687660",
            "runs\CartPole-v1__ppo__3__1736688619",
        ]
    ]
    labels = ["PPO-Combined2", "PPO-Combined10"]

    visualize_from_tfevents(
        folder_groups=folder_groups,
        labels=labels,
        tag_name="charts/episodic_return",
        smooth_window=200,
        title="RL Training Visualization from TensorBoard",
        save_path="rl_training_plot_tensorboard.png",
        step=1
    )
