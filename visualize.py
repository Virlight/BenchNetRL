import tensorflow as tf
import os
import utils  # Your utils.py functions

def read_tfevents(file_path, tag_name="reward", step=4):
    """
    Reads TensorBoard event file and extracts scalars for a specific tag,
    taking every `step` points.
    
    Args:
        file_path (str): Path to the .tfevents file.
        tag_name (str): Tag of the scalar to extract.
        step (int): Interval for downsampling points.
    
    Returns:
        list of [step, value] pairs.
    """
    scalar_data = []
    for i, event in enumerate(tf.compat.v1.train.summary_iterator(file_path)):
        for value in event.summary.value:
            if value.tag == tag_name and i % step == 0:
                scalar_data.append([event.step, value.simple_value])
    return scalar_data

def convert_tfevents_to_dict(event_paths, tag_name="reward", step=4):
    """
    Converts a list of TensorBoard .tfevents paths into a dictionary for visualization.

    Args:
        event_paths (list): List of .tfevents file paths.
        tag_name (str): The scalar tag to extract (e.g., 'reward').
        step (int): Interval for downsampling points.
    
    Returns:
        dict: Dictionary where keys are filenames, and values are episode rewards.
    """
    data_dict = {}
    for path in event_paths:
        label = os.path.basename(path).split('.')[0]  # Label based on file name
        data = read_tfevents(path, tag_name, step)
        data_dict[label] = [data]
    return data_dict

def visualize_from_tfevents(event_paths, labels, tag_name="reward", smooth_window=5, title="RL Training Visualization", save_path=None, step=4):
    """
    Visualizes RL training performance using TensorBoard event files, taking every `step` points.

    Args:
        event_paths (list): List of lists of .tfevents file paths.
        labels (list): List of labels corresponding to each group of paths.
        tag_name (str): Scalar tag to extract and visualize.
        smooth_window (int): Window size for smoothing.
        title (str): Title of the plot.
        save_path (str): Path to save the plot.
        step (int): Interval for downsampling points.
    """
    assert len(event_paths) == len(labels), "Each label must correspond to a set of event paths."

    data_dict = {}
    for label, paths in zip(labels, event_paths):
        episode_rewards = []
        for path in paths:
            list_data = read_tfevents(path, tag_name=tag_name, step=step)
            episode_rewards.append(list_data)
        data_dict[label] = episode_rewards

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
    event_paths = [
        [
            "events.out.tfevents1.0"
        ],
        [
            "runs\CartPole-v1__ppo__1__1728577357\events.out.tfevents.1728577357.Tornadosky.22916.0"
        ]
    ]
    labels = ["PPO-Mamba", "PPO-LSTM"]
    #print_tfevents("runs\CartPole-v1__ppo__1__1728577357\events.out.tfevents.1728577357.Tornadosky.22916.0")

    visualize_from_tfevents(
        event_paths=event_paths,
        labels=labels,
        tag_name="charts/episodic_return",
        smooth_window=100,
        title="RL Training Visualization from TensorBoard",
        save_path="rl_training_plot_tensorboard.png",
        step=50000
    )
