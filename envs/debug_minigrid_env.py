import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

def make_env(gym_id="MiniGrid-DoorKey-8x8-v0", seed=1):
    env = gym.make(
        gym_id,
        agent_view_size=3,  # just as an example
        tile_size=16,
        render_mode="rgb_array",  # needed for us to get image observations
    )
    env = ImgObsWrapper(RGBImgPartialObsWrapper(env, tile_size=16))
    env = gym.wrappers.TimeLimit(env, max_episode_steps=96)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

def debug_minigrid_plot_12():
    # 1. Create environment
    env = make_env()

    # 2. Reset environment
    obs, info = env.reset()
    print("Observation shape:", obs.shape)
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)

    # We'll store up to 12 steps of (observation, action)
    steps_to_plot = 12
    observations = []
    actions = []

    # 3. Append the initial observation to the list
    observations.append(obs)
    actions.append(-1)  # no action led to the initial state, so store a placeholder

    # 4. Step through the environment, restricting actions to {0,1,2,3}
    for i in range(steps_to_plot):
        # Sample from {0,1,2,3} only
        action = np.random.choice([0,1,2])
        obs, reward, done, truncated, info = env.step(action)

        observations.append(obs)
        actions.append(action)

        print(f"Step={i}, Action={action}, Reward={reward}, Done={done}, Truncated={truncated}")

        if done or truncated:
            obs, info = env.reset()

    # 5. Plot the 12 resulting observations
    # Actually we have 13 total observations: the initial + 12 after each action
    # We'll show the 12 *new* ones or you can choose to show all 13
    # Here we show all 12 steps in a 3x4 grid
    fig, axs = plt.subplots(3, 4, figsize=(12, 8))

    # If you want to show the initial obs plus 11 steps, adjust accordingly.
    # We'll show 12 steps: from index 1..12 (since index 0 is the initial obs)
    for idx in range(1, steps_to_plot+1):
        ax = axs[(idx-1)//4, (idx-1)%4]  # row, column in a 3x4
        obs_i = observations[idx]
        action_i = actions[idx]

        # Plot
        # obs_i is shape (height, width, 3) after wrappers => (28,28,3) typically
        ax.imshow(obs_i)
        ax.set_title(f"Step {idx}, Act={action_i}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    debug_minigrid_plot_12()
