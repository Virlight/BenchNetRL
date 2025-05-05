import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pom_env import PoMEnv

def debug_pom_env_with_render():
    """
    Debug the PoMEnv and capture rendered frames for visualization.
    """
    # 1) Initialize the environment in rgb_array mode
    env = PoMEnv(render_mode="rgb_array")

    print("=== PoMEnv debug with render ===")
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space, "| Number of possible discrete actions =", env.action_space.n)

    # 2) Reset environment and get initial observation
    obs, info = env.reset()
    print("Initial observation:", obs)

    max_steps = 48
    obs_list = []
    act_list = []
    rew_list = []
    done_list = []
    frames = []  # List to store rendered frames

    for step_i in range(max_steps):
        action = env.action_space.sample()
        act_list.append(action)

        next_obs, reward, done, truncated, info = env.step(action)
        
        obs_list.append(obs)
        rew_list.append(reward)
        done_list.append(done)

        # Capture frame after taking an action
        frame = env.render()
        frames.append(frame)

        print(f"Step={step_i} | Obs={obs} | Action={action} | Reward={reward:.2f} | Done={done}")

        obs = next_obs
        if done:
            print("Episode finished. Info:", info)
            break

    env.close()

    fig = plt.figure()
    plt.axis('off')

    def update(frame_idx):
        plt.imshow(frames[frame_idx])
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=300, repeat=False)
    plt.show()

if __name__ == "__main__":
    debug_pom_env_with_render()
