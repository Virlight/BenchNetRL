import numpy as np
import popgym
from popgym.envs.repeat_previous import RepeatPrevious  # Import the specific environment

def debug_repeat_previous_env():
    """
    Debug the RepeatPrevious environment:
      1) Initialize the environment
      2) Print action/observation space info
      3) Take random actions and record observations, actions, rewards, done flags
    """
    # 1) Initialize the environment
    env = RepeatPrevious()
    
    # 2) Print environment details
    print("=== RepeatPreviousEnv Debug ===")
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space, "| Number of possible discrete actions =", env.action_space.n)
    
    # 3) Reset environment and get initial observation
    obs = env.reset()
    print("Initial observation:", obs)
    
    max_steps = 10  # Number of steps to run for debugging
    for step in range(max_steps):
        action = env.action_space.sample()  # Sample a random action
        next_obs, reward, done, truncated, info = env.step(action)
        
        print(f"Step={step} | Obs={obs} | Action={action} | Reward={reward:.2f} | Done={done or truncated}")
        
        obs = next_obs
        if done or truncated:
            print("Episode finished. Info:", info)
            break
    
    env.close()

if __name__ == "__main__":
    debug_repeat_previous_env()
