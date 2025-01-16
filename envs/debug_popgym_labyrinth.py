import popgym

def debug_labyrinth_escape_hard_ascii():
    """
    Debug LabyrinthEscapeHard by printing ASCII representations of the maze.
    """
    env = popgym.envs.labyrinth_escape.LabyrinthEscapeHard()
    obs, info = env.reset(seed=42)
    
    print("=== LabyrinthEscapeHard Debug ===")
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Initial observation:", obs)

    max_steps = 10
    for step in range(max_steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nStep {step}")
        print(f"Action taken: {action}")
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}, Truncated: {truncated}")
        print("Maze state:")
        env.render()  # Prints the maze ASCII art
        
        if terminated or truncated:
            print("Episode finished!")
            break

    env.close()

if __name__ == "__main__":
    debug_labyrinth_escape_hard_ascii()