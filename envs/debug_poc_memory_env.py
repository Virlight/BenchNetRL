import numpy as np
import matplotlib.pyplot as plt
from poc_memory_env import PocMemoryEnv

def debug_poc_memory_env():
    """
    Debug the PocMemoryEnv:
      1) Initialize the environment
      2) Print action/observation info
      3) Take random actions
      4) Store (obs, action, reward)
      5) Plot agent position, goals, and rewards
    """
    # 1) Initialize the environment
    env = PocMemoryEnv(step_size=0.2, glob=False, freeze=False, max_episode_steps=50)

    # 2) Print environment details
    print("=== PocMemoryEnv debug ===")
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space, "| Number of possible discrete actions =", env.action_space.n)

    # 3) Reset and see initial obs
    obs = env.reset()
    print("Initial obs:", obs)  
    # obs ~ [goalLeft, agentPos, goalRight]
    #   if _num_show_steps > 0 at first, goals are visible, else they might be zeroed out

    # 4) Step with random actions
    max_steps = 12  # how many steps to run for debugging
    obs_list = []
    act_list = []
    rew_list = []
    done_list = []
    position_list = []  # just agent positions for easy plotting

    for step_i in range(max_steps):
        # Sample a random action from discrete(2) => 0 means "move left", 1 means "move right"
        action = np.array([env.action_space.sample()], dtype=int)  
        # env.step expects shape-lists if you handle them that way
        next_obs, reward, done, info = env.step(action)

        obs_list.append(obs)
        act_list.append(action.item())
        rew_list.append(reward)
        done_list.append(done)

        # The second entry in obs is the agent's position
        agent_pos = obs[1]  
        position_list.append(agent_pos)

        print(f"Step={step_i} | obs={obs} | action={action.item()} | reward={reward:.2f} | done={done}")
        obs = next_obs

        if done:
            print("Episode finished! Info:", info)
            break

    # 5) Plot agent position vs. time, along with the goals
    steps_range = np.arange(len(position_list))
    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot the agent position
    ax.plot(steps_range, position_list, marker='o', label='Agent position')

    # If you want to highlight left/right goal from the *first* observation
    #   obs[0] is left goal, obs[2] is right goal
    # but note that obs might have changed after multiple steps, so we might rely
    # on the initial obs or store the first. Or we can store them each step.

    # If the environment shows them at first, let's use the initial:
    initial_goals = obs_list[0] if len(obs_list) > 0 else [0,0,0]
    left_goal, right_goal = initial_goals[0], initial_goals[2]

    # Add horizontal lines or annotation for the two ends:
    ax.axhline(y=-1.0, color='red', linestyle='--', alpha=0.3, label='Left boundary')
    ax.axhline(y=+1.0, color='green', linestyle='--', alpha=0.3, label='Right boundary')

    # If we want to annotate which side is good vs. bad:
    left_label = f"LeftGoal=+1" if (left_goal>0) else f"LeftGoal=-1"
    right_label = f"RightGoal=+1" if (right_goal>0) else f"RightGoal=-1"
    ax.text(0.5, -1.05, left_label, ha='center', color='red')
    ax.text(0.5, 1.05, right_label, ha='center', color='green')

    ax.set_xlabel('Time step')
    ax.set_ylabel('Agent Position')
    ax.set_title('PocMemoryEnv Debug: Agent position over time')
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    debug_poc_memory_env()
