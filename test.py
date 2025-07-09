import gymnasium as gym
print([env_id for env_id in gym.envs.registry.keys() if "Breakout" in env_id])

import ale_py
gym.register_envs(ale_py)
print([env_id for env_id in gym.envs.registry.keys() if "Breakout" in env_id])