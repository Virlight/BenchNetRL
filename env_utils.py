import gymnasium as gym
import numpy as np
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self):
        observations = self.env.reset()
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, terminated, truncated, infos = self.env.step(action)
        self.episode_returns += rewards
        self.episode_lengths += 1
        infos["r"] = self.episode_returns.copy()
        infos["l"] = self.episode_lengths.copy()
        done = np.logical_or(terminated, truncated)
        self.episode_returns *= ~done
        self.episode_lengths *= ~done
        return observations, rewards, terminated, truncated, infos

def make_atari_env(gym_id, seed, idx, capture_video, run_name, frame_stack=1):
    """
    frame_stack: How many frames to stack for the observation.
    """
    def thunk():
        env = gym.make(gym_id, render_mode="rgb_array") if capture_video else gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, frame_stack)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk