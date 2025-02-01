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
  def __init__(self, env, deque_size=100):
    super(RecordEpisodeStatistics, self).__init__(env)
    self.num_envs = getattr(env, "num_envs", 1)
    self.episode_returns = None
    self.episode_lengths = None
    # get if the env has lives
    self.has_lives = False
    env.reset()
    info = env.step(np.zeros(self.num_envs, dtype=int))[-1]
    if info["lives"].sum() > 0:
      self.has_lives = True
      print("env has lives")

  def reset(self, **kwargs):
    observations = self.env.reset()
    self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
    self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
    self.lives = np.zeros(self.num_envs, dtype=np.int32)
    self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
    self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
    return observations

  def step(self, action):
    observations, rewards, term, trunc, infos = self.env.step(action)
    dones = term + trunc
    self.episode_returns += infos["reward"]
    self.episode_lengths += 1
    self.returned_episode_returns[:] = self.episode_returns
    self.returned_episode_lengths[:] = self.episode_lengths
    all_lives_exhausted = infos["lives"] == 0
    if self.has_lives:
      self.episode_returns *= 1 - all_lives_exhausted
      self.episode_lengths *= 1 - all_lives_exhausted
    else:
      self.episode_returns *= 1 - dones
      self.episode_lengths *= 1 - dones
    infos["r"] = self.returned_episode_returns
    infos["l"] = self.returned_episode_lengths
    return (
      observations,
      rewards,
      term,
      trunc,
      infos,
    )

def make_atari_env(gym_id, seed, idx, capture_video, run_name, frame_stack=1):
    """
    frame_stack: How many frames to stack for the observation.
    """
    def thunk():
        env = gym.make(gym_id, render_mode="rgb_array", repeat_action_probability=0.0) if capture_video else gym.make(gym_id, repeat_action_probability=0.0)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        #env = MaxAndSkipEnv(env, skip=4)
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