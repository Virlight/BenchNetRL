import gymnasium as gym
import numpy as np
# from stable_baselines3.common.atari_wrappers import (
#     ClipRewardEnv,
#     EpisodicLifeEnv,
#     FireResetEnv,
#     MaxAndSkipEnv,
#     NoopResetEnv,
# )

from gymnasium.wrappers import (
    RecordEpisodeStatistics,
    ResizeObservation,
    GrayscaleObservation,
    FrameStackObservation,
    RecordVideo,
    TransformReward,
    AtariPreprocessing,
)

import minigrid
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from envs.poc_memory_env import PocMemoryEnv
from collections import deque


class FireResetEnv(gym.Wrapper):
    """Auto-fire on reset for some Atari envs like Breakout."""
    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        self.fire_action = 1

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(self.fire_action)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class VecObservationStackWrapper(gym.ObservationWrapper):
    def __init__(self, env, num_stack):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        # Ensure the low/high bounds are NumPy arrays and repeat them along the stacking axis.
        low = np.repeat(np.array(env.observation_space.low, dtype=np.float32), num_stack, axis=0)
        high = np.repeat(np.array(env.observation_space.high, dtype=np.float32), num_stack, axis=0)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, **kwargs):
        # Reset the underlying env and ensure the observation is a NumPy array.
        obs, info = self.env.reset(**kwargs)
        obs = np.array(obs, dtype=np.float32)
        # Initialize the buffer with copies of the initial observation.
        self.frames = deque([obs.copy() for _ in range(self.num_stack)], maxlen=self.num_stack)
        return self._get_obs(), info

    def observation(self, observation):
        observation = np.array(observation, dtype=np.float32)
        # Append a copy to ensure consistency.
        self.frames.append(observation.copy())
        return self._get_obs()

    def _get_obs(self):
        # Concatenate along the first axis.
        return np.concatenate(list(self.frames), axis=0)

class MaskObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, mask_indices, mask_prob=1.0):
        super().__init__(env)
        self.mask_indices = mask_indices
        self.mask_prob = mask_prob

        # Keep original bounds for observation space (unchanged)
        self.observation_space = env.observation_space

    def observation(self, observation):
        observation = np.array(observation, dtype=np.float32)

        # Apply 50% masking probability for each index
        for i in self.mask_indices:
            if np.random.rand() < self.mask_prob:
                observation[i] = 0.0

        return observation

class RecordEpisodeStatistics(gym.Wrapper):
  def __init__(self, env, deque_size=100):
    super(RecordEpisodeStatistics, self).__init__(env)
    self.num_envs = getattr(env, "num_envs", 1)
    self.episode_returns = None
    self.episode_lengths = None
    # get if the env has lives
    self.has_lives = False
    env.reset()
    # info = env.step(np.zeros(self.num_envs, dtype=int))[-1]
    info = env.step(0)[-1]
    lives = info.get("lives", 0)
    if isinstance(lives, np.ndarray):
        has_lives = lives.sum() > 0
    else:
        has_lives = lives > 0
    # if info["lives"].sum() > 0:
    #   self.has_lives = True
    #   print("env has lives")
    if has_lives:
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
    # self.episode_returns += infos["reward"]
    self.episode_returns += rewards
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
    # def thunk():
    #     env = gym.make(gym_id, render_mode="rgb_array", repeat_action_probability=0.0) if capture_video else gym.make(gym_id, repeat_action_probability=0.0)
    #     env = gym.wrappers.RecordEpisodeStatistics(env)
    #     env = NoopResetEnv(env, noop_max=30)
    #     #env = MaxAndSkipEnv(env, skip=4)
    #     env = EpisodicLifeEnv(env)
    #     if "FIRE" in env.unwrapped.get_action_meanings():
    #         env = FireResetEnv(env)
    #     env = ClipRewardEnv(env)
    #     env = gym.wrappers.ResizeObservation(env, (84, 84))
    #     env = gym.wrappers.GrayscaleObservation(env)
    #     # env = GrayScaleObservation(env)
    #     env = gym.wrappers.FrameStackObservation(env, frame_stack)
    #     # env = FrameStackObservation(env, stack_size=frame_stack)
    #     if capture_video and idx == 0:
    #         env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    #     env.reset(seed=seed)
    #     env.action_space.seed(seed)
    #     env.observation_space.seed(seed)
    #     return env
    def thunk():
        env = gym.make(gym_id, render_mode="rgb_array" if capture_video else None, repeat_action_probability=0.0, frameskip=1)  # <--- 禁用原生 frame skip！)
        env = RecordEpisodeStatistics(env)

        # AtariPreprocessing handles NoopReset, MaxAndSkip, EpisodicLife, etc.
        # env = NoopResetEnv(env, noop_max=30)      # 等价于 noop_max=30
        # env = MaxAndSkipEnv(env, skip=4)          # 等价于 frame_skip=4
        # env = EpisodicLifeEnv(env)                # 等价于 terminal_on_life_loss=True
        # env = gym.wrappers.ResizeObservation(env, (84, 84))  # 等价于 screen_size=84
        # env = gym.wrappers.GrayscaleObservation(env)         # 等价于 grayscale_obs=True
        env = AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=4,
            terminal_on_life_loss=True,
            screen_size=84,   # Resize to 84x84
            grayscale_obs=True
        )

        # FireReset wrapper only for environments that need it
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)

        # Reward clipping: -1, 0, 1,  等同于 env = ClipRewardEnv(env)
        # Atari 游戏 reward 分布很不均（如 Q*Bert +25，Breakout +1，Pong ±1
        # PPO、DQN 等 policy/value based 方法在 early stage 训练时更关注“什么行为有奖励”而不是“奖励有多大”
        # 在 Atari 大规模实验中，reward clipping 通常提升 early training 收敛速度（但可能略损最终上限）
        env = TransformReward(env, lambda r: np.sign(r))

        # FrameStack over processed 84x84 grayscale images
        env = FrameStackObservation(env, stack_size=frame_stack)

        if capture_video and idx == 0:
            env = RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda x: True)

        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def make_classic_env(gym_id, seed, idx, capture_video, run_name, masked_indices=[], obs_stack=1):
    def thunk():
        env = gym.make(gym_id, render_mode="rgb_array") if capture_video else gym.make(gym_id)
        if masked_indices:
            env = MaskObservationWrapper(env, masked_indices)
        if obs_stack > 1:
            env = VecObservationStackWrapper(env, num_stack=obs_stack)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def make_memory_gym_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        import memory_gym
        env = gym.make(
            gym_id,
            render_mode="rgb_array" if capture_video else None,
        )
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def make_minigrid_env(gym_id, seed, idx, capture_video, run_name, agent_view_size=3, tile_size=28, max_episode_steps=96, frame_stack=1):
    def thunk():
        env = gym.make(
            gym_id,
            agent_view_size=agent_view_size,
            tile_size=tile_size,
            render_mode="rgb_array" if capture_video else None,
        )
        env = ImgObsWrapper(RGBImgPartialObsWrapper(env, tile_size=tile_size))
        if frame_stack > 1:
            env = gym.wrappers.FrameStack(env, frame_stack)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def make_poc_env(gym_id, seed, idx, capture_video, run_name, step_size=0.2, glob=False, freeze=False, max_episode_steps=96):
    def thunk():
        env = PocMemoryEnv(step_size=step_size, glob=glob, freeze=freeze, max_episode_steps=max_episode_steps)
        return env
    return thunk

def make_continuous_env(gym_id, seed, idx, capture_video, run_name, obs_stack=1):
    def thunk():
        env = gym.make(gym_id, render_mode="rgb_array") if capture_video else gym.make(gym_id)
        if obs_stack > 1:
            env = VecObservationStackWrapper(env, num_stack=obs_stack)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), observation_space=env.observation_space)
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk