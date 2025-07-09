import torch
import gymnasium as gym
import numpy as np
from ppo import Agent, parse_args
from env_utils import make_atari_env  # 如果你用的是 Breakout
import ale_py
import os
import argparse

gym.register_envs(ale_py)

# ====== 加载模型 ======
model_path = "runs/ALE/Breakout-v5__ppo_4__7__1748891698/ppo_4_update_600.cleanrl_model"  # 替换为你的路径
checkpoint = torch.load(model_path, map_location="cpu")
saved_args = checkpoint["args"]
saved_args = argparse.Namespace(**saved_args)
device = torch.device("cuda" if torch.cuda.is_available() and saved_args.cuda else "cpu")

# ====== 创建单个环境（非向量化） ======
env = make_atari_env(saved_args.gym_id, seed=0, idx=0, capture_video=True,
                     run_name="eval", frame_stack=saved_args.frame_stack)()
obs, _ = env.reset()
obs = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)

# ====== 创建 Agent 并加载参数 ======
dummy_vec_env = gym.vector.SyncVectorEnv([lambda: env])
agent = Agent(dummy_vec_env, saved_args).to(device)
agent.load_state_dict(checkpoint["model_weights"])
agent.eval()

# ====== Inference 循环 ======
done = False
total_reward = 0
frames = []

while not done:
    with torch.no_grad():
        action, _, _, _ = agent.get_action_and_value(obs)
    obs, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])
    done = terminated or truncated
    obs = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)
    total_reward += reward
    if "rgb_array" in env.metadata.get("render_modes", []):
        frame = env.render()
        frames.append(frame)

print(f"Total episode return: {total_reward:.2f}")
env.close()

# ====== 可选：保存 GIF ======
if frames:
    import imageio
    imageio.mimsave("inference.gif", frames, fps=30)
    print("Saved inference.gif")
