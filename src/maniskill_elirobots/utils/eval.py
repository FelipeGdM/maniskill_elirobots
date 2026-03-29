from typing import cast

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo  # pyright: ignore[reportPrivateImportUsage]
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.wrappers.record import RecordEpisode

from maniskill_elirobots import PushCubeEcEnv  # noqa: F401  # pyright: ignore[reportUnusedImport]

# Configuration
NUM_EVAL_EPISODES = 1
ENV_ID = "PushCubeEc-v1"  # Replace with your environment
ROBOT_UID = "ec63"
OUTPUT_FOLDER = "eval"

EPISODE_SIZE = 50

MAX_ANG_POS = 6.2832  # rad

MAX_ANG_VEL = 3.3161  # rad/s

env_kwargs = {"obs_mode": "state", "render_mode": "rgb_array", "sim_backend": "physx_cuda"}

# Create environment with recording capabilities
env = gym.make(
    id=ENV_ID,
    robot_uids=ROBOT_UID,
    obs_mode="state",
    render_mode="rgb_array",
    sim_backend="physx_cuda",
    # control_mode="pd_joint_pos",
    control_mode="pd_joint_delta_pos",
    sim_config={
        "sim_freq": 800,
        "control_freq": 80,
    },
)

print(f"{env.unwrapped.sim_freq=}")  # pyright: ignore[reportAttributeAccessIssue]
print(f"{env.unwrapped.control_freq=}")  # pyright: ignore[reportAttributeAccessIssue]

# Add video recording for every episode
env = RecordEpisode(
    env,
    output_dir="eval",  # Folder to save videos
    trajectory_name="video",  # Prefix for video filenames
)

# Add episode statistics tracking
# env = RecordEpisodeStatistics(env, NUM_EVAL_EPISODES)

print(f"Starting evaluation for {NUM_EVAL_EPISODES} episodes...")
print(f"Videos will be saved to: {OUTPUT_FOLDER}/")

obs_list = []

for episode_num in range(NUM_EVAL_EPISODES):
    obs, info = env.reset()
    obs_list.append(obs)
    episode_reward = 0
    step_count = 0

    episode_over = False
    while not episode_over:
        # Replace this with your trained agent's policy
        # action = env.action_space.sample()  # Random policy for demonstration

        action = torch.zeros((1, 7))

        # action[0][0] = 1 / env.unwrapped.control_freq if step_count < 25 else 0  # noqa: PLR2004
        # action[0][0] = 2 / EPISODE_SIZE if step_count < EPISODE_SIZE / 2 else 0
        # action[0][0] = 1.0 if step_count < EPISODE_SIZE / 2 else 0
        action[0][0] = 0.0
        # action[0][0] = 1e-2 / MAX_ANG_POS
        # print(f"{step_count}: {action[0][0]=}")
        # print(action)

        obs, reward, terminated, truncated, info = env.step(action)
        # print(info)
        obs_list.append(obs)

        print(f"{reward=}")

        qvel = torch.linalg.norm(obs.cpu().flatten()[8:16])

        print(f"{qvel=}")
        print(f"{torch.tanh(qvel)=}")

        # print(obs)
        # episode_reward += reward
        step_count += 1

        episode_over = terminated or truncated

    print(f"Episode {episode_num + 1}: {step_count} steps, reward = {episode_reward}")

env.close()

# [data["extra"]["tcp_pose"].cpu().flatten() for data in obs_list]

qpos = [data.cpu().flatten()[0:8] for data in obs_list]

# qvel = [data.cpu().flatten()[8:16] * MAX_ANG_VEL for data in obs_list]
qvel = [data.cpu().flatten()[8:16] for data in obs_list]

tcp_pose = [data.cpu().flatten()[16:23] for data in obs_list]

_ = plt.plot(qpos)  # pyright: ignore[reportUnknownArgumentType]

plt.savefig(f"{OUTPUT_FOLDER}/qpos.png")

plt.clf()

_ = plt.plot(qvel)  # pyright: ignore[reportUnknownArgumentType]

plt.savefig(f"{OUTPUT_FOLDER}/qvel.png")

plt.clf()

_ = plt.plot(tcp_pose, label=[f"q{i}" for i in range(7)])  # pyright: ignore[reportUnknownArgumentType]

_ = plt.legend()

plt.savefig(f"{OUTPUT_FOLDER}/tcp_pose.png")

plt.clf()

# print(f"{qpos[-1][0]=}")
# print(f"{qvel[-1][0]=}")
# Print summary statistics
# print(f"\nEvaluation Summary:")
# print(f"Episode durations: {list(env.time_queue)}")
# print(f"Episode rewards: {list(env.return_queue)}")
# print(f"Episode lengths: {list(env.length_queue)}")

# # Calculate some useful metrics
# avg_reward = np.sum(env.return_queue)
# avg_length = np.sum(env.length_queue)
# std_reward = np.std(env.return_queue)

# print(f"\nAverage reward: {avg_reward:.2f} ± {std_reward:.2f}")
# print(f"Average episode length: {avg_length:.1f} steps")
# print(f"Success rate: {sum(1 for r in env.return_queue if r > 0) / len(env.return_queue):.1%}")
