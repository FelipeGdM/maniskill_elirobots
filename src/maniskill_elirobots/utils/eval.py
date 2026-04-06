from typing import cast

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo  # pyright: ignore[reportPrivateImportUsage]
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.common import flatten_dict_keys, flatten_state_dict
from mani_skill.utils.structs.types import Array
from mani_skill.utils.wrappers.record import RecordEpisode

from maniskill_elirobots import PushCubeEcEnv  # noqa: F401  # pyright: ignore[reportUnusedImport]
from maniskill_elirobots.utils.agent import Agent

# Configuration
NUM_EVAL_EPISODES = 1
ENV_ID = "PushCubeEc-v1"  # Replace with your environment
CHECKPOINT = "/workspaces/maniskill_elirobots/pth/ckpt_8192000_slow.pt"
ROBOT_UID = "ec63"
OUTPUT_FOLDER = "eval"

ACTION_SPACE = 7
OBSERVATION_SPACE = 39

EPISODE_SIZE = 50

MAX_ANG_POS = 6.2832  # rad

MAX_ANG_VEL = 3.3161  # rad/s

# Create environment with recording capabilities
env = gym.make(
    id=ENV_ID,
    robot_uids=ROBOT_UID,
    obs_mode="state_dict",
    render_mode="rgb_array",
    sim_backend="physx_cpu",
    control_mode="pd_joint_delta_pos",
    sim_config={
        "sim_freq": 100,
        "control_freq": 2,
    },
)

print(f"{env.unwrapped.action_space=}")  # pyright: ignore[reportAttributeAccessIssue]
print(f"{env.unwrapped.observation_space=}")  # pyright: ignore[reportAttributeAccessIssue]

agent = Agent(OBSERVATION_SPACE, ACTION_SPACE, torch.device("cpu"))

_ = agent.load_state_dict(torch.load(CHECKPOINT, map_location=torch.device("cpu")))

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
    obs_dict, info = env.reset()

    obs: Array = flatten_state_dict(obs_dict, use_torch=True, device=torch.device("cpu"))  # pyright: ignore[reportUnknownArgumentType]

    obs_list.append(flatten_dict_keys(obs_dict))
    episode_reward = 0
    step_count = 0

    episode_over = False
    with torch.no_grad():
        while not episode_over:
            action = agent.get_action(obs, deterministic=True)

            obs_dict, reward, terminated, truncated, info = env.step(action)

            obs = flatten_state_dict(obs_dict, use_torch=True, device=torch.device("cpu"))  # pyright: ignore[reportUnknownArgumentType]

            obs_list[-1].update({"action": action})

            obs_list.append(flatten_dict_keys(obs_dict))

            step_count += 1
            episode_reward += reward

            episode_over = terminated or truncated

    print(info)

    print(f"Episode {episode_num + 1}: {step_count} steps, reward = {episode_reward}")

env.close()

out_obs_list = [{f"{k}/{n}": info for k in element for n, info in enumerate(element[k].flatten().tolist())} for element in obs_list]  # pyright: ignore[reportUnknownArgumentType]

df = pd.DataFrame(out_obs_list)
df.to_csv("output.csv", index=False)

# [data["extra"]["tcp_pose"].cpu().flatten() for data in obs_list]

# qpos = [data.cpu().flatten()[0:8] for data in obs_list]

# qvel = [data.cpu().flatten()[8:16] * MAX_ANG_VEL for data in obs_list]
# qvel = [data.cpu().flatten()[8:16] for data in obs_list]

# tcp_pose = [data.cpu().flatten()[16:23] for data in obs_list]

# _ = plt.plot(qpos)  # pyright: ignore[reportUnknownArgumentType]

# plt.savefig(f"{OUTPUT_FOLDER}/qpos.png")

# plt.clf()

# _ = plt.plot(qvel)  # pyright: ignore[reportUnknownArgumentType]

# plt.savefig(f"{OUTPUT_FOLDER}/qvel.png")

# plt.clf()

# _ = plt.plot(tcp_pose, label=[f"q{i}" for i in range(7)])  # pyright: ignore[reportUnknownArgumentType]

# _ = plt.legend()

# plt.savefig(f"{OUTPUT_FOLDER}/tcp_pose.png")

# plt.clf()

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
