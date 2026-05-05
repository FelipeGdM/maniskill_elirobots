from pprint import pprint
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
from PIL import Image
from torchvision.utils import save_image

import maniskill_elirobots
from maniskill_elirobots.utils.agent import Agent

gym.register_envs(maniskill_elirobots)

# Configuration
NUM_EVAL_EPISODES = 1
ENV_ID = "FlipCoin-v1"  # Replace with your environment
# ENV_ID = "PushCube-v1"  # Replace with your environment
CHECKPOINT = "/workspaces/maniskill_elirobots/pth/ckpt_8192000_slow.pt"
ROBOT_UID = "ec63"
OUTPUT_FOLDER = "eval"

ACTION_SPACE = 7
OBSERVATION_SPACE = 39

EPISODE_SIZE = 50

MAX_ANG_POS = 6.2832  # rad

MAX_ANG_VEL = 3.3161  # rad/s

env = gym.make(
    id=ENV_ID,
    robot_uids=ROBOT_UID,
    num_envs=1,
    obs_mode="state_dict",
    render_mode="rgb_array",
    sim_backend="physx_cpu",
    control_mode="pd_joint_pos",
    sim_config={
        "sim_freq": 100,
        "control_freq": 2,
    },
)

init_qpos = torch.tensor([0.0, -np.pi / 2, 0.0, -np.pi / 2, np.pi / 2, 0.0, 0.0, 0.0])

coin_xyz = torch.tensor([[2.4e-01, 1e-01, 3.7e-04]])

obs, info = env.reset(
    options={
        # "coin_xyz": coin_xyz,
        "init_qpos": init_qpos,
    },
)

# action = torch.tensor([0.0, 0.0, 0.0, -np.pi / 2, np.pi / 2, 0.0, 1.0])

# for _ in range(4):
#     obs, reward, term, trunc, info = env.step(action)

# action2 = torch.tensor([0.2, 0.0, 0.0, -np.pi / 2, np.pi / 2, 0.0, 1.0])

# for _ in range(4):
#     obs, reward, term, trunc, info = env.step(action2)

base_env = cast("maniskill_elirobots.FlipCoinEnv", env.unwrapped)

agent = base_env.agent

coin = base_env.coin

print(agent.is_grasping(coin))

pprint(obs)  # noqa: T203
pprint(info)  # noqa: T203

photo = cast("torch.Tensor", env.render()).cpu()  # pyright: ignore[reportInvalidCast]

print(f"{photo.shape=}")

# img = Image.fromarray(obj=photo.numpy().reshape(512, 512, 3))

# img.save("output.png")

save_image(photo[0].transpose(0, 2).transpose(1, 2) / 255, "output.png")

print("Image saved!")
