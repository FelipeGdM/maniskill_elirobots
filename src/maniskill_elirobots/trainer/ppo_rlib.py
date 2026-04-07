"""
Minimal PPO training with RLlib (Ray 2.x) — CUDA GPU
======================================================
Install: pip install "ray[rllib]" gymnasium torch torchvision
Requires: CUDA-capable GPU + matching PyTorch CUDA build
"""

import os
from pathlib import Path
from pprint import pprint
import gymnasium as gym
import ray
import torch
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from maniskill_elirobots.tasks import PushCubeEcEnv

os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"


def create_env(config):
    base_env = PushCubeEcEnv(**config)
    obs_env = gym.wrappers.FlattenObservation(base_env)
    return obs_env


register_env("table_env", create_env)

# f"ignore::DeprecationWarning,{os.environ['PYTHONWARNINGS']}"

# ── 0. Verify CUDA is available ───────────────────────────────────────────────
if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU detected. Ensure a CUDA-capable GPU is present and PyTorch was installed with CUDA support.\nCheck: https://pytorch.org/get-started/locally/")

print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# ── 1. Initialise Ray ─────────────────────────────────────────────────────────
ray.init(ignore_reinit_error=True)

env_kwargs = {
    # "env": "PushCubeEc-v1",
    "obs_mode": "state",
    "render_mode": "rgb_array",
    # "sim_backend": "physx_cuda",
    "sim_backend": "physx_cpu",
    # "render_backend": "sapien_cuda",
    "render_backend": "sapien_cpu",
    "control_mode": "pd_joint_delta_pos",
    "robot_uids": "ec63",
    "num_envs": 1,
    "sim_config": {
        "sim_freq": 100,
        "control_freq": 20,
    },
}

# ── 2. Build the PPO config ───────────────────────────────────────────────────
config = (
    PPOConfig()
    # Environment
    # .environment(ManiSkillVectorEnv, env_config=env_kwargs)
    .environment("table_env", env_config=env_kwargs)
    .env_runners(num_env_runners=1)
    .training(
        lr=0.0002,
        train_batch_size_per_learner=2000,
        num_epochs=10,
    )
)

# ──uild the algorithm ────────────────────────────────────────────────────
algo = config.build_algo()

# ── 4. Training loop ──────────────────────────────────────────────────────────
ITERATIONS = 15

print(f"\n{'Iter':>5}  {'Mean reward':>12}  {'Min':>8}  {'Max':>8}")
print("-" * 42)

for i in range(ITERATIONS):
    result = algo.train()

    env_runners = result["env_runners"]

    print(f"{i:>5}  {env_runners['episode_return_mean']:>12.1f}  {env_runners['episode_return_min']:>8.1f}  {env_runners['episode_return_max']:>8.1f}")


_ = algo.save_to_path(Path("./checkpoint").resolve())

# print(f"Evaluation episode reward: {total_reward}")

# ── 7. Restore from checkpoint (optional demo) ────────────────────────────────
# restored = PPOConfig().build()
# restored.restore(checkpoint_path)

# ray.shutdown 3. B()
