"""
Minimal PPO training with RLlib (Ray 2.x) — CUDA GPU
======================================================
Install: pip install "ray[rllib]" gymnasium torch torchvision
Requires: CUDA-capable GPU + matching PyTorch CUDA build
"""

from pprint import pprint

import gymnasium as gym
import ray
import torch
from ray.rllib.algorithms.ppo import PPOConfig

# ── 0. Verify CUDA is available ───────────────────────────────────────────────
if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU detected. Ensure a CUDA-capable GPU is present and PyTorch was installed with CUDA support.\nCheck: https://pytorch.org/get-started/locally/")
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# ── 1. Initialise Ray ─────────────────────────────────────────────────────────
ray.init(ignore_reinit_error=True)

# ── 2. Build the PPO config ───────────────────────────────────────────────────
config = (
    PPOConfig()
    # Environment
    .environment(env="CartPole-v1")
    # Rollout workers — each runs the env in a separate process
    .env_runners(num_env_runners=2, num_envs_per_env_runner=1)
    # Training hyperparams
    .training(
        lr=3e-4,
        gamma=0.99,  # discount factor
        clip_param=0.2,  # was: clip_param (old API stack)
        train_batch_size_per_learner=4000,  # was: train_batch_size (old API stack)
        minibatch_size=128,  # was: sgd_minibatch_size
        num_epochs=10,  # was: num_sgd_iter
    )
    # GPU placement: num_gpus in .resources() is deprecated — use .learners() only
    .learners(num_learners=1, num_gpus_per_learner=1)
)

# ── 3. Build the algorithm ────────────────────────────────────────────────────
algo = config.build()

# ── 4. Training loop ──────────────────────────────────────────────────────────
ITERATIONS = 15

print(f"\n{'Iter':>5}  {'Mean reward':>12}  {'Min':>8}  {'Max':>8}")
print("-" * 42)

for i in range(1, ITERATIONS + 1):
    result = algo.train()

    pprint(result)

    # Metrics now live under "env_runners" (was "rollout_worker" in old stack)
    runners = result.get("env_runners", {})
    mean = runners.get("episode_reward_mean", float("nan"))
    low = runners.get("episode_reward_min", float("nan"))
    high = runners.get("episode_reward_max", float("nan"))

    print(f"{i:>5}  {mean:>12.1f}  {low:>8.1f}  {high:>8.1f}")

    # CartPole is considered "solved" at mean reward >= 475
    if mean >= 475:
        print(f"\nSolved at iteration {i}!")
        break

# ── 5. Save a checkpoint ──────────────────────────────────────────────────────
# algo.save() now returns a ray.train.Checkpoint object, not a plain string
checkpoint = algo.save("./ppo_cartpole_checkpoint")
checkpoint_path = checkpoint.path if hasattr(checkpoint, "path") else str(checkpoint)
print(f"\nCheckpoint saved -> {checkpoint_path}")

# ── 6. Evaluate one episode with the trained policy ───────────────────────────
env = gym.make("CartPole-v1")
obs, _ = env.reset()
total_reward = 0

for _ in range(500):
    action = algo.compute_single_action(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

print(f"Evaluation episode reward: {total_reward}")
env.close()

# ── 7. Restore from checkpoint (optional demo) ────────────────────────────────
# restored = PPOConfig().build()
# restored.restore(checkpoint_path)

ray.shutdown()
