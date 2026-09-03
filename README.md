# ManiSkill with Elite robots

Reinforcement learning research project for robot manipulation built on top of [ManiSkill](https://maniskill.ai/).
It registers custom robot agents and manipulation environments for the **EC63** arm (a 6-DOF arm with a two-finger claw gripper), plus a dual-arm variant, and trains PPO policies that run on massively parallel GPU simulation.

## Features

- **Custom ManiSkill agents** registered with `@register_agent`:
  - `ec63` — 6-DOF arm + 2-finger claw gripper (URDF + STL meshes under `src/maniskill_elirobots/assets/`).
  - `dual_ec63` — two EC63 arms mounted on a shared base.
  - `panda` — a copy of the standard Panda used for baseline comparisons.
- **Custom tasks** registered with `@register_env` (see [Tasks](#tasks)).
- **CleanRL-style PPO** trainer with MLP actor-critic, GAE, checkpointing, video recording, early stopping and TensorBoard/W&B logging.
- **Massively parallel training** using ManiSkill's GPU (`physx_cuda`) simulation with up to 1024+ parallel environments.
- **Hyperparameter search** via Optuna backed by PostgreSQL.
- **Distributed job queues** (Celery + RQ) to sweep many seeds across worker machines.
- **Dev container** definitions for CPU, NVIDIA CUDA and AMD ROCm hosts, plus a GitHub Actions workflow that publishes a CUDA image to GHCR.

## Repository layout

```
src/maniskill_elirobots/
├── assets/            # URDF + STL meshes for ec63 and dual_ec63
├── robots/            # Agent definitions: ec63.py, dual_ec63.py, panda.py
├── tasks/             # Environments: flip_coin_ec.py, pick_cube_ec.py, push_cube_ec.py
│   └── scene_builder/ # Custom actor builders (two-color coin, transparent sphere, ...)
├── trainer/           # ppo_cleanrl.py (main), ppo_skrl.py, ppo_rllib.py, ppo2.py
├── optimize/          # Optuna study entrypoint (cleanrl.py)
├── jobs/              # Celery app/tasks and RQ worker scripts
├── runner/            # Minimal checkpoint-inference example
├── utils/             # Agent class, RecordEpisode wrapper, math, eval/analysis scripts
└── __init__.py        # Re-exports EC63, DualEC63, FlipCoinEnv, PushCubeEcEnv

train/                 # fish shell scripts that launch training runs
vendor/ManiSkill       # git submodule: fork of haosulab/ManiSkill (editable install)
.devcontainer/         # CPU / gpu-cuda / gpu-rocm devcontainer definitions
tests/                 # pytest suite (CUDA availability smoke test)
runs/                  # TensorBoard logs + checkpoints (gitignored)
pth/                   # exported checkpoints (gitignored)
eval/                  # evaluation videos / trajectories (gitignored)
```

## Tasks

| Env ID | Robot(s) | Description |
|---|---|---|
| `FlipCoin-v1` | `ec63` | Grasp a two-color coin and flip it to the opposite orientation over the table. |
| `PickCubeEc-v1` | `panda`, `fetch`, `xarm6_robotiq`, `so100`, `widowxai`, `ec63` | Grasp a cube and move it to a target goal position (adapted from ManiSkill's `PickCube`). |
| `PushCubeEc-v1` | `panda`, `fetch`, `ec63` | Push a cube to a red/white goal region on the table (adapted from ManiSkill's `PushCube`). |

All tasks use dense/normalized-dense rewards and support state observations as well as visual observations.

## Installation

The project is managed with [uv](https://docs.astral.sh/uv/) and requires Python >= 3.11.
ManiSkill is pinned as a git submodule and installed editable from `vendor/ManiSkill`.

```bash
# 1. Clone with submodules (or update them after a regular clone)
git clone --recurse-submodules git@github.com:FelipeGdM/maniskill_elirobots.git
# or: git submodule update --init --recursive

# 2. Install dependencies
uv sync
```

### Dev containers

Three ready-made development containers are provided under `.devcontainer/`:

- `cpu` — CPU-only (LLVMpipe / Intel rendering).
- `gpu-cuda` — NVIDIA CUDA (uses `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime`).
- `gpu-rocm` — AMD ROCm.

`postCreateCommand` runs `uv sync --inexact`. The GPU-CUDA image is rebuilt and published to GHCR on every push to `main` via `.github/workflows/docker-publish.yml`.

### Optional: Optuna PostgreSQL storage

The `docker-compose.yml` starts a PostgreSQL 18 container (`devuser`/`devpass`, database `optuna`) used by the Optuna study storage:

```bash
docker compose up -d db
```

## Quick start

Visualize the EC63 robot under random actions:

```bash
uv run show-robot
```

Train a PPO policy on the flip-coin task with the EC63:

```bash
python src/maniskill_elirobots/trainer/ppo_cleanrl.py \
    --env_id="FlipCoin-v1" \
    --robot_uid="ec63" \
    --exp_name="flipcoin-ec63-01"
```

Evaluate a checkpoint:

```bash
python src/maniskill_elirobots/trainer/ppo_cleanrl.py \
    --env_id="FlipCoin-v1" \
    --evaluate \
    --checkpoint="runs/<run>/ckpt_8192000.pt"
```

Watch training in TensorBoard:

```bash
tensorboard --bind_all --logdir ./runs
```

### Training scripts

The `train/` directory contains fish scripts for common configurations, e.g.:

```fish
set -e DISPLAY
python src/maniskill_elirobots/trainer/ppo_cleanrl.py \
    --env_id="FlipCoin-v1" \
    --exp_name="flipcoin-ec63-(date +%s)-(git rev-parse --short HEAD)"
```

## Training configuration

All hyperparameters are exposed via the `CliArgs` dataclass in `src/maniskill_elirobots/utils/cli.py` and can be overridden as CLI flags. Notable defaults:

| Flag | Default | Description |
|---|---|---|
| `--env_id` | `FlipCoin-v1` | Environment ID |
| `--robot_uid` | `ec63` | Robot unique ID |
| `--total_timesteps` | `8_192_000` | Total training timesteps |
| `--num_envs` | `1024` | Number of parallel environments |
| `--num_eval_envs` | `32` | Number of parallel eval environments |
| `--control_mode` | `pd_joint_delta_pos` | Arm controller (delta joint position) |
| `--learning_rate` | `3e-4` | Adam learning rate |
| `--gamma` / `--gae_lambda` | `0.8` / `0.9` | Discount / GAE lambda |
| `--update_epochs` | `10` | PPO update epochs |
| `--clip_coef` | `0.2` | PPO surrogate clipping |
| `--ent_coef` | `1e-2` | Entropy coefficient |
| `--eval_freq` | `16` | Evaluate every N iterations |
| `--early_stop` | `False` | Stop when eval success reaches 100% |
| `--capture_video` | `True` | Record eval (and optionally train) videos |
| `--track` | `False` | Log to Weights & Biases |
| `--tensorboard_folder` | `runs` | Root folder for TensorBoard data |

The trainer (`ppo_cleanrl.py`) uses the `normalized_dense` reward mode, seeds everything deterministically, saves checkpoints to `{tensorboard_folder}/{run_name}/ckpt_{step}.pt` and returns evaluation metrics that Optuna consumes.

## Hyperparameter optimization

`src/maniskill_elirobots/optimize/cleanrl.py` defines an Optuna study (`flipcoin_ppo_how_long`) that sweeps over seeds and returns the total timesteps needed to solve the task. It stores trials in the PostgreSQL database from `docker-compose.yml`:

```bash
uv run optimize     # entrypoint defined in pyproject.toml
```

`dash.sh` launches the Optuna dashboard:

```bash
./dash.sh
# optuna-dashboard postgresql+psycopg://devuser:devpass@<host>:5432/optuna
```

## Distributed training queues

Multiple training runs can be distributed via task queues:

- **Celery** (`jobs/celery_app.py`, `jobs/celery_tasks.py`) — queue a batch of seeds:
  ```bash
  ./celery.sh   # start a worker (concurrency 1, one GPU run at a time)
  ./flower.sh   # Flower monitoring dashboard on port 5555
  python -m maniskill_elirobots.jobs.celery_tasks  # enqueue seed jobs
  ```
- **RQ** (`jobs/redis_worker.py`) — alternative queue-based sweep (Redis-backed).

Both assume a Redis broker; the addresses can be adjusted in the respective files.

## Evaluation & analysis utilities

- `utils/eval.py` — load a checkpoint, run episodes on `PushCubeEc-v1`, record videos via `RecordEpisode` and dump observations to `output.csv`.
- `utils/analyze.py` — read `output.csv` and compare raw actions vs. target joint positions (used to validate the delta-position controller behavior).
- `utils/env_photo.py` — run a checkpoint and save rendered frames / sensor images.
- `utils/episode_logger.py` — a `RecordEpisode` gym wrapper for saving trajectories (`.h5` + `.json`) and videos.
- `utils/maniskill_vector_env.py` — an alternative Gymnasium `VectorEnv` implementation for ManiSkill GPU envs with episode metrics (`success_once`, `success_at_end`, ...).

## Observations

State observations are provided as a nested dict (`obs_mode="state_dict"`):

```python
from torch import Tensor

data = {
    "agent": {
        "qpos": Tensor([[0.0000, -2.3562, 1.9635, -1.1781, 1.5708, 0.0000, 0.0000, 0.0000]]), # 8
        "qvel": Tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), # 8
    },
    "extra": {
        # {x, y, z, q1, q2, q3, q4} => 7
        "tcp_pose": Tensor([[-2.5641e-01, 1.0300e-01, 2.3089e-01, -2.6125e-06, -7.0711e-01, 7.0711e-01, -7.7824e-08]]),
        # {x, y, z} => 3
        "goal_pos": Tensor([[0.1792, 0.0844, 0.0010]]),
        # {x, y, z, q1, q2, q3, q4} => 7
        "obj_pose": Tensor([[-0.0208, 0.0844, 0.0200, 1.0000, 0.0000, 0.0000, 0.0000]]),
    },
}
```

With `obs_mode="state"` the dict is flattened into a vector, which is what the MLP policy consumes.

## Lessons learned

- In `PDJointPosControllerConfig`, the option `normalize_action` means that the action produced by the agent will be clipped to [-1,1] and then scaled to the min and max of the controller.
- The output action of the agent may be any real number, `normalize_action=True` means that the action space is better used.
- For thin objects, setting `lower=None` on the gripper mimic controller is a trick to keep grasp force even when the object is thin.
- A large `found_lost_pairs_capacity` and `max_rigid_patch_count` in the GPU sim config helps avoid memory errors when many environments run in parallel.

## Development

Lint and type-check:

```bash
uv run ruff check src tests
uv run basedpyright src
```

Run tests:

```bash
uv run pytest tests
```

> Note: `tests/test_gpu.py` asserts that CUDA is available, so it only passes on a GPU host.

## TODO

See `TODO.md`:

- Write a numpy render wrapper to render videos with SB3.

## License

MIT — see `pyproject.toml`. ManiSkill is a third-party dependency (git submodule under `vendor/ManiSkill`).