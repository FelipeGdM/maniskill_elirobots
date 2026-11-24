# AGENTS.md

Context for AI coding agents working in this repository.

## Project

Reinforcement learning research project on top of ManiSkill for the EC63 robot
(6-DOF arm + two-finger claw) and its dual-arm variant (DualEC63). Registered
custom ManiSkill agents, custom tasks, PPO trainers, Optuna hyperparameter
search and distributed job queues (Celery/RQ). Author: Felipe Gomes de Melo.

## Key commands

- Install / sync deps: `uv sync`
- Train (main trainer): `python src/maniskill_elirobots/trainer/ppo_cleanrl.py --env_id="FlipCoin-v1" --robot_uid="ec63"`
- Evaluate a checkpoint: add `--evaluate --checkpoint="runs/<run>/ckpt_8192000.pt"`
- Show robot demo: `uv run show-robot`
- Lint: `uv run ruff check src tests` (ruff config in `pyproject.toml`, line-length 256)
- Type-check: `uv run basedpyright src`
- Tests: `uv run pytest tests` (note: `tests/test_gpu.py` asserts CUDA availability)
- TensorBoard: `tensorboard --bind_all --logdir ./runs`

## Project layout

- `src/maniskill_elirobots/`
  - `robots/` — agents: `ec63.py`, `dual_ec63.py`, `panda.py` (registered via `@register_agent`)
  - `tasks/` — envs: `flip_coin_ec.py`, `pick_cube_ec.py`, `push_cube_ec.py` (registered via `@register_env`)
  - `trainer/` — `ppo_cleanrl.py` is the main, actively used trainer; `ppo_skrl.py`, `ppo_rllib.py`, `ppo2.py` are secondary/experimental
  - `optimize/cleanrl.py` — Optuna study entrypoint (`uv run optimize`)
  - `jobs/` — Celery app/tasks and RQ worker
  - `utils/` — `Agent` class, `RecordEpisode` wrapper (`episode_logger.py`), eval/analyze scripts, `CliArgs` in `cli.py`
  - `assets/` — URDF + STL meshes for ec63/dual_ec63
- `vendor/ManiSkill` — git submodule, installed editable via `[tool.uv.sources]` in `pyproject.toml`
- `train/` — fish launch scripts; `runs/` — TensorBoard logs + checkpoints (gitignored)
- `.devcontainer/{cpu,gpu-cuda,gpu-rocm}` — dev containers; CUDA image published to GHCR on `main` push

## Conventions

- Python >= 3.11, package managed with uv (`uv_build` backend), code lives in `src/`.
- Ruff config: `select = ["ALL"]`, ignores `D, ANN, ERA, RUF012, T201, C408`; tests get extra per-file ignores.
- Do NOT add comments to code unless asked; keep response style concise.
- NEVER commit unless the user explicitly asks.

## Important runtime details

- `CliArgs` (utils/cli.py) is a `kw_only` dataclass; all training hyperparameters are CLI flags (tyro). Defaults: `env_id="FlipCoin-v1"`, `robot_uid="ec63"`, `control_mode="pd_joint_delta_pos"`, `num_envs=1024`, `total_timesteps=8_192_000`, `tensorboard_folder="runs"`.
- Trainer uses `reward_mode="normalized_dense"` (hardcoded in `ppo_cleanrl.py`), `physx_cuda` sim when CUDA is available.
- EC63 arm uses PD joint delta-position control (`use_delta=True`, `normalize_action=True`, arm limits ±0.1); gripper uses PD mimic controller (`lower=None` for thin objects).
- Checkpoints are raw `state_dict`s saved as `ckpt_{global_step}.pt`; step is parsed from filename on resume.
- Optuna study storage is PostgreSQL (`docker-compose.yml`, db `optuna`, user `devuser`/`devpass`, host from `POSTGRES_ADDR` env, default Tailscale IP 100.95.80.82).
- Redis broker/backend addresses are hardcoded in `jobs/celery_app.py` and `jobs/redis_worker.py` (172.17.0.x); update when hosts change.

## Gotchas

- `tests/test_gpu.py` will fail on CPU-only machines (asserts `torch.cuda.is_available()`).
- `runner/runner.py` and `utils/eval.py` reference absolute paths under `/workspaces/...` from a devcontainer; paths must be adjusted when run locally.
- Eval/train videos, `output.csv`, `pth/`, `eval/` are gitignored.