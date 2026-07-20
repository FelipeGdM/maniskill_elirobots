import datetime
import time
from typing import Any

import gymnasium as gym
import torch
import tyro
from gymnasium.core import Env
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from mani_skill.utils.gym_utils import ManiSkillVectorEnv
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.vector.wrappers.sb3 import ManiSkillSB3VectorEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecVideoRecorder

# from stable_baselines3.sac.policies import SACPolicy
from maniskill_elirobots.utils import CliArgs
from maniskill_elirobots.wrappers.debug_video_recorder import DebugVecVideoRecorder

ROBOT_UID = "ec63"
VIDEO_FOLDER = "eval/videos"


def list_wrappers(env: gym.Env):
    curr: Env[Any, Any] = env
    print("Wrappers applied (from outermost to innermost):")
    while hasattr(curr, "env"):
        print(f"- {type(curr).__name__}")
        curr = curr.env
    print(f"- Base Environment: {type(curr).__name__}")


def main(args: CliArgs) -> None:

    args.batch_size = int(args.num_envs * args.num_steps)
    # args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    print("####")
    print(f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}")
    print(f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size} args.update_epochs={args.update_epochs}")
    print("####")

    # torch.use_deterministic_algorithms()
    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True

    env_kwargs = {
        "obs_mode": "state",
        "render_mode": "rgb_array",
        "ignore_terminations": True,
        "reward_mode": "normalized_dense",
        # "metadata": {
        #     "render_fps": 20,
        # },
        "reconfiguration_freq": args.reconfiguration_freq,
        "control_mode": "pd_joint_delta_pos",
        "sim_config": {
            "sim_freq": 100,
            "control_freq": 20,
        },
    }

    envs = ManiSkillVectorEnv(
        env="maniskill_elirobots:FlipCoin-v1",
        robot_uids=ROBOT_UID,
        num_envs=args.num_envs,
        sim_backend="physx_cuda",
        render_backend="sapien_cuda",
        record_metrics=True,
        **env_kwargs,
    )

    eval_envs = ManiSkillVectorEnv(
        "maniskill_elirobots:FlipCoin-v1",
        robot_uids="ec63",
        num_envs=1,
        sim_backend="physx_cpu",
        # render_backend="sapien_cpu",
        **env_kwargs,
    )

    sb3_vec_envs = ManiSkillSB3VectorEnv(envs)

    sb3_eval_vec_env = ManiSkillSB3VectorEnv(eval_envs)

    video_sb3_eval_vec_env = DebugVecVideoRecorder(
        sb3_eval_vec_env,
        video_folder=f"{VIDEO_FOLDER}/{args.exp_name}",
        record_video_trigger=lambda _: True,
        video_length=args.num_steps - 1,
        name_prefix=args.exp_name,
    )

    eval_callback = EvalCallback(
        video_sb3_eval_vec_env,
        best_model_save_path="./logs_sb3/",
        log_path="./logs_sb3/",
        eval_freq=(args.total_timesteps // args.num_envs) // 8,
        deterministic=True,
        render=True,
        warn=False,
    )

    model_kwargs = {
        "learning_rate": args.learning_rate,
        "n_steps": args.num_steps,
        "batch_size": args.minibatch_size,
        "n_epochs": args.update_epochs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_coef,
        "clip_range_vf": None,
        "normalize_advantage": True,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "use_sde": False,
        "sde_sample_freq": -1,
        "rollout_buffer_class": None,
        "rollout_buffer_kwargs": None,
        "target_kl": args.target_kl,
        "stats_window_size": 100,
        "tensorboard_log": f"runs_sb3/{args.exp_name}",
        "policy_kwargs": {"net_arch": [256, 256, 256]},
        "verbose": 1,
        "seed": args.seed,
        "device": "cuda",
        "_init_setup_model": True,
    }

    model = (
        PPO(
            policy="MlpPolicy",
            env=sb3_vec_envs,
            **model_kwargs,
        )
        if args.checkpoint is None
        else PPO.load(
            path=args.checkpoint,
            env=sb3_vec_envs,
        )
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=(args.total_timesteps // args.num_envs) // 4,
        save_path=f"checkpoints/{args.exp_name}",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    _ = model.learn(
        total_timesteps=args.total_timesteps,
        progress_bar=True,
        callback=[checkpoint_callback, eval_callback],
    )

    model.save(f"{args.exp_name}")

    video_sb3_eval_vec_env.close()


if __name__ == "__main__":
    args = CliArgs(
        total_timesteps=16_384_000,
        # num_envs=2_048,
        # update_epochs=4,
        env_id="FlipCoin-v1",
        exp_name=f"flipcoin-ec63-{int(datetime.datetime.now(tz=datetime.UTC).timestamp())}",
        ent_coef=1e-2,
        num_steps=100,
        minibatch_size=4096,
        # checkpoint="flipcoin-ec63-1784392890",
    )
    main(args)
