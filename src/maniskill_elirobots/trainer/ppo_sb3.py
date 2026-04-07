from typing import Any

import gymnasium as gym
import torch
import tyro
from gymnasium.core import Env
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from mani_skill.utils.gym_utils import ManiSkillVectorEnv
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.sb3 import ManiSkillSB3VectorEnv
from stable_baselines3 import PPO

from maniskill_elirobots.utils import CliArgs

ROBOT_UID = "ec63"


def list_wrappers(env: gym.Env):
    curr: Env[Any, Any] = env
    print("Wrappers applied (from outermost to innermost):")
    while hasattr(curr, "env"):
        print(f"- {type(curr).__name__}")
        curr = curr.env
    print(f"- Base Environment: {type(curr).__name__}")


def main(args: CliArgs) -> None:

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    env_kwargs = {
        "obs_mode": "state",
        "render_mode": "rgb_array",
        "sim_backend": "physx_cuda",
        "render_backend": "sapien_cuda",
        "control_mode": "pd_joint_delta_pos",
        "sim_config": {
            "sim_freq": 100,
            "control_freq": 20,
        },
    }

    envs = gym.make(
        args.env_id,
        robot_uids=ROBOT_UID,
        num_envs=args.num_envs,
        reconfiguration_freq=args.reconfiguration_freq,
        **env_kwargs,
    )

    def save_video_trigger(x: int) -> bool:
        return (x % args.save_train_video_freq) == 0  # pyright: ignore[reportOperatorIssue]

    # video_envs = RecordEpisode(
    #     envs,
    #     output_dir="videos",
    #     save_video_trigger=save_video_trigger,
    #     max_steps_per_video=50,
    #     video_fps=30,
    #     info_on_video=True,
    # )

    video_envs = RecordVideo(
        envs,
        video_folder="videos2",
        step_trigger=save_video_trigger,
        video_length=50,
        fps=30,
    )

    sb3_vec_envs = ManiSkillSB3VectorEnv(video_envs)

    print(sb3_vec_envs.get_attr("render_mode"))

    model = PPO(
        "MlpPolicy",
        env=sb3_vec_envs,
        verbose=1,
        tensorboard_log=f"./runs/{args.exp_name}",
        device="cuda",
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        batch_size=args.minibatch_size,
        n_steps=args.num_steps,
        n_epochs=args.update_epochs,
        # clip_range=args.clip,
        # clip_range_vf: None | float | Schedule = None,
        normalize_advantage=True,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        rollout_buffer_class=None,
        rollout_buffer_kwargs=None,
        target_kl=None,
        stats_window_size=100,
        policy_kwargs=None,
    )

    _ = model.learn(total_timesteps=args.total_timesteps, progress_bar=True)

    model.save(f"{args.exp_name}")


if __name__ == "__main__":
    # args = tyro.cli(CliArgs)  # pyright: ignore[reportAny]

    args = CliArgs(
        exp_name="topster2",
        num_envs=256,
    )
    main(args)
