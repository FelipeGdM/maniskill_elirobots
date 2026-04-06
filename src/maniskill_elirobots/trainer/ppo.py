import gymnasium as gym
import tyro
from mani_skill.utils.gym_utils import ManiSkillVectorEnv
from stable_baselines3 import PPO

from maniskill_elirobots.utils import CliArgs

ROBOT_UID = "ec63"


def main(args: CliArgs) -> None:

    print(args)

    env_kwargs = {
        "obs_mode": "state",
        "render_mode": "rgb_array",
        "sim_backend": "physx_cuda",
    }

    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode

    envs = gym.make(args.env_id, robot_uids=ROBOT_UID, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)

    # envs = RecordEpisode(
    #     envs,
    #     output_dir=f"runs/{run_name}/train_videos",
    #     save_trajectory=False,
    #     save_video_trigger=save_video_trigger,
    #     max_steps_per_video=args.num_steps,
    #     video_fps=30,
    #     info_on_video=True,
    # )

    envs = ManiSkillVectorEnv(
        envs,
        args.num_envs,
        ignore_terminations=not args.partial_reset,
        record_metrics=True,
    )
    # eval_envs = gym.make(args.env_id, robot_uids=ROBOT_UID, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, **env_kwargs)

    model = PPO("MlpPolicy", env=envs, verbose=1, tensorboard_log="./runs/les_goo", device="cuda")

    _ = model.learn(total_timesteps=args.total_timesteps, progress_bar=True)

    model.save("les_goo")


if __name__ == "__main__":
    # args = tyro.cli(CliArgs)  # pyright: ignore[reportAny]

    args = CliArgs(
        exp_name="topster",
    )
    main(args)
