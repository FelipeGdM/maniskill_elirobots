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
from mani_skill.vector.wrappers.sb3 import ManiSkillSB3VectorEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecVideoRecorder

# from stable_baselines3.sac.policies import SACPolicy
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

    sb3_vec_envs = ManiSkillSB3VectorEnv(envs)

    model = PPO(
        policy="MlpPolicy",
        env=sb3_vec_envs,
        learning_rate=args.learning_rate,
        n_steps=args.num_steps,
        batch_size=args.minibatch_size,
        n_epochs=args.update_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_coef,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        use_sde=False,
        sde_sample_freq=-1,
        rollout_buffer_class=None,
        rollout_buffer_kwargs=None,
        target_kl=args.target_kl,
        stats_window_size=100,
        tensorboard_log=f"runs_sb3/{args.exp_name}",
        policy_kwargs={"net_arch": [256, 256]},
        verbose=1,
        seed=args.seed,
        device="cuda",
        _init_setup_model=True,
    )

    _ = model.learn(total_timesteps=args.total_timesteps, progress_bar=True)

    model.save(f"{args.exp_name}")


if __name__ == "__main__":
    args = CliArgs(
        env_id="FlipCoin-v1",
        exp_name=f"flipcoin-ec63-{int(datetime.datetime.now(tz=datetime.UTC).timestamp())}",
        # ent_coef=0.0,
    )
    main(args)
