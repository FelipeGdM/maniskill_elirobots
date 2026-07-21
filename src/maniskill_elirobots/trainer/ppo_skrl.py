import argparse
import os
import sys
from typing import override

import gymnasium as gym
import mani_skill.envs
import torch
import torch.nn as nn
import tyro  # needed to register the ManiSkill environment entry points
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

# import the skrl components to build the RL system
from skrl import logger
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from maniskill_elirobots.utils import CliArgs

# parse arguments
# parser = argparse.ArgumentParser()
# parser.add_argument("--num_envs", type=int, default=2048, help="Number of environments")
# parser.add_argument("--headless", action="store_true", help="Run in headless mode (no rendering)")
# parser.add_argument("--seed", type=int, default=None, help="Random seed")
# parser.add_argument("--checkpoint", type=str, default=None, help="Load checkpoint from path")
# parser.add_argument("--eval", action="store_true", help="Run in evaluation mode (logging/checkpointing disabled)")
# args, _ = parser.parse_known_args()

args = tyro.cli(CliArgs)


# seed for reproducibility
_ = set_seed(args.seed)  # e.g. `set_seed(42)` for fixed seed


# define models (stochastic and deterministic models) using mixins
class Policy(GaussianMixin, Model):
    def __init__(  # noqa: PLR0913
        self,
        observation_space,
        state_space,
        action_space,
        device,
        *,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20,
        max_log_std=2,
        reduction="sum",
    ):
        Model.__init__(self, observation_space=observation_space, state_space=state_space, action_space=action_space, device=device)
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction=reduction,
        )

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    @override
    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {"log_std": self.log_std_parameter}


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device):
        Model.__init__(self, observation_space=observation_space, state_space=state_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

    @override
    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {}


# load the environment
env_id = args.env_id
env_kwargs = {
    "obs_mode": "state",
    # "render_mode": "rgb_array",
    "render_mode": "sapien",
    "sim_backend": "physx_cuda",
    # "record_metrics": True,
    # "ignore_terminations": True,
    "reward_mode": "normalized_dense",
    # "render_backend": "sapien_cuda",
    "reconfiguration_freq": args.reconfiguration_freq,
    "control_mode": "pd_joint_delta_pos",
    "sim_config": {
        "sim_freq": 100,
        "control_freq": 20,
    },
}

# env = gym.make(
#     env_id,
#     num_envs=args.num_envs,
#     **env_kwargs,
# )

env = ManiSkillVectorEnv(
    env="maniskill_elirobots:FlipCoin-v1",
    robot_uids="ec63",
    num_envs=args.num_envs,
    render_backend="sapien_cuda",
    **env_kwargs,
)

# print(env)
# print(type(env))
# sys.exit()

# wrap the environment
env = wrap_env(env, wrapper="mani-skill")

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=args.num_steps, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = Policy(env.observation_space, env.state_space, env.action_space, device, clip_actions=True).to(device)
models["value"] = Value(env.observation_space, env.state_space, env.action_space, device).to(device)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_CFG()
cfg.rollouts = args.num_steps  # memory_size
cfg.learning_epochs = args.update_epochs
cfg.mini_batches = args.num_minibatches
cfg.discount_factor = args.gamma
cfg.gae_lambda = args.gae_lambda
cfg.learning_rate = args.learning_rate
cfg.learning_rate_scheduler = KLAdaptiveLR
cfg.learning_rate_scheduler_kwargs = {"kl_threshold": 0.008}
cfg.grad_norm_clip = args.max_grad_norm
cfg.ratio_clip = args.clip_coef
cfg.value_clip = 0.2
cfg.entropy_loss_scale = args.ent_coef
cfg.value_loss_scale = args.vf_coef
cfg.observation_preprocessor = RunningStandardScaler
cfg.observation_preprocessor_kwargs = {"size": env.observation_space, "device": device}
cfg.value_preprocessor = RunningStandardScaler
cfg.value_preprocessor_kwargs = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg.experiment.write_interval = 0  # "auto"
cfg.experiment.checkpoint_interval = 0  # "auto"
cfg.experiment.directory = f"runs_skrl/torch/{args.exp_name}"

agent = PPO(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    state_space=env.state_space,
    action_space=env.action_space,
    device=device,
)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": args.total_timesteps // args.num_envs, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

if args.checkpoint:
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint file not found: '{args.checkpoint}'")
        sys.exit(1)
    agent.load(args.checkpoint)

trainer.train()


# if args.checkpoint is not None else trainer.eval()
