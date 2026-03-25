from typing import TYPE_CHECKING, cast

import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from torch import Tensor, nn
from torch.distributions.normal import Normal

from maniskill_elirobots import EC63

SQRT_2 = cast("float", np.sqrt(2))

OBSERVATION_SPACE_SIZE = 33
ACTION_SPACE_SIZE = 7


def layer_init(layer: nn.Module, std: float = SQRT_2, bias_const: float = 0.0):
    _ = torch.nn.init.orthogonal_(layer.weight, std)
    _ = torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def build_state(qpos: Tensor, qvel: Tensor, tcp_pose: Tensor, goal_pos: Tensor, obj_pose: Tensor):
    return torch.cat((qpos, qvel, tcp_pose, goal_pos, obj_pose), -1)


class Agent(nn.Module):
    def __init__(self, observation_space_size: int, action_space_size: int):
        super().__init__()
        self.observation_space_size = observation_space_size
        self.action_space_size = action_space_size
        self.critic = nn.Sequential(
            layer_init(nn.Linear(observation_space_size, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(observation_space_size, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_space_size), std=0.01 * SQRT_2),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, action_space_size) * -0.5)

    def get_value(self, x: Tensor) -> Tensor:
        return cast("Tensor", self.critic(x))

    def get_action(self, x: Tensor, deterministic: bool = False):  # noqa: FBT001, FBT002
        action_mean = cast("Tensor", self.actor_mean(x))
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, x: Tensor, action: Tensor | None = None):
        action_mean = cast("Tensor", self.actor_mean(x))
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


if TYPE_CHECKING:
    from collections import OrderedDict

CHECKPOINT = "/workspaces/maniskill_elirobots/runs/PushCubeEc-v1__ppo__1__1774028711/ckpt_4096000.pt"
ENV_ID = "PushCubeEc-v1"
ROBOT_UID = "ec63"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = cast("OrderedDict[str, Tensor]", torch.load(CHECKPOINT, map_location=device))

agent = Agent(OBSERVATION_SPACE_SIZE, ACTION_SPACE_SIZE).to(device)

_ = agent.load_state_dict(model)

qpos = Tensor(EC63.keyframes["rest"].qpos).reshape((1, 8))

qvel = torch.zeros((1, 8))

tcp_pose = Tensor([[-2.5641e-01, 1.0300e-01, 2.3089e-01, -2.6125e-06, -7.0711e-01, 7.0711e-01, -7.7824e-08]])

goal_pos = Tensor([[0.1792, 0.0844, 0.0010]])

obj_pose = Tensor([[-0.0208, 0.0844, 0.0200, 1.0000, 0.0000, 0.0000, 0.0000]])

state = build_state(qpos, qvel, tcp_pose, goal_pos, obj_pose).to(device)

action = agent.get_action(state, deterministic=True)

print(action)
