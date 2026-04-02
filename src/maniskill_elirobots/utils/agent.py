from collections import OrderedDict
from typing import cast, final

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions.normal import Normal

SQRT_2 = cast("float", np.sqrt(2))


def layer_init(layer: nn.Module, std: float = SQRT_2, bias_const: float = 0.0):
    _ = torch.nn.init.orthogonal_(layer.weight, std)
    _ = torch.nn.init.constant_(layer.bias, bias_const)
    return layer


@final
class Agent(nn.Module):
    def __init__(self, observation_space_size: int, action_space_size: int, device: torch.device):
        super().__init__()
        self.observation_space_size = observation_space_size
        self.action_space_size = action_space_size
        self.device = device
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
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )
