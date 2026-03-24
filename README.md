# ManiSkill with Elite robots

```
sudo dnf build deps numpy sympy
PKG_CONFIG_PATH=(pwd)/.openblas poetry install
```

## Observation

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
