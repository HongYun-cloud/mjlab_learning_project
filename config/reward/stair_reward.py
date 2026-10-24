from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity_stair(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward for tracking the commanded base linear velocity (xy only).

    与 ``mdp.track_linear_velocity`` 的区别：
    1. 不惩罚垂直速度——爬楼梯必然有 vz，惩罚它会与任务目标冲突。
    2. 跟踪偏差大时返回负值（扣奖励），而不是只趋近于 0，从而打破"站姿局部最优"。
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_lin_vel_b
    xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
    # exp(-e/std^2) 在 e 较大时趋于 0，站着不动没有任何代价；
    # 减去 1.0 后，跟踪越差扣得越多（最大 -1），跟踪完美时为 0。
    return torch.exp(-xy_error / std**2) - 1.0