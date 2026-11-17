"""Step-level hard limit on the base XY linear velocity."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def clamp_base_speed(
    env: ManagerBasedRlEnv,
    env_ids,
    max_speed: float = 0.8,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Hard-clamp the body-frame XY linear velocity.

    The clamp is applied every control step after physics.  Z velocity and
    angular velocity are preserved.
    """
    asset = env.scene[asset_cfg.name]
    lin_b = asset.data.root_link_lin_vel_b.clone()
    ang_b = asset.data.root_link_ang_vel_b
    speed = torch.linalg.norm(lin_b[:, :2], dim=-1)
    scale = torch.clamp(
        max_speed / speed.clamp(min=1e-6),
        max=1.0,
    )
    lin_b[:, :2] *= scale[:, None]
    asset.write_root_link_velocity_b_to_sim(
        torch.cat([lin_b, ang_b], dim=-1)
    )
