from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import wrap_to_pi
from myrobot_mjlab.config.reward.Trackingreward import _get_goal_pos

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


# ==============================================================================
# Actor 观测项
# ==============================================================================

def direction_to_goal(
    env: ManagerBasedRlEnv,
    command_name: str = "Goal_direction",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    in_body_frame: bool = True,
    normalize: bool = False,
    **kwargs,
) -> torch.Tensor:
    """当前位置到目标位置的向量 (r_goal - x)。Shape: (num_envs, 2).

    Args:
        in_body_frame: 是否投影到机身局部坐标系 (推荐 True，符合 Actor 自身观测视角).
        normalize: 是否归一化为单位向量 (默认 False，保留距离幅度信息).
    """
    asset: Entity = env.scene[asset_cfg.name]
    robot_pos = asset.data.root_link_pos_w - env.scene.env_origins
    goal_pos = _get_goal_pos(env, command_name=command_name, **kwargs)

    to_goal_w = goal_pos[:, :2] - robot_pos[:, :2]

    if in_body_frame:
        # 旋转到机器人机身局部坐标系 (前向 +X, 左向 +Y)
        yaw = asset.data.heading_w
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        dx_b = cos_yaw * to_goal_w[:, 0] + sin_yaw * to_goal_w[:, 1]
        dy_b = -sin_yaw * to_goal_w[:, 0] + cos_yaw * to_goal_w[:, 1]
        vec = torch.stack([dx_b, dy_b], dim=-1)
    else:
        vec = to_goal_w

    if normalize:
        norm = torch.linalg.norm(vec, dim=-1, keepdim=True)
        vec = torch.where(norm > 1e-6, vec / norm, torch.zeros_like(vec))

    return vec


def heading_to_goal_error(
    env: ManagerBasedRlEnv,
    command_name: str = "Goal_direction",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    **kwargs,
) -> torch.Tensor:
    """当前机身朝向与目标方向的夹角误差 (wrap 到 [-pi, pi]). Shape: (num_envs, 1)."""
    asset: Entity = env.scene[asset_cfg.name]
    robot_pos = asset.data.root_link_pos_w - env.scene.env_origins
    goal_pos = _get_goal_pos(env, command_name=command_name, **kwargs)

    to_goal = goal_pos[:, :2] - robot_pos[:, :2]
    theta_goal = torch.atan2(to_goal[:, 1], to_goal[:, 0])
    heading = asset.data.heading_w
    head_err = wrap_to_pi(theta_goal - heading)
    return head_err.unsqueeze(-1)


def stair_boolean(
    env: ManagerBasedRlEnv,
    is_stair: bool | None = None,
    **kwargs,
) -> torch.Tensor:
    """标志位：当前任务是否为楼梯任务 (楼梯任务为 1.0，非楼梯任务为 0.0). Shape: (num_envs, 1).

    使用方式：
    1. 显式指定：params={"is_stair": True} 或 params={"is_stair": False}
    2. 自动判定（当 is_stair=None 时）：自动检测当前场景的 terrain 类型是否为楼梯。
    """
    if "is_stair" in kwargs and kwargs["is_stair"] is not None:
        flag = bool(kwargs["is_stair"])
    elif is_stair is not None:
        flag = bool(is_stair)
    else:
        # 自动检测 terrain 配置
        terrain = getattr(env.scene, "terrain", None)
        if terrain is not None and getattr(terrain.cfg, "terrain_type", None) == "generator":
            gen = getattr(terrain.cfg, "terrain_generator", None)
            gen_str = str(type(gen).__name__) + str(getattr(gen, "sub_terrains", ""))
            flag = "stair" in gen_str.lower()
        else:
            flag = False

    val = 1.0 if flag else 0.0
    return torch.full((env.num_envs, 1), val, device=env.device, dtype=torch.float32)


# ==============================================================================
# Critic 特权观测项
# ==============================================================================

def base_height(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """机身相对环境原点的真实垂直高度 (z - z_origin). Shape: (num_envs, 1)."""
    asset: Entity = env.scene[asset_cfg.name]
    pos_z = asset.data.root_link_pos_w[:, 2:3] - env.scene.env_origins[:, 2:3]
    return pos_z


def direction_real(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    eps: float = 1e-6,
) -> torch.Tensor:
    """机器人当前真实运动方向 (XY 平面归一化速度向量). Shape: (num_envs, 2)."""
    asset: Entity = env.scene[asset_cfg.name]
    vel_xy = asset.data.root_link_lin_vel_w[:, :2]
    speed = torch.linalg.norm(vel_xy, dim=-1, keepdim=True)
    return torch.where(speed > eps, vel_xy / speed, torch.zeros_like(vel_xy))


def friction_coefficient(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """地面/足端接触摩擦系数 mu. Shape: (num_envs, 1)."""
    asset: Entity = env.scene[asset_cfg.name]
    if hasattr(env.sim, "model") and hasattr(env.sim.model, "geom_friction"):
        fric = env.sim.model.geom_friction
        if fric.ndim == 3:  # (num_envs, n_geoms, 3)
            geom_idx = asset.indexing.geom_ids[0] if len(asset.indexing.geom_ids) > 0 else 0
            return fric[:, geom_idx, 0:1]
        elif fric.ndim == 2:  # (n_geoms, 3)
            geom_idx = asset.indexing.geom_ids[0] if len(asset.indexing.geom_ids) > 0 else 0
            val = fric[geom_idx, 0]
            return torch.full((env.num_envs, 1), float(val), device=env.device)
    return torch.ones((env.num_envs, 1), device=env.device)


__all__ = [
    "direction_to_goal",
    "heading_to_goal_error",
    "stair_boolean",
    "base_height",
    "direction_real",
    "friction_coefficient",
]
