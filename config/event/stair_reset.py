from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
import os

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_from_euler_xyz,
  quat_mul,
  sample_uniform,
)
if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer
from mjlab.envs.mdp.events import resolve_env_ids


def reset_root_state_from_stair(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: str = "robot",
    nominal_height: float = 0.28,  # 稍微压低一点标称高度（Go2趴下/微蹲约为 0.25~0.30m）
    num_steps: int = 5,            # 前面计算出的单侧台阶级数：5 级
    step_height_range: tuple[float, float] = (0.0, 0.1), # 与配置中的 (0.0, 0.1) 一致
    specified_level: int | str | None = None,
):
    if len(env_ids) == 0:
        return

    num_resets = len(env_ids)
    asset = env.scene[asset_cfg]

    # 0. 检查并同步指定的地形难度等级（Level 0 ~ num_rows-1）
    terrain = getattr(env.scene, "terrain", None)
    if terrain is not None and hasattr(terrain, "terrain_levels") and getattr(terrain, "terrain_origins", None) is not None:
        target_lvl = getattr(env, "specified_terrain_level", None)
        if target_lvl is None and specified_level is not None:
            target_lvl = specified_level
        if target_lvl is None:
            env_var = os.getenv("STAIR_LEVEL") or os.getenv("TERRAIN_LEVEL")
            if env_var is not None:
                try:
                    target_lvl = int(env_var)
                except ValueError:
                    target_lvl = env_var

        if target_lvl is not None:
            num_rows = terrain.terrain_origins.shape[0]
            if str(target_lvl).lower() == "random" or target_lvl == -1:
                terrain.terrain_levels[env_ids] = torch.randint(
                    0, num_rows, (num_resets,), device=env.device
                )
            else:
                try:
                    lvl = max(0, min(int(target_lvl), num_rows - 1))
                    terrain.terrain_levels[env_ids] = lvl
                except (ValueError, TypeError):
                    pass

            env.scene.env_origins[env_ids] = terrain.terrain_origins[
                terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
            ]
    
    # 1. 克隆默认状态
    root_states = asset.data.default_root_state[env_ids].clone()

    # 2. 采样平面 (x, y) 位置（倒金字塔中心凹底平地）
    dx = torch.empty(num_resets, device=env.device).uniform_(*pose_range.get("x", (-0.5, 0.5)))
    dy = torch.empty(num_resets, device=env.device).uniform_(*pose_range.get("y", (-0.5, 0.5)))

    origins = env.scene.env_origins[env_ids]
    target_x = origins[:, 0] + dx
    target_y = origins[:, 1] + dy

    # 3. 高度计算：倒金字塔的 origins[:, 2] 即为中心凹底地面的真实表面高度，无需落差补偿
    dz = torch.empty(num_resets, device=env.device).uniform_(*pose_range.get("z", (0.0, 0.0)))
    target_z = origins[:, 2] + nominal_height + dz

    # 写入目标位置
    root_states[:, 0] = target_x
    root_states[:, 1] = target_y
    root_states[:, 2] = target_z

    # 4. 设置朝向 Yaw
    if "yaw" in pose_range:
        yaw = torch.empty(num_resets, device=env.device).uniform_(*pose_range["yaw"])
        half_yaw = yaw * 0.5
        root_states[:, 3] = torch.cos(half_yaw)
        root_states[:, 4] = 0.0
        root_states[:, 5] = 0.0
        root_states[:, 6] = torch.sin(half_yaw)
    else:
        yaw = torch.zeros(num_resets, device=env.device)
        root_states[:, 3] = 1.0
        root_states[:, 4] = 0.0
        root_states[:, 5] = 0.0
        root_states[:, 6] = 0.0

    # 5. 速度清零
    root_states[:, 7:13] = 0.0

    # 6. 写回仿真环境
    asset.write_root_state_to_sim(root_states, env_ids=env_ids)

    # 7. 记录出生点 XY 坐标与朝向（供 Curriculum 课程学习计算真实行走位移）
    if not hasattr(env, "spawn_root_pos_w"):
        env.spawn_root_pos_w = torch.zeros((env.num_envs, 2), device=env.device)
    if not hasattr(env, "spawn_heading_w"):
        env.spawn_heading_w = torch.zeros(env.num_envs, device=env.device)

    env.spawn_root_pos_w[env_ids, 0] = target_x
    env.spawn_root_pos_w[env_ids, 1] = target_y
    if "yaw" in pose_range:
        env.spawn_heading_w[env_ids] = yaw
    else:
        env.spawn_heading_w[env_ids] = 0.0


def idle_termination(
    env: ManagerBasedRlEnv,
    asset_cfg: str = "robot",
    speed_threshold: float = 0.05,
    min_steps: int = 60,
) -> torch.Tensor:
    """Terminate idle envs whose robot base xy speed stays below threshold.

    Catches policy collapse where the robot stands still without triggering
    any other termination (no fall, no out-of-bounds), wasting the whole
    episode.  Only active after ``min_steps`` so a fresh reset has time to
    start moving.
    """
    asset = env.scene[asset_cfg]
    speed = torch.norm(asset.data.root_link_vel_w[:, :2], dim=-1)
    idle = speed < speed_threshold


    return idle

def out_of_patch_bounds(
    env: ManagerBasedRlEnv,
    half_size: float = 4.0,
    margin: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the robot wanders beyond its own sub-terrain patch.

    与内置 out_of_terrain_bounds 不同：
    - 后者以整个地形 grid + border_width 为边界
      （楼梯地形 border_width=20m，几乎永不触发）
    - 本函数以单个 sub-terrain patch 的半尺寸为边界，
      机器人离开自己的 patch 即终止，强迫 episode reset。

    Parameters
    ----------
    half_size:
        patch 的半边长（m），应与 terrain_generator.size[0] / 2 对应。
        楼梯地形 size=(8, 8) → 默认 4.0 m。
    margin:
        安全裕量：在 half_size 内缩，避免台阶边缘误触发，默认 0.3 m。
    """
    asset: Entity = env.scene[asset_cfg.name]
    # 机器人相对自身 env_origin 的 XY 位移
    rel_xy = asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    limit = half_size - margin
    out = (rel_xy[:, 0].abs() > limit) | (rel_xy[:, 1].abs() > limit)
    # 前 2 步跳过，避免 reset 后位置未同步时误触发
    out &= env.episode_length_buf > 2
    return out
