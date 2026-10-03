from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: str = "robot",
    nominal_height: float = 0.32,  # 机器人在地面上的站立高度/出生标称净空 (Go2约为 0.30~0.35m)
):
    """
    根据当前子环境的地形高度，动态计算初始根节点高度并重置机器人状态。
    兼容课程学习中不同难度导致台阶中心高度变化的情况。
    """
    if len(env_ids) == 0:
        return

    # 1. 获取目标资产（机器人）
    asset = env.scene[asset_cfg]

    # 2. 提取并克隆默认的 root 状态
    root_states = asset.data.default_root_state[env_ids].clone()

    # 3. 采样平面偏移 (x, y) 和 朝向 (yaw)
    dx = torch.empty(len(env_ids), device=env.device).uniform_(*pose_range.get("x", (0.0, 0.0)))
    dy = torch.empty(len(env_ids), device=env.device).uniform_(*pose_range.get("y", (0.0, 0.0)))
    
    # 获取各个子环境的原点 (通常由地形管理器根据网格生成)
    origins = env.scene.env_origins[env_ids]
    
    # 机器人的全局目标 (x, y) 坐标
    target_x = origins[:, 0] + dx
    target_y = origins[:, 1] + dy

    # 4. 【核心】：动态获取当前 (x, y) 处的地形实际高度
    terrain_z = None

    # 方案 A: 如果地形管理器提供了根据 (x, y) 获取高度的函数
    if hasattr(env.scene, "terrain") and hasattr(env.scene.terrain, "get_height_at"):
        terrain_z = env.scene.terrain.get_height_at(target_x, target_y)
    
    # 方案 B: 如果环境中保存了地形高度场或采样网格
    elif hasattr(env.scene, "terrain") and hasattr(env.scene.terrain, "sample_height"):
        terrain_z = env.scene.terrain.sample_height(target_x, target_y)

    # 方案 C: 解析法兜底（如果 env_origins[:, 2] 会随难度抬升，或者通过难度 level 计算）
    if terrain_z is None:
        if hasattr(env, "terrain_levels"):
            # 如果你有每个环境的难度等级，且楼梯高度正比于难度等级
            # platform_height = base_height + level * step_height * num_steps
            # 这里如果有具体数值，可按公式直接算
            terrain_z = origins[:, 2]
        else:
            # 默认直接取对应环境网格原点的 z（如果你的 stair_terrain.py 已经把中心点高度赋给了 env_origins 的 z）
            terrain_z = origins[:, 2]

    # 5. 计算最终 Z 高度 = 地面高度 + 机器人站立高度 + pose_range 中的微调偏移
    dz = torch.empty(len(env_ids), device=env.device).uniform_(*pose_range.get("z", (0.0, 0.0)))
    target_z = terrain_z + nominal_height + dz

    # 赋回绝对位置
    root_states[:, 0] = target_x
    root_states[:, 1] = target_y
    root_states[:, 2] = target_z

    # 6. 计算 Yaw 朝向四元数
    if "yaw" in pose_range:
        yaw = torch.empty(len(env_ids), device=env.device).uniform_(*pose_range["yaw"])
        half_yaw = yaw * 0.5
        cos_yaw = torch.cos(half_yaw)
        sin_yaw = torch.sin(half_yaw)

        # 四元数格式通常为 [w, x, y, z]
        root_states[:, 3] = cos_yaw   # w
        root_states[:, 4] = 0.0       # x
        root_states[:, 5] = 0.0       # y
        root_states[:, 6] = sin_yaw   # z

    # 7. 重置线速度和角速度（如果有配置则按范围采样，否则归零）
    for i, key in enumerate(["vx", "vy", "vz", "wx", "wy", "wz"]):
        if key in velocity_range:
            root_states[:, 7 + i] = torch.empty(len(env_ids), device=env.device).uniform_(*velocity_range[key])
        else:
            root_states[:, 7 + i] = 0.0

    # 8. 写回仿真环境
    asset.write_root_state_to_sim(root_states, env_ids=env_ids)