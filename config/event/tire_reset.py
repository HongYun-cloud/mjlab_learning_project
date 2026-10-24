from __future__ import annotations

from typing import TYPE_CHECKING
import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def reset_root_state_from_tire(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: str = "robot",
    nominal_height: float = 0.28,
):
    """Reset robot root state on the flat spawn platform of the tire array."""
    if len(env_ids) == 0:
        return

    num_resets = len(env_ids)
    asset = env.scene[asset_cfg]

    # 1. 克隆默认状态
    root_states = asset.data.default_root_state[env_ids].clone()

    # 2. 采样平面 (x, y) 位置（位于 Y 高端的出生平坦走道）
    dx = torch.empty(num_resets, device=env.device).uniform_(*pose_range.get("x", (-0.5, 0.5)))
    dy = torch.empty(num_resets, device=env.device).uniform_(*pose_range.get("y", (2.5, 3.2)))

    origins = env.scene.env_origins[env_ids]
    target_x = origins[:, 0] + dx
    target_y = origins[:, 1] + dy

    # 3. 高度计算：出生平台位于地面基准 z=0 处
    dz = torch.empty(num_resets, device=env.device).uniform_(*pose_range.get("z", (0.0, 0.0)))
    target_z = origins[:, 2] + nominal_height + dz

    root_states[:, 0] = target_x
    root_states[:, 1] = target_y
    root_states[:, 2] = target_z

    # 4. 设置朝向 Yaw (面向轮胎阵 -Y 方向前进)
    if "yaw" in pose_range:
        yaw = torch.empty(num_resets, device=env.device).uniform_(*pose_range["yaw"])
        half_yaw = yaw * 0.5
        root_states[:, 3] = torch.cos(half_yaw)
        root_states[:, 4] = 0.0
        root_states[:, 5] = 0.0
        root_states[:, 6] = torch.sin(half_yaw)
    else:
        # 默认面朝 -Y 方向 (yaw = -1.57)
        root_states[:, 3] = 0.7071068
        root_states[:, 4] = 0.0
        root_states[:, 5] = 0.0
        root_states[:, 6] = -0.7071068

    # 5. 初始速度清零
    root_states[:, 7:13] = 0.0

    # 6. 写回仿真环境
    asset.write_root_state_to_sim(root_states, env_ids=env_ids)

    # 7. 记录出生点 XY 与朝向，以供 Curriculum 课程学习计算前进位移
    if not hasattr(env, "spawn_root_pos_w"):
        env.spawn_root_pos_w = torch.zeros((env.num_envs, 2), device=env.device)
    if not hasattr(env, "spawn_heading_w"):
        env.spawn_heading_w = torch.zeros(env.num_envs, device=env.device)

    env.spawn_root_pos_w[env_ids, 0] = target_x
    env.spawn_root_pos_w[env_ids, 1] = target_y
    if "yaw" in pose_range:
        env.spawn_heading_w[env_ids] = yaw
    else:
        env.spawn_heading_w[env_ids] = -1.57

