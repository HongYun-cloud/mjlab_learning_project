from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from myrobot_mjlab.config.event.stair_reset import sync_specified_terrain_level
from myrobot_mjlab.config.reward.Trackingreward import reach_target

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _write_root_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    yaw: torch.Tensor,
    nominal_height: float,
    asset_cfg: SceneEntityCfg,
) -> None:
    asset: Entity = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()
    origins = env.scene.env_origins[env_ids]

    root_states[:, 0] = origins[:, 0] + x
    root_states[:, 1] = origins[:, 1] + y
    root_states[:, 2] = origins[:, 2] + nominal_height

    half_yaw = yaw * 0.5
    root_states[:, 3] = torch.cos(half_yaw)
    root_states[:, 4] = 0.0
    root_states[:, 5] = 0.0
    root_states[:, 6] = torch.sin(half_yaw)
    root_states[:, 7:13] = 0.0

    asset.write_root_state_to_sim(root_states, env_ids=env_ids)

    if not hasattr(env, "spawn_root_pos_w"):
        env.spawn_root_pos_w = torch.zeros(
            (env.num_envs, 2), device=env.device
        )
    if not hasattr(env, "spawn_heading_w"):
        env.spawn_heading_w = torch.zeros(env.num_envs, device=env.device)
    env.spawn_root_pos_w[env_ids] = torch.stack(
        [origins[:, 0] + x, origins[:, 1] + y], dim=-1
    )
    env.spawn_heading_w[env_ids] = yaw


def reset_root_state_at_stair_base_sides(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    side_distance: float = 5.0,
    lateral_jitter: float = 1.0,
    nominal_height: float = 0.28,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Spawn on the flat base outside the up pyramid, random E/W/N/S side."""
    if len(env_ids) == 0:
        return

    sync_specified_terrain_level(env, env_ids)

    n = len(env_ids)
    side = torch.randint(0, 4, (n,), device=env.device)
    lateral = (
        torch.rand((n,), device=env.device) * 2.0 - 1.0
    ) * lateral_jitter

    x = torch.zeros((n,), device=env.device)
    y = torch.zeros((n,), device=env.device)
    yaw = torch.zeros((n,), device=env.device)

    east = side == 0
    west = side == 1
    north = side == 2
    south = side == 3

    x[east] = side_distance
    y[east] = lateral[east]
    yaw[east] = math.pi

    x[west] = -side_distance
    y[west] = lateral[west]
    yaw[west] = 0.0

    y[north] = side_distance
    x[north] = lateral[north]
    yaw[north] = -math.pi / 2.0

    y[south] = -side_distance
    x[south] = lateral[south]
    yaw[south] = math.pi / 2.0

    _write_root_state(
        env,
        env_ids,
        x=x,
        y=y,
        yaw=yaw,
        nominal_height=nominal_height,
        asset_cfg=asset_cfg,
    )


def reset_root_state_west_of_stair(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    x_offset: float = -7.0,
    y_range: tuple[float, float] = (-0.5, 0.5),
    yaw: float = 0.0,
    nominal_height: float = 0.28,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Spawn west of the up pyramid, facing east."""
    if len(env_ids) == 0:
        return

    sync_specified_terrain_level(env, env_ids)

    n = len(env_ids)
    x = torch.full((n,), x_offset, device=env.device)
    y = torch.empty(n, device=env.device).uniform_(*y_range)
    yaw_t = torch.full((n,), yaw, device=env.device)
    _write_root_state(
        env,
        env_ids,
        x=x,
        y=y,
        yaw=yaw_t,
        nominal_height=nominal_height,
        asset_cfg=asset_cfg,
    )


def _get_goal_command(env: ManagerBasedRlEnv, command_name: str):
    if command_name not in env.command_manager.active_terms:
        return None
    return env.command_manager.get_term(command_name)


# Combined stair 5 个 waypoint 的相对高度（以单段金字塔总高度 H 为单位）：
# G1=up 顶中心(+H), G2=up 东底(0), G3=down 西顶(0),
# G4=down 底中心(-H), G5=down 东顶(0)。
WAYPOINT_HEIGHT_FRACTIONS: tuple[float, ...] = (1.0, 0.0, 0.0, -1.0, 0.0)


def _stair_total_height(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
) -> torch.Tensor:
    """按当前地形难度计算单段金字塔的总高度 H。"""
    terrain = getattr(env.scene, "terrain", None)
    generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    if generator is None:
        return torch.zeros(len(env_ids), device=env.device)

    sub_cfg = next(iter(generator.sub_terrains.values()))
    size = getattr(sub_cfg, "up_size", sub_cfg.size)
    num_steps_x = int(
        (size[0] - 2 * sub_cfg.border_width - sub_cfg.platform_width)
        / (2 * sub_cfg.step_width)
    )
    num_steps_y = int(
        (size[1] - 2 * sub_cfg.border_width - sub_cfg.platform_width)
        / (2 * sub_cfg.step_width)
    )
    num_steps = max(0, min(num_steps_x, num_steps_y))

    levels = terrain.terrain_levels[env_ids].float()
    num_rows = max(int(terrain.terrain_origins.shape[0]), 1)
    t = levels / max(num_rows - 1, 1)
    low, high = generator.difficulty_range
    difficulty = low + t * (high - low)

    step_low, step_high = sub_cfg.step_height_range
    step_height = step_low + difficulty * (step_high - step_low)
    return (num_steps + 1) * step_height


def _set_waypoint_height_target(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    waypoint_index: torch.Tensor | int,
    height_command_name: str | None,
) -> None:
    """将 Base_height_target 固定到当前 waypoint 的地形高度 + nominal。"""
    if height_command_name is None:
        return
    if height_command_name not in env.command_manager.active_terms:
        return
    command = env.command_manager.get_term(height_command_name)
    if not hasattr(command, "set_ground_z_override"):
        return

    total_height = _stair_total_height(env, env_ids)
    fractions = torch.tensor(
        WAYPOINT_HEIGHT_FRACTIONS, device=env.device, dtype=total_height.dtype
    )
    index = torch.as_tensor(waypoint_index, device=env.device)
    if index.ndim == 0:
        index = index.expand(len(env_ids))
    command.set_ground_z_override(
        env_ids, fractions[index] * total_height
    )
    command._update_command(env_ids)


def _ensure_target_positions(
    command, num_envs: int, device: torch.device
) -> None:
    if not hasattr(command, "target_positions"):
        command.target_positions = torch.zeros(
            (num_envs, 2), device=device, dtype=torch.float32
        )


def reset_stair_waypoints(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str = "Goal_direction",
    waypoints: tuple[tuple[float, float], ...] = (
        (-2.0, 0.0),
        (1.3, 0.0),
        (1.7, 0.0),
        (5.0, 0.0),
        (8.5, 0.0),
    ),
    height_command_name: str | None = "Base_height_target",
) -> None:
    """Initialize the 5-waypoint sequence for the combined stair task."""
    if len(env_ids) == 0:
        return

    waypoint_tensor = torch.tensor(
        waypoints, device=env.device, dtype=torch.float32
    )
    env.stair_waypoints = waypoint_tensor
    if not hasattr(env, "stair_waypoint_index"):
        env.stair_waypoint_index = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
    if not hasattr(env, "stair_waypoint_completed"):
        env.stair_waypoint_completed = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    if not hasattr(env, "stair_waypoint_start_time"):
        env.stair_waypoint_start_time = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.float32
        )
    env.stair_waypoint_index[env_ids] = 0
    env.stair_waypoint_completed[env_ids] = False
    env.stair_waypoint_start_time[env_ids] = 0.0

    command = _get_goal_command(env, command_name)
    if command is None:
        return
    _ensure_target_positions(command, env.num_envs, env.device)
    target = waypoint_tensor[0]
    command.target_positions[env_ids] = target
    command.target_pos[env_ids] = target
    _set_waypoint_height_target(env, env_ids, 0, height_command_name)
    command._update_command(env_ids)


def reset_stair_single_target(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str = "Goal_direction",
    target: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Set a fixed target (used by stair_up: top center)."""
    if len(env_ids) == 0:
        return
    command = _get_goal_command(env, command_name)
    if command is None:
        return
    _ensure_target_positions(command, env.num_envs, env.device)
    target_tensor = torch.tensor(
        target, device=env.device, dtype=torch.float32
    )
    command.target_positions[env_ids] = target_tensor
    command.target_pos[env_ids] = target_tensor
    command._update_command(env_ids)


def update_stair_waypoints(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None = None,
    command_name: str = "Goal_direction",
    distance_threshold: float = 0.15,
    foot_body_names: tuple[str, ...] = (
        "FR_foot",
        "FL_foot",
        "RR_foot",
        "RL_foot",
    ),
    height_command_name: str | None = "Base_height_target",
) -> None:
    """Advance the waypoint index when the current target is reached."""
    if not hasattr(env, "stair_waypoint_index"):
        return
    command = _get_goal_command(env, command_name)
    if command is None:
        return

    reached = reach_target(
        env,
        command_name=command_name,
        distance_threshold=distance_threshold,
        foot_body_names=foot_body_names,
        height_command_name=height_command_name,
    )
    index = env.stair_waypoint_index
    completed = env.stair_waypoint_completed

    advance = reached & (index < 4) & ~completed
    finish = reached & (index >= 4) & ~completed

    if advance.any():
        ids = advance.nonzero(as_tuple=False).squeeze(-1)
        new_index = index[ids] + 1
        index[ids] = new_index
        if hasattr(env, "stair_waypoint_start_time"):
            env.stair_waypoint_start_time[ids] = (
                env.episode_length_buf[ids].to(torch.float32) * env.step_dt
            )
        next_targets = env.stair_waypoints[new_index]
        command.target_positions[ids] = next_targets
        command.target_pos[ids] = next_targets
        _set_waypoint_height_target(env, ids, new_index, height_command_name)
        command._update_command(ids)

    if finish.any():
        completed[finish] = True


def stair_waypoint_success(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Metric: 1 only after all waypoints of the combined stair are reached."""
    if not hasattr(env, "stair_waypoint_completed"):
        return torch.zeros(env.num_envs, device=env.device)
    return env.stair_waypoint_completed.float()


__all__ = [
    "reset_root_state_at_stair_base_sides",
    "reset_root_state_west_of_stair",
    "reset_stair_waypoints",
    "reset_stair_single_target",
    "update_stair_waypoints",
    "stair_waypoint_success",
]
