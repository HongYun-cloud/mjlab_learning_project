from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import wrap_to_pi

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _get_goal_pos(
    env: ManagerBasedRlEnv,
    command_name: str = "Goal_direction",
    **kwargs,
) -> torch.Tensor:
    """Helper function to extract target position tensor."""
    if "target_pos" in kwargs and kwargs["target_pos"] is not None:
        goal_pos = kwargs["target_pos"]
    elif "goal_pos" in kwargs and kwargs["goal_pos"] is not None:
        goal_pos = kwargs["goal_pos"]
    elif command_name in env.command_manager.active_terms:
        command_term = env.command_manager.get_term(command_name)
        if hasattr(command_term, "target_pos"):
            goal_pos = command_term.target_pos
        elif hasattr(command_term, "goal_pos"):
            goal_pos = command_term.goal_pos
        elif hasattr(command_term, "target_position"):
            goal_pos = command_term.target_position
        else:
            goal_pos = env.command_manager.get_command(command_name)
    elif "Goal_direction" in env.command_manager.active_terms:
        command_term = env.command_manager.get_term("Goal_direction")
        if hasattr(command_term, "target_pos"):
            goal_pos = command_term.target_pos
        elif hasattr(command_term, "goal_pos"):
            goal_pos = command_term.goal_pos
        elif hasattr(command_term, "target_position"):
            goal_pos = command_term.target_position
        else:
            goal_pos = env.command_manager.get_command("Goal_direction")
    else:
        raise ValueError(
            f"Could not resolve goal position from command '{command_name}' or 'Goal_direction'."
        )

    if not isinstance(goal_pos, torch.Tensor):
        goal_pos = torch.tensor(goal_pos, device=env.device, dtype=torch.float32)
    return goal_pos


def track_target_position(
    env: ManagerBasedRlEnv,
    t_r: float = 2.0,
    command_name: str = "Goal_direction",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    dense: bool = True,
    **kwargs,
) -> torch.Tensor:
    """Reward for tracking target position.

    When dense=True (recommended), provides continuous dense reward throughout the episode:
        1 / (1 + ||x - r_goal||^2)
    When dense=False:
        (1 / T_r) * (1 / (1 + ||x - r_goal||^2))  if t > T - T_r else 0
    """
    asset: Entity = env.scene[asset_cfg.name]
    # 机器人当前在环境局部坐标系下的位置
    robot_pos = asset.data.root_link_pos_w - env.scene.env_origins
    goal_pos = _get_goal_pos(env, command_name=command_name, **kwargs)

    # 计算与目标的误差向量与距离平方 ||x - r_goal||^2
    dim = goal_pos.shape[-1]
    pos_error = robot_pos[:, :dim] - goal_pos
    dist_sq = torch.sum(torch.square(pos_error), dim=-1)

    if dense:
        # 全程提供连续的位置收敛奖励，让机器人每前进一步都有正向反馈
        return 1.0 / (1.0 + dist_sq)

    inv_t_r = 1.0 / t_r if t_r > 0 else 1.0
    reward = inv_t_r / (1.0 + dist_sq)

    # 条件判断: if t > T - T_r
    current_time = env.episode_length_buf * env.step_dt
    time_mask = current_time > (env.max_episode_length_s - t_r)

    return torch.where(time_mask, reward, torch.zeros_like(reward))


def track_target_bias(
    env: ManagerBasedRlEnv,
    command_name: str = "Goal_direction",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    eps: float = 1e-6,
    **kwargs,
) -> torch.Tensor:
    r"""Reward for tracking target bias: cosine similarity between velocity and direction to goal.

    Formula:
        (\dot{x} \cdot (r_{goal} - x)) / (||\dot{x}|| * ||r_{goal} - x||)
    """
    asset: Entity = env.scene[asset_cfg.name]
    robot_pos = asset.data.root_link_pos_w - env.scene.env_origins
    goal_pos = _get_goal_pos(env, command_name=command_name, **kwargs)

    dim = goal_pos.shape[-1]
    to_goal = goal_pos - robot_pos[:, :dim]
    vel = asset.data.root_link_lin_vel_w[:, :dim]

    to_goal_norm = torch.linalg.norm(to_goal, dim=-1)
    vel_norm = torch.linalg.norm(vel, dim=-1)

    dot = torch.sum(vel * to_goal, dim=-1)
    denom = torch.clamp(vel_norm * to_goal_norm, min=eps)
    return dot / denom


def track_base_height(
    env: ManagerBasedRlEnv,
    command_name: str = "Base_height_target",
    std: float = 0.04,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    **kwargs,
) -> torch.Tensor:
    """Reward tracking commanded base height (relative to env origin).

    Formula:
        exp(-((z_base - z_target) / std)^2)
    """
    asset: Entity = env.scene[asset_cfg.name]
    base_z = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]

    if command_name in env.command_manager.active_terms:
        target_z = env.command_manager.get_command(command_name)
        if target_z.ndim > 1:
            target_z = target_z.squeeze(-1)
    else:
        target_z = 0.28

    z_error = base_z - target_z
    return torch.exp(-torch.square(z_error / std))


def body_lateral_velocity_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    **kwargs,
) -> torch.Tensor:
    """Penalize sideways (Y) velocity in body frame to discourage crab-walking.

    Formula:
        v_y_b^2
    """
    asset: Entity = env.scene[asset_cfg.name]
    lat_vel = asset.data.root_link_lin_vel_b[:, 1]
    return torch.square(lat_vel)


def stall_penalty(
    env: ManagerBasedRlEnv,
    command_name: str = "Goal_direction",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    vel_threshold: float = 0.1,
    dist_threshold: float = 0.3,
    penalty_value: float = -2.0,
    **kwargs,
) -> torch.Tensor:
    """Penalty for stalling when far away from the goal.

    Formula:
        penalty_value (-2.0)  if ||x_dot|| < 0.1 m/s and ||x - r_goal|| > 0.5 m
        0                     otherwise
    """
    asset: Entity = env.scene[asset_cfg.name]
    robot_pos = asset.data.root_link_pos_w - env.scene.env_origins
    goal_pos = _get_goal_pos(env, command_name=command_name, **kwargs)

    dim = goal_pos.shape[-1]
    pos_error = robot_pos[:, :dim] - goal_pos
    dist = torch.linalg.norm(pos_error, dim=-1)
    # print(f"pos_error:{pos_error}")
    # print(f"dist:{dist}")

    vel = asset.data.root_link_lin_vel_w[:, :dim]
    vel_norm = torch.linalg.norm(vel, dim=-1)
    # print(f"vel:{vel}")
    # print(f"vel_norm:{vel_norm}")

    is_stalled = (vel_norm < vel_threshold) & (dist > dist_threshold)
    return torch.where(is_stalled, torch.full_like(dist, penalty_value), torch.zeros_like(dist))


def face_target(
    env: ManagerBasedRlEnv,
    command_name: str = "Heading_error",
    goal_command_name: str = "Goal_direction",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    dist_threshold: float = 0.5,
    **kwargs,
) -> torch.Tensor:
    """Reward for facing the target when far from goal.

    Formula:
        -||θ - θ_goal||  if ||x - r_goal|| > 0.5 m
        0                otherwise
    """
    asset: Entity = env.scene[asset_cfg.name]
    robot_pos = asset.data.root_link_pos_w - env.scene.env_origins
    goal_pos = _get_goal_pos(env, command_name=goal_command_name, **kwargs)

    dim = goal_pos.shape[-1]
    to_goal = goal_pos - robot_pos[:, :dim]
    dist = torch.linalg.norm(to_goal, dim=-1)

    # 计算航向偏差
    if "heading_error" in kwargs and kwargs["heading_error"] is not None:
        heading_err = torch.abs(kwargs["heading_error"])
    elif command_name in env.command_manager.active_terms:
        term = env.command_manager.get_term(command_name)
        if hasattr(term, "heading_error"):
            heading_err = torch.abs(term.heading_error)
        else:
            cmd = env.command_manager.get_command(command_name)
            if cmd.shape[-1] == 1 and cmd.abs().sum() > 0:
                heading_err = torch.abs(cmd.squeeze(-1))
            else:
                heading = asset.data.heading_w
                theta_goal = torch.atan2(to_goal[:, 1], to_goal[:, 0])
                heading_err = torch.abs(wrap_to_pi(heading - theta_goal))
    else:
        heading = asset.data.heading_w
        theta_goal = torch.atan2(to_goal[:, 1], to_goal[:, 0])
        heading_err = torch.abs(wrap_to_pi(heading - theta_goal))

    # -||θ - θ_goal||
    reward = -heading_err
    return torch.where(dist > dist_threshold, reward, torch.zeros_like(reward))


def base_angular_velocity_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    penalize_yaw: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Penalize excessive base angular velocity (roll and pitch to reduce shaking).

    Formula:
        ||ω_xy||^2  (or ||ω||^2 if penalize_yaw=True)
    """
    asset: Entity = env.scene[asset_cfg.name]
    ang_vel = asset.data.root_link_ang_vel_b
    if penalize_yaw:
        return torch.sum(torch.square(ang_vel), dim=-1)
    return torch.sum(torch.square(ang_vel[:, :2]), dim=-1)


def reach_target(
    env: ManagerBasedRlEnv,
    command_name: str = "Goal_direction",
    distance_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    min_steps: int = 2,
    **kwargs,
) -> torch.Tensor:
    """到达目标点指定距离（默认 5cm = 0.05m）以内后判定为成功，触发环境重置。

    Parameters
    ----------
    env:
        当前环境实例。
    command_name:
        目标指令名称，默认 "Goal_direction"。
    distance_threshold:
        成功判定阈值距离（米），默认 0.05（5cm）。
    asset_cfg:
        机器人实体配置。
    min_steps:
        最小步数限制，避免刚 reset 初始步误触发。

    Returns
    -------
    torch.Tensor (bool):
        达到 5cm 目标判据的环境掩码。
    """
    asset: Entity = env.scene[asset_cfg.name]
    robot_pos = asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    goal_pos = _get_goal_pos(env, command_name=command_name, **kwargs)[:, :2]

    dist = torch.linalg.norm(robot_pos - goal_pos, dim=-1)
    reached = dist <= distance_threshold

    # 忽略刚 reset 的前 min_steps 步
    reached = reached & (env.episode_length_buf >= min_steps)
    return reached


def _resolve_foot_body_ids(
    asset: Entity,
    contact_sensor,
) -> list[int]:
    """Return foot body indices ordered exactly like the contact sensor slots.

    The contact sensor expands its primary pattern into slots (e.g. FR, FL, RR,
    RL).  Element-wise products with foot velocities are only meaningful if the
    velocity tensor uses the same order.
    """
    body_names = list(asset.body_names)
    foot_body_ids: list[int] = []
    for primary_name in contact_sensor.primary_names:
        for candidate in (primary_name, f"{primary_name}_foot"):
            if candidate in body_names:
                foot_body_ids.append(body_names.index(candidate))
                break
        else:
            raise ValueError(
                f"Cannot map contact primary '{primary_name}' to a foot body. "
                f"Available bodies: {body_names}"
            )
    return foot_body_ids


def feet_slip_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str | None = None,
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    foot_names: tuple[str, ...] = ("FR_foot", "FL_foot", "RR_foot", "RL_foot"),
    **kwargs,
) -> torch.Tensor:
    """Penalize foot sliding (xy velocity while in contact), identical to official mdp.feet_slip.

    The only difference is that command_name is optional: if command is None or not in command_manager,
    it defaults to active=1.0 (always active).
    """
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor = env.scene[sensor_name]

    # 命令激活门控 (与官方一致，若无指定 command 则默认全程激活)
    if command_name is not None and command_name in env.command_manager.active_terms:
        command = env.command_manager.get_command(command_name)
        if command.shape[-1] >= 3:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
        else:
            total_command = torch.norm(command, dim=1)
        active = (total_command > command_threshold).float()
    else:
        active = 1.0

    assert contact_sensor.data.found is not None
    in_contact = (contact_sensor.data.found > 0).float()  # [B, N]

    # 获取足端 xy 线速度 [B, N, 2]，保证与 contact_sensor 槽位顺序完全一致
    if isinstance(asset_cfg.site_ids, list) and len(asset_cfg.site_ids) > 0:
        foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
    elif asset_cfg.site_names is not None and not isinstance(asset_cfg.site_ids, slice):
        foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
    else:
        foot_body_ids = _resolve_foot_body_ids(asset, contact_sensor)
        foot_vel_xy = asset.data.body_link_vel_w[:, foot_body_ids, :2]

    vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
    vel_xy_norm_sq = torch.square(vel_xy_norm)     # [B, N]

    cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
    num_in_contact = torch.sum(in_contact)
    mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(num_in_contact, min=1)
    env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
    return cost


def airborne_feet_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    max_airborne_feet: int = 2,
    mode: str = "binary",
    **kwargs,
) -> torch.Tensor:
    """Penalize having 3 or 4 legs in the air simultaneously (i.e. stance legs <= 1).

    In normal quadruped locomotion (trot, walk, crawl, pace), at least 2 legs support
    the body at any moment. When 3 or 4 legs are airborne, the robot is either
    hopping on a single leg or launching entirely into the air (jumping/pronking).

    Parameters
    ----------
    env:
        The RL environment instance.
    sensor_name:
        The name of the feet ground contact sensor.
    max_airborne_feet:
        Maximum allowed airborne feet without penalty. Defaults to 2 (meaning 3 or 4
        airborne feet will incur penalty).
    mode:
        'binary': returns 1.0 when airborne feet > max_airborne_feet (i.e. >= 3), 0.0 otherwise.
        'excess': returns excess count (airborne - max_airborne_feet), i.e. 1.0 for 3 feet,
                  2.0 for 4 feet.

    Returns
    -------
    torch.Tensor:
        Penalty value (>= 0.0) per environment.
    """
    contact_sensor = env.scene[sensor_name]
    assert contact_sensor.data.found is not None
    found = contact_sensor.data.found > 0
    if found.ndim == 3:
        found = found.squeeze(-1)

    num_contact = torch.sum(found.float(), dim=-1)
    num_airborne = 4.0 - num_contact

    if mode == "excess":
        cost = torch.clamp(num_airborne - max_airborne_feet, min=0.0)
    else:
        cost = (num_airborne > max_airborne_feet).float()

    if "log" in env.extras:
        env.extras["log"]["Metrics/airborne_excess_mean"] = (num_airborne > max_airborne_feet).float().mean()

    return cost


def diagonal_trot_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    speed_threshold: float = 0.2,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    **kwargs,
) -> torch.Tensor:
    """Strict diagonal trot reward: two diagonal pairs alternate.

    Kept for reference if explicit diagonal phase coordination is desired.
    """
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor = env.scene[sensor_name]
    assert contact_sensor.data.found is not None
    contact = contact_sensor.data.found > 0  # [B, N]
    if contact.ndim == 3:
        contact = contact.squeeze(-1)

    names = list(contact_sensor.primary_names)
    idx: dict[str, int] = {}
    for i, name in enumerate(names):
        for leg in ("FR", "FL", "RR", "RL"):
            if name in (leg, f"{leg}_foot"):
                idx[leg] = i
                break
    missing = [leg for leg in ("FR", "FL", "RR", "RL") if leg not in idx]
    if missing:
        raise ValueError(
            f"diagonal_trot_reward cannot find feet {missing} in sensor "
            f"'{sensor_name}' primaries {names}."
        )

    fr = contact[:, idx["FR"]]
    fl = contact[:, idx["FL"]]
    rr = contact[:, idx["RR"]]
    rl = contact[:, idx["RL"]]

    # 对角腿必须同相（同时在支撑或同时在摆动），两组对角腿必须反相。
    diagonal_in_phase = (fr == rl) & (fl == rr)
    pairs_out_of_phase = fr != fl
    correct = diagonal_in_phase & pairs_out_of_phase

    speed = torch.norm(asset.data.root_link_lin_vel_w[:, :2], dim=-1)
    moving = speed > speed_threshold

    reward = (correct & moving).float()
    if "log" in env.extras:
        env.extras["log"]["Metrics/trot_ratio_mean"] = correct.float().mean()
    return reward


def trot_contact_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    max_airborne_feet: int = 2,
    mode: str = "binary",
    **kwargs,
) -> torch.Tensor:
    """Backwards-compatible wrapper: redirects to airborne_feet_penalty."""
    return airborne_feet_penalty(
        env,
        sensor_name=sensor_name,
        max_airborne_feet=max_airborne_feet,
        mode=mode,
        **kwargs,
    )


def base_vertical_velocity_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    **kwargs,
) -> torch.Tensor:
    """Penalize base vertical (Z) velocity to suppress bouncing and hopping."""
    asset: Entity = env.scene[asset_cfg.name]
    vz = asset.data.root_link_lin_vel_b[:, 2]
    return torch.square(vz)

