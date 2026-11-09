from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat, wrap_to_pi

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


def _get_foot_center_xy(
    asset: Entity,
    foot_body_names: tuple[str, ...],
) -> torch.Tensor:
    """Mean XY position of the given foot bodies in the env-local frame."""
    body_names = list(asset.body_names)
    ids = [body_names.index(name) for name in foot_body_names]
    return asset.data.body_link_pos_w[:, ids, :2].mean(dim=1)


def track_target_position(
    env: ManagerBasedRlEnv,
    t_r: float = 2.0,
    command_name: str = "Goal_direction",
    foot_body_names: tuple[str, ...] | None = None,
    height_command_name: str | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    dense: bool = True,
    **kwargs,
) -> torch.Tensor:
    """Reward for tracking target position.

    XY uses the four-foot center when ``foot_body_names`` is provided; Z uses
    the base link height against ``Base_height_target`` when
    ``height_command_name`` is provided.

    When dense=True:
        1 / (1 + ||x - r_goal||^2)
    When dense=False:
        1 / (1 + ||x - r_goal||^2)  if t > T - t_r else 0
    """
    asset: Entity = env.scene[asset_cfg.name]
    goal_pos = _get_goal_pos(env, command_name=command_name, **kwargs)
    goal_xy = goal_pos[:, :2]

    if foot_body_names is not None:
        robot_xy = _get_foot_center_xy(asset, foot_body_names) - env.scene.env_origins[:, :2]
    else:
        robot_xy = asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]

    xy_error = robot_xy - goal_xy
    dist_sq = torch.sum(torch.square(xy_error), dim=-1)

    if (
        height_command_name is not None
        and height_command_name in env.command_manager.active_terms
    ):
        base_z = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
        target_z = env.command_manager.get_command(height_command_name)
        if target_z.ndim > 1:
            target_z = target_z.squeeze(-1)
        target_z = target_z.to(device=base_z.device, dtype=base_z.dtype)
        dist_sq = dist_sq + torch.square(base_z - target_z)

    if dense:
        # 全程提供连续的位置收敛奖励，让机器人每前进一步都有正向反馈
        return 1.0 / (1.0 + dist_sq)

    reward = 1.0 / (1.0 + dist_sq)

    # 条件判断: if t > T - T_r
    current_time = env.episode_length_buf * env.step_dt
    time_mask = current_time > (env.max_episode_length_s - t_r)

    return torch.where(time_mask, reward, torch.zeros_like(reward))


def standstill(
    env: ManagerBasedRlEnv,
    command_name: str = "Goal_direction",
    distance_threshold: float = 0.25,
    heading_threshold: float = 0.5,
    t_r: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    **kwargs,
) -> torch.Tensor:
    r"""Standstill penalty magnitude near the goal.

    Gate:
        ||p_t - p_t^*|| < distance_threshold
        ||q_t - q_*|| < heading_threshold   (yaw/heading error to goal, rad)
        t > T - t_r

    Value:
        2.5 * ||base_lin_vel|| + ||base_ang_vel||

    The task config gives this term a weight of -1, so the actual reward is
    negative inside the gate and zero outside.
    """
    asset: Entity = env.scene[asset_cfg.name]
    robot_pos = asset.data.root_link_pos_w - env.scene.env_origins
    goal_pos = _get_goal_pos(env, command_name=command_name, **kwargs)

    dim = goal_pos.shape[-1]
    dist = torch.linalg.norm(robot_pos[:, :dim] - goal_pos, dim=-1)

    heading_error = None
    if command_name in env.command_manager.active_terms:
        command_term = env.command_manager.get_term(command_name)
        if hasattr(command_term, "heading_error"):
            heading_error = command_term.heading_error
            if heading_error.ndim > 1:
                heading_error = heading_error.squeeze(-1)
    if heading_error is None:
        to_goal = goal_pos[:, :2] - robot_pos[:, :2]
        target_yaw = torch.atan2(to_goal[:, 1], to_goal[:, 0])
        heading_error = wrap_to_pi(target_yaw - asset.data.heading_w)
    heading_ok = heading_error.abs() < heading_threshold

    current_time = env.episode_length_buf * env.step_dt
    time_ok = current_time > (env.max_episode_length_s - t_r)
    gate = (dist < distance_threshold) & heading_ok & time_ok

    lin_speed = torch.linalg.norm(asset.data.root_link_lin_vel_w, dim=-1)
    ang_speed = torch.linalg.norm(asset.data.root_link_ang_vel_w, dim=-1)
    value = 2.5 * lin_speed + ang_speed
    return torch.where(gate, value, torch.zeros_like(value))


def base_accel_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    angular_scale: float = 0.02,
    **kwargs,
) -> torch.Tensor:
    r"""Penalize base linear/angular acceleration.

    Formula:
        ||base_lin_acc||^2 + angular_scale * ||base_ang_acc||^2
    """
    asset: Entity = env.scene[asset_cfg.name]
    adr = asset.indexing.free_joint_v_adr
    if len(adr) < 6:
        return torch.zeros(env.num_envs, device=env.device)
    acc = asset.data.data.qacc[:, adr]
    lin_sq = torch.sum(torch.square(acc[:, :3]), dim=-1)
    ang_sq = torch.sum(torch.square(acc[:, 3:6]), dim=-1)
    return lin_sq + angular_scale * ang_sq


def joint_vel_limit(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    vel_limit: float | tuple[float, ...] = 20.0,
    limit_ratio: float = 0.9,
    **kwargs,
) -> torch.Tensor:
    r"""Penalize joint velocities exceeding a soft limit.

    Formula:
        sum_i max(|dot{q}_i| - limit_ratio * dot{q}_lim_i, 0)
    """
    asset: Entity = env.scene[asset_cfg.name]
    vel = asset.data.joint_vel[:, asset_cfg.joint_ids].abs()
    if isinstance(vel_limit, (int, float)):
        limit = torch.full_like(vel, float(vel_limit))
    else:
        limit = torch.as_tensor(vel_limit, device=vel.device, dtype=vel.dtype)
        limit = limit.reshape(1, -1).expand_as(vel)
    excess = torch.clamp(vel - limit_ratio * limit, min=0.0)
    return torch.sum(excess, dim=-1)


def joint_power_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    vel_norm: float = 30.0,
    torque_norm: float = 25.0,
    **kwargs,
) -> torch.Tensor:
    r"""Penalize normalized squared joint power.

    Formula:
        sum_i ((dot{q}_i / vel_norm) * (tau_i / torque_norm))^2
    """
    asset: Entity = env.scene[asset_cfg.name]
    qdot = asset.data.joint_vel[:, asset_cfg.joint_ids]
    tau = asset.data.qfrc_actuator[:, asset_cfg.joint_ids]
    power = (qdot / vel_norm) * (tau / torque_norm)
    return torch.sum(torch.square(power), dim=-1)


class FootAccelPenalty:
    """Penalize the sum of per-foot linear acceleration magnitudes.

    Uses finite differences of foot body linear velocities:
        sum_{4 feet} ||(v_t - v_{t-1}) / dt||
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self._env = env
        self._prev_vel: torch.Tensor | None = None
        self._fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._fresh[env_ids] = True

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
        **kwargs,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        vel = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]
        if self._prev_vel is None or self._prev_vel.shape != vel.shape:
            self._prev_vel = torch.zeros_like(vel)
            self._fresh[:] = True

        acc = (vel - self._prev_vel) / env.step_dt
        acc = torch.where(self._fresh[:, None, None], torch.zeros_like(acc), acc)
        self._prev_vel.copy_(vel)
        self._fresh[:] = False
        return torch.sum(torch.linalg.norm(acc, dim=-1), dim=-1)


def normal_pose(
    env: ManagerBasedRlEnv,
    command_name: str = "Goal_direction",
    distance_threshold: float | None = 0.25,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    **kwargs,
) -> torch.Tensor:
    r"""Penalize body pitch and roll.

    Formula:
        |pitch| + |roll|

    ``distance_threshold=None`` makes the penalty active for the whole episode.
    Otherwise it only applies within ``distance_threshold`` of the goal.
    """
    asset: Entity = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_link_quat_w)
    value = torch.abs(pitch) + torch.abs(roll)

    if distance_threshold is None:
        return value

    robot_pos = asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    goal_pos = _get_goal_pos(env, command_name=command_name, **kwargs)[:, :2]
    dist = torch.linalg.norm(goal_pos - robot_pos, dim=-1)
    return torch.where(dist < distance_threshold, value, torch.zeros_like(value))


def end_position(
    env: ManagerBasedRlEnv,
    command_name: str = "Goal_direction",
    distance_threshold: float = 0.2,
    scale: float = 0.05,
    foot_body_names: tuple[str, ...] | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    **kwargs,
) -> torch.Tensor:
    r"""Dense exponential position reward near the goal.

    Formula (only when foot-center XY error < distance_threshold):
        exp(-error_xy / scale)
    """
    asset: Entity = env.scene[asset_cfg.name]
    goal_pos = _get_goal_pos(env, command_name=command_name, **kwargs)[:, :2]

    if foot_body_names is not None:
        robot_xy = (
            _get_foot_center_xy(asset, foot_body_names)
            - env.scene.env_origins[:, :2]
        )
    else:
        robot_xy = asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]

    error = torch.linalg.norm(robot_xy - goal_pos, dim=-1)
    value = torch.exp(-error / max(scale, 1e-6))
    return torch.where(
        error < distance_threshold, value, torch.zeros_like(value)
    )


def speed_limit_penalty(
    env: ManagerBasedRlEnv,
    max_speed: float = 0.5,
    **kwargs,
) -> torch.Tensor:
    r"""Penalize horizontal base speed above max_speed.

    Formula:
        x = max(||v_xy|| - max_speed, 0)
        reward = x^2
    """
    vel = env.scene["robot"].data.root_link_lin_vel_w[:, :2]
    speed = torch.linalg.norm(vel, dim=-1)
    excess = torch.clamp(speed - max_speed, min=0.0)
    return torch.square(excess)


def aggressive_motion_penalty(
    env: ManagerBasedRlEnv,
    max_speed: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    r"""Penalize horizontal base speed above max_speed.

    Formula:
        (||v_xy|| - max_speed)^2 * 1(||v_xy|| > max_speed)
    """
    vel = env.scene["robot"].data.root_link_lin_vel_w[:, :2]
    speed = torch.linalg.norm(vel, dim=-1)
    excess = torch.clamp(speed - max_speed, min=0.0)
    return torch.square(excess)


def foot_contact_pattern_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    min_contacts: int = 1,
    max_contacts: int = 3,
    goal_distance_threshold: float | None = None,
    goal_max_contacts: int = 4,
    command_name: str = "Goal_direction",
    **kwargs,
) -> torch.Tensor:
    """Return 1.0 when foot contact count is outside the allowed range.

    If ``goal_distance_threshold`` is set, allow up to ``goal_max_contacts``
    contacts near the goal (e.g. 4 feet for a stable final stance).
    """
    sensors = getattr(env.scene, "sensors", {})
    if sensor_name not in sensors:
        return torch.zeros(env.num_envs, device=env.device)
    found = sensors[sensor_name].data.found
    if found is None:
        return torch.zeros(env.num_envs, device=env.device)
    if found.ndim == 3:
        found = found.squeeze(-1)
    count = (found > 0).sum(dim=-1)

    max_allowed = torch.full_like(count, max_contacts)
    if (
        goal_distance_threshold is not None
        and command_name in env.command_manager.active_terms
    ):
        asset = env.scene["robot"]
        robot_pos = asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
        goal_pos = _get_goal_pos(env, command_name=command_name)[:, :2]
        dist = torch.linalg.norm(goal_pos - robot_pos, dim=-1)
        max_allowed = torch.where(
            dist < goal_distance_threshold,
            torch.full_like(count, goal_max_contacts),
            max_allowed,
        )

    invalid = (count < min_contacts) | (count > max_allowed)
    return invalid.float()


def goal_four_foot_contact_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    distance_threshold: float = 0.25,
    min_contacts: int = 4,
    command_name: str = "Goal_direction",
    **kwargs,
) -> torch.Tensor:
    """Return 1.0 when near the goal with fewer than min_contacts feet down."""
    sensors = getattr(env.scene, "sensors", {})
    if sensor_name not in sensors:
        return torch.zeros(env.num_envs, device=env.device)
    found = sensors[sensor_name].data.found
    if found is None:
        return torch.zeros(env.num_envs, device=env.device)
    if found.ndim == 3:
        found = found.squeeze(-1)
    count = (found > 0).sum(dim=-1)

    asset = env.scene["robot"]
    robot_pos = asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    goal_pos = _get_goal_pos(env, command_name=command_name)[:, :2]
    dist = torch.linalg.norm(goal_pos - robot_pos, dim=-1)
    return ((dist < distance_threshold) & (count < min_contacts)).float()


def collision_penalty(
    env: ManagerBasedRlEnv,
    sensor_names: tuple[str, ...] = (
        "self_collision",
        "thigh_ground_touch",
        "shank_ground_touch",
    ),
    **kwargs,
) -> torch.Tensor:
    """Return 1.0 if any non-foot collision sensor detects contact, else 0.0."""
    hit = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    sensors = getattr(env.scene, "sensors", {})
    for name in sensor_names:
        if name not in sensors:
            continue
        found = sensors[name].data.found
        if found is None:
            continue
        hit |= (found > 0).reshape(env.num_envs, -1).any(dim=-1)
    return hit.float()


def goal_forward_bonus(
    env: ManagerBasedRlEnv,
    command_name: str = "Goal_direction",
    min_forward_speed: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    **kwargs,
) -> torch.Tensor:
    """Fixed bonus while the base is moving toward the goal.

    Formula:
        1.0 if v_xy · direction_to_goal > min_forward_speed else 0.0
    """
    asset: Entity = env.scene[asset_cfg.name]
    robot_pos = asset.data.root_link_pos_w - env.scene.env_origins
    goal_pos = _get_goal_pos(env, command_name=command_name, **kwargs)

    to_goal = goal_pos[:, :2] - robot_pos[:, :2]
    dist = torch.linalg.norm(to_goal, dim=-1).clamp(min=1e-6)
    direction = to_goal / dist.unsqueeze(-1)

    vel = asset.data.root_link_lin_vel_w[:, :2]
    forward_speed = torch.sum(vel * direction, dim=-1)
    return (forward_speed > min_forward_speed).float()


def move_in_direction(
    env: ManagerBasedRlEnv,
    command_name: str = "Goal_direction",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    eps: float = 1e-6,
    goal_deadzone: float = 0.0,
    **kwargs,
) -> torch.Tensor:
    r"""Cosine similarity between base velocity and direction to goal.

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
    reward = dot / denom
    if goal_deadzone > 0.0:
        reward = torch.where(
            to_goal_norm > goal_deadzone,
            reward,
            torch.zeros_like(reward),
        )
    return reward


class MoveInDirectionFirstN:
    """Cosine move-in-direction reward active for the first N reward steps.

    The counter is per environment instance/run and is never restored from a
    checkpoint, so resuming training starts a new N-step window.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self._steps = 0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        switch_step: int = 3600,
        **kwargs,
    ) -> torch.Tensor:
        self._steps += 1
        if self._steps > switch_step:
            return torch.zeros(env.num_envs, device=env.device)
        return move_in_direction(env, **kwargs)


def stand_pose(
    env: ManagerBasedRlEnv,
    command_name: str = "Goal_direction",
    distance_threshold: float = 0.25,
    contact_distance_threshold: float = 0.15,
    contact_vel_threshold: float = 0.05,
    t_r: float = 1.0,
    height_command_name: str | None = "Base_height_target",
    nominal_height: float = 0.28,
    sensor_name: str | None = "feet_ground_contact",
    min_contacts: int = 4,
    contact_penalty: float = 1.0,
    foot_body_names: tuple[str, ...] | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    **kwargs,
) -> torch.Tensor:
    r"""Penalize base height error, body tilt, and non-four-foot stance near goal.

    Formula:
        |z - z*| + gx^2 + gy^2 + contact_penalty * 1(count < min_contacts)

    The height/tilt part is gated by ``t > T - t_r``. The four-foot requirement
    applies when the robot is within ``contact_distance_threshold`` and its
    linear speed is below ``contact_vel_threshold`` (i.e. it has stopped).
    """
    asset: Entity = env.scene[asset_cfg.name]
    robot_pos = asset.data.root_link_pos_w - env.scene.env_origins
    goal_pos = _get_goal_pos(env, command_name=command_name, **kwargs)

    dist = torch.linalg.norm(robot_pos[:, :2] - goal_pos[:, :2], dim=-1)
    near_goal = dist < distance_threshold

    current_time = env.episode_length_buf * env.step_dt
    time_ok = current_time > (env.max_episode_length_s - t_r)
    gate = near_goal & time_ok

    base_z = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    target_z = torch.full_like(base_z, nominal_height)
    if (
        height_command_name is not None
        and height_command_name in env.command_manager.active_terms
    ):
        command_z = env.command_manager.get_command(height_command_name)
        if command_z.ndim > 1:
            command_z = command_z.squeeze(-1)
        target_z = command_z.to(device=base_z.device, dtype=base_z.dtype)

    gravity = asset.data.projected_gravity_b
    tilt_sq = torch.square(gravity[:, 0]) + torch.square(gravity[:, 1])
    value = torch.where(
        gate,
        torch.abs(base_z - target_z) + tilt_sq,
        torch.zeros_like(base_z),
    )

    contact_violation = torch.zeros_like(base_z)
    sensors = getattr(env.scene, "sensors", {})
    if sensor_name is not None and sensor_name in sensors:
        found = sensors[sensor_name].data.found
        if found is not None:
            if found.ndim == 3:
                found = found.squeeze(-1)
            count = (found > 0).sum(dim=-1)
            contact_violation = (count < min_contacts).float() * contact_penalty

    lin_speed = torch.linalg.norm(asset.data.root_link_lin_vel_w, dim=-1)
    if foot_body_names is not None:
        contact_robot_xy = (
            _get_foot_center_xy(asset, foot_body_names)
            - env.scene.env_origins[:, :2]
        )
    else:
        contact_robot_xy = robot_pos[:, :2]
    contact_dist = torch.linalg.norm(
        contact_robot_xy - goal_pos[:, :2], dim=-1
    )
    contact_gate = (contact_dist < contact_distance_threshold) & (
        lin_speed <= contact_vel_threshold
    )
    return value + torch.where(
        contact_gate, contact_violation, torch.zeros_like(contact_violation)
    )


def track_target_bias(
    env: ManagerBasedRlEnv,
    command_name: str = "Goal_direction",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    distance_threshold: float | None = None,
    t_r: float | None = None,
    goal_deadzone: float = 0.0,
    **kwargs,
) -> torch.Tensor:
    r"""Reward for heading alignment with the target direction.

    Formula:
        1 / (1 + ||heading_error||^2)

    ``heading_error`` is the signed yaw difference between the robot heading
    and the direction from the robot to the goal, in radians.
    ``distance_threshold``/``t_r``: optional gate; when provided, the reward is
    nonzero only if distance to goal < threshold and t > T - t_r.
    ``goal_deadzone``: return 0 when the robot is within this distance of the
    goal.
    """
    asset: Entity = env.scene[asset_cfg.name]
    robot_pos = asset.data.root_link_pos_w - env.scene.env_origins
    goal_pos = _get_goal_pos(env, command_name=command_name, **kwargs)

    dim = goal_pos.shape[-1]
    dist = torch.linalg.norm(robot_pos[:, :dim] - goal_pos, dim=-1)

    heading_error = None
    if command_name in env.command_manager.active_terms:
        command_term = env.command_manager.get_term(command_name)
        if hasattr(command_term, "heading_error"):
            heading_error = command_term.heading_error
            if heading_error.ndim > 1:
                heading_error = heading_error.squeeze(-1)
    if heading_error is None:
        to_goal = goal_pos[:, :2] - robot_pos[:, :2]
        target_yaw = torch.atan2(to_goal[:, 1], to_goal[:, 0])
        heading_error = wrap_to_pi(target_yaw - asset.data.heading_w)

    reward = 1.0 / (1.0 + torch.square(heading_error))
    if goal_deadzone > 0.0:
        reward = torch.where(
            dist > goal_deadzone,
            reward,
            torch.zeros_like(reward),
        )
    if distance_threshold is not None:
        reward = torch.where(
            dist < distance_threshold,
            reward,
            torch.zeros_like(reward),
        )
    if t_r is not None:
        current_time = env.episode_length_buf * env.step_dt
        reward = torch.where(
            current_time > (env.max_episode_length_s - t_r),
            reward,
            torch.zeros_like(reward),
        )
    return reward


def track_base_height(
    env: ManagerBasedRlEnv,
    std: float = 0.04,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    **kwargs,
) -> torch.Tensor:
    """Reward tracking the ``Base_height_target`` command.

    Formula:
        exp(-((z_base - Base_height_target) / std)^2)

    仅使用命令项 ``Base_height_target`` 作为高度输入；命令项不存在时返回 0。
    """
    asset: Entity = env.scene[asset_cfg.name]
    base_z = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]

    if "Base_height_target" not in env.command_manager.active_terms:
        return torch.zeros_like(base_z)

    target_z = env.command_manager.get_command("Base_height_target")
    if target_z.ndim > 1:
        target_z = target_z.squeeze(-1)
    target_z = target_z.to(device=base_z.device, dtype=base_z.dtype)

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
    foot_body_names: tuple[str, ...] | None = None,
    vel_threshold: float = 0.2,
    dist_threshold: float = 0.1,
    penalty_value: float = -2.0,
    **kwargs,
) -> torch.Tensor:
    """Penalty for stalling when far away from the goal.

    Distance uses the four-foot center XY when ``foot_body_names`` is given;
    otherwise it falls back to the base link XY.
    """
    asset: Entity = env.scene[asset_cfg.name]
    goal_pos = _get_goal_pos(env, command_name=command_name, **kwargs)[:, :2]

    if foot_body_names is not None:
        robot_xy = (
            _get_foot_center_xy(asset, foot_body_names)
            - env.scene.env_origins[:, :2]
        )
    else:
        robot_xy = asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]

    dist = torch.linalg.norm(robot_xy - goal_pos, dim=-1)
    vel = asset.data.root_link_lin_vel_w[:, :2]
    vel_norm = torch.linalg.norm(vel, dim=-1)

    is_stalled = (vel_norm < vel_threshold) & (dist > dist_threshold)
    return torch.where(is_stalled, torch.full_like(dist, penalty_value), torch.zeros_like(dist))


def face_target(
    env: ManagerBasedRlEnv,
    command_name: str = "Heading_error",
    goal_command_name: str = "Goal_direction",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    dist_threshold: float = 0.5,
    min_heading_error: float = 0.0,
    **kwargs,
) -> torch.Tensor:
    """Reward for facing the target when far from goal.

    Formula:
        -||θ - θ_goal||  if ||x - r_goal|| > dist_threshold
                         and ||θ - θ_goal|| > min_heading_error
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
    gate = (dist > dist_threshold) & (heading_err > min_heading_error)
    return torch.where(gate, reward, torch.zeros_like(reward))


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
    foot_body_names: tuple[str, ...] | None = None,
    height_command_name: str | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    min_steps: int = 2,
    **kwargs,
) -> torch.Tensor:
    """到达目标点指定距离以内后判定为成功。

    XY 使用四足中心（``foot_body_names`` 提供时），Z 使用 base link 高度与
    ``Base_height_target`` 的误差。
    """
    asset: Entity = env.scene[asset_cfg.name]
    goal_pos = _get_goal_pos(env, command_name=command_name, **kwargs)
    goal_xy = goal_pos[:, :2]

    if foot_body_names is not None:
        robot_xy = _get_foot_center_xy(asset, foot_body_names) - env.scene.env_origins[:, :2]
    else:
        robot_xy = asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]

    pos_error_sq = torch.sum(torch.square(robot_xy - goal_xy), dim=-1)
    if (
        height_command_name is not None
        and height_command_name in env.command_manager.active_terms
    ):
        base_z = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
        target_z = env.command_manager.get_command(height_command_name)
        if target_z.ndim > 1:
            target_z = target_z.squeeze(-1)
        target_z = target_z.to(device=base_z.device, dtype=base_z.dtype)
        pos_error_sq = pos_error_sq + torch.square(base_z - target_z)

    dist = torch.sqrt(pos_error_sq)
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


# ==============================================================================
# 安全落脚奖励：平滑高程图 + 平地检测 + 落地高度差惩罚
# ==============================================================================

def _normalized_gaussian_blur(
    values: torch.Tensor,
    valid: torch.Tensor,
    kernel_size: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对 [B, H, W] 高程图做带有效 mask 的归一化高斯平滑。"""
    device = values.device
    coords = torch.arange(kernel_size, device=device, dtype=values.dtype)
    coords = coords - (kernel_size - 1) / 2.0
    sigma = max(kernel_size / 5.0, 1e-6)
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel = (g[:, None] * g[None, :]).view(1, 1, kernel_size, kernel_size)
    kernel = kernel / kernel.sum()

    values = values.unsqueeze(1)
    valid = valid.unsqueeze(1).to(dtype=values.dtype)
    pad = kernel_size // 2
    numerator = F.conv2d(values * valid, kernel, padding=pad)
    denominator = F.conv2d(valid, kernel, padding=pad)
    smooth = numerator.squeeze(1) / denominator.squeeze(1).clamp_min(1e-6)
    return smooth, denominator.squeeze(1)


def _bilinear_sample_grid(
    grid: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    x_min: float,
    y_min: float,
    resolution: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """在 [B, 1, H, W] 网格上按局部坐标做双线性采样。"""
    batch, _, height, width = grid.shape
    fx = (x - x_min) / resolution
    fy = (y - y_min) / resolution

    in_bounds = (
        (fx >= 0)
        & (fx <= width - 1)
        & (fy >= 0)
        & (fy <= height - 1)
    )

    x0 = torch.floor(fx).long()
    y0 = torch.floor(fy).long()
    x1 = x0 + 1
    y1 = y0 + 1
    wx = (fx - x0.to(fx.dtype)).clamp(0.0, 1.0)
    wy = (fy - y0.to(fy.dtype)).clamp(0.0, 1.0)

    x0 = x0.clamp(0, width - 1)
    x1 = x1.clamp(0, width - 1)
    y0 = y0.clamp(0, height - 1)
    y1 = y1.clamp(0, height - 1)

    batch_idx = torch.arange(batch, device=grid.device).unsqueeze(1)
    values = grid[:, 0]
    v00 = values[batch_idx, y0, x0]
    v10 = values[batch_idx, y0, x1]
    v01 = values[batch_idx, y1, x0]
    v11 = values[batch_idx, y1, x1]

    sampled = (
        v00 * (1.0 - wx) * (1.0 - wy)
        + v10 * wx * (1.0 - wy)
        + v01 * (1.0 - wx) * wy
        + v11 * wx * wy
    )
    return sampled, in_bounds
