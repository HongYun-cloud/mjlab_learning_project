"""TAMOLS & DTC (Deep Tracking Control) Motion Tracking Reward Terms.

Based on:
1. "TAMOLS: Terrain-Aware Motion Optimization for Legged Systems" (Paper)
2. "Deep Tracking Control: Agile Legged Locomotion via Trajectory Optimization and Reinforcement Learning"
   Table 3: Reward functions & parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import math
import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


# ==============================================================================
# Helper functions
# ==============================================================================

def _get_tamols_term(env: ManagerBasedRlEnv, command_name: str | None = None):
    """Retrieve TAMOLS motion command term with fallback candidates."""
    if command_name and command_name in env.command_manager.active_terms:
        return env.command_manager.get_term(command_name)
    for name in ("Tamols_footholds", "Tamols_motion", "tamols"):
        if name in env.command_manager.active_terms:
            return env.command_manager.get_term(name)
    raise ValueError(
        f"TAMOLS command term not found in command_manager. Active terms: {list(env.command_manager.active_terms.keys())}"
    )


def _resolve_foot_body_indices(
    asset: Entity,
    foot_names: tuple[str, ...] = ("FL_foot", "FR_foot", "RL_foot", "RR_foot"),
) -> list[int]:
    """Resolve indices of feet bodies matching the given order [FL, FR, RL, RR]."""
    body_names = list(asset.body_names)
    indices = []
    for name in foot_names:
        for candidate in (name, name.replace("_foot", ""), f"{name}_link"):
            if candidate in body_names:
                indices.append(body_names.index(candidate))
                break
        else:
            raise ValueError(f"Cannot resolve foot body '{name}' in robot bodies: {body_names}")
    return indices


def _resolve_contact_sensor_order(
    contact_sensor,
    target_legs: tuple[str, ...] = ("FL", "FR", "RL", "RR"),
) -> list[int]:
    """Map contact sensor slots to the target leg order [FL, FR, RL, RR]."""
    primaries = list(contact_sensor.primary_names)
    slot_indices = []
    for leg in target_legs:
        for idx, prim in enumerate(primaries):
            if prim == leg or prim.startswith(f"{leg}_") or prim == f"{leg}_foot":
                slot_indices.append(idx)
                break
        else:
            # Fallback to direct indexing if names don't match pattern
            slot_indices = list(range(min(len(primaries), len(target_legs))))
            break
    return slot_indices


def _as_tensor(x, ref: torch.Tensor) -> torch.Tensor:
    """Safely convert x to a torch.Tensor matching the device and dtype of ref."""
    if isinstance(x, torch.Tensor):
        return x.to(device=ref.device, dtype=ref.dtype)
    return torch.as_tensor(x, device=ref.device, dtype=ref.dtype)


# ==============================================================================
# 1. Foothold Tracking Reward (DTC Table 3: footholds, weight = 6.0)
# ==============================================================================

def track_tamols_footholds(
    env: ManagerBasedRlEnv,
    command_name: str = "Tamols_footholds",
    sensor_name: str = "feet_ground_contact",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    eps: float = 1e-5,
    std: float = 0.05,
    reach_threshold: float = 0.05,
    once_per_touchdown: bool = True,
    mode: str = "exp",
    only_on_contact: bool = True,
    **kwargs,
) -> torch.Tensor:
    r"""Reward for tracking TAMOLS optimal 3D footholds (FL, FR, RL, RR).

    当 once_per_touchdown=True 时，只有当足端在触地时到达参考点（误差在 reach_threshold 以内，默认 5cm）
    才会给一次分，到达后在当前整个支撑相期间不再重复给分，杜绝原地不迈步持续吃分刷分的投机行为。
    """
    term = _get_tamols_term(env, command_name)
    asset: Entity = env.scene[asset_cfg.name]

    # [num_envs, 4, 3] in FL, FR, RL, RR order
    foot_body_ids = _resolve_foot_body_indices(asset)
    p_act = asset.data.body_link_pos_w[:, foot_body_ids, :]
    # landing_footholds: 每条腿"正在飞向、触地时应落到"的目标 (支撑腿在触地瞬间被冻结)。
    # optimal_footholds 在触地瞬间会被重规划为"下一次触地"(整个步幅之前方)，
    # 直接用它会导致触地时永远对不上、跟踪奖励失效。
    p_des_src = getattr(term, "landing_footholds", None)
    if not isinstance(p_des_src, torch.Tensor):
        p_des_src = term.optimal_footholds
    p_des = _as_tensor(p_des_src, ref=p_act)

    # 距离计算: [num_envs, 4]
    dist_sq = torch.sum(torch.square(p_des - p_act), dim=-1)
    dist = torch.sqrt(torch.clamp(dist_sq, min=1e-8))

    if mode == "log":
        r_per_foot = -torch.log(dist_sq + eps) / (-math.log(eps))
    else:
        r_per_foot = torch.exp(-dist_sq / (std**2))

    # 判断接触传感器状态
    act_contact = None
    if only_on_contact:
        try:
            contact_sensor = env.scene[sensor_name]
            if contact_sensor is not None and hasattr(contact_sensor, "data") and contact_sensor.data.found is not None:
                found = contact_sensor.data.found > 0
                if found.ndim == 3:
                    found = found.squeeze(-1)
                sensor_order = _resolve_contact_sensor_order(contact_sensor)
                act_contact = found[:, sensor_order].float()
        except Exception:
            pass

    des_contact = _as_tensor(term.desired_contact_state, ref=dist).float()

    if once_per_touchdown and hasattr(term, "foothold_awarded"):
        # 到达判据：实际距离在阈值内 (<= 5cm)，且处于触地阶段
        is_reached = dist <= reach_threshold
        if act_contact is not None:
            is_reached = is_reached & (act_contact > 0.5)
        else:
            is_reached = is_reached & (des_contact > 0.5)

        # 仅未曾领过该落脚点分数的环境能够获取单次到达分
        newly_reached = is_reached & (~term.foothold_awarded)
        # 记录已领分状态
        term.foothold_awarded = term.foothold_awarded | is_reached

        reward_per_foot = torch.where(newly_reached, r_per_foot, torch.zeros_like(r_per_foot))
        return reward_per_foot.sum(dim=-1)

    if only_on_contact and act_contact is not None:
        mask = (des_contact > 0.5) & (act_contact > 0.5)
        swing_reward = r_per_foot * (1.0 - des_contact) * 0.2
        stance_reward = r_per_foot * mask.float()
        return (stance_reward + swing_reward).mean(dim=-1)

    return r_per_foot.mean(dim=-1)


# ==============================================================================
# 2. Base Pose Tracking Rewards (DTC Table 3: base position, base rotation)
# ==============================================================================

def track_tamols_base_position(
    env: ManagerBasedRlEnv,
    command_name: str = "Tamols_footholds",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    reach_threshold: float = 0.05,
    once_per_reach: bool = True,
    sigma: float = 1200.0,
    track_z_only: bool = False,
    std: float | None = 0.05,
    **kwargs,
) -> torch.Tensor:
    r"""Reward for tracking TAMOLS desired base 3D position.

    当 once_per_reach=True 时，仅在基座到达参考点（误差在 reach_threshold 以内，默认 5cm）
    时给一次分，到达后在该步周期内不再重复给分，直至下一阶段重新规划产生新的参考点。
    """
    term = _get_tamols_term(env, command_name)
    asset: Entity = env.scene[asset_cfg.name]

    act_pos = asset.data.root_link_pos_w
    des_pos = _as_tensor(term.desired_base_pos_w, ref=act_pos)

    if track_z_only:
        dist_sq = torch.square(des_pos[:, 2] - act_pos[:, 2])
        dist = torch.abs(des_pos[:, 2] - act_pos[:, 2])
    else:
        dist_sq = torch.sum(torch.square(des_pos - act_pos), dim=-1)
        dist = torch.sqrt(torch.clamp(dist_sq, min=1e-8))

    if std is not None:
        r_base = torch.exp(-dist_sq / (std**2))
    else:
        r_base = torch.exp(-sigma * dist_sq)

    if once_per_reach and hasattr(term, "base_pos_awarded"):
        is_reached = dist <= reach_threshold
        newly_reached = is_reached & (~term.base_pos_awarded)
        term.base_pos_awarded = term.base_pos_awarded | is_reached

        reward = torch.where(newly_reached, r_base, torch.zeros_like(r_base))
        return reward

    return r_base


def track_tamols_base_orientation(
    env: ManagerBasedRlEnv,
    command_name: str = "Tamols_footholds",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    sigma: float = 90.0,
    **kwargs,
) -> torch.Tensor:
    r"""Reward for tracking TAMOLS terrain-adaptive base orientation (pitch/roll/yaw).

    Directly fixes bad posture on stairs by motivating the torso to tilt
    parallel to the stair slope.
    DTC Table 3: r = exp(-sigma * ||q^*_B \ominus q_B||^2), sigma = 90.
    """
    term = _get_tamols_term(env, command_name)
    asset: Entity = env.scene[asset_cfg.name]

    act_quat = asset.data.root_link_quat_w  # [B, 4] (w, x, y, z)
    des_quat = _as_tensor(term.desired_base_quat_w, ref=act_quat)  # [B, 4] (w, x, y, z)

    # Quaternion dot product
    dot = torch.sum(des_quat * act_quat, dim=-1)
    quat_err_sq = 2.0 * (1.0 - torch.clamp(torch.abs(dot), max=1.0))

    return torch.exp(-sigma * quat_err_sq)


# ==============================================================================
# 3. Base Velocity Tracking Rewards (DTC Table 3: linear & angular velocity)
# ==============================================================================

def track_tamols_base_linear_velocity(
    env: ManagerBasedRlEnv,
    command_name: str = "Tamols_footholds",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    sigma: float = 10.0,
    std: float | None = 0.25,
    **kwargs,
) -> torch.Tensor:
    r"""Reward for tracking TAMOLS planned base linear velocity.

    Prevents the robot from sprinting too fast on stairs.
    DTC Table 3: r = exp(-sigma * ||v^*_B - v_B||^2), sigma = 10.
    """
    term = _get_tamols_term(env, command_name)
    asset: Entity = env.scene[asset_cfg.name]

    act_vel = asset.data.root_link_lin_vel_w
    des_vel = _as_tensor(term.desired_base_lin_vel_w, ref=act_vel)

    vel_err_sq = torch.sum(torch.square(des_vel - act_vel), dim=-1)

    if std is not None:
        return torch.exp(-vel_err_sq / (std**2))
    return torch.exp(-sigma * vel_err_sq)


def track_tamols_base_angular_velocity(
    env: ManagerBasedRlEnv,
    command_name: str = "Tamols_footholds",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    sigma: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    r"""Reward for tracking TAMOLS planned base angular velocity.

    DTC Table 3: r = exp(-sigma * ||\omega^*_B - \omega_B||^2), sigma = 1.
    """
    term = _get_tamols_term(env, command_name)
    asset: Entity = env.scene[asset_cfg.name]

    act_ang_vel = asset.data.root_link_ang_vel_w
    des_ang_vel = _as_tensor(term.desired_base_ang_vel_w, ref=act_ang_vel)

    ang_vel_err_sq = torch.sum(torch.square(des_ang_vel - act_ang_vel), dim=-1)
    return torch.exp(-sigma * ang_vel_err_sq)


# ==============================================================================
# 4. Base Acceleration Tracking Rewards (DTC Table 3: linear & angular acc)
# ==============================================================================

def track_tamols_base_linear_acceleration(
    env: ManagerBasedRlEnv,
    command_name: str = "Tamols_footholds",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    sigma: float = 0.05,
    **kwargs,
) -> torch.Tensor:
    r"""Reward for tracking TAMOLS planned base linear acceleration.

    DTC Table 3: r = exp(-sigma * ||a^*_B - a_B||^2), sigma = 0.05.
    """
    term = _get_tamols_term(env, command_name)
    asset: Entity = env.scene[asset_cfg.name]

    cur_vel = asset.data.root_link_lin_vel_w
    des_acc = _as_tensor(term.desired_base_lin_acc_w, ref=cur_vel)
    # Approximate actual acceleration using finite difference of velocities
    prev_vel_key = f"{asset_cfg.name}_prev_lin_vel"
    if prev_vel_key in env.extras:
        prev_vel = env.extras[prev_vel_key]
        act_acc = (cur_vel - prev_vel) / max(env.step_dt, 1e-4)
    else:
        act_acc = torch.zeros_like(cur_vel)
    env.extras[prev_vel_key] = cur_vel.clone()

    acc_err_sq = torch.sum(torch.square(des_acc - act_acc), dim=-1)
    return torch.exp(-sigma * acc_err_sq)


def track_tamols_base_angular_acceleration(
    env: ManagerBasedRlEnv,
    command_name: str = "Tamols_footholds",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    sigma: float = 0.005,
    **kwargs,
) -> torch.Tensor:
    r"""Reward for tracking TAMOLS planned base angular acceleration.

    DTC Table 3: r = exp(-sigma * ||\dot{\omega}^*_B - \dot{\omega}_B||^2), sigma = 0.005.
    """
    term = _get_tamols_term(env, command_name)
    asset: Entity = env.scene[asset_cfg.name]

    cur_ang_vel = asset.data.root_link_ang_vel_w
    des_ang_acc = _as_tensor(term.desired_base_ang_acc_w, ref=cur_ang_vel)
    prev_ang_vel_key = f"{asset_cfg.name}_prev_ang_vel"
    if prev_ang_vel_key in env.extras:
        prev_ang_vel = env.extras[prev_ang_vel_key]
        act_ang_acc = (cur_ang_vel - prev_ang_vel) / max(env.step_dt, 1e-4)
    else:
        act_ang_acc = torch.zeros_like(cur_ang_vel)
    env.extras[prev_ang_vel_key] = cur_ang_vel.clone()

    ang_acc_err_sq = torch.sum(torch.square(des_ang_acc - act_ang_acc), dim=-1)
    return torch.exp(-sigma * ang_acc_err_sq)


# ==============================================================================
# 5. Touchdown Joint Position Guidance (Analytical IK)
# ==============================================================================

def track_tamols_desired_joint_positions(
    env: ManagerBasedRlEnv,
    command_name: str = "Tamols_footholds",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.35,
    only_near_touchdown: bool = True,
    touchdown_time_threshold: float = 0.15,
    **kwargs,
) -> torch.Tensor:
    r"""Reward for aligning joint angles towards the analytical IK touchdown angles.

    Provides inductive kinematic guidance to ensure proper foot clearance and
    stair landing posture.
    """
    term = _get_tamols_term(env, command_name)
    asset: Entity = env.scene[asset_cfg.name]

    act_q = asset.data.joint_pos    # [num_envs, 12]
    des_q = _as_tensor(term.desired_joint_pos, ref=act_q)  # [num_envs, 12]

    # Error per leg: [num_envs, 4]
    leg_err_sq = torch.sum(
        torch.square((des_q - act_q).view(-1, 4, 3)),
        dim=-1,
    )
    r_per_leg = torch.exp(-leg_err_sq / (std**2))

    if only_near_touchdown:
        # Active during late swing phase (time left in swing < threshold) or early stance
        time_left = _as_tensor(term.time_left_in_phase, ref=r_per_leg)
        des_contact = _as_tensor(term.desired_contact_state, ref=r_per_leg)
        is_swing = des_contact < 0.5
        near_touchdown = is_swing & (time_left <= touchdown_time_threshold)
        is_stance = ~is_swing
        mask = (near_touchdown | is_stance).float()
        return (r_per_leg * mask).sum(dim=-1) / 4.0

    return r_per_leg.mean(dim=-1)


# ==============================================================================
# 6. Contact Schedule Tracking Reward
# ==============================================================================

def track_tamols_contact_schedule(
    env: ManagerBasedRlEnv,
    command_name: str = "Tamols_footholds",
    sensor_name: str = "feet_ground_contact",
    **kwargs,
) -> torch.Tensor:
    r"""Reward for matching the planned gait contact schedule (stance vs swing)."""
    term = _get_tamols_term(env, command_name)
    contact_sensor = env.scene[sensor_name]

    assert contact_sensor.data.found is not None
    found = contact_sensor.data.found > 0
    if found.ndim == 3:
        found = found.squeeze(-1)

    sensor_order = _resolve_contact_sensor_order(contact_sensor)
    act_contact = found[:, sensor_order].float()
    des_contact = _as_tensor(term.desired_contact_state, ref=act_contact).float()

    # Exact match ratio in [0, 1]
    match = 1.0 - torch.abs(act_contact - des_contact)
    return match.mean(dim=-1)


# ==============================================================================
# 7. DTC Regularization Penalties (Foot power & Joint power)
# ==============================================================================

def tamols_foot_power_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    **kwargs,
) -> torch.Tensor:
    r"""Penalize power at feet: -sum_i |v_i^T * f_i|.

    Discourages foot-scuffing, stubbing toes, and impact collisions.
    DTC Table 3: weight = -0.02.
    """
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor = env.scene[sensor_name]

    foot_body_ids = _resolve_foot_body_indices(asset)
    foot_vel = asset.data.body_link_vel_w[:, foot_body_ids, :3]  # [num_envs, 4, 3]

    if contact_sensor.data.force is not None:
        sensor_order = _resolve_contact_sensor_order(contact_sensor)
        # Force: [num_envs, num_slots, 3]
        foot_forces = contact_sensor.data.force[:, sensor_order, :]
        power = torch.abs(torch.sum(foot_vel * foot_forces, dim=-1))  # [num_envs, 4]
        return torch.sum(power, dim=-1)

    # Fallback if forces not available: penalize foot velocity while in contact
    found = contact_sensor.data.found > 0
    if found.ndim == 3:
        found = found.squeeze(-1)
    sensor_order = _resolve_contact_sensor_order(contact_sensor)
    in_contact = found[:, sensor_order].float()
    vel_norm = torch.norm(foot_vel, dim=-1)
    return torch.sum(vel_norm * in_contact, dim=-1)


def tamols_joint_power_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    **kwargs,
) -> torch.Tensor:
    r"""Penalize squared joint power: -sum_j (q_dot_j * tau_j)^2.

    Prevents violent twitching and excessive motor power.
    DTC Table 3: weight = -0.025.
    """
    asset: Entity = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel

    joint_torques = None
    # 1. Prefer mjlab standard qfrc_actuator (actuator forces projected to joint space)
    if hasattr(asset.data, "qfrc_actuator"):
        try:
            val = asset.data.qfrc_actuator
            if isinstance(val, torch.Tensor):
                joint_torques = val
        except (AttributeError, NotImplementedError):
            pass

    # 2. Try actuator_force
    if joint_torques is None and hasattr(asset.data, "actuator_force"):
        try:
            val = asset.data.actuator_force
            if isinstance(val, torch.Tensor):
                joint_torques = val
        except (AttributeError, NotImplementedError):
            pass

    # 3. Fallback to joint_torques if provided (e.g. in test mocks)
    if joint_torques is None:
        try:
            val = asset.data.joint_torques
            if isinstance(val, torch.Tensor):
                joint_torques = val
        except (AttributeError, NotImplementedError):
            pass

    if joint_torques is None:
        joint_torques = torch.zeros_like(joint_vel)

    # 归一化关节功率 (额定角速度 30rad/s, 额定力矩 25Nm)，避免量纲过大压制足端迈步
    power_norm = (joint_vel / 30.0) * (joint_torques / 25.0)
    power_sq = torch.square(power_norm)
    return torch.sum(power_sq, dim=-1)
