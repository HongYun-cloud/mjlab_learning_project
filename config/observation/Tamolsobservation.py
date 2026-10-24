from __future__ import annotations

from typing import TYPE_CHECKING
import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


# ==============================================================================
# Helper
# ==============================================================================

def _get_tamols_term(env: ManagerBasedRlEnv, command_name: str | None = None):
    """Retrieve TAMOLS motion/foothold command term with fallback candidates."""
    if command_name and command_name in env.command_manager.active_terms:
        return env.command_manager.get_term(command_name)
    for name in ("Tamols_footholds", "Tamols_motion", "tamols"):
        if name in env.command_manager.active_terms:
            return env.command_manager.get_term(name)
    raise ValueError(
        f"TAMOLS command term not found. Available active terms: {list(env.command_manager.active_terms.keys())}"
    )


# ==============================================================================
# Actor (Policy) 观测项
# ==============================================================================

def base_lin_vel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    in_body_frame: bool = True,
) -> torch.Tensor:
    """机身线速度 [v_x, v_y, v_z]。Shape: (num_envs, 3)."""
    asset: Entity = env.scene[asset_cfg.name]
    if in_body_frame:
        return asset.data.root_link_lin_vel_b
    return asset.data.root_link_lin_vel_w


def base_ang_vel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    in_body_frame: bool = True,
) -> torch.Tensor:
    """机身角速度 [w_x, w_y, w_z]。Shape: (num_envs, 3)."""
    asset: Entity = env.scene[asset_cfg.name]
    if in_body_frame:
        return asset.data.root_link_ang_vel_b
    return asset.data.root_link_ang_vel_w


def base_twist(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    in_body_frame: bool = True,
) -> torch.Tensor:
    """机身 6D 空间速度 Twist [v_x, v_y, v_z, w_x, w_y, w_z]。Shape: (num_envs, 6).

    对应 DTC / TAMOLS 论文规范中的 'base twist' (dim=6)。
    """
    lin_vel = base_lin_vel(env, asset_cfg=asset_cfg, in_body_frame=in_body_frame)
    ang_vel = base_ang_vel(env, asset_cfg=asset_cfg, in_body_frame=in_body_frame)
    return torch.cat([lin_vel, ang_vel], dim=-1)


def planar_footholds(
    env: ManagerBasedRlEnv,
    command_name: str = "Tamols_footholds",
    in_body_frame: bool = True,
    planar: bool = True,
) -> torch.Tensor:
    """机身局部航向系下的 4 腿期望落脚点平面坐标 (FL_xy, FR_xy, RL_xy, RR_xy)。Shape: (num_envs, 8).

    对应 DTC 论文 Table 2: 'planar footholds' (dim=8)。
    """
    term = _get_tamols_term(env, command_name)
    return term.planar_footholds


def planned_footholds(
    env: ManagerBasedRlEnv,
    command_name: str = "Tamols_footholds",
) -> torch.Tensor:
    """TAMOLS 规划的 4 腿 3D 期望落脚点世界坐标 [FL_xyz, FR_xyz, RL_xyz, RR_xyz]。Shape: (num_envs, 12)."""
    term = _get_tamols_term(env, command_name)
    return term.command


def desired_joint_positions(
    env: ManagerBasedRlEnv,
    command_name: str = "Tamols_footholds",
) -> torch.Tensor:
    """由解析逆运动学 (Analytical IK) 计算的触地期望关节位置。Shape: (num_envs, 12).

    对应 DTC 论文 Table 2: 'desired joint positions' (dim=12)。
    """
    term = _get_tamols_term(env, command_name)
    return term.desired_joint_pos


# 别名
tamols_desired_joint_pos = desired_joint_positions
tamols_planar_footholds = planar_footholds


def desired_contact_state(
    env: ManagerBasedRlEnv,
    command_name: str = "Tamols_footholds",
) -> torch.Tensor:
    """每条腿的接触计划 (1.0 为支撑相 Stance，0.0 为摆动相 Swing)。Shape: (num_envs, 4).

    对应 DTC 论文 Table 2: 'desired contact state' (dim=4)。
    """
    term = _get_tamols_term(env, command_name)
    return term.desired_contact_state


# 别名
tamols_desired_contact_state = desired_contact_state


def time_left_in_phase(
    env: ManagerBasedRlEnv,
    command_name: str = "Tamols_footholds",
) -> torch.Tensor:
    """每条腿当前步态相的剩余时间 (秒)。Shape: (num_envs, 4).

    支撑相为距离地剩余时间，摆动相为距触地剩余时间。
    对应 DTC 论文 Table 2: 'time left in phase' (dim=4)。
    """
    term = _get_tamols_term(env, command_name)
    return term.time_left_in_phase


# 别名
tamols_time_left_in_phase = time_left_in_phase


def gait_phase(
    env: ManagerBasedRlEnv,
    command_name: str = "Tamols_footholds",
) -> torch.Tensor:
    """每条腿的归一化步态相位 [0.0, 1.0)。Shape: (num_envs, 4)."""
    term = _get_tamols_term(env, command_name)
    return term.gait_phase


tamols_gait_phase = gait_phase


def reference_twist(
    env: ManagerBasedRlEnv,
    command_name: str = "Tamols_footholds",
    goal_command_name: str = "Goal_direction",
    velocity_command_name: str = "twist",
) -> torch.Tensor:
    """参考行进速度 Twist [cmd_vx, cmd_vy, cmd_yaw_rate]。Shape: (num_envs, 3).

    对应 DTC 论文 Table 2: 'reference twist' (dim=3)。
    优先读取速度指令项 (默认 "twist") 的机体系期望速度，保证策略观测与
    TAMOLS 五项规划使用的是同一个期望速度；没有速度指令项时才依次回退到
    Goal_direction 航向误差推导和机身实测速度。
    """
    if velocity_command_name and velocity_command_name in env.command_manager.active_terms:
        vel_term = env.command_manager.get_term(velocity_command_name)
        cmd = getattr(vel_term, "vel_command_b", None)
        if cmd is None:
            cmd = getattr(vel_term, "command", None)
        if cmd is not None:
            cmd = cmd.reshape(env.num_envs, -1)
            if cmd.shape[-1] >= 3:
                return cmd[:, :3]

    if goal_command_name in env.command_manager.active_terms:
        goal_term = env.command_manager.get_term(goal_command_name)
        if hasattr(goal_term, "heading_error"):
            head_err = goal_term.heading_error.squeeze(-1)  # [num_envs]
            cmd_vx = torch.clamp(torch.cos(head_err), min=0.1) * 0.55
            cmd_vy = torch.sin(head_err) * 0.55
            cmd_wz = torch.clamp(head_err * 1.5, min=-1.5, max=1.5)
            return torch.stack([cmd_vx, cmd_vy, cmd_wz], dim=-1)

    # 默认从机身当前速度提取
    asset: Entity = env.scene[_DEFAULT_ASSET_CFG.name]
    v_b = asset.data.root_link_lin_vel_b[:, :2]
    w_b = asset.data.root_link_ang_vel_b[:, 2:3]
    return torch.cat([v_b, w_b], dim=-1)


# ==============================================================================
# Critic (Privileged) 特权参考观测项
# ==============================================================================

def desired_base_pose(
    env: ManagerBasedRlEnv,
    command_name: str = "Tamols_footholds",
) -> torch.Tensor:
    """期望基座 3D 位置与四元数 [pos_xyz, quat_wxyz]。Shape: (num_envs, 7).

    对应 DTC 论文 Table 2 特权观测中的 'desired base position' (3) + 'desired base quaternion' (4)。
    """
    term = _get_tamols_term(env, command_name)
    return torch.cat([term.desired_base_pos_w, term.desired_base_quat_w], dim=-1)


tamols_desired_base_pose = desired_base_pose


def desired_base_acc(
    env: ManagerBasedRlEnv,
    command_name: str = "Tamols_footholds",
) -> torch.Tensor:
    """期望基座 3D 线加速度与 3D 角加速度 [lin_acc_xyz, ang_acc_xyz]。Shape: (num_envs, 6).

    对应 TAMOLS 与 DTC 论文中的基座参考线/角加速度。
    """
    term = _get_tamols_term(env, command_name)
    return torch.cat([term.desired_base_lin_acc_w, term.desired_base_ang_acc_w], dim=-1)


tamols_desired_base_acc = desired_base_acc


def desired_base_twist(
    env: ManagerBasedRlEnv,
    command_name: str = "Tamols_footholds",
) -> torch.Tensor:
    """TAMOLS 规划的期望基座 6D Twist。Shape: (num_envs, 6)."""
    # 结合参考 twist 与垂直期望速度
    ref_tw = reference_twist(env, command_name=command_name)
    vx = ref_tw[:, 0:1]
    vy = ref_tw[:, 1:2]
    vz = torch.zeros_like(vx)
    wx = torch.zeros_like(vx)
    wy = torch.zeros_like(vx)
    wz = ref_tw[:, 2:3]
    return torch.cat([vx, vy, vz, wx, wy, wz], dim=-1)


__all__ = [
    "base_twist",
    "base_lin_vel",
    "base_ang_vel",
    "planar_footholds",
    "planned_footholds",
    "desired_joint_positions",
    "desired_contact_state",
    "time_left_in_phase",
    "gait_phase",
    "reference_twist",
    "desired_base_pose",
    "desired_base_acc",
    "desired_base_twist",
    # 别名
    "tamols_planar_footholds",
    "tamols_desired_joint_pos",
    "tamols_desired_contact_state",
    "tamols_time_left_in_phase",
    "tamols_gait_phase",
    "tamols_desired_base_pose",
    "tamols_desired_base_acc",
]
