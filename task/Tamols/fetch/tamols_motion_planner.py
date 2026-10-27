from __future__ import annotations

import math
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn

from .tamols_foothold import TAMOLSFootholdPlanner, TorchTAMOLSFootholdPlanner, TAMOLSOptimizerCfg
from .elevation_history_map import ElevationHistoryMap
from .go2_kinematics import (
    go2_leg_analytical_ik_np,
    go2_analytical_ik_batch_torch,
    transform_footholds_to_hip_frame_torch,
    GO2_HIP_OFFSETS_TRUNK,
    GO2_L_HIP_SIGNS,
)


@dataclass
class TAMOLSMotionPlannerCfg:
    """Configuration for TAMOLS full motion planner."""
    optimizer_cfg: TAMOLSOptimizerCfg = field(default_factory=TAMOLSOptimizerCfg)

    # Reference acceleration tracking gains
    acc_filter_omega_lin: float = 15.0
    """Natural frequency for base linear position tracking filter (rad/s)."""

    acc_filter_zeta_lin: float = 1.0
    """Damping ratio for base linear position tracking filter."""

    acc_filter_omega_ang: float = 20.0
    """Natural frequency for base orientation tracking filter (rad/s)."""

    acc_filter_zeta_ang: float = 1.0
    """Damping ratio for base orientation tracking filter."""

    max_pitch: float = math.radians(35.0)
    """Maximum allowed base pitch adaptation in radians."""

    max_roll: float = math.radians(25.0)
    """Maximum allowed base roll adaptation in radians."""

    max_lin_acc: float = 8.0
    """Max linear acceleration clamp in m/s^2."""

    max_ang_acc: float = 15.0
    """Max angular acceleration clamp in rad/s^2."""


def euler_zyx_to_quat_np(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert ZYX Euler angles (roll, pitch, yaw) to quaternion [w, x, y, z] (NumPy)."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z], dtype=np.float32)


def euler_zyx_to_rotation_matrix_np(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Construct 3x3 rotation matrix R_w_b from ZYX Euler angles."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr               ],
    ], dtype=np.float32)
    return R


class TAMOLSMotionPlanner:
    """Full TAMOLS Motion Planner generating Base Pose, Accelerations, Footholds, and Touchdown Joint Angles.
    
    Adheres strictly to TAMOLS (Jenelten et al., 2022) & DTC (Science Robotics 2024):
    1. Footholds: Computes 3D optimal footholds on local elevation map avoiding edges & gaps;
    2. Base Pose: Fits terrain plane from footholds to compute adaptive height, pitch, and roll;
    3. Accelerations: Calculates reference linear acceleration and angular acceleration;
    4. Touchdown Joint Positions: Computes 12D joint angles via Analytical Inverse Kinematics.
    """

    def __init__(self, cfg: TAMOLSMotionPlannerCfg | None = None):
        self.cfg = cfg or TAMOLSMotionPlannerCfg()
        self.foothold_planner = TAMOLSFootholdPlanner(self.cfg.optimizer_cfg)

    def plan_motion(
        self,
        nominal_footholds_w: np.ndarray,
        base_pos_w: np.ndarray,
        base_yaw: float,
        base_vel_w: np.ndarray,
        cmd_vel_b: np.ndarray,
        base_ang_vel_w: np.ndarray,
        cmd_yaw_rate: float,
        elevation_map: ElevationHistoryMap,
        lead_time: np.ndarray | float | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute full TAMOLS motion references for a single robot.

        Args:
            nominal_footholds_w: [4, 3] Nominal landing positions in world frame.
            base_pos_w: [3] Measured base position in world frame.
            base_yaw: Measured base heading in radians.
            base_vel_w: [3] Measured base linear velocity in world frame.
            cmd_vel_b: [2] Commanded linear velocity (vx, vy) in body frame.
            base_ang_vel_w: [3] Measured base angular velocity in world frame.
            cmd_yaw_rate: Commanded yaw rate in rad/s.
            elevation_map: Local elevation history map.
            lead_time: [4] 或标量，每条腿距离其触地的提前时间 (秒)，用于把触地关节角
                的参考换算到该腿触地时刻的预测基座位置。
            
        Returns:
            Dictionary containing:
                - 'optimal_footholds_w': [4, 3] 3D foothold positions in world frame
                - 'desired_base_pos_w': [3] 3D planned base position
                - 'desired_base_quat_w': [4] Planned base quaternion [w, x, y, z]
                - 'desired_base_euler': [3] Planned base Euler angles [roll, pitch, yaw]
                - 'desired_base_lin_acc_w': [3] Planned base linear acceleration
                - 'desired_base_ang_acc_w': [3] Planned base angular acceleration
                - 'desired_joint_pos': [12] Planned touchdown joint angles via Analytical IK
        """
        # 1. Plan 3D optimal footholds using TAMOLS gradient & virtual floor costs
        optimal_footholds_w, debug_info = self.foothold_planner.plan_footholds(
            nominal_footholds_w=nominal_footholds_w,
            base_pos_w=base_pos_w,
            base_yaw=base_yaw,
            elevation_map=elevation_map,
        )

        # 2. Plan Terrain-Adaptive Base Pose (TAMOLS Section V.B.4 & Eq. 34)
        # Footholds order: 0: FL, 1: FR, 2: RL, 3: RR
        z_fl = optimal_footholds_w[0, 2]
        z_fr = optimal_footholds_w[1, 2]
        z_rl = optimal_footholds_w[2, 2]
        z_rr = optimal_footholds_w[3, 2]

        # Target base height: mean foot surface height + clearance h_des
        mean_ground_z = float(np.mean(optimal_footholds_w[:, 2]))
        des_pos_z = mean_ground_z + self.cfg.optimizer_cfg.h_des

        des_base_pos_w = np.array([
            base_pos_w[0],
            base_pos_w[1],
            des_pos_z,
        ], dtype=np.float32)

        # Orientation alignment:
        # Pitch: Front vs Rear foot height gradient along body X
        z_front = 0.5 * (z_fl + z_fr)
        z_rear = 0.5 * (z_rl + z_rr)
        dx_pitch = abs(GO2_HIP_OFFSETS_TRUNK[0, 0] - GO2_HIP_OFFSETS_TRUNK[2, 0])  # ~0.3868m
        raw_pitch = math.atan2(z_front - z_rear, max(dx_pitch, 0.1))
        des_pitch = float(np.clip(raw_pitch, -self.cfg.max_pitch, self.cfg.max_pitch))

        # Roll: Left vs Right foot height gradient along body Y
        z_left = 0.5 * (z_fl + z_rl)
        z_right = 0.5 * (z_fr + z_rr)
        dy_roll = 2.0 * abs(self.cfg.optimizer_cfg.hip_offsets[0, 1])  # ~0.29m
        raw_roll = math.atan2(z_left - z_right, max(dy_roll, 0.1))
        des_roll = float(np.clip(raw_roll, -self.cfg.max_roll, self.cfg.max_roll))

        # Yaw: Follows commanded heading
        des_yaw = base_yaw

        des_euler = np.array([des_roll, des_pitch, des_yaw], dtype=np.float32)
        des_quat_w = euler_zyx_to_quat_np(des_roll, des_pitch, des_yaw)
        R_w_b_des = euler_zyx_to_rotation_matrix_np(des_roll, des_pitch, des_yaw)
        R_b_w_des = R_w_b_des.T

        # 3. Reference Base Accelerations (Second-order tracking model)
        # Linear acceleration:
        # Horizontally: accelerate towards commanded velocity
        # Vertically: track desired clearance height
        cos_y = math.cos(base_yaw)
        sin_y = math.sin(base_yaw)
        cmd_vel_w_x = cos_y * cmd_vel_b[0] - sin_y * cmd_vel_b[1]
        cmd_vel_w_y = sin_y * cmd_vel_b[0] + cos_y * cmd_vel_b[1]

        omega_l = self.cfg.acc_filter_omega_lin
        zeta_l = self.cfg.acc_filter_zeta_lin

        a_x = omega_l * (cmd_vel_w_x - base_vel_w[0])
        a_y = omega_l * (cmd_vel_w_y - base_vel_w[1])
        a_z = omega_l**2 * (des_pos_z - base_pos_w[2]) - 2.0 * zeta_l * omega_l * base_vel_w[2]

        des_lin_acc_w = np.clip(
            np.array([a_x, a_y, a_z], dtype=np.float32),
            -self.cfg.max_lin_acc, self.cfg.max_lin_acc
        )

        # Angular acceleration:
        omega_a = self.cfg.acc_filter_omega_ang
        zeta_a = self.cfg.acc_filter_zeta_ang

        ang_acc_x = -omega_a**2 * (0.0 - des_roll) - 2.0 * zeta_a * omega_a * base_ang_vel_w[0]
        ang_acc_y = -omega_a**2 * (0.0 - des_pitch) - 2.0 * zeta_a * omega_a * base_ang_vel_w[1]
        ang_acc_z = omega_a * (cmd_yaw_rate - base_ang_vel_w[2])

        des_ang_acc_w = np.clip(
            np.array([ang_acc_x, ang_acc_y, ang_acc_z], dtype=np.float32),
            -self.cfg.max_ang_acc, self.cfg.max_ang_acc
        )

        # Desired velocities
        des_lin_vel_w = np.array([cmd_vel_w_x, cmd_vel_w_y, 0.0], dtype=np.float32)
        des_ang_vel_w = np.array([0.0, 0.0, cmd_yaw_rate], dtype=np.float32)

        # 4. Touchdown Desired Joint Positions via Analytical IK (DTC Section "Policy Observations")
        # Transform optimal footholds into the planned base frame, then hip frames
        desired_joint_pos = np.zeros(12, dtype=np.float32)

        # 每条腿触地时刻的预测基座位置: 落脚点代表的是"该腿触地时"的落点，
        # 因此关节角应相对触地时刻的基座位置计算；直接用当前基座会让高速下
        # 目标腿看起来被拉得非常直 (实际触地时基座已经前进了 lead*v)。
        if lead_time is None:
            lead_per_leg = np.full(4, 0.5 * self.cfg.optimizer_cfg.stance_time, dtype=np.float32)
        else:
            lead_per_leg = np.broadcast_to(np.asarray(lead_time, dtype=np.float32), (4,))
        cmd_vel_w_td = np.array([cmd_vel_w_x, cmd_vel_w_y, 0.0], dtype=np.float32)

        for leg_i in range(4):
            # Foothold vector relative to the base at this leg's touchdown in world frame
            p_foot_w = optimal_footholds_w[leg_i]
            base_pos_td = des_base_pos_w + lead_per_leg[leg_i] * cmd_vel_w_td
            delta_w = p_foot_w - base_pos_td
            # Rotate into planned base frame
            p_foot_b = R_b_w_des @ delta_w
            # Subtract hip offset
            p_foot_hip = p_foot_b - GO2_HIP_OFFSETS_TRUNK[leg_i]
            is_left = (GO2_L_HIP_SIGNS[leg_i] > 0)

            # Analytical IK
            q_hip, q_thigh, q_calf = go2_leg_analytical_ik_np(p_foot_hip, is_left=is_left)
            desired_joint_pos[leg_i * 3 + 0] = q_hip
            desired_joint_pos[leg_i * 3 + 1] = q_thigh
            desired_joint_pos[leg_i * 3 + 2] = q_calf

        return {
            "optimal_footholds_w": optimal_footholds_w,
            "desired_base_pos_w": des_base_pos_w,
            "desired_base_quat_w": des_quat_w,
            "desired_base_euler": des_euler,
            "desired_base_lin_vel_w": des_lin_vel_w,
            "desired_base_ang_vel_w": des_ang_vel_w,
            "desired_base_lin_acc_w": des_lin_acc_w,
            "desired_base_ang_acc_w": des_ang_acc_w,
            "desired_joint_pos": desired_joint_pos,
            "debug_info": debug_info,
        }


def euler_zyx_to_quat_torch(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Convert ZYX Euler angles (roll, pitch, yaw) to quaternion [w, x, y, z] (PyTorch)."""
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z], dim=-1)


class TorchTAMOLSMotionPlanner(nn.Module):
    """Full Batched PyTorch GPU TAMOLS Motion Planner.
    
    Computes optimal footholds, terrain-adaptive base pose, accelerations, and desired joint angles
    directly on CUDA tensors with near-zero latency.
    """
    def __init__(
        self,
        cfg: TAMOLSMotionPlannerCfg | None = None,
        cell_size: float = 0.05,
        x_min: float = -1.0,
        x_max: float = 1.4,
        y_min: float = -0.9,
        y_max: float = 0.9,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.cfg = cfg or TAMOLSMotionPlannerCfg()
        self.device = torch.device(device)
        self.foothold_planner = TorchTAMOLSFootholdPlanner(
            cfg=self.cfg.optimizer_cfg,
            cell_size=cell_size,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            device=self.device,
        )

    def plan_motion(
        self,
        nominal_footholds_w: torch.Tensor,
        base_pos_w: torch.Tensor,
        base_yaw: torch.Tensor,
        base_vel_w: torch.Tensor,
        cmd_vel_b: torch.Tensor,
        base_ang_vel_w: torch.Tensor | None,
        cmd_yaw_rate: torch.Tensor | float,
        terrain_maps: torch.Tensor,
        lead_time: torch.Tensor | float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute full TAMOLS motion references for a batch of robots on GPU.

        Args:
            lead_time: [B, 4] 或标量，每条腿距离其触地的提前时间 (秒)。用于把触地
                关节角的参考换算到该腿触地时刻的预测基座位置。
        """
        B = base_pos_w.shape[0]

        # 1. Plan 3D optimal footholds on GPU
        optimal_footholds_w, debug_info = self.foothold_planner.plan_footholds(
            nominal_footholds_w=nominal_footholds_w,
            base_pos_w=base_pos_w,
            base_yaw=base_yaw,
            terrain_maps=terrain_maps,
        )

        # 2. Plan Terrain-Adaptive Base Pose
        mean_ground_z = torch.mean(optimal_footholds_w[..., 2], dim=-1)  # [B]
        des_pos_z = mean_ground_z + self.cfg.optimizer_cfg.h_des

        des_base_pos_w = torch.stack([
            base_pos_w[:, 0],
            base_pos_w[:, 1],
            des_pos_z,
        ], dim=-1)  # [B, 3]

        # Pitch alignment
        z_fl = optimal_footholds_w[:, 0, 2]
        z_fr = optimal_footholds_w[:, 1, 2]
        z_rl = optimal_footholds_w[:, 2, 2]
        z_rr = optimal_footholds_w[:, 3, 2]

        z_front = 0.5 * (z_fl + z_fr)
        z_rear = 0.5 * (z_rl + z_rr)
        dx_pitch = abs(GO2_HIP_OFFSETS_TRUNK[0, 0] - GO2_HIP_OFFSETS_TRUNK[2, 0])
        raw_pitch = torch.atan2(z_front - z_rear, torch.full_like(z_front, dx_pitch))
        des_pitch = torch.clamp(raw_pitch, -self.cfg.max_pitch, self.cfg.max_pitch)

        # Roll alignment
        z_left = 0.5 * (z_fl + z_rl)
        z_right = 0.5 * (z_fr + z_rr)
        dy_roll = 2.0 * abs(self.cfg.optimizer_cfg.hip_offsets[0, 1])
        raw_roll = torch.atan2(z_left - z_right, torch.full_like(z_left, dy_roll))
        des_roll = torch.clamp(raw_roll, -self.cfg.max_roll, self.cfg.max_roll)

        des_yaw = base_yaw
        des_euler = torch.stack([des_roll, des_pitch, des_yaw], dim=-1)
        des_quat_w = euler_zyx_to_quat_torch(des_roll, des_pitch, des_yaw)

        # 3. Reference Base Accelerations & Velocities
        cos_y = torch.cos(base_yaw)
        sin_y = torch.sin(base_yaw)
        cmd_vx = cmd_vel_b[:, 0]
        cmd_vy = cmd_vel_b[:, 1] if cmd_vel_b.shape[-1] > 1 else torch.zeros_like(cmd_vx)

        cmd_vel_w_x = cos_y * cmd_vx - sin_y * cmd_vy
        cmd_vel_w_y = sin_y * cmd_vx + cos_y * cmd_vy

        des_lin_vel_w = torch.stack([cmd_vel_w_x, cmd_vel_w_y, torch.zeros_like(cmd_vx)], dim=-1)

        if isinstance(cmd_yaw_rate, (int, float)):
            cmd_wz = torch.full_like(base_yaw, float(cmd_yaw_rate))
        else:
            cmd_wz = cmd_yaw_rate

        des_ang_vel_w = torch.stack([torch.zeros_like(cmd_wz), torch.zeros_like(cmd_wz), cmd_wz], dim=-1)

        omega_l = self.cfg.acc_filter_omega_lin
        zeta_l = self.cfg.acc_filter_zeta_lin

        a_x = omega_l * (cmd_vel_w_x - base_vel_w[:, 0])
        a_y = omega_l * (cmd_vel_w_y - base_vel_w[:, 1])
        a_z = omega_l**2 * (des_pos_z - base_pos_w[:, 2]) - 2.0 * zeta_l * omega_l * base_vel_w[:, 2]

        des_lin_acc_w = torch.clamp(
            torch.stack([a_x, a_y, a_z], dim=-1),
            -self.cfg.max_lin_acc, self.cfg.max_lin_acc
        )

        omega_a = self.cfg.acc_filter_omega_ang
        zeta_a = self.cfg.acc_filter_zeta_ang

        if base_ang_vel_w is not None:
            w_x = base_ang_vel_w[:, 0]
            w_y = base_ang_vel_w[:, 1]
            w_z = base_ang_vel_w[:, 2]
        else:
            w_x = torch.zeros_like(cmd_wz)
            w_y = torch.zeros_like(cmd_wz)
            w_z = torch.zeros_like(cmd_wz)

        ang_acc_x = -omega_a**2 * (-des_roll) - 2.0 * zeta_a * omega_a * w_x
        ang_acc_y = -omega_a**2 * (-des_pitch) - 2.0 * zeta_a * omega_a * w_y
        ang_acc_z = omega_a * (cmd_wz - w_z)

        des_ang_acc_w = torch.clamp(
            torch.stack([ang_acc_x, ang_acc_y, ang_acc_z], dim=-1),
            -self.cfg.max_ang_acc, self.cfg.max_ang_acc
        )

        # 4. Touchdown Desired Joint Positions via Analytical IK (PyTorch)
        # 落脚点代表该腿触地时的落点，关节角应相对"触地时刻的预测基座位置"计算。
        zeros_b = torch.zeros_like(cmd_vx)
        cmd_vel_w = torch.stack([cmd_vel_w_x, cmd_vel_w_y, zeros_b], dim=-1)  # [B, 3]
        if lead_time is None:
            lead = torch.full(
                (B, 4), 0.5 * self.cfg.optimizer_cfg.stance_time,
                device=base_pos_w.device, dtype=base_pos_w.dtype,
            )
        elif isinstance(lead_time, (int, float)):
            lead = torch.full(
                (B, 4), float(lead_time),
                device=base_pos_w.device, dtype=base_pos_w.dtype,
            )
        else:
            lead = lead_time.to(device=base_pos_w.device, dtype=base_pos_w.dtype)
            if lead.dim() == 1:
                lead = lead.view(1, 4).expand(B, 4)
        base_pos_at_touchdown = (
            des_base_pos_w.unsqueeze(1) + lead.unsqueeze(-1) * cmd_vel_w.unsqueeze(1)
        )  # [B, 4, 3]

        foot_pos_hip = transform_footholds_to_hip_frame_torch(
            footholds_w=optimal_footholds_w,
            base_pos_w=base_pos_at_touchdown,
            base_quat_w=des_quat_w,
        )
        desired_joint_pos = go2_analytical_ik_batch_torch(foot_pos_hip)

        return {
            "optimal_footholds_w": optimal_footholds_w,
            "desired_base_pos_w": des_base_pos_w,
            "desired_base_quat_w": des_quat_w,
            "desired_base_euler": des_euler,
            "desired_base_lin_vel_w": des_lin_vel_w,
            "desired_base_ang_vel_w": des_ang_vel_w,
            "desired_base_lin_acc_w": des_lin_acc_w,
            "desired_base_ang_acc_w": des_ang_acc_w,
            "desired_joint_pos": desired_joint_pos,
            "debug_info": debug_info,
        }
