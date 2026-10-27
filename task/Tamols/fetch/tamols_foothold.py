from __future__ import annotations

import math
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .map_processing import process_height_maps
from .elevation_history_map import ElevationHistoryMap


@dataclass
class TAMOLSOptimizerCfg:
    """Hyperparameters for TAMOLS foothold optimization."""
    # Weights matching TAMOLS reference (costs.py)
    w_kinematic: float = 20.0
    """Weight pulling foot towards nominal kinematic stance position."""

    w_edge_raw: float = 3.0
    """Weight penalizing raw height gradient ||grad(h)||^2 (step/cliff edge)."""

    w_edge_smooth: float = 1.0
    """Weight penalizing smoothed gradient ||grad(h_s1)||^2."""

    w_virtual_floor: float = 2.0
    """Weight penalizing falling below virtual floor h_s2 into gaps."""

    # Search window around nominal foothold
    search_radius: float = 0.12
    """Search radius in meters around nominal foothold."""

    search_resolution: float = 0.02
    """Grid step for foothold candidate evaluation in meters."""

    # Robot kinematic limits (Unitree Go2)
    # 结合躯干 hip (y=±0.0465m) 与大腿连杆 thigh (y=±0.0955m)，实际横向展宽 y ≈ ±0.145m (站立宽度约 29cm)
    hip_offsets: np.ndarray = field(default_factory=lambda: np.array([
        [0.1934, 0.145, 0.0],   # FL (0)
        [0.1934, -0.145, 0.0],  # FR (1)
        [-0.1934, 0.145, 0.0],  # RL (2)
        [-0.1934, -0.145, 0.0], # RR (3)
    ], dtype=np.float32))

    l_min: float = 0.15
    """Minimum reachable leg extension from hip in meters."""

    l_max: float = 0.40
    """Maximum reachable leg extension from hip in meters."""

    h_des: float = 0.28
    """Desired standing trunk clearance in meters."""

    stance_time: float = 0.20
    """Default half-cycle stance duration in seconds."""

    k_velocity: float = 0.05
    """Velocity feedback gain for Raibert heuristic."""

    nominal_cmd_blend: float = 1.0
    """标称落脚点前馈速度的命令权重 (0..1)。

    ``1.0`` = 完全使用速度指令 (原行为); ``0.0`` = 完全使用实测基座速度。
    楼梯上策略实际速度通常明显低于指令 (1.0 m/s 指令、实测 0.5~0.7 m/s),
    若仍按指令前馈, 落点会被系统性地放到脚够不到的前方。取 0.5 时
    标称落点介于两者之间, 既保留一点前馈又不会超出腿的可达范围。
    """

    max_step_offset: float = 0.0
    """标称落点相对髋部的最大水平步幅 (m); ``<=0`` 关闭限幅。

    以 Go2 站立高度 ``h_des=0.28``、腿长 0.4 m 计算, 水平可达约 0.29 m;
    考虑抬腿后可达 0.34~0.36 m。楼梯建议 0.30~0.35, 防止规划点落在
    腿完全伸直也够不到的位置。"""


class TAMOLSFootholdPlanner:
    """TAMOLS-based expected foothold planner.
    
    Combines nominal inverted-pendulum kinematics with terrain-aware cost minimization:
      min  w_kin * ||p_xy - p_nom_xy||^2 
         + w_edge_raw * ||grad_h(p)||^2 
         + w_edge_smooth * ||grad_hs1(p)||^2
         + w_floor * (z - hs2(p))^2
      s.t. l_min <= ||p - p_hip|| <= l_max
    """

    def __init__(self, cfg: TAMOLSOptimizerCfg | None = None):
        self.cfg = cfg or TAMOLSOptimizerCfg()

        # Precompute candidate offsets in body frame
        r = self.cfg.search_radius
        res = self.cfg.search_resolution
        dx = np.arange(-r, r + 0.5 * res, res, dtype=np.float32)
        dy = np.arange(-r, r + 0.5 * res, res, dtype=np.float32)
        grid_dx, grid_dy = np.meshgrid(dx, dy, indexing="ij")
        self.cand_dx = grid_dx.flatten()
        self.cand_dy = grid_dy.flatten()
        self.num_candidates = len(self.cand_dx)

    def compute_nominal_footholds(
        self,
        base_pos_w: np.ndarray,
        base_yaw: float,
        base_vel_w: np.ndarray,
        cmd_vel_b: np.ndarray,
        base_ang_vel_w: np.ndarray | None = None,
        cmd_yaw_rate: float = 0.0,
        stance_time: float | None = None,
        lead_time: np.ndarray | float | None = None,
    ) -> np.ndarray:
        """Compute nominal footholds using Raibert heuristic with rotation compensation.

        Args:
            base_pos_w: [3] Base position in world frame (x, y, z).
            base_yaw: Heading angle in radians.
            base_vel_w: [3] Base linear velocity in world frame.
            cmd_vel_b: [2 or 3] Commanded linear velocity in body frame (vx, vy).
            base_ang_vel_w: [3] Base angular velocity in world frame (wx, wy, wz).
            cmd_yaw_rate: Commanded turning yaw rate in rad/s.
            stance_time: Stance duration in seconds (optional).
            lead_time: [4] or scalar 前馈提前时间 (秒)。每条腿的目标是"下次触地后
                半个支撑相"的位置，因此前馈量应为 (距该腿下次触地的时间 + 0.5*支撑相)。
                缺省为 0.5*stance_time (即假设脚尖落地即规划)。

        Returns:
            nominal_footholds_w: [4, 3] Nominal foot positions in world frame.
        """
        st = stance_time or self.cfg.stance_time
        if lead_time is None:
            lead = np.full(4, 0.5 * st, dtype=np.float32)
        else:
            lead = np.broadcast_to(np.asarray(lead_time, dtype=np.float32), (4,))
        cos_y = math.cos(base_yaw)
        sin_y = math.sin(base_yaw)
        R_yaw = np.array([
            [cos_y, -sin_y, 0.0],
            [sin_y,  cos_y, 0.0],
            [0.0,    0.0,   1.0],
        ], dtype=np.float32)

        # Convert world velocity to yaw-aligned body frame
        v_base_b_x = cos_y * base_vel_w[0] + sin_y * base_vel_w[1]
        v_base_b_y = -sin_y * base_vel_w[0] + cos_y * base_vel_w[1]

        cmd_vx = float(cmd_vel_b[0])
        cmd_vy = float(cmd_vel_b[1]) if len(cmd_vel_b) > 1 else 0.0

        omega_z = float(base_ang_vel_w[2]) if base_ang_vel_w is not None else 0.0
        alpha = float(np.clip(self.cfg.nominal_cmd_blend, 0.0, 1.0))
        max_step = float(self.cfg.max_step_offset)

        nominal_w = np.zeros((4, 3), dtype=np.float32)
        for i in range(4):
            hip_b = self.cfg.hip_offsets[i]

            # Rotation-induced tangential velocity at hip: omega x r_hip
            v_rot_x = -omega_z * hip_b[1]
            v_rot_y = omega_z * hip_b[0]
            cmd_rot_x = -cmd_yaw_rate * hip_b[1]
            cmd_rot_y = cmd_yaw_rate * hip_b[0]

            v_i_x = v_base_b_x + v_rot_x
            v_i_y = v_base_b_y + v_rot_y
            cmd_i_x = cmd_vx + cmd_rot_x
            cmd_i_y = cmd_vy + cmd_rot_y

            # 前馈速度 = 指令速度与实测速度的加权混合; alpha=1 时退化为原行为.
            v_feed_x = alpha * cmd_i_x + (1.0 - alpha) * v_i_x
            v_feed_y = alpha * cmd_i_y + (1.0 - alpha) * v_i_y

            # Raibert step offset per leg (lead[i] = 距该腿下次触地时间 + 半个支撑相)
            delta_x = lead[i] * v_feed_x + self.cfg.k_velocity * (v_i_x - v_feed_x)
            delta_y = lead[i] * v_feed_y + self.cfg.k_velocity * (v_i_y - v_feed_y)

            # 限制相对髋部的水平步幅, 避免落点超出腿的可达范围 (v*lead 过大时).
            if max_step > 0.0:
                norm = math.hypot(delta_x, delta_y)
                if norm > max_step:
                    scale = max_step / max(norm, 1e-9)
                    delta_x *= scale
                    delta_y *= scale

            step_offset_b = np.array([delta_x, delta_y, -self.cfg.h_des], dtype=np.float32)

            foot_b = hip_b + step_offset_b
            nominal_w[i] = base_pos_w + R_yaw @ foot_b

        return nominal_w

    def plan_footholds(
        self,
        nominal_footholds_w: np.ndarray,
        base_pos_w: np.ndarray,
        base_yaw: float,
        elevation_map: ElevationHistoryMap,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Optimize expected footholds on terrain elevation map using TAMOLS criteria.
        
        Args:
            nominal_footholds_w: [4, 3] Nominal landing targets in world frame.
            base_pos_w: [3] Robot base position in world frame.
            base_yaw: Base heading in radians.
            elevation_map: ElevationHistoryMap instance containing local terrain.
            
        Returns:
            optimal_footholds_w: [4, 3] Optimized expected foothold targets.
            debug_info: Dict containing cost breakdowns and nominal positions.
        """
        # 1. Process elevation map layers (h_s1, h_s2, gradients)
        h_raw = elevation_map.height_map
        cell_size = elevation_map.cell_size
        h_s1, h_s2, gradients = process_height_maps(h_raw, cell_size=cell_size)

        grad_h_x, grad_h_y = gradients["h"]
        grad_s1_x, grad_s1_y = gradients["h_s1"]

        cos_y = math.cos(base_yaw)
        sin_y = math.sin(base_yaw)
        R_yaw = np.array([
            [cos_y, -sin_y, 0.0],
            [sin_y,  cos_y, 0.0],
            [0.0,    0.0,   1.0],
        ], dtype=np.float32)

        optimal_w = np.copy(nominal_footholds_w)
        costs_per_leg = np.zeros(4, dtype=np.float32)

        for i in range(4):
            p_nom_w = nominal_footholds_w[i]
            hip_w = base_pos_w + R_yaw @ self.cfg.hip_offsets[i]

            # Transform nominal foot to body yaw frame for grid lookup
            rel_w = p_nom_w[:2] - base_pos_w[:2]
            p_nom_bx = cos_y * rel_w[0] + sin_y * rel_w[1]
            p_nom_by = -sin_y * rel_w[0] + cos_y * rel_w[1]

            # Generate candidate points in body yaw frame
            cand_bx = p_nom_bx + self.cand_dx
            cand_by = p_nom_by + self.cand_dy

            # Query grid coordinates (格中心在 x_min + i*cell_size)
            u = (cand_bx - elevation_map.x_min) / cell_size
            v = (cand_by - elevation_map.y_min) / cell_size

            u_clamped = np.clip(np.round(u).astype(int), 0, elevation_map.nx - 1)
            v_clamped = np.clip(np.round(v).astype(int), 0, elevation_map.ny - 1)

            # Terrain properties at candidates
            z_cands = h_raw[u_clamped, v_clamped]
            hs2_cands = h_s2[u_clamped, v_clamped]

            gx_raw = grad_h_x[u_clamped, v_clamped]
            gy_raw = grad_h_y[u_clamped, v_clamped]
            grad_sq_raw = gx_raw**2 + gy_raw**2

            gx_s1 = grad_s1_x[u_clamped, v_clamped]
            gy_s1 = grad_s1_y[u_clamped, v_clamped]
            grad_sq_s1 = gx_s1**2 + gy_s1**2

            # Candidate world coordinates
            cand_wx = base_pos_w[0] + cos_y * cand_bx - sin_y * cand_by
            cand_wy = base_pos_w[1] + sin_y * cand_bx + cos_y * cand_by

            # Kinematics check (leg extension length from hip)
            dx_hip = cand_wx - hip_w[0]
            dy_hip = cand_wy - hip_w[1]
            dz_hip = z_cands - hip_w[2]
            leg_len = np.sqrt(dx_hip**2 + dy_hip**2 + dz_hip**2)

            kin_violation = np.maximum(0.0, self.cfg.l_min - leg_len) + np.maximum(0.0, leg_len - self.cfg.l_max)

            # TAMOLS Cost computation
            # 1. Kinematic deviation cost (stay close to nominal)
            c_kin = (self.cand_dx**2 + self.cand_dy**2)
            # 2. Edge avoidance cost (steep slopes and step edges penalized)
            c_edge = self.cfg.w_edge_raw * grad_sq_raw + self.cfg.w_edge_smooth * grad_sq_s1
            # 3. Virtual floor gap penalty
            c_floor = np.maximum(0.0, hs2_cands - z_cands)**2

            total_cost = (
                self.cfg.w_kinematic * c_kin +
                c_edge +
                self.cfg.w_virtual_floor * c_floor +
                1000.0 * kin_violation
            )

            best_idx = np.argmin(total_cost)
            optimal_w[i, 0] = cand_wx[best_idx]
            optimal_w[i, 1] = cand_wy[best_idx]
            optimal_w[i, 2] = z_cands[best_idx]
            costs_per_leg[i] = total_cost[best_idx]

        debug_info = {
            "nominal_footholds": nominal_footholds_w,
            "optimal_footholds": optimal_w,
            "costs": costs_per_leg,
            "h_s1": h_s1,
            "h_s2": h_s2,
        }

        return optimal_w, debug_info


class TorchTAMOLSFootholdPlanner(nn.Module):
    """Batched PyTorch GPU foothold optimizer for TAMOLS using 2D grid_sample and argmin.
    
    Fully vectorized over [B, 4, K] candidate footholds, executing with near-zero latency
    (<0.05ms) directly on CUDA tensors.
    """
    def __init__(
        self,
        cfg: TAMOLSOptimizerCfg | None = None,
        cell_size: float = 0.05,
        x_min: float = -1.0,
        x_max: float = 1.4,
        y_min: float = -0.9,
        y_max: float = 0.9,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.cfg = cfg or TAMOLSOptimizerCfg()
        self.device = torch.device(device)
        self.cell_size = float(cell_size)
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        # 栅格中心位于 x_min + i*cell_size，因此末格中心是 x_min + (nx-1)*cell_size，
        # 而不是 x_max。grid_sample(align_corners=True) 的归一化必须用 (n-1)*cell。
        self.nx = int(round((self.x_max - self.x_min) / self.cell_size))
        self.ny = int(round((self.y_max - self.y_min) / self.cell_size))

        # Precompute candidate offsets in body frame
        r = float(self.cfg.search_radius)
        res = float(self.cfg.search_resolution)
        dx = torch.arange(-r, r + 0.5 * res, res, dtype=torch.float32)
        dy = torch.arange(-r, r + 0.5 * res, res, dtype=torch.float32)
        grid_dx, grid_dy = torch.meshgrid(dx, dy, indexing="ij")
        cand_dx = grid_dx.flatten()
        cand_dy = grid_dy.flatten()
        cand_kin_cost = cand_dx**2 + cand_dy**2

        hip_offsets = torch.tensor(self.cfg.hip_offsets, dtype=torch.float32)

        self.register_buffer("cand_dx", cand_dx.to(self.device))
        self.register_buffer("cand_dy", cand_dy.to(self.device))
        self.register_buffer("cand_kin_cost", cand_kin_cost.to(self.device))
        self.register_buffer("hip_offsets", hip_offsets.to(self.device))
        self.num_candidates = len(cand_dx)

    def compute_nominal_footholds(
        self,
        base_pos_w: torch.Tensor,
        base_yaw: torch.Tensor,
        base_vel_w: torch.Tensor,
        cmd_vel_b: torch.Tensor,
        base_ang_vel_w: torch.Tensor | None = None,
        cmd_yaw_rate: torch.Tensor | float = 0.0,
        stance_time: float | None = None,
        lead_time: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Vectorized nominal foothold generation using Raibert heuristic with rotation compensation.

        Args:
            base_pos_w: [B, 3] Robot base position in world frame.
            base_yaw: [B] Robot heading angle in radians.
            base_vel_w: [B, 3] Robot base linear velocity in world frame.
            cmd_vel_b: [B, 2] Commanded velocity (vx, vy) in body frame.
            base_ang_vel_w: [B, 3] Base angular velocity in world frame (optional).
            cmd_yaw_rate: [B] or float Commanded yaw rate in rad/s.
            stance_time: Stance duration in seconds (optional).
            lead_time: [B, 4] or scalar 前馈提前时间 (秒)。落脚点目标是"该腿下次
                触地后半个支撑相"的位置，必须补偿"距下次触地的时间"，否则机器人
                前进时目标会系统性地落在身后。缺省为 0.5*stance_time。

        Returns:
            nominal_footholds_w: [B, 4, 3] Nominal landing targets in world frame.
        """
        B = base_pos_w.shape[0]
        st = float(stance_time or self.cfg.stance_time)
        if lead_time is None:
            lead = torch.full((B, 4), 0.5 * st, device=base_pos_w.device, dtype=torch.float32)
        elif isinstance(lead_time, (int, float)):
            lead = torch.full((B, 4), float(lead_time), device=base_pos_w.device, dtype=torch.float32)
        elif lead_time.dim() == 1:
            lead = lead_time.to(base_pos_w.device).view(1, 4).expand(B, 4)
        else:
            lead = lead_time.to(dtype=torch.float32)
        cos_y = torch.cos(base_yaw)
        sin_y = torch.sin(base_yaw)

        # Body-frame linear velocity
        v_base_b_x = cos_y * base_vel_w[:, 0] + sin_y * base_vel_w[:, 1]
        v_base_b_y = -sin_y * base_vel_w[:, 0] + cos_y * base_vel_w[:, 1]

        cmd_vx = cmd_vel_b[:, 0]
        cmd_vy = cmd_vel_b[:, 1] if cmd_vel_b.shape[-1] > 1 else torch.zeros_like(cmd_vx)

        if base_ang_vel_w is not None:
            omega_z = base_ang_vel_w[:, 2]
        else:
            omega_z = torch.zeros_like(base_yaw)

        if isinstance(cmd_yaw_rate, (int, float)):
            cmd_wz = torch.full_like(base_yaw, float(cmd_yaw_rate))
        else:
            cmd_wz = cmd_yaw_rate

        # 4 legs: hip_offsets [4, 3]
        hip_b = self.hip_offsets.unsqueeze(0).expand(B, 4, 3)  # [B, 4, 3]

        # Tangential velocity at hip: omega x r_hip
        v_rot_x = -omega_z.unsqueeze(1) * hip_b[..., 1]
        v_rot_y = omega_z.unsqueeze(1) * hip_b[..., 0]
        cmd_rot_x = -cmd_wz.unsqueeze(1) * hip_b[..., 1]
        cmd_rot_y = cmd_wz.unsqueeze(1) * hip_b[..., 0]

        v_i_x = v_base_b_x.unsqueeze(1) + v_rot_x
        v_i_y = v_base_b_y.unsqueeze(1) + v_rot_y
        cmd_i_x = cmd_vx.unsqueeze(1) + cmd_rot_x
        cmd_i_y = cmd_vy.unsqueeze(1) + cmd_rot_y

        # Raibert step offset (lead = 距该腿下次触地时间 + 半个支撑相)
        alpha = float(min(max(self.cfg.nominal_cmd_blend, 0.0), 1.0))
        v_feed_x = alpha * cmd_i_x + (1.0 - alpha) * v_i_x
        v_feed_y = alpha * cmd_i_y + (1.0 - alpha) * v_i_y
        delta_x = lead * v_feed_x + self.cfg.k_velocity * (v_i_x - v_feed_x)
        delta_y = lead * v_feed_y + self.cfg.k_velocity * (v_i_y - v_feed_y)

        # 限制标称落点相对髋部的水平步幅 (运动学可达范围).
        max_step = float(self.cfg.max_step_offset)
        if max_step > 0.0:
            norm = torch.sqrt(delta_x * delta_x + delta_y * delta_y).clamp(min=1e-9)
            scale = torch.clamp(max_step / norm, max=1.0)
            delta_x = delta_x * scale
            delta_y = delta_y * scale

        foot_b_x = hip_b[..., 0] + delta_x
        foot_b_y = hip_b[..., 1] + delta_y
        foot_b_z = hip_b[..., 2] - self.cfg.h_des

        # Transform to world frame
        nom_wx = base_pos_w[:, 0:1] + cos_y.unsqueeze(1) * foot_b_x - sin_y.unsqueeze(1) * foot_b_y
        nom_wy = base_pos_w[:, 1:2] + sin_y.unsqueeze(1) * foot_b_x + cos_y.unsqueeze(1) * foot_b_y
        nom_wz = base_pos_w[:, 2:3] + foot_b_z

        return torch.stack([nom_wx, nom_wy, nom_wz], dim=-1)

    def plan_footholds(
        self,
        nominal_footholds_w: torch.Tensor,
        base_pos_w: torch.Tensor,
        base_yaw: torch.Tensor,
        terrain_maps: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Batched PyTorch GPU optimization of expected footholds on terrain maps.
        
        Args:
            nominal_footholds_w: [B, 4, 3] Nominal landing targets in world frame.
            base_pos_w: [B, 3] Base position in world frame.
            base_yaw: [B] Base heading angle in radians.
            terrain_maps: [B, 4, Nx, Ny] Stack of (h_raw, h_s2, grad_sq_raw, grad_sq_s1).
            
        Returns:
            optimal_footholds_w: [B, 4, 3] Optimized expected foothold targets.
            debug_info: Dict containing candidate costs.
        """
        B = nominal_footholds_w.shape[0]
        K = self.num_candidates

        cos_y = torch.cos(base_yaw)
        sin_y = torch.sin(base_yaw)

        # 1. Transform nominal footholds to body yaw frame
        rel_w = nominal_footholds_w[..., :2] - base_pos_w[:, None, :2]
        p_nom_bx = cos_y[:, None] * rel_w[..., 0] + sin_y[:, None] * rel_w[..., 1]  # [B, 4]
        p_nom_by = -sin_y[:, None] * rel_w[..., 0] + cos_y[:, None] * rel_w[..., 1]  # [B, 4]

        # 2. Expand candidate points in body yaw frame: [B, 4, K]
        cand_bx = p_nom_bx.unsqueeze(-1) + self.cand_dx.view(1, 1, K)
        cand_by = p_nom_by.unsqueeze(-1) + self.cand_dy.view(1, 1, K)

        # 3. Normalized coordinates for PyTorch F.grid_sample
        #    地图格中心在 x_min + i*cell_size，align_corners=True 下 -1/+1 映射到首/末格中心
        x_span = max((self.nx - 1) * self.cell_size, 1e-6)
        y_span = max((self.ny - 1) * self.cell_size, 1e-6)
        u_norm = 2.0 * (cand_bx - self.x_min) / x_span - 1.0
        v_norm = 2.0 * (cand_by - self.y_min) / y_span - 1.0

        grid_pts = torch.stack([v_norm, u_norm], dim=-1).view(B, 4 * K, 1, 2)
        sampled = F.grid_sample(
            terrain_maps, grid_pts, mode="bilinear", padding_mode="border", align_corners=True
        )
        sampled = sampled.view(B, 4, 4, K)

        z_cands = sampled[:, 0]       # [B, 4, K]
        hs2_cands = sampled[:, 1]     # [B, 4, K]
        grad_sq_raw = sampled[:, 2]   # [B, 4, K]
        grad_sq_s1 = sampled[:, 3]    # [B, 4, K]

        # 4. Candidate world coordinates
        cand_wx = base_pos_w[:, 0:1, None] + cos_y[:, None, None] * cand_bx - sin_y[:, None, None] * cand_by
        cand_wy = base_pos_w[:, 1:2, None] + sin_y[:, None, None] * cand_bx + cos_y[:, None, None] * cand_by

        # 5. Hip world coordinates for reachability check
        hip_b = self.hip_offsets  # [4, 3]
        hip_wx = base_pos_w[:, 0:1] + cos_y[:, None] * hip_b[:, 0] - sin_y[:, None] * hip_b[:, 1]
        hip_wy = base_pos_w[:, 1:2] + sin_y[:, None] * hip_b[:, 0] + cos_y[:, None] * hip_b[:, 1]
        hip_wz = base_pos_w[:, 2:3] + hip_b[:, 2]

        dx_hip = cand_wx - hip_wx.unsqueeze(-1)
        dy_hip = cand_wy - hip_wy.unsqueeze(-1)
        dz_hip = z_cands - hip_wz.unsqueeze(-1)
        leg_len = torch.sqrt(dx_hip**2 + dy_hip**2 + dz_hip**2 + 1e-6)

        kin_violation = (
            torch.clamp(self.cfg.l_min - leg_len, min=0.0) +
            torch.clamp(leg_len - self.cfg.l_max, min=0.0)
        )

        # 6. TAMOLS Cost computation
        c_kin = self.cfg.w_kinematic * self.cand_kin_cost.view(1, 1, K)
        c_edge = self.cfg.w_edge_raw * grad_sq_raw + self.cfg.w_edge_smooth * grad_sq_s1
        c_floor = self.cfg.w_virtual_floor * torch.clamp(hs2_cands - z_cands, min=0.0)**2

        total_cost = c_kin + c_edge + c_floor + 1000.0 * kin_violation  # [B, 4, K]

        best_idx = torch.argmin(total_cost, dim=-1, keepdim=True)  # [B, 4, 1]
        opt_wx = torch.gather(cand_wx, dim=-1, index=best_idx).squeeze(-1)
        opt_wy = torch.gather(cand_wy, dim=-1, index=best_idx).squeeze(-1)
        opt_wz = torch.gather(z_cands, dim=-1, index=best_idx).squeeze(-1)

        optimal_footholds_w = torch.stack([opt_wx, opt_wy, opt_wz], dim=-1)

        debug_info = {
            "costs": torch.gather(total_cost, dim=-1, index=best_idx).squeeze(-1),
            "best_indices": best_idx.squeeze(-1),
        }

        return optimal_footholds_w, debug_info
