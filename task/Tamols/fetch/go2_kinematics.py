from __future__ import annotations

import math
import numpy as np
import torch

# Unitree Go2 kinematic constants (meters)
GO2_HIP_OFFSETS_TRUNK = np.array([
    [0.1934, 0.0465, 0.0],   # FL (0)
    [0.1934, -0.0465, 0.0],  # FR (1)
    [-0.1934, 0.0465, 0.0],  # RL (2)
    [-0.1934, -0.0465, 0.0], # RR (3)
], dtype=np.float32)

GO2_L_HIP_SIGNS = [1.0, -1.0, 1.0, -1.0]  # FL, FR, RL, RR (+ for left, - for right)
GO2_L_HIP = 0.0955
GO2_L_THIGH = 0.213
GO2_L_CALF = 0.213


def go2_leg_forward_kinematics_np(
    q_hip: float,
    q_thigh: float,
    q_calf: float,
    is_left: bool = True,
) -> np.ndarray:
    """Compute 3D foot position in Hip joint frame for a single leg (NumPy)."""
    l_hip = GO2_L_HIP if is_left else -GO2_L_HIP
    l_thigh = GO2_L_THIGH
    l_calf = GO2_L_CALF

    c_h, s_h = math.cos(q_hip), math.sin(q_hip)
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, c_h, -s_h],
                   [0.0, s_h,  c_h]], dtype=np.float32)

    c_t, s_t = math.cos(q_thigh), math.sin(q_thigh)
    Ry_t = np.array([[ c_t, 0.0, s_t],
                    [ 0.0, 1.0, 0.0],
                    [-s_t, 0.0, c_t]], dtype=np.float32)

    q_tc = q_thigh + q_calf
    c_k, s_k = math.cos(q_tc), math.sin(q_tc)
    Ry_k = np.array([[ c_k, 0.0, s_k],
                    [ 0.0, 1.0, 0.0],
                    [-s_k, 0.0, c_k]], dtype=np.float32)

    p_thigh = Rx @ np.array([0.0, l_hip, 0.0], dtype=np.float32)
    p_calf = p_thigh + Rx @ Ry_t @ np.array([0.0, 0.0, -l_thigh], dtype=np.float32)
    p_foot = p_calf + Rx @ Ry_k @ np.array([0.0, 0.0, -l_calf], dtype=np.float32)
    return p_foot


def go2_leg_analytical_ik_np(
    p_foot_hip: np.ndarray,
    is_left: bool = True,
) -> tuple[float, float, float]:
    """Analytical Inverse Kinematics for a single Go2 leg (NumPy).
    
    Args:
        p_foot_hip: [3] Foot position in Hip joint frame (x, y, z).
        is_left: True for Left legs (FL, RL), False for Right legs (FR, RR).
        
    Returns:
        (q_hip, q_thigh, q_calf) in radians.
    """
    l_hip = GO2_L_HIP if is_left else -GO2_L_HIP
    l_thigh = GO2_L_THIGH
    l_calf = GO2_L_CALF

    px, py, pz = float(p_foot_hip[0]), float(p_foot_hip[1]), float(p_foot_hip[2])

    # 1. Hip Roll joint
    d_yz_sq = py**2 + pz**2
    d = math.sqrt(max(0.0, d_yz_sq - l_hip**2))
    q_hip = math.atan2(py, -pz) - math.atan2(l_hip, d)

    # 2. Leg Sagittal Plane (x, d)
    L_sq = px**2 + d**2
    l_max = (l_thigh + l_calf) * 0.999
    l_min = abs(l_thigh - l_calf) + 1e-4
    L_clamped_sq = np.clip(L_sq, l_min**2, l_max**2)
    L = math.sqrt(L_clamped_sq)

    # Calf / Knee joint (law of cosines)
    cos_calf = (l_thigh**2 + l_calf**2 - L_clamped_sq) / (2.0 * l_thigh * l_calf)
    cos_calf = np.clip(cos_calf, -1.0, 1.0)
    q_calf = -(math.pi - math.acos(cos_calf))

    # Thigh joint
    alpha = math.atan2(-px, d)
    cos_thigh = (l_thigh**2 + L_clamped_sq - l_calf**2) / (2.0 * l_thigh * L)
    cos_thigh = np.clip(cos_thigh, -1.0, 1.0)
    beta = math.acos(cos_thigh)
    q_thigh = alpha + beta

    return float(q_hip), float(q_thigh), float(q_calf)


def go2_analytical_ik_batch_torch(
    foot_pos_hip: torch.Tensor,
) -> torch.Tensor:
    """Vectorized Analytical Inverse Kinematics for Unitree Go2 (PyTorch).
    
    Args:
        foot_pos_hip: Foot positions in hip joint frames.
                      Shape: (..., 4, 3) where index 0=FL, 1=FR, 2=RL, 3=RR.
                      
    Returns:
        joint_positions: Joint angles in radians.
                         Shape: (..., 12) ordered as:
                         [FL_hip, FL_thigh, FL_calf,
                          FR_hip, FR_thigh, FR_calf,
                          RL_hip, RL_thigh, RL_calf,
                          RR_hip, RR_thigh, RR_calf]
    """
    device = foot_pos_hip.device
    dtype = foot_pos_hip.dtype

    l_hip_vals = torch.tensor(
        [GO2_L_HIP, -GO2_L_HIP, GO2_L_HIP, -GO2_L_HIP],
        device=device, dtype=dtype
    )  # [4]
    l_thigh = GO2_L_THIGH
    l_calf = GO2_L_CALF

    px = foot_pos_hip[..., 0]  # [..., 4]
    py = foot_pos_hip[..., 1]
    pz = foot_pos_hip[..., 2]

    # 1. Hip Roll joint
    d_yz_sq = py**2 + pz**2
    l_hip_sq = l_hip_vals**2
    d_eff_sq = torch.clamp(d_yz_sq - l_hip_sq, min=1e-8)
    d = torch.sqrt(d_eff_sq)

    q_hip = torch.atan2(py, -pz) - torch.atan2(l_hip_vals, d)

    # 2. Leg Sagittal Plane (x, d)
    L_sq = px**2 + d**2
    l_max = (l_thigh + l_calf) * 0.999
    l_min = abs(l_thigh - l_calf) + 1e-4
    L_clamped_sq = torch.clamp(L_sq, min=l_min**2, max=l_max**2)
    L = torch.sqrt(L_clamped_sq)

    # Calf / Knee joint
    cos_calf = (l_thigh**2 + l_calf**2 - L_clamped_sq) / (2.0 * l_thigh * l_calf)
    cos_calf = torch.clamp(cos_calf, -1.0, 1.0)
    q_calf = -(math.pi - torch.acos(cos_calf))

    # Thigh joint
    alpha = torch.atan2(-px, d)
    cos_thigh = (l_thigh**2 + L_clamped_sq - l_calf**2) / (2.0 * l_thigh * L)
    cos_thigh = torch.clamp(cos_thigh, -1.0, 1.0)
    beta = torch.acos(cos_thigh)
    q_thigh = alpha + beta

    # Stack: [..., 4, 3] -> [..., 12]
    # [q_hip, q_thigh, q_calf] for each of the 4 legs
    q_legs = torch.stack([q_hip, q_thigh, q_calf], dim=-1)  # [..., 4, 3]
    return q_legs.reshape(*foot_pos_hip.shape[:-2], 12)


def transform_footholds_to_hip_frame_torch(
    footholds_w: torch.Tensor,
    base_pos_w: torch.Tensor,
    base_quat_w: torch.Tensor,
) -> torch.Tensor:
    """Transform footholds from world frame to the 4 Hip joint frames.
    
    Args:
        footholds_w: [..., 4, 3] Foot positions in world frame.
        base_pos_w: [..., 3] Base position in world frame. 也可以给 [..., 4, 3]，
            表示每条腿各自参考的基座位置 (例如该腿触地时刻的预测基座位置)。
        base_quat_w: [..., 4] Base orientation quaternion [w, x, y, z] in world frame.

    Returns:
        foot_pos_hip: [..., 4, 3] Foot positions in hip frames.
    """
    from mjlab.utils.lab_api.math import quat_apply_inverse

    # Vector from base to foot in world frame
    if base_pos_w.dim() == footholds_w.dim():
        # 每条腿一个基座位置 [..., 4, 3]
        delta_w = footholds_w - base_pos_w
    else:
        delta_w = footholds_w - base_pos_w.unsqueeze(-2)  # [..., 4, 3]

    # Rotate into base frame
    b_shape = delta_w.shape
    delta_w_flat = delta_w.reshape(-1, 3)
    quat_flat = base_quat_w.unsqueeze(-2).expand(*b_shape[:-1], 4).reshape(-1, 4)
    delta_b_flat = quat_apply_inverse(quat_flat, delta_w_flat)
    foot_pos_b = delta_b_flat.reshape(b_shape)  # [..., 4, 3]

    # Subtract hip attachment offsets on trunk
    hip_offsets = torch.tensor(
        GO2_HIP_OFFSETS_TRUNK, device=footholds_w.device, dtype=footholds_w.dtype
    )  # [4, 3]
    foot_pos_hip = foot_pos_b - hip_offsets  # [..., 4, 3]

    return foot_pos_hip
