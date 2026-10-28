"""DTC actor 观测项 (TAMOLS 相关), 供 velocity 任务 cfg 接入 (本模块位于 DTC/observation).

用户要求的 actor obs 清单与现有/新增对照
========================================
  - 当前 twist (基座线/角速度)   -> 既有 base_lin_vel / base_ang_vel
  - 重力向量                     -> 既有 projected_gravity
  - 关节位置 / 关节速度          -> 既有 joint_pos / joint_vel
  - 上次 action                  -> 既有 actions
  - 期望 twist                   -> 既有 command (generated_commands)
  - TAMOLS 规划的落脚点           -> :func:`tamols_footholds` (新增)
  - 期望关节位置 (触地参考)       -> :func:`tamols_desired_joint_pos` (新增)
  - 期望接触状态 (固定 trot)      -> :func:`tamols_desired_contacts` (新增)
  - 高程图扫描                   -> mjlab ``mdp.height_scan`` (terrain_scan);
                                    flat 变体无该传感器, 变体 cfg 移除此
                                    obs 项 (见 config/env_cfg.py).

数值约定
--------
- 落脚点观测给**相对量**: 落点 - 基座位置, xy 旋转到机体系, z 为高度差
  [N,4,3] 摊平 [N,12] (顺序 FL/FR/RL/RR, 与命令项一致);
- 期望关节位置给**相对 default_joint_pos** 的偏移 (与既有 joint_pos_rel
  同一参考), [N,12];
- 期望接触状态 = 固定 trot 调度器的 contact_states (1=支撑, 0=摆动)
  [N,4], 无噪声 (确定性步态计划);
- 所有函数只读命令项/场景状态, 不改任何规划行为.
"""

from __future__ import annotations

import math

import torch

LEG_ORDER = ("FL", "FR", "RL", "RR")


def _command_term(env, command_name: str):
    term = env.command_manager.get_term(command_name)
    if term is None:
        raise KeyError(
            f"观测依赖命令项 {command_name!r}; 当前命令: "
            f"{list(env.command_manager.terms.keys()) if hasattr(env.command_manager, 'terms') else command_name}"
        )
    return term


def tamols_footholds(
    env,
    command_name: str = "Tamols_footholds",
    entity_name: str = "robot",
) -> torch.Tensor:
    """TAMOLS 规划落脚点 (相对基座, 机体系) [N,12].

    = (落点 - 基座位置) 旋转到机体系, z = 高度差; 顺序 FL/FR/RL/RR.
    """
    term = _command_term(env, command_name)
    asset = env.scene[entity_name]
    footholds = term.optimal_footholds            # [N,4,3] 世界系
    base_pos = asset.data.root_link_pos_w         # [N,3]
    quat = asset.data.root_link_quat_w            # [N,4] (w,x,y,z)
    yaw = torch.atan2(
        2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
        1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
    )
    rel = footholds - base_pos.unsqueeze(1)
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    bx = cos.unsqueeze(1) * rel[..., 0] + sin.unsqueeze(1) * rel[..., 1]
    by = -sin.unsqueeze(1) * rel[..., 0] + cos.unsqueeze(1) * rel[..., 1]
    out = torch.stack((bx, by, rel[..., 2]), dim=-1)
    return out.reshape(out.shape[0], 12)


def tamols_desired_joint_pos(
    env,
    command_name: str = "Tamols_footholds",
    entity_name: str = "robot",
) -> torch.Tensor:
    """触地时刻期望关节位置 (相对 default) [N,12].

    与既有 ``joint_pos_rel`` 同一参考 (减 default_joint_pos), 保证策略
    看到的目标与本体关节观测可直接比较.
    """
    term = _command_term(env, command_name)
    asset = env.scene[entity_name]
    default = asset.data.default_joint_pos
    assert default is not None, "robot 未配置 default_joint_pos"
    return term.desired_joint_pos - default


def tamols_desired_contacts(
    env,
    command_name: str = "Tamols_footholds",
) -> torch.Tensor:
    """固定 trot 的期望接触状态 [N,4] (1=支撑, 0=摆动; FL/FR/RL/RR)."""
    term = _command_term(env, command_name)
    return term.gait_scheduler.contact_states


def tamols_gait_phase(
    env,
    command_name: str = "Tamols_footholds",
    include_swing_mask: bool = True,
) -> torch.Tensor:
    """固定 trot 的每腿相位 (sin/cos 编码) [N,8] 或含摆动掩码 [N,12].

    - 相位 ``phi_i ∈ [0,1)`` 来自命令项共享的 GaitScheduler
      (对角 trot: FL/RR offset=0, FR/RL offset=0.5, T=0.6s);
    - 每腿编码为 ``[sin(2*pi*phi), cos(2*pi*phi)]`` (连续无跳变,
      周期任务标准做法; FL/FR/RL/RR 顺序展平);
    - ``include_swing_mask=True`` 时再拼 ``swing_mask [N,4]``
      (1=摆动), 共 12 维.
    """
    term = _command_term(env, command_name)
    phi = term.gait_scheduler.phases              # [N,4] in [0,1)
    ang = 2.0 * math.pi * phi
    encoded = torch.stack((torch.sin(ang), torch.cos(ang)), dim=-1)
    encoded = encoded.reshape(encoded.shape[0], 8)
    if include_swing_mask:
        mask = term.gait_scheduler.is_swing.float()
        return torch.cat((encoded, mask), dim=-1)
    return encoded


# ---------------------------------------------------------------------- #
# Critic 专用 (特权信息; actor 不给)
# ---------------------------------------------------------------------- #
def _base_yaw(asset) -> torch.Tensor:
    quat = asset.data.root_link_quat_w
    return torch.atan2(
        2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
        1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
    )


def tamols_base_pos_desired(
    env,
    command_name: str = "Tamols_footholds",
    entity_name: str = "robot",
) -> torch.Tensor:
    """TAMOLS 期望基座位置 (相对当前实际位置, 机体系) [N,3].

    = desired_base_pos_w - root_link_pos_w, xy 旋转到机体系, z 为高度差.
    """
    term = _command_term(env, command_name)
    asset = env.scene[entity_name]
    rel = term.desired_base_pos_w - asset.data.root_link_pos_w
    yaw = _base_yaw(asset)
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    return torch.stack(
        (
            cos * rel[:, 0] + sin * rel[:, 1],
            -sin * rel[:, 0] + cos * rel[:, 1],
            rel[:, 2],
        ),
        dim=-1,
    )


def tamols_base_quat_desired(
    env,
    command_name: str = "Tamols_footholds",
) -> torch.Tensor:
    """TAMOLS 期望基座姿态四元数 [N,4] (w,x,y,z; 已归一化)."""
    term = _command_term(env, command_name)
    quat = term.desired_base_quat_w
    return quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-9)


def tamols_base_twist_desired(
    env,
    command_name: str = "Tamols_footholds",
    entity_name: str = "robot",
) -> torch.Tensor:
    """TAMOLS 期望基座 twist (机体系) [N,6] = [lin(3), ang(3)].

    **与 actor 的 "command" 观测严格区分**: 这里是 TAMOLS 规划输出的
    基座线/角速度 (``desired_base_lin_vel_w`` / ``desired_base_ang_vel_w``,
    来自触地重规划的基座轨迹), 旋转到机体系; actor 的 ``command`` 是
    velocity 命令项 (UniformVelocityCommand) 的目标速度指令.
    """
    term = _command_term(env, command_name)
    asset = env.scene[entity_name]
    yaw = _base_yaw(asset)
    cos, sin = torch.cos(yaw), torch.sin(yaw)

    def rot(v: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                cos * v[:, 0] + sin * v[:, 1],
                -sin * v[:, 0] + cos * v[:, 1],
                v[:, 2],
            ),
            dim=-1,
        )

    return torch.cat(
        (
            rot(term.desired_base_lin_vel_w),
            rot(term.desired_base_ang_vel_w),
        ),
        dim=-1,
    )
