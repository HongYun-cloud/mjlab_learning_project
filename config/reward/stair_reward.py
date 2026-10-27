from __future__ import annotations

from typing import TYPE_CHECKING

import math

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity_stair(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward for tracking the commanded base linear velocity (xy only).

    与 ``mdp.track_linear_velocity`` 的区别：
    1. 不惩罚垂直速度——爬楼梯必然有 vz，惩罚它会与任务目标冲突。
    2. 跟踪偏差大时返回负值（扣奖励），而不是只趋近于 0，从而打破"站姿局部最优"。
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_lin_vel_b
    xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
    # exp(-e/std^2) 在 e 较大时趋于 0，站着不动没有任何代价；
    # 减去 1.0 后，跟踪越差扣得越多（最大 -1），跟踪完美时为 0。
    return torch.exp(-xy_error / std**2) - 1.0

def reach_peak_bonus(
    env,
    command_name: str = "twist",
    foothold_term: str = "foothold_hard_tracking",
    bonus_scale: float = 1.5,
    eps: float = 1e-3,
    default_d_min: float = 0.02,
    only_last_seconds: float | None = 2.0,
    goal_weight: float = 1.0 / 3.0,
    velocity_weight: float = 1.0 / 3.0,
    posture_weight: float = 1.0 / 3.0,
    goal_sigma: float = 0.40,
    velocity_sigma: float = 0.10,
    yaw_rate_sigma: float = 0.15,
    posture_sigma: float = 0.15,
) -> torch.Tensor:
    """episode 最后一段时间内、在终点停稳且姿态良好时的分项保持奖励 [N]。

    触发条件（每一步）:
    - ``arrived_target``: 当前在终点半径 (StairVelocityCommand,
      north_peak 模式) 内;
    - ``time_left <= only_last_seconds``: 只奖励 episode 最后 2 s;
    - 不再要求 ``peak_reached_now`` 上升沿, 因此提前到终点后一直站着,
      最后 2 s 仍然可以拿分; 到终点也不再终止 episode.

    奖励由三部分组成 (权重默认各 1/3, 越好越高):
    - 到达终点: ``0.5 + 0.5*exp(-(dist / goal_sigma)^2)``; dist 为到目标点
      的 xy 距离. 在到达半径内至少有 0.5 的基础分, 越靠近中心越高;
    - 速度为 0: 线速度与 yaw 角速度评分的平均; 既惩罚平移, 也惩罚
      到终点后原地转圈;
    - 姿态良好: ``exp(-(tilt / posture_sigma)^2)``, tilt 为机体系 z 轴
      与世界 z 轴的夹角.

    量级: 三部分满分且整个 ``only_last_seconds`` 窗口都保持时, 总奖励
    ≈ ``bonus_scale * 足端跟踪单次加权最大值`` (默认约 59.1). 只满足
    其中一部分、或只保持部分时间, 按比例拿分.
    """
    cmd = env.command_manager.get_term(command_name)
    arrived = getattr(cmd, "arrived_target", None)
    device = env.device
    if arrived is None or not bool(arrived.any()):
        return torch.zeros(env.num_envs, device=device)
    asset = env.scene["robot"]

    max_s = getattr(env, "max_episode_length_s", None)
    gate = arrived.clone()
    if only_last_seconds is not None and float(only_last_seconds) > 0.0:
        window = float(only_last_seconds)
        if max_s is not None:
            time_left = float(max_s) - env.episode_length_buf.to(
                device=device, dtype=torch.float32
            ) * float(env.step_dt)
            gate = gate & (time_left <= window + 1e-6)
    elif max_s is not None and float(max_s) < 1.0e8:
        # 不设时间窗时, 按整个 episode 归一化, 避免长期站在终点刷分.
        window = float(max_s)
    else:
        window = 1.0
    if not bool(gate.any()):
        return torch.zeros(env.num_envs, device=device)

    # 目标点 (与 StairVelocityCommand.north_peak 一致).
    target_mode = getattr(cmd.cfg, "target_mode", "center")
    if target_mode == "north_peak":
        offset = torch.tensor(
            getattr(cmd.cfg, "target_offset_w", (0.0, 0.0)),
            device=device,
            dtype=torch.float32,
        )
        target_xy = env.scene.env_origins[:, :2] + offset.unsqueeze(0)
    else:
        target_xy = env.scene.env_origins[:, :2]
    dist = (asset.data.root_link_pos_w[:, :2] - target_xy).norm(dim=-1)

    goal_score = 0.5 + 0.5 * torch.exp(
        -((dist / max(float(goal_sigma), 1e-6)) ** 2)
    )

    speed = asset.data.root_link_lin_vel_b[:, :2].norm(dim=-1)
    linear_score = torch.exp(
        -((speed / max(float(velocity_sigma), 1e-6)) ** 2)
    )
    ang_vel_b = getattr(asset.data, "root_link_ang_vel_b", None)
    if ang_vel_b is None:
        yaw_rate_score = torch.ones_like(linear_score)
    else:
        yaw_rate = ang_vel_b[:, 2].abs()
        yaw_rate_score = torch.exp(
            -((yaw_rate / max(float(yaw_rate_sigma), 1e-6)) ** 2)
        )
    velocity_score = 0.5 * (linear_score + yaw_rate_score)

    quat = asset.data.root_link_quat_w  # w, x, y, z
    # R[2,2] = 1 - 2(x^2 + y^2); acos 得到机体系 z 轴相对世界 z 的倾角.
    up_z = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    tilt = torch.acos(torch.clamp(up_z, -1.0, 1.0))
    posture_score = torch.exp(
        -((tilt / max(float(posture_sigma), 1e-6)) ** 2)
    )

    weight_sum = max(
        float(goal_weight) + float(velocity_weight) + float(posture_weight),
        1e-9,
    )
    score = (
        float(goal_weight) * goal_score
        + float(velocity_weight) * velocity_score
        + float(posture_weight) * posture_score
    ) / weight_sum

    weight, d_min = 6.0, default_d_min
    try:
        rcfg = env.reward_manager.get_term_cfg(foothold_term)
        weight = float(rcfg.weight)
        d_min = float(rcfg.params.get("d_min", default_d_min))
    except Exception:
        pass

    single_max = weight * (-math.log(d_min * d_min + eps))
    bonus_total = bonus_scale * single_max
    return gate.to(torch.float32) * score * (bonus_total / max(window, 1e-6))
