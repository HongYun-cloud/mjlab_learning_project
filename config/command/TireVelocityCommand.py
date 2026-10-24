from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass, field

import math
import torch

from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class TireTraverseVelocityCommand(UniformVelocityCommand):
    """轮胎阵穿越速度命令：持续引导机器人从当前出生点穿越轮胎阵到达对侧终点平地。

    原理
    ----
    1. 每个 step 计算当前机器人世界位置与对侧目标点 (0, target_y) 的相对向量。
    2. 计算对侧目标方向在机器人 body frame 中的投影 [vx_b, vy_b]，并乘以 forward_speed。
    3. 同时根据目标航向与当前航向的夹角偏差，施加航向纠偏角速度 wz_b。
    这样无论机器人在攀爬轮胎过程中发生侧向滑移还是旋转偏转，指令始终稳定拉向对侧终点。
    """

    cfg: TireTraverseVelocityCommandCfg

    def __init__(self, cfg: TireTraverseVelocityCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # 禁用父类随机模式
        self.is_heading_env[env_ids] = False
        self.is_standing_env[env_ids] = False
        self.is_world_env[env_ids] = False
        self.is_forward_env[env_ids] = False

        self.vel_command_b[env_ids, 0] = self.cfg.forward_speed
        self.vel_command_b[env_ids, 1] = 0.0
        self.vel_command_b[env_ids, 2] = 0.0

    def compute(self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        super().compute(dt, env_ids)

        robot_pos_w = self.robot.data.root_link_pos_w      # (N, 3)
        env_origins = self._env.scene.env_origins          # (N, 3)

        # 机器人相对于自身子环境原点的 XY 坐标
        rel_pos = robot_pos_w[:, :2] - env_origins[:, :2]  # (N, 2)

        # 对侧目标点在 sub-terrain 局部系中的位置: (0.0, target_y)
        # 例如 target_y = -3.0m
        target_local = torch.tensor([0.0, self.cfg.target_y], device=self.device)
        to_target = target_local - rel_pos                 # (N, 2)
        dist_to_target = torch.norm(to_target, dim=-1, keepdim=True).clamp(min=0.05)
        dir_w = to_target / dist_to_target                 # (N, 2) 单位方向向量

        # 机器人世界系航向角
        heading = self.robot.data.heading_w                # (N,)

        # 将 world-frame 方向向量旋转到 body frame
        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        vx_b =  cos_h * dir_w[:, 0] + sin_h * dir_w[:, 1]
        vy_b = -sin_h * dir_w[:, 0] + cos_h * dir_w[:, 1]

        # 计算目标航向角 (指向对侧目标点) 与当前航向的角差
        desired_heading = torch.atan2(dir_w[:, 1], dir_w[:, 0])
        heading_err = torch.atan2(
            torch.sin(desired_heading - heading),
            torch.cos(desired_heading - heading),
        )
        wz_b = torch.clamp(self.cfg.heading_gain * heading_err, -self.cfg.max_ang_vel, self.cfg.max_ang_vel)

        # 判断是否已经越过终点线（若 y <= target_y，说明已成功穿过轮胎阵）
        reached_target = rel_pos[:, 1] <= self.cfg.target_y
        
        # 写入线速度与角速度指令
        self.vel_command_b[:, 0] = self.cfg.forward_speed * vx_b
        self.vel_command_b[:, 1] = self.cfg.forward_speed * vy_b
        self.vel_command_b[:, 2] = wz_b

        # 越过终点后平稳保持直线冲刺
        if reached_target.any():
            reached_ids = reached_target.nonzero(as_tuple=False).flatten()
            self.vel_command_b[reached_ids, 0] = self.cfg.forward_speed
            self.vel_command_b[reached_ids, 1] = 0.0
            self.vel_command_b[reached_ids, 2] = 0.0


@dataclass(kw_only=True)
class TireTraverseVelocityCommandCfg(UniformVelocityCommandCfg):
    """轮胎阵穿越速度命令配置。"""
    forward_speed: float = 1.0
    """目标前进速率 (m/s)。"""

    target_y: float = -3.0
    """对侧目标区域在 sub-terrain local 坐标系中的 Y 坐标 (默认 -3.0m)。"""

    heading_gain: float = 1.5
    """对准终点的航向校正增益。"""

    max_ang_vel: float = 1.0
    """最大允许航向校正角速度 (rad/s)。"""

    ranges: UniformVelocityCommandCfg.Ranges = field(
        default_factory=lambda: UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=None,
        )
    )

    def build(self, env: ManagerBasedRlEnv) -> TireTraverseVelocityCommand:
        return TireTraverseVelocityCommand(self, env)

    def __post_init__(self):
        self.rel_standing_envs = 0.0
        self.rel_heading_envs  = 0.0
        self.rel_world_envs    = 0.0
        self.rel_forward_envs  = 0.0
        self.heading_command   = False
        self.init_velocity_prob = 0.0
        self.heading_control_stiffness = 2.0

