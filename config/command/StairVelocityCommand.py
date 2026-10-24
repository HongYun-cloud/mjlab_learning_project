from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass, field

import math
import torch

from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.managers.command_manager import CommandTermCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class StairVelocityCommand(UniformVelocityCommand):
    """楼梯速度命令。

    上楼阶段 (going_up=True)
    -----------------------
    每个 step 重新计算 world-frame 中 "机器人 → env中心" 方向，
    投影到 body frame 得到 [vx_b, vy_b]，乘以 forward_speed。
    这样无论机器人朝向如何，命令始终拉向中心。

    下楼阶段 (going_up=False)
    -----------------------
    切换到下楼时一次性采样一个随机的 world-frame 方向（远离中心的半球内），
    之后每 step 将该 world 方向投影到 body frame 得到命令。
    """

    cfg: StairVelocityCommandCfg

    def __init__(self, cfg: StairVelocityCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        self.going_up = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # 下楼时的 world-frame 目标方向单位向量 (N, 2)
        self._down_dir_w = torch.zeros(self.num_envs, 2, device=self.device)

    # ------------------------------------------------------------------
    # episode reset
    # ------------------------------------------------------------------
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # 禁用父类所有特殊模式
        self.is_heading_env[env_ids] = False
        self.is_standing_env[env_ids] = False
        self.is_world_env[env_ids] = False
        self.is_forward_env[env_ids] = False

        # 重置为上楼
        self.going_up[env_ids] = True
        self._step_count[env_ids] = 0

        # 初始命令占位（compute 里会立刻覆盖）
        self.vel_command_b[env_ids, 0] = self.cfg.forward_speed
        self.vel_command_b[env_ids, 1] = 0.0
        self.vel_command_b[env_ids, 2] = 0.0

    # ------------------------------------------------------------------
    # 辅助：world→body 旋转（与父类 _update_command 完全一致的公式）
    # ------------------------------------------------------------------
    def _world_to_body(self, dir_w_x: torch.Tensor, dir_w_y: torch.Tensor,
                       heading: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """将 world-frame 2D 方向旋转到 body frame。"""
        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        vx_b =  cos_h * dir_w_x + sin_h * dir_w_y
        vy_b = -sin_h * dir_w_x + cos_h * dir_w_y
        return vx_b, vy_b

    # ------------------------------------------------------------------
    # 每 step 计算命令
    # ------------------------------------------------------------------
    def compute(self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        # 父类：处理 resampling 计时、metrics 等
        # is_heading_env / is_standing_env / is_world_env 全 False → 不会覆盖 vel_command_b
        super().compute(dt, env_ids)

        # ---- 位置信息 ----
        robot_pos_w = self.robot.data.root_link_pos_w          # (N, 3)
        env_origins = self._env.scene.env_origins              # (N, 3)
        rel_pos = robot_pos_w[:, :2] - env_origins[:, :2]     # (N, 2) 机器人 - 中心
        dist = torch.norm(rel_pos, dim=-1)                     # (N,)

        # ---- 步数 & 方向切换 ----
        ids = slice(None) if env_ids is None else env_ids
        self._step_count[ids] += 1

        at_center = dist[ids] < self.cfg.center_threshold
        timed_out = self._step_count[ids] >= self.cfg.switch_steps
        should_switch = timed_out | (self.going_up[ids] & at_center)

        if isinstance(ids, torch.Tensor):
            switch_ids = ids[should_switch]
        else:
            switch_ids = should_switch.nonzero(as_tuple=False).flatten()

        if len(switch_ids) > 0:
            self.going_up[switch_ids] = ~self.going_up[switch_ids]
            self._step_count[switch_ids] = 0

            # 新切换到下楼的 env：采样随机 world 方向
            new_down_mask = ~self.going_up[switch_ids]
            new_down_ids = switch_ids[new_down_mask]
            if len(new_down_ids) > 0:
                angles = torch.empty(len(new_down_ids), device=self.device).uniform_(
                    0, 2 * math.pi
                )
                self._down_dir_w[new_down_ids, 0] = torch.cos(angles)
                self._down_dir_w[new_down_ids, 1] = torch.sin(angles)

        # ---- 上楼 env：速度指向中心 ----
        up_mask = self.going_up
        if up_mask.any():
            up_ids = up_mask.nonzero(as_tuple=False).flatten()

            # world-frame 方向：机器人 → 中心 = -rel_pos，归一化
            to_center = -rel_pos[up_ids]                         # (M, 2)
            to_center_norm = torch.norm(to_center, dim=-1, keepdim=True).clamp(min=0.01)
            dir_w = to_center / to_center_norm                   # 单位向量

            # world → body
            heading = self.robot.data.heading_w[up_ids]
            vx_b, vy_b = self._world_to_body(dir_w[:, 0], dir_w[:, 1], heading)

            self.vel_command_b[up_ids, 0] = self.cfg.forward_speed * vx_b
            self.vel_command_b[up_ids, 1] = self.cfg.forward_speed * vy_b
            self.vel_command_b[up_ids, 2] = 0.0

        # ---- 下楼 env：沿保存的随机 world 方向 ----
        down_mask = ~self.going_up
        if down_mask.any():
            down_ids = down_mask.nonzero(as_tuple=False).flatten()

            dir_w = self._down_dir_w[down_ids]                   # (M, 2)
            heading = self.robot.data.heading_w[down_ids]
            vx_b, vy_b = self._world_to_body(dir_w[:, 0], dir_w[:, 1], heading)

            self.vel_command_b[down_ids, 0] = self.cfg.forward_speed * vx_b
            self.vel_command_b[down_ids, 1] = self.cfg.forward_speed * vy_b
            self.vel_command_b[down_ids, 2] = 0.0
        # down_mask = ~self.going_up
        # if down_mask.any():
        #     down_ids = down_mask.nonzero(as_tuple=False).flatten()

        #     self.vel_command_b[down_ids, 0] = self.cfg.forward_speed
        #     self.vel_command_b[down_ids, 1] = 0.0
        #     self.vel_command_b[down_ids, 2] = 0.0


@dataclass(kw_only=True)
class StairVelocityCommandCfg(UniformVelocityCommandCfg):
    """楼梯场景速度命令配置。

    - forward_speed    : 命令速度大小 (m/s)
    - switch_steps     : 上楼超时后切换方向的步数
    - center_threshold : 距 env_origin 小于此距离视为"到顶"
    """
    forward_speed: float = 1.0
    switch_steps: int = 750
    center_threshold: float = 0.8

    ranges: UniformVelocityCommandCfg.Ranges = field(
        default_factory=lambda: UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=None,
        )
    )

    def build(self, env: ManagerBasedRlEnv) -> StairVelocityCommand:
        return StairVelocityCommand(self, env)

    def __post_init__(self):
        self.rel_standing_envs = 0.0
        self.rel_heading_envs  = 0.0
        self.rel_world_envs    = 0.0
        self.rel_forward_envs  = 0.0
        self.heading_command   = False
        self.init_velocity_prob = 0.0
        self.heading_control_stiffness = 2.0