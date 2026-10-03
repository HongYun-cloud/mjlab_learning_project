from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass, field

import torch

from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.managers.command_manager import CommandTermCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class StairVelocityCommand(UniformVelocityCommand):
    cfg: StairVelocityCommandCfg

    def __init__(self, cfg: StairVelocityCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        self.going_down = torch.ones(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self._step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._switch_steps = cfg.switch_steps
        self._terrain_size = cfg.terrain_size
        self._center_threshold = cfg.center_threshold

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        self.going_down[env_ids] = True
        self._step_count[env_ids] = 0

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        extras = super().reset(env_ids)
        if isinstance(env_ids, torch.Tensor):
            self.going_down[env_ids] = ~self.going_down[env_ids]
            self._step_count[env_ids] = 0
        elif env_ids == slice(None):
            self.going_down[:] = ~self.going_down[:]
            self._step_count[:] = 0
        return extras

    def compute(self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        super().compute(dt, env_ids)

        if env_ids is None:
            env_ids = slice(None)

        robot_pos_w = self.robot.data.root_link_pos_w
        env_origins = self._env.scene.env_origins

        rel_pos = robot_pos_w[:, :2] - env_origins[:, :2]
        dist_from_center = torch.norm(rel_pos, dim=-1)
        heading_to_center = torch.atan2(-rel_pos[:, 1], -rel_pos[:, 0])

        if isinstance(env_ids, torch.Tensor):
            self._step_count[env_ids] += 1
            at_center = dist_from_center[env_ids] < self._center_threshold
            switch_mask = (self._step_count[env_ids] >= self._switch_steps) | ((~self.going_down[env_ids]) & at_center)
            if switch_mask.any():
                switch_ids = env_ids[switch_mask]
                self.going_down[switch_ids] = ~self.going_down[switch_ids]
                self._step_count[switch_ids] = 0
        else:
            self._step_count += 1
            at_center = dist_from_center < self._center_threshold
            switch_mask = (self._step_count >= self._switch_steps) | ((~self.going_down) & at_center)
            if switch_mask.any():
                self.going_down[switch_mask] = ~self.going_down[switch_mask]
                self._step_count[switch_mask] = 0

        # 始终 body-frame 前进
        speed = self.cfg.forward_speed
        self.vel_command_b[:, 0] = speed
        self.vel_command_b[:, 1] = 0.0
        self.vel_command_b[:, 2] = 0.0

        # heading 指向目标方向：下楼朝外，上楼朝中心
        if self.going_down.any():
            down_ids = self.going_down.nonzero(as_tuple=False).flatten()
            away_dir = torch.atan2(rel_pos[down_ids, 1], rel_pos[down_ids, 0])
            center_mask = dist_from_center[down_ids] < 0.1
            if center_mask.any():
                away_dir[center_mask] = torch.rand_like(away_dir[center_mask]) * 2 * torch.pi
            self.heading_target[down_ids] = away_dir

        if (~self.going_down).any():
            up_ids = (~self.going_down).nonzero(as_tuple=False).flatten()
            self.heading_target[up_ids] = heading_to_center[up_ids]

        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0


@dataclass(kw_only=True)
class StairVelocityCommandCfg(UniformVelocityCommandCfg):
    forward_speed: float = 1.0
    switch_steps: int = 400
    terrain_size: float = 8.0
    center_threshold: float = 0.5
    ranges: UniformVelocityCommandCfg.Ranges = field(
        default_factory=lambda: UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-3.14, 3.14),
        )
    )

    def build(self, env: ManagerBasedRlEnv) -> StairVelocityCommand:
        return StairVelocityCommand(self, env)

    def __post_init__(self):
        self.rel_standing_envs = 0.0
        self.rel_heading_envs = 1.0
        self.rel_world_envs = 0.0
        self.rel_forward_envs = 0.0
        self.heading_command = True
        self.heading_control_stiffness = 2.0