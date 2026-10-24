from __future__ import annotations

from dataclasses import dataclass, field
import torch


@dataclass
class GaitSchedulerCfg:
    """Configuration for periodic gait schedule."""
    stride_duration: float = 0.60
    """Full gait cycle duration in seconds (T_stride)."""

    duty_factor: float = 0.50
    """Duty factor D: fraction of stride spent in stance phase (0.0 to 1.0)."""

    phase_offsets: tuple[float, float, float, float] = (0.0, 0.5, 0.5, 0.0)
    """Phase offsets for [FL, FR, RL, RR]. (0.0, 0.5, 0.5, 0.0) = Diagonal Trot."""


class GaitScheduler:
    """Periodic gait scheduler tracking phase, contact schedule, and remaining time.
    
    Adheres to DTC / TAMOLS formulation:
    - Trot gait by default (FL+RR pair alternating with FR+RL pair);
    - Evaluates per-leg phase phi_i in [0, 1);
    - Determines binary contact plan c_i in {0, 1} (1: Stance, 0: Swing);
    - Computes time left in current phase t_rem, i (seconds).
    """

    def __init__(
        self,
        cfg: GaitSchedulerCfg | None = None,
        num_envs: int = 1,
        device: torch.device | str = "cpu",
    ):
        self.cfg = cfg or GaitSchedulerCfg()
        self.num_envs = num_envs
        self.device = torch.device(device)

        # [4] phase offsets tensor: FL, FR, RL, RR
        self.phase_offsets = torch.tensor(
            self.cfg.phase_offsets, device=self.device, dtype=torch.float32
        )

        # Elapsed time in current episode per environment [num_envs]
        self.current_time = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        # Buffers for outputs: [num_envs, 4]
        self.phases = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)
        self.contact_states = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)
        self.time_left = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)

        self._update_all()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset elapsed time for specified environments."""
        if env_ids is None:
            self.current_time.zero_()
        else:
            self.current_time[env_ids] = 0.0
        self._update_all(env_ids)

    def step(self, dt: float, env_ids: torch.Tensor | None = None) -> None:
        """Advance time by dt and recompute gait states."""
        ids = slice(None) if env_ids is None else env_ids
        self.current_time[ids] += dt
        self._update_all(env_ids)

    def _update_all(self, env_ids: torch.Tensor | None = None) -> None:
        """Compute phase, contact schedule, and time left."""
        ids = slice(None) if env_ids is None else env_ids

        t = self.current_time[ids].unsqueeze(-1)  # [M, 1]
        T = self.cfg.stride_duration
        D = self.cfg.duty_factor

        # 1. Normalized Gait Phase in [0.0, 1.0)
        # phi_i = (t / T + offset_i) % 1.0
        phi = (t / T + self.phase_offsets.unsqueeze(0)) % 1.0  # [M, 4]
        self.phases[ids] = phi

        # 2. Desired Contact State c_i in {0.0, 1.0}
        # In stance if phi < D (c_i = 1.0), in swing if phi >= D (c_i = 0.0)
        is_stance = phi < D
        self.contact_states[ids] = is_stance.float()

        # 3. Time Left in Phase (seconds)
        # Stance: time until liftoff = (D - phi) * T
        # Swing: time until touchdown = (1.0 - phi) * T
        time_left_stance = (D - phi) * T
        time_left_swing = (1.0 - phi) * T
        self.time_left[ids] = torch.where(is_stance, time_left_stance, time_left_swing)

    @property
    def is_swing(self) -> torch.Tensor:
        """[num_envs, 4] boolean mask: True if leg is in swing phase."""
        return self.contact_states < 0.5

    @property
    def is_stance(self) -> torch.Tensor:
        """[num_envs, 4] boolean mask: True if leg is in stance phase."""
        return self.contact_states >= 0.5
