"""Per-env teacher routing for multi-teacher distillation.

The group mimics the subset of the ``MLPModel`` interface that
``rsl_rl.algorithms.Distillation`` uses (``forward``, ``reset``,
``get_hidden_state``, ``detach_hidden_state``, ``eval``), so the stock
distillation loop can stay unchanged.

Routing is based on ``env.scene.terrain.terrain_types``: each env is assigned
one terrain column, and that column selects the teacher.  The id getter is
read at every forward pass, so per-episode terrain randomization also works.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from tensordict import TensorDict


class TeacherGroup(nn.Module):
    """A module that routes each env to its terrain-specific teacher."""

    def __init__(
        self,
        teachers: dict[str, nn.Module],
        id_to_key: dict[int, str],
        id_getter: Callable[[], torch.Tensor],
    ) -> None:
        super().__init__()
        if not teachers:
            raise ValueError("TeacherGroup requires at least one teacher")
        missing = set(id_to_key.values()) - set(teachers)
        if missing:
            raise ValueError(
                f"TeacherGroup is missing checkpoints for: {sorted(missing)}"
            )

        self.teachers = nn.ModuleDict(teachers)
        self.id_to_key = dict(id_to_key)
        self.id_getter = id_getter
        # Non-persistent probe so ``_ids`` works even for parameterless
        # teachers and follows ``.to(device)``.
        self.register_buffer("_device_probe", torch.zeros(1), persistent=False)

    def _ids(self) -> torch.Tensor:
        ids = self.id_getter()
        return ids.to(
            device=self._device_probe.device,
            dtype=torch.long,
        )

    @torch.no_grad()
    def forward(self, obs: TensorDict) -> torch.Tensor:
        ids = self._ids()
        valid_ids = torch.tensor(
            list(self.id_to_key), device=ids.device, dtype=ids.dtype
        )
        if not torch.isin(ids, valid_ids).all():
            raise ValueError(
                f"TeacherGroup: unknown teacher ids {ids.unique().tolist()}, "
                f"expected one of {valid_ids.tolist()}"
            )

        out: torch.Tensor | None = None

        for teacher_id, key in self.id_to_key.items():
            idx = (ids == teacher_id).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue

            # Teachers only consume the privileged ``critic`` group.  Building
            # a small TensorDict avoids copying other groups (e.g. depth).
            sub_obs = TensorDict(
                {"critic": obs["critic"][idx]},
                batch_size=[idx.numel()],
                device=obs.device,
            )
            sub_out = self.teachers[key](sub_obs)

            if out is None:
                out = torch.empty(
                    (obs.batch_size[0], sub_out.shape[-1]),
                    device=sub_out.device,
                    dtype=sub_out.dtype,
                )
            out[idx] = sub_out

        if out is None:
            raise RuntimeError(
                "TeacherGroup: no env matched a teacher id; "
                f"ids={ids.unique().tolist()}, map={self.id_to_key}"
            )
        return out

    def reset(self, dones: torch.Tensor | None = None, hidden_state=None) -> None:
        """Reset teacher state. MLP teachers are stateless; RNN-ready."""
        if dones is None:
            for teacher in self.teachers.values():
                teacher.reset(hidden_state=hidden_state)
            return

        ids = self._ids()
        dones = dones.view(-1)
        for teacher_id, key in self.id_to_key.items():
            mask = ids == teacher_id
            if mask.any():
                self.teachers[key].reset(dones[mask])

    def get_hidden_state(self):
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        if dones is None:
            for teacher in self.teachers.values():
                teacher.detach_hidden_state()
            return

        ids = self._ids()
        dones = dones.view(-1)
        for teacher_id, key in self.id_to_key.items():
            mask = ids == teacher_id
            if mask.any():
                self.teachers[key].detach_hidden_state(dones[mask])
