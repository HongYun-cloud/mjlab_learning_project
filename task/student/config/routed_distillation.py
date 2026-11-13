"""Distillation variant for multi-teacher routing.

Teachers are external experts loaded from the registry.  They are intentionally
NOT stored in the student checkpoint:

- student checkpoints stay small;
- resuming a student never overwrites the teachers selected for this run;
- adding a new terrain only requires a new teacher checkpoint and a new
  registry entry, not a checkpoint format change.
"""

from __future__ import annotations

from rsl_rl.algorithms import Distillation


class RoutedDistillation(Distillation):
    """Distillation with externally managed teachers."""

    def save(self) -> dict:
        return {
            "student_state_dict": self._raw_student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if load_cfg is None:
            load_cfg = {
                "student": True,
                "optimizer": True,
                "iteration": True,
            }

        if load_cfg.get("student"):
            self._raw_student.load_state_dict(
                loaded_dict["student_state_dict"], strict=strict
            )
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])

        # Teachers are loaded by MultiTeacherDistillationRunner from the
        # registry.  Ignore a teacher_state_dict if an older checkpoint has one.
        return load_cfg.get("iteration", False)
