"""Distillation with deterministic student rollouts.

rsl_rl's stock ``Distillation.act`` samples the student action with
``stochastic_output=True``.  The distribution std is initialized to 1.0 and is
not part of the behavior-cloning loss, so it stays at 1.0 during training.
That injects a large amount of action noise into every rollout, which makes
the student fall early and limits the state distribution it learns from.

For teacher-student distillation the environment should follow the student's
current mean policy; the teacher is already deterministic.  This subclass only
changes the rollout action to the distribution mean.  The update/loss code is
unchanged.
"""

from __future__ import annotations

from rsl_rl.algorithms import Distillation


class DeterministicDistillation(Distillation):
    """Distillation that rolls out the student's mean action."""

    def act(self, obs):
        # Update the distribution so the runner can still log output_std, but
        # roll out the deterministic mean action.
        self.student(obs, stochastic_output=True)
        self.transition.actions = self.student.output_mean.detach()
        self.transition.privileged_actions = self.teacher(obs).detach()
        self.transition.observations = obs
        return self.transition.actions

    def save(self) -> dict:
        # Teachers are external, so keep the student checkpoint small.  This
        # also keeps the format compatible with RoutedDistillation checkpoints.
        return {
            "student_state_dict": self._raw_student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def load(
        self,
        loaded_dict: dict,
        load_cfg: dict | None = None,
        strict: bool = True,
    ) -> bool:
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
        if load_cfg.get("teacher"):
            self._raw_teacher.load_state_dict(
                loaded_dict.get("teacher_state_dict")
                or loaded_dict["actor_state_dict"],
                strict=strict,
            )
            self.teacher_loaded = True
        if load_cfg.get("optimizer") and "optimizer_state_dict" in loaded_dict:
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        # Teachers are loaded by MultiTeacherDistillationRunner from the
        # registry, not from the student checkpoint.
        return load_cfg.get("iteration", False)
