"""Multi-teacher distillation runner.

The student model is shared across student tasks.  Each task selects a teacher
checkpoint from the ``teachers`` registry using its ``terrain_key``.  For the
current per-terrain tasks only one teacher is active at a time; when the
terrains are merged into one environment, this runner can be extended to route
per-env teacher ids to simultaneous teacher models.
"""

from __future__ import annotations

from rsl_rl.runners import DistillationRunner


class MultiTeacherDistillationRunner(DistillationRunner):
    """Distillation runner that loads a teacher from a terrain registry."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
        self._terrain_key = str(train_cfg.get("terrain_key", ""))
        teachers = train_cfg.get("teachers") or {}
        teacher_checkpoint = train_cfg.get("teacher_checkpoint")
        if not teacher_checkpoint:
            teacher_checkpoint = teachers.get(self._terrain_key)
        self._teacher_checkpoint = teacher_checkpoint

        # rsl_rl MLPModel does not accept the optional mjlab model fields when
        # they are None.  Strip them the same way MjlabOnPolicyRunner does.
        for key in ("student", "teacher"):
            if key in train_cfg:
                for opt in ("cnn_cfg", "distribution_cfg"):
                    if train_cfg[key].get(opt) is None:
                        train_cfg[key].pop(opt, None)
                if train_cfg[key].get("rnn_type") is None:
                    for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
                        train_cfg[key].pop(opt, None)

        super().__init__(env, train_cfg, log_dir, device)

        if self._teacher_checkpoint:
            print(
                f"[MultiTeacher] loading teacher for terrain "
                f"'{self._terrain_key}': {self._teacher_checkpoint}"
            )
            self.load(
                self._teacher_checkpoint,
                load_cfg={"teacher": True},
                strict=True,
                map_location=device,
            )
        else:
            print(
                f"[MultiTeacher] no teacher checkpoint configured for terrain "
                f"'{self._terrain_key}'. Fill task/student/config/teachers.py "
                "or pass --agent.teacher-checkpoint."
            )

    def save(self, path: str, infos: dict | None = None) -> None:
        env_state = {
            "common_step_counter": self.env.unwrapped.common_step_counter,
        }
        infos = {**(infos or {}), "env_state": env_state}
        super().save(path, infos)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        # mjlab play.py calls load_cfg={"actor": True}; for a distillation
        # checkpoint the policy lives in the student state dict.
        if load_cfg == {"actor": True}:
            load_cfg = {"student": True}

        infos = super().load(path, load_cfg, strict, map_location)
        if infos and "env_state" in infos:
            env_state = infos["env_state"]
            if "common_step_counter" in env_state:
                self.env.unwrapped.common_step_counter = env_state[
                    "common_step_counter"
                ]
        return infos
