"""Multi-teacher distillation runner.

Two modes are supported:

- legacy single-teacher mode (``multi_teacher=False``): pick one checkpoint
  from the ``teachers`` registry by ``terrain_key``;
- per-env routing mode (``multi_teacher=True``): load every registry entry,
  keyed by sub-terrain name, and route each env to the teacher matching its
  ``terrain_types`` column.

Adding a new teacher only requires a new sub-terrain and a new registry entry;
the routing module and distillation loop do not change.
"""

from __future__ import annotations

import copy

import torch
from rsl_rl.runners import DistillationRunner
from rsl_rl.utils import resolve_callable

from .teacher_group import TeacherGroup


class MultiTeacherDistillationRunner(DistillationRunner):
    """Distillation runner with optional per-env teacher routing.

    ``multi_teacher=False`` keeps the legacy behavior: one teacher selected by
    ``terrain_key``.  ``multi_teacher=True`` loads every checkpoint in
    ``teachers`` (keyed by sub-terrain name) and routes each env to the
    teacher matching its terrain column.
    """

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
        self._multi_teacher = bool(train_cfg.get("multi_teacher", False))
        self._terrain_key = str(train_cfg.get("terrain_key", ""))
        self._teachers = dict(train_cfg.get("teachers") or {})

        teacher_checkpoint = train_cfg.get("teacher_checkpoint")
        if not teacher_checkpoint:
            teacher_checkpoint = self._teachers.get(self._terrain_key)
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

        # Keep a pristine copy before construct_algorithm pops class_name /
        # distribution class_name from the live config.
        self._teacher_template = copy.deepcopy(train_cfg.get("teacher", {}))

        super().__init__(env, train_cfg, log_dir, device)

        if self._multi_teacher:
            if log_dir is not None:
                self._install_teacher_group(env, train_cfg, device)
            else:
                # Playback only needs the student policy.
                print("[MultiTeacher] play mode: skipping teacher loading")
                self.alg.teacher_loaded = True
        elif self._teacher_checkpoint:
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

    def _install_teacher_group(self, env, train_cfg: dict, device: str) -> None:
        """Build one teacher per sub-terrain and install a routing module."""
        if not self._teachers:
            raise ValueError(
                "multi_teacher=True but 'teachers' is empty. "
                "Fill task/student/config/teachers.py."
            )

        obs = env.get_observations()
        template = copy.deepcopy(self._teacher_template)
        teacher_cls = resolve_callable(template.pop("class_name"))

        teachers = {}
        for key, path in self._teachers.items():
            if not path:
                raise ValueError(
                    f"No teacher checkpoint configured for terrain '{key}'. "
                    "Fill task/student/config/teachers.py."
                )

            cfg = copy.deepcopy(template)
            model = teacher_cls(
                obs,
                train_cfg["obs_groups"],
                "teacher",
                env.num_actions,
                **cfg,
            )

            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            state = ckpt.get("actor_state_dict")
            if state is None:
                raise KeyError(
                    f"Teacher checkpoint '{path}' has no 'actor_state_dict'. "
                    "Expected a PPO teacher checkpoint."
                )
            model.load_state_dict(state, strict=True)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            teachers[key] = model
            print(f"[MultiTeacher] loaded teacher '{key}': {path}")

        terrain = getattr(env.unwrapped.scene, "terrain", None)
        generator = getattr(
            getattr(terrain, "cfg", None), "terrain_generator", None
        )
        if terrain is None or generator is None:
            raise ValueError(
                "multi_teacher=True requires a terrain generator with "
                "per-env terrain_types."
            )

        sub_keys = list(generator.sub_terrains.keys())
        if set(sub_keys) != set(teachers):
            raise ValueError(
                "Teacher keys must match sub_terrains keys. "
                f"sub_terrains={sub_keys}, teachers={list(teachers.keys())}"
            )

        id_to_key = {i: key for i, key in enumerate(sub_keys)}

        def id_getter() -> torch.Tensor:
            # Read every forward so per-episode terrain randomization works.
            return terrain.terrain_types

        group = TeacherGroup(teachers, id_to_key, id_getter).to(device)
        self.alg.teacher = group
        self.alg._raw_teacher = group
        self.alg.teacher_loaded = True
        print(
            f"[MultiTeacher] routing {len(teachers)} teachers by terrain_types: "
            f"{id_to_key}"
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
