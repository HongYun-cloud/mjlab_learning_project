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
import os
from pathlib import Path

import torch
import wandb
from rsl_rl.runners import DistillationRunner
from rsl_rl.utils import resolve_callable

from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata

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

        if log_dir is None:
            # Playback only needs the student policy; teachers are training-only.
            print("[MultiTeacher] play mode: skipping teacher loading")
            self.alg.teacher_loaded = True
        elif self._multi_teacher:
            self._install_teacher_group(env, train_cfg, device)
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
            raise ValueError(
                f"No teacher checkpoint configured for terrain "
                f"'{self._terrain_key}'. Fill "
                "task/student/config/teachers.py or pass "
                "--agent.teacher-checkpoint."
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

    @staticmethod
    def _get_export_paths(checkpoint_path: str) -> tuple[Path, str, Path]:
        """Resolve ONNX export paths from a checkpoint path."""
        export_dir = Path(checkpoint_path).parent
        filename = f"{export_dir.name}.onnx"
        return export_dir, filename, export_dir / filename

    def export_policy_to_onnx(
        self, path: str, filename: str = "policy.onnx", verbose: bool = False
    ) -> None:
        """Export the student policy to ONNX."""
        onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
        onnx_model.to("cpu")
        onnx_model.eval()
        os.makedirs(path, exist_ok=True)
        torch.onnx.export(
            onnx_model,
            onnx_model.get_dummy_inputs(),  # type: ignore[operator]
            os.path.join(path, filename),
            export_params=True,
            opset_version=18,
            verbose=verbose,
            input_names=onnx_model.input_names,  # type: ignore[arg-type]
            output_names=onnx_model.output_names,  # type: ignore[arg-type]
            dynamic_axes={},
            dynamo=False,
        )

    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
    ) -> None:
        super().learn(num_learning_iterations, init_at_random_ep_len)
        # Export ONNX once, after the final update, instead of on every
        # periodic checkpoint save.
        self.export_final_onnx()

    def export_final_onnx(self) -> None:
        """Export the final student policy to the run directory."""
        log_dir = getattr(self.logger, "log_dir", None)
        if log_dir is None:
            return

        policy_dir = Path(log_dir)
        filename = f"{policy_dir.name}.onnx"
        onnx_path = policy_dir / filename
        try:
            self.export_policy_to_onnx(str(policy_dir), filename)
            run_name: str = (
                wandb.run.name
                if self.logger.logger_type in ("wandb", "WandbLogWriter")
                and wandb.run
                else policy_dir.name
            )
            metadata = get_base_metadata(self.env.unwrapped, run_name)
            attach_metadata_to_onnx(str(onnx_path), metadata)
        except Exception as e:
            print(f"[WARN] final ONNX export failed (training continues): {e}")
            return

        print(f"[ONNX] exported final student policy: {onnx_path}")
        if (
            self.logger.logger_type in ("wandb", "WandbLogWriter")
            and self.cfg["upload_model"]
        ):
            try:
                wandb.save(str(onnx_path), base_path=str(policy_dir))
            except Exception as e:
                print(f"[WARN] wandb ONNX upload skipped: {e}")

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
