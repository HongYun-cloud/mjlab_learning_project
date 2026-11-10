"""Multi-teacher distillation configuration for the student policy."""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.rl import RslRlBaseRunnerCfg, RslRlModelCfg

from .teachers import TEACHER_CHECKPOINTS


@dataclass
class MultiTeacherDistillationAlgorithmCfg:
    """rsl_rl ``Distillation`` algorithm configuration."""

    class_name: str = "Distillation"
    num_learning_epochs: int = 1
    gradient_length: int = 15
    learning_rate: float = 1.0e-3
    max_grad_norm: float = 1.0
    loss_type: str = "mse"
    optimizer: str = "adam"


@dataclass
class MultiTeacherDistillationRunnerCfg(RslRlBaseRunnerCfg):
    """Student/teacher observation split and teacher registry.

    ``teachers`` maps a terrain task key to a teacher PPO checkpoint.
    ``terrain_key`` selects the teacher used by this student task.  The
    student model is shared by using the same student checkpoint/config across
    tasks; true simultaneous multi-terrain routing is left for the combined
    terrain task.
    """

    class_name: str = "MultiTeacherDistillationRunner"

    student: RslRlModelCfg = field(
        default_factory=lambda: RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        )
    )
    teacher: RslRlModelCfg = field(
        default_factory=lambda: RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            # Keep the PPO actor distribution keys so a teacher checkpoint can
            # be loaded with strict=True.  Teacher inference is deterministic.
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        )
    )
    algorithm: MultiTeacherDistillationAlgorithmCfg = field(
        default_factory=MultiTeacherDistillationAlgorithmCfg
    )

    obs_groups: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "student": ("actor",),
            "teacher": ("critic",),
        }
    )

    teachers: dict[str, str] = field(
        default_factory=lambda: dict(TEACHER_CHECKPOINTS)
    )
    terrain_key: str = ""
    teacher_checkpoint: str | None = None

    experiment_name: str = "go2_velocity"
    save_interval: int = 50
    num_steps_per_env: int = 24
    max_iterations: int = 10_000


def my_go2_multi_teacher_distill_cfg(
    terrain_key: str,
    teachers: dict[str, str] | None = None,
    teacher_checkpoint: str | None = None,
) -> MultiTeacherDistillationRunnerCfg:
    """Create a student task config for one terrain teacher."""
    cfg = MultiTeacherDistillationRunnerCfg()
    cfg.terrain_key = terrain_key
    cfg.teachers = dict(teachers or TEACHER_CHECKPOINTS)
    if teacher_checkpoint is None:
        teacher_checkpoint = cfg.teachers.get(terrain_key) or None
    cfg.teacher_checkpoint = teacher_checkpoint
    return cfg
