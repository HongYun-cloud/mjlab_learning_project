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

    ``multi_teacher=False``: ``terrain_key`` selects one teacher checkpoint.
    ``multi_teacher=True``: ``teachers`` is keyed by sub-terrain name and each
    env is routed to the teacher matching its ``terrain_types`` column.  The
    student model is shared across all envs.
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
    multi_teacher: bool = False
    """If True, load all ``teachers`` and route per env by terrain column."""

    experiment_name: str = "go2_velocity"
    save_interval: int = 50
    num_steps_per_env: int = 24
    max_iterations: int = 10_000


def my_go2_multi_teacher_distill_cfg(
    terrain_key: str,
    teachers: dict[str, str] | None = None,
    teacher_checkpoint: str | None = None,
    multi_teacher: bool = False,
) -> MultiTeacherDistillationRunnerCfg:
    """Create a student task config.

    With ``multi_teacher=False`` this is the legacy single-teacher setup.
    With ``multi_teacher=True`` the ``teachers`` dict is keyed by sub-terrain
    name and every env is routed to its matching teacher.
    """
    cfg = MultiTeacherDistillationRunnerCfg()
    cfg.terrain_key = terrain_key
    cfg.teachers = dict(teachers or TEACHER_CHECKPOINTS)
    cfg.multi_teacher = multi_teacher

    if multi_teacher:
        # Teachers are external; never restore them from the student
        # checkpoint, and do not use the single-checkpoint override.
        cfg.teacher_checkpoint = None
        cfg.algorithm.class_name = (
            "myrobot_mjlab.task.student.config.routed_distillation:"
            "RoutedDistillation"
        )
    else:
        if teacher_checkpoint is None:
            teacher_checkpoint = cfg.teachers.get(terrain_key) or None
        cfg.teacher_checkpoint = teacher_checkpoint

    return cfg
