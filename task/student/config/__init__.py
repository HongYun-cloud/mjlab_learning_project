"""Student task registration for multi-teacher distillation."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import (
    my_go2_flat_env_cfg,
    my_go2_low_clearance_bar_env_cfg,
    my_go2_platform_env_cfg,
    my_go2_plum_env_cfg,
    my_go2_stair_down_env_cfg,
    my_go2_stair_up_env_cfg,
    my_go2_stones_2rows_env_cfg,
    my_go2_tire_env_cfg,
)
from .multi_teacher import MultiTeacherDistillationRunner
from .rl_cfg import my_go2_multi_teacher_distill_cfg
from .teachers import TEACHER_CHECKPOINTS

_TERRAINS = {
    "Flat": my_go2_flat_env_cfg,
    "StairUp": my_go2_stair_up_env_cfg,
    "StairDown": my_go2_stair_down_env_cfg,
    "Platform": my_go2_platform_env_cfg,
    "PlumBlossom": my_go2_plum_env_cfg,
    "Stones2Rows": my_go2_stones_2rows_env_cfg,
    "TireArray": my_go2_tire_env_cfg,
    "LowClearanceBar": my_go2_low_clearance_bar_env_cfg,
}

for _name, _env_cfg in _TERRAINS.items():
    register_mjlab_task(
        task_id=f"Mjlab-Tracking-{_name}-Student-Unitree-Go2",
        env_cfg=_env_cfg(),
        play_env_cfg=_env_cfg(play=True),
        rl_cfg=my_go2_multi_teacher_distill_cfg(
            terrain_key=_name,
            teachers=TEACHER_CHECKPOINTS,
        ),
        runner_cls=MultiTeacherDistillationRunner,
    )

__all__ = ["_TERRAINS"]
