"""DTC 速度任务配置 (基于 task/velocity 的副本 + TAMOLS 落足点可视化).

任务 id 加 DTC 前缀以避免与 velocity 包的同名注册冲突 (import 链上
myrobot_mjlab/__init__ 会先注册 velocity 的同名任务).
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from .rl_cfg import my_go2_ppo_runner_cfg
from .env_cfg import my_go2_flat_env_cfg
from .env_cfg import my_go2_stair_env_cfg
from .env_cfg import my_go2_tire_env_cfg
from .env_cfg import my_go2_platform_env_cfg
from .dtc_baseline import (
    my_go2_dtc_baseline_env_cfg,
    my_go2_dtc_baseline_flat_env_cfg,
)

register_mjlab_task(
    task_id="Mjlab-DTC-Velocity-Flat-Unitree-Go2",
    env_cfg=my_go2_flat_env_cfg(),
    play_env_cfg=my_go2_flat_env_cfg(play=True),
    rl_cfg=my_go2_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-DTC-Velocity-StairFromFlat-Unitree-Go2",
    env_cfg=my_go2_stair_env_cfg(),
    play_env_cfg=my_go2_stair_env_cfg(play=True),
    rl_cfg=my_go2_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-DTC-Velocity-TireArray-Unitree-Go2",
    env_cfg=my_go2_tire_env_cfg(),
    play_env_cfg=my_go2_tire_env_cfg(play=True),
    rl_cfg=my_go2_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-DTC-Velocity-Platform-Unitree-Go2",
    env_cfg=my_go2_platform_env_cfg(),
    play_env_cfg=my_go2_platform_env_cfg(play=True),
    rl_cfg=my_go2_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)


register_mjlab_task(
    task_id="Mjlab-DTC-Velocity-StairDTCBaseline-Unitree-Go2",
    env_cfg=my_go2_dtc_baseline_env_cfg(),
    play_env_cfg=my_go2_dtc_baseline_env_cfg(play=True),
    rl_cfg=my_go2_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)


register_mjlab_task(
    task_id="Mjlab-DTC-Velocity-FlatDTCBaseline-Unitree-Go2",
    env_cfg=my_go2_dtc_baseline_flat_env_cfg(),
    play_env_cfg=my_go2_dtc_baseline_flat_env_cfg(play=True),
    rl_cfg=my_go2_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
