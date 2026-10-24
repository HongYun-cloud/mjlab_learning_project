from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from .rl_cfg import my_go2_ppo_runner_cfg
from .env_cfg import my_go2_flat_env_cfg
from .env_cfg import my_go2_stair_env_cfg
from .env_cfg import my_go2_tire_env_cfg
from .env_cfg import my_go2_platform_env_cfg

register_mjlab_task(
    task_id="Mjlab-Tracking-Flat-Unitree-Go2",
    env_cfg=my_go2_flat_env_cfg(),
    play_env_cfg=my_go2_flat_env_cfg(play=True),
    rl_cfg=my_go2_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tracking-Unitree-Go2",
    env_cfg=my_go2_stair_env_cfg(),
    play_env_cfg=my_go2_stair_env_cfg(play=True),
    rl_cfg=my_go2_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tracking-TireArray-Unitree-Go2",
    env_cfg=my_go2_tire_env_cfg(),
    play_env_cfg=my_go2_tire_env_cfg(play=True),
    rl_cfg=my_go2_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tracking-Platform-Unitree-Go2",
    env_cfg=my_go2_platform_env_cfg(),
    play_env_cfg=my_go2_platform_env_cfg(play=True),
    rl_cfg=my_go2_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)