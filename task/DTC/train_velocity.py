"""DTC 速度任务的训练入口 (速度任务策略正常训练).

与 mjlab 标准 train CLI 相同, 只是先触发 DTC 任务注册::

    python myrobot_mjlab/task/DTC/train_velocity.py \
        Mjlab-DTC-Velocity-Flat-Unitree-Go2 \
        --env.scene.num-envs 2048 --agent.max-iterations 1000 \
        --agent.wandb-project go2 --agent.run-name <run_name>

训练本身不受 TAMOLS 命令项影响 (它不参与该任务的观测/奖励);
checkpoint 可用 play_velocity.py 播放并在 Viser 中观测落足点.
"""

if __name__ == "__main__":
    import myrobot_mjlab.task.DTC.config as _dtc  # noqa: F401 触发任务注册

    from mjlab.scripts.train import main

    main()
