"""DTC 速度任务的播放入口 (Viser 视图观测 TAMOLS 落足点).

任务注册发生在 ``myrobot_mjlab.task.DTC.config`` 的 import 中 (任务 id
带 DTC 前缀, 与 velocity 包不冲突)。本模块先触发注册, 再转发给 mjlab
标准播放 CLI::

    python myrobot_mjlab/task/DTC/play_velocity.py \
        Mjlab-DTC-Velocity-Flat-Unitree-Go2 \
        --checkpoint-file wandb/.../model_xxx.pt --viewer viser

浏览器页面左侧: **Debug Viz -> 勾选 Enabled**(默认关); 多环境时勾选
All envs。红点 = TAMOLS 重规划落点, 橙点 = 摆动腿当前落点目标,
蓝点 = 名义落点。
"""

if __name__ == "__main__":
    import myrobot_mjlab.task.DTC.config as _dtc  # noqa: F401 触发任务注册

    from mjlab.scripts.play import main

    main()
