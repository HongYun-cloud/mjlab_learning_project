"""DTC 速度任务配置 + TAMOLS 落足点可视化的测试 (全部在 task/DTC 内).

运行:
    python -m unittest myrobot_mjlab.task.DTC.tamols.test_velocity_task -v

覆盖:
  1. DTC 工厂组装: 三个速度变体的 commands 均含 ``Tamols_footholds``
     (DTCFootholdCommandCfg), debug_vis=True;
  2. 工厂 import 指向 DTC 副本 (不回引 velocity 包, 否则 DTC 命令项
     不会生效);
  3. DTCFootholdCommand 绘制 (shim, 未绑定调用): 红/橙/蓝三色足点,
     坐标取自命令项张量, 规划逻辑未改动 (不覆写 _resample/_update);
  4. 任务注册: ``Mjlab-DTC-Velocity-*`` 出现在 mjlab 注册表
     (真实注册链, 触发 myrobot_mjlab 父包 import)。
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from myrobot_mjlab.task.DTC.tamols.debug_viz import draw_command_footholds
from myrobot_mjlab.task.DTC.tamols.dtc_command import (
    DTCFootholdCommand,
    DTCFootholdCommandCfg,
)
from myrobot_mjlab.task.DTC.tamols.solver_backend import ca as _ca_unused  # noqa: F401


class TestDTCVelocityCfg(unittest.TestCase):
    def test_variants_have_foothold_command(self):
        from myrobot_mjlab.task.DTC.config import (
            my_go2_flat_env_cfg,
            my_go2_stair_env_cfg,
            my_go2_tire_env_cfg,
        )
        from myrobot_mjlab.task.DTC.config.env_cfg import my_go2_rough_env_cfg

        for name, fn in (
            ("rough", my_go2_rough_env_cfg),
            ("flat", my_go2_flat_env_cfg),
            ("stair", my_go2_stair_env_cfg),
            ("tire", my_go2_tire_env_cfg),
        ):
            cfg = fn(play=False)
            cmd = cfg.commands.get("Tamols_footholds")
            self.assertIsInstance(cmd, DTCFootholdCommandCfg, name)
            self.assertTrue(cmd.debug_vis, name)
            self.assertEqual(cmd.sensor_name, "terrain_scan", name)
        # play 变体同样带命令项 (可视化在 play 视图观测)
        cfg_play = my_go2_flat_env_cfg(play=True)
        self.assertIn("Tamols_footholds", cfg_play.commands)

    def test_factory_import_is_dtc_copy(self):
        """DTC/config/env_cfg.py 的工厂必须指向 DTC 副本.

        若回引 velocity 包的工厂, DTC 命令项会被静默绕过."""
        import inspect

        from myrobot_mjlab.task.DTC.config import env_cfg as dtc_env_cfg

        src = inspect.getsource(dtc_env_cfg)
        self.assertIn(
            "from myrobot_mjlab.task.DTC.velocity_env_cfg import "
            "make_velocity_env_cfg",
            src,
        )
        self.assertNotIn(
            "from myrobot_mjlab.task.velocity.velocity_env_cfg import",
            src,
        )


class _RecordingVisualizer:
    def __init__(self):
        self.env_idx = 0
        self.show_all_envs = False
        self.spheres = []

    def get_env_indices(self, num_envs):
        return [self.env_idx] if self.env_idx < num_envs else []

    @property
    def meansize(self):
        return 0.1

    def add_sphere(self, center, radius, color, label=None):
        self.spheres.append(
            {"center": tuple(float(v) for v in center), "label": str(label)}
        )


class TestDTCFootholdCommandDraw(unittest.TestCase):
    """绘制覆写 (shim): 规划逻辑零改动, 只换 debug 绘制."""

    def test_subclass_only_overrides_debug_vis(self):
        # 规划相关方法必须是继承原实现 (无覆写)
        for method in ("_resample_command", "_update_command", "_update_metrics"):
            self.assertEqual(
                DTCFootholdCommand.__dict__.get(method, None),
                None,
                f"{method} 被覆写: 规划逻辑应保持 fetch 原实现",
            )
        self.assertIn("_debug_vis_impl", DTCFootholdCommand.__dict__)

    def test_draw_three_colors_from_tensors(self):
        cmd = object.__new__(DTCFootholdCommand)
        # num_envs 是只读 property (读 self._env.num_envs): shim 注入 _env
        from types import SimpleNamespace

        cmd._env = SimpleNamespace(num_envs=1, device="cpu")
        from types import SimpleNamespace as _SN  # noqa: F401

        cmd.cfg = SimpleNamespace(
            dot_radius=0.035,
            dot_color=(1.0, 0.05, 0.05, 0.95),
            show_landing_dots=True,
            landing_dot_color=(1.0, 0.55, 0.0, 0.95),
            show_nominal_dots=True,
            nominal_dot_color=(0.2, 0.6, 1.0, 0.7),
        )
        base = torch.zeros(1, 4, 3)
        cmd.optimal_footholds = base.clone()
        cmd.optimal_footholds[0, 2] = torch.tensor([-0.21, 0.14, 0.02])  # RL
        cmd.landing_footholds = base.clone()
        cmd.landing_footholds[0, 2] = torch.tensor([-0.22, 0.15, 0.03])
        cmd.nominal_footholds = base.clone()
        cmd.nominal_footholds[0, 2] = torch.tensor([-0.2, 0.13, 0.0])

        viz = _RecordingVisualizer()
        DTCFootholdCommand._debug_vis_impl(cmd, viz)
        labels = {s["label"]: s["center"] for s in viz.spheres}
        self.assertEqual(len(viz.spheres), 12)
        # torch float32 张量转 float64 有 1e-8 级表示误差, 用 atol 比较
        np.testing.assert_allclose(
            labels["tamols_opt_RL_0"], [-0.21, 0.14, 0.02], atol=1e-7
        )
        np.testing.assert_allclose(
            labels["tamols_land_RL_0"], [-0.22, 0.15, 0.03], atol=1e-7
        )
        np.testing.assert_allclose(
            labels["tamols_nom_RL_0"], [-0.2, 0.13, 0.0], atol=1e-7
        )


class TestDTCRegistration(unittest.TestCase):
    def test_dtc_tasks_registered(self):
        """真实注册链: import DTC.config (触发父包与 entry-points 加载)."""
        import myrobot_mjlab.task.DTC.config as _dtc  # noqa: F401
        from mjlab.tasks.registry import list_tasks

        tasks = list_tasks()
        for expected in (
            "Mjlab-DTC-Velocity-Flat-Unitree-Go2",
            "Mjlab-DTC-Velocity-StairFromFlat-Unitree-Go2",
            "Mjlab-DTC-Velocity-TireArray-Unitree-Go2",
        ):
            self.assertIn(expected, tasks)


if __name__ == "__main__":
    unittest.main()
