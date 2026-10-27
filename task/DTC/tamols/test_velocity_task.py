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
from types import SimpleNamespace

import numpy as np
import torch

from myrobot_mjlab.task.DTC.tamols.debug_viz import draw_command_footholds
from myrobot_mjlab.task.DTC.tamols.dtc_command import (
    DTCFootholdCommand,
    DTCFootholdCommandCfg,
)
from myrobot_mjlab.task.DTC.tamols.solver_backend import ca as _ca_unused  # noqa: F401


class TestDTCVelocityCfg(unittest.TestCase):
    def test_dtc_baseline_paper_interface(self):
        from myrobot_mjlab.task.DTC.config.dtc_baseline import (
            my_go2_dtc_baseline_env_cfg,
            my_go2_dtc_baseline_flat_env_cfg,
        )

        variants = (
            ("stair", my_go2_dtc_baseline_env_cfg, "generator"),
            ("flat", my_go2_dtc_baseline_flat_env_cfg, "plane"),
        )
        for name, factory, terrain_type in variants:
            with self.subTest(name=name):
                cfg = factory(play=False)
                self.assertEqual(cfg.scene.terrain.terrain_type, terrain_type)
                actor = cfg.observations["actor"].terms
                critic = cfg.observations["critic"].terms
                self.assertIn("tamols_footholds_xy", actor)
                self.assertIn("tamols_time_left", actor)
                self.assertIn("tamols_path_height", actor)
                self.assertNotIn("height_scan", actor)
                self.assertNotIn("tamols_gait_phase", actor)
                self.assertIn("tamols_base_lin_acc_desired", critic)
                self.assertIn("tamols_base_ang_acc_desired", critic)
                # Paper gesture/main reward set, no custom goal/air-time/foot
                # clearance.
                for key in (
                    "air_time",
                    "upright",
                    "pose",
                    "foot_swing_height",
                    "reach_peak_bonus",
                    "stall_penalty",
                ):
                    self.assertNotIn(key, cfg.rewards)
                self.assertAlmostEqual(
                    cfg.rewards["base_pose_tracking_pos"].params["sigma"], 1200.0
                )
                self.assertAlmostEqual(
                    cfg.rewards["base_pose_tracking_rot"].params["sigma"], 90.0
                )
                self.assertAlmostEqual(
                    cfg.rewards["base_pose_tracking_acc_pos"].params["sigma"], 0.05
                )
                self.assertAlmostEqual(
                    cfg.rewards["base_pose_tracking_acc_rot"].params["sigma"], 0.005
                )
                self.assertTrue(
                    cfg.rewards["base_pose_tracking_pos"].params["squared"]
                )
                self.assertAlmostEqual(
                    cfg.rewards["action_consistency"].weight, 20.0
                )
                self.assertAlmostEqual(cfg.rewards["joint_power"].weight, 0.025)
                foothold_cmd = cfg.commands["Tamols_footholds"]
                self.assertTrue(foothold_cmd.replan_per_foot_touchdown)
                self.assertAlmostEqual(foothold_cmd.gait_cfg.stride_duration, 0.93)
                self.assertAlmostEqual(foothold_cmd.cell_size, 0.04)
                self.assertIn("base_contact", cfg.terminations)
                if name == "stair":
                    self.assertIn("out_of_patch_bounds", cfg.terminations)
                else:
                    self.assertNotIn("out_of_patch_bounds", cfg.terminations)
                    self.assertEqual(cfg.curriculum, {})

    def test_stair_uses_target_direction_and_in_place_gait(self):
        from myrobot_mjlab.task.DTC.config.env_cfg import my_go2_stair_env_cfg

        cfg = my_go2_stair_env_cfg(play=False)
        # 楼梯速度方向指向目标点, 大小仍为 forward_speed.
        self.assertFalse(cfg.commands["twist"].forward_only)
        self.assertTrue(cfg.commands["twist"].stop_at_target)
        foothold_cmd = cfg.commands["Tamols_footholds"]
        self.assertTrue(foothold_cmd.in_place_gait_enable)
        self.assertFalse(foothold_cmd.replan_after_all_four_touchdowns)
        self.assertAlmostEqual(foothold_cmd.gait_cfg.stride_duration, 0.8)

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
            self.assertFalse(cmd.debug_vis, name)  # 训练默认关
            self.assertEqual(cmd.sensor_name, "terrain_scan_map", name)
            # 训练态全部调试可视化关闭 (sensor + command)
            for s in cfg.scene.sensors or ():
                if hasattr(s, "debug_vis"):
                    self.assertFalse(s.debug_vis, f"{name}/{s.name}")
            for c in cfg.commands.values():
                self.assertFalse(c.debug_vis, f"{name}/{c}")
        # play 变体: 命令项存在且调试可视化统一打开
        cfg_play = my_go2_flat_env_cfg(play=True)
        self.assertIn("Tamols_footholds", cfg_play.commands)
        self.assertTrue(cfg_play.commands["Tamols_footholds"].debug_vis)
        self.assertTrue(cfg_play.commands["twist"].debug_vis)
        for s in cfg_play.scene.sensors or ():
            if hasattr(s, "debug_vis"):
                # terrain_scan_map 是 TAMOLS 内部 325 线扫描, play 里故意
                # 关闭其 debug 射线以降低 Viser 图元数量.
                if s.name == "terrain_scan_map":
                    self.assertFalse(s.debug_vis)
                else:
                    self.assertTrue(s.debug_vis, f"play/{s.name}")

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
    """play viewer 显示 TAMOLS optimal/landing/nominal 三组落点."""

    def test_subclass_only_overrides_debug_vis(self):
        # 核心规划方法不得覆写; 生命周期钩子 (_update_metrics/
        # _resample_command) 允许覆写但必须 super() 转发 (第一步的
        # 摆动观测扩展, 行为保留).
        self.assertNotIn("_update_command", DTCFootholdCommand.__dict__)
        import inspect

        for method in ("_update_metrics", "_resample_command"):
            if method in DTCFootholdCommand.__dict__:
                body = inspect.getsource(DTCFootholdCommand.__dict__[method])
                self.assertIn(f"super().{method}", body, method)
        self.assertIn("_debug_vis_impl", DTCFootholdCommand.__dict__)

    def test_debug_visualizer_shows_planned_footholds(self):
        cmd = object.__new__(DTCFootholdCommand)
        cmd._env = SimpleNamespace(num_envs=1, device="cpu")
        cmd.cfg = SimpleNamespace(
            dot_radius=0.035,
            dot_color=(1.0, 0.05, 0.05, 0.95),
            show_landing_dots=True,
            landing_dot_color=(1.0, 0.55, 0.0, 0.95),
            show_nominal_dots=True,
            nominal_dot_color=(0.2, 0.6, 1.0, 0.7),
        )
        cmd.optimal_footholds = torch.zeros(1, 4, 3)
        cmd.landing_footholds = torch.ones(1, 4, 3)
        cmd.nominal_footholds = torch.full((1, 4, 3), 2.0)
        viz = _RecordingVisualizer()
        DTCFootholdCommand._debug_vis_impl(cmd, viz)
        self.assertEqual(len(viz.spheres), 12)
        labels = {s["label"] for s in viz.spheres}
        self.assertIn("tamols_opt_FL_0", labels)
        self.assertIn("tamols_land_FL_0", labels)
        self.assertIn("tamols_nom_FL_0", labels)


class _FakeDropdown:
    def __init__(self, options, initial_value):
        self.options = list(options)
        self.value = initial_value
        self._cb = None

    def on_update(self, cb):
        self._cb = cb

    def select(self, value):
        """模拟用户在面板上选择."""
        self.value = value
        self._cb(None)


class _FakeFolder:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeGui:
    def __init__(self):
        self.folders = []
        self.dropdowns = []

    def add_folder(self, label):
        self.folders.append(label)
        return _FakeFolder()

    def add_dropdown(self, label, options, initial_value):
        dd = _FakeDropdown(options, initial_value)
        self.dropdowns.append((label, dd))
        return dd


class _FakeServer:
    def __init__(self):
        self.gui = _FakeGui()


def _shim_command(terrain):
    """构造只带 _env.scene.terrain 的 DTCFootholdCommand shim."""
    cmd = object.__new__(DTCFootholdCommand)
    cmd._env = SimpleNamespace(scene=SimpleNamespace(terrain=terrain))
    return cmd


class TestTerrainDifficultyGui(unittest.TestCase):
    """viewer 课程难度下拉框 (复制 track/AllTrackingCommand 方案)."""

    def _stair_terrain(self, num_levels=10, num_envs=8, cur_level=0):
        return SimpleNamespace(
            terrain_origins=torch.zeros(num_levels, 4, 3),
            terrain_levels=torch.full((num_envs,), cur_level, dtype=torch.long),
        )

    def test_dropdown_created_and_writes_level(self):
        terrain = self._stair_terrain()
        cmd = _shim_command(terrain)
        server = _FakeServer()
        actions = []
        cmd.create_gui(
            "Tamols_footholds", server, get_env_idx=lambda: 0,
            request_action=actions.append,
        )
        # 10 级 + Random
        self.assertEqual(len(server.gui.dropdowns), 1)
        _, dd = server.gui.dropdowns[0]
        self.assertEqual(len(dd.options), 11)
        self.assertEqual(dd.value, dd.options[0])
        # 选择 Level 5 -> 写入 env 并请求 RESET
        dd.select(dd.options[5])
        self.assertEqual(cmd._env.specified_terrain_level, 5)
        self.assertEqual(actions, ["RESET"])
        # 选择 Random
        dd.select(dd.options[-1])
        self.assertEqual(cmd._env.specified_terrain_level, "random")
        self.assertEqual(actions, ["RESET", "RESET"])

    def test_initial_value_from_env_or_terrain(self):
        # 已有 specified_terrain_level -> 下拉初值跟随
        terrain = self._stair_terrain()
        cmd = _shim_command(terrain)
        cmd._env.specified_terrain_level = 7
        server = _FakeServer()
        cmd.create_gui("t", server, get_env_idx=lambda: 0)
        _, dd = server.gui.dropdowns[0]
        self.assertEqual(dd.value, dd.options[7])
        # specified="random" -> 初值为最后一项
        cmd2 = _shim_command(self._stair_terrain())
        cmd2._env.specified_terrain_level = "random"
        server2 = _FakeServer()
        cmd2.create_gui("t", server2, get_env_idx=lambda: 0)
        self.assertEqual(
            server2.gui.dropdowns[0][1].value,
            server2.gui.dropdowns[0][1].options[-1],
        )
        # 无 specified -> 跟随当前 env 的 terrain_levels
        cmd3 = _shim_command(self._stair_terrain(cur_level=3))
        server3 = _FakeServer()
        cmd3.create_gui("t", server3, get_env_idx=lambda: 0)
        self.assertEqual(server3.gui.dropdowns[0][1].value,
                         server3.gui.dropdowns[0][1].options[3])

    def test_flat_terrain_skips_gui(self):
        # plane 地形无 terrain_origins -> 不创建控件
        cmd = _shim_command(SimpleNamespace())
        server = _FakeServer()
        cmd.create_gui("t", server, get_env_idx=lambda: 0)
        self.assertEqual(server.gui.dropdowns, [])
        # 无 terrain 实体也不报错
        cmd2 = _shim_command(None)
        server2 = _FakeServer()
        cmd2.create_gui("t", server2, get_env_idx=lambda: 0)
        self.assertEqual(server2.gui.dropdowns, [])

    def test_stair_reset_consumes_specified_level(self):
        """链路验证: stair 变体的 reset 事件确实消费 specified_terrain_level."""
        import inspect

        from myrobot_mjlab.config.event.stair_reset import (
            reset_root_state_from_stair,
        )
        from myrobot_mjlab.task.DTC.config.env_cfg import my_go2_stair_env_cfg

        src = inspect.getsource(reset_root_state_from_stair)
        self.assertIn("specified_terrain_level", src)
        cfg = my_go2_stair_env_cfg()
        self.assertEqual(
            cfg.events["reset_base"].func, reset_root_state_from_stair
        )


class TestDTCFootholdCommandBuild(unittest.TestCase):
    def test_cfg_overrides_build(self):
        """cfg.build 必须返回 DTC 子类 (否则 create_gui/三色绘制/
        摆动观测在真实 env 里全部静默失效 —— 父类 build 返回
        TamolsMotionCommand)."""
        import inspect

        self.assertIn("build", DTCFootholdCommandCfg.__dict__)
        src = inspect.getsource(DTCFootholdCommandCfg.build)
        self.assertIn("DTCFootholdCommand(", src)


class TestDTCRegistration(unittest.TestCase):
    def test_dtc_tasks_registered(self):
        """真实注册链: import DTC.config (触发父包与 entry-points 加载)."""
        import myrobot_mjlab.task.DTC.config as _dtc  # noqa: F401
        from mjlab.tasks.registry import list_tasks

        tasks = list_tasks()
        for expected in (
            "Mjlab-DTC-Velocity-Flat-Unitree-Go2",
            "Mjlab-DTC-Velocity-StairFromFlat-Unitree-Go2",
            "Mjlab-DTC-Velocity-StairDTCBaseline-Unitree-Go2",
            "Mjlab-DTC-Velocity-FlatDTCBaseline-Unitree-Go2",
            "Mjlab-DTC-Velocity-TireArray-Unitree-Go2",
        ):
            self.assertIn(expected, tasks)


if __name__ == "__main__":
    unittest.main()
