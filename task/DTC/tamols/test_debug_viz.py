"""DTC/TAMOLS 计划可视化 (debug_viz) 的测试 (假 visualizer, 无需 viewer).

运行:
    python -m unittest myrobot_mjlab.task.DTC.tamols.test_debug_viz -v

覆盖:
  1. draw_plan: 4 个期望落脚点 (红) 坐标 == plan.footholds_w;
  2. 基座轨迹采样点 (青) 数量与坐标 == base_reference_at;
  3. 触地腿的基座->落点箭头; horizon 终点坐标架 (开关);
  4. 未成功计划 / None 不绘制任何图元;
  5. 配置开关逐项生效; label 带腿名与环境 id;
  6. DTCPlanVizObserver 接入 PlanningWorker: 事件触发后 on_plan 更新
     快照, debug_vis 绘制最新计划 (worker->viz 全链路).
"""

from __future__ import annotations

import contextlib
import io
import time
import unittest
from dataclasses import dataclass, field

import numpy as np

from myrobot_mjlab.task.DTC.tamols.debug_viz import (
    DTCPlanVizObserver,
    DTCVizConfig,
    draw_plan,
)
from myrobot_mjlab.task.DTC.tamols.gn_sqp_solver import GNSQPOptions
from myrobot_mjlab.task.DTC.tamols.rl_bridge import (
    PlanningWorker,
    make_default_solve_fn,
)
from myrobot_mjlab.task.DTC.tamols.synthetic_scenarios import (
    CONTACT_NORMAL,
    build_scenario,
)
from myrobot_mjlab.task.DTC.tamols.trajectory_export import (
    export_trajectory,
    touchdown_joint_velocities,
)
from myrobot_mjlab.task.DTC.tamols.rl_bridge import plan_from_report


@dataclass
class RecordingVisualizer:
    env_idx: int = 0
    show_all_envs: bool = False
    spheres: list = field(default_factory=list)
    arrows: list = field(default_factory=list)
    frames: list = field(default_factory=list)

    def get_env_indices(self, num_envs: int):
        if self.show_all_envs:
            return range(num_envs)
        return [self.env_idx] if self.env_idx < num_envs else []

    @property
    def meansize(self) -> float:
        return 0.1

    def add_sphere(self, center, radius, color, label=None):
        self.spheres.append(
            {"center": np.asarray(center, float), "radius": float(radius),
             "color": tuple(float(v) for v in color), "label": str(label)}
        )

    def add_arrow(self, start, end, color, width=0.015, label=None):
        self.arrows.append(
            {"start": np.asarray(start, float), "end": np.asarray(end, float),
             "label": str(label)}
        )

    def add_frame(self, position, rotation_matrix, scale=0.3, label=None, **_):
        self.frames.append(
            {"pos": np.asarray(position, float), "label": str(label)}
        )

    def add_ghost_mesh(self, *a, **k):
        pass


def _make_plan():
    nlp, (coeffs, desired), _ = build_scenario("crawl_flat", CONTACT_NORMAL)
    nlp.add_initial_state_constraints()
    nlp.add_junction_constraints()
    nlp.add_dynamic_stability_constraints()
    nlp.add_kinematic_contact_constraints()
    nlp.add_objectives()
    nlp.finalize()
    blocks = [np.asarray(c, float) for c in coeffs] + [
        np.asarray(desired, float), np.zeros(3)
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        report = solve_gn_sqp_report = None
        from myrobot_mjlab.task.DTC.tamols.gn_sqp_solver import solve_gn_sqp

        report = solve_gn_sqp(nlp, blocks, GNSQPOptions())
    assert report.success, report.status
    export = export_trajectory(nlp, report)
    jvel = touchdown_joint_velocities(nlp, report, export)
    plan = plan_from_report(
        nlp, report, export, jvel, stamp=1, event="touchdown"
    )
    return nlp, plan


class TestDrawPlan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp, cls.plan = _make_plan()

    def test_foothold_dots_match_plan(self):
        viz = RecordingVisualizer()
        drawn = draw_plan(viz, self.plan, DTCVizConfig(
            draw_trajectory=False, draw_touchdown=False,
            draw_horizon_end=False,
        ))
        self.assertEqual(drawn, 4)
        self.assertEqual(len(viz.spheres), 4)
        for leg in range(4):
            s = [s for s in viz.spheres
                 if s["label"] == f"dtc_foothold_{'FL FR RL RR'.split()[leg]}_0"]
            self.assertEqual(len(s), 1)
            np.testing.assert_allclose(
                s[0]["center"], self.plan.footholds_w[leg], atol=1e-12
            )

    def test_trajectory_samples(self):
        cfg = DTCVizConfig(draw_footholds=False, draw_touchdown=False,
                           draw_horizon_end=False, trajectory_samples=7)
        viz = RecordingVisualizer()
        drawn = draw_plan(viz, self.plan, cfg)
        self.assertEqual(drawn, 7)
        for s in viz.spheres:
            t_idx = int(s["label"].split("_")[2])
            t = float(self.plan.t[t_idx])
            ref = self.plan.base_reference_at(t)
            np.testing.assert_allclose(s["center"], ref["pos_w"], atol=1e-12)

    def test_touchdown_arrows_for_touchdown_legs(self):
        cfg = DTCVizConfig(draw_footholds=False, draw_trajectory=False,
                           draw_horizon_end=False, draw_touchdown=True)
        viz = RecordingVisualizer()
        draw_plan(viz, self.plan, cfg)
        td_legs = self.nlp.touchdown_legs()
        self.assertEqual(len(viz.arrows), len(td_legs))
        for a, leg in zip(viz.arrows, td_legs):
            self.assertEqual(a["label"], f"dtc_td_{'FL FR RL RR'.split()[leg]}_0")
            np.testing.assert_allclose(
                a["end"][:2], self.plan.footholds_w[leg][:2], atol=0.02
            )

    def test_horizon_end_frame_toggle(self):
        cfg = DTCVizConfig(draw_footholds=False, draw_trajectory=False,
                           draw_touchdown=False, draw_horizon_end=True)
        viz = RecordingVisualizer()
        draw_plan(viz, self.plan, cfg)
        self.assertEqual(len(viz.frames), 1)
        self.assertEqual(viz.frames[0]["label"], "dtc_end_0")
        np.testing.assert_allclose(
            viz.frames[0]["pos"],
            self.plan.base_reference_at(self.plan.horizon[1])["pos_w"],
            atol=1e-12,
        )

    def test_none_and_failed_plan_draw_nothing(self):
        viz = RecordingVisualizer()
        self.assertEqual(draw_plan(viz, None, DTCVizConfig()), 0)
        self.assertEqual(len(viz.spheres), 0)
        failed = self.plan.__class__(
            stamp=0, status="x", success=False,
            horizon=(0.0, 1.0), t=np.zeros(0),
            base_pos_w=np.zeros((0, 3)), base_euler_zyx=np.zeros((0, 3)),
            base_lin_vel_w=np.zeros((0, 3)), base_lin_vel_b=np.zeros((0, 3)),
            base_ang_vel_b=np.zeros((0, 3)), base_lin_acc_w=np.zeros((0, 3)),
            footholds_w=np.full((4, 3), np.nan),
            touchdown_times=np.full(4, np.nan),
            touchdown_joint_pos=np.full((4, 3), np.nan),
            touchdown_joint_vel=np.full((4, 3), np.nan),
            contact_sets=(), spline_coeffs=np.zeros((1, 6, 5)),
            durations=(1.0,), state={},
        )
        self.assertEqual(draw_plan(viz, failed, DTCVizConfig()), 0)
        self.assertEqual(len(viz.spheres), 0)


class TestVizObserverWithWorker(unittest.TestCase):
    def test_worker_publishes_and_observer_draws(self):
        nlp, (coeffs, desired), _ = build_scenario("crawl_flat", CONTACT_NORMAL)
        nlp.add_initial_state_constraints()
        nlp.add_junction_constraints()
        nlp.add_dynamic_stability_constraints()
        nlp.add_kinematic_contact_constraints()
        nlp.add_objectives()
        nlp.finalize()
        blocks = [np.asarray(c, float) for c in coeffs] + [
            np.asarray(desired, float), np.zeros(3)
        ]
        viz_obs = DTCPlanVizObserver(env_idx=0)
        worker = PlanningWorker(
            make_default_solve_fn(nlp, blocks, GNSQPOptions()),
            name="viz-test",
        )
        worker.attach_observer(viz_obs)
        worker.start()
        try:
            worker.notify("touchdown")
            deadline = time.time() + 15
            while viz_obs.plan is None and time.time() < deadline:
                time.sleep(0.02)
            self.assertIsNotNone(viz_obs.plan)
            self.assertTrue(viz_obs.plan.success)
            # 每帧绘制入口: 从 observer 快照画
            viz = RecordingVisualizer()
            viz_obs.debug_vis(viz)
            opt = [s for s in viz.spheres
                   if s["label"].startswith("dtc_foothold_")]
            self.assertEqual(len(opt), 4)
            np.testing.assert_allclose(
                sorted(s["center"].tolist() for s in opt),
                sorted(self_plan.footholds_w.tolist())
                if (self_plan := viz_obs.plan) else [],
                atol=1e-10,
            )
        finally:
            worker.stop()


if __name__ == "__main__":
    unittest.main()
