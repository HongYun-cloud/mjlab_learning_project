"""RL 桥接层测试: 发布契约 / 触地关节速度 FK 闭合 / 独立线程语义.

运行:
    python -m unittest myrobot_mjlab.task.DTC.tamols.test_rl_bridge -v

用户约定的验证点:
  1. 输出契约 = 基座轨迹与位姿 + 四腿落脚点 + 触地期望关节位置/速度,
     且为**局部有限时域** (horizon = 一个步态周期), 不是全局路径;
  2. base_reference_at 在时域内任意时刻精确求值 (与导出网格一致);
  3. 触地关节速度的 FK 闭合: 基座按规划运动 + 关节按期望速度积分 =>
     足端世界位置不动 (触地静止);
  4. worker 线程: 事件触发 (触地/跌倒/推搡/指令变化), latest_plan 单调,
     事件合并, 失败保留旧解, stop 干净退出;
  5. obs / reward 钩子被回调 (接口预留, 行为由 RL 侧定义).
"""

from __future__ import annotations

import contextlib
import io
import time
import unittest

import numpy as np

from myrobot_mjlab.task.DTC.tamols.gn_sqp_solver import (
    GNSQPOptions,
    solve_gn_sqp,
)
from myrobot_mjlab.task.DTC.tamols.rl_bridge import (
    EVENT_KINDS,
    PlanObserver,
    PlanningWorker,
    RewardHook,
    TAMOLSPlan,
    make_default_solve_fn,
)
from myrobot_mjlab.task.DTC.tamols.trajectory_export import (
    export_trajectory,
    go2_leg_forward_kinematics_np,
    touchdown_joint_velocities,
)
from myrobot_mjlab.task.DTC.tamols.v_a_constraints import rotation_matrix_zyx
from myrobot_mjlab.task.DTC.tamols.synthetic_scenarios import (
    CONTACT_NORMAL,
    build_scenario,
)

GO2_HIP_TRUNK = np.array(
    [
        [0.1934, 0.0465, 0.0],
        [0.1934, -0.0465, 0.0],
        [-0.1934, 0.0465, 0.0],
        [-0.1934, -0.0465, 0.0],
    ]
)


def _solve_crawl():
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
        report = solve_gn_sqp(nlp, blocks, GNSQPOptions())
    return nlp, report, blocks


class TestPlanContract(unittest.TestCase):
    """验收 1+2: 输出契约 = 局部有限时域的基座/落脚/触地关节."""

    @classmethod
    def setUpClass(cls):
        cls.nlp, cls.report, cls.blocks = _solve_crawl()
        cls.export = export_trajectory(cls.nlp, cls.report)
        cls.jvel = touchdown_joint_velocities(cls.nlp, cls.report, cls.export)
        from myrobot_mjlab.task.DTC.tamols.rl_bridge import plan_from_report

        cls.plan = plan_from_report(
            cls.nlp, cls.report, cls.export, cls.jvel, stamp=1
        )

    def test_local_finite_horizon(self):
        """局部有限时域: horizon = 一个步态周期, 不是全局路径."""
        self.assertAlmostEqual(
            self.plan.horizon[1] - self.plan.horizon[0],
            sum(self.nlp.durations),
            places=12,
        )
        self.assertLess(self.plan.horizon[1], 2.0)  # 有限 (一个周期)
        # 每条腿在 horizon 内至多一个触地事件 (单步/腿, 论文 Fig.7)
        self.assertEqual(len(self.plan.contact_sets), len(self.nlp.durations))

    def test_contract_fields_and_units(self):
        plan = self.plan
        n = self.export.t.size
        for arr in (
            plan.base_pos_w, plan.base_euler_zyx, plan.base_lin_vel_w,
            plan.base_lin_vel_b, plan.base_ang_vel_b, plan.base_lin_acc_w,
        ):
            self.assertEqual(arr.shape, (n, 3))
        self.assertEqual(plan.footholds_w.shape, (4, 3))
        self.assertEqual(plan.touchdown_joint_pos.shape, (4, 3))
        self.assertEqual(plan.touchdown_joint_vel.shape, (4, 3))
        self.assertTrue(np.isfinite(plan.footholds_w).all())
        # 触地腿有有限关节目标, 不触地腿 NaN
        for leg in self.nlp.touchdown_legs():
            self.assertTrue(np.isfinite(plan.touchdown_joint_pos[leg]).all())
            self.assertTrue(np.isfinite(plan.touchdown_joint_vel[leg]).all())
        for leg in range(4):
            if leg not in self.nlp.touchdown_legs():
                self.assertTrue(np.isnan(plan.touchdown_joint_pos[leg]).all())

    def test_base_reference_at_matches_export(self):
        """任意时刻精确求值: 与导出网格逐点一致 (样条, 非插值近似)."""
        for t, i in zip(self.export.t[::5], range(0, len(self.export.t), 5)):
            ref = self.plan.base_reference_at(float(t))
            np.testing.assert_allclose(
                ref["pos_w"], self.export.base_pos_w[i], atol=1e-10
            )
            np.testing.assert_allclose(
                ref["lin_vel_b"], self.export.base_lin_vel_b[i], atol=1e-10
            )
            np.testing.assert_allclose(
                ref["ang_vel_b"], self.export.base_ang_vel_b[i], atol=1e-10
            )
        # 越界截断
        beyond = self.plan.base_reference_at(self.plan.horizon[1] + 5.0)
        np.testing.assert_allclose(
            beyond["pos_w"], self.export.base_pos_w[-1], atol=1e-10
        )

    def test_touchdown_joint_velocity_fk_closure(self):
        """验收 3: 触地静止的 FK 闭合——基座按规划运动 + 关节按期望
        速度积分 => 足端世界位置不变 (独立 FK 复算, 不复用被测雅可比)."""
        dt = 1e-3
        for leg in self.nlp.touchdown_legs():
            t_td = self.export.touchdown_times[leg]
            idx = min(
                int(np.searchsorted(self.export.t, t_td, side="left")),
                len(self.export.t) - 1,
            )
            q = self.export.touchdown_joint_pos[leg].copy()
            qd = self.plan.touchdown_joint_vel[leg]
            R = np.array(rotation_matrix_zyx(*self.export.base_euler_zyx[idx]))
            v = self.export.base_lin_vel_w[idx]
            is_left = leg in (0, 2)
            p0 = np.asarray(
                go2_leg_forward_kinematics_np(*q, is_left=is_left)
            )
            foot_before = (
                self.export.base_pos_w[idx]
                + R @ (GO2_HIP_TRUNK[leg] + p0)
            )
            p1 = np.asarray(
                go2_leg_forward_kinematics_np(*(q + qd * dt), is_left=is_left)
            )
            foot_after = (
                self.export.base_pos_w[idx] + v * dt
                + R @ (GO2_HIP_TRUNK[leg] + p1)
            )
            drift = float(np.abs(foot_after - foot_before).max())
            self.assertLess(
                drift, 1e-4,
                f"leg {leg}: 触地足端漂移 {drift:.2e} m (应≈0)",
            )


class TestPlanningWorker(unittest.TestCase):
    """验收 4+5: 独立线程 / 事件触发 / 合并 / 失败保留 / 钩子."""

    def _make_worker(self, nlp, blocks):
        solve_fn = make_default_solve_fn(nlp, blocks, GNSQPOptions())
        return PlanningWorker(solve_fn, name="test-planner")

    def test_event_triggered_and_latest_plan_semantics(self):
        nlp, _, blocks = _solve_crawl()
        worker = self._make_worker(nlp, blocks)
        seen = []
        worker.attach_observer(_Recorder(seen))
        worker.start()
        try:
            self.assertIsNone(worker.latest_plan())  # 更新前: 无解
            worker.notify("touchdown")
            deadline = time.time() + 15
            while worker.stamp == 0 and time.time() < deadline:
                time.sleep(0.02)
            plan = worker.latest_plan()
            self.assertIsNotNone(plan)
            self.assertTrue(plan.success)
            self.assertEqual(plan.event, "touchdown")
            self.assertEqual(worker.stamp, 1)
            stamp_before = worker.stamp
            plan_ref = worker.latest_plan()
            # 两次更新之间: 无新事件时策略读到**同一份**最新解
            time.sleep(0.2)
            self.assertEqual(worker.stamp, stamp_before)
            self.assertIs(worker.latest_plan(), plan_ref)
            # 其余事件类型同样触发
            for event in ("fall", "push", "cmd_change"):
                worker.notify(event)
                deadline = time.time() + 15
                while worker.stamp == stamp_before and time.time() < deadline:
                    time.sleep(0.02)
                self.assertGreater(worker.stamp, stamp_before)
                stamp_before = worker.stamp
            self.assertGreaterEqual(len(seen), 4)  # 每次发布都通知 observer
        finally:
            worker.stop()
        self.assertFalse(worker.is_running)

    def test_failure_keeps_previous_plan(self):
        """失败保留旧解: 第二次求解被注入错误后, latest_plan 仍是旧快照."""
        nlp, _, blocks = _solve_crawl()
        calls = {"n": 0}
        good = make_default_solve_fn(nlp, blocks, GNSQPOptions())

        def flaky(warm, event="", payload=None):
            calls["n"] += 1
            if calls["n"] >= 2:
                raise RuntimeError("injected solver crash")
            return good(warm, event, payload)

        worker = PlanningWorker(flaky, name="test-flaky")
        worker.start()
        try:
            worker.notify("touchdown")
            deadline = time.time() + 15
            while worker.stamp == 0 and time.time() < deadline:
                time.sleep(0.02)
            good_plan = worker.latest_plan()
            self.assertTrue(good_plan.success)
            worker.notify("push")  # 第二次: 注入崩溃
            deadline = time.time() + 15
            while worker.stamp == 1 and time.time() < deadline:
                time.sleep(0.02)
            self.assertIsNotNone(worker.last_error)
            self.assertIn("injected", worker.last_error)
            # 策略在两次更新之间仍使用**上一份有效解**
            current = worker.latest_plan()
            self.assertTrue(current.success)
            self.assertIs(current, good_plan)
            np.testing.assert_allclose(
                current.footholds_w, good_plan.footholds_w
            )
        finally:
            worker.stop()

    def test_reward_hook_called_per_event(self):
        nlp, _, blocks = _solve_crawl()
        events_seen = []
        worker = self._make_worker(nlp, blocks)

        class _Hook(RewardHook):
            def on_event(self, event, plan):
                events_seen.append((event, plan is not None and plan.success))

        worker.attach_reward_hook(_Hook())
        worker.start()
        try:
            worker.notify("cmd_change")
            deadline = time.time() + 15
            while worker.stamp == 0 and time.time() < deadline:
                time.sleep(0.02)
            time.sleep(0.1)
        finally:
            worker.stop()
        self.assertEqual([e for e, _ in events_seen], ["cmd_change"])
        self.assertTrue(all(ok for _, ok in events_seen))

    def test_unknown_event_rejected(self):
        nlp, _, blocks = _solve_crawl()
        worker = self._make_worker(nlp, blocks)
        with self.assertRaises(ValueError):
            worker.notify("teleport")
        self.assertEqual(EVENT_KINDS, ("touchdown", "fall", "push", "cmd_change"))


class _Recorder(PlanObserver):
    def __init__(self, sink):
        self.sink = sink

    def on_plan(self, plan):
        self.sink.append(plan)

    def observe(self, t, plan):
        return {"stamp": plan.stamp if plan else None}


if __name__ == "__main__":
    unittest.main()
