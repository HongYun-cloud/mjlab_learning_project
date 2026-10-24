"""eps<=0 修复后的轨迹物理合理性复核测试 (任务 B).

运行:
    python -m unittest myrobot_mjlab.task.DTC.tamols.test_trajectory_audit -v

复核内容 (对每个可解场景):
  - 摩擦利用率 ||F_xy||/(μ F_z) <= 1 且 F_z > 0 (F = m(p̈-g), 与符号约定无关);
  - 触地腿贴地误差 <= 1 cm (V-B1 软任务, 论文期望 mm 级);
  - 可达距离在 [l_min, l_max] 内;
  - 基座 z / 水平与竖直加速度在生理/工程合理区间;
  - 相位连接 C0/C1 跳变 <= 1e-8;
  - 松弛 eps <= 0 (式 29, 本次修复项);
  - slack 版与精确 GIAC 版的解差异有界 (两者都可行).
观察记录 (非断言): two_stones 场景摩擦利用率接近 1 (贴滑移极限),
见 test_two_stones_margin_note 的打印.
"""

from __future__ import annotations

import unittest

import numpy as np

from myrobot_mjlab.task.DTC.tamols.assumptions_config import AssumptionsConfig
from myrobot_mjlab.task.DTC.tamols.synthetic_scenarios import (
    CONTACT_NORMAL,
    build_scenario,
    run_scenario,
)
from myrobot_mjlab.task.DTC.tamols.trajectory_audit import (
    audit_trajectory,
    compare_solutions,
)


def _solve(name, cfg):
    nlp, guess, _ = build_scenario(name, cfg)
    if name == "static_flat":
        nlp.add_constant_base_pin(0)
    nlp.add_initial_state_constraints()
    nlp.add_junction_constraints()
    nlp.add_dynamic_stability_constraints()
    nlp.add_kinematic_contact_constraints()
    nlp.add_objectives()
    nlp.finalize()
    nlp.set_initial_guess(*guess)
    return nlp, nlp.solve(name)


class TestTrajectoryPlausibility(unittest.TestCase):
    def _check(self, name):
        nlp, report = _solve(name, CONTACT_NORMAL)
        self.assertTrue(report.success, report.status)
        audit = audit_trajectory(nlp, report)
        self.assertEqual(audit.violations, [], f"{name}: {audit.violations}")
        self.assertLessEqual(max(audit.friction_utilization.values()), 1.0 + 1e-6)
        self.assertGreater(min(audit.net_contact_force_z.values()), 0.0)
        self.assertLessEqual(max(audit.slack) if audit.slack.size else 0.0, 1e-6)
        for jump in audit.junction_jumps.values():
            self.assertLessEqual(jump, 1e-8)
        return audit

    def test_static_flat_plausible(self):
        audit = self._check("static_flat")
        # 静止: 合力 = m g 向上, 利用率 0
        self.assertAlmostEqual(
            min(audit.net_contact_force_z.values()), 15.2064 * 9.81, delta=1e-6
        )
        self.assertAlmostEqual(max(audit.friction_utilization.values()), 0.0, places=9)
        # 静止 GIAC 裕度 = |eps| = 8.366 (与 v_a 静态分析同值)
        self.assertAlmostEqual(audit.slack[0], -8.3666, delta=0.01)

    def test_crawl_flat_plausible(self):
        audit = self._check("crawl_flat")
        self.assertLess(max(audit.friction_utilization.values()), 0.9)
        self.assertLess(max(audit.ground_error.values()), 1e-3)
        self.assertLess(
            np.max(np.linalg.norm(audit.base_lin_acc_w[:, :2], axis=1)), 8.0
        )

    def test_crawl_step_plausible(self):
        audit = self._check("crawl_step")
        # 上台阶: 基座 z 应高于平地名义 (0.30) 一段
        self.assertGreater(audit.base_pos[:, 2].max(), 0.31)
        self.assertLess(max(audit.ground_error.values()), 1e-2)

    def test_two_stones_margin_note(self):
        audit = self._check("two_stones")
        util = max(audit.friction_utilization.values())
        self.assertLessEqual(util, 1.0 + 1e-6)
        # 观察: 该场景 N=3 相位基座在支撑三角形外, 摩擦利用率接近 1
        # (贴滑移极限). 仅作记录, 不作断言下限.
        print(f"[observe] two_stones friction utilization max = {util:.4f}")

    def test_audit_rejects_failed_report(self):
        res = run_scenario("infeasible_reach", CONTACT_NORMAL, verbose=False)
        nlp, _, _ = build_scenario("infeasible_reach", CONTACT_NORMAL)
        with self.assertRaises(ValueError):
            audit_trajectory(nlp, res.report)


class TestSlackVersusExact(unittest.TestCase):
    def test_both_feasible_and_bounded_difference(self):
        _, ra = _solve(
            "crawl_flat",
            AssumptionsConfig(
                inertial_accel_sign="contact_normal_explicit", use_giac_slack=True
            ),
        )
        _, rb = _solve(
            "crawl_flat",
            AssumptionsConfig(
                inertial_accel_sign="contact_normal_explicit", use_giac_slack=False
            ),
        )
        self.assertTrue(ra.success and rb.success)
        diff = compare_solutions(ra, rb)
        # slack 只改变裕度权衡, 不应把解搬到另一个盆地
        self.assertLess(diff["max_foothold_delta_m"], 0.05)
        self.assertEqual(rb.solution.get("slack", None), None)


if __name__ == "__main__":
    unittest.main()
