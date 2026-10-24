"""x -> 下游导出的测试 (任务 A).

运行:
    python -m unittest myrobot_mjlab.task.DTC.tamols.test_trajectory_export -v

覆盖:
  - 导出形状/时间网格/边界单次采样;
  - 初始位姿等式在导出轨迹首点成立;
  - 触地时刻 = 触地相位起始; 不触地腿为 NaN;
  - 触地关节角在 URDF 限位内, 且经现有 FK 闭合回髋系落脚点;
  - 触地脚高度与地形一致 (check_export 无违反);
  - 失败报告拒绝导出.
"""

from __future__ import annotations

import unittest

import numpy as np

from myrobot_mjlab.task.DTC.tamols.synthetic_scenarios import (
    CONTACT_NORMAL,
    build_scenario,
    run_scenario,
)
from myrobot_mjlab.task.DTC.tamols.trajectory_export import (
    JOINT_LIMITS,
    check_export,
    export_trajectory,
    phase_start_times,
)
from myrobot_mjlab.task.Tamols.fetch.go2_kinematics import (
    GO2_HIP_OFFSETS_TRUNK,
    GO2_L_HIP_SIGNS,
    go2_leg_forward_kinematics_np,
)
from myrobot_mjlab.task.DTC.tamols.v_a_constraints import rotation_matrix_zyx


def _solve(name):
    nlp, guess, _ = build_scenario(name, CONTACT_NORMAL)
    nlp.add_initial_state_constraints()
    nlp.add_junction_constraints()
    nlp.add_dynamic_stability_constraints()
    nlp.add_kinematic_contact_constraints()
    nlp.add_objectives()
    nlp.finalize()
    nlp.set_initial_guess(*guess)
    return nlp, nlp.solve(name)


class TestTrajectoryExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp, cls.report = _solve("crawl_flat")
        cls.export = export_trajectory(cls.nlp, cls.report, samples_per_phase=25)

    def test_grid_and_shapes(self):
        e = self.export
        n = 3 * 25 - 2  # 每相位 25 点, 两个边界各去重一次
        self.assertEqual(e.t.size, n)
        self.assertTrue(np.all(np.diff(e.t) > 0))
        for arr in (
            e.base_pos_w, e.base_lin_vel_w, e.base_lin_acc_w,
            e.base_lin_vel_b, e.base_ang_vel_b, e.base_euler_zyx,
        ):
            self.assertEqual(arr.shape, (n, 3))
        self.assertAlmostEqual(e.t[0], 0.0, places=12)
        self.assertAlmostEqual(e.t[-1], sum(self.nlp.durations), places=12)

    def test_initial_pose_at_first_sample(self):
        np.testing.assert_allclose(
            self.export.base_pos_w[0], [0.0, 0.0, 0.30], atol=1e-9
        )
        np.testing.assert_allclose(
            self.export.base_lin_vel_w[0], [0.25, 0.0, 0.0], atol=1e-9
        )

    def test_touchdown_times(self):
        starts = phase_start_times(self.nlp)
        e = self.export
        self.assertAlmostEqual(e.touchdown_times[0], starts[1], places=12)
        self.assertAlmostEqual(e.touchdown_times[1], starts[2], places=12)
        self.assertTrue(np.isnan(e.touchdown_times[2]))
        self.assertTrue(np.isnan(e.touchdown_times[3]))

    def test_joint_limits_and_fk_closure(self):
        e = self.export
        for leg in (0, 1):
            q = e.touchdown_joint_pos[leg]
            self.assertTrue(np.isfinite(q).all())
            for idx, name in enumerate(("hip", "thigh", "calf")):
                lo, hi = JOINT_LIMITS[name]
                self.assertGreaterEqual(q[idx], lo - 1e-9)
                self.assertLessEqual(q[idx], hi + 1e-9)
            # FK 闭合: 关节角 -> 髋系脚点, 应等于导出时用的髋系目标
            p_fk = go2_leg_forward_kinematics_np(
                q[0], q[1], q[2], is_left=GO2_L_HIP_SIGNS[leg] > 0
            )
            phase = self.nlp.schedules[leg].touchdown_phase
            pose = e.base_pos_w[
                np.searchsorted(e.t, e.touchdown_times[leg], side="left")
            ]
            euler = e.base_euler_zyx[
                np.searchsorted(e.t, e.touchdown_times[leg], side="left")
            ]
            R = np.array(rotation_matrix_zyx(euler[0], euler[1], euler[2]))
            p_hip_target = R.T @ (
                e.footholds_w[leg] - pose
            ) - GO2_HIP_OFFSETS_TRUNK[leg]
            np.testing.assert_allclose(p_fk, p_hip_target, atol=1e-6)

    def test_check_export_clean(self):
        self.assertEqual(check_export(self.nlp, self.export), [])

    def test_static_export_no_touchdowns(self):
        # 不触地场景: 导出接口仍可用, 触地时刻/关节角全为 NaN
        nlp, report = _solve("static_flat")
        e = export_trajectory(nlp, report, samples_per_phase=7)
        self.assertTrue(np.isnan(e.touchdown_times).all())
        self.assertTrue(np.isnan(e.touchdown_joint_pos).all())

    def test_failed_report_rejected(self):
        res = run_scenario("infeasible_reach", CONTACT_NORMAL, verbose=False)
        nlp, _, _ = build_scenario("infeasible_reach", CONTACT_NORMAL)
        with self.assertRaises(ValueError):
            export_trajectory(nlp, res.report)


if __name__ == "__main__":
    unittest.main()
