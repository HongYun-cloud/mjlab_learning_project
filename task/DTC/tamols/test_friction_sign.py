"""摩擦锥符号审计的物理基准测试 (用户任务书 C.3/C.4).

运行:
    python -m unittest myrobot_mjlab.task.DTC.tamols.test_friction_sign -v

用例覆盖: 静止站立 / 正向竖直加速度 / 向下竖直加速度 / 锥边界水平
加速度 / 锥外水平加速度 / 自由落体式 (法向违反). 每个用例用**独立
NumPy** 由线动量方程恢复接触合力 F = m(p̈-g), 在力级锥 (论文式 (3) 对
合力) 上判定, 再核对:
  - 派生残差 (force_balance_derived) 的可行性判定与力级判据一致;
  - 字面残差 (paper_literal, 论文印刷式 (5)) 在哪些用例上给出与物理
    相反的结论 (静止/边界/向下加速度: 字面法向残差 > 0).

本测试**能**验证: 合力锥必要条件 (残差形式与 F=m(p̈-g) 的锥隶属一致)、
边界/锥外/法向的解析符号与量值。
本测试**不能**验证: 逐接触力分配 -- NLP 决策变量不含接触力
(contact-force-free 模型, 论文 Assumption 1-2), 见
``TestModelLimits.test_no_contact_force_variables``.
"""

from __future__ import annotations

import unittest

import numpy as np

from myrobot_mjlab.task.DTC.tamols.assumptions_config import (
    DEFAULT_ASSUMPTIONS,
    AssumptionsConfig,
    SAFETY_NOTE,
)
from myrobot_mjlab.task.DTC.tamols.friction_sign import (
    GRAVITY,
    PhysicsCase,
    contact_force_cone_residuals,
    derived_cone_residuals,
    force_balance_verdict,
    friction_records_derived,
    literal_cone_residuals,
    net_contact_force,
    physics_cases,
)
from myrobot_mjlab.task.DTC.tamols.nlp_integration import (
    IntegratedNLP,
    LegSchedule,
    NLPParameters,
)
from myrobot_mjlab.task.DTC.tamols.synthetic_scenarios import (
    GO2_FEET_STANCE,
    flat_maps,
)

MASS = 15.2064
MU = 0.6


def aB_of(p_ddot):
    """论文符号 aB = g - p̈."""
    g = np.array(GRAVITY)
    return g - np.asarray(p_ddot, dtype=float)


def feasible(pair):
    cone, normal = pair
    return cone <= 1e-9 and normal <= 1e-9


class TestForceBalancePhysics(unittest.TestCase):
    def test_force_level_verdicts_match_analytic_expectation(self):
        for case in physics_cases(MU):
            cone, normal = force_balance_verdict(case, MASS)
            self.assertEqual(
                cone <= 1e-9, case.expect_force_cone_ok,
                f"{case.name}: 力级锥残差 {cone:.4g}",
            )
            self.assertEqual(
                normal <= 1e-9, case.expect_force_normal_ok,
                f"{case.name}: 力级法向残差 {normal:.4g}",
            )

    def test_static_contact_force_points_up(self):
        F = net_contact_force(MASS, (0.0, 0.0, 0.0))
        np.testing.assert_allclose(F, [0.0, 0.0, MASS * 9.81], atol=1e-9)
        self.assertGreater(F[2], 0.0)  # 地面对机器人的法向接触力向上

    def test_derived_matches_force_level_feasibility(self):
        for case in physics_cases(MU):
            aB = aB_of(case.p_ddot)
            derived = derived_cone_residuals(aB, case.mu)
            force = force_balance_verdict(case, MASS)
            self.assertEqual(
                feasible(derived), feasible(force),
                f"{case.name}: derived={derived} force={force}",
            )

    def test_literal_contradicts_physics_on_supported_cases(self):
        """字面式 (5) 在物理上可行的支撑用例上判不可行 (符号错误证据)."""
        supported = [
            c for c in physics_cases(MU)
            if c.expect_force_cone_ok and c.expect_force_normal_ok
        ]
        self.assertTrue(supported)
        for case in supported:
            aB = aB_of(case.p_ddot)
            literal = literal_cone_residuals(aB, case.mu)
            if abs(case.p_ddot[0]) < 1e-12 and abs(case.p_ddot[1]) < 1e-12:
                # 纯竖直/静止: 字面法向 -aB_z = +(p̈_z - g_z)*-1... = p̈_z+9.81 的负号错
                self.assertGreater(
                    literal[1], 0.0,
                    f"{case.name}: 字面法向残差应 >0 (与物理相反)",
                )
                self.assertFalse(feasible(literal))
            else:
                # 水平加速度用例: 字面法向同样 > 0 (aB_z = -9.81)
                self.assertGreater(literal[1], 0.0)
                self.assertFalse(feasible(literal))

    def test_cone_boundary_values(self):
        case = [c for c in physics_cases(MU)
                if c.name == "horizontal_on_cone_boundary"][0]
        aB = aB_of(case.p_ddot)
        d_cone, d_norm = derived_cone_residuals(aB, case.mu)
        self.assertAlmostEqual(d_cone, 0.0, places=9)   # 恰在锥面
        self.assertLess(d_norm, 0.0)
        F = net_contact_force(MASS, case.p_ddot)
        f_cone, f_norm = contact_force_cone_residuals(F, case.mu)
        self.assertAlmostEqual(f_cone, 0.0, places=6)
        self.assertLess(f_norm, 0.0)
        outside = [c for c in physics_cases(MU)
                   if c.name == "horizontal_outside_cone"][0]
        aB_o = aB_of(outside.p_ddot)
        self.assertGreater(derived_cone_residuals(aB_o, outside.mu)[0], 0.0)
        F_o = net_contact_force(MASS, outside.p_ddot)
        self.assertGreater(contact_force_cone_residuals(F_o, outside.mu)[0], 0.0)

    def test_freefall_normal_violation_both_forms(self):
        case = [c for c in physics_cases(MU) if c.name == "freefall_like"][0]
        aB = aB_of(case.p_ddot)
        self.assertFalse(feasible(derived_cone_residuals(aB, case.mu)))
        self.assertFalse(feasible(force_balance_verdict(case, MASS)))
        # 字面形式在自由落体式用例上法向"满足"(-aB_z<0), 恰是其符号错误
        # 的另一面: 它把无法推地的状态判为法向可行.
        self.assertLess(literal_cone_residuals(aB, case.mu)[1], 0.0)


class TestDerivedRecords(unittest.TestCase):
    def test_records_at_rest(self):
        aB = aB_of((0.0, 0.0, 0.0))
        records = friction_records_derived(aB, (0.0, 0.0, 0.0), MU)
        by_name = {r.name: float(r.residual) for r in records}
        self.assertAlmostEqual(by_name["5fb_cone"], -(MU * 9.81) ** 2, places=9)
        self.assertAlmostEqual(by_name["5fb_normal"], -9.81, places=9)
        self.assertEqual(len(records), 2)
        with_push = friction_records_derived(
            aB, (0.0, 0.0, 0.0), MU, include_push=True
        )
        by_name2 = {r.name: float(r.residual) for r in with_push}
        # 推地残差 g_z - p̈_z 与派生法向伴生 aB_z 代数相同 (冗余, 审计用)
        self.assertAlmostEqual(
            by_name2["V-A-4_push"], by_name2["5fb_normal"], places=12
        )

    def test_literal_function_matches_v_a_module(self):
        from myrobot_mjlab.task.DTC.tamols.giac_constraints import (
            friction_squared_residual,
        )

        aB = aB_of((0.3, -0.1, 0.2))
        cone, _ = literal_cone_residuals(aB, MU)
        self.assertAlmostEqual(
            float(cone), float(friction_squared_residual(aB, MU)), places=12
        )


class TestStaticNLPWithDerivedCone(unittest.TestCase):
    def _assemble(self, cfg):
        maps = flat_maps()
        schedules = {
            leg: LegSchedule(True, measured_position=tuple(GO2_FEET_STANCE[leg]))
            for leg in range(4)
        }
        params = NLPParameters(
            base_pose0=[0, 0, 0.30, 0, 0, 0],
            base_twist0=np.zeros(6),
            measured_feet={leg: GO2_FEET_STANCE[leg] for leg in range(4)},
            ref_twist=lambda p, t: (np.zeros(3), np.zeros(3)),
            solver_options={"print_time": False,
                            "ipopt": {"print_level": 0, "max_iter": 200}},
        )
        nlp = IntegratedNLP(
            durations=[0.4], schedules=schedules, maps=maps, params=params,
            assumptions=cfg,
        )
        nlp.add_constant_base_pin(0)
        nlp.add_initial_state_constraints()
        nlp.add_junction_constraints()
        nlp.add_dynamic_stability_constraints()
        nlp.add_kinematic_contact_constraints()
        nlp.add_objectives()
        nlp.finalize()
        nlp.set_initial_guess([np.zeros((6, 5))], GO2_FEET_STANCE.copy())
        return nlp

    def test_derived_cone_static_feasible_giac_untouched(self):
        cfg = AssumptionsConfig(friction_cone_form="force_balance_derived")
        nlp = self._assemble(cfg)
        # GIAC 仍用字面 aB + CCW (未被全局翻转)
        self.assertEqual(nlp.assumptions.inertial_accel_sign, "paper_literal")
        self.assertEqual(nlp.assumptions.resolved_winding(), "ccw")
        cats = {e.category for e in nlp.constraint_entries}
        self.assertIn("17b-17f", cats)
        names = [e.name for e in nlp.constraint_entries if e.category == "5"]
        self.assertTrue(all(n.startswith("5fb_") for n in names))
        report = nlp.solve("static-derived")
        self.assertTrue(report.success, report.status)
        self.assertLess(max(report.max_violation_by_category.values()), 1e-6)

    def test_norm_form_with_derived_rejected(self):
        cfg = AssumptionsConfig(
            friction_cone_form="force_balance_derived", friction_form="norm"
        )
        maps = flat_maps()
        schedules = {
            leg: LegSchedule(True, measured_position=tuple(GO2_FEET_STANCE[leg]))
            for leg in range(4)
        }
        params = NLPParameters(
            base_pose0=[0, 0, 0.30, 0, 0, 0],
            base_twist0=np.zeros(6),
            measured_feet={leg: GO2_FEET_STANCE[leg] for leg in range(4)},
            solver_options={"print_time": False,
                            "ipopt": {"print_level": 0, "max_iter": 50}},
        )
        nlp = IntegratedNLP(
            durations=[0.4], schedules=schedules, maps=maps, params=params,
            assumptions=cfg,
        )
        nlp.add_initial_state_constraints()
        with self.assertRaises(ValueError):
            nlp.add_dynamic_stability_constraints()

    def test_default_config_unchanged(self):
        self.assertEqual(DEFAULT_ASSUMPTIONS.friction_cone_form, "paper_literal")
        self.assertEqual(DEFAULT_ASSUMPTIONS.inertial_accel_sign, "paper_literal")
        self.assertEqual(DEFAULT_ASSUMPTIONS.friction_mode, "enabled")
        self.assertFalse(DEFAULT_ASSUMPTIONS.include_push)


class TestAuditTrail(unittest.TestCase):
    def test_audit_doc_records_sources_and_uncertainty(self):
        from myrobot_mjlab.task.DTC.tamols import friction_sign

        doc = friction_sign.__doc__
        for marker in (
            "arXiv", "未能确认", "constraints.py", "FIXED", "(5)", "(4)",
            "不作权威依据",
        ):
            self.assertIn(marker, doc)

    def test_safety_note_in_describe(self):
        text = DEFAULT_ASSUMPTIONS.describe()
        self.assertIn("不可用于安全结论", text)
        self.assertIn("UNRESOLVED", SAFETY_NOTE)
        self.assertIn("friction_cone_form", text)


class TestModelLimits(unittest.TestCase):
    def test_no_contact_force_variables(self):
        """模型变量不含接触力: 基准只能验证合力锥必要条件."""
        maps = flat_maps()
        schedules = {
            leg: LegSchedule(True, measured_position=tuple(GO2_FEET_STANCE[leg]))
            for leg in range(4)
        }
        params = NLPParameters(
            base_pose0=[0, 0, 0.30, 0, 0, 0],
            base_twist0=np.zeros(6),
            measured_feet={leg: GO2_FEET_STANCE[leg] for leg in range(4)},
        )
        nlp = IntegratedNLP(
            durations=[0.4], schedules=schedules, maps=maps, params=params,
        )
        names = {name for name, _ in nlp.backend.variables}
        self.assertEqual(
            names, {"spline_coeffs_0", "desired_footholds", "slack"}
        )
        self.assertNotIn("contact_forces", names)


if __name__ == "__main__":
    unittest.main()
