"""GIAC / 摩擦约束的单元测试.

覆盖:
- ``det(a, b, c) == dot(a, cross(b, c))``;
- 已知可行点满足全部 GIAC 与摩擦约束;
- 故意违反约束的点能被检测出来;
- 无序对的枚举、交换 i/j 的符号、约束组装;
- 安装了 CasADi 时, 用 IPOPT 实际求解最小 NLP.

运行:
    python -m unittest myrobot_mjlab.task.DTC.tamols.test_giac_constraints -v
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from myrobot_mjlab.task.DTC.tamols import giac_example
from myrobot_mjlab.task.DTC.tamols.giac_constraints import (
    EQUALITY,
    INEQUALITY,
    add_contact_constraints,
    add_friction_constraint,
    add_giac_constraints,
    ballistic_residuals,
    contact_constraint_records,
    contact_moment,
    contact_pairs,
    det3,
    double_support_line_residual,
    double_support_moment_residual,
    friction_residual,
    friction_squared_residual,
    giac_residual,
    giac_residuals,
    gravity_inertial_acceleration,
    single_support_residuals,
)
from myrobot_mjlab.task.DTC.tamols.giac_example import (
    FEASIBILITY_TOL,
    G_VEC,
    evaluate_residuals,
    reference_case,
)

CASADI_AVAILABLE = giac_example.ca is not None


class RecordingOpti:
    """只记录 subject_to 调用的假求解器对象."""

    def __init__(self) -> None:
        self.constraints: list[Any] = []

    def subject_to(self, expr: Any) -> None:
        self.constraints.append(expr)


class TestDet3(unittest.TestCase):
    def test_matches_dot_cross(self):
        rng = np.random.default_rng(1234)
        for _ in range(100):
            a = rng.normal(size=3)
            b = rng.normal(size=3)
            c = rng.normal(size=3)
            self.assertAlmostEqual(
                float(det3(a, b, c)),
                float(np.dot(a, np.cross(b, c))),
                places=12,
            )

    def test_anticommutes(self):
        rng = np.random.default_rng(7)
        a = rng.normal(size=3)
        b = rng.normal(size=3)
        c = rng.normal(size=3)
        self.assertAlmostEqual(
            float(det3(a, b, c)), -float(det3(b, a, c)), places=12
        )
        self.assertAlmostEqual(
            float(det3(a, b, c)), -float(det3(a, c, b)), places=12
        )

    def test_standard_basis_orientation(self):
        ex = np.array([1.0, 0.0, 0.0])
        ey = np.array([0.0, 1.0, 0.0])
        ez = np.array([0.0, 0.0, 1.0])
        self.assertAlmostEqual(float(det3(ex, ey, ez)), 1.0, places=12)
        self.assertAlmostEqual(float(det3(ex, ez, ey)), -1.0, places=12)

    @unittest.skipUnless(CASADI_AVAILABLE, "CasADi 未安装")
    def test_matches_casadi_vector_operations(self):
        ca = giac_example.ca
        rng = np.random.default_rng(99)
        a_val = rng.normal(size=3)
        b_val = rng.normal(size=3)
        c_val = rng.normal(size=3)

        a = ca.MX.sym("a", 3)
        b = ca.MX.sym("b", 3)
        c = ca.MX.sym("c", 3)
        f = ca.Function("det_test", [a, b, c], [det3(a, b, c)])
        got = float(f(a_val, b_val, c_val))
        expected = float(np.dot(a_val, np.cross(b_val, c_val)))
        self.assertAlmostEqual(got, expected, places=12)


class TestGravityInertialAcceleration(unittest.TestCase):
    def test_definition(self):
        p_ddot = np.array([1.0, -2.0, 3.0])
        a_B = gravity_inertial_acceleration(p_ddot, G_VEC)
        np.testing.assert_allclose(a_B, [-1.0, 2.0, -12.81], atol=1e-15)

    def test_zero_acceleration_is_gravity(self):
        a_B = gravity_inertial_acceleration(np.zeros(3), G_VEC)
        np.testing.assert_allclose(a_B, [0.0, 0.0, -9.81], atol=1e-15)

    def test_bad_shape(self):
        with self.assertRaises(ValueError):
            gravity_inertial_acceleration([0.0, 0.0], G_VEC)
        with self.assertRaises(ValueError):
            gravity_inertial_acceleration([0.0, 0.0, 0.0], [0.0, -9.81])


class TestGiacResidual(unittest.TestCase):
    def setUp(self):
        self.case = reference_case()
        self.a_B = np.asarray(
            gravity_inertial_acceleration(
                self.case.feasible_p_ddot, G_VEC
            )
        )

    def test_matches_manual_formula(self):
        mass = 2.5
        p_B = np.array([0.1, -0.2, 0.35])
        p_i = np.array([0.2, 0.05, 0.0])
        p_j = np.array([-0.1, 0.15, 0.0])
        a_B = np.array([0.3, -0.4, 1.2])
        L_dot = np.array([0.2, 0.1, -0.3])
        p_ij = p_j - p_i
        expected = (
            mass * np.dot(p_ij, np.cross(p_B - p_i, a_B))
            - np.dot(p_ij, L_dot)
        )
        self.assertAlmostEqual(
            float(
                giac_residual(mass, p_B, a_B, L_dot, p_i, p_j)
            ),
            float(expected),
            places=12,
        )

    def test_swap_flips_lhs(self):
        p_i = self.case.feet[0]
        p_j = self.case.feet[1]
        forward = giac_residual(
            self.case.mass,
            self.case.p_B,
            self.a_B,
            self.case.L_dot,
            p_i,
            p_j,
        )
        reverse = giac_residual(
            self.case.mass,
            self.case.p_B,
            self.a_B,
            self.case.L_dot,
            p_j,
            p_i,
        )
        self.assertAlmostEqual(
            float(forward), -float(reverse), places=12
        )

    def test_cyclic_pair_order_matches_support_polygon(self):
        # 机身位于三角形内部, 静态 a_B = g = (0, 0, -9.81), L_dot = 0.
        # 按支撑多边形环绕方向 0->1->2->0 取边, 三条 GIAC 都满足;
        # 直接取所有 i<j 会把 (0,2) 写成反向的 (2,0), 从而翻号.
        case = self.case
        a_B = np.array([0.0, 0.0, -9.81])
        L_dot = np.zeros(3)

        cyclic = giac_residuals(
            case.mass,
            case.p_B,
            a_B,
            L_dot,
            case.feet,
            pairs=[(0, 1), (1, 2), (2, 0)],
        )
        for pair, value in cyclic:
            self.assertLessEqual(
                float(value), FEASIBILITY_TOL, msg=f"pair {pair}"
            )
        self.assertLess(max(float(v) for _, v in cyclic), -1.0)

        default = giac_residuals(
            case.mass, case.p_B, a_B, L_dot, case.feet
        )
        self.assertTrue(
            any(float(value) > 0.0 for _, value in default),
            "默认 i<j 枚举中应能观察到反向边的正残差",
        )

    def test_default_pairs_are_unordered_i_less_than_j(self):
        self.assertEqual(
            contact_pairs(4), [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        )
        self.assertEqual(contact_pairs(3), [(0, 1), (0, 2), (1, 2)])

    def test_explicit_pairs_keep_order(self):
        self.assertEqual(
            contact_pairs(3, pairs=[(2, 0), (1, 2)]), [(2, 0), (1, 2)]
        )

    def test_invalid_pairs(self):
        with self.assertRaises(ValueError):
            contact_pairs(3, pairs=[(0, 0)])
        with self.assertRaises(ValueError):
            contact_pairs(3, pairs=[(0, 3)])
        with self.assertRaises(ValueError):
            contact_pairs(3, pairs=[(-1, 1)])
        with self.assertRaises(ValueError):
            contact_pairs(3, pairs=[(0, 1, 2)])


class TestFrictionResidual(unittest.TestCase):
    def test_known_cone_boundary(self):
        # ||(0.3, 0.4)|| = 0.5, mu * a_z = 0.5 -> 恰好位于锥面.
        a_B = np.array([0.3, 0.4, 1.0])
        self.assertAlmostEqual(
            float(friction_residual(a_B, 0.5)), 0.0, places=12
        )
        self.assertAlmostEqual(
            float(friction_squared_residual(a_B, 0.5)), 0.0, places=12
        )

    def test_inside_and_outside(self):
        mu = 0.5
        inside = np.array([0.1, 0.1, 1.0])
        outside = np.array([0.6, 0.0, 1.0])
        self.assertLess(float(friction_residual(inside, mu)), 0.0)
        self.assertGreater(float(friction_residual(outside, mu)), 0.0)
        self.assertLess(float(friction_squared_residual(inside, mu)), 0.0)
        self.assertGreater(float(friction_squared_residual(outside, mu)), 0.0)

    def test_zero_horizontal_acceleration(self):
        a_B = np.array([0.0, 0.0, 0.2])
        self.assertAlmostEqual(
            float(friction_residual(a_B, 0.6)), -0.12, places=12
        )
        self.assertAlmostEqual(
            float(friction_squared_residual(a_B, 0.6)),
            -0.0144,
            places=12,
        )


class TestKnownFeasibleCase(unittest.TestCase):
    def setUp(self):
        self.case = reference_case()

    def test_feasible_point_satisfies_all_constraints(self):
        residuals = evaluate_residuals(
            self.case, self.case.feasible_p_ddot
        )
        self.assertTrue(residuals["feasible"])
        self.assertGreater(residuals["a_B"][2], 0.0)
        for value in residuals["giac"].values():
            self.assertLessEqual(value, FEASIBILITY_TOL)
        self.assertLessEqual(
            residuals["friction_norm"], FEASIBILITY_TOL
        )
        self.assertLessEqual(
            residuals["friction_squared"], FEASIBILITY_TOL
        )
        # 可行点应当不是把所有残差都取成 0 的平凡点.
        self.assertLess(min(residuals["giac"].values()), -1e-3)
        self.assertLess(residuals["friction_norm"], -1e-3)

    def test_violating_point_is_detected(self):
        residuals = evaluate_residuals(
            self.case, self.case.violating_p_ddot
        )
        self.assertFalse(residuals["feasible"])
        self.assertGreater(max(residuals["giac"].values()), 0.0)
        self.assertGreater(residuals["friction_norm"], 0.0)
        self.assertLess(residuals["normal_accel"], 0.0)

    def test_squared_form_alone_would_accept_negative_normal(self):
        # 说明为什么平方形式必须同时加 a_B_z >= 0:
        # a_B_z < 0 时平方残差也可以为负, 但这并不是摩擦锥内的点.
        residuals = evaluate_residuals(
            self.case, self.case.violating_p_ddot
        )
        self.assertLess(residuals["friction_squared"], 0.0)
        self.assertGreater(residuals["friction_norm"], 0.0)
        self.assertLess(residuals["normal_accel"], 0.0)


class TestConstraintAssembly(unittest.TestCase):
    def setUp(self):
        self.case = reference_case()
        self.a_B = np.asarray(
            gravity_inertial_acceleration(
                self.case.feasible_p_ddot, G_VEC
            )
        )

    def test_adds_one_constraint_per_unordered_pair(self):
        opti = RecordingOpti()
        residuals = add_giac_constraints(
            opti,
            self.case.mass,
            self.case.p_B,
            self.a_B,
            self.case.L_dot,
            self.case.feet,
        )
        self.assertEqual(len(residuals), 3)
        self.assertEqual(len(opti.constraints), 3)
        for residual in residuals:
            self.assertLessEqual(float(residual), FEASIBILITY_TOL)

    def test_requires_at_least_three_feet(self):
        opti = RecordingOpti()
        with self.assertRaises(ValueError):
            add_giac_constraints(
                opti,
                self.case.mass,
                self.case.p_B,
                self.a_B,
                self.case.L_dot,
                self.case.feet[:2],
            )
        self.assertEqual(opti.constraints, [])

    def test_friction_squared_adds_normal_constraint(self):
        opti = RecordingOpti()
        expressions = add_friction_constraint(
            opti, self.a_B, self.case.mu, form="squared"
        )
        self.assertEqual(len(expressions), 2)
        self.assertEqual(len(opti.constraints), 2)
        self.assertLessEqual(float(expressions[0]), FEASIBILITY_TOL)
        self.assertGreaterEqual(float(expressions[1]), 0.0)

    def test_squared_form_explicitly_rejects_negative_normal(self):
        # a_B_z < 0 时平方残差也可能为负, 所以必须显式加 a_B_z >= 0.
        a_B = np.asarray(
            gravity_inertial_acceleration(
                self.case.violating_p_ddot, G_VEC
            )
        )
        opti = RecordingOpti()
        expressions = add_friction_constraint(
            opti, a_B, self.case.mu, form="squared"
        )
        self.assertLess(float(expressions[0]), 0.0)
        self.assertLess(float(expressions[1]), 0.0)
        # opti.constraints[1] 是 (a_B_z >= 0) 的布尔结果.
        self.assertFalse(bool(opti.constraints[1]))

    def test_friction_norm_adds_one_constraint(self):
        opti = RecordingOpti()
        expressions = add_friction_constraint(
            opti, self.a_B, self.case.mu, form="norm"
        )
        self.assertEqual(len(expressions), 1)
        self.assertEqual(len(opti.constraints), 1)
        self.assertAlmostEqual(
            float(expressions[0]),
            float(friction_residual(self.a_B, self.case.mu)),
            places=12,
        )

    def test_norm_and_squared_have_same_sign_inside_cone(self):
        self.assertLessEqual(
            float(friction_residual(self.a_B, self.case.mu)), 0.0
        )
        self.assertLessEqual(
            float(friction_squared_residual(self.a_B, self.case.mu)), 0.0
        )

    def test_invalid_friction_form_and_mu(self):
        with self.assertRaises(ValueError):
            add_friction_constraint(
                RecordingOpti(), self.a_B, self.case.mu, form="bogus"
            )
        with self.assertRaises(ValueError):
            add_friction_constraint(
                RecordingOpti(), self.a_B, 0.0, form="norm"
            )
        with self.assertRaises(ValueError):
            add_friction_constraint(
                RecordingOpti(), self.a_B, -0.5, form="squared"
            )

    def test_explicit_pair_order_changes_sign(self):
        a_B = self.a_B
        forward = giac_residuals(
            self.case.mass,
            self.case.p_B,
            a_B,
            self.case.L_dot,
            self.case.feet,
            pairs=[(0, 1)],
        )[0][1]
        reverse = giac_residuals(
            self.case.mass,
            self.case.p_B,
            a_B,
            self.case.L_dot,
            self.case.feet,
            pairs=[(1, 0)],
        )[0][1]
        self.assertAlmostEqual(
            float(forward), -float(reverse), places=12
        )


class TestDoubleSupport(unittest.TestCase):
    """N = 2: 17c 等式 + 17d 力矩不等式 (+ 摩擦 17a)."""

    def setUp(self):
        self.mass = 10.0
        self.mu = 0.6
        self.p_B = np.array([0.2, 0.1, 0.3])
        self.a_B = np.array([0.2, -0.1, 9.0])
        self.feet = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]])
        # 同时满足 17c 与 17d 的构造值.
        self.L_dot = np.array([9.3, -30.0, 0.0])

    def _records(self, L_dot=None, mu=0.6):
        return contact_constraint_records(
            self.mass,
            self.p_B,
            self.a_B,
            self.L_dot if L_dot is None else L_dot,
            self.feet,
            mu=mu,
        )

    def test_17c_and_17d_are_satisfied(self):
        line = double_support_line_residual(
            self.mass, self.p_B, self.a_B, self.L_dot, *self.feet
        )
        moment = double_support_moment_residual(
            self.mass, self.p_B, self.a_B, self.L_dot, *self.feet
        )
        self.assertAlmostEqual(float(line), 0.0, places=12)
        self.assertLessEqual(float(moment), 0.0)

    def test_17d_matches_det_definition(self):
        p_12 = self.feet[1] - self.feet[0]
        M_1 = contact_moment(
            self.mass, self.p_B, self.a_B, self.L_dot, self.feet[0]
        )
        expected = float(det3([0.0, 0.0, 1.0], p_12, M_1))
        moment = float(
            double_support_moment_residual(
                self.mass, self.p_B, self.a_B, self.L_dot, *self.feet
            )
        )
        self.assertAlmostEqual(moment, -expected, places=12)
        self.assertGreaterEqual(expected, 0.0)

    def test_17d_paper_formula_det_ez_p_ij_M_i_ge_zero(self):
        """原文公式: det(e_z, p_ij, M_i) >= 0, 用独立 NumPy 复算.

        该测试保留论文印刷方向 (>= 0), 不依赖代码选择的残差符号.
        """
        p_1, p_2 = self.feet
        p_12 = p_2 - p_1
        M_1 = np.cross(self.p_B - p_1, self.a_B) - self.L_dot / self.mass
        paper_det = float(
            np.dot(np.array([0.0, 0.0, 1.0]), np.cross(p_12, M_1))
        )
        self.assertGreaterEqual(paper_det, 0.0)
        residual = float(
            double_support_moment_residual(
                self.mass, self.p_B, self.a_B, self.L_dot, *self.feet
            )
        )
        self.assertAlmostEqual(residual, -paper_det, places=12)

        # 违反原文方向的构造: 同一个 det 变为负.
        L_dot_bad = np.array([9.3, 0.0, 0.0])
        M_bad = np.cross(self.p_B - p_1, self.a_B) - L_dot_bad / self.mass
        bad_det = float(
            np.dot(np.array([0.0, 0.0, 1.0]), np.cross(p_12, M_bad))
        )
        self.assertLess(bad_det, 0.0)
        self.assertGreater(
            float(
                double_support_moment_residual(
                    self.mass, self.p_B, self.a_B, L_dot_bad, *self.feet
                )
            ),
            0.0,
        )

    def test_17d_engineering_interpretation_matches_paper(self):
        """工程/参考实现写法与论文写法在 a_B = g - p_ddot 下等价.

        参考实现里 ``a_B`` 变量实际是 ``p_ddot``, 其力矩写成
        ``(p_B - p_i) x (g - a_ref) - L_dot / m``. 把
        ``a_ref = p_ddot``, ``a_B = g - p_ddot`` 代入后两者逐项相等.
        """
        p_ddot = np.array([0.5, -0.2, 3.0])
        g = np.array([0.0, 0.0, -9.81])
        a_B_paper = g - p_ddot
        p_1, p_2 = self.feet
        M_paper = (
            np.cross(self.p_B - p_1, a_B_paper) - self.L_dot / self.mass
        )
        M_reference = (
            np.cross(self.p_B - p_1, g - p_ddot)
            - self.L_dot / self.mass
        )
        np.testing.assert_allclose(M_paper, M_reference, atol=1e-15)

        paper_det = float(
            np.dot(
                np.array([0.0, 0.0, 1.0]),
                np.cross(p_2 - p_1, M_paper),
            )
        )
        residual = float(
            double_support_moment_residual(
                self.mass, self.p_B, a_B_paper, self.L_dot, p_1, p_2
            )
        )
        self.assertAlmostEqual(residual, -paper_det, places=12)

    def test_17d_alternative_moment_sign_is_not_silently_accepted(self):
        """把 M_i 整体取反会翻转不等式方向.

        论文方向固定为 det(e_z, p_ij, M_i) >= 0; 代码只实现这一种.
        该测试明确记录相反符号约定会得到 ``-det``, 防止后续重构时
        把两个方向混用. 注意只翻转 ``(p_B - p_i)`` 而保留
        ``-L_dot/m`` 并不是整体取反, 数值会同时依赖 L_dot; 这里比较
        的是整体符号约定.
        """
        p_1, p_2 = self.feet
        p_12 = p_2 - p_1
        M_paper = np.cross(self.p_B - p_1, self.a_B) - self.L_dot / self.mass
        M_opposite_convention = -M_paper
        ez = np.array([0.0, 0.0, 1.0])
        det_paper = float(np.dot(ez, np.cross(p_12, M_paper)))
        det_opposite = float(
            np.dot(ez, np.cross(p_12, M_opposite_convention))
        )
        self.assertAlmostEqual(det_opposite, -det_paper, places=12)
        self.assertGreater(det_paper, 0.0)

        # 只翻转 (p_B - p_i)、不翻转 L_dot 项会得到第三种结果,
        # 不能当作整体的符号约定.
        M_partial = (
            np.cross(p_1 - self.p_B, self.a_B) - self.L_dot / self.mass
        )
        det_partial = float(np.dot(ez, np.cross(p_12, M_partial)))
        self.assertNotAlmostEqual(det_partial, -det_paper, places=6)
        self.assertNotAlmostEqual(det_partial, det_paper, places=6)

    def test_violating_17c_is_detected(self):
        # 只有线动量等式被破坏; 17d 仍然满足.
        L_dot = np.array([0.0, -30.0, 0.0])
        line = float(
            double_support_line_residual(
                self.mass, self.p_B, self.a_B, L_dot, *self.feet
            )
        )
        moment = float(
            double_support_moment_residual(
                self.mass, self.p_B, self.a_B, L_dot, *self.feet
            )
        )
        self.assertGreater(line, 1e-6)
        self.assertLessEqual(moment, 0.0)

    def test_violating_17d_is_detected(self):
        # 17c 仍成立, 但 17d 的 det 为负.
        L_dot = np.array([9.3, 0.0, 0.0])
        line = float(
            double_support_line_residual(
                self.mass, self.p_B, self.a_B, L_dot, *self.feet
            )
        )
        moment = float(
            double_support_moment_residual(
                self.mass, self.p_B, self.a_B, L_dot, *self.feet
            )
        )
        self.assertAlmostEqual(line, 0.0, places=12)
        self.assertGreater(moment, 1e-6)

    def test_swapping_feet_flips_17c_sign(self):
        # 17c 的左端严格反对称. 17d 不同: 它的参考点 p_1 也随脚的
        # 顺序改变, 所以不能假设 17d 只是变号.
        L_dot = np.array([0.0, -30.0, 0.0])
        forward = float(
            double_support_line_residual(
                self.mass, self.p_B, self.a_B, L_dot, *self.feet
            )
        )
        reverse = float(
            double_support_line_residual(
                self.mass,
                self.p_B,
                self.a_B,
                L_dot,
                self.feet[1],
                self.feet[0],
            )
        )
        self.assertAlmostEqual(forward, -reverse, places=12)
        self.assertGreater(abs(forward), 1e-6)

        # 同一组脚下 17d 的数值随参考点改变, 不是简单取反.
        moment_forward = float(
            double_support_moment_residual(
                self.mass, self.p_B, self.a_B, self.L_dot, *self.feet
            )
        )
        moment_reverse = float(
            double_support_moment_residual(
                self.mass,
                self.p_B,
                self.a_B,
                self.L_dot,
                self.feet[1],
                self.feet[0],
            )
        )
        self.assertNotAlmostEqual(
            moment_forward, -moment_reverse, places=6
        )

    def test_records_contain_only_double_support_constraints(self):
        records = self._records()
        self.assertEqual(records.num_contacts, 2)
        self.assertEqual(set(records.by_category()), {"17c", "17d", "17a"})
        self.assertEqual(len(records.equalities), 1)
        self.assertEqual(records.equalities[0].category, "17c")
        self.assertEqual(records.equalities[0].kind, EQUALITY)
        self.assertEqual(records.equalities[0].name, "17c_line")
        self.assertEqual(len(records.inequalities), 3)
        self.assertEqual(records.by_category()["17d"][0].kind, INEQUALITY)
        values = records.residual_values()
        self.assertAlmostEqual(values["17c_line"], 0.0, places=12)
        self.assertLessEqual(values["17d_moment"], 0.0)
        self.assertTrue(all(r.is_satisfied() for r in records))

    def test_add_contact_constraints_assembly(self):
        opti = RecordingOpti()
        records = add_contact_constraints(
            opti,
            self.mass,
            self.p_B,
            self.a_B,
            self.L_dot,
            self.feet,
            mu=self.mu,
        )
        self.assertEqual(len(records), 4)
        self.assertEqual(len(opti.constraints), 4)

    def test_mu_required_when_contacts_exist(self):
        with self.assertRaises(ValueError):
            contact_constraint_records(
                self.mass,
                self.p_B,
                self.a_B,
                self.L_dot,
                self.feet,
                mu=None,
            )

    def test_pairs_not_allowed_for_two_contacts(self):
        with self.assertRaises(ValueError):
            contact_constraint_records(
                self.mass,
                self.p_B,
                self.a_B,
                self.L_dot,
                self.feet,
                mu=self.mu,
                pairs=[(0, 1)],
            )


class TestSingleSupport(unittest.TestCase):
    """N = 1: 17e 向量等式拆成 3 条分量等式 (+ 摩擦 17a)."""

    def setUp(self):
        self.mass = 10.0
        self.mu = 0.6
        self.p_B = np.array([0.2, 0.1, 0.3])
        self.a_B = np.array([0.5, -0.2, 9.0])
        self.p_i = np.array([0.1, -0.05, 0.0])
        moment = np.cross(self.p_B - self.p_i, self.a_B)
        # 令 17e 恰好成立: L_dot = m (p_B - p_i) x a_B.
        self.L_dot = self.mass * moment

    def _records(self, L_dot=None, mu=0.6):
        return contact_constraint_records(
            self.mass,
            self.p_B,
            self.a_B,
            self.L_dot if L_dot is None else L_dot,
            [self.p_i],
            mu=mu,
        )

    def test_vector_equation_is_satisfied(self):
        residuals = single_support_residuals(
            self.mass, self.p_B, self.a_B, self.L_dot, self.p_i
        )
        for residual in residuals:
            self.assertAlmostEqual(float(residual), 0.0, places=12)

    def test_vector_equation_is_split_into_components(self):
        records = self._records()
        names = [record.name for record in records.equalities]
        self.assertEqual(names, ["17e_x", "17e_y", "17e_z"])
        self.assertTrue(
            all(record.kind == EQUALITY for record in records.equalities)
        )
        self.assertEqual(len(records.equalities), 3)

    def test_only_violated_component_is_nonzero(self):
        # 只改动 L_dot_x, 应当只让 x 分量残差非零.
        L_dot = self.L_dot + np.array([1.0, 0.0, 0.0])
        values = self._records(L_dot=L_dot).residual_values()
        self.assertAlmostEqual(values["17e_x"], -1.0, places=12)
        self.assertAlmostEqual(values["17e_y"], 0.0, places=12)
        self.assertAlmostEqual(values["17e_z"], 0.0, places=12)
        self.assertFalse(
            self._records(L_dot=L_dot).by_category()["17e"][0].is_satisfied()
        )

    def test_records_contain_only_single_support_constraints(self):
        records = self._records()
        self.assertEqual(set(records.by_category()), {"17e", "17a"})
        self.assertEqual(len(records.inequalities), 2)
        self.assertTrue(all(r.is_satisfied() for r in records))

    def test_add_contact_constraints_assembly(self):
        opti = RecordingOpti()
        records = add_contact_constraints(
            opti,
            self.mass,
            self.p_B,
            self.a_B,
            self.L_dot,
            [self.p_i],
            mu=self.mu,
        )
        self.assertEqual(len(records), 5)
        self.assertEqual(len(opti.constraints), 5)
        self.assertEqual(len(records.equalities), 3)
        self.assertEqual(len(records.inequalities), 2)

    def test_mu_required_when_contacts_exist(self):
        with self.assertRaises(ValueError):
            contact_constraint_records(
                self.mass,
                self.p_B,
                self.a_B,
                self.L_dot,
                [self.p_i],
                mu=None,
            )


class TestFlight(unittest.TestCase):
    """N = 0: 17f 弹道条件, 拆成 6 条分量等式, 不加摩擦."""

    def setUp(self):
        self.mass = 10.0
        self.p_B = np.array([0.2, 0.1, 0.3])
        self.a_B = np.zeros(3)
        self.L_dot = np.zeros(3)

    def _records(self, a_B=None, L_dot=None, mu=None):
        return contact_constraint_records(
            self.mass,
            self.p_B,
            self.a_B if a_B is None else a_B,
            self.L_dot if L_dot is None else L_dot,
            [],
            mu=mu,
        )

    def test_ballistic_conditions_hold(self):
        residuals = ballistic_residuals(self.a_B, self.L_dot)
        self.assertEqual(len(residuals), 6)
        for residual in residuals:
            self.assertAlmostEqual(float(residual), 0.0, places=12)

    def test_conditions_are_split_into_six_equalities(self):
        records = self._records()
        self.assertEqual(records.num_contacts, 0)
        self.assertEqual(len(records), 6)
        self.assertEqual(len(records.equalities), 6)
        self.assertEqual(records.inequalities, ())
        self.assertEqual(
            [record.name for record in records.records],
            [
                "17f_a_B_x",
                "17f_a_B_y",
                "17f_a_B_z",
                "17f_L_dot_x",
                "17f_L_dot_y",
                "17f_L_dot_z",
            ],
        )
        self.assertEqual(set(records.by_category()), {"17f"})

    def test_each_component_violation_is_detected(self):
        records = self._records(
            a_B=np.array([0.0, 0.0, 0.1]),
            L_dot=np.array([0.2, 0.0, 0.0]),
        )
        values = records.residual_values()
        self.assertAlmostEqual(values["17f_a_B_z"], 0.1, places=12)
        self.assertAlmostEqual(values["17f_L_dot_x"], 0.2, places=12)
        satisfied = {
            record.name: record.is_satisfied() for record in records
        }
        self.assertFalse(satisfied["17f_a_B_z"])
        self.assertFalse(satisfied["17f_L_dot_x"])
        self.assertEqual(
            sum(not value for value in satisfied.values()), 2
        )

    def test_no_friction_during_flight(self):
        with_mu = self._records(mu=0.6)
        without_mu = self._records(mu=None)
        self.assertEqual(set(with_mu.by_category()), {"17f"})
        self.assertEqual(
            with_mu.residual_values(), without_mu.residual_values()
        )

    def test_add_contact_constraints_assembly(self):
        opti = RecordingOpti()
        records = add_contact_constraints(
            opti,
            self.mass,
            self.p_B,
            self.a_B,
            self.L_dot,
            [],
            mu=None,
        )
        self.assertEqual(len(records), 6)
        self.assertEqual(len(opti.constraints), 6)

    def test_invalid_friction_form_still_rejected(self):
        with self.assertRaises(ValueError):
            contact_constraint_records(
                self.mass,
                self.p_B,
                self.a_B,
                self.L_dot,
                [],
                mu=None,
                friction_form="bogus",
            )


class TestContactStateDispatch(unittest.TestCase):
    """按 len(feet) 分派, 各状态互不复用."""

    def test_four_contacts_use_only_17b(self):
        feet = np.array(
            [
                [0.2, 0.0, 0.0],
                [0.0, 0.2, 0.0],
                [-0.2, 0.0, 0.0],
                [0.0, -0.2, 0.0],
            ]
        )
        records = contact_constraint_records(
            10.0,
            np.zeros(3),
            np.array([0.0, 0.0, 9.81]),
            np.zeros(3),
            feet,
            mu=0.6,
        )
        self.assertEqual(records.num_contacts, 4)
        self.assertEqual(len(records.by_category()["17b"]), 6)
        self.assertTrue(
            all(
                record.kind == INEQUALITY
                for record in records.by_category()["17b"]
            )
        )
        self.assertEqual(set(records.by_category()), {"17b", "17a"})

    def test_states_do_not_overlap(self):
        a_B = np.array([0.0, 0.0, 9.81])
        L_dot = np.zeros(3)
        cases = {
            3: np.array([[0.2, 0.0, 0.0], [0.0, 0.2, 0.0],
                         [-0.2, 0.0, 0.0]]),
            2: np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]]),
            1: np.array([[0.0, 0.0, 0.0]]),
            0: np.empty((0, 3)),
        }
        expected = {
            3: {"17b", "17a"},
            2: {"17c", "17d", "17a"},
            1: {"17e", "17a"},
            0: {"17f"},
        }
        for num_contacts, feet in cases.items():
            records = contact_constraint_records(
                10.0, np.zeros(3), a_B, L_dot, feet, mu=0.6
            )
            self.assertEqual(
                set(records.by_category()),
                expected[num_contacts],
                msg=f"N={num_contacts}",
            )

    def test_invalid_vector_shapes(self):
        a_B = np.array([0.0, 0.0, 9.81])
        L_dot = np.zeros(3)
        feet = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]])
        with self.assertRaises(ValueError):
            contact_constraint_records(
                10.0, np.zeros(2), a_B, L_dot, feet, mu=0.6
            )
        with self.assertRaises(ValueError):
            contact_constraint_records(
                10.0, np.zeros(3), a_B[:2], L_dot, feet, mu=0.6
            )
        with self.assertRaises(ValueError):
            contact_constraint_records(
                10.0, np.zeros(3), a_B, L_dot[:2], feet, mu=0.6
            )
        with self.assertRaises(ValueError):
            contact_constraint_records(
                10.0,
                np.zeros(3),
                a_B,
                L_dot,
                [np.zeros(3), np.zeros(2)],
                mu=0.6,
            )

    def test_missing_mu_rejected_for_any_contact(self):
        a_B = np.array([0.0, 0.0, 9.81])
        for feet in (
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]]),
            np.array(
                [[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [-0.2, 0.0, 0.0]]
            ),
        ):
            with self.assertRaises(ValueError):
                contact_constraint_records(
                    10.0, np.zeros(3), a_B, np.zeros(3), feet, mu=None
                )

    @unittest.skipUnless(CASADI_AVAILABLE, "CasADi 未安装")
    def test_casadi_accepts_all_contact_states(self):
        ca = giac_example.ca
        for feet in (
            np.array(
                [[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [-0.2, 0.0, 0.0]]
            ),
            np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0]]),
            np.empty((0, 3)),
        ):
            opti = ca.Opti()
            # 用变量而不是常数: CasADi 的 subject_to 不接受常值约束.
            variable_a_B = opti.variable(3)
            variable_L_dot = opti.variable(3)
            opti.set_initial(variable_a_B, [0.0, 0.0, 9.81])
            opti.set_initial(variable_L_dot, [0.0, 0.0, 0.0])
            records = add_contact_constraints(
                opti,
                10.0,
                np.zeros(3),
                variable_a_B,
                variable_L_dot,
                feet,
                mu=0.6,
            )
            self.assertGreater(len(records), 0)


@unittest.skipUnless(CASADI_AVAILABLE, "CasADi 未安装")
class TestIPOPTMinimalExample(unittest.TestCase):
    def test_invalid_objective(self):
        with self.assertRaises(ValueError):
            giac_example.build_problem(
                reference_case(), objective="bogus"
            )

    def test_norm_friction_form_builds(self):
        case = reference_case()
        _, _, _, giac_exprs, friction_exprs = (
            giac_example.build_problem(
                case, objective="norm", friction_form="norm"
            )
        )
        self.assertEqual(len(giac_exprs), 3)
        self.assertEqual(len(friction_exprs), 1)

    def test_ipopt_finds_feasible_point(self):
        case = reference_case()
        opti, p_B_ddot, a_B, giac_exprs, friction_exprs = (
            giac_example.build_problem(case)
        )
        self.assertEqual(len(giac_exprs), 3)
        self.assertEqual(len(friction_exprs), 2)

        solution = opti.solve()
        stats = solution.stats()
        self.assertTrue(bool(stats.get("success", False)))

        p_ddot = np.asarray(
            solution.value(p_B_ddot), dtype=float
        ).reshape(3)
        residuals = evaluate_residuals(case, p_ddot)
        self.assertTrue(residuals["feasible"])

        # 默认目标 min ||p_ddot - p_ref||^2, p_ref 严格可行,
        # 因此最优点就是 p_ref, 且所有约束都有非零余量.
        np.testing.assert_allclose(
            p_ddot, case.feasible_p_ddot, atol=1e-6
        )
        self.assertLess(max(residuals["giac"].values()), -1e-3)
        self.assertLess(residuals["friction_norm"], -1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
