"""V-A 约束 (式 30 / 式 5 + V-A-4 / 式 31) 的单元测试.

运行:
    python -m unittest myrobot_mjlab.task.DTC.tamols.test_v_a_constraints -v
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from myrobot_mjlab.task.DTC.tamols import solver_backend
from myrobot_mjlab.task.DTC.tamols.giac_constraints import (
    friction_squared_residual,
)
from myrobot_mjlab.task.DTC.tamols.v_a_constraints import (
    CCW_INDICES,
    CCW_LEG_ORDER,
    DEFAULT_HIP_OFFSETS,
    DEFAULT_L_MAX,
    DEFAULT_L_MIN,
    EZ,
    GRAVITY,
    GO2_HIP_JOINT_OFFSETS,
    GO2_STANCE_HIP_OFFSETS,
    LEG_ORDER,
    VAConstraint,
    VAConstraintSet,
    add_convexity_constraints,
    add_friction_push_constraints,
    add_reachability_constraints,
    add_spline_phase_friction_constraints,
    add_spline_phase_reachability_constraints,
    complete_footprint,
    convexity_fourth_residual,
    convexity_records,
    convexity_records_from_events,
    convexity_residuals,
    counterclockwise_order,
    FootholdEvent,
    friction_push_records,
    IncompleteFootprintError,
    leg_index,
    matvec3,
    reachability_records,
    reachability_residuals,
    rotation_matrix_zyx,
    spline_base_acceleration,
    spline_value,
)

CASADI_AVAILABLE = solver_backend.ca is not None

# 工程名义站姿: 索引顺序 FL, FR, RL, RR; z 取任意固定高度.
GO2_FEET = np.array(
    [
        [0.1934, 0.145, -0.28],   # FL
        [0.1934, -0.145, -0.28],  # FR
        [-0.1934, 0.145, -0.28],  # RL
        [-0.1934, -0.145, -0.28], # RR
    ],
    dtype=float,
)

# 逆时针单位正方形: 数组顺序本身就是 CCW.
SQUARE = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)
CCW_SQUARE_ORDER = (0, 1, 2, 3)


def signed_area(points) -> float:
    points = np.asarray(points, dtype=float)
    x, y = points[:, 0], points[:, 1]
    return float(
        0.5
        * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    )


def explicit_rotation(roll, pitch, yaw) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


# ---------------------------------------------------------------------- #
# 足端编号 / 绕向
# ---------------------------------------------------------------------- #
class TestLegOrdering(unittest.TestCase):
    def test_leg_order_matches_project(self):
        self.assertEqual(LEG_ORDER, ("FL", "FR", "RL", "RR"))
        self.assertEqual(
            [leg_index(name) for name in LEG_ORDER], [0, 1, 2, 3]
        )

    def test_leg_index_accepts_names_and_indices(self):
        self.assertEqual(leg_index("FL"), 0)
        self.assertEqual(leg_index("rr"), 3)
        self.assertEqual(leg_index(2), 2)
        with self.assertRaises(ValueError):
            leg_index("XX")
        with self.assertRaises(ValueError):
            leg_index(4)

    def test_ccw_order_is_counterclockwise(self):
        self.assertEqual(CCW_LEG_ORDER, ("FL", "RL", "RR", "FR"))
        self.assertEqual(CCW_INDICES, (0, 2, 3, 1))
        ordered = GO2_FEET[list(CCW_INDICES)]
        self.assertGreater(signed_area(ordered), 0.0)
        self.assertAlmostEqual(signed_area(ordered), 0.112172, places=6)

    def test_raw_project_order_is_a_bowtie(self):
        """FL,FR,RL,RR 直接相连是自交的, 有向面积为 0."""
        self.assertAlmostEqual(signed_area(GO2_FEET), 0.0, places=12)
        with self.assertRaises(ValueError):
            convexity_records(GO2_FEET, order=(0, 1, 2, 3))

    def test_counterclockwise_order_returns_ccw_permutation(self):
        order = counterclockwise_order(GO2_FEET)
        self.assertEqual(sorted(order), [0, 1, 2, 3])
        self.assertGreater(signed_area(GO2_FEET[list(order)]), 0.0)

    def test_counterclockwise_order_requires_numeric_points(self):
        if not CASADI_AVAILABLE:
            self.skipTest("CasADi 未安装")
        ca = solver_backend.ca
        points = ca.MX.sym("points", 4, 3)
        with self.assertRaises(ValueError):
            counterclockwise_order([points[i, :] for i in range(4)])


# ---------------------------------------------------------------------- #
# 式 (30): 足迹凸性
# ---------------------------------------------------------------------- #
class TestConvexity(unittest.TestCase):
    def test_square_in_ccw_order_passes(self):
        residuals = convexity_residuals(*SQUARE)
        np.testing.assert_allclose(residuals, [-1.0, -1.0, -1.0, -1.0])
        result = convexity_records(
            SQUARE, order=CCW_SQUARE_ORDER, require_ccw=True
        )
        self.assertEqual(len(result), 4)
        self.assertTrue(result.is_satisfied())

    def test_nominal_go2_footprint_passes_with_default_order(self):
        result = convexity_records(GO2_FEET)
        self.assertEqual(len(result), 4)
        self.assertTrue(result.is_satisfied())
        self.assertIn("winding=ccw", result.note)
        # 矩形四条边的对应残差都是 -w*h.
        area = signed_area(GO2_FEET[list(CCW_INDICES)])
        np.testing.assert_allclose(
            [float(record.residual) for record in result],
            [-area, -area, -area, -area],
            atol=1e-12,
        )

    def test_records_carry_names_indices_and_bounds(self):
        result = convexity_records(GO2_FEET)
        names = [record.name for record in result]
        self.assertEqual(
            names,
            [
                "30_convex_p13_p12",
                "30_convex_p14_p13",
                "30_convex_p24_p23",
                "30_convex_p21_p24",
            ],
        )
        self.assertEqual(result.by_category()["30"], tuple(result.records))
        self.assertEqual(result.records[0].foot_indices, (0, 3, 2))
        self.assertEqual(result.records[3].foot_indices, (2, 0, 1))
        for record in result:
            self.assertEqual(record.category, "30")
            self.assertEqual(record.bound, 0.0)

    def test_concave_quad_is_detected(self):
        concave = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.4, 0.4, 0.0],  # 位于三角形 p1-p2-p4 内部
                [0.0, 1.0, 0.0],
            ]
        )
        self.assertGreater(signed_area(concave), 0.0)
        result = convexity_records(
            concave, order=CCW_SQUARE_ORDER
        )
        self.assertFalse(result.is_satisfied())
        # 第三项 (p24 x p23) 检测到 p3 的凹陷.
        self.assertAlmostEqual(
            float(result.records[2].residual), 0.2, places=12
        )
        self.assertAlmostEqual(result.max_violation(), 0.2, places=12)

    def test_reversed_order_rejected_by_default(self):
        with self.assertRaises(ValueError):
            convexity_records(SQUARE, order=(0, 3, 2, 1))

    def test_reversed_order_violates_when_check_disabled(self):
        result = convexity_records(
            SQUARE, order=(0, 3, 2, 1), require_ccw=False
        )
        self.assertFalse(result.is_satisfied())
        np.testing.assert_allclose(
            [float(record.residual) for record in result],
            [1.0, 1.0, 1.0, 1.0],
        )

    def test_collinear_three_points_is_a_boundary_case(self):
        """三共线时一条残差为 0: 不等式允许, 但不是严格凸."""
        degenerate = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        self.assertGreater(signed_area(degenerate), 0.0)
        result = convexity_records(
            degenerate, order=CCW_SQUARE_ORDER
        )
        self.assertEqual(float(result.records[0].residual), 0.0)
        self.assertTrue(result.is_satisfied())
        self.assertLess(float(result.records[1].residual), 0.0)

    def test_four_collinear_points_rejected_by_default(self):
        line = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        )
        with self.assertRaises(ValueError):
            convexity_records(line, order=CCW_SQUARE_ORDER)
        result = convexity_records(
            line, order=CCW_SQUARE_ORDER, require_ccw=False
        )
        np.testing.assert_allclose(
            [float(record.residual) for record in result],
            [0.0, 0.0, 0.0, 0.0],
        )

    def test_order_must_be_permutation(self):
        with self.assertRaises(ValueError):
            convexity_records(SQUARE, order=(0, 1, 2))
        with self.assertRaises(ValueError):
            convexity_records(SQUARE, order=(0, 1, 2, 2))

    def test_point_count_must_be_four(self):
        with self.assertRaises(ValueError):
            convexity_records(SQUARE[:3], order=(0, 1, 2))

    def test_foot_sources_are_propagated(self):
        sources = ("stance", "phase 2", "phase 3", "stance")
        result = convexity_records(GO2_FEET, foot_sources=sources)
        for record in result:
            self.assertEqual(len(record.foot_sources), 3)
            self.assertEqual(
                record.foot_sources,
                tuple(sources[i] for i in record.foot_indices),
            )


class TestConvexityP20Ambiguity(unittest.TestCase):
    """专门验证式 (30) 第四项印刷的 p20 为什么是 p21 的笔误."""

    @staticmethod
    def _p(v):
        return (float(v[0]), float(v[1]))

    def test_adopted_fourth_term_is_p21_cross_p24(self):
        p1, p2, p3, p4 = (self._p(p) for p in SQUARE)
        p21 = (p1[0] - p2[0], p1[1] - p2[1])
        p24 = (p4[0] - p2[0], p4[1] - p2[1])
        expected = p21[0] * p24[1] - p21[1] * p24[0]
        value = float(
            convexity_fourth_residual(SQUARE[0], SQUARE[1], SQUARE[2], SQUARE[3])
        )
        self.assertAlmostEqual(value, expected, places=12)
        self.assertAlmostEqual(value, -1.0, places=12)

    def test_paper_printed_p20_formula_is_kept(self):
        """保留原文 (p20 x p24) 的字面读法, 并固定它的行为.

        论文没有定义 "脚 0". 唯一字面读法是把 p0 当成世界原点,
        即 ``p20 = -p2``. 该式在平移下会改变符号, 因此不可能是有意义
        的凸性条件; 本测试与采用 p21 的工程解释分开保留, 避免只测试
        agent 选择的版本.
        """

        def paper_printed_term(points):
            p20 = (-float(points[1][0]), -float(points[1][1]))
            p24 = (
                float(points[3][0] - points[1][0]),
                float(points[3][1] - points[1][1]),
            )
            return p20[0] * p24[1] - p20[1] * p24[0]

        self.assertAlmostEqual(paper_printed_term(SQUARE), -1.0, places=12)
        shifted = SQUARE + np.array([-10.0, 0.0, 0.0])
        self.assertGreater(paper_printed_term(shifted), 0.0)
        # 同一平移下工程采用的 (p21 x p24) 不变, 两者是不同的约束.
        self.assertAlmostEqual(
            float(convexity_fourth_residual(*shifted)), -1.0, places=12
        )

    def test_adopted_term_is_translation_invariant(self):
        base = float(convexity_fourth_residual(*SQUARE))
        for shift in ((10.0, -7.0, 3.0), (-10.0, 0.0, 0.0)):
            moved = SQUARE + np.asarray(shift)
            value = float(convexity_fourth_residual(*moved))
            self.assertAlmostEqual(value, base, places=12)

    def test_origin_interpretation_of_p20_is_not_translation_invariant(self):
        def origin_term(points):
            # p20 = p0 - p2, 把不存在的脚 0 当成世界原点.
            p20 = (0.0 - points[1][0], 0.0 - points[1][1])
            p24 = (
                points[3][0] - points[1][0],
                points[3][1] - points[1][1],
            )
            return p20[0] * p24[1] - p20[1] * p24[0]

        base = origin_term(SQUARE)
        shifted = origin_term(SQUARE + np.array([-10.0, 0.0, 0.0]))
        self.assertAlmostEqual(base, -1.0, places=12)
        self.assertGreater(shifted, 0.0)
        self.assertNotAlmostEqual(base, shifted, places=6)

    def test_p0_as_p4_makes_the_term_vacuous(self):
        # 若 p0 解释成第 4 个脚, p20 = p4 - p2 = p24, 叉积恒为 0.
        points = SQUARE + np.array([[0.0, 0.0, 0.0]])
        p20 = (points[3][0] - points[1][0], points[3][1] - points[1][1])
        p24 = (points[3][0] - points[1][0], points[3][1] - points[1][1])
        self.assertEqual(p20[0] * p24[1] - p20[1] * p24[0], 0.0)

    def test_reference_expression_has_opposite_sign(self):
        # 参考实现用的 (p24 x p21), 对逆时针正方形会给出 +1, 即误判为凸.
        p1, _, _, p4 = (self._p(p) for p in SQUARE)
        p2 = (1.0, 0.0)
        p24 = (p4[0] - p2[0], p4[1] - p2[1])
        p21 = (p1[0] - p2[0], p1[1] - p2[1])
        reference = p24[0] * p21[1] - p24[1] * p21[0]
        self.assertAlmostEqual(reference, 1.0, places=12)
        self.assertGreater(
            reference, float(convexity_fourth_residual(*SQUARE))
        )


# ---------------------------------------------------------------------- #
# 非完整支撑: 未来相位补齐
# ---------------------------------------------------------------------- #
class TestFootprintCompletion(unittest.TestCase):
    def test_completes_from_future_phase_events(self):
        stance = {
            "FL": GO2_FEET[0],
            "FR": GO2_FEET[1],
        }
        events = [
            FootholdEvent("RR", GO2_FEET[3], phase=3),
            FootholdEvent("RL", GO2_FEET[2], phase=2),
        ]
        completed = complete_footprint(stance, events)
        self.assertEqual(completed.sources, ("stance", "stance", "phase 2", "phase 3"))
        self.assertEqual(completed.phases, (None, None, 2, 3))
        for got, expected in zip(completed.points, GO2_FEET):
            np.testing.assert_allclose(got, expected)
        self.assertEqual(completed.source_of("RL"), "phase 2")

    def test_earliest_future_event_wins(self):
        stance = {"FL": GO2_FEET[0], "FR": GO2_FEET[1], "RR": GO2_FEET[3]}
        events = [
            FootholdEvent("RL", [9.0, 9.0, 9.0], phase=5),
            FootholdEvent("RL", GO2_FEET[2], phase=2),
        ]
        completed = complete_footprint(stance, events)
        np.testing.assert_allclose(completed.points[2], GO2_FEET[2])
        self.assertEqual(completed.sources[2], "phase 2")

    def test_stance_foothold_is_not_overwritten(self):
        stance = {"FL": GO2_FEET[0], "FR": GO2_FEET[1], "RL": GO2_FEET[2]}
        events = [FootholdEvent("RL", [9.0, 9.0, 9.0], phase=1),
                  FootholdEvent("RR", GO2_FEET[3], phase=4)]
        completed = complete_footprint(stance, events)
        np.testing.assert_allclose(completed.points[2], GO2_FEET[2])
        self.assertEqual(completed.sources[2], "stance")

    def test_incomplete_footprint_raises(self):
        stance = {"FL": GO2_FEET[0]}
        events = [FootholdEvent("FR", GO2_FEET[1], phase=2)]
        with self.assertRaises(IncompleteFootprintError) as ctx:
            complete_footprint(stance, events)
        message = str(ctx.exception)
        self.assertIn("RL", message)
        self.assertIn("RR", message)

    def test_convexity_from_events_records_sources(self):
        stance = {"FL": GO2_FEET[0], "FR": GO2_FEET[1]}
        events = [
            FootholdEvent("RR", GO2_FEET[3], phase=3),
            FootholdEvent("RL", GO2_FEET[2], phase=2),
        ]
        result = convexity_records_from_events(
            stance, events, phase=1
        )
        self.assertEqual(len(result), 4)
        self.assertTrue(result.is_satisfied())
        self.assertIn("FL<-stance", result.note)
        self.assertIn("RL<-phase 2", result.note)
        self.assertIn("RR<-phase 3", result.note)
        for record in result:
            self.assertTrue(record.foot_sources)
            self.assertEqual(record.phase, 1)

    def test_events_out_of_order_are_sorted(self):
        stance = {"FL": GO2_FEET[0], "FR": GO2_FEET[1]}
        events = [
            FootholdEvent("RR", GO2_FEET[3], phase=7),
            FootholdEvent("RL", GO2_FEET[2], phase=4),
        ]
        completed = complete_footprint(stance, events)
        self.assertEqual(completed.phases, (None, None, 4, 7))


# ---------------------------------------------------------------------- #
# 式 (5) / V-A-4: 摩擦锥与推地
# ---------------------------------------------------------------------- #
class TestFrictionPush(unittest.TestCase):
    def test_pure_vertical_up_acceleration_is_inside(self):
        # 竖直方向的 a_B 只检验摩擦锥; 推地条件在字面约定下与它冲突,
        # 由 test_literal_forms_are_jointly_inconsistent 单独覆盖.
        result = friction_push_records(
            (0.0, 0.0, 5.0),
            (0.0, 0.0, -14.81),
            mu=0.6,
            include_push=False,
        )
        self.assertTrue(result.is_satisfied())
        values = result.residual_values()
        # 平方形式: 0 - (0.6 * 5)^2 = -9.
        self.assertAlmostEqual(values["5_friction_cone"], -9.0)
        self.assertAlmostEqual(values["5_friction_normal"], -5.0)

    def test_negative_normal_is_rejected(self):
        result = friction_push_records(
            (0.0, 0.0, -1.0),
            (0.0, 0.0, -8.81),
            mu=0.6,
            include_push=False,
        )
        self.assertFalse(result.is_satisfied())
        self.assertAlmostEqual(
            result.residual_values()["5_friction_normal"], 1.0
        )

    def test_cone_boundary_inside_and_outside(self):
        mu = 0.6
        boundary = friction_push_records(
            (0.6, 0.0, 1.0), (0.0, 0.0, -10.81), mu=mu,
            include_push=False,
        )
        inside = friction_push_records(
            (0.5, 0.0, 1.0), (0.0, 0.0, -10.81), mu=mu,
            include_push=False,
        )
        outside = friction_push_records(
            (0.7, 0.0, 1.0), (0.0, 0.0, -10.81), mu=mu,
            include_push=False,
        )
        self.assertAlmostEqual(
            boundary.residual_values()["5_friction_cone"], 0.0, places=12
        )
        self.assertAlmostEqual(
            inside.residual_values()["5_friction_cone"], -0.11, places=12
        )
        self.assertAlmostEqual(
            outside.residual_values()["5_friction_cone"], 0.13, places=12
        )

    def test_norm_form_matches_squared_form(self):
        mu = 0.6
        for point in ((0.5, 0.2, 1.0), (0.6, 0.0, 1.0), (0.0, 0.0, 2.0)):
            norm = friction_push_records(
                point, (0, 0, 0), mu=mu, form="norm", include_push=False
            )
            squared = friction_push_records(
                point, (0, 0, 0), mu=mu, form="squared", include_push=False
            )
            self.assertTrue(norm.is_satisfied())
            self.assertTrue(squared.is_satisfied())

    def test_squared_form_without_normal_admits_false_solutions(self):
        # 只写平方形式时 a_B_z < 0 也会 "满足"; 这就是必须加
        # a_B_z >= 0 的原因.
        a_B = (0.1, 0.0, -1.0)
        self.assertLess(friction_squared_residual(a_B, 0.6), 0.0)
        norm = friction_push_records(
            a_B, (0, 0, 0), mu=0.6, form="norm", include_push=False
        )
        self.assertFalse(norm.is_satisfied())

    def test_push_boundary(self):
        result = friction_push_records(
            (0.0, 0.0, 0.0), (0.0, 0.0, -9.81), mu=0.6,
            include_push=True,
        )
        self.assertAlmostEqual(
            result.residual_values()["V-A-4_push"], 0.0, places=12
        )
        self.assertTrue(result.is_satisfied())

    def test_push_satisfied_and_violated(self):
        satisfied = friction_push_records(
            (0.0, 0.0, 0.0), (0.0, 0.0, -5.0), mu=0.6,
            include_push=True,
        )
        violated = friction_push_records(
            (0.0, 0.0, 0.0), (0.0, 0.0, -12.0), mu=0.6,
            include_push=True,
        )
        self.assertAlmostEqual(
            satisfied.residual_values()["V-A-4_push"], -4.81
        )
        self.assertAlmostEqual(
            violated.residual_values()["V-A-4_push"], 2.19
        )
        self.assertTrue(satisfied.is_satisfied())
        self.assertFalse(violated.is_satisfied())

    def test_literal_forms_are_jointly_inconsistent(self):
        """a_B = g - p_ddot 时, 字面摩擦锥与推地条件只在 a_B=0 相容."""
        p_ddot = (0.0, 0.0, -5.0)
        a_B = (0.0, 0.0, GRAVITY[2] - p_ddot[2])  # -4.81
        result = friction_push_records(
            a_B, p_ddot, mu=0.6, include_push=True
        )
        values = result.residual_values()
        self.assertAlmostEqual(values["V-A-4_push"], -4.81)
        self.assertAlmostEqual(values["5_friction_normal"], 4.81)
        self.assertGreaterEqual(result.max_violation(), 4.81)

        neutral = friction_push_records(
            (0.0, 0.0, 0.0), GRAVITY, mu=0.6, include_push=True
        )
        self.assertTrue(neutral.is_satisfied())

    def test_static_state_literal_convention_is_friction_infeasible(self):
        """静止: p_ddot = 0, L_dot = 0.

        按论文字面 a_B = g - p_ddot = [0,0,-9.81], 摩擦锥要求
        a_B_z >= 0, 因此**不可行**; 推地条件却满足. 这一测试固定
        "原文公式"的行为, 不跟随任何工程改写.
        """
        p_ddot = (0.0, 0.0, 0.0)
        a_B = (0.0, 0.0, GRAVITY[2])
        for form in ("norm", "squared"):
            result = friction_push_records(
                a_B, p_ddot, mu=0.6, form=form, include_push=False
            )
            self.assertFalse(result.is_satisfied())

        squared = friction_push_records(
            a_B, p_ddot, mu=0.6, form="squared", include_push=False
        )
        self.assertAlmostEqual(
            squared.residual_values()["5_friction_normal"], 9.81
        )
        norm = friction_push_records(
            a_B, p_ddot, mu=0.6, form="norm", include_push=False
        )
        self.assertAlmostEqual(
            norm.residual_values()["5_friction_cone"], 0.6 * 9.81
        )

        # 推地条件本身在静止时满足: g_z - 0 = -9.81 <= 0.
        with_push = friction_push_records(
            a_B, p_ddot, mu=0.6, include_push=True
        )
        self.assertAlmostEqual(
            with_push.residual_values()["V-A-4_push"], -9.81
        )
        push_records = with_push.by_category()["V-A-4"]
        self.assertTrue(
            VAConstraintSet(push_records).is_satisfied()
        )

    def test_static_state_physical_convention_is_feasible(self):
        """工程解释 a_B' = p_ddot - g (字面式整体取负).

        静止时 a_B' = [0,0,9.81], 摩擦锥与推地条件**同时满足**.
        该约定不是论文原文, 仅用于对照, 代码不会自动采用.
        """
        p_ddot = (0.0, 0.0, 0.0)
        a_B_engineering = (0.0, 0.0, -GRAVITY[2])
        for form in ("norm", "squared"):
            result = friction_push_records(
                a_B_engineering,
                p_ddot,
                mu=0.6,
                form=form,
                include_push=True,
            )
            self.assertTrue(result.is_satisfied())
            self.assertAlmostEqual(
                result.residual_values()["V-A-4_push"], -9.81
            )

    def test_friction_and_push_do_not_take_L_dot(self):
        # 式 (5) 与推地条件只用到 a_B / p_ddot, 不接收 L_dot.
        # 静止状态里的 L_dot = 0 只影响 GIAC (17b-17f) 的检验.
        import inspect

        parameters = inspect.signature(friction_push_records).parameters
        self.assertNotIn("L_dot", parameters)

    def test_push_is_opt_in(self):
        # 符号确认前, 默认不把推地条件与摩擦锥合并.
        default = friction_push_records(
            (0.0, 0.0, 1.0), (0.0, 0.0, -10.81), mu=0.6
        )
        self.assertNotIn("V-A-4", default.by_category())
        self.assertEqual(set(default.by_category()), {"5"})

    def test_unnamed_push_can_be_disabled(self):
        result = friction_push_records(
            (0.0, 0.0, 1.0), (0.0, 0.0, -12.0), mu=0.6,
            include_push=False,
        )
        self.assertNotIn("V-A-4", result.by_category())
        self.assertTrue(result.is_satisfied())

    def test_invalid_mu_and_form(self):
        with self.assertRaises(ValueError):
            friction_push_records((0, 0, 1), (0, 0, 0), mu=0.0)
        with self.assertRaises(ValueError):
            friction_push_records((0, 0, 1), (0, 0, 0), mu=0.6, form="bad")

    def test_record_names_and_categories(self):
        result = friction_push_records(
            (0.0, 0.0, 1.0), (0.0, 0.0, -10.81), mu=0.6,
            include_push=True,
        )
        self.assertEqual(
            [record.name for record in result],
            ["5_friction_cone", "5_friction_normal", "V-A-4_push"],
        )
        self.assertEqual(result.by_category()["5"][0].kind, "inequality")

    @unittest.skipUnless(CASADI_AVAILABLE, "CasADi 未安装")
    def test_squared_form_is_smooth_at_zero_horizontal(self):
        ca = solver_backend.ca
        a = ca.MX.sym("a", 3)
        a_B = (a[0], a[1], a[2])
        residual = friction_squared_residual(a_B, 0.6)
        function = ca.Function(
            "f", [a], [residual, ca.gradient(residual, a)]
        )
        value, gradient = function([0.0, 0.0, 1.0])
        self.assertTrue(np.isfinite(np.asarray(gradient)).all())
        np.testing.assert_allclose(
            np.asarray(gradient).reshape(-1),
            [0.0, 0.0, -0.72],
            atol=1e-12,
        )
        self.assertAlmostEqual(float(value), -0.36, places=12)

    @unittest.skipUnless(CASADI_AVAILABLE, "CasADi 未安装")
    def test_norm_form_has_undefined_gradient_at_zero(self):
        ca = solver_backend.ca
        a = ca.MX.sym("a", 3)
        residual = (
            (a[0] ** 2 + a[1] ** 2) ** 0.5 - 0.6 * a[2]
        )
        function = ca.Function(
            "f", [a], [residual, ca.gradient(residual, a)]
        )
        _, gradient = function([0.0, 0.0, 1.0])
        self.assertTrue(np.isnan(np.asarray(gradient)).any())


# ---------------------------------------------------------------------- #
# 式 (31): 运动学可达范围
# ---------------------------------------------------------------------- #
class TestRotationAndSpline(unittest.TestCase):
    def test_identity_rotation(self):
        rotation = np.array(rotation_matrix_zyx(0.0, 0.0, 0.0))
        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-15)

    def test_yaw_90_maps_body_x_to_world_y(self):
        rotation = np.array(rotation_matrix_zyx(0.0, 0.0, math.pi / 2))
        np.testing.assert_allclose(
            rotation @ np.array([1.0, 0.0, 0.0]),
            [0.0, 1.0, 0.0],
            atol=1e-12,
        )

    def test_pitch_90_maps_body_x_to_world_minus_z(self):
        rotation = np.array(rotation_matrix_zyx(0.0, math.pi / 2, 0.0))
        np.testing.assert_allclose(
            rotation @ np.array([1.0, 0.0, 0.0]),
            [0.0, 0.0, -1.0],
            atol=1e-12,
        )

    def test_rotation_matches_rz_ry_rx(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            angles = rng.uniform(-1.0, 1.0, size=3)
            got = np.array(rotation_matrix_zyx(*angles))
            np.testing.assert_allclose(
                got, explicit_rotation(*angles), atol=1e-12
            )

    def test_rotation_matches_project_spline(self):
        import torch

        from myrobot_mjlab.task.DTC.tamols.base_spline import Spline6D

        angles = (0.21, -0.13, 0.77)
        coeffs = torch.zeros(6, 5, dtype=torch.float64)
        coeffs[:, 0] = torch.tensor([0.0, 0.0, 0.0, *angles])
        expected = Spline6D().rotation_matrix(coeffs, 0.0)[0].numpy()
        got = np.array(rotation_matrix_zyx(*angles))
        # torch 与 math 的三角函数实现有 ~1e-9 的浮点差异.
        np.testing.assert_allclose(got, expected, atol=1e-7)

    def test_spline_value_constant_and_derivatives(self):
        coeffs = np.zeros((6, 5))
        coeffs[:, 0] = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
        np.testing.assert_allclose(
            spline_value(coeffs, 0.0), [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
        )
        coeffs[0, 1] = 2.0
        coeffs[0, 2] = 3.0
        # x(tau) = 1 + 2 tau + 3 tau^2
        self.assertAlmostEqual(spline_value(coeffs, 2.0)[0], 17.0)
        self.assertAlmostEqual(spline_value(coeffs, 2.0, 1)[0], 14.0)
        self.assertAlmostEqual(spline_value(coeffs, 2.0, 2)[0], 6.0)
        self.assertAlmostEqual(spline_value(coeffs, 2.0, 3)[0], 0.0)

    def test_spline_value_rejects_bad_shape(self):
        with self.assertRaises(ValueError):
            spline_value(np.zeros((3, 5)), 0.0)
        with self.assertRaises(ValueError):
            spline_value(np.zeros((6,)), 0.0)
        with self.assertRaises(ValueError):
            spline_value(np.zeros((6, 5)), 0.0, order=-1)


class TestReachability(unittest.TestCase):
    def setUp(self):
        self.p_B = (0.0, 0.0, 0.3)
        self.R_B = rotation_matrix_zyx(0.0, 0.0, 0.0)
        self.r_i = (0.1, 0.0, 0.0)
        self.l_min = 0.15
        self.l_max = 0.4

    def test_exact_lower_bound_passes(self):
        foot = (0.1, 0.0, 0.3 - self.l_min)
        lower, upper = reachability_residuals(
            self.p_B, self.R_B, foot, self.r_i, self.l_min, self.l_max
        )
        self.assertAlmostEqual(float(lower), 0.0, places=12)
        self.assertLess(float(upper), 0.0)

    def test_exact_upper_bound_passes(self):
        foot = (0.1, 0.0, 0.3 - self.l_max)
        lower, upper = reachability_residuals(
            self.p_B, self.R_B, foot, self.r_i, self.l_min, self.l_max
        )
        self.assertLess(float(lower), 0.0)
        self.assertAlmostEqual(float(upper), 0.0, places=12)

    def test_below_lower_bound_violates(self):
        foot = (0.1, 0.0, 0.3 - 0.1)
        lower, upper = reachability_residuals(
            self.p_B, self.R_B, foot, self.r_i, self.l_min, self.l_max
        )
        self.assertGreater(float(lower), 0.0)
        self.assertLess(float(upper), 0.0)

    def test_above_upper_bound_violates(self):
        foot = (0.1, 0.0, 0.3 - 0.5)
        lower, upper = reachability_residuals(
            self.p_B, self.R_B, foot, self.r_i, self.l_min, self.l_max
        )
        self.assertLess(float(lower), 0.0)
        self.assertGreater(float(upper), 0.0)

    def test_rotation_is_applied_to_hip_offset(self):
        yaw = math.pi / 2
        rotation = rotation_matrix_zyx(0.0, 0.0, yaw)
        r_i = (0.2, 0.0, 0.0)
        # 旋转后髋在 (0, 0.2, 0.3), 脚在其正下方 0.35 m.
        foot = (0.0, 0.2, -0.05)
        lower, upper = reachability_residuals(
            self.p_B, rotation, foot, r_i, self.l_min, self.l_max
        )
        self.assertLessEqual(float(lower), 0.0)
        self.assertLessEqual(float(upper), 0.0)

        # 若忘记旋转, 髋会被算在 (0.2, 0, 0.3), 距离超过 l_max.
        identity = rotation_matrix_zyx(0.0, 0.0, 0.0)
        _, upper_no_rotation = reachability_residuals(
            self.p_B, identity, foot, r_i, self.l_min, self.l_max
        )
        self.assertGreater(float(upper_no_rotation), 0.0)

    def test_split_lower_and_upper_records(self):
        foot = (0.1, 0.0, 0.0)
        lower, upper = reachability_records(
            self.p_B, self.R_B, foot, self.r_i,
            0.15, 0.4, leg_index=3, phase=2, tau=0.1,
        )
        self.assertEqual(lower.category, "31_lower")
        self.assertEqual(upper.category, "31_upper")
        self.assertEqual(lower.bound, 0.15**2)
        self.assertEqual(upper.bound, 0.4**2)
        self.assertEqual(lower.foot_indices, (3,))
        self.assertEqual(lower.phase, 2)
        self.assertAlmostEqual(lower.tau, 0.1)
        self.assertEqual(lower.name, "31_lower_leg3_phase2_tau0.1")

    def test_leg_mapping_uses_matching_hip_offset(self):
        r_hips = np.array(
            [
                [0.1, 0.2, 0.0],
                [0.1, -0.2, 0.0],
                [-0.1, 0.2, 0.0],
                [-0.1, -0.2, 0.0],
            ]
        )
        feet = np.array(
            [
                [0.1, 0.2, 0.0],   # FL, dist 0.3
                [0.1, -0.2, 0.0],  # FR, dist 0.3
                [-0.1, 0.2, 0.0],  # RL, dist 0.3
                [-0.1, -0.2, 0.0], # RR, dist 0.3
            ]
        )
        identity = np.eye(3)
        for index in range(4):
            records = reachability_records(
                (0.0, 0.0, 0.3),
                identity,
                feet[index],
                r_hips[index],
                0.15,
                0.4,
                leg_index=index,
            )
            self.assertEqual(records[0].foot_indices, (index,))
            self.assertTrue(records[0].is_satisfied())
            self.assertTrue(records[1].is_satisfied())

        # 用错髋位置 (FL 配 RR 的偏移) 会检测出超出可达范围.
        wrong = reachability_records(
            (0.0, 0.0, 0.3),
            identity,
            feet[0],
            r_hips[3],
            0.15,
            0.4,
            leg_index=0,
        )
        self.assertFalse(wrong[1].is_satisfied())

    def test_invalid_limits(self):
        with self.assertRaises(ValueError):
            reachability_residuals(
                self.p_B, self.R_B, (0, 0, 0), self.r_i, 0.0, 0.4
            )
        with self.assertRaises(ValueError):
            reachability_residuals(
                self.p_B, self.R_B, (0, 0, 0), self.r_i, 0.5, 0.4
            )
        with self.assertRaises(ValueError):
            reachability_residuals(
                self.p_B, self.R_B, (0, 0, 0), self.r_i, float("nan"), 0.4
            )

    def test_invalid_rotation_shape(self):
        with self.assertRaises(ValueError):
            reachability_residuals(
                self.p_B, np.eye(2), (0, 0, 0), self.r_i, 0.15, 0.4
            )

    def test_defaults_match_project(self):
        self.assertEqual(DEFAULT_L_MIN, 0.15)
        self.assertEqual(DEFAULT_L_MAX, 0.40)
        np.testing.assert_allclose(DEFAULT_HIP_OFFSETS, GO2_STANCE_HIP_OFFSETS)
        self.assertAlmostEqual(GO2_HIP_JOINT_OFFSETS[0, 1], 0.0465)


class TestSplinePhaseReachability(unittest.TestCase):
    def _constant_coeffs(self, pose):
        coeffs = np.zeros((6, 5))
        coeffs[:, 0] = pose
        return coeffs

    def test_default_sampling_excludes_phase_end(self):
        coeffs = self._constant_coeffs([0.0, 0.0, 0.3, 0.0, 0.0, 0.0])
        footholds = np.array([[0.0, 0.0, 0.0]] * 4)
        r_hips = np.zeros((4, 3))
        class Recorder:
            def __init__(self):
                self.items = []
            def add_inequality(self, expr, lb, ub, name=""):
                self.items.append((expr, lb, ub, name))
        backend = Recorder()
        result = add_spline_phase_reachability_constraints(
            backend,
            coeffs,
            duration=0.3,
            footholds=footholds,
            r_hips=r_hips,
            stance_legs=("FL", "RR"),
            l_min=0.15,
            l_max=0.4,
            phase=1,
            num_samples=3,
        )
        self.assertEqual(len(result), 3 * 2 * 2)
        self.assertEqual(len(backend.items), 12)
        np.testing.assert_allclose(
            [record.tau for record in result][0::2],
            [0.0, 0.0, 0.1, 0.1, 0.2, 0.2],
            atol=1e-12,
        )
        self.assertTrue(all(record.phase == 1 for record in result))
        self.assertTrue(all(item[1] == -float("inf") for item in backend.items))
        self.assertTrue(all(item[2] == 0.0 for item in backend.items))

    def test_feasible_and_infeasible_samples(self):
        coeffs = self._constant_coeffs([0.0, 0.0, 0.3, 0.0, 0.0, 0.0])
        r_hips = np.zeros((4, 3))
        feasible = np.zeros((4, 3))
        class Recorder:
            def add_inequality(self, expr, lb, ub, name=""):
                pass
        good = add_spline_phase_reachability_constraints(
            Recorder(), coeffs, 0.3, feasible, r_hips, ("FL",),
            num_samples=3,
        )
        self.assertTrue(good.is_satisfied())

        too_far = feasible.copy()
        too_far[0] = [0.0, 0.0, 0.9]  # 与 base 的距离 0.6 > l_max
        bad = add_spline_phase_reachability_constraints(
            Recorder(), coeffs, 0.3, too_far, r_hips, ("FL",),
            num_samples=3,
        )
        self.assertFalse(bad.is_satisfied())
        self.assertGreater(bad.max_violation(), 0.0)

    def test_rotation_used_in_spline_path(self):
        coeffs = self._constant_coeffs(
            [0.0, 0.0, 0.3, 0.0, 0.0, math.pi / 2]
        )
        r_hips = np.zeros((4, 3))
        r_hips[0] = [0.2, 0.0, 0.0]
        footholds = np.zeros((4, 3))
        # yaw=90° 后髋在 (0, 0.2, 0.3); 脚在其下方 0.35 m.
        footholds[0] = [0.0, 0.2, -0.05]
        class Recorder:
            def add_inequality(self, expr, lb, ub, name=""):
                pass
        good = add_spline_phase_reachability_constraints(
            Recorder(), coeffs, 0.3, footholds, r_hips, ("FL",),
            num_samples=1,
        )
        self.assertTrue(good.is_satisfied())

    def test_sample_time_validation(self):
        coeffs = self._constant_coeffs([0.0, 0.0, 0.3, 0.0, 0.0, 0.0])
        class Recorder:
            def add_inequality(self, expr, lb, ub, name=""):
                pass
        args = (Recorder(), coeffs, 0.3, np.zeros((4, 3)), np.zeros((4, 3)),
                ("FL",))
        with self.assertRaises(ValueError):
            add_spline_phase_reachability_constraints(
                *args, sample_times=(-0.1,)
            )
        with self.assertRaises(ValueError):
            add_spline_phase_reachability_constraints(
                *args, sample_times=(0.4,)
            )
        with self.assertRaises(ValueError):
            add_spline_phase_reachability_constraints(
                *args, sample_times=()
            )
        with self.assertRaises(ValueError):
            add_spline_phase_reachability_constraints(
                *args, num_samples=0
            )
        with self.assertRaises(ValueError):
            add_spline_phase_reachability_constraints(
                Recorder(), coeffs, 0.0, np.zeros((4, 3)),
                np.zeros((4, 3)), ("FL",)
            )


class TestSplineFriction(unittest.TestCase):
    def test_spline_base_acceleration(self):
        coeffs = np.zeros((6, 5))
        coeffs[0, 0] = 1.0
        coeffs[0, 1] = 2.0
        coeffs[0, 2] = 3.0  # x(tau) = 1 + 2 tau + 3 tau^2
        p_ddot, a_B = spline_base_acceleration(coeffs, 0.5)
        np.testing.assert_allclose(p_ddot, [6.0, 0.0, 0.0])
        np.testing.assert_allclose(a_B, [-6.0, 0.0, -9.81])

    def test_sampled_friction_records(self):
        coeffs = np.zeros((6, 5))
        coeffs[:, 0] = [0.0, 0.0, 0.3, 0.0, 0.0, 0.0]

        class Recorder:
            def __init__(self):
                self.names = []
            def add_inequality(self, expr, lb, ub, name=""):
                self.names.append(name)

        backend = Recorder()
        result = add_spline_phase_friction_constraints(
            backend,
            coeffs,
            duration=0.3,
            mu=0.6,
            include_push=False,
            phase=0,
            num_samples=3,
        )
        self.assertEqual(len(result), 6)
        self.assertEqual(len(backend.names), 6)
        self.assertTrue(any("_phase0_tau" in name for name in backend.names))
        # a_B = g = (0,0,-9.81): 字面摩擦锥被违反 (法向为负).
        self.assertAlmostEqual(result.max_violation(), 9.81)
        self.assertTrue(
            all(record.phase == 0 for record in result)
        )
        np.testing.assert_allclose(
            [record.tau for record in result][0::2],
            [0.0, 0.1, 0.2],
        )

    def test_sampled_push_is_satisfied_at_rest(self):
        coeffs = np.zeros((6, 5))
        coeffs[:, 0] = [0.0, 0.0, 0.3, 0.0, 0.0, 0.0]

        class Recorder:
            def add_inequality(self, expr, lb, ub, name=""):
                pass

        result = add_spline_phase_friction_constraints(
            Recorder(), coeffs, 0.3, 0.6, phase=1,
            sample_times=(0.0, 0.15), include_push=True,
        )
        push = result.by_category()["V-A-4"]
        self.assertEqual(len(push), 2)
        for record in push:
            # 静止时 p_ddot = 0, 残差 = g_z = -9.81 <= 0, 条件满足.
            self.assertAlmostEqual(float(record.residual), -9.81, places=12)
            self.assertTrue(record.is_satisfied())
        self.assertIn("_phase1_tau0.15", push[1].name)


class TestIndependentNumpyRecomputation(unittest.TestCase):
    """用裸 numpy 公式复算, 避免只验证 "函数等于自己"."""

    def test_convexity_matches_numpy_cross(self):
        p1, p2, p3, p4 = (row for row in SQUARE)
        expected = [
            np.cross(p3 - p1, p2 - p1)[2],
            np.cross(p4 - p1, p3 - p1)[2],
            np.cross(p4 - p2, p3 - p2)[2],
            np.cross(p1 - p2, p4 - p2)[2],
        ]
        np.testing.assert_allclose(
            [float(value) for value in convexity_residuals(*SQUARE)],
            expected,
            atol=1e-15,
        )

    def test_friction_matches_raw_numpy(self):
        a = np.array([0.5, 0.2, 1.0])
        mu = 0.6
        norm_expected = float(
            np.hypot(a[0], a[1]) - mu * a[2]
        )
        squared_expected = float(
            a[0] ** 2 + a[1] ** 2 - (mu * a[2]) ** 2
        )
        norm = friction_push_records(
            a, (0, 0, 0), mu, form="norm", include_push=False
        )
        squared = friction_push_records(
            a, (0, 0, 0), mu, form="squared", include_push=False
        )
        self.assertAlmostEqual(
            norm.residual_values()["5_friction_cone"],
            norm_expected,
            places=15,
        )
        self.assertAlmostEqual(
            squared.residual_values()["5_friction_cone"],
            squared_expected,
            places=15,
        )
        self.assertAlmostEqual(
            squared.residual_values()["5_friction_normal"], -a[2]
        )

    def test_push_matches_raw_numpy(self):
        p_ddot = np.array([0.1, -0.2, -5.0])
        g = np.array(GRAVITY)
        expected = float(g[2] - p_ddot[2])
        result = friction_push_records(
            (0, 0, 0), p_ddot, 0.6, include_push=True
        )
        self.assertAlmostEqual(
            result.residual_values()["V-A-4_push"], expected, places=15
        )

    def test_reachability_matches_raw_numpy(self):
        p_B = np.array([0.1, -0.2, 0.3])
        R = explicit_rotation(0.1, -0.2, 0.3)
        r = np.array([0.1934, 0.145, 0.0])
        p_i = np.array([0.4, 0.1, 0.0])
        diff = p_B + R @ r - p_i
        distance_sq = float(diff @ diff)
        expected = (
            0.15**2 - distance_sq,
            distance_sq - 0.4**2,
        )
        got = reachability_residuals(
            p_B, R, p_i, r, 0.15, 0.4
        )
        np.testing.assert_allclose(
            [float(value) for value in got], expected, atol=1e-15
        )

    def test_spline_value_matches_numpy_polynomial(self):
        rng = np.random.default_rng(2)
        coeffs = rng.normal(size=(6, 5))
        for tau in (0.0, 0.3, 1.7):
            for order in (0, 1, 2, 3):
                expected = np.array(
                    [
                        np.polynomial.Polynomial(coeffs[dim])
                        .deriv(order)(tau)
                        for dim in range(6)
                    ]
                )
                got = np.array(
                    spline_value(coeffs, tau, order=order), dtype=float
                )
                np.testing.assert_allclose(got, expected, atol=1e-12)


# ---------------------------------------------------------------------- #
# 与 CasADi + IPOPT 的组合
# ---------------------------------------------------------------------- #
@unittest.skipUnless(CASADI_AVAILABLE, "CasADi 未安装")
class TestCasadiIPOPT(unittest.TestCase):
    def _backend(self):
        from myrobot_mjlab.task.DTC.tamols.solver_backend import CasadiBackend

        return CasadiBackend(
            solver_options={"ipopt": {"print_level": 0, "max_iter": 200}}
        )

    @staticmethod
    def _status(backend) -> str:
        return str(backend.opti.stats()["return_status"])

    def test_convexity_feasibility_nlp(self):
        ca = solver_backend.ca
        backend = self._backend()
        feet = backend.variable((4, 3), name="feet")
        target = GO2_FEET.copy()
        # RL 缩到三角形 FL-RR-FR 内部 -> 凹足迹 (r1 被违反).
        target[2] = [0.05, 0.0, -0.28]
        backend.opti.minimize(ca.sumsqr(feet - target))
        for index in range(4):
            backend.opti.subject_to(feet[index, 2] == -0.28)
        backend.set_initial(feet, GO2_FEET)
        result = add_convexity_constraints(backend, feet, phase=0)
        backend.solve()

        self.assertEqual(self._status(backend), "Solve_Succeeded")
        values = result.residual_values(evaluate=backend.value)
        self.assertLessEqual(max(values.values()), 1e-6)
        solution = np.asarray(backend.value(feet)).reshape(4, 3)
        self.assertGreater(
            signed_area(solution[list(CCW_INDICES)]), 0.0
        )
        # 解确实被推到凸位置, 而不是停在凹的目标上.
        self.assertGreater(
            np.linalg.norm(solution[2] - target[2]), 0.01
        )

    def test_friction_inside_target_is_kept(self):
        ca = solver_backend.ca
        backend = self._backend()
        a = backend.variable((3,), name="a_B")
        target = np.array([0.5, 0.2, 1.0])
        backend.opti.minimize(ca.sumsqr(a - target))
        backend.set_initial(a, [0.0, 0.0, 1.0])
        p_ddot = np.array(
            [
                GRAVITY[0] - target[0],
                GRAVITY[1] - target[1],
                GRAVITY[2] - target[2],
            ]
        )
        # 推地条件在这里是常值 (不影响变量), CasADi 不接受常值约束,
        # 因此只加入摩擦锥; 推地条件由 test_push_condition_nlp 覆盖.
        result = add_friction_push_constraints(
            backend, (a[0], a[1], a[2]), p_ddot, mu=0.6,
            include_push=False,
        )
        backend.solve()

        self.assertEqual(self._status(backend), "Solve_Succeeded")
        solution = np.asarray(backend.value(a)).reshape(-1)
        np.testing.assert_allclose(solution, target, atol=1e-6)
        self.assertLessEqual(
            result.max_violation(evaluate=backend.value), 1e-7
        )

    def test_friction_outside_target_hits_cone_boundary(self):
        ca = solver_backend.ca
        backend = self._backend()
        a = backend.variable((3,), name="a_B")
        target = np.array([0.9, 0.0, 1.0])
        backend.opti.minimize(ca.sumsqr(a - target))
        backend.set_initial(a, [0.0, 0.0, 1.0])
        result = add_friction_push_constraints(
            backend,
            (a[0], a[1], a[2]),
            (0.0, 0.0, -10.81),
            mu=0.6,
            include_push=False,
        )
        backend.solve()

        self.assertEqual(self._status(backend), "Solve_Succeeded")
        self.assertLessEqual(
            result.max_violation(evaluate=backend.value), 1e-7
        )
        solution = np.asarray(backend.value(a)).reshape(-1)
        cone = solution[0] ** 2 + solution[1] ** 2 - (0.6 * solution[2]) ** 2
        self.assertAlmostEqual(cone, 0.0, places=6)
        self.assertGreater(np.linalg.norm(solution - target), 0.05)

    def test_push_condition_nlp(self):
        ca = solver_backend.ca
        backend = self._backend()
        pz = backend.variable((1,), name="p_ddot_z")
        # a_B 用变量并固定为 0, 避免摩擦锥变成常值约束.
        a = backend.variable((3,), name="a_B")
        for dim in range(3):
            backend.opti.subject_to(a[dim] == 0.0)
        backend.opti.minimize((pz - (-12.0)) ** 2)
        backend.set_initial(pz, 0.0)
        result = add_friction_push_constraints(
            backend,
            (a[0], a[1], a[2]),
            (0.0, 0.0, pz),
            mu=0.6,
            include_push=True,
        )
        backend.solve()

        self.assertEqual(self._status(backend), "Solve_Succeeded")
        np.testing.assert_allclose(
            np.asarray(backend.value(pz)).reshape(-1), [-9.81], atol=1e-6
        )
        self.assertLessEqual(
            result.max_violation(evaluate=backend.value), 1e-7
        )

    def test_literal_cone_and_push_force_zero_acceleration(self):
        """把 a_B = g - p_ddot 作为等式时, 两个字面约束只剩 a_B=0."""
        ca = solver_backend.ca
        backend = self._backend()
        a = backend.variable((3,), name="a_B")
        p_ddot = backend.variable((3,), name="p_ddot")
        for dim in range(3):
            backend.opti.subject_to(a[dim] + p_ddot[dim] == GRAVITY[dim])
        target = np.array([0.05, 0.0, 1.0])
        backend.opti.minimize(ca.sumsqr(a - target))
        backend.set_initial(a, [0.0, 0.0, 1.0])
        backend.set_initial(p_ddot, [0.0, 0.0, -10.81])
        # 仅作为 "字面符号互相矛盾" 的诊断, 不要用于完整优化.
        result = add_friction_push_constraints(
            backend,
            (a[0], a[1], a[2]),
            (p_ddot[0], p_ddot[1], p_ddot[2]),
            mu=0.6,
            include_push=True,
        )
        backend.solve()

        self.assertEqual(self._status(backend), "Solve_Succeeded")
        solution = np.asarray(backend.value(a)).reshape(-1)
        # IPOPT 的约束容差允许 ~1e-4 的数值残差.
        np.testing.assert_allclose(solution, np.zeros(3), atol=1e-3)
        self.assertLessEqual(
            result.max_violation(evaluate=backend.value), 1e-7
        )
        # 目标远离可行集, 说明两个字面约束确实把可行集压成了单点.
        self.assertGreater(np.linalg.norm(solution - target), 0.9)

    def test_reachability_nlp_pushes_to_bounds(self):
        ca = solver_backend.ca
        p_B = np.array([0.0, 0.0, 0.3])
        R_B = np.eye(3)
        r_i = np.array([0.1, 0.0, 0.0])
        for target_dist, expected in ((0.05, 0.15), (0.55, 0.40)):
            backend = self._backend()
            foot = backend.variable((3,), name="foot")
            # 目标在髋下方 target_dist 处.
            target = np.array([0.1, 0.0, 0.3 - target_dist])
            backend.opti.minimize(ca.sumsqr(foot - target))
            backend.set_initial(foot, [0.1, 0.0, 0.0])
            result = add_reachability_constraints(
                backend, p_B, R_B, foot, r_i, legs=(0,),
                l_min=0.15, l_max=0.4,
            )
            backend.solve()
            self.assertEqual(self._status(backend), "Solve_Succeeded")
            self.assertLessEqual(
                result.max_violation(evaluate=backend.value), 1e-7
            )
            solution = np.asarray(backend.value(foot)).reshape(-1)
            hip = p_B + r_i
            distance = float(np.linalg.norm(solution - hip))
            self.assertAlmostEqual(distance, expected, places=4)

    def test_spline_reachability_with_symbolic_coeffs(self):
        ca = solver_backend.ca
        backend = self._backend()
        coeffs = backend.variable((6, 5), name="coeffs")
        backend.opti.subject_to(
            coeffs[:, 0] == [0.0, 0.0, 0.3, 0.0, 0.0, 0.0]
        )
        for k in range(1, 5):
            backend.opti.subject_to(coeffs[:, k] == 0.0)

        foot = backend.variable((3,), name="foot")
        target = np.array([0.0, 0.0, -0.2])  # 距离髋 0.5 m, 超出 l_max
        backend.opti.minimize(ca.sumsqr(foot - target))
        backend.set_initial(foot, [0.0, 0.0, -0.1])
        result = add_spline_phase_reachability_constraints(
            backend,
            coeffs,
            duration=0.3,
            footholds=foot,
            r_hips=np.zeros((4, 3)),
            stance_legs=("FL",),
            l_min=0.15,
            l_max=0.4,
            phase=0,
            num_samples=2,
        )
        backend.solve()

        self.assertEqual(self._status(backend), "Solve_Succeeded")
        self.assertLessEqual(
            result.max_violation(evaluate=backend.value), 1e-7
        )
        solution = np.asarray(backend.value(foot)).reshape(-1)
        hip = np.array([0.0, 0.0, 0.3])  # p_B, 髋偏移为 0
        self.assertAlmostEqual(
            float(np.linalg.norm(solution - hip)), 0.4, places=4
        )

    def test_assembly_names_and_sets(self):
        ca = solver_backend.ca
        backend = self._backend()
        feet = backend.variable((4, 3), name="feet")
        backend.opti.minimize(ca.sumsqr(feet))
        backend.set_initial(feet, GO2_FEET)
        result = add_convexity_constraints(backend, feet, phase=2)
        self.assertIsInstance(result, VAConstraintSet)
        self.assertEqual(len(backend.constraints), 4)
        self.assertTrue(all(kind == "inequality" for kind, _ in backend.constraints))
        self.assertTrue(all(record.phase == 2 for record in result))
        backend.solve()
        self.assertLessEqual(
            result.max_violation(evaluate=backend.value), 1e-7
        )
        self.assertLessEqual(
            max(result.max_violation_by_category(
                evaluate=backend.value
            ).values()),
            1e-7,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
