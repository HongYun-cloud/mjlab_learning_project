"""optimization_problem 的单元测试.

不依赖 CasADi 的部分用一个记录型假后端验证变量布局和约束组装;
安装了 CasADi 时额外用 IPOPT 实际求解, 验证初始位姿约束确实被满足.

运行:
    python -m unittest myrobot_mjlab.task.DTC.tamols.test_optimization_problem -v
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import unittest

from myrobot_mjlab.task.DTC.tamols import solver_backend
from myrobot_mjlab.task.DTC.tamols.optimization_problem import (
    TrajectoryOptimizationProblem,
)
from myrobot_mjlab.task.DTC.tamols.optimization_state import OptimizationStateLayout
from myrobot_mjlab.task.DTC.tamols.solver_backend import (
    CasadiBackend,
    SolverBackend,
    make_backend,
)

POSE = np.array([0.3, -0.2, 0.45, 0.05, -0.04, 0.8])


class FakeNode:
    """支持 +/* 的占位表达式节点, 用于记录约束表达式树."""

    def __mul__(self, scalar: Any) -> "FakeScaled":
        return FakeScaled(self, scalar)

    def __rmul__(self, scalar: Any) -> "FakeScaled":
        return FakeScaled(self, scalar)

    def __add__(self, other: Any) -> "FakeSum":
        return FakeSum([self, other])

    def __radd__(self, other: Any) -> "FakeSum":
        return FakeSum([other, self])


class FakeExpr(FakeNode):
    """记录变量索引的占位表达式."""

    def __init__(self, var: "FakeVar", key: Any):
        self.var = var
        self.key = key

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"{self.var.name}[{self.key!r}]"


class FakeScaled(FakeNode):
    def __init__(self, expr: FakeNode, scale: Any):
        self.expr = expr
        self.scale = scale


class FakeSum(FakeNode):
    def __init__(self, terms: list[Any]):
        self.terms = terms


def evaluate_expr(expr: Any, blocks: dict[str, np.ndarray]) -> np.ndarray:
    """用给定的系数数值求占位表达式树的值."""
    if isinstance(expr, FakeExpr):
        rows, col = expr.key
        return np.asarray(blocks[expr.var.name])[rows, col]
    if isinstance(expr, FakeScaled):
        return expr.scale * evaluate_expr(expr.expr, blocks)
    if isinstance(expr, FakeSum):
        return sum(evaluate_expr(term, blocks) for term in expr.terms)
    raise TypeError(f"无法求值的表达式节点: {type(expr).__name__}")


class FakeVar:
    def __init__(self, name: str, shape: tuple[int, ...]):
        self.name = name
        self.shape = shape

    def __getitem__(self, key: Any) -> FakeExpr:
        return FakeExpr(self, key)


class FakeBackend(SolverBackend):
    """只记录调用, 不做任何数值计算."""

    def __init__(self):
        self.variables: list[FakeVar] = []
        self.equalities: list[tuple[Any, Any, str]] = []
        self.inequalities: list[tuple[Any, Any, Any, str]] = []
        self.initials: list[tuple[FakeVar, Any]] = []
        self.solved = 0

    def variable(self, shape: tuple[int, ...], name: str = "") -> FakeVar:
        var = FakeVar(name, tuple(shape))
        self.variables.append(var)
        return var

    def add_equality(self, lhs: Any, rhs: Any, name: str = "") -> None:
        self.equalities.append((lhs, rhs, name))

    def add_inequality(self, expr: Any, lb: Any, ub: Any, name: str = "") -> None:
        self.inequalities.append((expr, lb, ub, name))

    def set_initial(self, var: Any, value: Any) -> None:
        self.initials.append((var, value))

    def solve(self) -> str:
        self.solved += 1
        return "fake-solution"

    def value(self, expr: Any) -> np.ndarray:
        return np.zeros(1)


class TestVariableLayout(unittest.TestCase):
    def test_default_variables(self):
        backend = FakeBackend()
        problem = TrajectoryOptimizationProblem(backend=backend)

        self.assertEqual(len(problem.vars.spline_coeffs), 1)
        self.assertEqual(problem.vars.spline_coeffs[0].name, "spline_coeffs_0")
        self.assertEqual(problem.vars.spline_coeffs[0].shape, (6, 5))
        self.assertEqual(problem.vars.landing_footholds.shape, (4, 3))
        self.assertEqual(problem.vars.desired_footholds.shape, (4, 3))
        self.assertEqual(problem.vars.slack.shape, (1,))
        # 4 blocks: 1 spline + landing + desired + slack
        self.assertEqual(len(backend.variables), 4)

    def test_custom_layout_variables(self):
        layout = OptimizationStateLayout(
            num_phases=3, num_legs=2, num_coeffs=6, num_slack=4
        )
        backend = FakeBackend()
        problem = TrajectoryOptimizationProblem(layout=layout, backend=backend)

        self.assertEqual(len(problem.vars.spline_coeffs), 3)
        for phase, var in enumerate(problem.vars.spline_coeffs):
            self.assertEqual(var.name, f"spline_coeffs_{phase}")
            self.assertEqual(var.shape, (6, 6))
        self.assertEqual(problem.vars.landing_footholds.shape, (2, 3))
        self.assertEqual(problem.vars.desired_footholds.shape, (2, 3))
        self.assertEqual(problem.vars.slack.shape, (4,))

    def test_problem_uses_same_layout_as_state_vector(self):
        layout = OptimizationStateLayout(num_phases=2, num_legs=3, num_coeffs=5)
        problem = TrajectoryOptimizationProblem(
            layout=layout, backend=FakeBackend()
        )
        coeffs_size = sum(
            var.shape[0] * var.shape[1] for var in problem.vars.spline_coeffs
        )
        landing_size = int(np.prod(problem.vars.landing_footholds.shape))
        desired_size = int(np.prod(problem.vars.desired_footholds.shape))
        slack_size = int(np.prod(problem.vars.slack.shape))
        self.assertEqual(
            coeffs_size + landing_size + desired_size + slack_size,
            layout.total_size,
        )


class TestInitialPoseConstraint(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.problem = TrajectoryOptimizationProblem(backend=self.backend)

    def test_constrains_first_column_of_first_phase(self):
        self.problem.add_initial_pose_constraint(POSE)

        self.assertEqual(len(self.backend.equalities), 1)
        lhs, rhs, name = self.backend.equalities[0]
        self.assertEqual(name, "initial_pose")
        self.assertEqual(lhs.var.name, "spline_coeffs_0")

        dims, coeff_index = lhs.key
        self.assertEqual(dims, list(range(6)))
        self.assertEqual(coeff_index, 0)

        self.assertEqual(rhs.shape, (6, 1))
        np.testing.assert_array_equal(rhs[:, 0], POSE)
        np.testing.assert_array_equal(self.problem.initial_pose, POSE)

    def test_select_position_or_orientation_dims(self):
        self.problem.add_initial_pose_constraint(POSE, dims=(0, 1, 2))
        lhs, rhs, _ = self.backend.equalities[-1]
        self.assertEqual(lhs.key, ([0, 1, 2], 0))
        np.testing.assert_array_equal(rhs[:, 0], POSE[:3])

        self.problem.add_initial_pose_constraint(POSE, dims=[3, 4, 5])
        lhs, rhs, _ = self.backend.equalities[-1]
        self.assertEqual(lhs.key, ([3, 4, 5], 0))
        np.testing.assert_array_equal(rhs[:, 0], POSE[3:])

    def test_constraint_on_later_phase(self):
        layout = OptimizationStateLayout(num_phases=3)
        backend = FakeBackend()
        problem = TrajectoryOptimizationProblem(layout=layout, backend=backend)
        problem.add_initial_pose_constraint(POSE, phase=2)
        lhs, _, _ = backend.equalities[0]
        self.assertEqual(lhs.var.name, "spline_coeffs_2")
        self.assertEqual(lhs.key[1], 0)

    def test_pose_can_be_list(self):
        self.problem.add_initial_pose_constraint(list(POSE))
        _, rhs, _ = self.backend.equalities[0]
        np.testing.assert_array_equal(rhs[:, 0], POSE)

    def test_invalid_pose(self):
        with self.assertRaises(ValueError):
            self.problem.add_initial_pose_constraint(POSE[:5])
        with self.assertRaises(ValueError):
            self.problem.add_initial_pose_constraint(np.append(POSE, 0.0))
        bad = POSE.copy()
        bad[2] = float("nan")
        with self.assertRaises(ValueError):
            self.problem.add_initial_pose_constraint(bad)

    def test_invalid_phase_and_dims(self):
        with self.assertRaises(ValueError):
            self.problem.add_initial_pose_constraint(POSE, phase=1)
        with self.assertRaises(ValueError):
            self.problem.add_initial_pose_constraint(POSE, phase=-1)
        with self.assertRaises(ValueError):
            self.problem.add_initial_pose_constraint(POSE, dims=())
        with self.assertRaises(ValueError):
            self.problem.add_initial_pose_constraint(POSE, dims=(0, 6))
        with self.assertRaises(ValueError):
            self.problem.add_initial_pose_constraint(POSE, dims=(1, 1))

    def test_solve_delegates_to_backend(self):
        self.problem.add_initial_pose_constraint(POSE)
        self.assertEqual(self.problem.solve(), "fake-solution")
        self.assertEqual(self.backend.solved, 1)


class TestJunctionConstraints(unittest.TestCase):
    def _make_problem(self, num_phases: int = 3, num_coeffs: int = 5):
        layout = OptimizationStateLayout(
            num_phases=num_phases, num_legs=2, num_coeffs=num_coeffs
        )
        backend = FakeBackend()
        problem = TrajectoryOptimizationProblem(layout=layout, backend=backend)
        return problem, backend

    @staticmethod
    def _random_coeffs(problem: TrajectoryOptimizationProblem):
        rng = np.random.default_rng(0)
        return {
            var.name: rng.normal(size=var.shape)
            for var in problem.vars.spline_coeffs
        }

    @staticmethod
    def _poly_end(
        coeffs: np.ndarray, duration: float, dims: list[int]
    ) -> np.ndarray:
        value = np.zeros(len(dims))
        for i in range(coeffs.shape[1]):
            value = value + coeffs[dims, i] * duration**i
        return value

    def test_one_equality_per_junction(self):
        problem, backend = self._make_problem(num_phases=3)
        added = problem.add_junction_constraints([0.3, 0.4, 0.5])

        self.assertEqual(added, 2)
        self.assertEqual(len(backend.equalities), 2)
        names = [name for _, _, name in backend.equalities]
        self.assertEqual(names, ["junction_0_1", "junction_1_2"])

        for k, (lhs, rhs, _) in enumerate(backend.equalities):
            # 右端是下一段的 a0, 即 [:, 0]
            self.assertEqual(rhs.var.name, f"spline_coeffs_{k + 1}")
            self.assertEqual(rhs.key, (list(range(6)), 0))

    def test_endpoint_is_previous_phase_end(self):
        problem, backend = self._make_problem(num_phases=3)
        durations = [0.3, 0.4, 0.5]
        problem.add_junction_constraints(durations)

        blocks = self._random_coeffs(problem)
        for k, (lhs, rhs, _) in enumerate(backend.equalities):
            expected_end = self._poly_end(
                blocks[f"spline_coeffs_{k}"], durations[k], list(range(6))
            )
            np.testing.assert_allclose(
                evaluate_expr(lhs, blocks), expected_end, atol=1e-12
            )
            np.testing.assert_allclose(
                evaluate_expr(rhs, blocks),
                blocks[f"spline_coeffs_{k + 1}"][:, 0],
                atol=1e-12,
            )

    def test_scalar_duration(self):
        problem, backend = self._make_problem(num_phases=2)
        problem.add_junction_constraints(0.25)

        blocks = self._random_coeffs(problem)
        lhs, _, _ = backend.equalities[0]
        np.testing.assert_allclose(
            evaluate_expr(lhs, blocks),
            self._poly_end(
                blocks["spline_coeffs_0"], 0.25, list(range(6))
            ),
            atol=1e-12,
        )

    def test_dims_subset(self):
        problem, backend = self._make_problem(num_phases=2)
        dims = [0, 2, 5]
        problem.add_junction_constraints([0.4, 0.4], dims=dims)

        blocks = self._random_coeffs(problem)
        lhs, rhs, _ = backend.equalities[0]
        np.testing.assert_allclose(
            evaluate_expr(lhs, blocks),
            self._poly_end(blocks["spline_coeffs_0"], 0.4, dims),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            evaluate_expr(rhs, blocks),
            blocks["spline_coeffs_1"][dims, 0],
            atol=1e-12,
        )

    def test_single_phase_is_noop(self):
        problem, backend = self._make_problem(num_phases=1)
        self.assertEqual(problem.add_junction_constraints(0.3), 0)
        self.assertEqual(len(backend.equalities), 0)

    def test_invalid_durations(self):
        problem, _ = self._make_problem(num_phases=3)
        with self.assertRaises(ValueError):
            problem.add_junction_constraints([0.3, 0.4])  # 少一个
        with self.assertRaises(ValueError):
            problem.add_junction_constraints([0.3, 0.0, 0.5])
        with self.assertRaises(ValueError):
            problem.add_junction_constraints([0.3, -0.4, 0.5])
        with self.assertRaises(ValueError):
            problem.add_junction_constraints([0.3, float("nan"), 0.5])


class TestBackendFactory(unittest.TestCase):
    def test_unknown_backend(self):
        with self.assertRaises(ValueError):
            make_backend("bogus")

    def test_missing_casadi_has_clear_error(self):
        if solver_backend.ca is not None:
            self.skipTest("当前环境已安装 CasADi")
        with self.assertRaises(ImportError):
            CasadiBackend()


CASADI_AVAILABLE = solver_backend.ca is not None


@unittest.skipUnless(CASADI_AVAILABLE, "CasADi 未安装")
class TestCasadiIPOPT(unittest.TestCase):
    """真实求解: 用二次目标让只含初始位姿约束的问题有唯一解."""

    def _solve_with_regularization(self, problem: TrajectoryOptimizationProblem):
        ca = solver_backend.ca
        # 只约束了位姿时问题欠定; 加一个很小的正则项使所有变量都有梯度.
        objective = ca.sumsqr(problem.vars.spline_coeffs[0])
        for coeffs in problem.vars.spline_coeffs[1:]:
            objective += ca.sumsqr(coeffs)
        objective += 1e-6 * ca.sumsqr(problem.vars.landing_footholds)
        objective += 1e-6 * ca.sumsqr(problem.vars.desired_footholds)
        objective += 1e-6 * ca.sumsqr(problem.vars.slack)
        problem.backend.opti.minimize(objective)
        problem.solve()

    def test_solution_starts_at_current_pose(self):
        problem = TrajectoryOptimizationProblem(
            solver_options={"ipopt": {"print_level": 0}}
        )
        problem.add_initial_pose_constraint(POSE)
        self._solve_with_regularization(problem)

        coeffs = problem.value(problem.vars.spline_coeffs[0])
        np.testing.assert_allclose(coeffs[:, 0], POSE, atol=1e-7)
        # 目标只依赖系数的平方 -> 除被约束的第 0 列外其余系数应为 0
        np.testing.assert_allclose(coeffs[:, 1:], 0.0, atol=1e-6)

    def test_partial_dims(self):
        problem = TrajectoryOptimizationProblem(
            solver_options={"ipopt": {"print_level": 0}}
        )
        problem.add_initial_pose_constraint(POSE, dims=(0, 1, 2))
        self._solve_with_regularization(problem)

        coeffs = problem.value(problem.vars.spline_coeffs[0])
        np.testing.assert_allclose(coeffs[:3, 0], POSE[:3], atol=1e-7)
        np.testing.assert_allclose(coeffs[3:, 0], 0.0, atol=1e-6)

    def test_junction_connects_phase_end_to_next_start(self):
        layout = OptimizationStateLayout(num_phases=2)
        problem = TrajectoryOptimizationProblem(
            layout=layout, solver_options={"ipopt": {"print_level": 0}}
        )
        problem.add_initial_pose_constraint(POSE)
        self.assertEqual(problem.add_junction_constraints([0.3, 0.4]), 1)
        self._solve_with_regularization(problem)

        coeffs0 = problem.value(problem.vars.spline_coeffs[0])
        coeffs1 = problem.value(problem.vars.spline_coeffs[1])

        # 注意: phase 0 的自由系数不会全部为 0. 连接约束把 phase 1 的
        # 起点 s1 = sum_k a_k^0 T^k 与 phase 0 的系数绑在一起, 而目标
        # 同时惩罚 ||s1||^2. 最优折中是 "最小范数插值":
        #     a_k^0 = -POSE * T^k / (1 + C),  C = sum_{k=1}^{K-1} T^{2k},
        #     s1    = POSE / (1 + C),
        # 而不是 s1 = POSE. 因此这里直接验证连接等式和该解析解.
        np.testing.assert_allclose(coeffs0[:, 0], POSE, atol=1e-7)

        duration = 0.3
        num_coeffs = coeffs0.shape[1]
        norm = 1.0 + sum(duration ** (2 * k) for k in range(1, num_coeffs))
        expected_end = POSE / norm
        for k in range(1, num_coeffs):
            np.testing.assert_allclose(
                coeffs0[:, k], -POSE * duration**k / norm, atol=1e-7
            )

        # 连接约束的实质: 上一段终点 == 下一段起点.
        end0 = sum(
            coeffs0[:, k] * duration**k for k in range(num_coeffs)
        )
        np.testing.assert_allclose(end0, coeffs1[:, 0], atol=1e-7)
        np.testing.assert_allclose(coeffs1[:, 0], expected_end, atol=1e-7)
        # phase 1 除起点列外没有被约束, 二次目标把它们压到 0.
        np.testing.assert_allclose(coeffs1[:, 1:], 0.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
