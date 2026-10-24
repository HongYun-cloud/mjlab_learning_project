"""JIT 快路径 (codegen 混合路线) 的正确性验证与计时.

运行:
    python -m unittest myrobot_mjlab.task.DTC.tamols.test_gn_sqp_fast -v

快路径 = ``GNSQPOptions(jit=True)``: 求值函数编译为 C (数学不变),
extract 结果按结构版本缓存, 线搜索用轻量值函数.

正确性验证 (对用户要求 "和 python 的离线版本做比较"):
  1. 同输入下 fast(JIT) vs python(非 JIT) vs IPOPT 三方解一致
     (落脚点/系数/松弛/目标值);
  2. fast 解通过独立 NumPy 参考复算 (independent_residual_check);
  3. fast 解通过物理审计 (trajectory_audit);
  4. KKT: fixture 级已由 test_qp_reliability 覆盖 (与 jit 无关);
  5. 计时: 首次 (含 JIT 编译 ~30 s, 一次性) / 冷启动 / 暖启动分项;
     暖启动预期 10 ms 级 (论文 V-C 量级), 断言用宽松上界防回归,
     实际值以打印为准 (机器相关).
"""

from __future__ import annotations

import unittest

import numpy as np

from myrobot_mjlab.task.DTC.tamols.gn_sqp_solver import (
    GNSQPOptions,
    solve_gn_sqp,
)
from myrobot_mjlab.task.DTC.tamols.synthetic_scenarios import (
    CONTACT_NORMAL,
    build_scenario,
    check_consistency,
    independent_residual_check,
    run_scenario,
)
from myrobot_mjlab.task.DTC.tamols.trajectory_audit import (
    audit_trajectory,
    compare_solutions,
)

FAST = GNSQPOptions(jit=True)
PYTHON = GNSQPOptions(jit=False)


def _blocks(nlp, guess):
    coeffs, desired = guess
    return [np.asarray(c, float) for c in coeffs] + [
        np.asarray(desired, float)
    ] + ([np.zeros(nlp.layout.num_phases)] if nlp.layout.use_slack else [])


def _build(name, pin=False):
    nlp, guess, _ = build_scenario(name, CONTACT_NORMAL)
    if pin:
        nlp.add_constant_base_pin(0)
    nlp.add_initial_state_constraints()
    nlp.add_junction_constraints()
    nlp.add_dynamic_stability_constraints()
    nlp.add_kinematic_contact_constraints()
    nlp.add_objectives()
    nlp.finalize()
    return nlp, guess


class TestFastPathCorrectness(unittest.TestCase):
    """fast(JIT) 与 python 离线版在同一 NLP/同初值上的交叉验证."""

    @classmethod
    def setUpClass(cls):
        cls.nlp, cls.guess = _build("crawl_flat")
        blocks = _blocks(cls.nlp, cls.guess)
        with __import__("contextlib").redirect_stdout(
            __import__("io").StringIO()
        ):
            import time

            t0 = time.perf_counter()
            cls.rep_fast = solve_gn_sqp(cls.nlp, blocks, FAST)
            cls.t_first = time.perf_counter() - t0
            cls.rep_py = solve_gn_sqp(cls.nlp, blocks, PYTHON)
            cls.nlp.set_initial_guess(*cls.guess)
            cls.rep_ipopt = cls.nlp.solve("fast-check-ipopt")

    def test_fast_matches_python_offline(self):
        """用户要求的核心比较: fast vs python 离线版."""
        self.assertTrue(self.rep_fast.success, self.rep_fast.status)
        self.assertTrue(self.rep_py.success, self.rep_py.status)
        diff = compare_solutions(self.rep_fast, self.rep_py)
        self.assertLess(diff["max_foothold_delta_m"], 1e-8)
        self.assertLess(diff["max_coeff_delta"], 1e-6)
        self.assertLess(
            abs(self.rep_fast.objective_total - self.rep_py.objective_total),
            1e-8,
        )

    def test_fast_matches_ipopt(self):
        self.assertTrue(self.rep_ipopt.success)
        diff = compare_solutions(self.rep_fast, self.rep_ipopt)
        self.assertLess(diff["max_foothold_delta_m"], 1e-4)
        self.assertLess(
            abs(self.rep_fast.objective_total - self.rep_ipopt.objective_total),
            1e-4,
        )

    def test_fast_passes_independent_residual_check(self):
        checks = independent_residual_check(self.nlp, self.rep_fast)
        self.assertTrue(checks)
        self.assertEqual(check_consistency(checks), [])

    def test_fast_passes_physical_audit(self):
        audit = audit_trajectory(self.nlp, self.rep_fast)
        self.assertEqual(audit.violations, [])

    def test_fast_static_matches_python(self):
        nlp, guess = _build("static_flat", pin=True)
        blocks = _blocks(nlp, guess)
        with __import__("contextlib").redirect_stdout(
            __import__("io").StringIO()
        ):
            rep_f = solve_gn_sqp(nlp, blocks, FAST)
            rep_p = solve_gn_sqp(nlp, blocks, PYTHON)
        self.assertTrue(rep_f.success and rep_p.success)
        diff = compare_solutions(rep_f, rep_p)
        self.assertLess(diff["max_coeff_delta"], 1e-6)
        self.assertLess(
            abs(rep_f.objective_total - rep_p.objective_total), 1e-8
        )


class TestFastPathTiming(unittest.TestCase):
    """计时分项: 首次 (编译, 一次性) / 冷启动 / 暖启动."""

    def test_warm_start_budget(self):
        import contextlib
        import io
        import time

        import numpy as np

        nlp, guess = _build("crawl_flat")
        blocks = _blocks(nlp, guess)
        with contextlib.redirect_stdout(io.StringIO()):
            t0 = time.perf_counter()
            solve_gn_sqp(nlp, blocks, FAST)  # 首次: extract + JIT 编译
            t_first = time.perf_counter() - t0
            t0 = time.perf_counter()
            r_cold = solve_gn_sqp(nlp, blocks, FAST)
            t_cold = time.perf_counter() - t0
            sol = r_cold.solution
            warm = [np.asarray(c, float) for c in sol["spline_coeffs"]] + [
                np.asarray(sol["desired_footholds"], float),
                np.asarray(sol["slack"], float),
            ]
            ts = []
            for _ in range(50):
                t0 = time.perf_counter()
                r_w = solve_gn_sqp(nlp, warm, FAST)
                ts.append(time.perf_counter() - t0)
        warm_ms = float(np.mean(ts)) * 1000
        print(
            f"[timing] first={t_first*1000:.0f} ms (one-off compile) | "
            f"cold={t_cold*1000:.1f} ms ({r_cold.iterations} it) | "
            f"warm={warm_ms:.1f} ms ({r_w.iterations} it, {r_w.status})"
        )
        # 宽松上界防回归 (机器相关, 实际值以打印为准);
        # 暖启动为论文重规划场景 (1-2 迭代), 预期 10 ms 级.
        self.assertLess(warm_ms, 100.0)
        self.assertTrue(r_w.success)
        # 暖启动解必须与冷启动一致
        diff = compare_solutions(r_w, r_cold)
        self.assertLess(diff["max_foothold_delta_m"], 1e-8)


class TestFastDoesNotAffectDefault(unittest.TestCase):
    """默认 (jit=False) 路径行为不变: 场景套件照常通过."""

    def test_default_python_path_unchanged(self):
        r = run_scenario("crawl_flat", CONTACT_NORMAL, verbose=False)
        self.assertTrue(r.report.success)
        self.assertEqual(check_consistency(r.independent_checks), [])


if __name__ == "__main__":
    unittest.main()
