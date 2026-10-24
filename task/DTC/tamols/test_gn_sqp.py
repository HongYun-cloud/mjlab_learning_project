"""GN+SQP 第二求解路径的测试 (与 IPOPT 交叉验证).

运行:
    python -m unittest myrobot_mjlab.task.DTC.tamols.test_gn_sqp -v

验证协议
--------
1. 冷启动一致性: 静置/平地行走等"盆地单一"的场景, 两条路径应收敛到
   同一局部解 (落脚点/目标值在紧容差内一致);
2. 热启动 KKT 交叉检查: 以 IPOPT 解为初值跑 GN-SQP, 应停留在同一解
   附近 (目标值差 <= 1e-2 绝对量, 落脚点差 <= 1e-3 m) —— 这验证 IPOPT
   解对 SQP 的 QP 序列也是 (近似) 驻点;
3. 不同局部解如实记录: crawl_step 冷启动 GN-SQP 可能落入另一局部最优,
   测试只要求其可行且物理审计无违反, **不**要求与 IPOPT 同值;
4. GN Hessian 半正定 (J_r^T J_r + reg I);
5. 不可行场景: GN-SQP 不得报告 success;
6. 下游兼容: trajectory_audit / trajectory_export 对 GN-SQP 报告同样可用.
"""

from __future__ import annotations

import unittest

import numpy as np

from myrobot_mjlab.task.DTC.tamols.gn_sqp_solver import (
    GNSQPOptions,
    _make_qp,
    extract_nlp,
    solve_gn_sqp,
)
from myrobot_mjlab.task.DTC.tamols.synthetic_scenarios import (
    CONTACT_NORMAL,
    build_scenario,
    run_scenario,
)
from myrobot_mjlab.task.DTC.tamols.trajectory_audit import (
    audit_trajectory,
    compare_solutions,
)
from myrobot_mjlab.task.DTC.tamols.trajectory_export import (
    check_export,
    export_trajectory,
)


def _build(name, cfg=CONTACT_NORMAL, pin=False):
    nlp, guess, _ = build_scenario(name, cfg)
    if pin:
        nlp.add_constant_base_pin(0)
    nlp.add_initial_state_constraints()
    nlp.add_junction_constraints()
    nlp.add_dynamic_stability_constraints()
    nlp.add_kinematic_contact_constraints()
    nlp.add_objectives()
    nlp.finalize()
    return nlp, guess


def _blocks_from_guess(nlp, guess):
    coeffs, desired = guess
    blocks = [np.asarray(c, dtype=float) for c in coeffs] + [
        np.asarray(desired, dtype=float)
    ]
    if nlp.layout.use_slack:
        blocks.append(np.zeros(nlp.layout.num_phases))
    return blocks


def _blocks_from_report(nlp, report):
    sol = report.solution
    blocks = [np.asarray(c, dtype=float) for c in sol["spline_coeffs"]] + [
        np.asarray(sol["desired_footholds"], dtype=float)
    ]
    if nlp.layout.use_slack:
        blocks.append(np.asarray(sol["slack"], dtype=float))
    return blocks


class _QuadProgAdapterCases:
    """解析可判的 QP 用例; 由 py 包装与 C++ shim 两个子类共用."""

    adapter_cls = None
    skip_reason = None

    def setUp(self):
        if self.skip_reason:
            raise unittest.SkipTest(self.skip_reason)
        self.qp = self.adapter_cls()

    def test_unconstrained_like_inequality_inactive(self):
        H = np.eye(3) * 2.0
        g = np.array([1.0, 2.0, 3.0])
        A = np.array([[1.0, 1.0, 0.0]])
        r = self.qp(
            h=H, g=g, a=A, lba=np.array([-np.inf]), uba=np.array([1.0])
        )
        # 无约束最优 x = -g/2 = (-0.5,-1,-1.5), 约束 x0+x1=-1.5 <= 1 不激活
        np.testing.assert_allclose(r["x"], [-0.5, -1.0, -1.5], atol=1e-10)

    def test_equality_constraint_analytic(self):
        # min 0.5||x||^2 - c.x s.t. sum(x)=1 -> x = c + λ1, λ=(1-Σc)/3
        c = np.array([1.0, 2.0, 3.0])
        H = np.eye(3)
        A = np.ones((1, 3))
        r = self.qp(h=H, g=-c, a=A, lba=np.array([1.0]), uba=np.array([1.0]))
        lam = (1.0 - c.sum()) / 3.0
        np.testing.assert_allclose(r["x"], c + lam, atol=1e-10)

    def test_active_inequality(self):
        # min 0.5||x||^2 - c.x s.t. x0 + x1 <= 1, c=(2,2,0): 约束激活
        c = np.array([2.0, 2.0, 0.0])
        H = np.eye(3)
        A = np.array([[1.0, 1.0, 0.0]])
        r = self.qp(
            h=H, g=-c, a=A, lba=np.array([-np.inf]), uba=np.array([1.0])
        )
        # KKT: x = c - μ(1,1,0), x0+x1=1 -> 4-2μ=1 -> μ=1.5 -> x=(0.5,0.5,0)
        np.testing.assert_allclose(r["x"], [0.5, 0.5, 0.0], atol=1e-10)

    def test_infeasible_qp_raises(self):
        H = np.eye(2)
        A = np.array([[1.0, 0.0], [-1.0, 0.0]])
        with self.assertRaises(Exception):
            self.qp(
                h=H, g=np.zeros(2), a=A,
                lba=np.array([2.0, 2.0]), uba=np.array([2.0, 2.0]),
            )


def _make_adapter_classes():
    from myrobot_mjlab.task.DTC.tamols.gn_sqp_solver import (
        QuadProgCppQP,
        QuadProgQP,
    )

    py_skip = None
    try:
        import quadprog  # noqa: F401
    except ImportError:
        py_skip = "python quadprog 未安装"
    cpp_skip = None
    try:
        QuadProgCppQP()
    except (FileNotFoundError, OSError):
        cpp_skip = "QuadProg++ C++ shim 未编译 (cpp/build_shim.sh)"

    class TestQuadProgPyAdapter(_QuadProgAdapterCases, unittest.TestCase):
        adapter_cls = QuadProgQP
        skip_reason = py_skip

    class TestQuadProgCppAdapter(_QuadProgAdapterCases, unittest.TestCase):
        adapter_cls = QuadProgCppQP
        skip_reason = cpp_skip

    return TestQuadProgPyAdapter, TestQuadProgCppAdapter, cpp_skip


(
    TestQuadProgPyAdapter,
    TestQuadProgCppAdapter,
    _CPP_SKIP,
) = _make_adapter_classes()


class TestGNSQPBasics(unittest.TestCase):
    def test_qp_plugin_available(self):
        nlp, _ = _build("static_flat", pin=True)
        ex = extract_nlp(nlp)
        qp, plugin = _make_qp(ex, GNSQPOptions())
        self.assertIn(
            plugin, ("quadprog_cpp", "quadprog", "qpoases", "qrqp", "osqp")
        )

    def test_gn_hessian_psd(self):
        nlp, guess = _build("crawl_flat")
        ex = extract_nlp(nlp)
        rng = np.random.default_rng(5)
        x = np.concatenate([b.reshape(-1) for b in _blocks_from_guess(nlp, guess)])
        x = x + rng.normal(0, 0.02, size=x.size)
        out = ex.f_eval(x)
        Jr = np.asarray(out[7])
        H = Jr.T @ Jr + GNSQPOptions().hessian_reg * np.eye(ex.n)
        ev = np.linalg.eigvalsh(0.5 * (H + H.T))
        self.assertGreaterEqual(ev.min(), GNSQPOptions().hessian_reg * 0.5)

    def test_extract_dimensions(self):
        nlp, _ = _build("crawl_flat")
        ex = extract_nlp(nlp)
        self.assertEqual(ex.n, nlp.layout.total_size)
        self.assertEqual(ex.m_ineq + ex.m_eq, len(nlp.constraint_entries))
        self.assertEqual(ex.m_eq, sum(
            1 for e in nlp.constraint_entries if e.kind == "equality"
        ))


class TestCrossValidation(unittest.TestCase):
    def test_static_cold_agreement(self):
        nlp, guess = _build("static_flat", pin=True)
        rep_g = solve_gn_sqp(nlp, _blocks_from_guess(nlp, guess))
        self.assertTrue(rep_g.success, rep_g.status)
        nlp.set_initial_guess(*guess)
        rep_i = nlp.solve("static-ipopt")
        self.assertTrue(rep_i.success)
        diff = compare_solutions(rep_g, rep_i)
        self.assertLess(diff["max_foothold_delta_m"], 1e-6)
        self.assertLess(diff["max_coeff_delta"], 1e-6)
        self.assertLess(diff["max_slack_delta"], 1e-3)
        self.assertLess(
            abs(rep_g.objective_total - rep_i.objective_total), 1e-5
        )

    def test_crawl_flat_cold_agreement(self):
        nlp, guess = _build("crawl_flat")
        rep_g = solve_gn_sqp(nlp, _blocks_from_guess(nlp, guess))
        self.assertTrue(rep_g.success, rep_g.status)
        nlp.set_initial_guess(*guess)
        rep_i = nlp.solve("crawl-ipopt")
        self.assertTrue(rep_i.success)
        diff = compare_solutions(rep_g, rep_i)
        self.assertLess(diff["max_foothold_delta_m"], 1e-4)
        self.assertLess(
            abs(rep_g.objective_total - rep_i.objective_total), 1e-4
        )

    def test_crawl_step_warm_kkt_check(self):
        nlp, guess = _build("crawl_step")
        nlp.set_initial_guess(*guess)
        rep_i = nlp.solve("step-ipopt")
        self.assertTrue(rep_i.success)
        rep_g = solve_gn_sqp(nlp, _blocks_from_report(nlp, rep_i))
        self.assertTrue(rep_g.success, rep_g.status)
        diff = compare_solutions(rep_g, rep_i)
        self.assertLess(diff["max_foothold_delta_m"], 1e-3)
        self.assertLess(
            abs(rep_g.objective_total - rep_i.objective_total), 1e-2
        )
        audit = audit_trajectory(nlp, rep_g)
        self.assertEqual(audit.violations, [])

    def test_crawl_step_cold_may_find_other_local_optimum(self):
        """冷启动 GN-SQP 允许落入不同局部最优: 只要求可行+物理自洽."""
        nlp, guess = _build("crawl_step")
        rep_g = solve_gn_sqp(nlp, _blocks_from_guess(nlp, guess))
        self.assertTrue(rep_g.success, rep_g.status)
        self.assertLess(
            max(rep_g.max_violation_by_category.values()), 1e-6
        )
        audit = audit_trajectory(nlp, rep_g)
        self.assertEqual(audit.violations, [])
        print(
            "[observe] crawl_step cold GN-SQP obj =",
            f"{rep_g.objective_total:.6g}",
        )

    def test_infeasible_scenario_not_success(self):
        nlp, guess = _build("infeasible_reach")
        rep_g = solve_gn_sqp(
            nlp, _blocks_from_guess(nlp, guess),
            GNSQPOptions(max_iter=15),
        )
        self.assertFalse(rep_g.success)
        self.assertNotEqual(rep_g.solution_source, "gn_sqp_solution")


class TestQPPluginAgreement(unittest.TestCase):
    def test_default_plugin_is_quadprog_cpp(self):
        nlp, guess = _build("static_flat", pin=True)
        rep = solve_gn_sqp(nlp, _blocks_from_guess(nlp, guess))
        if _CPP_SKIP:
            self.assertIn(rep.versions["qp"], ("quadprog", "qpoases"))
            raise unittest.SkipTest(_CPP_SKIP)
        self.assertEqual(rep.versions["qp"], "quadprog_cpp")

    def test_three_qp_backends_agree_on_crawl_flat(self):
        nlp, guess = _build("crawl_flat")
        blocks = _blocks_from_guess(nlp, guess)
        reports = {}
        for plugin in ("quadprog_cpp", "quadprog", "qpoases"):
            if plugin == "quadprog_cpp" and _CPP_SKIP:
                continue
            reports[plugin] = solve_gn_sqp(
                nlp, blocks, GNSQPOptions(qp_plugin=plugin)
            )
        self.assertGreaterEqual(len(reports), 2)
        for plugin, rep in reports.items():
            self.assertTrue(rep.success, f"{plugin}: {rep.status}")
            self.assertEqual(rep.versions["qp"], plugin)
        keys = list(reports)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                diff = compare_solutions(reports[keys[i]], reports[keys[j]])
                self.assertLess(diff["max_foothold_delta_m"], 1e-6, keys)
                self.assertLess(
                    abs(
                        reports[keys[i]].objective_total
                        - reports[keys[j]].objective_total
                    ),
                    1e-6,
                    keys,
                )


class TestDownstreamCompatibility(unittest.TestCase):
    def test_audit_and_export_on_gn_sqp_report(self):
        nlp, guess = _build("crawl_flat")
        rep_g = solve_gn_sqp(nlp, _blocks_from_guess(nlp, guess))
        self.assertTrue(rep_g.success)
        audit = audit_trajectory(nlp, rep_g)
        self.assertEqual(audit.violations, [])
        export = export_trajectory(nlp, rep_g, samples_per_phase=13)
        self.assertEqual(check_export(nlp, export), [])
        self.assertTrue(np.isfinite(export.touchdown_joint_pos[0]).all())
        # state_vector 对 GN-SQP 报告同样可用
        x = rep_g.state_vector(nlp.layout)
        self.assertEqual(x.size, nlp.layout.total_size)


if __name__ == "__main__":
    unittest.main()
