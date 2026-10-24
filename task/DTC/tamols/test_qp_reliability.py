"""QP 后端可靠性验收测试 (任务书四).

运行:
    python -m unittest myrobot_mjlab.task.DTC.tamols.test_qp_reliability -v

验收项:
  1. 原始失败 fixture (crawl_flat it0, n=105/p=42/m=223): C++ 后端
     (Jacobi 变量缩放) 的原问题线性化约束违反从 ~1.31e-2 降至 <=1e-8;
     关闭缩放可复现原始失败 (并验证保护层会拒绝);
  2. 同一 fixture 上多后端同输入比较: 比较原问题残差与目标值
     (独立 NumPy 复算, 不复用被测残差函数);
  3. static 场景与解析 shim 用例不回归 (由 test_gn_sqp 的用例覆盖,
     这里只断言 fixture 级的等式/激活/不可行行为);
  4. 随机凸 QP 回归 (固定种子): 可行性/有限解/残差;
  5. 集成: crawl_flat 全 SQP 用 cpp 后端, 无回退且与 IPOPT 交叉一致;
     raw 模式下保护层回退仍生效.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from myrobot_mjlab.task.DTC.tamols.gn_sqp_solver import (
    GNSQPOptions,
    QuadProgCppQP,
    QuadProgQP,
    solve_gn_sqp,
)
from myrobot_mjlab.task.DTC.tamols.qp_reliability import (
    QPData,
    diagnose_qp,
    kkt_metrics,
    load_qp,
    qp_metrics,
    solve_qp_with,
)
from myrobot_mjlab.task.DTC.tamols.synthetic_scenarios import (
    CONTACT_NORMAL,
    build_scenario,
    run_scenario,
)
from myrobot_mjlab.task.DTC.tamols.trajectory_audit import compare_solutions

FIXTURE = (
    Path(__file__).parent
    / "test" / "fixtures" / "qp_crawl_flat_iter0.npz"
)
TOL = 1e-8


def _fixture() -> QPData:
    return load_qp(FIXTURE)


def _blocks(nlp, guess):
    coeffs, desired = guess
    return [np.asarray(c, float) for c in coeffs] + [
        np.asarray(desired, float)
    ] + ([np.zeros(nlp.layout.num_phases)] if nlp.layout.use_slack else [])


class TestFixtureRepair(unittest.TestCase):
    """验收 1: 原始失败 fixture 的修复效果 (原问题空间独立复算)."""

    @classmethod
    def setUpClass(cls):
        cls.qp = _fixture()
        cls.diag = diagnose_qp(cls.qp)

    def test_fixture_documents_original_failure(self):
        # fixture 复现原始失败: raw (无缩放) 的 C++ 后端 ineq viol ~1e-2
        x_raw, _, _ = solve_qp_with(self.qp, "quadprog_cpp_raw")
        m = qp_metrics(self.qp, x_raw)
        self.assertGreater(m["ineq_violation"], 1e-4)
        # 修复后: 缩放版通过容差
        x_fix, _, _ = solve_qp_with(self.qp, "quadprog_cpp")
        m2 = qp_metrics(self.qp, x_fix)
        self.assertLess(m2["eq_residual"], TOL)
        self.assertLess(m2["ineq_violation"], TOL)
        self.assertTrue(m2["finite"])

    def test_repaired_objective_matches_reference_backends(self):
        x_fix, _, _ = solve_qp_with(self.qp, "quadprog_cpp")
        obj_fix = qp_metrics(self.qp, x_fix)["objective"]
        for be in ("quadprog", "qpoases"):
            x_ref, _, _ = solve_qp_with(self.qp, be)
            m = qp_metrics(self.qp, x_ref)
            self.assertLess(m["eq_residual"], TOL)
            self.assertLess(m["ineq_violation"], TOL)
            self.assertLess(abs(obj_fix - m["objective"]), 1e-8, be)

    def test_raw_backend_is_rejected_by_protection_layer(self):
        """raw C++ 后端不通过线性化检查 => SQP 循环必须回退, 不得计成功."""
        nlp, guess, _ = build_scenario("crawl_flat", CONTACT_NORMAL)
        nlp.add_initial_state_constraints()
        nlp.add_junction_constraints()
        nlp.add_dynamic_stability_constraints()
        nlp.add_kinematic_contact_constraints()
        nlp.add_objectives()
        nlp.finalize()
        rep = solve_gn_sqp(
            nlp, _blocks(nlp, guess),
            GNSQPOptions(qp_plugin="quadprog_cpp", cpp_var_scaling=False),
        )
        # 保护层: 每个含问题 QP 的迭代都回退到可信后端, 结果仍收敛
        self.assertTrue(rep.success, rep.status)
        notes = [w for w in rep.warnings if "改用" in w]
        self.assertGreater(len(notes), 0, "期望保护层回退记录")

    def test_diagnostic_properties_recorded(self):
        d = self.diag
        self.assertTrue(d["finite_input"])
        self.assertLess(d["hessian_symmetry_err"], 1e-12)
        self.assertGreater(d["hessian_cond"], 1e11)  # 记录病态程度
        self.assertGreater(d["a_ineq_rownorm_spread"], 1e2)  # 行尺度差异


class TestRandomConvexQPs(unittest.TestCase):
    """验收 5: 随机凸 QP (固定种子), 可行性/有限解/残差."""

    def test_random_qps_cpp_matches_reference(self):
        rng = np.random.default_rng(42)
        for case in range(12):
            n = rng.integers(8, 40)
            # 随机 SPD Hessian (可控条件数 1e2..1e8)
            cond = 10 ** rng.uniform(2, 8)
            Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
            ev = np.geomspace(1.0, 1.0 / cond, n)
            H = Q @ np.diag(ev) @ Q.T
            g = rng.normal(size=n)
            A = rng.normal(size=(n // 2 + 1, n))
            b_ub = np.abs(A @ rng.normal(size=n)) + 1.0  # 0 可行
            qp = QPData(H=H, g=g, A_eq=np.zeros((0, n)), b_eq=np.zeros(0),
                        A_ineq=A, b_ub=b_ub, meta={"case": int(case)})
            x_fix, _, _ = solve_qp_with(qp, "quadprog_cpp")
            m_fix = qp_metrics(qp, x_fix)
            self.assertTrue(m_fix["finite"], case)
            # 随机用例 cond 可达 1e8: 验收线取 SQP 保护层阈值 qp_lin_tol=1e-6
            # (fixture 更严: <=1e-8, 见 TestFixtureRepair).
            self.assertLess(m_fix["ineq_violation"], 1e-6, case)
            x_ref, _, _ = solve_qp_with(qp, "quadprog")
            m_ref = qp_metrics(qp, x_ref)
            self.assertLess(abs(m_fix["objective"] - m_ref["objective"]),
                            1e-7 * max(1.0, abs(m_ref["objective"])), case)
            x_q, _, _ = solve_qp_with(qp, "qpoases")
            self.assertLess(
                abs(m_fix["objective"]
                    - qp_metrics(qp, x_q)["objective"]),
                1e-7 * max(1.0, abs(m_ref["objective"])), case,
            )

    def test_random_qps_with_equalities(self):
        rng = np.random.default_rng(7)
        for case in range(6):
            n = 20
            Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
            H = Q @ np.diag(np.geomspace(1.0, 1e-4, n)) @ Q.T
            g = rng.normal(size=n)
            A_eq = rng.normal(size=(3, n))
            x0 = rng.normal(size=n)
            b_eq = A_eq @ x0
            A = rng.normal(size=(10, n))
            b_ub = A @ x0 + np.abs(rng.normal(size=10)) + 0.5
            qp = QPData(H=H, g=g, A_eq=A_eq, b_eq=b_eq,
                        A_ineq=A, b_ub=b_ub, meta={"case": int(case)})
            x_fix, _, _ = solve_qp_with(qp, "quadprog_cpp")
            m = qp_metrics(qp, x_fix)
            self.assertTrue(m["finite"], case)
            self.assertLess(m["eq_residual"], 1e-8, case)
            self.assertLess(m["ineq_violation"], 1e-8, case)


class TestAnalyticCasesBothCppModes(unittest.TestCase):
    """验收 3: 解析用例在缩放开/关下都正确 (缩放是数学等价变换)."""

    def test_scaled_and_raw_agree_on_analytic(self):
        H = np.eye(3) * 2.0
        g = np.array([1.0, 2.0, 3.0])
        A = np.array([[1.0, 1.0, 0.0]])
        for var_scaling in (True, False):
            qp = QuadProgCppQP(var_scaling=var_scaling)
            r = qp(h=H, g=g, a=A,
                   lba=np.array([-np.inf]), uba=np.array([1.0]))
            np.testing.assert_allclose(r["x"], [-0.5, -1.0, -1.5], atol=1e-10)

    def test_infeasible_still_raises_in_both_modes(self):
        H = np.eye(2)
        A = np.array([[1.0, 0.0], [-1.0, 0.0]])
        for var_scaling in (True, False):
            qp = QuadProgCppQP(var_scaling=var_scaling)
            with self.assertRaises(Exception):
                qp(h=H, g=np.zeros(2), a=A,
                   lba=np.array([2.0, 2.0]), uba=np.array([2.0, 2.0]))


class TestKKTOptimality(unittest.TestCase):
    """求解正确性 = 可行性 (qp_metrics) + 最优性 (KKT, 凸 QP 充要).

    乘子由独立 NumPy/SciPy 从解恢复, 不复用任何被测后端的内部状态.
    """

    def test_fixture_backends_satisfy_kkt(self):
        qp = _fixture()
        for be in ("quadprog_cpp", "quadprog", "qpoases"):
            x, _, _ = solve_qp_with(qp, be)
            k = kkt_metrics(qp, x, active_tol=1e-6)
            scale = max(1.0, k["grad_inf_norm"])
            self.assertLess(k["stationarity"], 1e-5 * scale, be)
            self.assertLess(k["complementarity"], 1e-5, be)
            self.assertEqual(k["dual_infeasibility"], 0.0, be)

    def test_raw_backend_infeasible_so_kkt_inapplicable(self):
        """反例的正确表述: raw C++ 解 **primal 不可行** (ineq ~1e-2).
        KKT 最优性判据只对可行解适用; raw 解的 stationarity 小恰说明它
        是 "把违反的约束当作激活边界" 的移位可行域最优——这正是该后端
        数值缺陷的表现, 由可行性检查 (与 SQP 保护层) 拒绝, 不计入成功."""
        qp = _fixture()
        x_raw, _, _ = solve_qp_with(qp, "quadprog_cpp_raw")
        m = qp_metrics(qp, x_raw)
        self.assertGreater(m["ineq_violation"], 1e-4)

    def test_random_qps_satisfy_kkt(self):
        rng = np.random.default_rng(123)
        for case in range(6):
            n = 25
            Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
            H = Q @ np.diag(np.geomspace(1.0, 10 ** -rng.uniform(2, 7), n)) @ Q.T
            g = rng.normal(size=n)
            A_eq = rng.normal(size=(2, n))
            x0 = rng.normal(size=n)
            A = rng.normal(size=(8, n))
            qp = QPData(H=H, g=g, A_eq=A_eq, b_eq=A_eq @ x0,
                        A_ineq=A, b_ub=A @ x0 + np.abs(rng.normal(size=8)) + 0.3,
                        meta={"case": case})
            x, _, _ = solve_qp_with(qp, "quadprog_cpp")
            k = kkt_metrics(qp, x, active_tol=1e-6)
            scale = max(1.0, k["grad_inf_norm"])
            self.assertLess(k["stationarity"], 1e-5 * scale, case)
            self.assertLess(k["complementarity"], 1e-5, case)
            self.assertEqual(k["dual_infeasibility"], 0.0, case)




class TestNLPSolutionCorrectness(unittest.TestCase):
    """NLP 级求解正确性: 两条独立求解路径的解都通过独立参考复算,
    且总目标一致; 最优性由 QP 层 KKT (凸子问题) 与双路径交叉共同保证."""

    def test_both_solvers_pass_independent_residual_check(self):
        from myrobot_mjlab.task.DTC.tamols.synthetic_scenarios import (
            build_scenario,
            check_consistency,
            independent_residual_check,
        )
        from myrobot_mjlab.task.DTC.tamols.gn_sqp_solver import solve_gn_sqp

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
        rep_g = solve_gn_sqp(nlp, blocks)
        self.assertTrue(rep_g.success, rep_g.status)
        nlp.set_initial_guess(coeffs, desired)
        rep_i = nlp.solve("correctness-ipopt")
        self.assertTrue(rep_i.success, rep_i.status)
        for label, report in (("gn_sqp", rep_g), ("ipopt", rep_i)):
            checks = independent_residual_check(nlp, report)
            self.assertTrue(checks, label)
            self.assertEqual(check_consistency(checks), [], label)
        # 两条路径总目标一致 (同一 NLP 的同一局部盆地)
        self.assertLess(
            abs(rep_g.objective_total - rep_i.objective_total), 1e-4
        )


class TestIntegrationWithScaling(unittest.TestCase):
    """验收 6: 集成层——缩放后 cpp 无回退且与 IPOPT 交叉一致."""

    def test_crawl_flat_pure_cpp_no_fallback(self):
        nlp, guess, _ = build_scenario("crawl_flat", CONTACT_NORMAL)
        nlp.add_initial_state_constraints()
        nlp.add_junction_constraints()
        nlp.add_dynamic_stability_constraints()
        nlp.add_kinematic_contact_constraints()
        nlp.add_objectives()
        nlp.finalize()
        rep_g = solve_gn_sqp(nlp, _blocks(nlp, guess))
        self.assertTrue(rep_g.success, rep_g.status)
        self.assertEqual(rep_g.versions["qp"], "quadprog_cpp")
        notes = [w for w in rep_g.warnings if "改用" in w]
        self.assertEqual(notes, [], f"期望零回退, 实际 {notes}")
        nlp.set_initial_guess(*guess)
        rep_i = nlp.solve("crawl-ipopt")
        self.assertTrue(rep_i.success)
        diff = compare_solutions(rep_g, rep_i)
        self.assertLess(diff["max_foothold_delta_m"], 1e-4)
        self.assertLess(abs(rep_g.objective_total - rep_i.objective_total), 1e-4)


if __name__ == "__main__":
    unittest.main()
