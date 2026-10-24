"""完整 NLP 集成的单元测试.

运行:
    python -m unittest myrobot_mjlab.task.DTC.tamols.test_nlp_integration -v

覆盖 (对应用户任务书):
  一、变量布局: 形状/索引/单位表, pack/unpack 往返, 系数展平顺序与
      ``spline_value`` 一致, 参数不进入决策变量块.
  二、统一 L_dot: 解析式 vs CasADi 符号时间求导 vs 独立数值差分;
      欧拉角导数 -> 机体系角速度映射不与恒等映射混淆;
      集成接口拒绝外部独立 L_dot.
  三、集成: 权重只乘一次; V-B2 距离/平方距离单位区分; 采样个数与
      T_k 权重因子; 目标分项 (未加权/权重/加权/占比).
  四、假设: 静止基准 (字面符号不可行, contact-normal 可行);
      凸性 p20 字面 vs p21 工程版在平移/旋转下的行为; 常数约束装配期
      不可行路径.
  五、场景: 5 个合成场景状态/独立复算一致性 (详细打印在
      ``synthetic_scenarios`` CLI).
  六、数值: 有限差分梯度 (总目标/地形插值/GIAC/L_dot); NaN/Inf 与
      地图覆盖警告; 热启动 vs 冷启动.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from myrobot_mjlab.task.DTC.tamols import solver_backend
from myrobot_mjlab.task.DTC.tamols.assumptions_config import AssumptionsConfig
from myrobot_mjlab.task.DTC.tamols.nlp_integration import (
    IntegratedNLP,
    IntegratedNLPLayout,
    LegSchedule,
    NLPParameters,
    trajectory_angular_momentum,
    trajectory_angular_momentum_rate,
)
from myrobot_mjlab.task.DTC.tamols.synthetic_scenarios import (
    CONTACT_NORMAL,
    GO2_FEET_STANCE,
    LITERAL,
    build_scenario,
    check_consistency,
    flat_maps,
    ref_giac,
    ref_L_dot_fd,
    ref_omega,
    ref_rot_zyx,
    run_scenario,
)
from myrobot_mjlab.task.DTC.tamols.v_b_objectives import (
    GO2_INERTIA_B,
    VBWeights,
    assemble_total_objective,
    spline_value,
)

CASADI_AVAILABLE = solver_backend.ca is not None


def num_grad(fn, x0, eps=1e-6):
    x0 = np.asarray(x0, dtype=float)
    grad = np.zeros_like(x0)
    for i in range(x0.size):
        plus = x0.copy()
        minus = x0.copy()
        plus[i] += eps
        minus[i] -= eps
        grad[i] = (fn(plus) - fn(minus)) / (2 * eps)
    return grad


def build_simple_nlp(assumptions=None, weights=None, **param_kwargs):
    maps = flat_maps()
    schedules = {
        leg: LegSchedule(True, measured_position=tuple(GO2_FEET_STANCE[leg]))
        for leg in range(4)
    }
    solver_options = param_kwargs.pop(
        "solver_options",
        {"print_time": False, "ipopt": {"print_level": 0, "max_iter": 200}},
    )
    params = NLPParameters(
        base_pose0=[0, 0, 0.30, 0, 0, 0],
        base_twist0=np.zeros(6),
        measured_feet={leg: GO2_FEET_STANCE[leg] for leg in range(4)},
        ref_twist=lambda p, t: (np.zeros(3), np.zeros(3)),
        solver_options=solver_options,
        **param_kwargs,
    )
    nlp = IntegratedNLP(
        durations=[0.3], schedules=schedules, maps=maps, params=params,
        assumptions=assumptions or CONTACT_NORMAL,
        weights=weights or VBWeights(),
    )
    return nlp




# ---------------------------------------------------------------------- #
# 〇、集成一致性: 全部组件进入同一个 NLP; GIAC 与 V-B8 共用 L_dot 节点
# ---------------------------------------------------------------------- #
class TestIntegrationConsistency(unittest.TestCase):
    def _build(self):
        from myrobot_mjlab.task.DTC.tamols.synthetic_scenarios import (
            _crawl_schedules,
            flat_maps,
        )

        maps = flat_maps()
        params = NLPParameters(
            base_pose0=[0, 0, 0.30, 0, 0, 0],
            base_twist0=[0.25, 0, 0, 0, 0, 0],
            measured_feet={leg: GO2_FEET_STANCE[leg] for leg in range(4)},
            ref_twist=lambda p, t: (np.array([0.25, 0, 0.0]), np.zeros(3)),
            previous_footholds=GO2_FEET_STANCE.copy(),
            solver_options={"print_time": False,
                            "ipopt": {"print_level": 0, "max_iter": 200}},
        )
        nlp = IntegratedNLP(
            durations=[0.3, 0.3, 0.3], schedules=_crawl_schedules(),
            maps=maps, params=params, assumptions=CONTACT_NORMAL,
        )
        nlp.add_initial_state_constraints()
        nlp.add_junction_constraints()
        nlp.add_dynamic_stability_constraints()
        nlp.add_kinematic_contact_constraints()
        nlp.add_objectives()
        return nlp

    def test_all_components_in_single_nlp(self):
        nlp = self._build()
        cats = {e.category for e in nlp.constraint_entries}
        for want in ("V-A-1", "17b-17f", "5", "30", "31_lower", "31_upper", "layout"):
            self.assertIn(want, cats, f"约束类别 {want} 未进入 NLP")
        # 初始/连接等式确实存在
        self.assertTrue([e for e in nlp.constraint_entries
                         if e.name.startswith("initial_pose")])
        self.assertTrue([e for e in nlp.constraint_entries
                         if e.name.startswith("initial_twist")])
        self.assertTrue([e for e in nlp.constraint_entries
                         if e.name.startswith("junction_")])
        obj_cats = set(nlp.objective_set.by_category())
        self.assertEqual(
            obj_cats,
            {"V-B1", "V-B2", "V-B3", "V-B4", "V-B5", "V-B6", "V-B7", "V-B8", "29"},
        )
        # 同一个 backend: 约束条数 == 进入求解器的 entry 数 (常数 entry 除外)
        n_const = sum(
            1 for e in nlp.constraint_entries
            if isinstance(e.residual, (int, float, np.integer, np.floating))
        )
        self.assertEqual(
            len(nlp.backend.constraints), len(nlp.constraint_entries) - n_const
        )
        nlp.finalize()
        self.assertEqual(
            len(nlp.backend.objective_terms()), len(nlp.objective_set.records)
        )
        # 单一决策向量: 后端变量总维度 == 布局 total_size
        total_var = sum(int(np.prod(s)) for _, s in nlp.backend.variables)
        self.assertEqual(total_var, nlp.layout.total_size)

    def test_giac_and_smoothness_share_same_ldot_object(self):
        """GIAC 装配与 V-B8 装配必须拿到同一个缓存 L_dot 表达式对象."""
        from myrobot_mjlab.task.DTC.tamols.synthetic_scenarios import (
            _crawl_schedules,
            flat_maps,
        )

        maps = flat_maps()
        params = NLPParameters(
            base_pose0=[0, 0, 0.30, 0, 0, 0],
            base_twist0=np.zeros(6),
            measured_feet={leg: GO2_FEET_STANCE[leg] for leg in range(4)},
            ref_twist=lambda p, t: (np.zeros(3), np.zeros(3)),
            solver_options={"print_time": False,
                            "ipopt": {"print_level": 0, "max_iter": 50}},
        )
        nlp = IntegratedNLP(
            durations=[0.3], schedules=_crawl_schedules(), maps=maps,
            params=params, assumptions=CONTACT_NORMAL,
        )
        calls: list[tuple[str, int, float, int]] = []
        stage = {"now": "init"}
        orig = nlp.l_dot_expression

        def spy(phase, tau):
            out = orig(phase, tau)
            calls.append((stage["now"], int(phase), float(tau), id(out)))
            return out

        nlp.l_dot_expression = spy
        nlp.add_initial_state_constraints()
        stage["now"] = "giac"
        nlp.add_dynamic_stability_constraints()
        stage["now"] = "vb8"
        nlp.add_objectives()

        giac_calls = [c for c in calls if c[0] == "giac"]
        vb8_calls = [c for c in calls if c[0] == "vb8"]
        self.assertTrue(giac_calls and vb8_calls)
        # 同一 (phase, tau): 两阶段拿到的对象 id 必须相同 (缓存同一表达式)
        for key in {(c[1], c[2]) for c in giac_calls} & {(c[1], c[2]) for c in vb8_calls}:
            ids = {c[3] for c in calls if (c[1], c[2]) == key}
            self.assertEqual(len(ids), 1, f"(phase,tau)={key} 的 L_dot 对象不唯一: {ids}")
        # 每个采样点只构造一次 (缓存生效, 无重复表达式树)
        per_key = {}
        for c in calls:
            per_key.setdefault((c[1], c[2]), set()).add(c[3])
        self.assertTrue(all(len(v) == 1 for v in per_key.values()))

    def test_giac_and_smoothness_ldot_numerically_same(self):
        """随机样条点上: GIAC 残差与 V-B8 残差都与同一条独立 FD L_dot 一致."""
        import casadi as ca

        nlp = self._build()
        phase = 0
        tau = nlp.sample_times(0)[0]
        giac = [
            e for e in nlp.constraint_entries
            if e.category == "17b-17f" and e.phase == phase and e.tau == tau
        ]
        vb8 = [
            r for r in nlp.objective_set.records
            if r.category == "V-B8" and r.phase == phase and r.tau == tau
        ]
        self.assertTrue(giac and vb8)
        cv = nlp.vars["spline_coeffs"][phase]
        sv = nlp.vars["slack"]
        # GIAC 残差含 slack 项: 函数输入同时给样条与 slack (取 0)
        f_g = ca.Function(
            "fg", [cv, sv], [ca.vertcat(*[e.residual for e in giac])]
        )
        f_8 = ca.Function("f8", [cv], [vb8[0].residual])
        rng = np.random.default_rng(3)
        coeffs = rng.normal(0.0, 0.05, size=(6, 5))
        coeffs[0:3, 0] = [0.0, 0.0, 0.30]
        L_fd = ref_L_dot_fd(coeffs, GO2_INERTIA_B, tau)
        # V-B8: ||L_dot||^2 与 FD 一致
        self.assertAlmostEqual(
            float(f_8(coeffs)), float(L_fd @ L_fd), places=8
        )
        # GIAC: 用 FD 的 L_dot 复算每条残差 (N=3 相位, 脚为测量参数)
        pose = np.array([float(v) for v in spline_value(coeffs, tau, 0)])
        accel = np.array([float(v) for v in spline_value(coeffs, tau, 2)])
        a_B = accel[0:3] - np.array([0, 0, -9.81])  # CONTACT_NORMAL
        ordered, feet = nlp._ordered_stance(phase, nlp.assumptions.resolved_winding())
        vals = np.array(
            f_g(coeffs, np.zeros(nlp.layout.num_phases))
        ).ravel()
        n = len(ordered)
        pairs = [(i, (i + 1) % n) for i in range(n)]
        for idx, (i, j) in enumerate(pairs):
            ref = ref_giac(
                nlp.params.mass, pose[0:3], a_B, L_fd,
                np.array(feet[i]), np.array(feet[j]),
            )
            self.assertAlmostEqual(vals[idx], ref, places=8)

    def test_slack_upper_bound_enforced(self):
        """式 (29): eps <= 0 必须作为约束存在, 且解满足 (不允许正 slack)."""
        nlp = self._build()
        eps_entries = [
            e for e in nlp.constraint_entries
            if e.name.startswith("29_eps_upper")
        ]
        self.assertEqual(len(eps_entries), nlp.layout.num_phases)
        report = run_scenario("crawl_flat", CONTACT_NORMAL, verbose=False)
        self.assertTrue(report.report.success, report.report.status)
        slack = report.report.solution["slack"]
        self.assertLessEqual(float(slack.max()), 1e-6)
        # eps<=0 的约束违反统计中不应出现正的违反
        self.assertLessEqual(
            report.report.max_violation_by_category.get("29", 0.0), 1e-6
        )

    def test_ldot_shared_value_at_solution(self):
        """解处: GIAC 与 V-B8 的残差与同一条独立 FD L_dot 一致."""
        r = run_scenario("crawl_flat", CONTACT_NORMAL, verbose=False)
        self.assertTrue(r.report.success)
        checks = r.independent_checks
        giac_names = [k for k in checks if k.startswith("17b_pair")]
        vb8_names = [k for k in checks if k.startswith("VB8_smooth")]
        self.assertTrue(giac_names and vb8_names)
        for name in giac_names + vb8_names:
            rep, ref = checks[name]
            scale = max(1.0, abs(rep), abs(ref))
            self.assertLessEqual(abs(rep - ref), 1e-5 * scale, name)


# ---------------------------------------------------------------------- #
# 〇b、静止基准 (平地/静止机身/静态支撑): 当前默认配置的可行性报告
# ---------------------------------------------------------------------- #
class TestStaticBenchmark(unittest.TestCase):
    """纪律: 默认配置 (paper_literal + friction enabled) 下静止基准不可行;
    测试只确认并定位该事实, 不允许通过翻转符号让它通过."""

    def _assemble(self, cfg):
        nlp, guess, _ = build_scenario("static_flat", cfg)
        nlp.add_constant_base_pin(0)
        nlp.add_initial_state_constraints()
        nlp.add_junction_constraints()
        nlp.add_dynamic_stability_constraints()
        nlp.add_kinematic_contact_constraints()
        nlp.add_objectives()
        nlp.finalize()
        coeffs, desired = guess
        nlp.set_initial_guess(coeffs, desired)
        return nlp

    def test_default_config_reports_infeasible_with_source(self):
        from myrobot_mjlab.task.DTC.tamols.assumptions_config import (
            DEFAULT_ASSUMPTIONS,
        )
        from myrobot_mjlab.task.DTC.tamols.static_benchmark import (
            rest_residual_evidence,
        )

        # 默认配置必须是字面符号, 未被私自翻转
        self.assertEqual(DEFAULT_ASSUMPTIONS.inertial_accel_sign, "paper_literal")
        self.assertEqual(DEFAULT_ASSUMPTIONS.friction_mode, "enabled")
        self.assertFalse(DEFAULT_ASSUMPTIONS.include_push)
        nlp = self._assemble(DEFAULT_ASSUMPTIONS)
        self.assertEqual(nlp.assumptions.resolved_winding(), "ccw")
        # _accel_sign 在零加速度下返回字面 a_B = g (不是 -g)
        a_B = [float(v) for v in nlp._accel_sign((0.0, 0.0, 0.0))]
        np.testing.assert_allclose(a_B, [0.0, 0.0, -9.81], atol=1e-12)
        # 静止点残差证据: 摩擦法向 +9.81 (唯一违反), 其余动力学约束满足
        ev = rest_residual_evidence(nlp)
        normal = [v for k, v in ev["5"].items() if "normal" in k]
        cone = [v for k, v in ev["5"].items() if "cone" in k]
        self.assertTrue(normal)
        np.testing.assert_allclose(normal, 9.81, atol=1e-9)
        self.assertTrue(all(v <= 1e-9 for v in cone))  # 平方锥本身满足
        self.assertTrue(all(v <= 1e-9 for v in ev["17b-17f"].values()))
        self.assertTrue(all(v <= 1e-9 for v in ev["31_lower"].values()))
        self.assertTrue(all(v <= 1e-9 for v in ev["31_upper"].values()))
        self.assertTrue(all(v <= 1e-9 for v in ev["30"].values()))
        # 求解: 不可行, 且报告状态如实记录
        report = nlp.solve("static-default")
        self.assertFalse(report.success)
        self.assertIn("Infeasible", report.status)
        self.assertEqual(report.solution_source, "last_iterate_not_a_solution")

    def test_friction_disabled_is_explicit_config_and_feasible(self):
        from myrobot_mjlab.task.DTC.tamols.assumptions_config import (
            AssumptionsConfig,
        )

        cfg = AssumptionsConfig(friction_mode="disabled")
        nlp = self._assemble(cfg)
        report = nlp.solve("static-friction-off")
        self.assertTrue(report.success, report.status)
        self.assertLess(max(report.max_violation_by_category.values()), 1e-6)
        # 关闭后类别 5 的约束不应存在
        self.assertFalse(
            [e for e in nlp.constraint_entries if e.category == "5"]
        )

    def test_benchmark_cli_matrix(self):
        from myrobot_mjlab.task.DTC.tamols.static_benchmark import (
            run_static_benchmark,
        )

        results = run_static_benchmark(verbose=False)
        self.assertEqual(len(results), 4)
        self.assertFalse(results[0]["report"].success)   # 当前默认: 不可行
        self.assertTrue(results[1]["report"].success)    # 显式关闭摩擦: 可行
        self.assertTrue(results[2]["report"].success)    # 显式换约: 可行
        self.assertTrue(results[3]["report"].success)    # 派生修正锥: 可行
        # 派生修正不得改动 GIAC 符号约定 (仍字面 aB + CCW)
        self.assertEqual(
            results[3]["config"].inertial_accel_sign, "paper_literal"
        )
        self.assertEqual(results[3]["config"].resolved_winding(), "ccw")
        # 可行配置 (显式关闭摩擦) 的独立复算一致性
        from myrobot_mjlab.task.DTC.tamols.assumptions_config import (
            AssumptionsConfig,
        )
        from myrobot_mjlab.task.DTC.tamols.synthetic_scenarios import (
            run_scenario,
        )
        res = run_scenario(
            "static_flat", AssumptionsConfig(friction_mode="disabled"),
            verbose=False,
        )
        self.assertEqual(check_consistency(res.independent_checks), [])


# ---------------------------------------------------------------------- #
# 一、变量布局
# ---------------------------------------------------------------------- #
class TestLayout(unittest.TestCase):
    def test_layout_table_and_slices(self):
        layout = IntegratedNLPLayout(num_phases=3, use_slack=True)
        self.assertEqual(
            layout.block_shapes["spline_coeffs"], (3, 6, 5)
        )
        self.assertEqual(layout.block_shapes["desired_footholds"], (4, 3))
        self.assertEqual(layout.block_shapes["slack"], (3,))
        self.assertEqual(layout.slices["spline_coeffs"], slice(0, 90))
        self.assertEqual(layout.slices["desired_footholds"], slice(90, 102))
        self.assertEqual(layout.slices["slack"], slice(102, 105))
        self.assertEqual(layout.total_size, 105)
        text = layout.describe()
        self.assertIn("decision variables", text)
        self.assertIn("parameters (not decision variables)", text)
        self.assertIn("base_pose0", text)
        self.assertIn("measured_stance_feet", text)

    def test_pack_unpack_roundtrip(self):
        layout = IntegratedNLPLayout(num_phases=2, use_slack=True)
        coeffs = [np.arange(30.0).reshape(6, 5), -np.arange(30.0).reshape(6, 5)]
        desired = np.arange(12.0).reshape(4, 3)
        slack = np.array([-0.1, -0.2])
        x = layout.pack(coeffs, desired, slack)
        self.assertEqual(x.size, layout.total_size)
        out = layout.unpack(x)
        np.testing.assert_allclose(out["spline_coeffs"][0], coeffs[0])
        np.testing.assert_allclose(out["spline_coeffs"][1], coeffs[1])
        np.testing.assert_allclose(out["desired_footholds"], desired)
        np.testing.assert_allclose(out["slack"], slack)
        with self.assertRaises(ValueError):
            layout.unpack(np.zeros(layout.total_size + 1))
        with self.assertRaises(ValueError):
            layout.pack(coeffs, desired, None)

    def test_no_slack_layout(self):
        layout = IntegratedNLPLayout(num_phases=1, use_slack=False)
        self.assertNotIn("slack", layout.block_shapes)
        self.assertEqual(layout.total_size, 30 + 12)

    def test_coeff_flatten_order_matches_spline_value(self):
        """展平顺序必须与 spline_value 的 coeffs[dim, k] 索引一致."""
        layout = IntegratedNLPLayout(num_phases=1, use_slack=False)
        coeffs = np.zeros((6, 5))
        coeffs[2, 3] = 7.0  # z 维 a3
        x = layout.pack([coeffs], np.zeros((4, 3)))
        back = layout.unpack(x)["spline_coeffs"][0]
        self.assertEqual(back[2, 3], 7.0)
        # spline_value 读同一位置: z(tau) 含 7*tau^3
        val = spline_value(back, 0.5, order=0)
        self.assertAlmostEqual(val[2], 7.0 * 0.5**3, places=12)

    def test_state_vector_roundtrip(self):
        """最优输出 = 状态向量 x: pack 后 unpack 必须还原 report.solution."""
        nlp = build_simple_nlp()
        nlp.add_initial_state_constraints()
        nlp.add_dynamic_stability_constraints()
        nlp.add_kinematic_contact_constraints()
        nlp.add_objectives()
        nlp.finalize()
        coeffs = [np.zeros((6, 5))]
        coeffs[0][0:3, 0] = [0, 0, 0.30]
        nlp.set_initial_guess(coeffs, GO2_FEET_STANCE.copy())
        report = nlp.solve("state-vector")
        self.assertTrue(report.success, report.status)
        x = report.state_vector(nlp.layout)
        self.assertEqual(x.size, nlp.layout.total_size)
        out = nlp.layout.unpack(x)
        np.testing.assert_allclose(
            out["spline_coeffs"], report.solution["spline_coeffs"], atol=1e-12
        )
        np.testing.assert_allclose(
            out["desired_footholds"], report.solution["desired_footholds"],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            out["slack"], report.solution["slack"], atol=1e-12
        )
        # 布局不匹配的 layout 必须报错 (P 不同 -> total_size 不同)
        other = IntegratedNLPLayout(num_phases=2, use_slack=True)
        with self.assertRaises(ValueError):
            report.state_vector(other)

    def test_measured_feet_are_parameters_not_variables(self):
        nlp = build_simple_nlp()
        names = [name for name, _ in nlp.backend.variables]
        self.assertIn("desired_footholds", names)
        self.assertIn("spline_coeffs_0", names)
        self.assertNotIn("landing_footholds", names)
        self.assertNotIn("measured_feet", names)
        text = nlp.layout.describe()
        self.assertIn("measured_stance_feet", text)


# ---------------------------------------------------------------------- #
# 二、统一 L_dot
# ---------------------------------------------------------------------- #
class TestLDotUnification(unittest.TestCase):
    COEFFS = np.array(
        [
            [0.0, 0.1, 0.05, -0.02, 0.01],   # x
            [0.0, -0.05, 0.02, 0.01, 0.0],   # y
            [0.3, 0.0, -0.03, 0.02, 0.0],    # z
            [0.0, 0.05, -0.01, 0.0, 0.02],   # roll
            [0.0, -0.08, 0.03, 0.01, -0.01], # pitch
            [0.0, 0.12, 0.0, -0.02, 0.0],    # yaw
        ]
    )

    def test_euler_rate_to_omega_mapping_not_identity(self):
        euler = (0.3, -0.4, 0.7)
        rates = (0.5, -0.2, 0.9)
        omega = np.array(
            [float(v) for v in body_omega_tuple(euler, rates)]
        )
        ref = ref_omega(euler, rates)
        np.testing.assert_allclose(omega, ref, atol=1e-12)
        self.assertTrue(np.linalg.norm(omega - np.array(rates)) > 1e-3)

    @staticmethod
    def _cm_vector(coeffs):
        """ca.reshape(c, 6, 5) 的列优先布局: c[dim + 6k] = coeffs[dim, k]."""
        return np.asarray(coeffs, dtype=float).T.ravel()

    def test_analytic_vs_casadi_symbolic_time_vs_fd(self):
        import casadi as ca

        I = ca.DM(GO2_INERTIA_B)
        t = ca.MX.sym("t")
        c = ca.MX.sym("c", 30)
        coeffs = ca.reshape(c, 6, 5)
        L = trajectory_angular_momentum(coeffs, t, I)
        L_dot_sym = ca.jacobian(ca.vertcat(*L), t).T
        f = ca.Function("f", [c, t], [L_dot_sym])
        mine = trajectory_angular_momentum_rate(self.COEFFS, 0.17, GO2_INERTIA_B)
        mine = np.array([float(v) for v in mine])
        cas = np.array(f(self._cm_vector(self.COEFFS), 0.17)).ravel()
        fd = ref_L_dot_fd(self.COEFFS, GO2_INERTIA_B, 0.17)
        np.testing.assert_allclose(mine, cas, atol=1e-9)
        np.testing.assert_allclose(mine, fd, atol=1e-5)

    @unittest.skipUnless(CASADI_AVAILABLE, "CasADi 未安装")
    def test_jacobian_wrt_coeffs_vs_fd(self):
        import casadi as ca

        I = ca.DM(GO2_INERTIA_B)
        c = ca.MX.sym("c", 30)
        coeffs = ca.reshape(c, 6, 5)
        L_dot = trajectory_angular_momentum_rate(coeffs, 0.21, I)
        jac_expr = ca.jacobian(ca.vertcat(*L_dot), c)
        f = ca.Function("f", [c], [jac_expr])
        x0 = self._cm_vector(self.COEFFS)
        J = np.array(f(x0))  # 3x30, 列优先布局

        def ref_row(xv, row):
            cc = np.asarray(xv, dtype=float).reshape(6, 5, order="F")
            return ref_L_dot_fd(cc, GO2_INERTIA_B, 0.21)[row]

        for row in range(3):
            fd = num_grad(lambda xv: ref_row(xv, row), x0, eps=1e-6)
            np.testing.assert_allclose(J[row], fd, atol=1e-5, rtol=1e-4)

    def test_integration_rejects_independent_l_dot(self):
        nlp = build_simple_nlp()
        nlp.add_initial_state_constraints()
        with self.assertRaises(ValueError):
            nlp.add_dynamic_stability_constraints(
                l_dot_override=np.array([0.1, 0.2, 0.3])
            )

    def test_giac_uses_trajectory_l_dot_expression(self):
        """GIAC 残差表达式必须依赖样条变量 (即 L_dot/a_B 来自轨迹)."""
        import casadi as ca

        nlp = build_simple_nlp()
        nlp.add_initial_state_constraints()
        entries = [
            e for e in nlp.add_dynamic_stability_constraints()
            if e.category == "17b-17f"
        ]
        self.assertTrue(entries)
        coeffs_var = nlp.vars["spline_coeffs"][0]
        for e in entries:
            self.assertTrue(
                ca.depends_on(ca.MX(e.residual), coeffs_var),
                f"{e.name} 不依赖样条变量: L_dot 未与轨迹统一",
            )


def body_omega_tuple(euler, rates):
    from myrobot_mjlab.task.DTC.tamols.v_b_objectives import body_omega_zyx

    return body_omega_zyx(euler, rates)


# ---------------------------------------------------------------------- #
# 三、集成: 权重/单位/采样/分项
# ---------------------------------------------------------------------- #
class TestIntegrationBookkeeping(unittest.TestCase):
    def _nlp_full(self, weights=None):
        nlp = build_simple_nlp()
        nlp.add_initial_state_constraints()
        nlp.add_junction_constraints()
        nlp.add_dynamic_stability_constraints()
        nlp.add_kinematic_contact_constraints()
        nlp.objective_set = nlp.add_objectives()
        return nlp

    def test_weights_applied_once(self):
        nlp = self._nlp_full()
        nlp.finalize()
        coeffs = [np.zeros((6, 5))]
        coeffs[0][0:3, 0] = [0, 0, 0.30]
        nlp.set_initial_guess(coeffs, GO2_FEET_STANCE.copy())
        report = nlp.solve("weight-once")
        self.assertTrue(report.success, report.status)
        total = 0.0
        for row in report.objective_lines:
            # 加权贡献 = 权重 * 未加权值, 且只乘一次
            self.assertAlmostEqual(row.weighted, row.weight * row.unweighted, places=10)
            total += row.weighted
        self.assertAlmostEqual(report.objective_total, total, places=10)
        self.assertAlmostEqual(
            sum(r.fraction for r in report.objective_lines), 1.0, places=8
        )

    def test_weight_scaling_doubles_contributions(self):
        nlp1 = build_simple_nlp(weights=VBWeights())
        nlp2 = build_simple_nlp(
            weights=VBWeights(footholds_on_ground=2e4, edge_avoidance=6.0)
        )
        for nlp in (nlp1, nlp2):
            nlp.add_objectives()
        rec1 = {r.name: r for r in nlp1.objective_set.records}
        rec2 = {r.name: r for r in nlp2.objective_set.records}
        for name in rec1:
            if name.startswith("VB1"):
                self.assertAlmostEqual(
                    rec2[name].weight, 2 * rec1[name].weight, places=12
                )
            if name.startswith("VB5"):
                self.assertAlmostEqual(
                    rec2[name].weight, 2 * rec1[name].weight, places=12
                )
            if name.startswith("VB3"):
                self.assertAlmostEqual(
                    rec2[name].weight, rec1[name].weight, places=12
                )

    def test_vb2_distance_vs_squared_units(self):
        params = NLPParameters(
            base_pose0=[0, 0, 0.3, 0, 0, 0],
            base_twist0=np.zeros(6),
            measured_feet={0: GO2_FEET_STANCE[0]},
            min_foot_distance_m=0.2,
        )
        self.assertAlmostEqual(params.eps_min_m2, 0.04, places=12)
        nlp = build_simple_nlp()
        self.assertAlmostEqual(nlp.params.eps_min_m2, 0.01, places=12)
        # 0.2 m 间距 -> z = 0.04 = eps_min -> barrier 恰为 0
        from myrobot_mjlab.task.DTC.tamols.v_b_objectives import leg_collision_barrier

        self.assertAlmostEqual(
            leg_collision_barrier([0.0, 0.0, 0.0], [0.2, 0.0, 0.0], 0.04), 0.0,
            places=12,
        )
        # 0.1 m 间距 -> z = 0.01 < 0.04 -> barrier = (0.03)^2
        self.assertAlmostEqual(
            leg_collision_barrier([0.0, 0.0, 0.0], [0.1, 0.0, 0.0], 0.04),
            0.03**2, places=12,
        )

    def test_sampling_counts_and_Tk_weights(self):
        nlp = build_simple_nlp()
        nlp.add_objectives()
        per_phase = len(nlp.sample_times(0))
        self.assertEqual(per_phase, 6)
        align = [r for r in nlp.objective_set.records if r.category == "V-B4"]
        self.assertEqual(len(align), 6 * 4)
        for r in align:
            self.assertAlmostEqual(r.weight, 100.0 * 0.3 / 6, places=12)
        track = [r for r in nlp.objective_set.records if r.category == "V-B7"]
        self.assertEqual(len(track), 6)
        for r in track:
            self.assertAlmostEqual(r.weight, 2.0 * 0.3 / 6, places=12)
        slack = [r for r in nlp.objective_set.records if r.category == "29"]
        self.assertEqual(len(slack), 1)
        self.assertAlmostEqual(slack[0].weight, 7e-3, places=12)

    def test_report_objective_lines_have_fraction(self):
        nlp = self._nlp_full()
        nlp.finalize()
        coeffs = [np.zeros((6, 5))]
        coeffs[0][0:3, 0] = [0, 0, 0.30]
        nlp.set_initial_guess(coeffs, GO2_FEET_STANCE.copy())
        report = nlp.solve("bookkeeping")
        self.assertTrue(report.success)
        self.assertEqual(
            len(report.objective_lines), len(nlp.objective_set.records)
        )
        frac_sum = sum(r.fraction for r in report.objective_lines)
        self.assertAlmostEqual(frac_sum, 1.0, places=8)
        for row in report.objective_lines:
            self.assertAlmostEqual(
                row.weighted, row.weight * row.unweighted, places=10
            )
        self.assertIn("solution_source=solver_solution", report.format())


# ---------------------------------------------------------------------- #
# 四、假设与静止基准
# ---------------------------------------------------------------------- #
class TestAssumptions(unittest.TestCase):
    def test_static_baseline_literal_infeasible(self):
        r = run_scenario("static_flat", LITERAL, verbose=False)
        self.assertFalse(r.report.success)
        self.assertIn("Infeasible", r.report.status)
        # 静止基准: 在零加速度 (基座被场景等式钉住) 处直接求字面摩擦锥
        # 法向残差, 应等于 +9.81 (a_B = g -> -a_B_z = +g), 与求解过程无关.
        import casadi as ca

        nlp, _, _ = build_scenario("static_flat", LITERAL)
        nlp.add_initial_state_constraints()
        entries = [
            e for e in nlp.add_dynamic_stability_constraints()
            if e.category == "5" and "normal" in e.name
        ]
        self.assertTrue(entries)
        cv = nlp.vars["spline_coeffs"][0]
        f = ca.Function("f", [cv], [ca.vertcat(*[e.residual for e in entries])])
        static_coeffs = np.zeros((6, 5))
        static_coeffs[0:3, 0] = [0, 0, 0.30]
        vals = np.array(f(static_coeffs)).ravel()
        np.testing.assert_allclose(vals, 9.81, atol=1e-9)

    def test_static_baseline_contact_feasible(self):
        r = run_scenario("static_flat", CONTACT_NORMAL, verbose=False)
        self.assertTrue(r.report.success)
        self.assertLess(
            max(r.report.max_violation_by_category.values()), 1e-6
        )
        self.assertEqual(check_consistency(r.independent_checks), [])

    def test_convexity_p20_vs_p21_translation_rotation(self):
        feet = [
            (0.19, 0.145, 0.0),
            (-0.19, 0.145, 0.0),
            (-0.19, -0.145, 0.0),
            (0.19, -0.145, 0.0),
        ]  # CCW: FL,RL,RR,FR
        nlp_eng = build_simple_nlp(
            assumptions=AssumptionsConfig(convexity_fourth_term="p21_engineering")
        )
        nlp_lit = build_simple_nlp(
            assumptions=AssumptionsConfig(convexity_fourth_term="p20_literal")
        )
        res_eng = nlp_eng._convexity_residuals(feet, "p21_engineering")
        res_lit = nlp_lit._convexity_residuals(feet, "p20_literal")
        self.assertNotAlmostEqual(res_eng[3], res_lit[3], places=6)
        # 平移: p21 不变, p20 改变
        moved = [(x + 1.7, y - 0.9, z) for x, y, z in feet]
        self.assertAlmostEqual(
            nlp_eng._convexity_residuals(moved, "p21_engineering")[3],
            res_eng[3], places=9,
        )
        self.assertNotAlmostEqual(
            nlp_lit._convexity_residuals(moved, "p20_literal")[3],
            res_lit[3], places=6,
        )
        # 绕世界原点旋转: 两版的第四项都只由向量叉积 z 分量构成, 均不变;
        # 区别只在平移 (p20 依赖原点, p21 不依赖).
        th = 0.7
        rot = [
            (x * math.cos(th) - y * math.sin(th),
             x * math.sin(th) + y * math.cos(th), z)
            for x, y, z in feet
        ]
        self.assertAlmostEqual(
            nlp_eng._convexity_residuals(rot, "p21_engineering")[3],
            res_eng[3], places=9,
        )
        self.assertAlmostEqual(
            nlp_lit._convexity_residuals(rot, "p20_literal")[3],
            res_lit[3], places=9,
        )
        # 前三项与版本无关
        self.assertEqual(res_eng[:3], res_lit[:3])

    def test_constant_constraint_assembly_infeasible(self):
        """测量脚排成顺时针 -> 凸性常数残差违反 -> 装配期判不可行, 不求解."""
        nlp = build_simple_nlp()
        cw_feet = [tuple(GO2_FEET_STANCE[i]) for i in (0, 1, 3, 2)]
        res = nlp._convexity_residuals(cw_feet, "p21_engineering")
        self.assertGreater(max(float(r) for r in res), 0.0)
        # 直接走 _add_entry 的常数分支
        from myrobot_mjlab.task.DTC.tamols.giac_constraints import INEQUALITY
        from myrobot_mjlab.task.DTC.tamols.nlp_integration import ConstraintEntry

        for i, r in enumerate(res):
            nlp._add_entry(
                ConstraintEntry(
                    name=f"const30_{i}", category="30", kind=INEQUALITY,
                    residual=float(r), source="test",
                )
            )
        self.assertTrue(nlp.assembly_infeasible)
        nlp.add_initial_state_constraints()
        nlp.add_objectives()
        nlp.finalize()
        nlp.set_initial_guess([np.zeros((6, 5))], GO2_FEET_STANCE.copy())
        report = nlp.solve("const-infeasible")
        self.assertEqual(report.status, "Infeasible_At_Assembly")
        self.assertFalse(report.success)
        self.assertEqual(report.iterations, 0)

    def test_assumptions_describe_in_report(self):
        r = run_scenario("static_flat", CONTACT_NORMAL, verbose=False)
        self.assertIn("AssumptionsConfig", r.report.assumptions_text)
        self.assertIn("inertial_accel_sign", r.report.assumptions_text)
        self.assertIn("convexity_fourth_term", r.report.assumptions_text)


# ---------------------------------------------------------------------- #
# 五、合成场景
# ---------------------------------------------------------------------- #
class TestScenarios(unittest.TestCase):
    def test_s2_crawl_flat(self):
        r = run_scenario("crawl_flat", CONTACT_NORMAL, verbose=False)
        self.assertTrue(r.report.success, r.report.status)
        self.assertLess(max(r.report.max_violation_by_category.values()), 1e-6)
        self.assertEqual(check_consistency(r.independent_checks), [])

    def test_s3_crawl_step(self):
        r = run_scenario("crawl_step", CONTACT_NORMAL, verbose=False)
        self.assertTrue(r.report.success, r.report.status)
        self.assertLess(max(r.report.max_violation_by_category.values()), 1e-6)
        self.assertEqual(check_consistency(r.independent_checks), [])
        sol = r.report.solution["desired_footholds"]
        # 前腿登上 +0.15 m 平台
        self.assertAlmostEqual(sol[0, 2], 0.15, delta=0.02)

    def test_s4_two_stones_choice(self):
        r = run_scenario("two_stones", CONTACT_NORMAL, verbose=False)
        self.assertTrue(r.report.success, r.report.status)
        self.assertLess(max(r.report.max_violation_by_category.values()), 1e-6)
        self.assertEqual(check_consistency(r.independent_checks), [])
        sol = r.report.solution["desired_footholds"]
        # 初值在石台 A: 局部选择应停留在 A (中心 0.30, 0.28, r=0.10)
        dA = math.hypot(sol[0, 0] - 0.30, sol[0, 1] - 0.28)
        dB = math.hypot(sol[0, 0] - 0.30, sol[0, 1] - 0.06)
        self.assertLess(dA, 0.10, f"FL 应停留在石台 A, 实际距 A {dA:.3f} 距 B {dB:.3f}")
        self.assertAlmostEqual(sol[0, 2], 0.0, delta=0.02)

    def test_s5_infeasible_distinguished(self):
        r = run_scenario("infeasible_reach", CONTACT_NORMAL, verbose=False)
        self.assertFalse(r.report.success)
        self.assertIn("Infeasible", r.report.status)
        # 解析论证 (不依赖求解器): tau=0 处基座位姿被初始等式钉在
        # (0,0,0.30,0,0,0), 支撑脚为测量参数 z=0, 髋偏移 r_z=0, R=I,
        # 故 d = ||hip - foot|| >= 0.30 与任何决策变量取值无关; 而
        # l_max = 0.06 => 31_upper 残差 = d^2 - l_max^2 >= 0.0864 > 0,
        # 该等式+参数组合使可行集为空 => 问题确实无解.
        from myrobot_mjlab.task.DTC.tamols.synthetic_scenarios import ref_reach

        _, upper = ref_reach(
            np.array([0.0, 0.0, 0.30]), np.eye(3),
            np.array([0.1934, 0.0465, 0.0]),  # 髋关节偏移 (r_z=0)
            np.array([0.1934, 0.145, 0.0]), 0.02, 0.06,
        )
        self.assertGreater(upper, 0.05)
        # 报告中的 31_upper 条目存在 (最后迭代点数值仅作诊断, 不作证明)
        self.assertTrue(
            [c for c in r.report.constraint_lines if c.category == "31_upper"]
        )


# ---------------------------------------------------------------------- #
# 六、数值与健壮性
# ---------------------------------------------------------------------- #
@unittest.skipUnless(CASADI_AVAILABLE, "CasADi 未安装")
class TestNumerics(unittest.TestCase):
    def test_total_objective_fd_gradient_foothold_tasks(self):
        import casadi as ca

        from myrobot_mjlab.task.DTC.tamols.v_b_objectives import (
            edge_avoidance_records,
            foothold_ground_records,
            leg_collision_records,
            merge_objective_sets,
            previous_solution_records,
        )

        maps = flat_maps()
        backend = solver_backend.make_backend("casadi")
        x = ca.MX.sym("x", 12)
        p = ca.reshape(x, 4, 3)
        prev = GO2_FEET_STANCE * 0.95
        sets = merge_objective_sets(
            foothold_ground_records(maps, p),
            leg_collision_records(p),
            edge_avoidance_records(maps, p),
            previous_solution_records(p, prev),
        )
        total = assemble_total_objective(sets)
        f = ca.Function("f", [x], [total])
        # 避开格线 (dx=0.05, origin=(-1,-1)): 选非 0.05 整数倍坐标.
        # 双线性插值的梯度在格线上不连续 (CasADi 取右格单侧导数, 中心
        # 差分平均两侧), 故评估点必须严格在格内.
        pts = GO2_FEET_STANCE + np.array([0.013, -0.021, 0.007])
        x0 = pts.T.ravel()
        J = np.array(ca.Function("g", [x], [ca.jacobian(total, x)])(x0)).ravel()

        h_grid = maps.h.data
        gx = maps.grad_h_x.data
        gy = maps.grad_h_y.data
        g1x = maps.grad_hs1_x.data
        g1y = maps.grad_hs1_y.data
        dx, dy = maps.h.dx, maps.h.dy
        w = VBWeights()

        def bilinear(grid, fx, fy):
            from myrobot_mjlab.task.DTC.tamols.test_v_b_objectives import (
                bilinear_manual,
            )
            return bilinear_manual(grid, fx, fy)

        def ref_total(xv):
            q = xv.reshape(3, 4).T
            cost = 0.0
            for i in range(4):
                fx = (q[i, 0] - maps.h.origin[0]) / dx
                fy = (q[i, 1] - maps.h.origin[1]) / dy
                hv = bilinear(h_grid, fx, fy)
                cost += w.footholds_on_ground * (hv - q[i, 2]) ** 2
                gh = (bilinear(gx, fx, fy), bilinear(gy, fx, fy))
                gh1 = (bilinear(g1x, fx, fy), bilinear(g1y, fx, fy))
                cost += w.edge_avoidance * (
                    gh[0] ** 2 + gh[1] ** 2 + gh1[0] ** 2 + gh1[1] ** 2
                )
                cost += w.previous_solution * float(np.sum((q[i] - prev[i]) ** 2))
            for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
                z = (q[i, 0] - q[j, 0]) ** 2 + (q[i, 1] - q[j, 1]) ** 2
                cost += w.leg_collision * (max(0.01 - z, 0.0) ** 2)
            return cost

        fd = num_grad(ref_total, x0, eps=1e-6)
        np.testing.assert_allclose(J, fd, rtol=1e-4, atol=1e-6)

    def test_giac_residual_fd_gradient(self):
        import casadi as ca

        nlp = build_simple_nlp()
        nlp.add_initial_state_constraints()
        entries = [
            e for e in nlp.add_dynamic_stability_constraints()
            if e.category == "17b-17f"
        ]
        self.assertTrue(entries)
        target = entries[0]
        coeffs_var = nlp.vars["spline_coeffs"][0]
        jac = ca.jacobian(ca.MX(target.residual), coeffs_var)  # 1x30
        f = ca.Function("f", [coeffs_var], [jac])
        x0 = np.zeros(30)
        x0[0:3] = [0.0, 0.0, 0.30]
        J = np.array(f(x0.reshape(6, 5))).ravel()

        def ref_res(xv):
            cc = np.asarray(xv, dtype=float).reshape(6, 5)
            pose = np.array(
                [float(v) for v in spline_value(cc, 0.0, order=0)]
            )
            accel = np.array(
                [float(v) for v in spline_value(cc, 0.0, order=2)]
            )
            rate = np.array(
                [float(v) for v in spline_value(cc, 0.0, order=1)]
            )
            if nlp.assumptions.inertial_accel_sign == "paper_literal":
                a_B = np.array([0, 0, -9.81]) - accel[0:3]
            else:
                a_B = accel[0:3] - np.array([0, 0, -9.81])
            R = ref_rot_zyx(pose[3], pose[4], pose[5])
            omega = ref_omega(pose[3:6], rate[3:6])
            L = R @ np.asarray(GO2_INERTIA_B) @ omega
            L_dot = ref_L_dot_fd(cc, GO2_INERTIA_B, 0.0)
            # target 是第一相位第一个采样点 (tau=0) 的第一对循环边脚
            ordered, feet = nlp._ordered_stance(0, nlp.assumptions.resolved_winding())
            self.assertTrue(all(isinstance(v, float) for v in feet[0] + feet[1]))
            p_i = np.array(feet[0], dtype=float)
            p_j = np.array(feet[1], dtype=float)
            p_ij = p_j - p_i
            res = nlp.params.mass * float(
                np.dot(p_ij, np.cross(pose[0:3] - p_i, a_B))
            ) - float(np.dot(p_ij, L_dot))
            if nlp.layout.use_slack:
                res -= 0.0  # slack 初值 0, 且不在 x 中
            return res

        fd = num_grad(ref_res, x0, eps=1e-6)
        # J 是 ca.jacobian 对矩阵变量的列优先展平; fd 用行优先, 需转置对比
        G_cas = J.reshape(6, 5, order="F")
        G_fd = fd.reshape(6, 5)
        np.testing.assert_allclose(G_cas, G_fd, rtol=1e-4, atol=1e-5)

    def test_map_coverage_warning_when_solution_outside(self):
        """小地图: 名义落脚点在图外, 解必然出界 -> 报告必须带覆盖警告."""
        from myrobot_mjlab.task.DTC.tamols.map_query import TerrainMaps

        small = TerrainMaps(
            np.zeros((9, 9)), np.zeros((9, 9)), np.zeros((9, 9)),
            dx=0.05, dy=0.05, origin=(-0.6, -0.6),
        )
        schedules = {
            0: LegSchedule(False, touchdown_phase=0),
            1: LegSchedule(True, measured_position=tuple(GO2_FEET_STANCE[1])),
            2: LegSchedule(True, measured_position=tuple(GO2_FEET_STANCE[2])),
            3: LegSchedule(True, measured_position=tuple(GO2_FEET_STANCE[3])),
        }
        params = NLPParameters(
            base_pose0=[0, 0, 0.30, 0, 0, 0],
            base_twist0=np.zeros(6),
            measured_feet={leg: GO2_FEET_STANCE[leg] for leg in (1, 2, 3)},
            ref_twist=lambda p, t: (np.zeros(3), np.zeros(3)),
            solver_options={"print_time": False,
                            "ipopt": {"print_level": 0, "max_iter": 200}},
        )
        nlp = IntegratedNLP(
            durations=[0.3], schedules=schedules, maps=small,
            params=params, assumptions=CONTACT_NORMAL,
        )
        nlp.add_initial_state_constraints()
        nlp.add_dynamic_stability_constraints()
        nlp.add_kinematic_contact_constraints()
        nlp.add_objectives()
        nlp.finalize()
        coeffs = [np.zeros((6, 5))]
        coeffs[0][0:3, 0] = [0, 0, 0.30]
        guess = GO2_FEET_STANCE.copy()
        nlp.set_initial_guess(coeffs, guess)
        report = nlp.solve("coverage")
        self.assertTrue(report.success, report.status)
        self.assertTrue(
            any("超出地图范围" in w or "有效性" in w for w in report.warnings),
            f"期望地图覆盖警告, 实际 {report.warnings}",
        )
        for arr in report.solution.values():
            self.assertTrue(np.isfinite(arr).all())

    def test_nan_initial_rejected_before_solver(self):
        """NaN 初值在 set_initial 阶段就被后端拒绝, 不会静默进入求解器."""
        nlp = build_simple_nlp()
        nlp.add_initial_state_constraints()
        nlp.add_dynamic_stability_constraints()
        nlp.add_objectives()
        nlp.finalize()
        with self.assertRaises(Exception):
            nlp.set_initial_guess([np.full((6, 5), np.nan)], GO2_FEET_STANCE.copy())

    def test_finite_check_helper(self):
        nlp = build_simple_nlp()
        bad = nlp.check_finite({"a": np.array([1.0, np.nan]), "b": np.ones(2)})
        self.assertEqual(len(bad), 1)
        self.assertIn("a", bad[0])
        self.assertEqual(nlp.check_finite({"a": np.ones(2)}), [])

    def test_warm_start_matches_cold(self):
        cold = run_scenario("crawl_flat", CONTACT_NORMAL, verbose=False)
        self.assertTrue(cold.report.success)
        nlp, guess, _ = build_scenario("crawl_flat", CONTACT_NORMAL)
        nlp.add_initial_state_constraints()
        nlp.add_junction_constraints()
        nlp.add_dynamic_stability_constraints()
        nlp.add_kinematic_contact_constraints()
        nlp.add_objectives()
        nlp.finalize()
        nlp.set_warm_start(cold.report)
        warm = nlp.solve("crawl_flat_warm")
        self.assertTrue(warm.success, warm.status)
        np.testing.assert_allclose(
            warm.solution["desired_footholds"],
            cold.report.solution["desired_footholds"],
            atol=1e-4,
        )
        self.assertLessEqual(warm.iterations, cold.report.iterations + 3)

    def test_solver_failure_handled_gracefully(self):
        """max_iter=1 -> 求解"失败"状态, solve() 不抛异常且报告完整."""
        nlp = build_simple_nlp(
            solver_options={"print_time": False,
                            "ipopt": {"print_level": 0, "max_iter": 1}}
        )
        nlp.add_initial_state_constraints()
        nlp.add_dynamic_stability_constraints()
        nlp.add_objectives()
        nlp.finalize()
        coeffs = [np.zeros((6, 5))]
        coeffs[0][0:3, 0] = [0, 0, 0.30]
        nlp.set_initial_guess(coeffs, GO2_FEET_STANCE.copy())
        report = nlp.solve("max-iter-1")
        self.assertIsInstance(report.status, str)
        self.assertFalse(report.success)
        self.assertEqual(report.solution_source, "last_iterate_not_a_solution")
        self.assertIn("iterations", report.format())


if __name__ == "__main__":
    unittest.main()
