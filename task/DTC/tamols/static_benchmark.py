"""最简单基准: 平地 / 静止机身 / 静态支撑 (四足全站立, P=1).

运行::

    cd /home/hong/myrobot_mjlab/src && \
    /home/hong/myrobot_mjlab/.venv/bin/python -m \
        myrobot_mjlab.task.DTC.tamols.static_benchmark

目的与纪律
----------
摩擦锥 (式 5 / 17a) 的符号约定**仍待核实** (见
:mod:`assumptions_config` 文档): 按论文字面 ``a_B = g - p_B_ddot``, 静止
站立时 ``a_B = [0,0,-9.81]``, 字面摩擦锥要求 ``a_B_z >= 0`` 而被违反.
本基准的任务是**检查并报告**当前默认配置下该场景是否可行, 而不是让
它通过: 不翻转任何符号; 需要其他约定时只能显式配置
(``friction_mode="disabled"`` 或 ``inertial_accel_sign="contact_normal_explicit"``),
且这两种路径在报告中标记为 UNVERIFIED.

报告内容 (每种配置)
-------------------
1. IPOPT 状态 / 迭代数 / 目标值 / 各类约束最大违反;
2. 静止点 (零加速度, 基座被场景等式钉住) 处各动力学残差的**数值证据**:
   - 摩擦锥平方残差与法向残差 (类别 "5");
   - GIAC (17b) 循环边残差 (类别 "17b-17f");
   - 可达 (31) 与凸性 (30) 在静止点的值;
3. 结论行: 当前默认配置是否可行; 若不可行, 指出唯一违反来源.
"""

from __future__ import annotations

import numpy as np

from myrobot_mjlab.task.DTC.tamols.assumptions_config import (
    DEFAULT_ASSUMPTIONS,
    AssumptionsConfig,
)
from myrobot_mjlab.task.DTC.tamols.nlp_integration import IntegratedNLP
from myrobot_mjlab.task.DTC.tamols.synthetic_scenarios import (
    GO2_FEET_STANCE,
    build_scenario,
    check_consistency,
)

STATIC_COEFFS = np.zeros((6, 5))
STATIC_COEFFS[0:3, 0] = [0.0, 0.0, 0.30]


def rest_residual_evidence(nlp: IntegratedNLP) -> dict[str, dict[str, float]]:
    """在静止点 (样条全导数为 0) 数值求各动力学约束残差, 与求解无关."""
    import casadi as ca

    cv = nlp.vars["spline_coeffs"][0]
    inputs = [cv]
    zero_extra: list[np.ndarray] = [STATIC_COEFFS]
    if nlp.layout.use_slack:
        # GIAC 残差含 slack 项; 静止证据点取 slack = 0
        inputs.append(nlp.vars["slack"])
        zero_extra.append(np.zeros(nlp.layout.num_phases))
    out: dict[str, dict[str, float]] = {}
    groups: dict[str, list] = {}
    for e in nlp.constraint_entries:
        if e.phase == 0 and e.category in ("5", "17b-17f", "31_lower", "31_upper", "30"):
            groups.setdefault(e.category, []).append(e)
    for cat, entries in groups.items():
        values = {}
        for e in entries:
            res = e.residual
            if isinstance(res, (int, float, np.integer, np.floating)):
                values[e.name] = float(res)
            else:
                f = ca.Function("f", inputs, [ca.MX(res)])
                values[e.name] = float(f(*zero_extra))
        out[cat] = values
    return out


def run_static_benchmark(verbose: bool = True) -> list[dict]:
    configs = [
        ("CURRENT DEFAULT (paper_literal, friction enabled)", DEFAULT_ASSUMPTIONS),
        ("paper_literal, friction DISABLED (未验证, 不可用于安全结论)",
         AssumptionsConfig(friction_mode="disabled")),
        ("contact_normal_explicit (未验证, 不可用于安全结论)",
         AssumptionsConfig(inertial_accel_sign="contact_normal_explicit")),
        ("force_balance_derived cone, GIAC literal+CCW "
         "(派生修正, 待作者勘误确认, 不可用于安全结论)",
         AssumptionsConfig(friction_cone_form="force_balance_derived")),
    ]
    results = []
    for label, cfg in configs:
        nlp, guess, note = build_scenario("static_flat", cfg)
        nlp.add_constant_base_pin(0)  # "静止机身" 的严格表达
        nlp.add_initial_state_constraints()
        nlp.add_junction_constraints()
        nlp.add_dynamic_stability_constraints()
        nlp.add_kinematic_contact_constraints()
        nlp.add_objectives()
        nlp.finalize()
        coeffs, desired = guess
        nlp.set_initial_guess(coeffs, desired)
        report = nlp.solve(f"static_flat[{cfg.inertial_accel_sign},"
                           f"friction={cfg.friction_mode}]")
        evidence = rest_residual_evidence(nlp)
        results.append(
            {"label": label, "config": cfg, "report": report,
             "evidence": evidence}
        )
        if verbose:
            print(f"=== {label} ===")
            print(f"status={report.status.splitlines()[0]} "
                  f"success={report.success} iterations={report.iterations} "
                  f"objective={report.objective_total:.6g}")
            print("max violation by category:",
                  {k: f"{v:.3g}" for k, v in report.max_violation_by_category.items()})
            for cat, values in evidence.items():
                worst = max(values.values(), key=abs) if values else 0.0
                print(f"  rest residuals [{cat}]: n={len(values)} "
                      f"max|value|={abs(worst):.4g} "
                      f"max violation={max(0.0, max(values.values())):.4g}")
            fric_normal = [
                v for k, v in evidence.get("5", {}).items() if "normal" in k
            ]
            giac = list(evidence.get("17b-17f", {}).values())
            verdict = "FEASIBLE" if report.success else "INFEASIBLE"
            src = ""
            if not report.success and fric_normal and max(fric_normal) > 0:
                src = (f" source: literal friction normal residual at rest = "
                       f"{max(fric_normal):+.4f} > 0 (a_B = g at rest); "
                       f"GIAC rest residuals all <= 0: "
                       f"{all(v <= 1e-9 for v in giac)}")
            print(f"  VERDICT: {verdict}{src}")
            print(f"  winding={cfg.resolved_winding()} "
                  f"include_push={cfg.include_push} (sign NOT flipped by code)")
            print()
    return results


def main() -> int:
    results = run_static_benchmark(verbose=True)
    default = results[0]
    rep = default["report"]
    fric_normal = [
        v for k, v in default["evidence"].get("5", {}).items() if "normal" in k
    ]
    print("=== summary ===")
    for r in results:
        print(f"{r['label']:<52s} success={r['report'].success!s:<5} "
              f"status={r['report'].status.splitlines()[0][:40]}")
    print()
    print("结论 (仅陈述所测结果, 不声称全局最优/实机可用):")
    print(f"  当前默认配置下静止基准: {'可行' if rep.success else '不可行'}.")
    if not rep.success and fric_normal and max(fric_normal) > 0:
        ev = default["evidence"]
        others_ok = all(
            max(vals.values()) <= 1e-9
            for cat, vals in ev.items()
            if cat != "5"
        )
        cone_ok = all(
            v <= 1e-9
            for k, v in ev.get("5", {}).items()
            if "cone" in k
        )
        if others_ok and cone_ok:
            print("  静止基准按当前字面公式不可行, 摩擦法向是唯一违反项.")
        print(f"  不可行来源: 字面摩擦锥法向残差在静止点 = {max(fric_normal):+.4f} > 0.")
        print("  该符号约定 UNRESOLVED; 本基准未翻转符号. 显式诊断配置: "
              "friction_mode='disabled' / inertial_accel_sign="
              "'contact_normal_explicit' (未验证, 不可用于安全结论); "
              "friction_cone_form='force_balance_derived' (论文内部证据派生, "
              "待作者勘误确认, 不可用于安全结论).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
