"""V-A 约束的最小数值验证脚本 (CasADi + IPOPT).

对每组约束单独构造一个最小 NLP, 打印 IPOPT 状态、迭代数、目标值、
求解后的各约束最大违反量 (按类别) 和解向量, 作为测试之外的
"可运行证据". 这不是完整 TAMOLS 问题, 也不是实机安全性证明.

运行::

    python -m myrobot_mjlab.task.DTC.tamols.v_a_verification
"""

from __future__ import annotations

import numpy as np

from myrobot_mjlab.task.DTC.tamols import solver_backend
from myrobot_mjlab.task.DTC.tamols.solver_backend import CasadiBackend
from myrobot_mjlab.task.DTC.tamols.v_a_constraints import (
    DEFAULT_HIP_OFFSETS,
    GRAVITY,
    add_convexity_constraints,
    add_friction_push_constraints,
    add_reachability_constraints,
    add_spline_phase_friction_constraints,
    add_spline_phase_reachability_constraints,
    complete_footprint,
    convexity_records_from_events,
    FootholdEvent,
    friction_push_records,
    spline_base_acceleration,
)

MU = 0.6


def _backend() -> CasadiBackend:
    return CasadiBackend(
        solver_options={"ipopt": {"print_level": 0, "max_iter": 300}}
    )


def _report(title, backend, result, extra=""):
    stats = backend.opti.stats()
    history = stats.get("iterations", {})
    print(f"\n=== {title} ===")
    print(f"status        : {stats.get('return_status')}")
    print(f"iterations    : {stats.get('iter_count')}")
    if history:
        print(f"objective     : {history['obj'][-1]}")
        print(
            f"inf_pr/inf_du : {history['inf_pr'][-1]} / "
            f"{history['inf_du'][-1]}"
        )
    try:
        violations = result.max_violation_by_category(
            evaluate=backend.value
        )
    except RuntimeError as exc:  # 求解失败时仍尝试用 debug 值
        print(f"violation eval failed: {exc}")
    else:
        text = ", ".join(
            f"{key}={value:.3e}" for key, value in violations.items()
        )
        print(f"max violation : {text}")
    if extra:
        print(extra)


def scenario_convexity():
    ca = solver_backend.ca
    backend = _backend()
    feet = backend.variable((4, 3), name="feet")
    target = np.array(
        [
            [0.1934, 0.145, -0.28],
            [0.1934, -0.145, -0.28],
            [0.05, 0.0, -0.28],   # RL 被移到三角形内部 -> 凹
            [-0.1934, -0.145, -0.28],
        ]
    )
    backend.opti.minimize(ca.sumsqr(feet - target))
    for index in range(4):
        backend.opti.subject_to(feet[index, 2] == -0.28)
    backend.set_initial(feet, target)
    result = add_convexity_constraints(backend, feet, phase=0)
    backend.solve()
    solution = np.asarray(backend.value(feet)).reshape(4, 3)
    _report(
        "Eq (30) convexity, full stance",
        backend,
        result,
        "solution:\n"
        + np.array2string(solution, precision=5)
        + f"\nRL displacement from concave target: "
        f"{np.linalg.norm(solution[2] - target[2]):.4f} m",
    )
    return solution


def scenario_completion():
    stance = {
        "FL": (0.2, 0.15, 0.0),
        "RR": (-0.2, -0.15, 0.0),
    }
    events = (
        FootholdEvent("FR", (0.2, -0.15, 0.0), phase=1),
        FootholdEvent("RL", (-0.2, 0.15, 0.0), phase=2),
    )
    completed = complete_footprint(stance, events)
    records = convexity_records_from_events(stance, events)
    print("\n=== Eq (30) incomplete stance -> future events ===")
    print(f"sources   : {completed.sources}")
    print(f"phases    : {completed.phases}")
    print(
        "residuals : "
        + ", ".join(
            f"{name}={value:.3e}"
            for name, value in records.residual_values().items()
        )
    )
    print(f"max violation: {records.max_violation():.3e}")
    print(f"note      : {records.note}")


def scenario_static_state():
    """静止: p_ddot = 0, L_dot = 0 (L_dot 不进入式 (5)/V-A-4)."""
    p_ddot = (0.0, 0.0, 0.0)
    literal_a_B = (0.0, 0.0, GRAVITY[2])
    engineering_a_B = (0.0, 0.0, -GRAVITY[2])

    print("\n=== Static state: p_ddot = 0, L_dot = 0 ===")
    for label, a_B in (
        ("paper  a_B = g - p_ddot", literal_a_B),
        ("engineering a_B = p_ddot - g", engineering_a_B),
    ):
        print(f"{label}: a_B = {a_B}")
        for form in ("norm", "squared"):
            friction = friction_push_records(
                a_B, p_ddot, MU, form=form, include_push=False
            )
            residual_text = ", ".join(
                f"{name}={value:.3e}"
                for name, value in friction.residual_values().items()
            )
            print(
                f"  {form:7s} friction: {residual_text} "
                f"-> feasible={friction.is_satisfied()}"
            )
        combined = friction_push_records(
            a_B, p_ddot, MU, include_push=True
        )
        push_value = combined.residual_values()["V-A-4_push"]
        print(f"  push residual g_z - p_ddot_z = {push_value:.3e} "
              f"(satisfied={push_value <= 0.0})")


def scenario_friction_inside():
    ca = solver_backend.ca
    backend = _backend()
    a = backend.variable((3,), name="a_B")
    target = np.array([0.5, 0.2, 1.0])
    p_ddot = tuple(GRAVITY[i] - a[i] for i in range(3))
    backend.opti.minimize(ca.sumsqr(a - target))
    backend.set_initial(a, [0.0, 0.0, 1.0])
    result = add_friction_push_constraints(
        backend, (a[0], a[1], a[2]), p_ddot, MU,
        form="squared", include_push=False,
    )
    backend.solve()
    solution = np.asarray(backend.value(a)).reshape(-1)
    _report(
        "Eq (5) friction, target inside cone",
        backend,
        result,
        "solution a_B: "
        + np.array2string(solution, precision=6)
        + f"\ncone slack (||.|| - mu*az): "
        f"{np.linalg.norm(solution[:2]) - MU * solution[2]:.6f}",
    )


def scenario_friction_outside():
    ca = solver_backend.ca
    backend = _backend()
    a = backend.variable((3,), name="a_B")
    target = np.array([0.9, 0.0, 1.0])
    p_ddot = tuple(GRAVITY[i] - a[i] for i in range(3))
    backend.opti.minimize(ca.sumsqr(a - target))
    backend.set_initial(a, [0.0, 0.0, 1.0])
    result = add_friction_push_constraints(
        backend, (a[0], a[1], a[2]), p_ddot, MU,
        form="squared", include_push=False,
    )
    backend.solve()
    solution = np.asarray(backend.value(a)).reshape(-1)
    _report(
        "Eq (5) friction, target outside cone -> boundary",
        backend,
        result,
        "solution a_B: "
        + np.array2string(solution, precision=6)
        + f"\ncone residual: "
        f"{solution[0] ** 2 + solution[1] ** 2 - (MU * solution[2]) ** 2:.3e}",
    )


def scenario_push():
    ca = solver_backend.ca
    backend = _backend()
    pz = backend.variable((1,), name="p_ddot_z")
    a = backend.variable((3,), name="a_B")
    for dim in range(3):
        backend.opti.subject_to(a[dim] == 0.0)
    backend.opti.minimize((pz - (-12.0)) ** 2)
    backend.set_initial(pz, 0.0)
    result = add_friction_push_constraints(
        backend,
        (a[0], a[1], a[2]),
        (0.0, 0.0, pz),
        MU,
        include_push=True,
    )
    backend.solve()
    _report(
        "V-A-4 push condition",
        backend,
        result,
        "solution p_ddot_z: "
        + np.array2string(
            np.asarray(backend.value(pz)).reshape(-1), precision=6
        ),
    )


def scenario_literal_contradiction():
    ca = solver_backend.ca
    backend = _backend()
    a = backend.variable((3,), name="a_B")
    p_ddot = backend.variable((3,), name="p_ddot")
    for dim in range(3):
        backend.opti.subject_to(a[dim] + p_ddot[dim] == GRAVITY[dim])
    target = np.array([0.05, 0.0, 1.0])
    backend.opti.minimize(ca.sumsqr(a - target))
    backend.set_initial(a, [0.0, 0.0, 1.0])
    backend.set_initial(p_ddot, [0.0, 0.0, -10.81])
    # 仅用于诊断: 字面摩擦锥与字面推地条件同时加入.
    result = add_friction_push_constraints(
        backend, (a[0], a[1], a[2]),
        (p_ddot[0], p_ddot[1], p_ddot[2]), MU,
        include_push=True,
    )
    backend.solve()
    solution = np.asarray(backend.value(a)).reshape(-1)
    _report(
        "Eq (5) + V-A-4 literal signs (a_B = g - p_ddot)",
        backend,
        result,
        "solution a_B: "
        + np.array2string(solution, precision=6)
        + f"\ndistance to target: {np.linalg.norm(solution - target):.4f}",
    )


def scenario_reachability():
    ca = solver_backend.ca
    p_B = np.array([0.0, 0.0, 0.3])
    R_B = np.eye(3)
    r_i = DEFAULT_HIP_OFFSETS[0]
    for label, target_dist, expected in (
        ("below l_min", 0.05, 0.15),
        ("above l_max", 0.55, 0.40),
    ):
        backend = _backend()
        foot = backend.variable((3,), name="foot")
        target = np.array([r_i[0], r_i[1], p_B[2] - target_dist])
        backend.opti.minimize(ca.sumsqr(foot - target))
        backend.set_initial(foot, [r_i[0], r_i[1], p_B[2] - 0.2])
        result = add_reachability_constraints(
            backend, p_B, R_B, foot, r_i, legs=(0,),
            l_min=0.15, l_max=0.40,
        )
        backend.solve()
        solution = np.asarray(backend.value(foot)).reshape(-1)
        hip = p_B + r_i
        _report(
            f"Eq (31) reachability, {label}",
            backend,
            result,
            "solution foot: "
            + np.array2string(solution, precision=6)
            + f"\ndistance to hip: "
            f"{np.linalg.norm(solution - hip):.6f} (expect {expected})",
        )


def scenario_spline_reachability():
    ca = solver_backend.ca
    backend = _backend()
    coeffs = backend.variable((6, 5), name="coeffs")
    backend.opti.subject_to(
        coeffs[:, 0] == [0.0, 0.0, 0.3, 0.0, 0.0, 0.0]
    )
    for k in range(1, 5):
        backend.opti.subject_to(coeffs[:, k] == 0.0)
    foot = backend.variable((3,), name="foot")
    target = np.array([0.0, 0.0, -0.2])
    backend.opti.minimize(ca.sumsqr(foot - target))
    backend.set_initial(foot, [0.0, 0.0, -0.1])
    result = add_spline_phase_reachability_constraints(
        backend, coeffs, duration=0.3, footholds=foot,
        r_hips=np.zeros((4, 3)), stance_legs=("FL",),
        l_min=0.15, l_max=0.40, phase=0, num_samples=2,
    )
    backend.solve()
    solution = np.asarray(backend.value(foot)).reshape(-1)
    hip = np.array([0.0, 0.0, 0.3])  # coeffs 的零阶项; r_hips 为零
    _report(
        "Eq (31) reachability sampled over a spline phase",
        backend,
        result,
        "solution foot: "
        + np.array2string(solution, precision=6)
        + f"\ndistance to hip: {np.linalg.norm(solution - hip):.6f} "
        "(expect 0.40)",
    )


def scenario_spline_friction():
    """令 p_ddot = (0.1, 0.1, -10.81) => a_B = (-0.1,-0.1,1.0)."""
    ca = solver_backend.ca
    backend = _backend()
    coeffs = backend.variable((6, 5), name="coeffs")
    backend.opti.subject_to(coeffs[:, 0] == 0.0)
    backend.opti.subject_to(coeffs[:, 1] == 0.0)
    for k in (3, 4):
        backend.opti.subject_to(coeffs[:, k] == 0.0)
    backend.opti.subject_to(
        coeffs[:, 2] == [0.05, 0.05, -5.405, 0.0, 0.0, 0.0]
    )
    backend.opti.minimize(ca.sumsqr(coeffs))
    _, a_B = spline_base_acceleration(coeffs, 0.0)
    result = add_spline_phase_friction_constraints(
        backend, coeffs, duration=0.3, mu=MU,
        include_push=False, phase=0, num_samples=3,
    )
    backend.solve()
    a_solution = np.array(
        [float(backend.value(value)) for value in a_B]
    )
    _report(
        "Eq (5) friction sampled over a spline phase",
        backend,
        result,
        "a_B at tau=0: "
        + np.array2string(a_solution, precision=6)
        + f"\ncone slack (||.|| - mu*az): "
        f"{np.linalg.norm(a_solution[:2]) - MU * a_solution[2]:.6f}",
    )


def main():
    if solver_backend.ca is None:
        raise SystemExit("CasADi 不可用, 无法运行验证脚本")
    print(f"CasADi {solver_backend.ca.__version__}")
    scenario_convexity()
    scenario_completion()
    scenario_static_state()
    scenario_friction_inside()
    scenario_friction_outside()
    scenario_push()
    scenario_literal_contradiction()
    scenario_reachability()
    scenario_spline_reachability()
    scenario_spline_friction()


if __name__ == "__main__":
    main()
