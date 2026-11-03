"""Step 2 验证: C++ GIAC/摩擦/凸性/可达 vs Python/CasADi.

覆盖:
  - 17b (N>=3 cyclic pairs)
  - 17c/17d (N=2)
  - 17e (N=1)
  - 17f (N=0)
  - 摩擦 5 / V-A-4
  - 式 (30) 凸性
  - 式 (31) 可达

运行:
  bash task/DTC/tamols/cpp/nlp/build_core.sh
  uv run python task/DTC/tamols/cpp/nlp/test_step2.py
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from myrobot_mjlab.task.DTC.tamols.assumptions_config import AssumptionsConfig
from myrobot_mjlab.task.DTC.tamols.synthetic_scenarios import (
    CONTACT_NORMAL,
    build_scenario,
)


HERE = Path(__file__).resolve().parent
LIB = HERE / "libtamols_core.so"


def flatten_entries(nlp):
    values = []
    opti = nlp.backend.opti
    init = opti.initial()
    for entry in nlp.constraint_entries:
        values.append(
            np.asarray(opti.debug.value(entry.residual, init)).reshape(-1)
        )
    return np.concatenate(values)


def eval_py_residuals(nlp, x):
    P = nlp.layout.num_phases
    K = nlp.layout.num_coeffs
    coeff_count = P * 6 * K
    coeffs = x[:coeff_count].reshape(P, 6, K)
    desired = x[coeff_count : coeff_count + 12].reshape(4, 3)
    slack = x[coeff_count + 12 :]
    nlp.set_initial_guess([coeffs[p] for p in range(P)], desired, slack)
    return flatten_entries(nlp)


def fd_jacobian(nlp, x, eps=1e-6):
    n = x.size
    base = eval_py_residuals(nlp, x)
    J = np.zeros((base.size, n), dtype=float)
    for i in range(n):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        J[:, i] = (
            eval_py_residuals(nlp, xp) - eval_py_residuals(nlp, xm)
        ) / (2.0 * eps)
    return J


def build_reference(assumptions=CONTACT_NORMAL):
    nlp, (coeffs, desired), _ = build_scenario("crawl_flat", assumptions)
    nlp.add_initial_state_constraints()
    nlp.add_junction_constraints()
    nlp.add_dynamic_stability_constraints()
    nlp.add_kinematic_contact_constraints()

    P = nlp.layout.num_phases
    K = nlp.layout.num_coeffs
    coeff_count = P * 6 * K
    x0 = np.concatenate(
        [
            np.asarray(coeffs, dtype=float).reshape(-1),
            np.asarray(desired, dtype=float).reshape(-1),
            np.zeros(P, dtype=float),
        ]
    )

    pin_mask = np.array(
        [0 if nlp.schedules[leg].touchdown_leg else 1 for leg in range(4)],
        dtype=np.int32,
    )
    stance_mask = np.zeros(P * 4, dtype=np.int32)
    measured_mask = np.zeros(P * 4, dtype=np.int32)
    for p in range(P):
        for leg in range(4):
            stance_mask[p * 4 + leg] = int(nlp.schedules[leg].in_stance(p))
            measured_mask[p * 4 + leg] = int(
                nlp.schedules[leg].foot_is_measured(p)
            )

    measured = np.zeros(12, dtype=float)
    for leg in range(4):
        if leg in nlp.params.measured_feet:
            measured[3 * leg : 3 * leg + 3] = nlp.params.measured_feet[leg]

    a = nlp.assumptions
    friction_mode = 0
    if a.friction_mode == "enabled":
        friction_mode = (
            2 if a.friction_cone_form == "force_balance_derived" else 1
        )
    args = dict(
        P=P,
        K=K,
        x0=x0,
        durations=np.asarray(nlp.durations, dtype=float),
        stance_mask=stance_mask,
        measured_mask=measured_mask,
        measured=measured,
        pin_mask=pin_mask,
        base_pose0=np.asarray(nlp.params.base_pose0, dtype=float),
        base_twist0=np.asarray(nlp.params.base_twist0, dtype=float),
        mass=float(nlp.params.mass),
        inertia=np.asarray(nlp.params.inertia_B, dtype=float).reshape(-1),
        hip_offsets=np.asarray(nlp.hip_offsets, dtype=float).reshape(-1),
        l_min=float(nlp.params.l_min),
        l_max=float(nlp.params.l_max),
        mu=float(a.mu),
        num_samples=int(nlp.weights.num_samples),
        giac_pairs=int(a.giac_pairs != "cyclic_edges"),
        winding=int(a.resolved_winding() == "cw"),
        friction_mode=friction_mode,
        friction_form=int(a.friction_form != "norm"),
        include_push=int(a.include_push),
        inertial_sign=int(a.inertial_accel_sign != "paper_literal"),
        convexity_fourth=int(a.convexity_fourth_term != "p21_engineering"),
        use_slack=int(nlp.layout.use_slack),
    )
    return nlp, x0, args


def call_cpp(args, with_jac=True):
    lib = ctypes.CDLL(str(LIB))
    lib.tamols_step2_eval.restype = ctypes.c_int
    D = ctypes.POINTER(ctypes.c_double)
    I = ctypes.POINTER(ctypes.c_int)
    lib.tamols_step2_eval.argtypes = [
        ctypes.c_int, ctypes.c_int, D,
        D, I, I, D, I, D, D,
        ctypes.c_double, D, D,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        D, D, I,
    ]

    def ptr(a):
        return a.ctypes.data_as(D)

    def iptr(a):
        return a.ctypes.data_as(I)

    m = ctypes.c_int(0)
    rc = lib.tamols_step2_eval(
        args["P"], args["K"], ptr(args["x0"]),
        ptr(args["durations"]), iptr(args["stance_mask"]),
        iptr(args["measured_mask"]), ptr(args["measured"]),
        iptr(args["pin_mask"]), ptr(args["base_pose0"]),
        ptr(args["base_twist0"]),
        args["mass"], ptr(args["inertia"]), ptr(args["hip_offsets"]),
        args["l_min"], args["l_max"], args["mu"],
        args["num_samples"],
        args["giac_pairs"], args["winding"],
        args["friction_mode"], args["friction_form"], args["include_push"],
        args["inertial_sign"], args["convexity_fourth"], args["use_slack"],
        None, None, ctypes.byref(m),
    )
    assert rc == 0, rc
    r = np.zeros(m.value, dtype=float)
    J = None
    if with_jac:
        J = np.zeros((m.value, args["x0"].size), dtype=float)
    rc = lib.tamols_step2_eval(
        args["P"], args["K"], ptr(args["x0"]),
        ptr(args["durations"]), iptr(args["stance_mask"]),
        iptr(args["measured_mask"]), ptr(args["measured"]),
        iptr(args["pin_mask"]), ptr(args["base_pose0"]),
        ptr(args["base_twist0"]),
        args["mass"], ptr(args["inertia"]), ptr(args["hip_offsets"]),
        args["l_min"], args["l_max"], args["mu"],
        args["num_samples"],
        args["giac_pairs"], args["winding"],
        args["friction_mode"], args["friction_form"], args["include_push"],
        args["inertial_sign"], args["convexity_fourth"], args["use_slack"],
        ptr(r), ptr(J) if J is not None else None, ctypes.byref(m),
    )
    assert rc == 0, rc
    return r, J


def main():
    cases = [
        ("contact_normal + paper friction", CONTACT_NORMAL),
        ("paper_literal + paper friction", AssumptionsConfig()),
        (
            "contact_normal + derived friction",
            AssumptionsConfig(
                inertial_accel_sign="contact_normal_explicit",
                friction_cone_form="force_balance_derived",
            ),
        ),
        (
            "friction disabled",
            AssumptionsConfig(friction_mode="disabled"),
        ),
        (
            "all_unordered GIAC pairs",
            AssumptionsConfig(giac_pairs="all_unordered"),
        ),
        (
            "convexity p20 literal",
            AssumptionsConfig(convexity_fourth_term="p20_literal"),
        ),
    ]
    all_ok = True
    for label, assumptions in cases:
        nlp, x0, args = build_reference(assumptions)
        r_py = eval_py_residuals(nlp, x0)
        J_py = fd_jacobian(nlp, x0)
        r_cpp, J_cpp = call_cpp(args)
        er = float(np.max(np.abs(r_cpp - r_py)))
        ej = float(np.max(np.abs(J_cpp - J_py)))
        ok = (
            r_cpp.size == r_py.size
            and er < 1e-10
            and ej < 1e-7
        )
        all_ok = all_ok and ok
        print(
            f"[{'PASS' if ok else 'FAIL'}] {label}: m={r_py.size}, "
            f"max|r|={er:.2e}, max|J|={ej:.2e}"
        )
        if not ok:
            idx = np.argsort(np.abs(r_cpp - r_py))[-5:]
            for i in idx:
                print(f"  worst residual {i}: cpp={r_cpp[i]:.6g} py={r_py[i]:.6g}")
    print("STEP 2:", "PASS" if all_ok else "FAIL")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
