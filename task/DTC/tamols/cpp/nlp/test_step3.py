"""Step 3 验证: C++ V-B1..V-B8 + slack 的目标值/总梯度 vs Python/CasADi.

运行:
  bash task/DTC/tamols/cpp/nlp/build_core.sh
  uv run python task/DTC/tamols/cpp/nlp/test_step3.py
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from myrobot_mjlab.task.DTC.tamols.synthetic_scenarios import (
    CONTACT_NORMAL,
    build_scenario,
)


HERE = Path(__file__).resolve().parent
LIB = HERE / "libtamols_core.so"


def setup():
    nlp, (coeffs, desired), _ = build_scenario("crawl_flat", CONTACT_NORMAL)
    prev = np.asarray(desired, dtype=float) + np.array(
        [[0.01, -0.02, 0.0]] * 4
    )
    nlp.params.previous_footholds = prev
    obj = nlp.add_objectives()

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

    stance_mask = np.zeros(P * 4, dtype=np.int32)
    for p in range(P):
        for leg in range(4):
            stance_mask[p * 4 + leg] = int(nlp.schedules[leg].in_stance(p))

    m = nlp.maps
    args = dict(
        P=P,
        K=K,
        x=x0,
        durations=np.asarray(nlp.durations, dtype=float),
        stance_mask=stance_mask,
        td_legs=np.asarray(nlp.touchdown_legs(), dtype=np.int32),
        prev=prev.reshape(-1),
        inertia=np.asarray(nlp.params.inertia_B, dtype=float).reshape(-1),
        hip_offsets=np.asarray(nlp.hip_offsets, dtype=float).reshape(-1),
        h_des=float(nlp.params.h_des),
        eps_min=float(nlp.params.eps_min_m2),
        num_samples=int(nlp.weights.num_samples),
        ref_vx=0.25,
        ref_vy=0.0,
        ref_wz=0.0,
        ny=int(m.h.data.shape[0]),
        nx=int(m.h.data.shape[1]),
        h=m.h.data,
        hs1=m.hs1.data,
        hs2=m.hs2.data,
        ghx=m.grad_h_x.data,
        ghy=m.grad_h_y.data,
        gs1x=m.grad_hs1_x.data,
        gs1y=m.grad_hs1_y.data,
        dx=float(m.h.dx),
        dy=float(m.h.dy),
        ox=float(m.h.origin[0]),
        oy=float(m.h.origin[1]),
        yaw=float(m.h.yaw),
        weights=dict(
            vb1=float(nlp.weights.footholds_on_ground),
            vb2=float(nlp.weights.leg_collision),
            vb3=float(nlp.weights.nominal_kinematics),
            vb4=float(nlp.weights.base_pose_alignment),
            vb5=float(nlp.weights.edge_avoidance),
            vb6=float(nlp.weights.previous_solution),
            vb7=float(nlp.weights.tracking),
            vb8=float(nlp.weights.smoothness),
            slack=float(nlp.weights.robustness_margin),
        ),
        use_slack=int(nlp.layout.use_slack),
    )
    return nlp, obj, x0, args


def eval_record(nlp, x, expr):
    P = nlp.layout.num_phases
    K = nlp.layout.num_coeffs
    cc = P * 6 * K
    coeffs = x[:cc].reshape(P, 6, K)
    desired = x[cc : cc + 12].reshape(4, 3)
    slack = x[cc + 12 :]
    nlp.set_initial_guess([coeffs[p] for p in range(P)], desired, slack)
    opti = nlp.backend.opti
    return float(np.asarray(opti.debug.value(expr, opti.initial())))


def fd_total_gradient(nlp, obj, x, eps=1e-6):
    total = obj.total()
    n = x.size
    g = np.zeros(n, dtype=float)
    for i in range(n):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        g[i] = (
            eval_record(nlp, xp, total) - eval_record(nlp, xm, total)
        ) / (2.0 * eps)
    return g


def call_cpp(args, py_count):
    lib = ctypes.CDLL(str(LIB))
    lib.tamols_step3_eval.restype = ctypes.c_int
    D = ctypes.POINTER(ctypes.c_double)
    I = ctypes.POINTER(ctypes.c_int)
    sig = [
        ctypes.c_int, ctypes.c_int, D, D, I, I, ctypes.c_int, D, ctypes.c_int,
        D, D, ctypes.c_double, ctypes.c_double, ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_int, ctypes.c_int,
        D, D, D, D, D, D, D,
        ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        *([ctypes.c_double] * 9),
        ctypes.c_int,
        D, D, I, D, D,
    ]
    lib.tamols_step3_eval.argtypes = sig

    def ptr(a):
        return a.ctypes.data_as(D)

    def iptr(a):
        return a.ctypes.data_as(I)

    w = args["weights"]
    base = [
        args["P"], args["K"], ptr(args["x"]), ptr(args["durations"]),
        iptr(args["stance_mask"]), iptr(args["td_legs"]),
        int(args["td_legs"].size), ptr(args["prev"]), 1,
        ptr(args["inertia"]), ptr(args["hip_offsets"]),
        args["h_des"], args["eps_min"], args["num_samples"],
        args["ref_vx"], args["ref_vy"], args["ref_wz"],
        args["ny"], args["nx"],
        ptr(args["h"]), ptr(args["hs1"]), ptr(args["hs2"]),
        ptr(args["ghx"]), ptr(args["ghy"]), ptr(args["gs1x"]), ptr(args["gs1y"]),
        args["dx"], args["dy"], args["ox"], args["oy"], args["yaw"],
        w["vb1"], w["vb2"], w["vb3"], w["vb4"], w["vb5"],
        w["vb6"], w["vb7"], w["vb8"], w["slack"],
        args["use_slack"],
    ]

    count = ctypes.c_int(0)
    rc = lib.tamols_step3_eval(
        *base, None, None, ctypes.byref(count), None, None
    )
    assert rc == 0, rc
    assert count.value == py_count, (count.value, py_count)
    res = np.zeros(count.value, dtype=float)
    wts = np.zeros(count.value, dtype=float)
    total = ctypes.c_double(0.0)
    grad = np.zeros(args["x"].size, dtype=float)
    rc = lib.tamols_step3_eval(
        *base, ptr(res), ptr(wts), ctypes.byref(count),
        ctypes.byref(total), ptr(grad),
    )
    assert rc == 0, rc
    return res, wts, float(total.value), grad


def main():
    nlp, obj, x0, args = setup()
    records = list(obj.records)
    py_res = np.array([eval_record(nlp, x0, r.residual) for r in records])
    py_w = np.array([r.weight for r in records])
    py_total = eval_record(nlp, x0, obj.total())
    py_grad = fd_total_gradient(nlp, obj, x0)

    cpp_res, cpp_w, cpp_total, cpp_grad = call_cpp(args, len(records))
    print(f"P={args['P']}, K={args['K']}, n={x0.size}, records={len(records)}")
    print(f"max |record residual| = {np.max(np.abs(cpp_res - py_res)):.3e}")
    print(f"max |record weight|   = {np.max(np.abs(cpp_w - py_w)):.3e}")
    print(f"|total_cpp - total_py| = {abs(cpp_total - py_total):.3e}")
    print(f"max |grad_cpp - grad_py| = {np.max(np.abs(cpp_grad - py_grad)):.3e}")

    if np.max(np.abs(cpp_res - py_res)) > 1e-9:
        for i in np.argsort(np.abs(cpp_res - py_res))[-5:]:
            print(f"  rec {i} {records[i].name}: cpp={cpp_res[i]:.8g} py={py_res[i]:.8g}")
    ok = (
        np.max(np.abs(cpp_res - py_res)) < 1e-9
        and np.max(np.abs(cpp_w - py_w)) < 1e-12
        and abs(cpp_total - py_total) < 1e-7
        and np.max(np.abs(cpp_grad - py_grad)) < 1e-5
    )
    print("STEP 3:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
