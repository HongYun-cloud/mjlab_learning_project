"""Step 4 验证: C++ solver eval / GN-SQP vs Python CasADi GN-SQP.

运行:
  bash task/DTC/tamols/cpp/nlp/build_core.sh
  uv run python task/DTC/tamols/cpp/nlp/test_step4.py
"""

from __future__ import annotations

import ctypes
import time
from pathlib import Path

import numpy as np

from myrobot_mjlab.task.DTC.tamols.gn_sqp_solver import (
    GNSQPOptions,
    extract_nlp,
    solve_gn_sqp,
)
from myrobot_mjlab.task.DTC.tamols.synthetic_scenarios import (
    CONTACT_NORMAL,
    build_scenario,
)


HERE = Path(__file__).resolve().parent
LIB = HERE / "libtamols_core.so"

D = ctypes.POINTER(ctypes.c_double)
I = ctypes.POINTER(ctypes.c_int)


class Problem(ctypes.Structure):
    _fields_ = [
        ("num_phases", ctypes.c_int),
        ("num_coeffs", ctypes.c_int),
        ("num_samples", ctypes.c_int),
        ("num_td", ctypes.c_int),
        ("use_slack", ctypes.c_int),
        ("giac_pairs_mode", ctypes.c_int),
        ("winding", ctypes.c_int),
        ("friction_mode", ctypes.c_int),
        ("friction_form", ctypes.c_int),
        ("include_push", ctypes.c_int),
        ("inertial_sign", ctypes.c_int),
        ("convexity_fourth", ctypes.c_int),
        ("has_prev", ctypes.c_int),
        ("include_dynamics", ctypes.c_int),
        ("include_kinematic", ctypes.c_int),
        ("h_des", ctypes.c_double),
        ("eps_min", ctypes.c_double),
        ("mu", ctypes.c_double),
        ("l_min", ctypes.c_double),
        ("l_max", ctypes.c_double),
        ("mass", ctypes.c_double),
        ("ref_vx", ctypes.c_double),
        ("ref_vy", ctypes.c_double),
        ("ref_wz", ctypes.c_double),
        ("map_ny", ctypes.c_int),
        ("map_nx", ctypes.c_int),
        ("map_dx", ctypes.c_double),
        ("map_dy", ctypes.c_double),
        ("origin_x", ctypes.c_double),
        ("origin_y", ctypes.c_double),
        ("yaw", ctypes.c_double),
        ("weights", ctypes.c_double * 9),
        ("durations", D),
        ("stance_mask", I),
        ("measured_mask", I),
        ("pin_mask", I),
        ("td_legs", I),
        ("measured_feet", D),
        ("base_pose0", D),
        ("base_twist0", D),
        ("inertia9", D),
        ("hip_offsets12", D),
        ("previous_footholds", D),
        ("map_h", D),
        ("map_hs1", D),
        ("map_hs2", D),
        ("grad_hx", D),
        ("grad_hy", D),
        ("grad_s1x", D),
        ("grad_s1y", D),
    ]


def arr(a):
    a = np.ascontiguousarray(a, dtype=np.float64)
    return a, a.ctypes.data_as(D)


def iarr(a):
    a = np.ascontiguousarray(a, dtype=np.int32)
    return a, a.ctypes.data_as(I)


def setup():
    nlp, (coeffs, desired), _ = build_scenario("crawl_flat", CONTACT_NORMAL)
    prev = np.asarray(desired, dtype=float) + np.array([[0.01, -0.02, 0.0]] * 4)
    nlp.params.previous_footholds = prev
    nlp.add_initial_state_constraints()
    nlp.add_junction_constraints()
    nlp.add_dynamic_stability_constraints()
    nlp.add_kinematic_contact_constraints()
    nlp.add_objectives()
    nlp.finalize()

    P = nlp.layout.num_phases
    K = nlp.layout.num_coeffs
    cc = P * 6 * K
    x0 = np.concatenate(
        [
            np.asarray(coeffs, dtype=float).reshape(-1),
            np.asarray(desired, dtype=float).reshape(-1),
            np.zeros(P, dtype=float),
        ]
    )
    blocks = [np.asarray(c, dtype=float) for c in coeffs] + [
        np.asarray(desired, dtype=float),
        np.zeros(P, dtype=float),
    ]

    stance = np.zeros(P * 4, dtype=np.int32)
    measured = np.zeros(P * 4, dtype=np.int32)
    for p in range(P):
        for leg in range(4):
            stance[p * 4 + leg] = int(nlp.schedules[leg].in_stance(p))
            measured[p * 4 + leg] = int(nlp.schedules[leg].foot_is_measured(p))
    pin = np.array([0 if nlp.schedules[leg].touchdown_leg else 1 for leg in range(4)], dtype=np.int32)
    mfeet = np.zeros(12)
    for leg in range(4):
        if leg in nlp.params.measured_feet:
            mfeet[3 * leg : 3 * leg + 3] = nlp.params.measured_feet[leg]

    a = nlp.assumptions
    friction_mode = 0
    if a.friction_mode == "enabled":
        friction_mode = 2 if a.friction_cone_form == "force_balance_derived" else 1
    m = nlp.maps
    weights = [
        float(nlp.weights.footholds_on_ground),
        float(nlp.weights.leg_collision),
        float(nlp.weights.nominal_kinematics),
        float(nlp.weights.base_pose_alignment),
        float(nlp.weights.edge_avoidance),
        float(nlp.weights.previous_solution),
        float(nlp.weights.tracking),
        float(nlp.weights.smoothness),
        float(nlp.weights.robustness_margin),
    ]

    keep = []
    def setv(name, val):
        keep.append(val)
    p = Problem()
    p.num_phases = P
    p.num_coeffs = K
    p.num_samples = int(nlp.weights.num_samples)
    p.num_td = len(nlp.touchdown_legs())
    p.use_slack = int(nlp.layout.use_slack)
    p.giac_pairs_mode = int(a.giac_pairs != "cyclic_edges")
    p.winding = int(a.resolved_winding() == "cw")
    p.friction_mode = friction_mode
    p.friction_form = int(a.friction_form != "norm")
    p.include_push = int(a.include_push)
    p.inertial_sign = int(a.inertial_accel_sign != "paper_literal")
    p.convexity_fourth = int(a.convexity_fourth_term != "p21_engineering")
    p.has_prev = 1
    p.include_dynamics = 1
    p.include_kinematic = 1
    p.h_des = float(nlp.params.h_des)
    p.eps_min = float(nlp.params.eps_min_m2)
    p.mu = float(a.mu)
    p.l_min = float(nlp.params.l_min)
    p.l_max = float(nlp.params.l_max)
    p.mass = float(nlp.params.mass)
    p.ref_vx, p.ref_vy, p.ref_wz = 0.25, 0.0, 0.0
    p.map_ny, p.map_nx = int(m.h.data.shape[0]), int(m.h.data.shape[1])
    p.map_dx, p.map_dy = float(m.h.dx), float(m.h.dy)
    p.origin_x, p.origin_y = float(m.h.origin[0]), float(m.h.origin[1])
    p.yaw = float(m.h.yaw)
    for i, w in enumerate(weights):
        p.weights[i] = w

    def set_arr(field, name, value, ints=False):
        a_, ptr = iarr(value) if ints else arr(value)
        setv(name, a_)
        setattr(p, field, ptr)

    set_arr("durations", "durations", nlp.durations)
    set_arr("stance_mask", "stance", stance, True)
    set_arr("measured_mask", "measured", measured, True)
    set_arr("pin_mask", "pin", pin, True)
    set_arr("td_legs", "td", np.array(nlp.touchdown_legs(), dtype=np.int32), True)
    set_arr("measured_feet", "mfeet", mfeet)
    set_arr("base_pose0", "bp0", nlp.params.base_pose0)
    set_arr("base_twist0", "bt0", nlp.params.base_twist0)
    set_arr("inertia9", "inertia", np.asarray(nlp.params.inertia_B).reshape(-1))
    set_arr("hip_offsets12", "hip", np.asarray(nlp.hip_offsets).reshape(-1))
    set_arr("previous_footholds", "prev", prev.reshape(-1))
    set_arr("map_h", "h", m.h.data)
    set_arr("map_hs1", "hs1", m.hs1.data)
    set_arr("map_hs2", "hs2", m.hs2.data)
    set_arr("grad_hx", "ghx", m.grad_h_x.data)
    set_arr("grad_hy", "ghy", m.grad_h_y.data)
    set_arr("grad_s1x", "gs1x", m.grad_hs1_x.data)
    set_arr("grad_s1y", "gs1y", m.grad_hs1_y.data)

    return nlp, x0, blocks, p, keep


def cpp_eval(p, x, reg=1e-8):
    lib = ctypes.CDLL(str(LIB))
    lib.tamols_solver_eval.restype = ctypes.c_int
    lib.tamols_solver_eval.argtypes = [
        ctypes.POINTER(Problem), D,
        D, D, D, D, D, D, D,
        I, I, I, ctypes.c_double,
    ]
    xa, xp = arr(x)
    n = ctypes.c_int(0)
    mg = ctypes.c_int(0)
    mh = ctypes.c_int(0)
    rc = lib.tamols_solver_eval(
        ctypes.byref(p), xp,
        None, None, None, None, None, None, None,
        ctypes.byref(n), ctypes.byref(mg), ctypes.byref(mh), reg,
    )
    assert rc == 0
    f = ctypes.c_double(0.0)
    g = np.zeros(mg.value)
    h = np.zeros(mh.value)
    gf = np.zeros(n.value)
    Jg = np.zeros(mg.value * n.value)
    Jh = np.zeros(mh.value * n.value)
    H = np.zeros(n.value * n.value)
    rc = lib.tamols_solver_eval(
        ctypes.byref(p), xp,
        ctypes.byref(f), arr(g)[1], arr(h)[1], arr(gf)[1],
        arr(Jg)[1], arr(Jh)[1], arr(H)[1],
        ctypes.byref(n), ctypes.byref(mg), ctypes.byref(mh), reg,
    )
    assert rc == 0
    return f.value, g, h, gf, Jg, Jh, H, n.value


def cpp_solve(p, x0, max_iter=60, tol_step=1e-9, tol_viol=1e-6,
              tol_stationary=1e-9, qp_lin_tol=1e-6,
              merit_rho=1e3, armijo_eta=1e-4, t_min=1e-4, reg=1e-8):
    lib = ctypes.CDLL(str(LIB))
    lib.tamols_gn_sqp_solve.restype = ctypes.c_int
    lib.tamols_gn_sqp_solve.argtypes = [
        ctypes.POINTER(Problem), D, ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        D, ctypes.POINTER(ctypes.c_double), I,
        ctypes.POINTER(ctypes.c_double), ctypes.c_char_p, ctypes.c_int,
    ]
    xa, xp = arr(x0)
    out = np.zeros_like(xa)
    f = ctypes.c_double(0.0)
    it = ctypes.c_int(0)
    max_viol = ctypes.c_double(0.0)
    status = ctypes.create_string_buffer(128)
    rc = lib.tamols_gn_sqp_solve(
        ctypes.byref(p), xp, max_iter,
        tol_step, tol_viol, tol_stationary, qp_lin_tol,
        merit_rho, armijo_eta, t_min, reg,
        arr(out)[1], ctypes.byref(f), ctypes.byref(it),
        ctypes.byref(max_viol), status, len(status),
    )
    assert rc == 0
    return out, f.value, it.value, status.value.decode(), max_viol.value


def main():
    nlp, x0, blocks, p, keep = setup()
    ex = extract_nlp(nlp)
    t = time.perf_counter()
    out = ex.f_eval(x0)
    py_f = float(out[0])
    py_g = np.asarray(out[1]).ravel()
    py_h = np.asarray(out[2]).ravel()
    py_gf = np.asarray(out[3]).ravel()
    py_Jg = np.asarray(out[4])
    py_Jh = np.asarray(out[5])
    py_Jr = np.asarray(out[7])
    py_H = py_Jr.T @ py_Jr + 1e-8 * np.eye(x0.size)

    c_f, c_g, c_h, c_gf, c_Jg, c_Jh, c_H, n = cpp_eval(p, x0)
    print(f"n={n}, mg={py_g.size}, mh={py_h.size}")
    print(f"|f_cpp - f_py| = {abs(c_f - py_f):.3e}")
    print(f"max |g| = {np.max(np.abs(c_g - py_g)):.3e}")
    print(f"max |h| = {np.max(np.abs(c_h - py_h)):.3e}")
    print(f"max |gf| = {np.max(np.abs(c_gf - py_gf)):.3e}")
    c_Jg = c_Jg.reshape(py_Jg.shape)
    c_Jh = c_Jh.reshape(py_Jh.shape)
    c_H = c_H.reshape(py_H.shape)
    print(f"max |Jg| = {np.max(np.abs(c_Jg - py_Jg)):.3e}")
    print(f"max |Jh| = {np.max(np.abs(c_Jh - py_Jh)):.3e}")
    print(f"max |H|  = {np.max(np.abs(c_H - py_H)):.3e}")

    opts = GNSQPOptions()
    t = time.perf_counter()
    rep = solve_gn_sqp(nlp, blocks, opts)
    py_time = time.perf_counter() - t
    py_x = np.concatenate([np.asarray(b).reshape(-1) for b in blocks])
    py_sol = rep.solution
    py_vec = np.concatenate(
        [np.asarray(py_sol["spline_coeffs"]).reshape(-1),
         np.asarray(py_sol["desired_footholds"]).reshape(-1)]
        + ([np.asarray(py_sol["slack"]).reshape(-1)] if "slack" in py_sol else [])
    )
    cpp_x, cpp_f, cpp_it, cpp_status, cpp_viol = cpp_solve(p, x0, max_iter=opts.max_iter)
    print(f"py status={rep.status}, it={rep.iterations}, f={rep.objective_total:.8g}, {py_time*1e3:.1f} ms")
    print(f"cpp status={cpp_status}, it={cpp_it}, f={cpp_f:.8g}")
    print(f"max |x_cpp - x_py| = {np.max(np.abs(cpp_x - py_vec)):.3e}")
    print(f"|f_cpp - f_py| = {abs(cpp_f - rep.objective_total):.3e}")

    ok = (
        np.max(np.abs(c_g - py_g)) < 1e-9
        and np.max(np.abs(c_h - py_h)) < 1e-9
        and np.max(np.abs(c_gf - py_gf)) < 1e-7
        and np.max(np.abs(c_Jg - py_Jg)) < 1e-7
        and np.max(np.abs(c_Jh - py_Jh)) < 1e-7
        and np.max(np.abs(c_H - py_H)) < 1e-6
        and np.max(np.abs(cpp_x - py_vec)) < 1e-1
    )
    print("STEP 4:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
