"""Step 1 验证: C++ CppAD 核心 vs Python/CasADi.

覆盖:
  1. 变量布局 (spline_coeffs | desired_footholds | slack)
  2. 初始位姿/速度约束
  3. horizon 内不触地腿的 desired 钉扎
  4. 相邻相位 C0/C1 连接约束
  5. 局部高程图 h / h_s1 / h_s2 双线性查询

运行:
  bash task/DTC/tamols/cpp/nlp/build_core.sh
  uv run python task/DTC/tamols/cpp/nlp/test_step1.py
"""

from __future__ import annotations

import ctypes
import math
from pathlib import Path

import numpy as np

from myrobot_mjlab.task.DTC.tamols.synthetic_scenarios import (
    CONTACT_NORMAL,
    build_scenario,
)


HERE = Path(__file__).resolve().parent
LIB = HERE / "libtamols_core.so"


def build_reference():
    nlp, (coeffs, desired), _ = build_scenario("crawl_flat", CONTACT_NORMAL)
    nlp.add_initial_state_constraints()
    nlp.add_junction_constraints()

    P = nlp.layout.num_phases
    K = nlp.layout.num_coeffs
    x0 = np.concatenate(
        [
            np.asarray(coeffs, dtype=float).reshape(-1),
            np.asarray(desired, dtype=float).reshape(-1),
            np.zeros(P, dtype=float),
        ]
    )

    entries = list(nlp.constraint_entries)
    r_py = eval_py_residuals(nlp, x0, entries)
    j_py = fd_jacobian(nlp, x0, entries)

    pin_mask = np.array(
        [0 if nlp.schedules[leg].touchdown_leg else 1 for leg in range(4)],
        dtype=np.int32,
    )
    measured = np.zeros(12, dtype=float)
    for leg in range(4):
        if leg in nlp.params.measured_feet:
            measured[3 * leg : 3 * leg + 3] = nlp.params.measured_feet[leg]

    return nlp, x0, r_py, j_py, pin_mask, measured


def eval_py_residuals(nlp, x, entries):
    P = nlp.layout.num_phases
    K = nlp.layout.num_coeffs
    coeff_count = P * 6 * K
    coeffs = x[:coeff_count].reshape(P, 6, K)
    desired = x[coeff_count : coeff_count + 12].reshape(4, 3)
    slack = x[coeff_count + 12 :]
    nlp.set_initial_guess([coeffs[p] for p in range(P)], desired, slack)
    opti = nlp.backend.opti
    values = opti.initial()
    out = []
    for entry in entries:
        out.append(
            np.asarray(
                opti.debug.value(entry.residual, values)
            ).reshape(-1)
        )
    return np.concatenate(out)


def fd_jacobian(nlp, x, entries, eps=1e-6):
    n = x.size
    base = eval_py_residuals(nlp, x, entries)
    J = np.zeros((base.size, n), dtype=float)
    for i in range(n):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        J[:, i] = (
            eval_py_residuals(nlp, xp, entries)
            - eval_py_residuals(nlp, xm, entries)
        ) / (2.0 * eps)
    return J


def call_cpp(nlp, x0, pin_mask, measured):
    lib = ctypes.CDLL(str(LIB))
    lib.tamols_step1_eval.restype = ctypes.c_int
    lib.tamols_step1_eval.argtypes = [
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
    ]
    P = nlp.layout.num_phases
    K = nlp.layout.num_coeffs
    n = x0.size
    # 先算 m
    m = ctypes.c_int(0)
    rc = lib.tamols_step1_eval(
        P, K,
        x0.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        nlp.params.base_pose0.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        nlp.params.base_twist0.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        np.asarray(nlp.durations, dtype=float).ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)
        ),
        pin_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        measured.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        None, None, ctypes.byref(m),
    )
    assert rc == 0, rc
    r = np.zeros(m.value, dtype=float)
    J = np.zeros((m.value, n), dtype=float)
    rc = lib.tamols_step1_eval(
        P, K,
        x0.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        nlp.params.base_pose0.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        nlp.params.base_twist0.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        np.asarray(nlp.durations, dtype=float).ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)
        ),
        pin_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        measured.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        J.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(m),
    )
    assert rc == 0, rc
    return r, J


def check_map(nlp):
    maps = nlp.maps
    h = maps.h
    hs1 = maps.hs1
    hs2 = maps.hs2
    assert hs1 is not None and hs2 is not None
    ny, nx = h.data.shape
    origin = np.array(h.origin, dtype=float)
    yaw = float(h.yaw)
    dx, dy = h.dx, h.dy

    # 覆盖地图范围的采样点 (世界系), 包含中心和边缘附近
    xs = np.linspace(origin[0] - 0.2, origin[0] + (nx - 1) * dx + 0.2, 11)
    ys = np.linspace(origin[1] - 0.2, origin[1] + (ny - 1) * dy + 0.2, 11)
    px, py = np.meshgrid(xs, ys)
    px = px.reshape(-1)
    py = py.reshape(-1)

    lib = ctypes.CDLL(str(LIB))
    lib.tamols_map_eval.restype = ctypes.c_int
    lib.tamols_map_eval.argtypes = [
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    out_h = np.zeros_like(px)
    out_s1 = np.zeros_like(px)
    out_s2 = np.zeros_like(px)
    rc = lib.tamols_map_eval(
        ny, nx,
        h.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        hs1.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        hs2.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dx, dy, origin[0], origin[1], yaw,
        px.size,
        px.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        py.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_h.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_s1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_s2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    assert rc == 0, rc

    py_h = np.array([maps.height_h(float(x), float(y)) for x, y in zip(px, py)])
    py_s1 = np.array([hs1.query(float(x), float(y)) for x, y in zip(px, py)])
    py_s2 = np.array([maps.height_hs2(float(x), float(y)) for x, y in zip(px, py)])
    return (
        float(np.max(np.abs(out_h - py_h))),
        float(np.max(np.abs(out_s1 - py_s1))),
        float(np.max(np.abs(out_s2 - py_s2))),
    )


def main():
    nlp, x0, r_py, j_py, pin_mask, measured = build_reference()
    r_cpp, J_cpp = call_cpp(nlp, x0, pin_mask, measured)
    print(f"P={nlp.layout.num_phases}, K={nlp.layout.num_coeffs}, n={x0.size}")
    print(f"residuals: python={r_py.size}, cpp={r_cpp.size}")
    print(f"max |r_cpp - r_py| = {np.max(np.abs(r_cpp - r_py)):.3e}")
    print(f"max |J_cpp - J_py| = {np.max(np.abs(J_cpp - j_py)):.3e}")

    e_h, e_s1, e_s2 = check_map(nlp)
    print(f"max |h_cpp - h_py|   = {e_h:.3e}")
    print(f"max |s1_cpp - s1_py| = {e_s1:.3e}")
    print(f"max |s2_cpp - s2_py| = {e_s2:.3e}")

    tol_r = 1e-10
    tol_j = 1e-8  # Python 参考是中心差分, C++ 是 CppAD 精确 Jacobian
    ok = (
        r_cpp.size == r_py.size
        and np.max(np.abs(r_cpp - r_py)) < tol_r
        and np.max(np.abs(J_cpp - j_py)) < tol_j
        and e_h < 1e-12 and e_s1 < 1e-12 and e_s2 < 1e-12
    )
    print("STEP 1:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
