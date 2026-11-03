"""C++ GN-SQP 求解器的 ctypes 封装 (Step 5).

只负责:
  - 把 DTC 的纯数据容器 (durations/schedules/maps/params) 直接打包成
    C 结构体 (``make_problem_from_data``), 不需要构建 ``IntegratedNLP``;
  - 兼容入口 ``make_problem`` 从已存在的 ``IntegratedNLP`` 打包;
  - 调 ``tamols_gn_sqp_solve`` / ``tamols_solver_eval``;
  - 解包 x.

C++ 侧仍用 Eigen + CppAD + QuadProg++，不经过 CasADi.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
_LIB_PATH = _HERE / "cpp" / "nlp" / "libtamols_core.so"

D = ctypes.POINTER(ctypes.c_double)
I = ctypes.POINTER(ctypes.c_int)


class TamolsSolverProblem(ctypes.Structure):
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


def _d(a: np.ndarray):
    a = np.ascontiguousarray(a, dtype=np.float64)
    return a, a.ctypes.data_as(D)


def _i(a: np.ndarray):
    a = np.ascontiguousarray(a, dtype=np.int32)
    return a, a.ctypes.data_as(I)


@dataclass
class CppProblem:
    struct: TamolsSolverProblem
    keep: list[Any] = field(default_factory=list)
    x0: np.ndarray | None = None
    n: int = 0

    def solve(
        self,
        x0: np.ndarray | None = None,
        *,
        max_iter: int = 60,
        tol_step: float = 1e-9,
        tol_viol: float = 1e-6,
        tol_stationary: float = 1e-9,
        qp_lin_tol: float = 1e-6,
        merit_rho: float = 1e3,
        armijo_eta: float = 1e-4,
        t_min: float = 1e-4,
        hessian_reg: float = 1e-8,
    ) -> tuple[np.ndarray, float, int, str]:
        if not _LIB_PATH.exists():
            raise FileNotFoundError(
                f"C++ solver 未编译: {_LIB_PATH}; 运行 build_core.sh"
            )
        lib = ctypes.CDLL(str(_LIB_PATH))
        lib.tamols_gn_sqp_solve.restype = ctypes.c_int
        lib.tamols_gn_sqp_solve.argtypes = [
            ctypes.POINTER(TamolsSolverProblem), D, ctypes.c_int,
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            D, ctypes.POINTER(ctypes.c_double), I,
            ctypes.POINTER(ctypes.c_double), ctypes.c_char_p, ctypes.c_int,
        ]
        start = self.x0 if x0 is None else np.asarray(x0, dtype=float)
        start = np.ascontiguousarray(start, dtype=float)
        out = np.zeros_like(start)
        f = ctypes.c_double(0.0)
        it = ctypes.c_int(0)
        max_viol = ctypes.c_double(0.0)
        status = ctypes.create_string_buffer(128)
        rc = lib.tamols_gn_sqp_solve(
            ctypes.byref(self.struct), start.ctypes.data_as(D), int(max_iter),
            tol_step, tol_viol, tol_stationary, qp_lin_tol,
            merit_rho, armijo_eta, t_min, hessian_reg,
            out.ctypes.data_as(D), ctypes.byref(f), ctypes.byref(it),
            ctypes.byref(max_viol), status, len(status),
        )
        if rc != 0:
            raise RuntimeError(f"tamols_gn_sqp_solve 返回错误码 {rc}")
        return (
            out,
            float(f.value),
            int(it.value),
            status.value.decode(),
            float(max_viol.value),
        )


def eval_problem(
    problem: CppProblem,
    x: np.ndarray,
    hessian_reg: float = 1e-8,
) -> dict[str, Any]:
    """对 TamolsSolverProblem 求 f/g/h 及其导数，用于 direct builder 对拍."""
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.tamols_solver_eval.restype = ctypes.c_int
    lib.tamols_solver_eval.argtypes = [
        ctypes.POINTER(TamolsSolverProblem), D,
        D, D, D, D, D, D, D,
        I, I, I, ctypes.c_double,
    ]
    x = np.ascontiguousarray(x, dtype=float)
    n = ctypes.c_int(0)
    mg = ctypes.c_int(0)
    mh = ctypes.c_int(0)
    rc = lib.tamols_solver_eval(
        ctypes.byref(problem.struct), x.ctypes.data_as(D),
        None, None, None, None, None, None, None,
        ctypes.byref(n), ctypes.byref(mg), ctypes.byref(mh), hessian_reg,
    )
    if rc != 0:
        raise RuntimeError(f"tamols_solver_eval 返回错误码 {rc}")
    f = ctypes.c_double(0.0)
    g = np.zeros(mg.value)
    h = np.zeros(mh.value)
    gf = np.zeros(n.value)
    Jg = np.zeros(mg.value * n.value)
    Jh = np.zeros(mh.value * n.value)
    H = np.zeros(n.value * n.value)
    rc = lib.tamols_solver_eval(
        ctypes.byref(problem.struct), x.ctypes.data_as(D),
        ctypes.byref(f), g.ctypes.data_as(D), h.ctypes.data_as(D),
        gf.ctypes.data_as(D), Jg.ctypes.data_as(D), Jh.ctypes.data_as(D),
        H.ctypes.data_as(D),
        ctypes.byref(n), ctypes.byref(mg), ctypes.byref(mh), hessian_reg,
    )
    if rc != 0:
        raise RuntimeError(f"tamols_solver_eval 返回错误码 {rc}")
    return {
        "f": float(f.value),
        "g": g,
        "h": h,
        "gf": gf,
        "Jg": Jg.reshape(mg.value, n.value),
        "Jh": Jh.reshape(mh.value, n.value),
        "H": H.reshape(n.value, n.value),
        "n": n.value,
    }


def make_problem_from_data(
    *,
    durations: Sequence[float],
    schedules: Any,
    maps: Any,
    params: Any,
    assumptions: Any,
    td_legs: Sequence[int],
    ref_vx: float,
    ref_vy: float,
    ref_wz: float,
    previous_footholds: np.ndarray | None,
    include_dynamics: bool = True,
    include_kinematic: bool = True,
    num_coeffs: int = 5,
    use_slack: bool | None = None,
    weights_obj: Any | None = None,
    hip_offsets: np.ndarray | None = None,
) -> CppProblem:
    """从纯数据容器 (durations/schedules/maps/params) 构造 C++ 问题.

    与 :func:`make_problem` 的唯一区别是不需要 ``IntegratedNLP``：不再
    创建 CasADi 表达式，只读取数组与调度表。
    """
    P = len(durations)
    if use_slack is None:
        use_slack = bool(assumptions.use_giac_slack)
    if weights_obj is None:
        from myrobot_mjlab.task.DTC.tamols.v_b_objectives import VBWeights

        weights_obj = VBWeights()

    stance = np.zeros(P * 4, dtype=np.int32)
    measured = np.zeros(P * 4, dtype=np.int32)
    for ph in range(P):
        for leg in range(4):
            stance[ph * 4 + leg] = int(schedules[leg].in_stance(ph))
            measured[ph * 4 + leg] = int(schedules[leg].foot_is_measured(ph))
    pin = np.array(
        [0 if schedules[leg].touchdown_leg else 1 for leg in range(4)],
        dtype=np.int32,
    )
    measured_feet = np.zeros(12)
    for leg in range(4):
        if leg in params.measured_feet:
            measured_feet[3 * leg : 3 * leg + 3] = params.measured_feet[leg]

    friction_mode = 0
    if assumptions.friction_mode == "enabled":
        friction_mode = (
            2
            if assumptions.friction_cone_form == "force_balance_derived"
            else 1
        )

    weights = [
        float(weights_obj.footholds_on_ground),
        float(weights_obj.leg_collision),
        float(weights_obj.nominal_kinematics),
        float(weights_obj.base_pose_alignment),
        float(weights_obj.edge_avoidance),
        float(weights_obj.previous_solution),
        float(weights_obj.tracking),
        float(weights_obj.smoothness),
        float(weights_obj.robustness_margin),
    ]

    m = maps
    return make_problem_direct(
        num_phases=P,
        num_coeffs=int(num_coeffs),
        num_samples=int(weights_obj.num_samples),
        durations=np.asarray(durations),
        stance_mask=stance,
        measured_mask=measured,
        measured_feet=measured_feet,
        pin_mask=pin,
        base_pose0=np.asarray(params.base_pose0),
        base_twist0=np.asarray(params.base_twist0),
        mass=float(params.mass),
        inertia9=np.asarray(params.inertia_B).reshape(-1),
        hip_offsets12=np.asarray(
            hip_offsets
            if hip_offsets is not None
            else (
                getattr(params, "hip_offsets", None)
                if getattr(params, "hip_offsets", None) is not None
                else _default_hip_offsets(assumptions)
            )
        ).reshape(-1),
        h_des=float(params.h_des),
        eps_min=float(params.eps_min_m2),
        mu=float(assumptions.mu),
        l_min=float(params.l_min),
        l_max=float(params.l_max),
        giac_pairs_mode=int(assumptions.giac_pairs != "cyclic_edges"),
        winding=int(assumptions.resolved_winding() == "cw"),
        friction_mode=friction_mode,
        friction_form=int(assumptions.friction_form != "norm"),
        include_push=int(assumptions.include_push),
        inertial_sign=int(assumptions.inertial_accel_sign != "paper_literal"),
        convexity_fourth=int(
            assumptions.convexity_fourth_term != "p21_engineering"
        ),
        use_slack=int(bool(use_slack)),
        weights=weights,
        td_legs=list(td_legs),
        previous_footholds=previous_footholds,
        ref_vx=ref_vx,
        ref_vy=ref_vy,
        ref_wz=ref_wz,
        map_h=np.asarray(m.h.data),
        map_hs1=np.asarray(m.hs1.data),
        map_hs2=np.asarray(m.hs2.data),
        grad_hx=np.asarray(m.grad_h_x.data),
        grad_hy=np.asarray(m.grad_h_y.data),
        grad_s1x=np.asarray(m.grad_hs1_x.data),
        grad_s1y=np.asarray(m.grad_hs1_y.data),
        map_dx=float(m.h.dx),
        map_dy=float(m.h.dy),
        origin_x=float(m.h.origin[0]),
        origin_y=float(m.h.origin[1]),
        yaw=float(m.h.yaw),
        include_dynamics=include_dynamics,
        include_kinematic=include_kinematic,
    )


def _default_hip_offsets(assumptions: Any) -> np.ndarray:
    from myrobot_mjlab.task.DTC.tamols.nlp_integration import (
        GO2_HIP_JOINT_OFFSETS,
        GO2_STANCE_HIP_OFFSETS,
    )

    if getattr(assumptions, "hip_offsets", "stance") == "stance":
        return GO2_STANCE_HIP_OFFSETS
    return GO2_HIP_JOINT_OFFSETS


def make_problem(
    nlp: Any,
    *,
    ref_vx: float,
    ref_vy: float,
    ref_wz: float,
    previous_footholds: np.ndarray | None,
    include_dynamics: bool = True,
    include_kinematic: bool = True,
) -> CppProblem:
    """从已构造的 IntegratedNLP 打包 C++ solver problem."""
    return make_problem_from_data(
        durations=nlp.durations,
        schedules=nlp.schedules,
        maps=nlp.maps,
        params=nlp.params,
        assumptions=nlp.assumptions,
        td_legs=nlp.touchdown_legs(),
        ref_vx=ref_vx,
        ref_vy=ref_vy,
        ref_wz=ref_wz,
        previous_footholds=previous_footholds,
        include_dynamics=include_dynamics,
        include_kinematic=include_kinematic,
        num_coeffs=nlp.layout.num_coeffs,
        use_slack=nlp.layout.use_slack,
        weights_obj=nlp.weights,
        hip_offsets=nlp.hip_offsets,
    )


def make_problem_direct(
    *,
    num_phases: int,
    num_coeffs: int,
    num_samples: int,
    durations: np.ndarray,
    stance_mask: np.ndarray,
    measured_mask: np.ndarray,
    measured_feet: np.ndarray,
    pin_mask: np.ndarray,
    base_pose0: np.ndarray,
    base_twist0: np.ndarray,
    mass: float,
    inertia9: np.ndarray,
    hip_offsets12: np.ndarray,
    h_des: float,
    eps_min: float,
    mu: float,
    l_min: float,
    l_max: float,
    giac_pairs_mode: int,
    winding: int,
    friction_mode: int,
    friction_form: int,
    include_push: int,
    inertial_sign: int,
    convexity_fourth: int,
    use_slack: int,
    weights: Sequence[float],
    td_legs: Sequence[int],
    previous_footholds: np.ndarray | None,
    ref_vx: float,
    ref_vy: float,
    ref_wz: float,
    map_h: np.ndarray,
    map_hs1: np.ndarray,
    map_hs2: np.ndarray,
    grad_hx: np.ndarray,
    grad_hy: np.ndarray,
    grad_s1x: np.ndarray,
    grad_s1y: np.ndarray,
    map_dx: float,
    map_dy: float,
    origin_x: float,
    origin_y: float,
    yaw: float,
    include_dynamics: bool = True,
    include_kinematic: bool = True,
) -> CppProblem:
    """完全从原始数组/状态构造 C++ 问题，不依赖 IntegratedNLP/CasADi."""
    p = TamolsSolverProblem()
    p.num_phases = int(num_phases)
    p.num_coeffs = int(num_coeffs)
    p.num_samples = int(num_samples)
    p.num_td = len(td_legs)
    p.use_slack = int(use_slack)
    p.giac_pairs_mode = int(giac_pairs_mode)
    p.winding = int(winding)
    p.friction_mode = int(friction_mode)
    p.friction_form = int(friction_form)
    p.include_push = int(include_push)
    p.inertial_sign = int(inertial_sign)
    p.convexity_fourth = int(convexity_fourth)
    p.has_prev = int(previous_footholds is not None)
    p.include_dynamics = int(include_dynamics)
    p.include_kinematic = int(include_kinematic)
    p.h_des = float(h_des)
    p.eps_min = float(eps_min)
    p.l_min = float(l_min)
    p.l_max = float(l_max)
    p.mu = float(mu)
    p.mass = float(mass)
    p.ref_vx, p.ref_vy, p.ref_wz = float(ref_vx), float(ref_vy), float(ref_wz)
    p.map_ny, p.map_nx = int(map_h.shape[0]), int(map_h.shape[1])
    p.map_dx, p.map_dy = float(map_dx), float(map_dy)
    p.origin_x, p.origin_y = float(origin_x), float(origin_y)
    p.yaw = float(yaw)
    for i, w in enumerate(weights):
        p.weights[i] = float(w)

    keep: list[Any] = []

    def set_arr(field: str, value: np.ndarray, ints: bool = False):
        a_, ptr = _i(value) if ints else _d(value)
        keep.append(a_)
        setattr(p, field, ptr)

    set_arr("durations", np.asarray(durations))
    set_arr("stance_mask", np.asarray(stance_mask, dtype=np.int32), True)
    set_arr("measured_mask", np.asarray(measured_mask, dtype=np.int32), True)
    set_arr("pin_mask", np.asarray(pin_mask, dtype=np.int32), True)
    set_arr("td_legs", np.asarray(td_legs, dtype=np.int32), True)
    set_arr("measured_feet", np.asarray(measured_feet))
    set_arr("base_pose0", np.asarray(base_pose0))
    set_arr("base_twist0", np.asarray(base_twist0))
    set_arr("inertia9", np.asarray(inertia9).reshape(-1))
    set_arr("hip_offsets12", np.asarray(hip_offsets12).reshape(-1))
    set_arr(
        "previous_footholds",
        np.zeros(12) if previous_footholds is None
        else np.asarray(previous_footholds).reshape(-1),
    )
    set_arr("map_h", np.asarray(map_h))
    set_arr("map_hs1", np.asarray(map_hs1))
    set_arr("map_hs2", np.asarray(map_hs2))
    set_arr("grad_hx", np.asarray(grad_hx))
    set_arr("grad_hy", np.asarray(grad_hy))
    set_arr("grad_s1x", np.asarray(grad_s1x))
    set_arr("grad_s1y", np.asarray(grad_s1y))

    n = num_phases * 6 * num_coeffs + 12 + (num_phases if use_slack else 0)
    return CppProblem(struct=p, keep=keep, x0=None, n=n)


def unpack_x(nlp: Any, x: np.ndarray) -> dict[str, np.ndarray]:
    """C++ x -> [spline_coeffs | desired_footholds | slack] (layout 兼容)."""
    P = nlp.layout.num_phases
    K = nlp.layout.num_coeffs
    cc = P * 6 * K
    out = {
        "spline_coeffs": np.asarray(x[:cc]).reshape(P, 6, K),
        "desired_footholds": np.asarray(x[cc : cc + 12]).reshape(4, 3),
    }
    if nlp.layout.use_slack:
        out["slack"] = np.asarray(x[cc + 12 :])
    return out
