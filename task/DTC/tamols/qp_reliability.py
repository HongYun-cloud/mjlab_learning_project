"""QP 后端可靠性: 失败 fixture 的提取/加载、独立 NumPy 诊断与多后端比较.

背景
----
C++ 源码版 QuadProg++ (Di Gaspero, 经 ``cpp/quadprog_shim.cc``) 在行走
规模 QP (GN Hessian cond ~1e12) 上返回的步长违反自身线性化约束
(lin-viol ~1e-2), 而 python ``quadprog`` 包装版在同输入上 ~2.6e-9.
本模块提供:

- :class:`QPData`: 规范形式的凸 QP (min 0.5 d'Hd + g'd s.t.
  A_eq d = b_eq, A_ineq d <= b_ub), 附元信息 (场景/迭代/维度/后端版本).
- :func:`dump_qp` / :func:`load_qp`: fixture 持久化 (.npz).
- :func:`qp_metrics`: **独立 NumPy** 残差/目标复算 (不经过任何被测
  后端, 也不复用适配器内部检查).
- :func:`diagnose_qp`: 输入健全性 + 谱 + 约束条件数诊断.
- :func:`scale_qp`: 可审计的变量/行缩放 (Jacobi 变量缩放 + 约束行
  单位化), 返回精确正/反变换; 反变换由 :func:`unscale_solution` 完成.
- :func:`solve_qp_with`: 用统一输入调用各后端 (cpp 无缩放 / cpp 缩放 /
  py / qpoases / qrqp / osqp), 返回原问题空间中的解与耗时.

所有比较都在**同一份数据**上进行; 残差一律回到原始 (未缩放) 问题计算.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import casadi as ca
except ImportError:  # pragma: cover
    ca = None

from myrobot_mjlab.task.DTC.tamols.gn_sqp_solver import (
    QuadProgCppQP,
    QuadProgQP,
    GNSQPOptions,
    _build_qp_plugin,
    _Extracted,
)


@dataclass
class QPData:
    H: np.ndarray
    g: np.ndarray
    A_eq: np.ndarray
    b_eq: np.ndarray
    A_ineq: np.ndarray
    b_ub: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.H.shape[0]

    @property
    def p(self) -> int:
        return self.A_eq.shape[0]

    @property
    def m(self) -> int:
        return self.A_ineq.shape[0]


def dump_qp(qp: QPData, path: str | Path) -> None:
    np.savez_compressed(
        path, H=qp.H, g=qp.g, A_eq=qp.A_eq, b_eq=qp.b_eq,
        A_ineq=qp.A_ineq, b_ub=qp.b_ub,
        meta=np.array([str(qp.meta)], dtype=object),
    )


def load_qp(path: str | Path) -> QPData:
    data = np.load(path, allow_pickle=True)
    return QPData(
        H=data["H"], g=data["g"], A_eq=data["A_eq"], b_eq=data["b_eq"],
        A_ineq=data["A_ineq"], b_ub=data["b_ub"],
        meta=dict(eval(str(data["meta"][0]))),  # noqa: S307 - 自产 fixture
    )


# ---------------------------------------------------------------------- #
# 独立 NumPy 复算 (测试与验收使用; 不复用被测代码的残差函数)
# ---------------------------------------------------------------------- #
def qp_metrics(qp: QPData, x: np.ndarray) -> dict[str, float]:
    """原问题空间中的独立复算: 等式残差/不等式违反/目标值/有限性."""
    x = np.asarray(x, dtype=float).ravel()
    eq = float(np.abs(qp.A_eq @ x - qp.b_eq).max()) if qp.p else 0.0
    ineq = float(
        np.maximum(qp.A_ineq @ x - qp.b_ub, 0.0).max()
    ) if qp.m else 0.0
    obj = float(0.5 * x @ qp.H @ x + qp.g @ x)
    return {
        "eq_residual": eq,
        "ineq_violation": ineq,
        "objective": obj,
        "finite": bool(np.isfinite(x).all()),
    }


def kkt_metrics(qp: QPData, x: np.ndarray, active_tol: float = 1e-6) -> dict[str, float]:
    """凸 QP 解的独立 KKT 验证 (NumPy/SciPy, 不复用被测代码).

    KKT 是凸 QP 最优性的充要条件, 因此本函数给出的是**最优性**判据
    (qp_metrics 只给可行性). 乘子从解恢复:
      - 激活集: 不等式 slack <= active_tol 的行;
      - 等式乘子自由 (无符号), 激活不等式乘子 >= 0: 先把残差投影到
        range(A_eq^T) 的正交补, 再用 scipy NNLS 恢复非负乘子;
      - 等式乘子由最小二乘闭式恢复.
    返回 (均为绝对量, 调用方与 ||grad|| 等尺度比较):
      stationarity  = ||Hx + g + A_eq^T λe + A_act^T λi||_inf
      complementarity = max_i λi_i * (A_i x - b_i)   (应 ≈ 0)
      dual_infeasibility = max(0, -min λi)            (应 = 0)
      n_active = 激活行数
    """
    from scipy.optimize import nnls

    x = np.asarray(x, dtype=float).ravel()
    grad = qp.H @ x + qp.g
    slack = qp.b_ub - qp.A_ineq @ x if qp.m else np.zeros(0)
    act = slack <= active_tol if qp.m else np.zeros(0, dtype=bool)
    B = qp.A_ineq[act].T if qp.m else np.zeros((qp.n, 0))
    # 等式乘子可吸收 range(A_eq^T) 内的分量: 残差投影到其正交补
    if qp.p:
        # 投影到 range(A_eq^T) 的正交补: P = I - A_eq^T (A_eq A_eq^T)^+ A_eq
        AAt = qp.A_eq @ qp.A_eq.T
        P = np.eye(qp.n) - qp.A_eq.T @ np.linalg.pinv(AAt) @ qp.A_eq
    else:
        P = np.eye(qp.n)
    M = P @ B
    y = -P @ grad
    if B.shape[1]:
        lam_i, _ = nnls(M, y)
    else:
        lam_i = np.zeros(0)
    # 等式乘子闭式恢复: A_eq^T λe = -(grad + B lam_i)
    if qp.p:
        lam_e = np.linalg.lstsq(qp.A_eq.T, -(grad + B @ lam_i), rcond=None)[0]
    else:
        lam_e = np.zeros(0)
    stat = grad + qp.A_eq.T @ lam_e + B @ lam_i
    comp = float(np.max(lam_i * (qp.A_ineq[act] @ x - qp.b_ub[act]))) \
        if (qp.m and act.any()) else 0.0
    return {
        "stationarity": float(np.abs(stat).max()),
        "complementarity": comp,
        "dual_infeasibility": float(max(0.0, -lam_i.min())) if lam_i.size else 0.0,
        "n_active": int(act.sum()),
        "grad_inf_norm": float(np.abs(grad).max()),
    }


def diagnose_qp(qp: QPData) -> dict[str, float]:
    """输入健全性与谱诊断 (全部 NumPy, 独立于后端)."""
    H = qp.H
    sym_err = float(np.abs(H - H.T).max())
    ev = np.linalg.eigvalsh(0.5 * (H + H.T))
    h_dyn = float(np.abs(H[np.isfinite(H) & (H != 0)]).max()
                  / np.abs(H[H != 0]).min()) if (H != 0).any() else 1.0
    out = {
        "n": qp.n, "p_eq": qp.p, "m_ineq": qp.m,
        "finite_input": bool(
            np.isfinite(H).all() and np.isfinite(qp.g).all()
            and np.isfinite(qp.A_eq).all() and np.isfinite(qp.b_eq).all()
            and np.isfinite(qp.A_ineq).all() and np.isfinite(qp.b_ub).all()
        ),
        "hessian_symmetry_err": sym_err,
        "hessian_eig_min": float(ev.min()),
        "hessian_eig_max": float(ev.max()),
        "hessian_cond": float(ev.max() / max(ev.min(), 1e-300)),
        "hessian_dynamic_range": h_dyn,
    }
    if qp.p:
        sv = np.linalg.svd(qp.A_eq, compute_uv=False)
        out["a_eq_svd_max"] = float(sv.max())
        out["a_eq_svd_min"] = float(sv.min())
        out["a_eq_cond"] = float(sv.max() / max(sv.min(), 1e-300))
    if qp.m:
        row = np.linalg.norm(qp.A_ineq, axis=1)
        out["a_ineq_rownorm_min"] = float(row.min())
        out["a_ineq_rownorm_max"] = float(row.max())
        out["a_ineq_rownorm_spread"] = float(row.max() / max(row.min(), 1e-300))
    return out


# ---------------------------------------------------------------------- #
# 可审计的缩放 (变量 Jacobi + 约束行单位化)
# ---------------------------------------------------------------------- #
@dataclass
class QPScale:
    d_var: np.ndarray   # d = d_var * z  (变量缩放)
    r_eq: np.ndarray    # 行缩放 (等式)
    r_ineq: np.ndarray  # 行缩放 (不等式)

    def describe(self) -> dict[str, Any]:
        return {
            "var_scale_min": float(self.d_var.min()),
            "var_scale_max": float(self.d_var.max()),
            "var_scale_spread": float(self.d_var.max() / self.d_var.min()),
            "eq_rowscale_min": float(self.r_eq.min()) if self.r_eq.size else None,
            "eq_rowscale_max": float(self.r_eq.max()) if self.r_eq.size else None,
            "ineq_rowscale_min": float(self.r_ineq.min()) if self.r_ineq.size else None,
            "ineq_rowscale_max": float(self.r_ineq.max()) if self.r_ineq.size else None,
        }


def scale_qp(qp: QPData, row_scaling: bool = True) -> tuple[QPData, QPScale]:
    """Jacobi 变量缩放 + (可选) 约束行单位化.

    变量缩放 ``d = D z`` (D = diag(1/sqrt(diag(H)))) 使 H' = D H D 的
    对角为 1; 约束行缩放 ``R`` 只乘约束行 (A'' = R A D, b'' = R b),
    不改变可行集与最优解. 反变换: z* -> d* = D z* (行缩放不作用于解).
    """
    diag = np.diag(qp.H).copy()
    diag[diag <= 0] = np.abs(diag).max() if (np.abs(diag) > 0).any() else 1.0
    d_var = 1.0 / np.sqrt(diag)
    H2 = qp.H * np.outer(d_var, d_var)
    g2 = qp.g * d_var
    A_eq2, b_eq2 = qp.A_eq * d_var[None, :], qp.b_eq.copy()
    A_in2, b_ub2 = qp.A_ineq * d_var[None, :], qp.b_ub.copy()
    r_eq = np.ones(qp.p)
    r_ineq = np.ones(qp.m)
    if row_scaling:
        if qp.p:
            r_eq = 1.0 / np.maximum(
                np.abs(A_eq2).max(axis=1), 1e-300
            )
            A_eq2 = A_eq2 * r_eq[:, None]
            b_eq2 = b_eq2 * r_eq
        if qp.m:
            r_ineq = 1.0 / np.maximum(
                np.abs(A_in2).max(axis=1), 1e-300
            )
            A_in2 = A_in2 * r_ineq[:, None]
            b_ub2 = b_ub2 * r_ineq
    scaled = QPData(H=H2, g=g2, A_eq=A_eq2, b_eq=b_eq2,
                    A_ineq=A_in2, b_ub=b_ub2, meta=dict(qp.meta))
    return scaled, QPScale(d_var=d_var, r_eq=r_eq, r_ineq=r_ineq)


def unscale_solution(z: np.ndarray, scale: QPScale) -> np.ndarray:
    """缩放后问题的解 z* -> 原问题解 d* = D z*."""
    return scale.d_var * np.asarray(z, dtype=float).ravel()


# ---------------------------------------------------------------------- #
# 统一后端调用 (同一输入, 返回原问题空间解)
# ---------------------------------------------------------------------- #
_BACKENDS = ("quadprog_cpp", "quadprog_cpp_raw",
             "quadprog", "qpoases", "qrqp", "osqp")


def _dummy_extracted(qp: QPData) -> _Extracted:
    return _Extracted(
        inputs=[], dvars=None, f_eval=None, rec_eval=None,
        n_rec=0, n_ent=0, n=qp.n, m_ineq=qp.m, m_eq=qp.p,
        entry_kinds=[], entry_ubs=[0.0] * qp.m,
    )


def _to_bounds(qp: QPData):
    A = np.vstack([qp.A_ineq, qp.A_eq]) if (qp.m + qp.p) else np.zeros((0, qp.n))
    lba = np.concatenate(
        [np.full(qp.m, -np.inf), qp.b_eq]
    ) if qp.p else np.full(qp.m, -np.inf)
    uba = np.concatenate([qp.b_ub, qp.b_eq]) if qp.p else qp.b_ub
    return A, lba, uba


def solve_qp_with(
    qp: QPData, backend: str, *, qp_options: GNSQPOptions | None = None,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """用指定后端解 QP; 返回 (原问题空间解, 耗时 s, 信息).

    后端语义 (同一 QPData 输入):
      - ``quadprog_cpp``: C++ 适配器, **默认带 Jacobi 变量缩放** (修复后);
      - ``quadprog_cpp_raw``: 关闭缩放 (复现原始失败行为, 仅诊断);
      - ``quadprog`` / ``qpoases`` / ``qrqp`` / ``osqp``: 原样求解.
    所有结果都在**原问题**空间做独立残差检查 (调用方用 qp_metrics).
    """
    import time

    opts = qp_options or GNSQPOptions()
    ex = _dummy_extracted(qp)

    if backend == "quadprog_cpp":
        inst = QuadProgCppQP(var_scaling=True)
    elif backend == "quadprog_cpp_raw":
        inst = QuadProgCppQP(var_scaling=False)
    elif backend == "quadprog":
        inst = QuadProgQP()
    else:
        inst = _build_qp_plugin(ex, backend, opts)
    A, lba, uba = _to_bounds(qp)
    t0 = time.perf_counter()
    x = np.asarray(inst(
        h=qp.H, g=qp.g, a=A, lba=lba, uba=uba,
    )["x"]).ravel()
    dt = time.perf_counter() - t0
    return x, dt, {"backend": backend}
