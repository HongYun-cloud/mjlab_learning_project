"""高斯-牛顿 + SQP 第二求解路径 (论文 V-C.1), 与 IPOPT 并存用于交叉验证.

设计纪律
--------
- **不替换**原求解器: :meth:`IntegratedNLP.solve` 仍走 CasADi+IPOPT;
  本模块提供 :func:`solve_gn_sqp`, 输入同一个 :class:`IntegratedNLP`,
  输出同一种 :class:`SolveReport`, 因此 ``trajectory_audit`` /
  ``trajectory_export`` / ``compare_solutions`` 等下游工具对两条路径
  通用, 交叉验证只需比较两份报告.
- 论文 V-C.1 路线: 目标为加权平方和, 用高斯-牛顿提取正定部分
  ``H = Σ_k 2 w_k ∇s_k ∇s_k^T`` (s_k 为 :attr:`ObjectiveRecord.roots`
  根残差); 线性项 (式 29 的 eps) 无根残差, 曲率按 0, 梯度精确进入 QP
  线性项; 每步解一个 QP (默认 QuadProg++ 的 python 包装, 备选 casadi
  conic: qpoases/qrqp/osqp), 再用
  精确惩罚函数 + Armijo 回溯线搜索全球化.
- QP 约束: 不等式/等式残差的一阶线性化 ``g + J_g d <= 0``,
  ``h + J_h d = 0`` (与论文 SQP 一致); GIAC/摩擦等约束的**非线性**
  满足性只在线搜索与终止判据处用真值检查, 不用线性化值冒充.

能验证什么 / 不能验证什么
------------------------
能: 同一 NLP 的另一个局部解路径 (KKT 点一致性、目标值/落脚点/轨迹
差异); GN 正定近似的数值行为; QP 子问题可行性.
不能: 全局最优性 (SQP 与 IPOPT 都是局部方法); 论文 QuadProg++ 实现的
位级复现 (本模块用 CasADi 内置 QP 求解器).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

try:
    import casadi as ca
except ImportError:  # pragma: no cover
    ca = None

from myrobot_mjlab.task.DTC.tamols.giac_constraints import EQUALITY
from myrobot_mjlab.task.DTC.tamols.nlp_integration import (
    ConstraintLine,
    IntegratedNLP,
    ObjectiveLine,
    SolveReport,
)


@dataclass
class GNSQPOptions:
    max_iter: int = 60
    tol_step: float = 1e-9
    tol_viol: float = 1e-6
    tol_stationary: float = 1e-9
    """线搜索拒绝时, 若可行且目标相对上次被接受迭代的改变 <= 该相对量,
    判为数值驻点收敛."""
    qp_lin_tol: float = 1e-6
    """QP 解校验容差: QP 返回的 d 必须满足线性化约束到该精度, 否则视为
    该 QP 后端数值失败, 本步改用备选后端重解 (见 solve_gn_sqp)."""
    jit: bool = False
    """把求值函数编译为 C (数学不变): 全量求值 2.5->0.08 ms.
    编译 ~30 s/图, 仅在 extract 缓存未命中时发生; 默认关 (测试套件
    用纯 Python 路径), 快路径显式开启."""
    use_extract_cache: bool = True
    """复用 ``IntegratedNLP.cached_extract`` 的提取结果 (结构不变时
    符号构建与 Jacobian 求值零成本)."""
    merit_rho: float = 1e3
    armijo_eta: float = 1e-4
    t_min: float = 1e-4
    hessian_reg: float = 1e-8
    qp_plugin: str = "quadprog_cpp"
    """默认 QuadProg++ C++ 实现 (ctypes shim, 论文 V-C.1 同款算法);
    备选链: python quadprog 包装 -> qpoases -> qrqp -> osqp.
    更换 QP 实现只替换适配器类, SQP 主循环不变."""
    qp_fallbacks: tuple[str, ...] = ("quadprog", "qpoases", "qrqp", "osqp")
    cpp_var_scaling: bool = True
    """QuadProg++ C++ 后端的 Jacobi 变量缩放 (根因修复, 见
    :class:`QuadProgCppQP` 文档); False 用于复现原始失败行为."""


@dataclass
class _Extracted:
    inputs: list
    dvars: Any
    f_eval: Any          # Function(x) -> [f, g, h, gf, Jg, Jh, roots, Jr]
    rec_eval: Any        # Function(x) -> [record residuals..., entry residuals...]
    n_rec: int
    n_ent: int
    n: int
    m_ineq: int
    m_eq: int
    entry_kinds: list[str]
    entry_ubs: list[float]
    v_eval: Any = None   # Function(x) -> [f, g, h] (线搜索轻量路径)


def _cvec(var: Any) -> Any:
    """矩阵变量 -> 列向量, C 序展平 (与 IntegratedNLPLayout.pack 一致)."""
    shape = tuple(int(s) for s in var.shape)
    if len(shape) == 2:
        return ca.reshape(ca.transpose(var), shape[0] * shape[1], 1)
    return ca.reshape(var, shape[0], 1)


def _block_from_vector(vec: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    return np.asarray(vec, dtype=float).reshape(shape)


def _unflatten(vec: Any, shape: tuple[int, ...]) -> Any:
    """C 序向量 -> 矩阵表达式 (与 _cvec 互逆)."""
    if len(shape) == 2:
        return ca.transpose(ca.reshape(vec, (shape[1], shape[0])))
    return ca.reshape(vec, (shape[0], 1))


def extract_nlp(nlp: IntegratedNLP, jit: bool = False) -> _Extracted:
    """从 IntegratedNLP 的符号记录中提取 NLP (f, g, h, 根残差, 雅可比).

    所有表达式最终替换到单一原始符号向量 ``x`` (C 序, 与
    ``IntegratedNLPLayout.pack`` 一致), 以便 ``ca.gradient`` / 数值迭代.

    ``jit=True`` 时把全量/轻量求值函数编译为 C (一次性编译成本, 配合
    :meth:`IntegratedNLP.cached_extract` 摊销); 求值结果与非 JIT 数学
    一致 (同一表达式图, 仅编译执行), 由
    ``test_gn_sqp_fast`` 与 Python 离线版交叉验证. 含地图插值的图在
    MX 层展开 (expand) 反而慢 ~6x, 故不用 SX 展开.
    """
    if nlp.objective_set is None:
        raise RuntimeError("extract_nlp 前必须先 add_objectives()")
    inputs: list[Any] = list(nlp.vars["spline_coeffs"]) + [
        nlp.vars["desired_footholds"]
    ]
    if nlp.vars["slack"] is not None:
        inputs.append(nlp.vars["slack"])
    shapes = [tuple(int(s) for s in v.shape) for v in inputs]
    sizes = [int(np.prod(s)) for s in shapes]
    n = sum(sizes)
    xs = ca.MX.sym("x", n)
    repl_new = []
    off = 0
    for shape, size in zip(shapes, sizes):
        repl_new.append(_unflatten(xs[off:off + size], shape))
        off += size

    records = nlp.objective_set.records
    f = sum((r.weighted_residual() for r in records), 0.0)
    roots_parts = []
    for r in records:
        if r.roots:
            scale = float(np.sqrt(2.0 * r.weight))
            roots_parts.append(ca.vertcat(*[scale * s for s in r.roots]))
    roots = (
        ca.vertcat(*roots_parts) if roots_parts else ca.DM.zeros(0, 1)
    )

    ineq = [e.residual for e in nlp.constraint_entries if e.kind != EQUALITY]
    eq = [e.residual for e in nlp.constraint_entries if e.kind == EQUALITY]
    ubs = [e.ub for e in nlp.constraint_entries if e.kind != EQUALITY]
    kinds = [e.kind for e in nlp.constraint_entries]
    g = ca.vertcat(*ineq) if ineq else ca.DM.zeros(0, 1)
    h = ca.vertcat(*eq) if eq else ca.DM.zeros(0, 1)

    # 全部表达式映射到原始符号向量 xs (gradient/jacobian 要求纯符号自变量)
    f, g, h, roots = ca.substitute([f, g, h, roots], inputs, repl_new)
    dvars = xs
    gf = ca.gradient(f, dvars)
    Jg = ca.jacobian(g, dvars) if ineq else ca.DM.zeros(0, n)
    Jh = ca.jacobian(h, dvars) if eq else ca.DM.zeros(0, n)
    Jr = ca.jacobian(roots, dvars) if roots_parts else ca.DM.zeros(0, n)

    jit_opts = {
        "jit": True,
        "jit_options": {
            "flags": ["-O2"],
            "verbose": False,
        },
    } if jit else {}

    def _fun(name, outs):
        try:
            return ca.Function(name, [xs], outs, jit_opts)
        except Exception:
            # JIT 不可用 (工具链/图不支持): 回退非 JIT, 数学不变
            return ca.Function(name, [xs], outs)

    f_eval = _fun("gn_sqp_eval", [f, g, h, gf, Jg, Jh, roots, Jr])
    # 轻量值函数: 线搜索 trial 只需要 f/g/h (无 Jacobian)
    v_eval = _fun("gn_sqp_value", [f, g, h])
    rec_res = [r.residual for r in records]
    ent_res = [e.residual for e in nlp.constraint_entries]
    rec_res, ent_res = ca.substitute(
        [ca.vertcat(*rec_res), ca.vertcat(*ent_res)], inputs, repl_new
    )
    rec_eval = ca.Function("gn_sqp_records", [xs], [rec_res, ent_res])
    return _Extracted(
        inputs=inputs, dvars=dvars, f_eval=f_eval, v_eval=v_eval,
        rec_eval=rec_eval,
        n_rec=len(records), n_ent=len(nlp.constraint_entries), n=n,
        m_ineq=len(ineq), m_eq=len(eq), entry_kinds=kinds, entry_ubs=ubs,
    )


def _default_shim_path() -> str:
    import os
    from pathlib import Path

    env = os.environ.get("QUADPROG_SHIM_LIB")
    if env:
        return env
    return str(Path(__file__).parent / "cpp" / "libquadprog_shim.so")


class QuadProgCppQP:
    """QuadProg++ **C++** 适配器 (ctypes 调用本地编译的 shim).

    shim 源码与构建脚本见 ``cpp/quadprog_shim.cc`` / ``cpp/build_shim.sh``
    (编译自 Di Gaspero 的 QuadProg++, 与 python ``quadprog`` 包装同一
    Goldfarb-Idnani 算法). 调用约定与 :class:`QuadProgQP` 完全相同, 因此
    SQP 主循环与测试对两者无差别; 库缺失时 ``__init__`` 抛
    ``FileNotFoundError``, 由 :func:`_make_qp` 落入备选链.

    变量缩放 (根因修复, 默认开启)
    ------------------------------
    Di Gaspero 源码版的双重方法对 ill-conditioned Hessian 数值脆弱:
    TAMOLS 的 GN Hessian 因 Table I 权重跨 1e-3..1e4, 元素动态范围
    ~8e25, cond~2e12, 此时 C++ 版返回的解违反自身线性化约束 (~1e-2,
    python 包装版同输入 ~1e-9; fixture 见 ``test/fixtures/qp_crawl_flat
    _iter0.npz`` 与 ``test_qp_reliability``). 修复: 对 QP 做 Jacobi 变量
    缩放 ``d = D z, D = diag(1/sqrt(diag(H)))``, 使缩放问题的 Hessian
    对角为 1::

        min 0.5 z' (D H D) z + (D g)' z  s.t. (A D) z 与原约束同界

    D 可逆 => 数学严格等价; 解反变换 ``d = D z`` 后在本适配器**原问题
    空间**返回. 行缩放经消融评估会使该后端更差 (单独记录于
    ``qp_reliability``), 不采用; 无正则化 (不引入阻尼偏差).
    ``var_scaling=False`` 可关闭缩放 (用于复现原始失败行为).
    """

    name = "quadprog_cpp"

    def __init__(self, lib_path: str | None = None, var_scaling: bool = True):
        import ctypes

        path = lib_path or _default_shim_path()
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"QuadProg++ shim 未编译: {path}; 运行 cpp/build_shim.sh"
            )
        self._lib = ctypes.CDLL(path)
        self._lib.qp_solve_cpp.restype = ctypes.c_int
        self._lib.qp_solve_cpp.argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 8
        self._ctypes = ctypes
        self.lib_path = path
        self.var_scaling = bool(var_scaling)

    def __call__(self, h=None, g=None, a=None, lba=None, uba=None, **_):
        np_ = np
        H = np_.ascontiguousarray(h, dtype=np.float64)
        gv = np_.asarray(g, dtype=np.float64).ravel()
        A = np_.asarray(a, dtype=np.float64).reshape(-1, H.shape[0])
        lb = np_.asarray(lba, dtype=np.float64).ravel()
        ub = np_.asarray(uba, dtype=np.float64).ravel()
        d_var = None
        if self.var_scaling and H.shape[0]:
            diag = np_.diag(H).copy()
            if not np_.all(diag > 0):
                fallback = float(np_.max(np_.abs(diag))) or 1.0
                diag[diag <= 0] = fallback
            d_var = 1.0 / np_.sqrt(diag)
            H = H * np_.outer(d_var, d_var)
            gv = gv * d_var
            A = A * d_var[None, :]
        ce_rows, ce0, ci_rows, ci0 = [], [], [], []
        for i in range(A.shape[0]):
            if lb[i] == ub[i]:
                ce_rows.append(A[i])
                ce0.append(-ub[i])  # CE d + ce0 = 0, ce0 = -b
            else:
                if np_.isfinite(ub[i]):
                    ci_rows.append(-A[i])
                    ci0.append(ub[i])  # -A d + ub >= 0
                if np_.isfinite(lb[i]):
                    ci_rows.append(A[i])
                    ci0.append(-lb[i])  # A d - lb >= 0
        n = H.shape[0]
        # 注: QuadProg++ (含本 shim) 对 0 行等式/不等式可直接处理 (已测),
        # 不需要占位约束; 占位零行反而触发其内部异常.
        CE = (
            np_.ascontiguousarray(np_.stack(ce_rows), dtype=np.float64)
            if ce_rows else np_.zeros((0, n), dtype=np.float64)
        )
        CI = (
            np_.ascontiguousarray(np_.stack(ci_rows), dtype=np.float64)
            if ci_rows else np_.zeros((0, n), dtype=np.float64)
        )
        ce0v = np_.ascontiguousarray(ce0, dtype=np.float64)
        ci0v = np_.ascontiguousarray(ci0, dtype=np.float64)
        x = np_.zeros(n, dtype=np.float64)
        cost = np_.zeros(1, dtype=np.float64)

        def ptr(arr):
            return arr.ctypes.data_as(self._ctypes.c_void_p)

        rc = self._lib.qp_solve_cpp(
            n, CE.shape[0], CI.shape[0],
            ptr(H), ptr(np_.ascontiguousarray(gv)), ptr(CE), ptr(ce0v),
            ptr(CI), ptr(ci0v), ptr(x), ptr(cost),
        )
        if rc == 1:
            raise ValueError("quadprog_cpp: QP infeasible (cost=inf)")
        if rc != 0:
            raise RuntimeError(f"quadprog_cpp: shim 返回错误码 {rc}")
        if d_var is not None:
            x = d_var * x  # 反变换回原问题空间 (行缩放不作用于解)
        return {"x": x}


class QuadProgQP:
    """QuadProg++ (python ``quadprog`` 包装) 适配器.

    把本模块的 QP 调用约定 (与 casadi conic 一致)::

        min 0.5 d'H d + g'd   s.t.  lba <= A d <= uba

    映射到 ``quadprog.solve_qp`` 的约定::

        min 0.5 x'G x - a'x   s.t.  C'x >= b, 前 meq 列为等式

    映射规则: 等式行 (lba==uba) 放最前 (meq); 上界行取 ``-A_i d >= -ub_i``;
    有限下界行取 ``A_i d >= lb_i``. G 必须正定: 调用方保证
    ``H = Jr'Jr + reg I``. 不可行 QP 时 ``solve_qp`` 抛 ValueError, 由
    SQP 主循环按 qp_failed 处理.

    将来换 C++ QuadProg++ (pybind11 / 直接链接 libQuadProg) 时, 只需
    替换本类的 ``__call__`` 实现, 保持输入输出约定不变.
    """

    name = "quadprog"

    def __call__(self, h=None, g=None, a=None, lba=None, uba=None, **_):
        import quadprog

        H = np.asarray(h, dtype=float)
        gv = np.asarray(g, dtype=float).ravel()
        A = np.asarray(a, dtype=float).reshape(-1, H.shape[0])
        lb = np.asarray(lba, dtype=float).ravel()
        ub = np.asarray(uba, dtype=float).ravel()
        cols: list[np.ndarray] = []
        bs: list[float] = []
        meq = 0
        for i in range(A.shape[0]):  # 等式在前
            if lb[i] == ub[i]:
                cols.append(A[i])
                bs.append(lb[i])
                meq += 1
        for i in range(A.shape[0]):
            if lb[i] == ub[i]:
                continue
            if np.isfinite(ub[i]):
                cols.append(-A[i])
                bs.append(-ub[i])
            if np.isfinite(lb[i]):
                cols.append(A[i])
                bs.append(lb[i])
        n = H.shape[0]
        C = np.stack(cols, axis=1) if cols else np.zeros((n, 0))
        b = np.array(bs, dtype=float) if bs else np.zeros(0)
        x, _f, _xu, _iters, _lagr, _iact = quadprog.solve_qp(
            H, -gv, C, b, meq=meq
        )
        return {"x": np.asarray(x, dtype=float).ravel()}


def _build_qp_plugin(ex: _Extracted, plugin: str, opts: GNSQPOptions):
    """按名字构造单个 QP 后端 (统一调用约定); 不可用则抛异常."""
    n, m = ex.n, ex.m_ineq + ex.m_eq
    if plugin == "quadprog_cpp":
        return QuadProgCppQP(var_scaling=opts.cpp_var_scaling)
    if plugin == "quadprog":
        import quadprog  # noqa: F401

        return QuadProgQP()
    common = {"print_time": False, "print_in": False, "print_out": False}
    extras = {
        "qpoases": {"print_level": "none"},
        "osqp": {"verbose": False},
    }
    dims = {"h": ca.Sparsity.dense(n, n), "a": ca.Sparsity.dense(m, n)}
    last_err = None
    for opt in (dict(common, **extras.get(plugin, {})), common):
        try:
            return ca.conic("gn_sqp_qp", plugin, dims, opt)
        except Exception as exc:  # 插件缺失或选项不被接受
            last_err = exc
    raise RuntimeError(f"QP 插件 {plugin} 不可用: {last_err}")


def _make_qp(ex: _Extracted, opts: GNSQPOptions):
    last_err = None
    for plugin in (opts.qp_plugin, *opts.qp_fallbacks):
        try:
            return _build_qp_plugin(ex, plugin, opts), plugin
        except Exception as exc:
            last_err = exc
    raise RuntimeError(
        f"没有可用的 QP 求解器插件 (尝试 {opts.qp_plugin} + "
        f"{opts.qp_fallbacks}): {last_err}"
    )


def solve_gn_sqp(
    nlp: IntegratedNLP,
    initial_blocks: Sequence[np.ndarray],
    options: GNSQPOptions | None = None,
) -> SolveReport:
    """用 GN+SQP 求解 ``nlp`` 描述的同一个 NLP, 返回与 IPOPT 同构的报告.

    Args:
      initial_blocks: 与 ``nlp.vars`` 顺序一致的初值块
        ``[coeffs_0, ..., coeffs_{P-1}, desired, (slack)]``.
    """
    opts = options or GNSQPOptions()
    t0 = time.perf_counter()
    if opts.use_extract_cache and hasattr(nlp, "cached_extract"):
        ex = nlp.cached_extract(opts.jit)
    else:
        ex = extract_nlp(nlp, jit=opts.jit)
    blocks = [np.asarray(b, dtype=float) for b in initial_blocks]
    if len(blocks) != len(ex.inputs):
        raise ValueError(
            f"initial_blocks 个数 {len(blocks)} 与变量块 {len(ex.inputs)} 不一致"
        )
    x = np.concatenate([b.reshape(-1) for b in blocks])  # C 序, 同 layout.pack
    qp, qp_plugin = _make_qp(ex, opts)
    qp_cache = {qp_plugin: qp}
    qp_fallback_notes: list[str] = []

    def lin_viol(dd, gv, hv, Jgv, Jhv):
        """QP 解 d 的线性化约束违反 (QP 自身约束的满足度自检)."""
        a = 0.0
        if Jgv.size:
            vi = Jgv @ dd - (np.array(ex.entry_ubs[: ex.m_ineq]) - gv)
            a = float(np.maximum(vi, 0.0).max())
        b = 0.0
        if Jhv.size and hv.size:
            b = float(np.abs(Jhv @ dd + hv).max())
        return max(a, b)

    def eval_at(xv):
        out = ex.f_eval(xv)
        return (
            float(out[0]),
            np.asarray(out[1]).ravel(),
            np.asarray(out[2]).ravel(),
            np.asarray(out[3]).ravel(),
            np.asarray(out[4]),
            np.asarray(out[5]),
            np.asarray(out[7]),
        )

    def value_at(xv):
        """轻量求值 (线搜索): 只取 f/g/h."""
        v = ex.v_eval if ex.v_eval is not None else ex.f_eval
        out = v(xv)
        return (
            float(out[0]),
            np.asarray(out[1]).ravel(),
            np.asarray(out[2]).ravel(),
        )

    def violation(gv, hv):
        vi = np.maximum(gv - np.array(ex.entry_ubs[: ex.m_ineq]), 0.0)
        return float(vi.sum() + np.abs(hv).sum())

    def merit(fv, gv, hv):
        return fv + opts.merit_rho * violation(gv, hv)

    f, g, h, gf, Jg, Jh, Jr = eval_at(x)
    prev_f = f
    status = "GN-SQP: max_iter"
    iterations = 0
    for it in range(opts.max_iter):
        iterations = it + 1
        H = Jr.T @ Jr + opts.hessian_reg * np.eye(ex.n)
        A = np.vstack([Jg, Jh]) if (ex.m_ineq + ex.m_eq) else np.zeros((0, ex.n))
        # 不等式: g + Jg d <= ub  ->  uba = ub - g; 等式: h + Jh d = 0 -> -h
        if ex.m_eq:
            lba = np.concatenate([np.full(ex.m_ineq, -np.inf), -h])
            uba = np.concatenate([np.array(ex.entry_ubs[: ex.m_ineq]) - g, -h])
        else:
            lba = np.full(ex.m_ineq, -np.inf)
            uba = np.array(ex.entry_ubs[: ex.m_ineq]) - g
        try:
            sol = qp(
                h=ca.DM(H), g=ca.DM(gf), a=ca.DM(A),
                lba=ca.DM(lba), uba=ca.DM(uba),
            )
            d = np.asarray(sol["x"]).ravel()
        except Exception:
            status = "GN-SQP: qp_failed"
            break
        if not np.isfinite(d).all():
            status = "GN-SQP: qp_nonfinite"
            break
        lv = lin_viol(d, g, h, Jg, Jh)
        if lv > opts.qp_lin_tol:
            # QP 后端数值失败 (例如老版 QuadProg++ 源码在 cond(H)~1e12 时
            # 返回不满足自身约束的解). 逐级补救: 同后端加大 LM 阻尼重试
            # (不改变定点, 只改善条件数), 仍失败再换备选后端.
            replaced = None
            candidates: list[tuple[str, Any, np.ndarray]] = []
            # 优先: 无阻尼的备选后端 (步长语义不变);
            # 兜底: 同后端加大 LM 阻尼 (仅在无备选可用时).
            for alt in (opts.qp_plugin, *opts.qp_fallbacks):
                if alt == qp_plugin:
                    continue
                inst = qp_cache.get(alt)
                if inst is None:
                    try:
                        inst = _build_qp_plugin(ex, alt, opts)
                    except Exception:
                        continue
                    qp_cache[alt] = inst
                candidates.append((alt, inst, H))
            for mult in (1e2, 1e4):
                candidates.append(
                    (
                        f"{qp_plugin}[reg*{mult:g}]",
                        qp,
                        Jr.T @ Jr + (opts.hessian_reg * mult) * np.eye(ex.n),
                    )
                )
            for label, inst, H_use in candidates:
                try:
                    d2 = np.asarray(
                        inst(
                            h=ca.DM(H_use), g=ca.DM(gf), a=ca.DM(A),
                            lba=ca.DM(lba), uba=ca.DM(uba),
                        )["x"]
                    ).ravel()
                except Exception:
                    continue
                if np.isfinite(d2).all() and lin_viol(d2, g, h, Jg, Jh) <= opts.qp_lin_tol:
                    d = d2
                    replaced = label
                    break
            if replaced is None:
                status = "GN-SQP: qp_failed"
                break
            qp_fallback_notes.append(
                f"iter {it}: QP {qp_plugin} 线性化违反 {lv:.2e} > "
                f"{opts.qp_lin_tol:.0e}, 本步改用 {replaced}"
            )
        step_inf = float(np.max(np.abs(d)))
        viol_now = violation(g, h)
        if step_inf < opts.tol_step:
            # QP 给出零步: 已是一阶驻点. 可行则收敛, 否则是驻点不可行.
            status = (
                "GN-SQP: converged"
                if viol_now < opts.tol_viol
                else "GN-SQP: stationary_infeasible"
            )
            break
        d0 = float(gf @ d) - opts.merit_rho * viol_now
        phi0 = merit(f, g, h)
        t = 1.0
        accepted = False
        trial = x
        while t >= opts.t_min:
            trial = x + t * d
            ft, gt, ht = value_at(trial)
            if merit(ft, gt, ht) <= phi0 + opts.armijo_eta * t * d0:
                accepted = True
                break
            t *= 0.5
        if not accepted:
            # 线搜索拒绝: 若当前点可行且目标相对上一次被接受的迭代已无
            # 变化 (数值驻点), 判收敛; 否则才是真正的停滞.
            if viol_now < opts.tol_viol and abs(f - prev_f) <= (
                opts.tol_stationary * max(1.0, abs(f))
            ):
                status = "GN-SQP: converged"
                break
            status = "GN-SQP: line_search_stalled"
            break
        prev_f = f
        x = trial
        f, g, h, gf, Jg, Jh, Jr = eval_at(x)
        if violation(g, h) < opts.tol_viol and (
            step_inf * t < opts.tol_step or d0 >= -opts.tol_step
        ):
            status = "GN-SQP: converged"
            break
    else:
        status = "GN-SQP: max_iter"
    if status == "GN-SQP: converged" and violation(g, h) > opts.tol_viol:
        status = "GN-SQP: converged_with_violation"

    dt = time.perf_counter() - t0
    rec_vals, ent_vals = [np.asarray(v).ravel() for v in ex.rec_eval(x)]
    success = status == "GN-SQP: converged"
    solution = nlp.layout.unpack(x)
    report = _build_report(
        nlp, solution, rec_vals, ent_vals, status, success, iterations, dt,
        qp_plugin,
    )
    if qp_fallback_notes:
        shown = qp_fallback_notes[:10]
        extra = (
            f"... 共 {len(qp_fallback_notes)} 次 QP 回退"
            if len(qp_fallback_notes) > 10
            else ""
        )
        report.warnings = tuple(report.warnings) + tuple(shown) + ((extra,) if extra else ())
    return report


def _build_report(
    nlp: IntegratedNLP,
    solution: dict[str, np.ndarray],
    rec_vals: np.ndarray,
    ent_vals: np.ndarray,
    status: str,
    success: bool,
    iterations: int,
    seconds: float,
    qp_plugin: str,
) -> SolveReport:
    records = nlp.objective_set.records
    lines = []
    total = 0.0
    weighted = []
    for rec, val in zip(records, rec_vals):
        wv = rec.weight * float(val)
        weighted.append(wv)
        total += wv
    for rec, val, wv in zip(records, rec_vals, weighted):
        lines.append(
            ObjectiveLine(
                rec.name, rec.category, rec.phase, rec.tau, float(val),
                rec.weight, wv, (wv / total if total else 0.0),
            )
        )
    con_lines = []
    by_cat: dict[str, float] = {}
    for entry, val in zip(nlp.constraint_entries, ent_vals):
        arr = np.asarray(val, dtype=float).reshape(-1)
        viol = entry.violation(arr)
        con_lines.append(
            ConstraintLine(
                entry.name, entry.category, entry.kind,
                float(arr[0]) if arr.size else 0.0, viol, entry.phase,
                entry.tau, entry.source,
            )
        )
        by_cat[entry.category] = max(by_cat.get(entry.category, 0.0), viol)
    warnings = list(nlp.assembly_warnings)
    warnings.extend(nlp.check_finite(solution))
    return SolveReport(
        scenario=f"{status}",
        status=status,
        success=success,
        iterations=iterations,
        solve_seconds=seconds,
        layout_text=nlp.layout.describe(),
        assumptions_text=nlp.assumptions.describe(),
        objective_total=float(total),
        objective_lines=tuple(lines),
        constraint_lines=tuple(con_lines),
        max_violation_by_category=by_cat,
        variable_ranges={
            k: (float(v.min()), float(v.max())) for k, v in solution.items()
        },
        warnings=tuple(warnings),
        solution=solution,
        versions={"casadi": ca.__version__, "qp": qp_plugin},
        solution_source="gn_sqp_solution" if success else "gn_sqp_last_iterate",
        expected_dec_dim=nlp.layout.total_size,
    )
