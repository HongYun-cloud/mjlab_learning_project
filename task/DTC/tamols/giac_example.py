"""TAMOLS GIAC + 摩擦约束的最小可运行 NLP (CasADi + IPOPT).

这个脚本**只**验证约束代码, 不是完整的 TAMOLS 轨迹优化器:

- 固定机身位置 ``p_B`` 和 3 个支撑脚位置;
- 只把机身加速度 ``p_B_ddot`` 作为决策变量;
- 默认目标函数取 ``||p_B_ddot - p_ref||^2``, 其中 ``p_ref`` 是一个严格
  可行点; 这只是为了让 IPOPT 有一个非退化的最优点、方便检查约束,
  不代表论文的完整目标函数. 也可以切换成 ``||p_B_ddot||^2``;
- 加入多足 GIAC 不等式和水平地面摩擦锥不等式, 用 IPOPT 求解;
- 打印求解器状态、解、每条约束的数值残差.

安装依赖 (CasADi 是可选依赖, 未安装时本模块仍可导入, 单元测试会跳过
IPOPT 部分)::

    uv pip install casadi

或者使用项目虚拟环境::

    uv pip install --python /home/hong/myrobot_mjlab/.venv/bin/python casadi

运行::

    python -m myrobot_mjlab.task.DTC.tamols.giac_example

关于解的物理解释
----------------
本模块严格实现用户给出的符号约定 ``a_B = g - p_B_ddot`` 和
``||a_B_xy|| <= mu * a_B_z``. 该约定下, 摩擦锥要求 ``a_B_z >= 0``,
也就是 ``p_B_ddot_z <= -9.81`` (向下加速至少 1g). 若使用
``objective="norm"``, 最优解会被推到 ``p_B_ddot_z = -9.81`` 的边界,
对应 ``a_B = 0``, 此时平方形式的摩擦锥在原点处退化; 默认的
``objective="reference"`` 则停在严格可行点 ``p_ref``, 数值上更稳.
无论哪种目标, 都只说明约束代码可行, 并不代表常见的站立/行走工况.
仅就摩擦锥而言,
若把 ``a_B`` 整体取负 (等价于 ``a_B = p_B_ddot - g``), 静止时
``a_B_z = +9.81``, 才对应通常的支撑力方向; 但 GIAC 的符号和不等式
方向也必须一起核对, 不能只改其中一条. 详见
:mod:`myrobot_mjlab.task.DTC.tamols.giac_constraints` 的模块文档.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from myrobot_mjlab.task.DTC.tamols.giac_constraints import (
    add_friction_constraint,
    add_giac_constraints,
    friction_residual,
    friction_squared_residual,
    giac_residuals,
    gravity_inertial_acceleration,
)

try:  # CasADi 是可选依赖; 未安装时仍可导入本模块并运行纯数值测试.
    import casadi as ca
except ImportError:  # pragma: no cover - 取决于运行环境
    ca = None


G_VEC = np.array([0.0, 0.0, -9.81])
FEASIBILITY_TOL = 1e-9


@dataclass(frozen=True)
class GiacTestCase:
    """可重复的最小测试数据.

    支撑脚按逆时针顺序给出 (从 +z 向下看)::

        p0 = ( 0.20,  0.0000, 0)
        p1 = (-0.10,  0.1732, 0)
        p2 = (-0.10, -0.1732, 0)

    机身位于三角形内部正上方. 默认 GIAC 枚举与参考实现一致, 使用
    所有 ``i < j`` 的无序对; 其中 ``(0, 2)`` 相对 0->1->2 的环绕方向
    是反向的, 会翻转该条不等式的符号. 这里的 ``L_dot`` 取一个小值,
    使这些按字面枚举的 GIAC 不等式在 ``feasible_p_ddot`` 处同时可行.
    这只是约束代码的可重复测试数据, 不宣称它是 ZMP/支撑锥的最优
    物理解释; 若要严格按支撑多边形环绕方向建模, 请显式传入
    ``pairs=[(0, 1), (1, 2), (2, 0)]`` 一类有序对.
    """

    mass: float = 10.0
    mu: float = 0.6
    p_B: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.3])
    )
    feet: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [0.20, 0.0, 0.0],
                [-0.10, 0.1732, 0.0],
                [-0.10, -0.1732, 0.0],
            ]
        )
    )
    L_dot: np.ndarray = field(
        default_factory=lambda: np.array([-0.5, -0.3, 0.0])
    )
    # a_B = g - p_ddot = (0, 0, 0.19): 所有不等式严格可行.
    feasible_p_ddot: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, -10.0])
    )
    # a_B = (0, 0, -0.81): 同时违反一条 GIAC 和摩擦约束.
    violating_p_ddot: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, -9.0])
    )


def reference_case() -> GiacTestCase:
    """返回脚本/测试共用的固定测试数据."""
    return GiacTestCase()


def _as_numeric(value: Any) -> np.ndarray:
    """把 CasADi DM / numpy / Python 标量统一转成 numpy 数组."""
    if hasattr(value, "full"):  # CasADi DM
        value = value.full()
    return np.asarray(value, dtype=float)


def _as_float(value: Any) -> float:
    return float(_as_numeric(value).reshape(-1)[0])


def evaluate_residuals(
    case: GiacTestCase,
    p_B_ddot: np.ndarray,
) -> dict[str, Any]:
    """在给定加速度处计算所有约束的数值残差 (numpy, 不需要 CasADi).

    返回字段:
      - ``a_B``: 重力-惯性加速度
      - ``giac``: ``{(i, j): residual}``, 约束要求 residual <= 0
      - ``friction_norm``: ``||a_B_xy|| - mu * a_B_z``, 要求 <= 0
      - ``friction_squared``: 等价平方形式, 要求 <= 0
      - ``normal_accel``: ``a_B_z``, 平方形式要求 >= 0
      - ``feasible``: 在容差内是否全部满足
    """
    p_B_ddot = np.asarray(p_B_ddot, dtype=float).reshape(3)
    a_B = np.asarray(
        gravity_inertial_acceleration(p_B_ddot, G_VEC), dtype=float
    )

    giac = {
        pair: _as_float(value)
        for pair, value in giac_residuals(
            case.mass, case.p_B, a_B, case.L_dot, case.feet
        )
    }
    friction_norm = _as_float(friction_residual(a_B, case.mu))
    friction_squared = _as_float(friction_squared_residual(a_B, case.mu))
    normal_accel = _as_float(a_B[2])

    feasible = (
        all(value <= FEASIBILITY_TOL for value in giac.values())
        and friction_norm <= FEASIBILITY_TOL
        and normal_accel >= -FEASIBILITY_TOL
    )
    return {
        "a_B": a_B,
        "giac": giac,
        "friction_norm": friction_norm,
        "friction_squared": friction_squared,
        "normal_accel": normal_accel,
        "feasible": feasible,
    }


def build_problem(
    case: GiacTestCase,
    *,
    objective: str = "reference",
    friction_form: str = "squared",
):
    """构造只含 GIAC/摩擦约束的 IPOPT 可行性问题.

    Args:
      objective: ``"reference"`` (默认) 最小化
        ``||p_B_ddot - p_ref||^2``, ``p_ref`` 取 ``case.feasible_p_ddot``;
        最优点严格位于可行域内部, 避免摩擦锥在 ``a_B = 0`` 处的退化.
        ``"norm"`` 最小化 ``||p_B_ddot||^2``, 这就是需求里提到的
        加速度范数平方目标 (但它会把解推到 ``a_B = 0`` 的边界).
        两者都只是为了测试可行性的选择, 不是论文的完整目标.
      friction_form: 传给 :func:`add_friction_constraint`; 默认
        ``"squared"`` (等价且光滑). 设为 ``"norm"`` 可复现原始 norm
        形式, 但它在 ``a_B_xy = 0`` 处梯度未定义, IPOPT 可能失败.

    Returns:
      ``(opti, p_B_ddot, a_B, giac_exprs, friction_exprs)``.

    Raises:
      ImportError: 当前环境没有安装 CasADi.
      ValueError: objective/friction_form 非法.
    """
    if ca is None:
        raise ImportError(
            "当前环境没有安装 CasADi. 请先安装, 例如:\n"
            "  uv pip install casadi\n"
            "或使用项目虚拟环境:\n"
            "  uv pip install --python /home/hong/myrobot_mjlab/.venv/bin/python casadi"
        )

    opti = ca.Opti()

    # 唯一的决策变量. p_B 和脚点位置在本小问题中固定.
    p_B_ddot = opti.variable(3)

    # a_B = g - p_B_ddot. 这里保留用户要求的符号约定, 不做隐式取反.
    a_B = gravity_inertial_acceleration(p_B_ddot, ca.DM(G_VEC))

    # 3 个支撑脚 -> C(3, 2) = 3 条 GIAC 不等式.
    giac_exprs = add_giac_constraints(
        opti,
        case.mass,
        case.p_B,
        a_B,
        case.L_dot,
        case.feet,
        pairs=None,  # 默认 i < j 的无序对
    )

    # 摩擦约束: 用与原 norm 形式等价的平方形式, 并显式加入 a_B_z >= 0.
    # 这样在 a_B_xy = 0 处仍然光滑; 可行集与 norm 形式完全相同.
    friction_exprs = add_friction_constraint(
        opti, a_B, case.mu, form=friction_form
    )

    # 仅为可复现的可行性小实验而选的目标; 不是 TAMOLS 的完整目标函数.
    if objective == "reference":
        # 参考点严格可行, 因此最优解不会落在摩擦锥的退化顶点 a_B = 0.
        opti.minimize(
            ca.sumsqr(p_B_ddot - ca.DM(case.feasible_p_ddot))
        )
    elif objective == "norm":
        opti.minimize(ca.sumsqr(p_B_ddot))
    else:
        raise ValueError(
            f"未知目标 {objective!r}; 可选 'reference' 或 'norm'"
        )

    # 从一个严格可行点出发, 便于 IPOPT 收敛.
    opti.set_initial(p_B_ddot, case.feasible_p_ddot)

    opti.solver(
        "ipopt",
        {
            "print_time": False,
            "ipopt": {
                "print_level": 0,
                "max_iter": 300,
                "tol": 1e-9,
            },
        },
    )
    return opti, p_B_ddot, a_B, giac_exprs, friction_exprs


def _print_residuals(residuals: dict[str, Any]) -> None:
    a_B = residuals["a_B"]
    print("  a_B = g - p_B_ddot = "
          f"[{a_B[0]:+.9f}, {a_B[1]:+.9f}, {a_B[2]:+.9f}]")
    print("  GIAC (要求 <= 0):")
    for (i, j), value in residuals["giac"].items():
        print(f"    pair ({i},{j}): {value:+.12e}")
    print(f"  friction norm residual  : {residuals['friction_norm']:+.12e}"
          "  (要求 <= 0)")
    print(f"  friction squared residual: {residuals['friction_squared']:+.12e}"
          "  (要求 <= 0)")
    print(f"  a_B_z                    : {residuals['normal_accel']:+.12e}"
          "  (平方形式要求 >= 0)")
    print(f"  all constraints satisfied: {residuals['feasible']}")


def main() -> int:
    case = reference_case()

    print("=" * 72)
    print("TAMOLS GIAC + friction minimal NLP (CasADi + IPOPT)")
    print("=" * 72)
    print("mass =", case.mass, " mu =", case.mu)
    print("p_B  =", case.p_B)
    print("feet =")
    for i, foot in enumerate(case.feet):
        print(f"  p{i} = {foot}")
    print("L_dot =", case.L_dot)
    print("p_ref =", case.feasible_p_ddot)
    print("objective = ||p_B_ddot - p_ref||^2  (feasibility test only)")
    print()

    if ca is None:
        print("[跳过求解] 当前 Python 环境没有安装 CasADi.")
        print("安装命令:")
        print("  uv pip install casadi")
        print("或:")
        print("  uv pip install --python "
              "/home/hong/myrobot_mjlab/.venv/bin/python casadi")
        return 1

    print("--- 已知可行点 (求解前检查) ---")
    feasible_residuals = evaluate_residuals(
        case, case.feasible_p_ddot
    )
    print("p_B_ddot =", case.feasible_p_ddot)
    _print_residuals(feasible_residuals)
    print()

    print("--- 故意违反约束的点 (求解前检查) ---")
    violating_residuals = evaluate_residuals(
        case, case.violating_p_ddot
    )
    print("p_B_ddot =", case.violating_p_ddot)
    _print_residuals(violating_residuals)
    print()

    print("--- IPOPT 求解 ---")
    opti, p_B_ddot, a_B, giac_exprs, friction_exprs = build_problem(case)
    try:
        solution = opti.solve()
    except RuntimeError as exc:
        print("[IPOPT 求解失败]", exc)
        try:
            print("status:", opti.stats().get("return_status", "unknown"))
        except Exception:  # pragma: no cover - 诊断信息不可用
            pass
        return 2

    try:
        stats = solution.stats()
    except AttributeError:  # pragma: no cover - 兼容较老版本的 CasADi
        stats = opti.stats()
    print("status        :", stats.get("return_status", "unknown"))
    print("success       :", stats.get("success", "unknown"))
    print("iterations    :", stats.get("iter_count", "unknown"))
    print()

    p_B_ddot_value = _as_numeric(
        solution.value(p_B_ddot)
    ).reshape(3)
    a_B_value = np.asarray(
        [_as_float(solution.value(component)) for component in a_B],
        dtype=float,
    )
    delta = p_B_ddot_value - case.feasible_p_ddot
    objective_value = float(np.dot(delta, delta))

    print("p_B_ddot* = "
          f"[{p_B_ddot_value[0]:+.9f}, "
          f"{p_B_ddot_value[1]:+.9f}, "
          f"{p_B_ddot_value[2]:+.9f}]")
    print("a_B*      = "
          f"[{a_B_value[0]:+.9f}, "
          f"{a_B_value[1]:+.9f}, "
          f"{a_B_value[2]:+.9f}]")
    print(f"objective = {objective_value:.12f}")
    print(f"number of GIAC constraints     = {len(giac_exprs)}")
    print(f"number of friction constraints = {len(friction_exprs)}"
          "  (squared residual + a_B_z >= 0)")
    print()

    print("--- 求解后约束残差 ---")
    residuals = evaluate_residuals(case, p_B_ddot_value)
    _print_residuals(residuals)
    print()

    success = bool(stats.get("success", False))
    if not success:
        print("[警告] IPOPT 未报告成功; 上面的点仅用于诊断.")
        return 2
    if not residuals["feasible"]:
        print("[警告] 求解器返回的点未满足全部约束 (残差超出容差).")
        return 3
    print("结论: IPOPT 找到了满足 GIAC 与摩擦约束的数值点.")
    print("注意: 这是数值可行性证据, 不是全局可行性证明;")
    print("      本脚本只验证约束代码, 不等于完整 TAMOLS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
