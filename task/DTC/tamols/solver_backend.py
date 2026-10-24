"""求解器后端抽象层.

目的: 优化问题的 "变量 / 约束 / 目标" 组装代码只依赖这里定义的
:class:`SolverBackend` 接口, 不直接依赖 CasADi. 当前默认后端是
CasADi + IPOPT; 之后如果换成自定义求解器, 只需要实现同样的接口,
上层 :class:`~myrobot_mjlab.task.DTC.tamols.optimization_problem.TrajectoryOptimizationProblem`
不需要改动.

接口刻意保持很小:
  - :meth:`variable`       创建决策变量
  - :meth:`add_equality`   添加等式约束  lhs == rhs
  - :meth:`add_inequality` 添加双边不等式 lb <= expr <= ub
  - :meth:`add_objective`  累加一个目标项 (标量代价, 最小化)
  - :meth:`objective`      返回已累加目标项之和
  - :meth:`minimize`       把累加的目标交给求解器 (每个问题调用一次)
  - :meth:`set_initial`    设置初值 (warm start)
  - :meth:`solve`          调用具体求解器
  - :meth:`value`          读取表达式在当前解/初值处的数值

注意: 表达式本身仍然是后端类型 (CasADi 的 MX). 上层只通过上述方法
与后端交互, 因此更换求解器时, 约束的数学表达形式可以逐条移植, 而不必
改动变量布局和状态向量定义.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

try:  # CasADi 是可选依赖: 未安装时仍可导入本模块和使用纯布局逻辑.
    import casadi as ca
except ImportError:  # pragma: no cover - 取决于运行环境
    ca = None


class SolverBackend(ABC):
    """优化问题组装所需的最小求解器接口."""

    @abstractmethod
    def variable(self, shape: tuple[int, ...], name: str = "") -> Any:
        """创建一个形状为 ``shape`` 的决策变量, 返回后端表达式."""

    @abstractmethod
    def add_equality(self, lhs: Any, rhs: Any, name: str = "") -> None:
        """添加等式约束 ``lhs == rhs``."""

    @abstractmethod
    def add_inequality(self, expr: Any, lb: Any, ub: Any, name: str = "") -> None:
        """添加双边不等式约束 ``lb <= expr <= ub`` (lb/ub 可为 +-inf)."""

    @abstractmethod
    def set_initial(self, var: Any, value: Any) -> None:
        """设置变量初值."""

    @abstractmethod
    def solve(self) -> Any:
        """求解当前问题."""

    @abstractmethod
    def value(self, expr: Any) -> np.ndarray:
        """返回表达式在当前解处的数值 (未求解时退回当前初值)."""

    # -- 目标函数 (累加式) ---------------------------------------------- #
    def add_objective(self, term: Any, name: str = "") -> None:
        """累加一个标量目标项 (将被最小化).

        目标项应是**已含权重**的标量代价表达式 (数值或后端表达式).
        多次调用等价于把各项相加; :meth:`objective` 返回总和.
        """
        terms = getattr(self, "_objective_terms", None)
        if terms is None:
            terms = []
            self._objective_terms = terms
        terms.append((name, term))

    def objective_terms(self) -> list[tuple[str, Any]]:
        """已累加的目标项 ``[(name, term), ...]``."""
        return list(getattr(self, "_objective_terms", ()))

    def objective(self) -> Any:
        """已累加目标项之和; 没有目标项时返回 ``0.0``."""
        terms = self.objective_terms()
        if not terms:
            return 0.0
        total = terms[0][1]
        for _, term in terms[1:]:
            total = total + term
        return total

    def minimize(self) -> None:
        """把 :meth:`objective` 交给求解器.

        每个问题在添加完全部目标项后调用**一次**; 基类不支持时抛
        ``NotImplementedError``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} 未实现 minimize(); 请在后端子类中把 "
            "objective() 交给具体求解器."
        )

    def solve_diagnostics(self) -> dict[str, Any]:
        """最近一次 solve() 的状态诊断; 基类返回空字段."""
        return {"status": None, "iterations": None, "success": None}


@dataclass
class CasadiBackend(SolverBackend):
    """CasADi + IPOPT 后端.

    Args:
      plugin: CasADi 求解器插件名, 默认 ``"ipopt"``.
      solver_options: 传给 ``opti.solver(plugin, options)`` 的选项, 例如
        ``{"ipopt": {"max_iter": 200, "print_level": 0}}``.
      print_time: 是否打印 CasADi 求解耗时.
    """

    plugin: str = "ipopt"
    solver_options: Mapping[str, Any] = field(default_factory=dict)
    print_time: bool = False

    def __post_init__(self) -> None:
        if ca is None:
            raise ImportError(
                "当前环境没有安装 CasADi. 请先安装, 例如:\n"
                "  uv pip install --python /home/hong/myrobot_mjlab/.venv/bin/python casadi\n"
                "或改用实现 SolverBackend 接口的自定义后端."
            )
        self.opti = ca.Opti()
        self._solver_options = dict(self.solver_options)
        self._solver_options.setdefault("print_time", self.print_time)
        self._variables: list[tuple[str, tuple[int, ...]]] = []
        self._constraints: list[tuple[str, str]] = []

    def variable(self, shape: tuple[int, ...], name: str = "") -> Any:
        shape = tuple(int(s) for s in shape)
        if len(shape) == 1:
            var = self.opti.variable(shape[0])
        elif len(shape) == 2:
            var = self.opti.variable(shape[0], shape[1])
        else:
            raise ValueError(f"CasADi 后端只支持 1D/2D 变量, 实际 shape={shape}")
        self._variables.append((name, shape))
        return var

    def add_equality(self, lhs: Any, rhs: Any, name: str = "") -> None:
        self.opti.subject_to(lhs == rhs)
        self._constraints.append(("equality", name))

    def add_inequality(
        self, expr: Any, lb: Any, ub: Any, name: str = ""
    ) -> None:
        self.opti.subject_to(self.opti.bounded(lb, expr, ub))
        self._constraints.append(("inequality", name))

    def set_initial(self, var: Any, value: Any) -> None:
        self.opti.set_initial(var, value)

    def solve(self) -> Any:
        # 每次求解都重新构造 solver, 这样求解之后再添加约束/目标也不会失效.
        self.opti.solver(self.plugin, self._solver_options)
        return self.opti.solve()

    def value(self, expr: Any) -> np.ndarray:
        try:
            return np.asarray(self.opti.value(expr))
        except RuntimeError:
            # 尚未 solve() 时 opti.value 不可用, 用当前初值求值.
            return np.asarray(self.opti.debug.value(expr))

    def minimize(self) -> None:
        """把累加的目标项之和设为 ``opti`` 的目标函数.

        没有目标项时不设 (纯可行性问题). 每个问题调用一次; 重复调用
        会以最后一次的总和覆盖之前的目标.
        """
        terms = self.objective_terms()
        if not terms:
            return
        self.opti.minimize(self.objective())

    def solve_diagnostics(self) -> dict[str, Any]:
        """最近一次 solve() 的状态诊断 (未求解时字段为 None).

        返回 ``{"status", "iterations", "success"}``; 状态字符串来自
        IPOPT (例如 ``"Solve_Succeeded"`` / ``"Infeasible_Problem_Detected"``).
        注意: 求解状态**不能**替代独立残差检查.
        """
        try:
            stats = self.opti.stats()
        except Exception:  # 尚未 solve() 时 stats 不可用
            return {"status": None, "iterations": None, "success": None}
        return {
            "status": stats.get("return_status"),
            "iterations": stats.get("iter_count"),
            "success": bool(stats.get("success", False)),
        }

    @property
    def variables(self) -> list[tuple[str, tuple[int, ...]]]:
        """已创建的变量 (name, shape), 便于调试/换后端时核对."""
        return list(self._variables)

    @property
    def constraints(self) -> list[tuple[str, str]]:
        """已添加的约束 (类型, 名称)."""
        return list(self._constraints)


def make_backend(name: str = "casadi", **kwargs: Any) -> SolverBackend:
    """按名字创建求解器后端.

    新增自定义求解器时, 在这里注册即可::

        if name == "my_solver":
            return MySolverBackend(**kwargs)
    """
    key = name.lower()
    if key == "casadi":
        return CasadiBackend(**kwargs)
    raise ValueError(
        f"未知求解器后端 {name!r}; 当前支持: casadi. "
        "自定义后端需实现 SolverBackend 接口并在 make_backend() 中注册."
    )
