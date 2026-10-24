"""高程图的双线性插值查询层 (numpy / CasADi MX 通用).

论文 IV-B ("Numerical Differentiation and Interpolation") 指出: 网格图的
离散梯度会让梯度类求解器的解在相邻两次优化之间在格点间跳动, 因此所有
地图查询都要用双线性插值. 本模块提供该插值查询层, 供
:mod:`~myrobot_mjlab.task.DTC.tamols.v_b_objectives` 中的目标函数
(V-B1 的 ``h(p_i)``、V-B4 的 ``h_s2(p_B)``、V-B5 的 ``∇h(p_i)`` 与
``∇h_s1(p_i)``) 使用.

与 :mod:`~myrobot_mjlab.task.DTC.tamols.map_derivatives` 的分工
--------------------------------------------------------------
- ``map_derivatives`` 负责**网格上**的差分 (五点核) 与标量坐标的 torch
  双线性查询, 面向 RL 侧;
- 本模块负责**优化侧**: 输入是已经处理好的网格 (h / h_s1 / h_s2 及其
  梯度场), 输出是可以在 CasADi 表达式树里出现的符号插值, 同时保留
  纯 numpy 的数值查询路径, 便于单元测试复算.
  梯度场本身由 ``map_derivatives.compute_derivatives`` (torch 五点核)
  计算后转成 numpy 缓存, 本模块不重新实现差分格式.

坐标约定
--------
- 网格 ``V[iy, ix]``: ``ix`` 沿地图 x (机头方向), ``iy`` 沿地图 y (机体
  左侧); 地图系内 ``x[ix] = ix*dx``, ``y[iy] = iy*dy``;
- 地图系相对世界系: 原点 ``origin=(x0, y0)`` (世界系, m), 绕 z 旋转
  ``yaw`` (terrain_scan 的 ``ray_alignment="yaw"`` 即地图系 = 机身偏航
  系). 查询时先做 ``p_map = R(yaw)^T (p_world - origin)``; ``yaw=0`` 且
  ``origin=0`` 时地图系即世界系;
- 高度单位 m, 梯度单位 m/m (无量纲).

边界处理 (out_of_bounds)
------------------------
- ``"clamp"`` (默认): 查询点截断到网格边界. 符号查询**必须**用 clamp,
  因为 CasADi 的 linear interpolant 在网格外会线性外推 (已验证), 外推
  值没有物理意义且会把落脚点拉出地图; 截断用 ``fmin/fmax`` 实现, 几乎
  处处可微;
- ``"nan"``: 数值查询越界返回 NaN (不能用于符号装配);
- ``"raise"``: 数值查询越界抛 ``ValueError`` (不能用于符号装配).
符号查询越界时不抛错 (优化过程中脚点可能瞬时出界), 但
:meth:`GridMapField.covers` 可以在初值/解上检查覆盖情况.

缺失值处理 (fill)
-----------------
网格中的 NaN/inf 视为缺失 (传感器打空、地图空洞):
- ``"median"`` (默认): 用该网格有效格的中位数填充, 与
  ``map_process.HeightMapProcessor.fill_invalid`` 的策略一致;
- ``"value"``: 用 ``fill_value`` 填充;
- ``"raise"``: 存在缺失即抛 ``ValueError``.
填充只是让插值有定义, **不代表**这些格有观测: 每个字段额外维护一张
0/1 有效性掩码的双线性场 :meth:`GridMapField.validity`, 调用方可以用它
判断某个查询点周围是否真的有观测 (V-B5 的梯度在缺失格附近不可信,
论文也警告不要假设地图梯度对所有位置都有效). 填充比例记录在
``filled_fraction`` 中, 便于诊断.

插值
----
符号路径用 ``casadi.interpolant(..., "linear", ...)``: 数据布局为
``V[iy, ix].flatten()`` (C 序), 网格 ``[x_grid, y_grid]`` (已用节点值
单测验证). 它的 jacobian 是逐格常数 (双线性插值在该格内的梯度), 与
论文 IV-B 的双线性导数插值一致. 数值路径用同一套权重的纯 numpy 实现,
两条路径在单测中互相对照.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np

try:  # CasADi 可选: 没有数值优化需求时仍可只用 numpy 查询路径.
    import casadi as ca
except ImportError:  # pragma: no cover - 取决于运行环境
    ca = None

OutOfBoundsMode = Literal["nan", "clamp", "raise"]
FillMode = Literal["median", "value", "raise"]

_BOUNDS_EPS = 1e-9


def _is_symbolic(value: Any) -> bool:
    module = type(value).__module__
    return module == "casadi" or module.startswith("casadi.")


def _as_grid(array: Any, name: str) -> np.ndarray:
    """把 torch / numpy / 嵌套序列规范化为二维 float64 网格 ``[Ny, Nx]``."""
    if hasattr(array, "detach"):  # torch.Tensor
        array = array.detach().cpu().numpy()
    grid = np.asarray(array, dtype=float)
    while grid.ndim > 2 and grid.shape[0] == 1:
        grid = grid[0]
    if grid.ndim != 2:
        raise ValueError(
            f"{name} 需要二维网格 [Ny, Nx] (或可 squeeze 成二维, 例如 "
            f"[1, 1, Ny, Nx]), 实际 shape={tuple(np.asarray(array).shape)}. "
            "多环境 batch 地图请先选出单个环境."
        )
    if grid.shape[0] < 2 or grid.shape[1] < 2:
        raise ValueError(
            f"{name} 每条轴至少需要 2 个网格点才能插值, 实际 {grid.shape}"
        )
    return grid


def _clamp_symbolic(value: Any, lo: float, hi: float) -> Any:
    return ca.fmin(ca.fmax(value, lo), hi)


class GridMapField:
    """单个二维网格字段的双线性查询 (高度或某个梯度分量).

    Args:
      data: ``[Ny, Nx]`` 网格 (torch/numpy 均可); NaN/inf 视为缺失.
      dx, dy: 网格分辨率 [m/cell], 必须为有限正数.
      origin: 地图系原点在世界系 xy 中的位置 ``(x0, y0)`` [m].
      yaw: 地图系相对世界系绕 z 的旋转 [rad]; 查询做
        ``p_map = R(yaw)^T (p_world - origin)``. 默认 0.
      out_of_bounds: 数值查询的越界策略, 见模块文档; 符号查询恒为 clamp.
      fill: 缺失值填充策略, 见模块文档.
      fill_value: ``fill="value"`` 时的填充值.
      name: 字段名, 仅用于报错与诊断.
    """

    def __init__(
        self,
        data: Any,
        dx: float,
        dy: float,
        origin: tuple[float, float] = (0.0, 0.0),
        yaw: float = 0.0,
        out_of_bounds: OutOfBoundsMode = "clamp",
        fill: FillMode = "median",
        fill_value: float = 0.0,
        name: str = "field",
    ):
        if out_of_bounds not in ("nan", "clamp", "raise"):
            raise ValueError(
                "out_of_bounds 只能是 'nan'/'clamp'/'raise', "
                f"实际为 {out_of_bounds!r}"
            )
        if fill not in ("median", "value", "raise"):
            raise ValueError(
                f"fill 只能是 'median'/'value'/'raise', 实际为 {fill!r}"
            )
        for label, value in (("dx", dx), ("dy", dy)):
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{label} 必须是有限正数, 实际为 {value!r}")
        x0, y0 = (float(v) for v in origin)
        if not (math.isfinite(x0) and math.isfinite(y0)):
            raise ValueError(f"origin 必须有限, 实际为 {origin!r}")
        if not math.isfinite(float(yaw)):
            raise ValueError(f"yaw 必须有限, 实际为 {yaw!r}")

        self.name = name
        self.dx = float(dx)
        self.dy = float(dy)
        self.origin = (x0, y0)
        self.yaw = float(yaw)
        self.out_of_bounds = out_of_bounds
        self.fill = fill

        raw = _as_grid(data, name)
        valid = np.isfinite(raw)
        self.invalid_cells = int(valid.size - np.count_nonzero(valid))
        self.fill_value_used: float | None = None
        if self.invalid_cells:
            if fill == "raise":
                raise ValueError(
                    f"{name} 含 {self.invalid_cells} 个缺失格 (NaN/inf) 且 "
                    "fill='raise'; 请先处理缺失值或改用 'median'/'value'."
                )
            filled = raw.copy()
            if fill == "median":
                fallback = (
                    float(np.median(raw[valid])) if valid.any() else 0.0
                )
            else:
                fallback = float(fill_value)
                if not math.isfinite(fallback):
                    raise ValueError(
                        f"fill_value 必须有限, 实际为 {fill_value!r}"
                    )
            filled[~valid] = fallback
            self.fill_value_used = fallback
        else:
            filled = raw
        self.data = filled
        self.valid_mask = valid
        self.filled_fraction = self.invalid_cells / valid.size

        ny, nx = filled.shape
        self.ny, self.nx = ny, nx
        # 地图系内的范围 (地图系原点在网格 (0, 0) 格).
        self.bounds = (
            0.0,
            (nx - 1) * self.dx,
            0.0,
            (ny - 1) * self.dy,
        )
        self._valid_field = np.where(valid, 1.0, 0.0)

        self._interpolant = None
        self._valid_interpolant = None
        if ca is not None:
            x_grid = np.arange(nx) * self.dx
            y_grid = np.arange(ny) * self.dy
            self._interpolant = ca.interpolant(
                f"{name}_bilinear", "linear", [x_grid, y_grid],
                filled.flatten(),
            )
            self._valid_interpolant = ca.interpolant(
                f"{name}_valid_bilinear", "linear", [x_grid, y_grid],
                self._valid_field.flatten(),
            )

    # ------------------------------------------------------------------ #
    # 坐标变换与覆盖检查
    # ------------------------------------------------------------------ #
    def to_map_frame(self, x: Any, y: Any) -> tuple[Any, Any]:
        """世界系 xy -> 地图系 xy (支持符号)."""
        dx = x - self.origin[0]
        dy = y - self.origin[1]
        if self.yaw == 0.0:
            return dx, dy
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return c * dx + s * dy, -s * dx + c * dy

    def covers(self, x: float, y: float, margin: float = 0.0) -> bool:
        """数值查询点是否落在地图范围内 (地图系, 含 margin [m])."""
        xm, ym = self.to_map_frame(float(x), float(y))
        x_min, x_max, y_min, y_max = self.bounds
        return (
            x_min - margin <= float(xm) <= x_max + margin
            and y_min - margin <= float(ym) <= y_max + margin
        )

    # ------------------------------------------------------------------ #
    # 查询
    # ------------------------------------------------------------------ #
    def query(self, x: Any, y: Any) -> Any:
        """双线性插值查询; ``x, y`` 为世界系坐标, 数值或 CasADi 标量.

        符号输入: 地图系坐标先截断到网格范围 (clamp), 返回 MX 标量.
        数值输入: 按 ``self.out_of_bounds`` 处理越界, 返回 float.
        """
        xm, ym = self.to_map_frame(x, y)
        if _is_symbolic(xm) or _is_symbolic(ym):
            if self._interpolant is None:
                raise RuntimeError(
                    "未安装 CasADi, 无法做符号插值查询; 请改用数值查询."
                )
            xc = _clamp_symbolic(xm, self.bounds[0], self.bounds[1])
            yc = _clamp_symbolic(ym, self.bounds[2], self.bounds[3])
            return self._interpolant(ca.vertcat(xc, yc))
        return self._query_numeric(float(xm), float(ym), self.data)

    def validity(self, x: Any, y: Any) -> Any:
        """查询点周围有效观测比例 (0..1, 双线性); 缺失格附近 < 1."""
        xm, ym = self.to_map_frame(x, y)
        if _is_symbolic(xm) or _is_symbolic(ym):
            if self._valid_interpolant is None:
                raise RuntimeError("未安装 CasADi, 无法做符号插值查询.")
            xc = _clamp_symbolic(xm, self.bounds[0], self.bounds[1])
            yc = _clamp_symbolic(ym, self.bounds[2], self.bounds[3])
            return self._valid_interpolant(ca.vertcat(xc, yc))
        return self._query_numeric(float(xm), float(ym), self._valid_field)

    def _query_numeric(self, xm: float, ym: float, field: np.ndarray) -> float:
        """地图系坐标 (xm, ym) 上的 numpy 双线性插值."""
        if not (math.isfinite(xm) and math.isfinite(ym)):
            raise ValueError(f"查询坐标必须有限, 实际为 ({xm}, {ym})")
        fx = xm / self.dx
        fy = ym / self.dy
        inside = (
            -_BOUNDS_EPS <= fx <= (self.nx - 1) + _BOUNDS_EPS
            and -_BOUNDS_EPS <= fy <= (self.ny - 1) + _BOUNDS_EPS
        )
        if not inside:
            if self.out_of_bounds == "raise":
                raise ValueError(
                    f"{self.name}: 地图系查询点 ({xm:.4g}, {ym:.4g}) 超出范围 "
                    f"x∈[{self.bounds[0]}, {self.bounds[1]}], "
                    f"y∈[{self.bounds[2]}, {self.bounds[3]}]"
                )
            if self.out_of_bounds == "nan":
                return float("nan")
            fx = min(max(fx, 0.0), float(self.nx - 1))
            fy = min(max(fy, 0.0), float(self.ny - 1))
        j0 = min(int(math.floor(fx)), self.nx - 2)
        i0 = min(int(math.floor(fy)), self.ny - 2)
        tx = fx - j0
        ty = fy - i0
        v00 = field[i0, j0]
        v01 = field[i0, j0 + 1]
        v10 = field[i0 + 1, j0]
        v11 = field[i0 + 1, j0 + 1]
        return float(
            (1 - ty) * (1 - tx) * v00
            + (1 - ty) * tx * v01
            + ty * (1 - tx) * v10
            + ty * tx * v11
        )


# ---------------------------------------------------------------------- #
# 地形三图容器: h / h_s1 / h_s2 及其梯度场
# ---------------------------------------------------------------------- #
class TerrainMaps:
    """V-B 目标所需的地图字段集合 (h, ∇h, h_s1, h_s2, 有效性).

    Args:
      h: 原始高程图 ``[Ny, Nx]`` (或 [B,1,Ny,Nx] 单环境).
      hs1: 平滑图 h_s1 (V-B5 第二项); None 时 :meth:`grad_hs1` 不可用.
      hs2: 虚拟平面 h_s2 (V-B4); None 时 :meth:`height_hs2` 不可用.
      dx, dy: 网格分辨率 [m/cell].
      origin, yaw: 地图系相对世界系的原点与绕 z 旋转, 见
        :class:`GridMapField`.
      gradient_fields: 可选, 预先算好的梯度场字典
        ``{"grad_h_x", "grad_h_y", "grad_hs1_x", "grad_hs1_y"}``;
        None 时用 ``map_derivatives.compute_derivatives`` (torch 五点核)
        对 h / hs1 现算.
      out_of_bounds, fill, fill_value: 传给每个字段.

    所有查询方法同时接受数值与 CasADi 标量.
    """

    def __init__(
        self,
        h: Any,
        hs1: Any | None = None,
        hs2: Any | None = None,
        dx: float = 0.05,
        dy: float = 0.05,
        origin: tuple[float, float] = (0.0, 0.0),
        yaw: float = 0.0,
        gradient_fields: dict[str, Any] | None = None,
        out_of_bounds: OutOfBoundsMode = "clamp",
        fill: FillMode = "median",
        fill_value: float = 0.0,
    ):
        common = dict(
            dx=dx, dy=dy, origin=origin, yaw=yaw,
            out_of_bounds=out_of_bounds, fill=fill, fill_value=fill_value,
        )
        self.h = GridMapField(h, name="h", **common)

        if gradient_fields is None:
            gradient_fields = self.compute_gradient_fields(h, hs1, dx, dy)
        else:
            gradient_fields = {
                key: _as_grid(value, key)
                for key, value in gradient_fields.items()
            }

        def field(key: str) -> np.ndarray:
            if key not in gradient_fields:
                raise ValueError(
                    f"gradient_fields 缺少 '{key}'; "
                    f"已有: {sorted(gradient_fields)}"
                )
            return gradient_fields[key]

        self.grad_h_x = GridMapField(field("grad_h_x"), name="grad_h_x", **common)
        self.grad_h_y = GridMapField(field("grad_h_y"), name="grad_h_y", **common)
        if hs1 is not None:
            self.hs1: GridMapField | None = GridMapField(hs1, name="hs1", **common)
            self.grad_hs1_x: GridMapField | None = GridMapField(
                field("grad_hs1_x"), name="grad_hs1_x", **common
            )
            self.grad_hs1_y: GridMapField | None = GridMapField(
                field("grad_hs1_y"), name="grad_hs1_y", **common
            )
        else:
            self.hs1 = None
            self.grad_hs1_x = None
            self.grad_hs1_y = None
        self.hs2 = (
            GridMapField(hs2, name="hs2", **common) if hs2 is not None else None
        )

    # -- 查询接口 ------------------------------------------------------- #
    def height_h(self, x: Any, y: Any) -> Any:
        """h(p) [m]; V-B1 与 V-B5 (通过梯度场) 使用."""
        return self.h.query(x, y)

    def height_hs2(self, x: Any, y: Any) -> Any:
        """h_s2(p) [m]; V-B4 使用. 未提供 hs2 时抛 RuntimeError."""
        if self.hs2 is None:
            raise RuntimeError("TerrainMaps 未提供 hs2, 无法查询 h_s2.")
        return self.hs2.query(x, y)

    def grad_h(self, x: Any, y: Any) -> tuple[Any, Any]:
        """∇h(p) = (∂h/∂x, ∂h/∂y) [m/m]; V-B5 第一项."""
        return self.grad_h_x.query(x, y), self.grad_h_y.query(x, y)

    def grad_hs1(self, x: Any, y: Any) -> tuple[Any, Any]:
        """∇h_s1(p) [m/m]; V-B5 第二项. 未提供 hs1 时抛 RuntimeError."""
        if self.grad_hs1_x is None or self.grad_hs1_y is None:
            raise RuntimeError("TerrainMaps 未提供 hs1, 无法查询 ∇h_s1.")
        return self.grad_hs1_x.query(x, y), self.grad_hs1_y.query(x, y)

    def validity_at(self, x: float, y: float) -> dict[str, float]:
        """数值查询各字段在 (x, y) 处的有效观测比例, 便于诊断."""
        out = {"h": float(self.h.validity(x, y))}
        if self.hs1 is not None:
            out["hs1"] = float(self.hs1.validity(x, y))
        if self.hs2 is not None:
            out["hs2"] = float(self.hs2.validity(x, y))
        return out

    def describe(self) -> str:
        """字段形状 / 范围 / 缺失填充情况的诊断文本."""
        lines = [
            f"TerrainMaps: dx={self.h.dx}, dy={self.h.dy}, "
            f"origin={self.h.origin}, yaw={self.yaw:.4g} rad"
        ]
        for name in (
            "h", "hs1", "hs2",
            "grad_h_x", "grad_h_y", "grad_hs1_x", "grad_hs1_y",
        ):
            field = getattr(self, name)
            if field is None:
                lines.append(f"  {name:<10s} (未提供)")
                continue
            lines.append(
                f"  {name:<10s} shape=({field.ny}, {field.nx}) "
                f"missing={field.invalid_cells} "
                f"filled_fraction={field.filled_fraction:.3g} "
                f"map-frame bounds=x[{field.bounds[0]:.3g}, {field.bounds[1]:.3g}] "
                f"y[{field.bounds[2]:.3g}, {field.bounds[3]:.3g}]"
            )
        return "\n".join(lines)

    # -- 梯度场计算 ----------------------------------------------------- #
    @staticmethod
    def compute_gradient_fields(
        h: Any, hs1: Any | None, dx: float, dy: float
    ) -> dict[str, np.ndarray]:
        """用 map_derivatives 的五点核计算 ∇h 与 ∇h_s1 (torch -> numpy).

        返回 ``{"grad_h_x", "grad_h_y", ("grad_hs1_x", "grad_hs1_y")}``,
        形状与输入网格相同; 缺失格 (NaN) 附近的导数按 map_derivatives
        的约定为 NaN, 交给 :class:`GridMapField` 的填充策略处理.
        """
        import torch

        from myrobot_mjlab.task.DTC.tamols.map_derivatives import (
            compute_derivatives,
        )

        def grad_of(grid: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
            array = _as_grid(grid, name)
            derivs = compute_derivatives(
                torch.from_numpy(array)[None, None], dx=dx, dy=dy
            )
            return (
                np.asarray(derivs.grad_x[0, 0].numpy(), dtype=float),
                np.asarray(derivs.grad_y[0, 0].numpy(), dtype=float),
            )

        out: dict[str, np.ndarray] = {}
        out["grad_h_x"], out["grad_h_y"] = grad_of(h, "h")
        if hs1 is not None:
            out["grad_hs1_x"], out["grad_hs1_y"] = grad_of(hs1, "hs1")
        return out
