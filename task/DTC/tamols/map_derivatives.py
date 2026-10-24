"""高程图的数值微分与双线性插值.

本模块只负责两件事:
  1. 对规则二维高度网格 h[i, j] 计算一阶/二阶导数与曲率场;
  2. 在任意连续坐标 (x, y) 处对上述导数场做双线性插值.
不涉及机器人动力学或落脚点优化.

坐标与索引约定
--------------
h[i, j] 与 map_process.HeightMapProcessor 的输出布局一致:
  - j (最后一维 / 列) 沿 x 方向 (机体前方), i (倒数第二维 / 行) 沿 y 方向 (机体左侧);
  - x_j = x0 + j * dx, y_i = y0 + i * dy, 由 origin=(x0, y0) 指定;
  - 高度 h 单位 [m], 网格分辨率 dx, dy 单位 [m/cell].

导数与曲率的定义 (z = h(x, y), 向上为 +z)
------------------------------------------
  - grad_x = ∂h/∂x, grad_y = ∂h/∂y                [m/m, 无量纲]
  - hess_xx = ∂²h/∂x², hess_xy = ∂²h/∂x∂y, hess_yy = ∂²h/∂y²   [1/m]
  - slope = sqrt(grad_x² + grad_y²)               [m/m, 无量纲]
  - mean_curvature (平均曲率):
        H = [(1+gy²) hxx - 2 gx gy hxy + (1+gx²) hyy] / (2 (1+gx²+gy²)^{3/2})
    单位 [1/m]; 碗形 (凹向上) H > 0, 穹顶 H < 0.
  - gaussian_curvature (高斯曲率):
        K = (hxx hyy - hxy²) / (1+gx²+gy²)²
    单位 [1/m²]; 碗/穹顶 K > 0, 鞍面 K < 0.

离散格式 (五点中心差分, 内部 4 阶精度)
---------------------------------------
记格距 d, 对内部点 (两侧至少各有 2 个邻居):
    f'_i  = ( f_{i-2} - 8 f_{i-1} + 8 f_{i+1} - f_{i+2} ) / (12 d)
    f''_i = ( -f_{i-2} + 16 f_{i-1} - 30 f_i + 16 f_{i+1} - f_{i+2} ) / (12 d²)
混合导数 hess_xy 通过对 y、x 依次应用五点一阶差分得到.

边界处理
--------
每侧最外 2 行/列无法使用中心模板, 改用 5 点单侧差分, 保证输出形状
与输入完全一致、边界点也有定义. 一阶导数单侧公式为 4 阶精度, 二阶导数
单侧公式 (5 点) 为 3 阶精度:
    f'_0     = (-25 f0 + 48 f1 - 36 f2 + 16 f3 - 3 f4) / (12 d)
    f'_1     = (-3 f0 - 10 f1 + 18 f2 - 6 f3 + f4) / (12 d)
    f'_{n-2} = (-f_{n-5} + 6 f_{n-4} - 18 f_{n-3} + 10 f_{n-2} + 3 f_{n-1}) / (12 d)
    f'_{n-1} = (3 f_{n-5} - 16 f_{n-4} + 36 f_{n-3} - 48 f_{n-2} + 25 f_{n-1}) / (12 d)
二阶导数使用对应的 5 点单侧公式 (见 _diff2_axis). 每条轴至少需要 5 个网格点,
否则抛出 ValueError (无法构造五点模板).

无效值
------
h 中的 NaN/inf 视为缺失. 不填补、不伪造: 某个差分模板只要触到无效格,
该点的导数就为 NaN; 无效格自身的导数也强制为 NaN (五点一阶差分在中心点权重为 0,
否则该点会"借用"邻居值而看起来有效). 内部点受影响半径约 2 格; 最外两行/列
使用 5 点单侧模板, NaN 最多可向内传播 4 格.
双线性插值时, 若 4 个角点中有任一无效, 默认返回 NaN; 传入 fill_value 可回退.

复杂度与缓存
------------
对 Ny x Nx 的地图, 全图导数/曲率计算时间 O(Ny*Nx), 每个分量显存 O(Ny*Nx),
每个网格点仅为常数次乘加; 单点查询 O(1), 不会触发全图重算.
HeightMapDerivativeProvider 缓存最近一次 update() 的结果: 同一帧内多次查询
只计算一次导数场, 只有显式 update() 或 invalidate() 才会使缓存失效.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

DERIVATIVE_COMPONENTS: tuple[str, ...] = (
  "grad_x",
  "grad_y",
  "hess_xx",
  "hess_xy",
  "hess_yy",
  "slope",
  "mean_curvature",
  "gaussian_curvature",
)

OutOfBoundsMode = Literal["nan", "clamp", "raise"]

# 网格索引容差: 查询点落在网格边界 ±1e-6 个格距内仍视为有效.
_BOUNDS_EPS = 1e-6


# --------------------------------------------------------------------- #
# 五点差分核 (逐轴 1D)
# --------------------------------------------------------------------- #
def _diff1_axis(v: torch.Tensor, spacing: float, axis: int) -> torch.Tensor:
  """沿指定轴做五点一阶差分 (内部中心差分, 边界 4 阶单侧差分)."""
  n = v.shape[axis]
  if n < 5:
    raise ValueError(
      f"五点差分要求该轴至少 5 个网格点, 实际为 {n} (axis={axis})"
    )
  x = v.movedim(axis, -1)
  out = torch.empty_like(x)
  denom = 12.0 * spacing

  # 内部: (f_{i-2} - 8 f_{i-1} + 8 f_{i+1} - f_{i+2}) / (12 d)
  out[..., 2:-2] = (
    x[..., :-4] - 8.0 * x[..., 1:-3] + 8.0 * x[..., 3:-1] - x[..., 4:]
  ) / denom

  # 左边界 (4 阶单侧)
  out[..., 0] = (
    -25.0 * x[..., 0]
    + 48.0 * x[..., 1]
    - 36.0 * x[..., 2]
    + 16.0 * x[..., 3]
    - 3.0 * x[..., 4]
  ) / denom
  out[..., 1] = (
    -3.0 * x[..., 0]
    - 10.0 * x[..., 1]
    + 18.0 * x[..., 2]
    - 6.0 * x[..., 3]
    + x[..., 4]
  ) / denom

  # 右边界 (与左侧镜像对称)
  out[..., n - 2] = (
    -x[..., n - 5]
    + 6.0 * x[..., n - 4]
    - 18.0 * x[..., n - 3]
    + 10.0 * x[..., n - 2]
    + 3.0 * x[..., n - 1]
  ) / denom
  out[..., n - 1] = (
    3.0 * x[..., n - 5]
    - 16.0 * x[..., n - 4]
    + 36.0 * x[..., n - 3]
    - 48.0 * x[..., n - 2]
    + 25.0 * x[..., n - 1]
  ) / denom

  return out.movedim(-1, axis)


def _diff2_axis(v: torch.Tensor, spacing: float, axis: int) -> torch.Tensor:
  """沿指定轴做五点二阶差分 (内部中心差分, 边界 5 点单侧差分)."""
  n = v.shape[axis]
  if n < 5:
    raise ValueError(
      f"五点差分要求该轴至少 5 个网格点, 实际为 {n} (axis={axis})"
    )
  x = v.movedim(axis, -1)
  out = torch.empty_like(x)
  denom = 12.0 * spacing * spacing

  # 内部: (-f_{i-2} + 16 f_{i-1} - 30 f_i + 16 f_{i+1} - f_{i+2}) / (12 d²)
  out[..., 2:-2] = (
    -x[..., :-4]
    + 16.0 * x[..., 1:-3]
    - 30.0 * x[..., 2:-2]
    + 16.0 * x[..., 3:-1]
    - x[..., 4:]
  ) / denom

  # 左边界 (5 点单侧, 3 阶精度)
  out[..., 0] = (
    35.0 * x[..., 0]
    - 104.0 * x[..., 1]
    + 114.0 * x[..., 2]
    - 56.0 * x[..., 3]
    + 11.0 * x[..., 4]
  ) / denom
  out[..., 1] = (
    11.0 * x[..., 0]
    - 20.0 * x[..., 1]
    + 6.0 * x[..., 2]
    + 4.0 * x[..., 3]
    - x[..., 4]
  ) / denom

  # 右边界
  out[..., n - 2] = (
    -x[..., n - 5]
    + 4.0 * x[..., n - 4]
    + 6.0 * x[..., n - 3]
    - 20.0 * x[..., n - 2]
    + 11.0 * x[..., n - 1]
  ) / denom
  out[..., n - 1] = (
    11.0 * x[..., n - 5]
    - 56.0 * x[..., n - 4]
    + 114.0 * x[..., n - 3]
    - 104.0 * x[..., n - 2]
    + 35.0 * x[..., n - 1]
  ) / denom

  return out.movedim(-1, axis)


# --------------------------------------------------------------------- #
# 结果容器与全图计算
# --------------------------------------------------------------------- #
@dataclass
class MapDerivatives:
  """一张高程图上的全部导数/曲率场, 形状均为 [..., Ny, Nx]."""

  grad_x: torch.Tensor
  grad_y: torch.Tensor
  hess_xx: torch.Tensor
  hess_xy: torch.Tensor
  hess_yy: torch.Tensor
  slope: torch.Tensor
  mean_curvature: torch.Tensor
  gaussian_curvature: torch.Tensor

  def as_dict(self) -> dict[str, torch.Tensor]:
    return {name: getattr(self, name) for name in DERIVATIVE_COMPONENTS}

  def __getitem__(self, name: str) -> torch.Tensor:
    if name not in DERIVATIVE_COMPONENTS:
      raise KeyError(f"未知导数分量 '{name}', 可选: {DERIVATIVE_COMPONENTS}")
    return getattr(self, name)


def _as_height_tensor(h: torch.Tensor) -> torch.Tensor:
  if not isinstance(h, torch.Tensor):
    raise TypeError(f"高度图必须是 torch.Tensor, 实际为 {type(h).__name__}")
  if h.dim() < 2:
    raise ValueError(f"高度图至少需要 2 维 [..., Ny, Nx], 实际 shape={tuple(h.shape)}")
  if not h.is_floating_point():
    h = h.to(torch.get_default_dtype())
  elif h.dtype in (torch.float16, torch.bfloat16):
    # 差分对数值很敏感, 半精度会放大截断误差, 统一提升到 float32.
    h = h.to(torch.float32)
  return h


def _check_spacing(dx: float, dy: float) -> None:
  for name, value in (("dx", dx), ("dy", dy)):
    if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
      raise ValueError(f"{name} 必须是有限正数 (m/cell), 实际为 {value!r}")


def compute_derivatives(h: torch.Tensor, dx: float, dy: float) -> MapDerivatives:
  """对整张高程图计算梯度、Hessian、坡度与曲率.

  Args:
    h: [..., Ny, Nx] 高度网格, 单位为 m; i (倒数第二维) 沿 y, j (最后一维) 沿 x.
    dx: x 方向网格分辨率 [m/cell].
    dy: y 方向网格分辨率 [m/cell].

  Returns:
    MapDerivatives, 所有分量形状与 h 相同.

  Raises:
    TypeError: h 不是 torch.Tensor.
    ValueError: h 维数不足、任一网格轴少于 5 个点、或 dx/dy 非有限正数.
  """
  h = _as_height_tensor(h)
  _check_spacing(dx, dy)
  ny, nx = h.shape[-2], h.shape[-1]
  if ny < 5 or nx < 5:
    raise ValueError(
      f"五点差分要求每条轴至少 5 个网格点, 实际网格为 Ny={ny}, Nx={nx}"
    )

  grad_x = _diff1_axis(h, dx, axis=-1)
  grad_y = _diff1_axis(h, dy, axis=-2)
  hess_xx = _diff2_axis(h, dx, axis=-1)
  hess_yy = _diff2_axis(h, dy, axis=-2)
  # 混合导数: 先去 y 再对 x 求导 (与小斜率无关, 只需一阶差分算子可分离).
  hess_xy = _diff1_axis(grad_y, dx, axis=-1)

  one = torch.ones_like(grad_x)
  g2 = grad_x * grad_x + grad_y * grad_y
  slope = torch.sqrt(g2)
  mean_curvature = (
    (one + grad_y * grad_y) * hess_xx
    - 2.0 * grad_x * grad_y * hess_xy
    + (one + grad_x * grad_x) * hess_yy
  ) / (2.0 * (one + g2).pow(1.5))
  gaussian_curvature = (hess_xx * hess_yy - hess_xy * hess_xy) / (
    one + g2
  ).pow(2.0)

  # 输入无效 (NaN/inf) 的格点, 其导数一律视为无效.
  # 注意五点一阶差分在中心点权重为 0, 不显式屏蔽的话该点梯度会"借用"邻居,
  # 让缺失高度的地方看起来仍有有效导数; 这里统一置为 NaN, 语义更清楚.
  valid_input = torch.isfinite(h)
  nan = float("nan")
  components = MapDerivatives(
    grad_x=torch.where(valid_input, grad_x, torch.full_like(grad_x, nan)),
    grad_y=torch.where(valid_input, grad_y, torch.full_like(grad_y, nan)),
    hess_xx=torch.where(valid_input, hess_xx, torch.full_like(hess_xx, nan)),
    hess_xy=torch.where(valid_input, hess_xy, torch.full_like(hess_xy, nan)),
    hess_yy=torch.where(valid_input, hess_yy, torch.full_like(hess_yy, nan)),
    slope=torch.where(valid_input, slope, torch.full_like(slope, nan)),
    mean_curvature=torch.where(
      valid_input, mean_curvature, torch.full_like(mean_curvature, nan)
    ),
    gaussian_curvature=torch.where(
      valid_input, gaussian_curvature, torch.full_like(gaussian_curvature, nan)
    ),
  )
  return components


# --------------------------------------------------------------------- #
# 带缓存的高程图导数提供者 + 双线性插值
# --------------------------------------------------------------------- #
class HeightMapDerivativeProvider:
  """缓存一张高程图的导数场, 并支持任意连续坐标的双线性查询.

  典型用法:
      provider = HeightMapDerivativeProvider(dx=0.1, dy=0.1, origin=(-0.8, -0.5))
      derivs = provider.update(h)          # 每帧/每次地图更新调用一次
      gx = provider.query("grad_x", x, y)  # 同一帧内可任意多次查询, O(1)
  """

  def __init__(
    self,
    dx: float,
    dy: float,
    origin: tuple[float, float] = (0.0, 0.0),
  ):
    _check_spacing(dx, dy)
    x0, y0 = float(origin[0]), float(origin[1])
    if not (math.isfinite(x0) and math.isfinite(y0)):
      raise ValueError(f"origin 必须是有限值, 实际为 {origin!r}")
    self.dx = float(dx)
    self.dy = float(dy)
    self.origin = (x0, y0)
    self._derivatives: MapDerivatives | None = None

  # -- 缓存管理 -------------------------------------------------------- #
  def update(self, h: torch.Tensor) -> MapDerivatives:
    """计算并缓存一帧的导数场; 覆盖旧缓存."""
    self._derivatives = compute_derivatives(h, self.dx, self.dy)
    return self._derivatives

  def invalidate(self) -> None:
    """丢弃缓存; 之后必须先 update() 才能查询."""
    self._derivatives = None

  @property
  def derivatives(self) -> MapDerivatives:
    if self._derivatives is None:
      raise RuntimeError("导数场尚未计算, 请先调用 update(h)")
    return self._derivatives

  # -- 双线性插值 ------------------------------------------------------ #
  def query(
    self,
    component: str,
    x: float,
    y: float,
    *,
    out_of_bounds: OutOfBoundsMode = "nan",
    fill_value: float | None = None,
  ) -> torch.Tensor:
    """在连续坐标 (x, y) 处双线性插值某个导数分量.

    Args:
      component: DERIVATIVE_COMPONENTS 中的分量名, 例如 "grad_x".
      x, y: 连续坐标 [m] (标量).
      out_of_bounds: 越界策略.
        - "nan"  (默认): 返回 NaN;
        - "clamp": 截断到最近的网格边界点;
        - "raise": 抛出 ValueError.
      fill_value: 若 4 个角点中存在无效值 (NaN/inf), 用该值回退;
        None (默认) 表示保持 NaN.

    Returns:
      标量张量; 若地图带前导 batch 维 [..., Ny, Nx], 返回 [...] 形状的张量.

    Raises:
      KeyError: component 名称非法.
      ValueError: 越界且策略为 "raise", 或坐标为非有限值.
      RuntimeError: 尚未调用 update().
    """
    if component not in DERIVATIVE_COMPONENTS:
      raise KeyError(
        f"未知导数分量 '{component}', 可选: {DERIVATIVE_COMPONENTS}"
      )
    if out_of_bounds not in ("nan", "clamp", "raise"):
      raise ValueError(
        f"out_of_bounds 只能是 'nan' / 'clamp' / 'raise', 实际为 {out_of_bounds!r}"
      )
    return self._bilinear(
      self.derivatives[component],
      x,
      y,
      out_of_bounds=out_of_bounds,
      fill_value=fill_value,
    )

  def query_point(
    self,
    x: float,
    y: float,
    *,
    out_of_bounds: OutOfBoundsMode = "nan",
    fill_value: float | None = None,
  ) -> dict[str, torch.Tensor]:
    """一次返回 (x, y) 处所有导数/曲率分量."""
    return {
      name: self.query(
        name,
        x,
        y,
        out_of_bounds=out_of_bounds,
        fill_value=fill_value,
      )
      for name in DERIVATIVE_COMPONENTS
    }

  def _bilinear(
    self,
    field: torch.Tensor,
    x: float,
    y: float,
    *,
    out_of_bounds: OutOfBoundsMode,
    fill_value: float | None,
  ) -> torch.Tensor:
    nx = field.shape[-1]
    ny = field.shape[-2]
    x0, y0 = self.origin

    try:
      xf, yf = float(x), float(y)
    except (TypeError, ValueError) as exc:
      raise ValueError(f"查询坐标必须是标量, 实际为 x={x!r}, y={y!r}") from exc
    if not (math.isfinite(xf) and math.isfinite(yf)):
      raise ValueError(f"查询坐标必须是有限值, 实际为 x={xf}, y={yf}")

    # 连续坐标 -> 网格索引
    fx = (xf - x0) / self.dx
    fy = (yf - y0) / self.dy
    inside = (
      -_BOUNDS_EPS <= fx <= (nx - 1) + _BOUNDS_EPS
      and -_BOUNDS_EPS <= fy <= (ny - 1) + _BOUNDS_EPS
    )

    if not inside:
      x_min, x_max = x0, x0 + (nx - 1) * self.dx
      y_min, y_max = y0, y0 + (ny - 1) * self.dy
      if out_of_bounds == "raise":
        raise ValueError(
          f"查询坐标 ({xf}, {yf}) 超出高程图范围 "
          f"x∈[{x_min}, {x_max}], y∈[{y_min}, {y_max}]"
        )
      if out_of_bounds == "nan":
        return torch.full(
          field.shape[:-2],
          float("nan"),
          dtype=field.dtype,
          device=field.device,
        )
      # clamp
      fx = min(max(fx, 0.0), float(nx - 1))
      fy = min(max(fy, 0.0), float(ny - 1))

    # 取左侧/下侧格点索引与分数 (贴右/上边界时退化为最后一格, 分数=1)
    j0 = min(int(math.floor(fx)), nx - 2)
    i0 = min(int(math.floor(fy)), ny - 2)
    tx = fx - j0
    ty = fy - i0

    v00 = field[..., i0, j0]
    v01 = field[..., i0, j0 + 1]
    v10 = field[..., i0 + 1, j0]
    v11 = field[..., i0 + 1, j0 + 1]

    value = (
      (1.0 - ty) * (1.0 - tx) * v00
      + (1.0 - ty) * tx * v01
      + ty * (1.0 - tx) * v10
      + ty * tx * v11
    )

    valid = (
      torch.isfinite(v00)
      & torch.isfinite(v01)
      & torch.isfinite(v10)
      & torch.isfinite(v11)
    )
    if fill_value is None:
      fallback = torch.full_like(value, float("nan"))
    else:
      fallback = torch.as_tensor(
        fill_value, dtype=field.dtype, device=field.device
      ).expand_as(value)
    return torch.where(valid, value, fallback)
