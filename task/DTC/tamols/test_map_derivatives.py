"""map_derivatives 的单元测试.

覆盖: 平地、已知斜坡、已知曲率曲面、混合导数、网格点查询、网格间查询、
地图边界/越界、无效格、缓存与异常路径.

运行:
    python -m unittest myrobot_mjlab.task.DTC.tamols.test_map_derivatives -v
或直接:
    python test_map_derivatives.py
"""

from __future__ import annotations

import math
import unittest

import torch

from myrobot_mjlab.task.DTC.tamols.map_derivatives import (
  DERIVATIVE_COMPONENTS,
  HeightMapDerivativeProvider,
  compute_derivatives,
)

DX = 0.1
DY = 0.2
NY = 7
NX = 9


def make_grid(
  ny: int = NY,
  nx: int = NX,
  dx: float = DX,
  dy: float = DY,
  origin: tuple[float, float] = (0.0, 0.0),
  dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
  """返回网格点坐标 X, Y, 形状均为 [ny, nx]."""
  xs = origin[0] + torch.arange(nx, dtype=dtype) * dx
  ys = origin[1] + torch.arange(ny, dtype=dtype) * dy
  return torch.meshgrid(xs, ys, indexing="xy")


class TestFullMapDerivatives(unittest.TestCase):
  def test_flat_plane(self):
    """平地: 所有导数与曲率分量恒为 0, 形状与输入一致."""
    h = torch.full((NY, NX), 2.5, dtype=torch.float64)
    d = compute_derivatives(h, DX, DY)
    for name in DERIVATIVE_COMPONENTS:
      value = getattr(d, name)
      self.assertEqual(tuple(value.shape), tuple(h.shape))
      self.assertTrue(
        torch.allclose(value, torch.zeros_like(h), atol=1e-12), name
      )

  def test_linear_ramp(self):
    """已知斜坡: 梯度恒定, Hessian/曲率为 0; 边界点也精确 (单侧差分)."""
    X, Y = make_grid()
    a, b, c = 0.3, -0.7, 2.0
    h = a * X + b * Y + c

    d = compute_derivatives(h, DX, DY)
    zeros = torch.zeros_like(h)

    self.assertTrue(torch.allclose(d.grad_x, torch.full_like(h, a), atol=1e-12))
    self.assertTrue(torch.allclose(d.grad_y, torch.full_like(h, b), atol=1e-12))
    for name in ("hess_xx", "hess_xy", "hess_yy", "mean_curvature", "gaussian_curvature"):
      self.assertTrue(torch.allclose(getattr(d, name), zeros, atol=1e-10), name)
    self.assertTrue(
      torch.allclose(
        d.slope, torch.full_like(h, math.hypot(a, b)), atol=1e-12
      )
    )

  def test_quadratic_curvature(self):
    """已知曲率曲面 h = x² - y²: 检查 Hessian、平均曲率与高斯曲率."""
    X, Y = make_grid()
    kx, ky = 2.0, -2.0
    h = 0.5 * kx * X**2 + 0.5 * ky * Y**2

    d = compute_derivatives(h, DX, DY)
    gx, gy = kx * X, ky * Y
    g2 = gx**2 + gy**2
    hxx = torch.full_like(h, kx)
    hyy = torch.full_like(h, ky)

    self.assertTrue(torch.allclose(d.grad_x, gx, atol=1e-9))
    self.assertTrue(torch.allclose(d.grad_y, gy, atol=1e-9))
    self.assertTrue(torch.allclose(d.hess_xx, hxx, atol=1e-9))
    self.assertTrue(torch.allclose(d.hess_yy, hyy, atol=1e-9))
    self.assertTrue(
      torch.allclose(d.hess_xy, torch.zeros_like(h), atol=1e-9)
    )

    mean_expected = (
      (1.0 + gy**2) * hxx - 2.0 * gx * gy * 0.0 + (1.0 + gx**2) * hyy
    ) / (2.0 * (1.0 + g2) ** 1.5)
    gauss_expected = (hxx * hyy) / (1.0 + g2) ** 2
    self.assertTrue(torch.allclose(d.mean_curvature, mean_expected, atol=1e-9))
    self.assertTrue(
      torch.allclose(d.gaussian_curvature, gauss_expected, atol=1e-9)
    )
    # 鞍面: 高斯曲率处处为负
    self.assertTrue(bool((d.gaussian_curvature < 0).all()))

  def test_mixed_derivative(self):
    """h = x*y: hess_xy 恒为 1, 包括边界 (单侧差分对双线性函数精确)."""
    X, Y = make_grid()
    h = X * Y
    d = compute_derivatives(h, DX, DY)
    self.assertTrue(torch.allclose(d.grad_x, Y, atol=1e-10))
    self.assertTrue(torch.allclose(d.grad_y, X, atol=1e-10))
    self.assertTrue(
      torch.allclose(d.hess_xy, torch.ones_like(h), atol=1e-10)
    )
    self.assertTrue(torch.allclose(d.hess_xx, torch.zeros_like(h), atol=1e-10))
    self.assertTrue(torch.allclose(d.hess_yy, torch.zeros_like(h), atol=1e-10))

  def test_batched_leading_dims(self):
    """支持 [B, Ny, Nx] 等高维前导 batch 维度, 形状保持不变."""
    X, Y = make_grid()
    base = X * Y
    h = torch.stack([base, 2.0 * base])  # [2, NY, NX]
    d = compute_derivatives(h, DX, DY)
    self.assertEqual(tuple(d.grad_x.shape), (2, NY, NX))
    self.assertTrue(torch.allclose(d.grad_x[1], 2.0 * Y, atol=1e-10))

  def test_invalid_cells_propagate(self):
    """NaN 视为缺失: 模板触到它的点输出 NaN, 未触到的点保持有效."""
    ny, nx = 11, 13
    X, Y = make_grid(ny=ny, nx=nx)
    h = X * Y
    h[5, 6] = float("nan")
    d = compute_derivatives(h, DX, DY)

    # x 方向: 同一行 ±2 格内受影响
    self.assertTrue(bool(torch.isnan(d.grad_x[5, 6])))
    self.assertTrue(bool(torch.isnan(d.grad_x[5, 4])))
    self.assertTrue(bool(torch.isnan(d.grad_x[5, 8])))
    self.assertFalse(bool(torch.isnan(d.grad_x[5, 3])))
    self.assertFalse(bool(torch.isnan(d.grad_x[5, 9])))
    # 其它行不受 x 方向差分影响
    self.assertFalse(bool(torch.isnan(d.grad_x[6, 6])))
    # y 方向: 同一列 ±2 格内受影响
    self.assertTrue(bool(torch.isnan(d.grad_y[3, 6])))
    self.assertFalse(bool(torch.isnan(d.grad_y[2, 6])))
    # NaN 不会扩散到整张图
    self.assertTrue(bool(torch.isfinite(d.grad_x[0, 0])))
    self.assertTrue(bool(torch.isfinite(d.grad_y[-1, -1])))

  def test_small_grid_raises(self):
    with self.assertRaises(ValueError):
      compute_derivatives(torch.zeros(4, 9), DX, DY)
    with self.assertRaises(ValueError):
      compute_derivatives(torch.zeros(9, 4), DX, DY)

  def test_bad_spacing_raises(self):
    h = torch.zeros(NY, NX)
    for dx, dy in ((0.0, DY), (DX, -0.1), (float("nan"), DY), (float("inf"), DY)):
      with self.assertRaises(ValueError):
        compute_derivatives(h, dx, dy)

  def test_non_tensor_raises(self):
    with self.assertRaises(TypeError):
      compute_derivatives([[0.0] * NX] * NY, DX, DY)  # type: ignore[arg-type]


class TestInterpolation(unittest.TestCase):
  def setUp(self):
    self.X, self.Y = make_grid()
    # h = x² - y² -> grad_x = 2x (沿 x 线性), grad_y = -2y
    self.h = self.X**2 - self.Y**2
    self.provider = HeightMapDerivativeProvider(DX, DY)
    self.provider.update(self.h)

  def test_query_at_grid_points(self):
    """网格点查询必须精确复现该点的导数场值."""
    for i in (0, 3, NY - 1):
      for j in (0, 4, NX - 1):
        x, y = float(j * DX), float(i * DY)
        self.assertAlmostEqual(
          float(self.provider.query("grad_x", x, y)), 2.0 * x, places=10
        )
        self.assertAlmostEqual(
          float(self.provider.query("grad_y", x, y)), -2.0 * y, places=10
        )
        self.assertAlmostEqual(
          float(self.provider.query("hess_xx", x, y)), 2.0, places=10
        )

  def test_between_grid_points(self):
    """网格间查询: 对线性变化的 grad 场, 双线性插值应精确."""
    j, i = 3, 3
    x = (j + 0.5) * DX
    y = (i + 0.5) * DY
    self.assertAlmostEqual(float(self.provider.query("grad_x", x, y)), 2.0 * x, places=10)
    self.assertAlmostEqual(float(self.provider.query("grad_y", x, y)), -2.0 * y, places=10)
    # 常数场插值后仍为常数
    self.assertAlmostEqual(float(self.provider.query("hess_yy", x, y)), -2.0, places=10)

  def test_query_point_returns_all_components(self):
    values = self.provider.query_point(DX, DY)
    self.assertEqual(set(values), set(DERIVATIVE_COMPONENTS))
    for name, value in values.items():
      self.assertTrue(bool(torch.isfinite(value)), name)
      self.assertEqual(tuple(value.shape), ())  # 无 batch 维时返回 0 维张量

  def test_map_boundary(self):
    """边界查询: 四个角点可用, 且与解析值一致."""
    x_max = (NX - 1) * DX
    y_max = (NY - 1) * DY
    for x, y in ((0.0, 0.0), (x_max, 0.0), (0.0, y_max), (x_max, y_max)):
      self.assertAlmostEqual(float(self.provider.query("grad_x", x, y)), 2.0 * x, places=10)
      self.assertAlmostEqual(float(self.provider.query("grad_y", x, y)), -2.0 * y, places=10)

  def test_out_of_bounds_policies(self):
    x, y = -DX, -DY  # 图外一格
    self.assertTrue(bool(torch.isnan(self.provider.query("grad_x", x, y))))
    clamped = float(self.provider.query("grad_x", x, y, out_of_bounds="clamp"))
    self.assertAlmostEqual(clamped, 0.0, places=10)  # 角点 (0,0) 的 grad_x
    with self.assertRaises(ValueError):
      self.provider.query("grad_x", x, y, out_of_bounds="raise")
    with self.assertRaises(ValueError):
      self.provider.query("grad_x", 0.0, 0.0, out_of_bounds="bogus")  # type: ignore[arg-type]

  def test_bad_query_inputs(self):
    with self.assertRaises(KeyError):
      self.provider.query("not_a_component", 0.0, 0.0)
    with self.assertRaises(ValueError):
      self.provider.query("grad_x", float("nan"), 0.0)

  def test_invalid_corner_policies(self):
    """角点无效: 默认返回 NaN, 传入 fill_value 时回退."""
    ny, nx = 11, 13
    X, Y = make_grid(ny=ny, nx=nx)
    h = X * Y
    h[5, 6] = float("nan")
    provider = HeightMapDerivativeProvider(DX, DY)
    provider.update(h)

    x, y = 6 * DX, 5 * DY  # 正好是无效网格点
    self.assertTrue(bool(torch.isnan(provider.query("grad_x", x, y))))
    self.assertAlmostEqual(
      float(provider.query("grad_x", x, y, fill_value=-1.0)), -1.0, places=12
    )
    # 远离无效点的查询不受影响
    self.assertTrue(
      bool(torch.isfinite(provider.query("grad_x", 0.0, 0.0)))
    )

  def test_origin_offset(self):
    """origin 指定 h[0,0] 的世界坐标; 地图按世界坐标构建, 查询结果随之平移."""
    origin = (-0.8, -0.5)
    X, Y = make_grid(origin=origin)  # 用世界坐标构造高度图
    provider = HeightMapDerivativeProvider(DX, DY, origin=origin)
    provider.update(X**2 - Y**2)
    x = origin[0] + 3 * DX
    y = origin[1] + 4 * DY
    self.assertAlmostEqual(float(provider.query("grad_x", x, y)), 2.0 * x, places=10)
    self.assertAlmostEqual(float(provider.query("grad_y", x, y)), -2.0 * y, places=10)

  def test_batched_query(self):
    h = torch.stack([self.h, 2.0 * self.h])  # [2, NY, NX]
    provider = HeightMapDerivativeProvider(DX, DY)
    provider.update(h)
    x, y = 3 * DX, 4 * DY
    gx = provider.query("grad_x", x, y)
    self.assertEqual(tuple(gx.shape), (2,))
    self.assertAlmostEqual(float(gx[0]), 2.0 * x, places=10)
    self.assertAlmostEqual(float(gx[1]), 4.0 * x, places=10)
    # 越界 NaN 也保持 batch 形状
    oob = provider.query("grad_x", -1.0, -1.0)
    self.assertEqual(tuple(oob.shape), (2,))
    self.assertTrue(bool(torch.isnan(oob).all()))

  def test_cache_lifecycle(self):
    d1 = self.provider.update(self.h)
    self.assertIs(self.provider.derivatives, d1)
    _ = self.provider.query("grad_x", 0.0, 0.0)
    self.assertIs(self.provider.derivatives, d1)  # 查询不重算

    self.provider.invalidate()
    with self.assertRaises(RuntimeError):
      _ = self.provider.derivatives

    d2 = self.provider.update(self.h)
    self.assertIsNot(d1, d2)


if __name__ == "__main__":
  unittest.main(verbosity=2)
