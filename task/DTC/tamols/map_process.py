"""terrain_scan 高程图处理: h0 -> 高斯滤波 h1, 虚拟平面 h2.

流程:
  1. 读取 velocity_env_cfg 中配置的 terrain_scan (RayCastSensor + GridPatternCfg):
     该传感器每条射线对应网格上的一个采样点, 按网格顺序重排即可得到原始高程图 h0;
  2. 对 h0 做高斯滤波, 得到平滑高程图 h1 (对应 TAMOLS 中的 h_s1);
  3. 虚拟平面 h2 (对应 TAMOLS 中的 h_s2):
       a. delta_h = h0 - median_filter(h0), 中值滤波窗口 3x3;
       b. 候选格分类: delta_h > 0 -> 落脚点(stepping stone), delta_h < 0 -> 间隙(gap),
          两者都令掩码 m=1 (要求 delta_threshold < |delta_h| < delta_max);
          平坦地面 (|delta_h| <= delta_threshold) 及其他异常 m=0;
       c. 掩码膨胀: 被膨胀掩码覆盖的格子, 其高度替换为邻域内最大值 h_d = dilate(h),
          从而加宽落脚点、缩小间隙;
       d. h2 = gaussian(h_d, sigma2), 其中 sigma2 > sigma1.

坐标与数据布局约定:
  - terrain_scan 使用 ray_alignment="yaw", 射线网格位于机身偏航系:
    x 向前, y 向左, 原点为传感器挂载 body (velocity_env_cfg 中为 robot 的 base body);
  - 高度取命中点的世界系 z (与既有 TAMOLS 实现中 h_raw 的语义一致);
  - 输出张量布局 [B, 1, Ny, Nx]: 第 -2 维是 y (行), 第 -1 维是 x (列),
    与 GridPatternCfg.generate_rays() 的展平顺序一致 (x 方向索引变化最快).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from mjlab.sensor import GridPatternCfg

if TYPE_CHECKING:
  from mjlab.sensor import RayCastSensor


@dataclass
class HeightMapProcessorCfg:
  """高程图处理参数."""

  sigma1: float = 1.0
  """h0 -> h1 的高斯核标准差 (单位: 栅格). <=0 时 h1 == h0."""

  sigma2: float = 2.0
  """h_d -> h2 的高斯核标准差 (单位: 栅格), 需要比 sigma1 大 (虚拟平面更平滑)."""

  median_size: int = 3
  """h0 中值滤波窗口尺寸 (像素), 用于计算高度差 delta_h."""

  delta_threshold: float = 0.0
  """|delta_h| 下阈值: |delta_h| <= delta_threshold 视为平坦地面, 不是候选格."""

  delta_max: float = 1.0
  """|delta_h| 上阈值: 超过视为异常/大落差, 不参与膨胀."""

  truncate: float = 4.0
  """高斯核截断半径 (以 sigma 为单位), 4.0 与 scipy.ndimage.gaussian_filter 默认一致."""

  max_range_ratio: float = 0.95
  """命中距离超过 max_distance * ratio 视为打空 (warp 会把打空距离饱和到接近 max_distance)."""

  eps: float = 1e-6
  """归一化卷积分母的下限."""


class HeightMapProcessor:
  """把 terrain_scan 的射线命中结果整理成高程图 h0, 并输出 h1 (平滑图) 与 h2 (虚拟平面)."""

  def __init__(self, cfg: HeightMapProcessorCfg | None = None):
    self.cfg = cfg if cfg is not None else HeightMapProcessorCfg()

    self._kernels: dict[tuple, torch.Tensor] = {}
    self._grid_shape: dict[int, tuple[int, int]] = {}

    self.valid: torch.Tensor | None = None
    """最近一次 extract_h0() 的有效性掩码 [B, 1, Ny, Nx] (bool)."""

    self.cell_class: torch.Tensor | None = None
    """最近一次 compute_h2() 的候选格分类 [B, 1, Ny, Nx] (int8): 1=落脚点, -1=间隙, 0=平坦/无效."""

  # ------------------------------------------------------------------ #
  # h0: 传感器 -> 原始高程图
  # ------------------------------------------------------------------ #
  @staticmethod
  def grid_shape(sensor: RayCastSensor) -> tuple[int, int]:
    """由传感器网格模式推出高程图尺寸 (Ny, Nx)."""
    pattern = sensor.cfg.pattern
    if not isinstance(pattern, GridPatternCfg):
      raise TypeError(
        f"terrain_scan 需要 GridPatternCfg 才能重排成高程图, 实际为 {type(pattern).__name__}"
      )
    res = float(pattern.resolution)
    size_x, size_y = pattern.size
    # 与 GridPatternCfg.generate_rays() 的 arange 采样点数一致.
    nx = int(round(size_x / res)) + 1
    ny = int(round(size_y / res)) + 1
    return ny, nx

  def extract_h0(self, sensor: RayCastSensor) -> tuple[torch.Tensor, torch.Tensor]:
    """从 terrain_scan 读取本帧命中点, 重排为原始高程图 h0.

    Returns:
      h0: [B, 1, Ny, Nx], 世界系 z 高度; 无效格填 0 (配合 valid 使用).
      valid: [B, 1, Ny, Nx], True 表示该格由真实命中得到.
    """
    data = sensor.data
    distances = data.distances  # [B, N]
    if sensor.num_frames != 1:
      raise ValueError(
        f"extract_h0() 仅支持单挂载帧的 terrain_scan, 实际 num_frames={sensor.num_frames}"
      )

    ny, nx = self._grid_shape_cached(sensor)
    num_rays = distances.shape[1]
    if num_rays != ny * nx:
      raise ValueError(
        f"terrain_scan 射线数 {num_rays} 与网格尺寸 ({ny}, {nx}) 不匹配"
      )

    # GridPatternCfg 展平顺序: index = iy * nx + ix -> reshape(B, ny, nx)
    h0 = data.hit_pos_w[..., 2].reshape(-1, ny, nx)

    max_dist = float(sensor.cfg.max_distance)
    valid = (
      (distances > 0.0)
      & (distances < self.cfg.max_range_ratio * max_dist)
      & torch.isfinite(h0.reshape(distances.shape))
    ).reshape(-1, ny, nx)

    h0 = torch.where(valid, h0, torch.zeros_like(h0))
    self.valid = valid.unsqueeze(1)
    return h0.unsqueeze(1), self.valid

  def _grid_shape_cached(self, sensor: RayCastSensor) -> tuple[int, int]:
    key = id(sensor)
    if key not in self._grid_shape:
      self._grid_shape[key] = self.grid_shape(sensor)
    return self._grid_shape[key]

  # ------------------------------------------------------------------ #
  # h1: 高斯滤波
  # ------------------------------------------------------------------ #
  def _get_kernel(
    self, sigma: float, device: torch.device, dtype: torch.dtype
  ) -> torch.Tensor:
    """构造并缓存 2D 高斯核 [1, 1, K, K]."""
    sigma = float(sigma)
    radius = max(1, int(self.cfg.truncate * sigma + 0.5))
    size = 2 * radius + 1
    key = (sigma, device, dtype, size)
    kernel = self._kernels.get(key)
    if kernel is None:
      coords = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
      k1d = torch.exp(-0.5 * (coords / sigma) ** 2)
      k2d = torch.outer(k1d, k1d)
      k2d = k2d / k2d.sum()
      kernel = k2d.view(1, 1, size, size).to(dtype)
      self._kernels[key] = kernel
    return kernel

  def gaussian_blur(
    self,
    h0: torch.Tensor,
    valid: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """对 h0 做高斯滤波得到 h1.

    采用归一化卷积 (masked normalized convolution): 只在有效格上加权,
    避免空洞 (未命中) 和地图边界被当作 0 高度拉低滤波结果.

    Args:
      h0: [B, 1, Ny, Nx] 原始高程图.
      valid: [B, 1, Ny, Nx] 有效性掩码; None 表示全部有效.

    Returns:
      h1: [B, 1, Ny, Nx], 平滑高程图.
    """
    if h0.dim() != 4:
      raise ValueError(f"h0 需为 [B, 1, Ny, Nx], 实际 shape={tuple(h0.shape)}")
    if self.cfg.sigma1 <= 0.0:
      return h0

    dtype = h0.dtype
    device = h0.device
    h0 = h0.to(torch.float32)
    w = (
      torch.ones_like(h0)
      if valid is None
      else valid.to(device=device, dtype=torch.float32)
    )

    kernel = self._get_kernel(self.cfg.sigma1, device, torch.float32)
    radius = kernel.shape[-1] // 2

    weighted = h0 * w
    num = F.conv2d(weighted, kernel, padding=radius)
    den = F.conv2d(w, kernel, padding=radius)

    h1 = num / den.clamp_min(self.cfg.eps)

    # 整片邻域都没有观测时, 退回该环境有效高度的均值; 一个有效点都没有则填 0.
    weight_sum = w.sum(dim=(1, 2, 3), keepdim=True)
    mean_h = (h0 * w).sum(dim=(1, 2, 3), keepdim=True) / weight_sum.clamp_min(
      self.cfg.eps
    )
    fallback = torch.where(weight_sum > 0, mean_h, torch.zeros_like(mean_h))
    h1 = torch.where(den > self.cfg.eps, h1, fallback.expand_as(h1))

    return h1.to(dtype)

  def _gaussian_smooth(self, h: torch.Tensor, sigma: float) -> torch.Tensor:
    """普通高斯平滑 (边界 replicate, 不归一化), 用于已经填充完整的 h_d."""
    if sigma <= 0.0:
      return h
    dtype = h.dtype
    h = h.to(torch.float32)
    kernel = self._get_kernel(sigma, h.device, torch.float32)
    radius = kernel.shape[-1] // 2
    h = F.pad(h, (radius,) * 4, mode="replicate")
    return F.conv2d(h, kernel).to(dtype)

  @staticmethod
  def median_filter(h: torch.Tensor, size: int) -> torch.Tensor:
    """3x3 (可配置) 中值滤波, h: [B, 1, Ny, Nx]."""
    radius = size // 2
    if radius == 0:
      return h
    hp = F.pad(h, (radius,) * 4, mode="replicate")
    windows = F.unfold(hp, kernel_size=size)  # [B, C*size*size, L], C=1
    return windows.median(dim=1).values.view_as(h)

  @staticmethod
  def dilate(h: torch.Tensor, mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """掩码膨胀 + 高度膨胀 (TAMOLS Eq. 虚拟地板).

    先把掩码 m 做 (2r+1)x(2r+1) 二值膨胀; 被膨胀掩码覆盖的格子,
    其高度替换为邻域内的最大值:
        h_d[i] = max{ h[j] | j 在 i 的 (2r+1) 邻域内 },  m_d[i] = 1
        h_d[i] = h[i],                                m_d[i] = 0
    取邻域最大值会把间隙抬到周围地表、把落脚点向四周加宽.
    """
    kernel_size = 2 * radius + 1
    covered = (
      F.max_pool2d(mask.to(h.dtype), kernel_size, stride=1, padding=radius) > 0
    )
    h_max = F.max_pool2d(h, kernel_size, stride=1, padding=radius)
    return torch.where(covered, h_max, h)

  @staticmethod
  def fill_invalid(h0: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """用该环境内有效高度的中位数填充未命中格, 避免 0 高度污染中值滤波/膨胀."""
    h_nan = h0.masked_fill(~valid, float("nan"))
    med = torch.nanmedian(h_nan.view(h0.shape[0], -1), dim=1).values
    med = torch.where(torch.isfinite(med), med, torch.zeros_like(med))
    return torch.where(valid, h0, med.view(-1, 1, 1, 1))

  # ------------------------------------------------------------------ #
  # h2: 虚拟平面
  # ------------------------------------------------------------------ #
  def classify_cells(
    self,
    delta_h: torch.Tensor,
    valid: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """把候选格分类为落脚点 / 间隙.

    分类规则 (要求该格有效):
      - delta_h >  delta_threshold 且 < delta_max ->  1 (落脚点 / stepping stone);
      - delta_h < -delta_threshold 且 > -delta_max -> -1 (间隙 / gap);
      - 其余 (|delta_h| <= delta_threshold 的平地, 或超阈值的异常) -> 0.

    Returns:
      [B, 1, Ny, Nx] int8 分类图.
    """
    if valid is None:
      valid = torch.ones_like(delta_h, dtype=torch.bool)
    else:
      valid = valid.to(device=delta_h.device, dtype=torch.bool)

    abs_delta = delta_h.abs()
    candidate = (
      (abs_delta > self.cfg.delta_threshold)
      & (abs_delta < self.cfg.delta_max)
      & valid
    )

    cell_class = torch.zeros_like(delta_h, dtype=torch.int8)
    cell_class = torch.where(
      candidate & (delta_h > 0), torch.ones_like(cell_class), cell_class
    )
    cell_class = torch.where(
      candidate & (delta_h < 0), -torch.ones_like(cell_class), cell_class
    )
    return cell_class
  def compute_h2(
    self,
    h0: torch.Tensor,
    valid: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """由 h0 计算虚拟平面 h2.

    步骤:
      1. delta_h = h0 - median_filter(h0);
      2. 候选格分类 (见 classify_cells): delta_h > 0 -> 落脚点, delta_h < 0 -> 间隙,
         两类都令掩码 m=1; 平坦地面 (|delta_h| <= delta_threshold) m=0;
      3. 掩码膨胀, 被膨胀覆盖的格子高度替换为邻域最大值 h_d = dilate(h, m),
         从而加宽落脚点、缩小间隙;
      4. h2 = gaussian(h_d, sigma2), sigma2 > sigma1.

    Args:
      h0: [B, 1, Ny, Nx] 原始高程图.
      valid: [B, 1, Ny, Nx] 有效性掩码; None 表示全部有效.

    Returns:
      h2: [B, 1, Ny, Nx], 虚拟平面高度.
    """
    if h0.dim() != 4:
      raise ValueError(f"h0 需为 [B, 1, Ny, Nx], 实际 shape={tuple(h0.shape)}")

    dtype = h0.dtype
    h0 = h0.to(torch.float32)
    if valid is None:
      valid = torch.ones_like(h0, dtype=torch.bool)
    else:
      valid = valid.to(device=h0.device, dtype=torch.bool)

    # 1. 无效格先填中位数, 再算高度差
    h = self.fill_invalid(h0, valid)
    h_med = self.median_filter(h, self.cfg.median_size)
    delta_h = h - h_med

    # 2. 候选格分类: 落脚点(+1) / 间隙(-1) / 平地或异常(0); 两类候选都进入掩码
    self.cell_class = self.classify_cells(delta_h, valid)
    mask = self.cell_class != 0

    # 3. 掩码膨胀 + 邻域取最大 -> 加宽落脚点、缩小间隙
    h_d = self.dilate(h, mask)

    # 4. 大标准差高斯平滑
    return self._gaussian_smooth(h_d, self.cfg.sigma2).to(dtype)

  # ------------------------------------------------------------------ #
  # 一步完成: sensor -> (h0, h1, h2)
  # ------------------------------------------------------------------ #
  def __call__(
    self, sensor: RayCastSensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """读取 terrain_scan 并返回 (h0, h1, h2), 均为 [B, 1, Ny, Nx]."""
    h0, valid = self.extract_h0(sensor)
    h1 = self.gaussian_blur(h0, valid)
    h2 = self.compute_h2(h0, valid)
    return h0, h1, h2
