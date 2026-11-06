from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import (
    SubTerrainCfg,
    TerrainGeneratorCfg,
    TerrainGeometry,
    TerrainOutput,
)


def _schedule_index(difficulty: float, length: int) -> int:
    """Map difficulty in [0, 1] to the nearest schedule index."""
    if length <= 1:
        return 0
    return int(round(float(np.clip(difficulty, 0.0, 1.0)) * (length - 1)))


@dataclass(kw_only=True)
class PlumBlossomFieldTerrainCfg(SubTerrainCfg):
    """中心平台 + 周围梅花桩地形。

    设计尺寸（全部使用米）：
    - 地形 patch：15m × 15m
    - 中心平台：7m × 7m × 10m
    - 难度 0：8 根 4m × 4m × 10m 大桩，围在中心平台四周
    - 难度 1~15：中心平台保持 7m × 7m（0~10），
      11~15 逐渐缩小到 0.8m × 0.8m；
      周围改为 0.362m × 0.362m × 10m 小桩，
      桩间距按难度随机排列，最大不超过 0.3m。
    """

    # --- 地形范围 ---
    field_size: tuple[float, float] = (15.0, 15.0)
    """梅花桩区域尺寸（米）。"""

    flat_border_width: float = 2.0
    """梅花桩区域外侧的平地宽度（米）。总 patch 尺寸会增加 2 倍该值。"""

    flat_color: tuple[float, float, float, float] = (0.45, 0.55, 0.40, 1.0)
    """外侧平地颜色。"""

    # --- 中心平台 ---
    platform_size_schedule: tuple[float, ...] = (
        7.00,
        7.00,
        7.00,
        7.00,
        7.00,
        7.00,
        7.00,
        7.00,
        7.00,
        7.00,
        7.00,
        5.76,
        4.52,
        3.28,
        2.04,
        0.80,
    )
    """按难度索引的中心平台边长。"""

    platform_height: float = 10.0
    """中心平台高度（米）。"""

    platform_color: tuple[float, float, float, float] = (0.28, 0.52, 0.72, 1.0)
    """中心平台颜色。"""

    # --- 梅花桩 ---
    pillar_size_schedule: tuple[float, ...] = (
        4.000,
        0.362,
        0.362,
        0.362,
        0.362,
        0.362,
        0.362,
        0.362,
        0.362,
        0.362,
        0.362,
        0.362,
        0.362,
        0.362,
        0.362,
        0.362,
    )
    """按难度索引的单根桩边长。"""

    pillar_height: float = 10.0
    """桩高度（米）。"""

    num_large_pillars: int = 8
    """难度 0 时的大桩数量。"""

    pillar_gap_max: float = 0.30
    """小桩之间允许的最大间距（米）。"""

    pillar_gap_min_level1: float = 0.05
    """难度 1 时的小桩最大间距（米）。"""

    pillar_jitter: float = 0.02
    """小桩位置随机抖动量（米）。"""

    platform_to_pillar_gap: float = 0.05
    """中心平台与周围小桩之间的最小净间距（米）。"""

    pillar_left_color: tuple[float, float, float, float] = (0.85, 0.35, 0.35, 1.0)
    """交替桩颜色 A。"""

    pillar_right_color: tuple[float, float, float, float] = (0.90, 0.60, 0.25, 1.0)
    """交替桩颜色 B。"""

    floor_thickness: float = 0.20
    """基准地面厚度（米）。"""

    floor_color: tuple[float, float, float, float] = (0.32, 0.35, 0.38, 1.0)
    """基准地面颜色。"""

    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    """接触摩擦参数。"""

    center_offset: tuple[float, float] = (0.0, 0.0)
    """在当前 sub-terrain patch 内的中心偏移。"""

    def _add_box(
        self,
        geometries: list[TerrainGeometry],
        body: mujoco.MjSpec.body,
        *,
        name: str,
        size: tuple[float, float, float],
        pos: tuple[float, float, float],
        color: tuple[float, float, float, float],
    ) -> None:
        geom = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=tuple(float(np.maximum(1e-4, s)) for s in size),
            pos=tuple(float(p) for p in pos),
        )
        geom.contype = 1
        geom.conaffinity = 1
        for idx, value in enumerate(self.friction):
            geom.friction[idx] = value
        geometries.append(TerrainGeometry(geom=geom, color=color))

    def _large_pillar_positions(self) -> list[tuple[float, float]]:
        """难度 0 的 8 根大桩中心，相对 patch 中心。

        针对 15m 梅花桩区域、7m 中心平台、4m 大桩：
        4m 宽的环形区域里放上方 3 根、下方 3 根、左右各 1 根。
        """
        field_half = self.field_size[0] / 2.0
        platform_size = float(self.platform_size_schedule[0])
        ring_center = platform_size / 2.0 + (
            field_half - platform_size / 2.0
        ) / 2.0
        return [
            (-5.0, ring_center),
            (0.0, ring_center),
            (5.0, ring_center),
            (-5.0, -ring_center),
            (0.0, -ring_center),
            (5.0, -ring_center),
            (ring_center, 0.0),
            (-ring_center, 0.0),
        ]

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        body = spec.body("terrain")
        geometries: list[TerrainGeometry] = []

        level = _schedule_index(
            difficulty, len(self.platform_size_schedule)
        )
        platform_size = float(self.platform_size_schedule[level])
        pillar_size = float(self.pillar_size_schedule[level])

        patch_half_x = self.size[0] / 2.0
        patch_half_y = self.size[1] / 2.0
        field_half_x = self.field_size[0] / 2.0
        field_half_y = self.field_size[1] / 2.0
        cx = patch_half_x + self.center_offset[0]
        cy = patch_half_y + self.center_offset[1]

        # 0. 基准地面
        self._add_box(
            geometries,
            body,
            name="floor",
            size=(patch_half_x, patch_half_y, self.floor_thickness / 2.0),
            pos=(cx, cy, -self.floor_thickness / 2.0),
            color=self.floor_color,
        )

        # 1. 中心平台
        platform_half = platform_size / 2.0
        self._add_box(
            geometries,
            body,
            name="center_platform",
            size=(
                platform_half,
                platform_half,
                self.platform_height / 2.0,
            ),
            pos=(cx, cy, self.platform_height / 2.0),
            color=self.platform_color,
        )

        # 1.1 梅花桩区域外侧的环形平地
        border_x = patch_half_x - field_half_x
        border_y = patch_half_y - field_half_y
        if border_x > 0.0 and border_y > 0.0:
            # 上下两条
            for sign in (1.0, -1.0):
                self._add_box(
                    geometries,
                    body,
                    name="outer_flat_top" if sign > 0 else "outer_flat_bottom",
                    size=(
                        patch_half_x,
                        border_y / 2.0,
                        self.platform_height / 2.0,
                    ),
                    pos=(
                        cx,
                        cy + sign * (field_half_y + border_y / 2.0),
                        self.platform_height / 2.0,
                    ),
                    color=self.flat_color,
                )
            # 左右两条（上下两条已经覆盖四个角）
            for sign in (1.0, -1.0):
                self._add_box(
                    geometries,
                    body,
                    name="outer_flat_right" if sign > 0 else "outer_flat_left",
                    size=(
                        border_x / 2.0,
                        field_half_y,
                        self.platform_height / 2.0,
                    ),
                    pos=(
                        cx + sign * (field_half_x + border_x / 2.0),
                        cy,
                        self.platform_height / 2.0,
                    ),
                    color=self.flat_color,
                )

        # 2. 梅花桩
        if level == 0:
            # 难度 0：8 根 4m × 4m × 10m 大桩
            large_half = pillar_size / 2.0
            for idx, (dx, dy) in enumerate(self._large_pillar_positions()):
                self._add_box(
                    geometries,
                    body,
                    name=f"large_pillar_{idx + 1}",
                    size=(
                        large_half,
                        large_half,
                        self.pillar_height / 2.0,
                    ),
                    pos=(cx + dx, cy + dy, self.pillar_height / 2.0),
                    color=(
                        self.pillar_left_color
                        if idx % 2 == 0
                        else self.pillar_right_color
                    ),
                )
        else:
            # 难度 1~15：0.362m 小桩，间距随难度增大，最大 0.3m。
            gap_max = self.pillar_gap_min_level1 + (
                self.pillar_gap_max - self.pillar_gap_min_level1
            ) * (level - 1) / 14.0
            gap_max = float(np.clip(gap_max, 0.0, self.pillar_gap_max))

            # 保证 base_gap + 2 * jitter <= gap_max，避免随机抖动后超过上限。
            # 六角排列下最近邻包含斜向邻居；留 2*sqrt(2)*jitter 的余量，
            # 保证两根桩相向抖动时不会重叠，同时最近邻最大间距不超过 gap_max。
            jitter = min(self.pillar_jitter, gap_max / 6.0)
            base_gap = max(gap_max - 2.0 * np.sqrt(2.0) * jitter, 0.0)
            pitch = pillar_size + base_gap
            # 六角/梅花交错排列：所有最近邻中心距都等于 pitch，
            # 这样相邻桩的最大净间距就是 base_gap + 2*jitter = gap_max。
            row_pitch = pitch * np.sqrt(3.0) / 2.0

            half_pillar = pillar_size / 2.0
            start_x = -field_half_x + half_pillar
            end_x = field_half_x - half_pillar
            start_y = -field_half_y + half_pillar
            end_y = field_half_y - half_pillar

            nx = int(np.floor((end_x - start_x) / pitch)) + 1
            ny = int(np.floor((end_y - start_y) / row_pitch)) + 1

            platform_clear = platform_half + half_pillar + self.platform_to_pillar_gap

            for iy in range(ny):
                y = start_y + iy * row_pitch
                # 奇数行交错半个 pitch
                row_offset = (pitch / 2.0) if (iy % 2 == 1) else 0.0
                for ix in range(nx):
                    x = start_x + ix * pitch + row_offset
                    if (
                        x < -field_half_x + half_pillar
                        or x > field_half_x - half_pillar
                    ):
                        continue

                    xj = x + float(rng.uniform(-jitter, jitter))
                    yj = y + float(rng.uniform(-jitter, jitter))
                    if (
                        abs(xj) > field_half_x - half_pillar
                        or abs(yj) > field_half_y - half_pillar
                    ):
                        continue

                    # 跳过中心平台及其净空区域
                    if abs(xj) < platform_clear and abs(yj) < platform_clear:
                        continue

                    self._add_box(
                        geometries,
                        body,
                        name=f"pillar_{iy}_{ix}",
                        size=(
                            half_pillar,
                            half_pillar,
                            self.pillar_height / 2.0,
                        ),
                        pos=(cx + xj, cy + yj, self.pillar_height / 2.0),
                        color=(
                            self.pillar_left_color
                            if (ix + iy) % 2 == 0
                            else self.pillar_right_color
                        ),
                    )

        # 3. 采样点：中心平台顶部中心
        origin = np.array(
            [float(cx), float(cy), float(self.platform_height)]
        )
        flat_patches = {
            "start": np.array([[float(cx), float(cy), float(self.platform_height)]]),
            "platform": np.array(
                [[float(cx), float(cy), float(self.platform_height)]]
            ),
            "goal": np.array(
                [[float(cx), float(cy), float(self.platform_height)]]
            ),
        }

        return TerrainOutput(
            origin=origin,
            geometries=geometries,
            flat_patches=flat_patches,
        )


def plum_blossom_field_terrain(
    **overrides: Any,
) -> PlumBlossomFieldTerrainCfg:
    return PlumBlossomFieldTerrainCfg(**overrides)


# 16 个难度：0~15
PLUM_BLOSSOM_FIELD_TERRAINS_CFG = TerrainGeneratorCfg(
    # 15m 梅花桩区域 + 四周各 2m 平地
    size=(19.0, 19.0),
    border_width=20.0,
    num_rows=16,
    num_cols=1,
    curriculum=True,
    sub_terrains={
        "plum_blossom_field": plum_blossom_field_terrain(
            proportion=1.0,
            field_size=(15.0, 15.0),
            flat_border_width=2.0,
        ),
    },
    add_lights=True,
)
