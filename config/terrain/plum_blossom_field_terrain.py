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
    - 地形区：6.52m × 6.52m，外侧再加 2m 平地
    - 中心平台：2m × 2m，厚度 0.2m
    - 难度 0：严格 7×7 = 49 根桩，桩间距和到外侧平台的距离都是 0.01m
    - 难度 1~9：桩尺寸 0.92m -> 0.20m，间距 0.01m -> 0.26m
    - 难度 10~12：保持 0.20m / 0.26m，中心平台 2.0m -> 0.8m
    - 地形区中间不再铺基准地面，只有平台、桩和外侧平地是实体薄片。
    """

    # --- 地形范围 ---
    field_size: tuple[float, float] = (6.52, 6.52)
    """地形区尺寸（米）：7*0.92m 桩 + 6*0.01m 桩间 gap + 2*0.01m 外侧 gap。"""

    flat_border_width: float = 2.0
    """地形区外侧的平地宽度（米）。总 patch 尺寸会增加 2 倍该值。"""

    flat_color: tuple[float, float, float, float] = (0.45, 0.55, 0.40, 1.0)
    """外侧平地颜色。"""

    # --- 中心平台 ---
    platform_size_schedule: tuple[float, ...] = (
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        1.6,
        1.2,
        0.8,
    )
    """按难度索引的中心平台边长（13 级：0~12）。"""

    platform_height: float = 0.2
    """中心平台厚度（米）。地形为薄片悬空结构，不是 10m 高台。"""

    platform_color: tuple[float, float, float, float] = (0.28, 0.52, 0.72, 1.0)
    """中心平台颜色。"""

    # --- 梅花桩 ---
    pillar_size_schedule: tuple[float, ...] = (
        0.92,
        0.84,
        0.76,
        0.68,
        0.60,
        0.52,
        0.44,
        0.36,
        0.28,
        0.20,
        0.20,
        0.20,
        0.20,
    )
    """按难度索引的单根桩边长（13 级：0~12，0.92 -> 0.20）。"""

    pillar_height: float = 0.2
    """桩厚度（米）。地形为薄片悬空结构，不是 10m 高柱。"""

    pillar_gap_max: float = 0.30
    """小桩之间允许的最大间距（米）。"""

    pillar_gap_min_level1: float = 0.05
    """难度 1 时的小桩最大间距（米）。"""

    pillar_gap_schedule: tuple[float, ...] | None = (
        0.01,
        0.10,
        0.12,
        0.14,
        0.16,
        0.18,
        0.20,
        0.22,
        0.24,
        0.26,
        0.26,  # level 10 开始缩小中心平台
        0.26,
        0.26,
    )
    """按难度索引的小桩最小间距（米）。设置后优先于线性插值。"""

    pillar_jitter: float = 0.02
    """小桩位置随机抖动量（米）。"""

    platform_to_pillar_gap: float = 0.05
    """中心平台与周围小桩之间的最小净间距（米）。"""

    pillar_left_color: tuple[float, float, float, float] = (0.85, 0.35, 0.35, 1.0)
    """交替桩颜色 A。"""

    pillar_right_color: tuple[float, float, float, float] = (0.90, 0.60, 0.25, 1.0)
    """交替桩颜色 B。"""

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

        # 不再创建整块基准地面：梅花桩区域中间是空的，
        # 只有中心平台、梅花桩和外侧平地是实体薄片。

        # 1. 中心平台尺寸（最后添加几何，确保叠在桩上面）
        platform_half = platform_size / 2.0

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
        if (
            self.pillar_gap_schedule is not None
            and level < len(self.pillar_gap_schedule)
        ):
            gap = float(self.pillar_gap_schedule[level])
        else:
            gap = self.pillar_gap_min_level1 + (
                self.pillar_gap_max - self.pillar_gap_min_level1
            ) * (level - 1) / max(len(self.platform_size_schedule) - 2, 1)
        gap = float(np.clip(gap, 0.0, self.pillar_gap_max))

        half_pillar = pillar_size / 2.0

        # 所有难度使用同一规则：按 size/gap 用正方形网格尽量填满地形区。
        # 网格覆盖整个地形区，允许与中心平台重叠；平台最后添加并叠在上面。
        pitch = pillar_size + gap
        num = int(np.floor((2.0 * field_half_x + gap) / pitch))
        num = max(1, num)
        span = num * pillar_size + (num - 1) * gap
        start = -span / 2.0 + half_pillar

        for iy in range(num):
            y = start + iy * pitch
            for ix in range(num):
                x = start + ix * pitch
                self._add_box(
                    geometries,
                    body,
                    name=f"pillar_{iy}_{ix}",
                    size=(
                        half_pillar,
                        half_pillar,
                        self.pillar_height / 2.0,
                    ),
                    pos=(cx + x, cy + y, self.pillar_height / 2.0),
                    color=(
                        self.pillar_left_color
                        if (ix + iy) % 2 == 0
                        else self.pillar_right_color
                    ),
                )

        # 2.1 中心平台最后添加，叠在桩上面。
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


# 13 个难度：0~12
PLUM_BLOSSOM_FIELD_TERRAINS_CFG = TerrainGeneratorCfg(
    # 6.52m 地形区 + 四周各 2m 平地
    size=(10.52, 10.52),
    border_width=20.0,
    num_rows=13,
    num_cols=1,
    curriculum=True,
    sub_terrains={
        "plum_blossom_field": plum_blossom_field_terrain(
            proportion=1.0,
            field_size=(6.52, 6.52),
            flat_border_width=2.0,
        ),
    },
    add_lights=True,
)
