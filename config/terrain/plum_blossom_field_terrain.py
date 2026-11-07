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
    - 梅花桩区域：15m × 15m，外侧再加 2m 平地
    - 中心平台：3m × 3m，厚度 0.2m
    - 难度 0~10：桩尺寸从 2.0m 逐渐减到 0.2m，
      桩间距从 0.05m 逐渐增大到 0.3m；
    - 难度 11~13：桩保持 0.2m / 0.3m，
      中心平台从 3.0m 逐渐缩小到 0.8m。
    - 梅花桩区域中间不再铺基准地面，只有平台、桩和外侧平地是实体薄片。
    """

    # --- 地形范围 ---
    field_size: tuple[float, float] = (8.92, 8.92)
    """地形区尺寸（米）：2m 平台 + 7*0.92m 桩 + 6*0.08m gap。"""

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

    num_pillars_level0: int = 49
    """难度 0 的实际桩数量；更高难度按 size/gap 在固定地形区内生成。"""

    pillar_gap_max: float = 0.30
    """小桩之间允许的最大间距（米）。"""

    pillar_gap_min_level1: float = 0.05
    """难度 1 时的小桩最大间距（米）。"""

    pillar_gap_schedule: tuple[float, ...] | None = (
        0.08,
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

        jitter = min(self.pillar_jitter, gap / 4.0)
        pitch = pillar_size + gap + 2.0 * jitter
        row_pitch = pitch * np.sqrt(3.0) / 2.0

        half_pillar = pillar_size / 2.0
        platform_clear = platform_half + half_pillar + self.platform_to_pillar_gap

        # 以中心平台为原点，向四周交错扩展。
        nx_half = int(np.floor((field_half_x - half_pillar) / pitch))
        ny_half = int(np.floor((field_half_y - half_pillar) / row_pitch))

        candidates: list[tuple[float, float, int, int]] = []
        for iy in range(-ny_half, ny_half + 1):
            y = iy * row_pitch
            row_offset = (pitch / 2.0) if (iy % 2 != 0) else 0.0
            for ix in range(-nx_half, nx_half + 1):
                x = ix * pitch + row_offset
                if abs(x) > field_half_x - half_pillar:
                    continue

                xj = x + float(rng.uniform(-jitter, jitter))
                yj = y + float(rng.uniform(-jitter, jitter))
                # 抖动后越界或进入平台净空时，退回未抖动位置。
                if (
                    abs(xj) > field_half_x - half_pillar
                    or abs(yj) > field_half_y - half_pillar
                    or (abs(xj) < platform_clear and abs(yj) < platform_clear)
                ):
                    xj, yj = x, y
                if abs(xj) < platform_clear and abs(yj) < platform_clear:
                    continue
                candidates.append((xj, yj, ix, iy))

        if level == 0:
            # 难度 0 固定 49 根，从候选位置中随机选 49 个。
            if len(candidates) > self.num_pillars_level0:
                chosen = rng.choice(
                    len(candidates),
                    size=self.num_pillars_level0,
                    replace=False,
                )
                selected = [candidates[int(i)] for i in chosen]
            else:
                selected = candidates
        else:
            # 更高难度在固定地形区内按当前 size/gap 生成。
            selected = candidates

        for xj, yj, ix, iy in selected:
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


# 13 个难度：0~12
PLUM_BLOSSOM_FIELD_TERRAINS_CFG = TerrainGeneratorCfg(
    # 8.92m 地形区 + 四周各 2m 平地
    size=(12.92, 12.92),
    border_width=20.0,
    num_rows=13,
    num_cols=1,
    curriculum=True,
    sub_terrains={
        "plum_blossom_field": plum_blossom_field_terrain(
            proportion=1.0,
            field_size=(8.92, 8.92),
            flat_border_width=2.0,
        ),
    },
    add_lights=True,
)
