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
    if length <= 1:
        return 0
    return int(round(float(np.clip(difficulty, 0.0, 1.0)) * (length - 1)))


@dataclass(kw_only=True)
class Stones2RowsTerrainCfg(SubTerrainCfg):
    """Stones-2Rows：中心平台 + 四个方向的双排交错梅花桩。

    - 中心平台：1.5m（level 0~10），level 11/12 缩小到 1.15/0.8
    - 每个方向 2 排 × 5 根 = 10 根桩，按 5 组布置
    - 桩尺寸固定 0.2m，只有 gap 随难度从 0.01 增到 0.2
    - 每组 = 内外两排各一根桩，横向中心距 L，横向净距 gap
    - 组内第二根相对第一根的纵向错位 f：
      level 0 时 f=0（同组两根并排），level 9 时 f=L（完全交错）
    - 组间：上一组第二根到下一组第一根的纵向中心距 L，
      所以相邻两组第一根之间的纵向间距为 L + f
    - 第一组第一根到平台边缘 gap，最后一根到外侧平台 gap
    - 外侧平台尺寸与中心平台相同
    - 目标点固定在四个外侧平台中心，每次 reset 在 E/W/N/S 中随机选一个
    - 所有实体厚度 0.2m，其他区域为空域
    """

    # --- 难度 schedule ---
    platform_size_schedule: tuple[float, ...] = (
        1.50,
        1.50,
        1.50,
        1.50,
        1.50,
        1.50,
        1.50,
        1.50,
        1.50,
        1.50,
        1.50,
        1.15,
        0.80,
    )
    """中心平台和外侧平台的边长（13 级：0~12）。"""

    pillar_size_schedule: tuple[float, ...] = (
        0.200000,
        0.200000,
        0.200000,
        0.200000,
        0.200000,
        0.200000,
        0.200000,
        0.200000,
        0.200000,
        0.200000,
        0.200000,
        0.200000,
        0.200000,
    )
    """单根桩边长（所有难度固定 0.2m）。"""

    pillar_gap_schedule: tuple[float, ...] = (
        0.010000,
        0.031111,
        0.052222,
        0.073333,
        0.094444,
        0.115556,
        0.136667,
        0.157778,
        0.178889,
        0.200000,
        0.200000,
        0.200000,
        0.200000,
    )
    """沿路径的净间距（0~9 从 0.01 线性增到 0.2，之后保持 0.2）。"""

    num_pillars_per_path: int = 10
    """每个方向的总桩数。2 排 × 5 根。"""

    num_rows: int = 2
    """每个方向的桩排数。"""

    stagger_levels: int = 9
    """0~9 级内第二排的纵向错位从 0 线性增加到 L，之后保持不变。"""

    platform_height: float = 0.2
    """中心/外侧平台厚度。"""

    pillar_height: float = 0.2
    """桩厚度。"""

    # --- 颜色 ---
    platform_color: tuple[float, float, float, float] = (0.28, 0.52, 0.72, 1.0)
    outer_platform_color: tuple[float, float, float, float] = (
        0.22,
        0.65,
        0.55,
        1.0,
    )
    pillar_left_color: tuple[float, float, float, float] = (0.85, 0.35, 0.35, 1.0)
    pillar_right_color: tuple[float, float, float, float] = (0.90, 0.60, 0.25, 1.0)

    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    center_offset: tuple[float, float] = (0.0, 0.0)

    def _pitch(self, level: int) -> float:
        """L = gap + 桩 size。"""
        return float(self.pillar_size_schedule[level]) + float(
            self.pillar_gap_schedule[level]
        )

    def _row_stagger(self, level: int) -> float:
        """第二排相对第一排的纵向错位 f(L)。

        level 0 时 f=0（两排并排），level 9 时 f=L（两排完全交错）。
        10~12 保持 level 9 的 f=L，几何不变。
        """
        if self.stagger_levels <= 0:
            return self._pitch(level)
        t = float(np.clip(level, 0, self.stagger_levels)) / self.stagger_levels
        return t * self._pitch(level)

    def _target_distance(self, level: int) -> float:
        """中心平台中心到外侧平台中心的距离。"""
        platform = float(self.platform_size_schedule[level])
        size = float(self.pillar_size_schedule[level])
        gap = float(self.pillar_gap_schedule[level])
        pitch = size + gap
        stagger = self._row_stagger(level)
        per_row = self.num_pillars_per_path // self.num_rows
        # 最后一根桩到平台边缘的纵向距离：
        #   gap + per_row * stagger + (per_row - 1) * pitch
        return (
            platform
            + 2.0 * gap
            + size
            + per_row * stagger
            + (per_row - 1) * pitch
        )

    def _add_box(
        self,
        geometries: list[TerrainGeometry],
        body: mujoco.MjSpec.body,
        *,
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
        del rng
        body = spec.body("terrain")
        geometries: list[TerrainGeometry] = []

        level = _schedule_index(
            difficulty, len(self.platform_size_schedule)
        )
        platform_size = float(self.platform_size_schedule[level])
        pillar_size = float(self.pillar_size_schedule[level])
        gap = float(self.pillar_gap_schedule[level])

        half_patch_x = self.size[0] / 2.0
        half_patch_y = self.size[1] / 2.0
        cx = half_patch_x + self.center_offset[0]
        cy = half_patch_y + self.center_offset[1]

        # 中心平台
        platform_half = platform_size / 2.0
        self._add_box(
            geometries,
            body,
            size=(platform_half, platform_half, self.platform_height / 2.0),
            pos=(cx, cy, self.platform_height / 2.0),
            color=self.platform_color,
        )

        target_distance = self._target_distance(level)
        pillar_half = pillar_size / 2.0
        pitch = pillar_size + gap
        stagger = self._row_stagger(level)
        per_row = self.num_pillars_per_path // self.num_rows
        row_lateral = pitch / 2.0

        # 四个方向：东、西、北、南
        directions = (
            (1.0, 0.0),
            (-1.0, 0.0),
            (0.0, 1.0),
            (0.0, -1.0),
        )

        for dx, dy in directions:
            # 路径的横向单位向量
            px, py = -dy, dx

            # 外侧平台
            self._add_box(
                geometries,
                body,
                size=(
                    platform_half,
                    platform_half,
                    self.platform_height / 2.0,
                ),
                pos=(
                    cx + dx * target_distance,
                    cy + dy * target_distance,
                    self.platform_height / 2.0,
                ),
                color=self.outer_platform_color,
            )

            # 2 排 × 5 根，按 5 组布置：
            #   组内第二排相对第一排纵向错位 stagger（0 -> L）
            #   组间上一组第二根到下一组第一根纵向中心距 L
            # 所以相邻两组第一根之间的间距为 stagger + pitch。
            group_pitch = stagger + pitch
            for row in range(self.num_rows):
                row_sign = 1.0 if row % 2 == 0 else -1.0
                lateral = row_sign * row_lateral
                stagger_offset = 0.0 if row == 0 else stagger

                for idx in range(per_row):
                    longitudinal = (
                        platform_half
                        + gap
                        + pillar_half
                        + stagger_offset
                        + idx * group_pitch
                    )

                    self._add_box(
                        geometries,
                        body,
                        size=(
                            pillar_half,
                            pillar_half,
                            self.pillar_height / 2.0,
                        ),
                        pos=(
                            cx + dx * longitudinal + px * lateral,
                            cy + dy * longitudinal + py * lateral,
                            self.pillar_height / 2.0,
                        ),
                        color=(
                            self.pillar_left_color
                            if row == 0
                            else self.pillar_right_color
                        ),
                    )

        origin = np.array(
            [float(cx), float(cy), float(self.platform_height)]
        )
        flat_patches = {
            "start": np.array(
                [[float(cx), float(cy), float(self.platform_height)]]
            ),
            "platform": np.array(
                [[float(cx), float(cy), float(self.platform_height)]]
            ),
        }
        goal_names = ("goal_east", "goal_west", "goal_north", "goal_south")
        for goal_name, (dx, dy) in zip(goal_names, directions, strict=True):
            flat_patches[goal_name] = np.array(
                [[
                    float(cx + dx * target_distance),
                    float(cy + dy * target_distance),
                    float(self.platform_height),
                ]]
            )
        flat_patches["goal"] = flat_patches["goal_east"]

        return TerrainOutput(
            origin=origin,
            geometries=geometries,
            flat_patches=flat_patches,
        )


def stones_2rows_terrain(**overrides: Any) -> Stones2RowsTerrainCfg:
    return Stones2RowsTerrainCfg(**overrides)


# 13 个难度：0~12
STONES_2ROWS_TERRAINS_CFG = TerrainGeneratorCfg(
    # patch 需要容纳最远外侧平台（level 9/10 约 5.70m）
    size=(27.0, 27.0),
    border_width=20.0,
    num_rows=13,
    num_cols=1,
    curriculum=True,
    sub_terrains={
        "stones_2rows": stones_2rows_terrain(proportion=1.0),
    },
    add_lights=True,
)
