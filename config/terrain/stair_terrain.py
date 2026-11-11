from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_generator import (
    SubTerrainCfg,
    TerrainGeneratorCfg,
    TerrainGeometry,
    TerrainOutput,
)


class BoxPyramidStairsUpTerrainCfg(terrain_gen.BoxPyramidStairsTerrainCfg):
    """正金字塔楼梯，原点设在底面平面（而不是顶平台）。"""

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        out = super().function(difficulty, spec, rng)
        # BoxPyramidStairsTerrainCfg 的 origin z 是顶平台高度；
        # up 任务需要从底面出生，所以把 origin 改到底面平面。
        out.origin[2] = 0.0
        return out


def pyramid_stairs_up(**overrides: Any) -> BoxPyramidStairsUpTerrainCfg:
    """正金字塔楼梯配置（中心高，四周台阶向下）。"""
    defaults: dict[str, Any] = dict(
        step_height_range=(0.0, 0.12),
        step_width=0.3,
        platform_width=3.0,
        border_width=1.0,
    )
    defaults.update(overrides)
    return BoxPyramidStairsUpTerrainCfg(**defaults)


@dataclass(kw_only=True)
class UpStairsApronTerrainCfg(SubTerrainCfg):
    """8m 正金字塔 + 四周平坦外圈，保证 5m 出生点在 patch 内。

    pyramid 和难度与原 8m 地形完全一致；外圈只增加可站立的平地。
    """

    step_height_range: tuple[float, float] = (0.0, 0.30)
    step_width: float = 0.3
    platform_width: float = 3.0
    border_width: float = 1.0
    pyramid_size: tuple[float, float] = (8.0, 8.0)
    apron: float = 2.0

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        pyramid = BoxPyramidStairsUpTerrainCfg(
            proportion=1.0,
            size=self.pyramid_size,
            step_height_range=self.step_height_range,
            step_width=self.step_width,
            platform_width=self.platform_width,
            border_width=self.border_width,
        )
        out = pyramid.function(difficulty, spec, rng)

        patch_size = (
            self.pyramid_size[0] + 2.0 * self.apron,
            self.pyramid_size[1] + 2.0 * self.apron,
        )
        for geometry in out.geometries:
            pos = np.array(geometry.geom.pos, dtype=float)
            pos[0] += self.apron
            pos[1] += self.apron
            geometry.geom.pos = pos

        step_height = self.step_height_range[0] + difficulty * (
            self.step_height_range[1] - self.step_height_range[0]
        )
        num_steps_x = int(
            (self.pyramid_size[0] - 2 * self.border_width - self.platform_width)
            / (2 * self.step_width)
        )
        num_steps_y = int(
            (self.pyramid_size[1] - 2 * self.border_width - self.platform_width)
            / (2 * self.step_width)
        )
        num_steps = max(0, min(num_steps_x, num_steps_y))
        depth = max((num_steps + 1) * step_height, 0.1) + 0.2

        body = spec.body("terrain")
        slab = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(patch_size[0] / 2.0, patch_size[1] / 2.0, depth / 2.0),
            pos=(patch_size[0] / 2.0, patch_size[1] / 2.0, -depth / 2.0),
        )
        slab_color = (
            out.geometries[0].color if out.geometries else (0.45, 0.45, 0.45, 1.0)
        )
        geometries = [TerrainGeometry(geom=slab, color=slab_color), *out.geometries]
        origin = np.array(
            [patch_size[0] / 2.0, patch_size[1] / 2.0, 0.0], dtype=float
        )
        return TerrainOutput(origin=origin, geometries=geometries)


def pyramid_stairs_down(
    **overrides: Any,
) -> terrain_gen.BoxInvertedPyramidStairsTerrainCfg:
    """倒金字塔楼梯配置（中心低，四周台阶向上）。"""
    defaults: dict[str, Any] = dict(
        step_height_range=(0.0, 0.12),
        step_width=0.3,
        platform_width=3.0,
        border_width=1.0,
    )
    defaults.update(overrides)
    return terrain_gen.BoxInvertedPyramidStairsTerrainCfg(**defaults)


# 兼容原有导入：pyramid_stairs_inv / pyramid_stairs 都指倒金字塔。
pyramid_stairs_inv = pyramid_stairs_down
pyramid_stairs = pyramid_stairs_down


@dataclass(kw_only=True)
class CombinedStairTerrainCfg(SubTerrainCfg):
    """正金字塔 + 1m 平地 + 倒金字塔。

    西侧是 up（正金字塔），东侧是 down（倒金字塔）。
    up 的底面和 down 的顶面都在 z=0 平面。
    """

    step_height_range: tuple[float, float] = (0.0, 0.30)
    step_width: float = 0.3
    platform_width: float = 3.0
    border_width: float = 1.0
    gap: float = 1.0
    up_size: tuple[float, float] = (8.0, 8.0)
    down_size: tuple[float, float] = (8.0, 8.0)
    west_apron: float = 3.0
    """up 西侧额外平坦出生区宽度（米）。出生点在其中，距最低台阶 2m。"""

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        up = BoxPyramidStairsUpTerrainCfg(
            proportion=1.0,
            size=self.up_size,
            step_height_range=self.step_height_range,
            step_width=self.step_width,
            platform_width=self.platform_width,
            border_width=self.border_width,
        )
        down = terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
            proportion=1.0,
            size=self.down_size,
            step_height_range=self.step_height_range,
            step_width=self.step_width,
            platform_width=self.platform_width,
            border_width=self.border_width,
        )

        up_out = up.function(difficulty, spec, rng)
        down_out = down.function(difficulty, spec, rng)

        up_base_half = self.up_size[0] / 2.0 - self.border_width
        down_base_half = self.down_size[0] / 2.0 - self.border_width
        up_center_x = self.west_apron + self.up_size[0] / 2.0
        down_center_x = up_center_x + up_base_half + self.gap + down_base_half
        shift_x = down_center_x - self.down_size[0] / 2.0

        geometries = []
        for geometry in up_out.geometries:
            pos = np.array(geometry.geom.pos, dtype=float)
            pos[0] += self.west_apron
            geometry.geom.pos = pos
            geometries.append(geometry)
        for geometry in down_out.geometries:
            pos = np.array(geometry.geom.pos, dtype=float)
            pos[0] += shift_x
            geometry.geom.pos = pos
            geometries.append(geometry)

        patch_height = max(self.up_size[1], self.down_size[1])
        patch_width = max(
            up_center_x + self.up_size[0] / 2.0,
            shift_x + self.down_size[0],
        )

        # up 西侧平坦出生区：补齐 [0, west_apron] 的地面，避免出生在相邻地形行。
        if self.west_apron > 0.0:
            step_height = self.step_height_range[0] + difficulty * (
                self.step_height_range[1] - self.step_height_range[0]
            )
            num_steps_x = int(
                (self.up_size[0] - 2 * self.border_width - self.platform_width)
                / (2 * self.step_width)
            )
            num_steps_y = int(
                (self.up_size[1] - 2 * self.border_width - self.platform_width)
                / (2 * self.step_width)
            )
            num_steps = max(0, min(num_steps_x, num_steps_y))
            depth = max((num_steps + 1) * step_height, 0.1) + 0.2
            body = spec.body("terrain")
            apron = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(
                    self.west_apron / 2.0,
                    patch_height / 2.0,
                    depth / 2.0,
                ),
                pos=(self.west_apron / 2.0, patch_height / 2.0, -depth / 2.0),
            )
            apron_color = (
                up_out.geometries[0].color
                if up_out.geometries
                else (0.45, 0.45, 0.45, 1.0)
            )
            geometries.insert(
                0, TerrainGeometry(geom=apron, color=apron_color)
            )

        origin = np.array(
            [patch_width / 2.0, patch_height / 2.0, 0.0], dtype=float
        )
        return TerrainOutput(origin=origin, geometries=geometries)


def _stairs_terrains_cfg(
    sub_terrain_factory: Callable[..., SubTerrainCfg],
) -> TerrainGeneratorCfg:
    return TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=10,
        num_cols=1,
        curriculum=True,
        sub_terrains={
            "pyramid_stairs": sub_terrain_factory(
                proportion=1.0,
                step_height_range=(0.0, 0.30),
            ),
        },
        add_lights=True,
    )


STAIRS_UP_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(12.0, 12.0),
    border_width=20.0,
    num_rows=10,
    num_cols=1,
    curriculum=True,
    sub_terrains={
        "pyramid_stairs": UpStairsApronTerrainCfg(
            proportion=1.0,
            step_height_range=(0.0, 0.30),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            pyramid_size=(8.0, 8.0),
            apron=2.0,
        ),
    },
    add_lights=True,
)
STAIRS_DOWN_TERRAINS_CFG = _stairs_terrains_cfg(pyramid_stairs_down)

STAIRS_COMBINED_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(18.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=1,
    curriculum=True,
    sub_terrains={
        "combined_stairs": CombinedStairTerrainCfg(
            proportion=1.0,
            step_height_range=(0.0, 0.30),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            gap=1.0,
            up_size=(8.0, 8.0),
            down_size=(8.0, 8.0),
            west_apron=3.0,
        ),
    },
    add_lights=True,
)

# 旧名字保留为倒金字塔，避免影响 Tamols/DTC/Velocity 等旧任务。
STAIRS_TERRAINS_CFG = STAIRS_DOWN_TERRAINS_CFG
