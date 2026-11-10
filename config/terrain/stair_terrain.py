from collections.abc import Callable

from typing import Any

import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeneratorCfg


def pyramid_stairs_up(**overrides: Any) -> terrain_gen.BoxPyramidStairsTerrainCfg:
    """正金字塔楼梯配置（中心高，四周台阶向下）。"""
    defaults: dict[str, Any] = dict(
        step_height_range=(0.0, 0.12),
        step_width=0.3,
        platform_width=3.0,
        border_width=1.0,
    )
    defaults.update(overrides)
    return terrain_gen.BoxPyramidStairsTerrainCfg(**defaults)


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


STAIRS_UP_TERRAINS_CFG = _stairs_terrains_cfg(pyramid_stairs_up)
STAIRS_DOWN_TERRAINS_CFG = _stairs_terrains_cfg(pyramid_stairs_down)

# 旧名字保留为倒金字塔，避免影响其他任务。
STAIRS_TERRAINS_CFG = STAIRS_DOWN_TERRAINS_CFG
