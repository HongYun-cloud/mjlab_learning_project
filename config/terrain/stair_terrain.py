
from collections.abc import Callable
from typing import Any, TypeVar

import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_entity import TerrainEntity, TerrainEntityCfg
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeneratorCfg

def pyramid_stairs_inv(**overrides: Any) -> terrain_gen.BoxInvertedPyramidStairsTerrainCfg:
  """倒金字塔楼梯地形配置（中心出生点低，四周台阶高）。"""
  defaults: dict[str, Any] = dict(
    step_height_range=(0.0, 0.12),
    step_width=0.3,
    platform_width=3.0,
    border_width=1.0,
  )
  defaults.update(overrides)
  return terrain_gen.BoxInvertedPyramidStairsTerrainCfg(**defaults)


# 保留别名以兼容原有导入
pyramid_stairs = pyramid_stairs_inv

STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=1,
    curriculum=True,

    sub_terrains={
        "pyramid_stairs": pyramid_stairs_inv(
            proportion=1.0,
            step_height_range=(0.0, 0.30),
        ),
    },

    add_lights=True,
)