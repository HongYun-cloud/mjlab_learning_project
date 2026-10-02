
from collections.abc import Callable
from typing import Any, TypeVar

import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_entity import TerrainEntity, TerrainEntityCfg
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeneratorCfg

def pyramid_stairs(**overrides: Any) -> terrain_gen.BoxPyramidStairsTerrainCfg:
  defaults: dict[str, Any] = dict(
    step_height_range=(0.0, 0.2),
    step_width=0.3,
    platform_width=3.0,
    border_width=1.0,
  )
  defaults.update(overrides)
  return terrain_gen.BoxPyramidStairsTerrainCfg(**defaults)

STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=1,

    sub_terrains={
        "pyramid_stairs": pyramid_stairs(
            proportion=1.0,
            step_height_range=(0.0, 0.1),
        ),
    },

    add_lights=True,
)