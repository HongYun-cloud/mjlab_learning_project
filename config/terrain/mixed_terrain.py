"""Mixed flat + stair-up terrain for multi-teacher distillation.

The generator runs in curriculum mode, so each sub-terrain gets exactly one
column.  Column 0 is flat and column 1 is the up-pyramid stair.  Every env is
assigned one column (``terrain_types``) and one difficulty row
(``terrain_levels``).  Teacher routing uses the column index.

This is also the extension point for more terrains: add another key to
``sub_terrains`` and a matching teacher checkpoint; the teacher group will
route it automatically.
"""

from __future__ import annotations

from mjlab.terrains import BoxFlatTerrainCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from myrobot_mjlab.config.terrain.stair_terrain import UpStairsApronTerrainCfg


FLAT_STAIR_UP_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(12.0, 12.0),
    border_width=20.0,
    num_rows=10,
    num_cols=2,  # ignored in curriculum mode; kept for readability
    curriculum=True,
    sub_terrains={
        # Column 0: flat.
        "flat": BoxFlatTerrainCfg(proportion=0.5),
        # Column 1: up pyramid with a 2 m flat apron.
        # To add a third teacher later, add another key here and a matching
        # key in TEACHER_CHECKPOINTS_FLAT_STAIR_UP; the runner routes by column.
        "stair_up": UpStairsApronTerrainCfg(
            proportion=0.5,
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
