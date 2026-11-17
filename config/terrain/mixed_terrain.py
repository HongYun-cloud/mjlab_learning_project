"""Mixed terrains for multi-teacher and generalist-teacher training.

Four generators are provided:

- ``FLAT_STAIR_UP_TERRAINS_CFG``: flat + up pyramid (12 x 12 m patch);
- ``FLAT_STAIR_DOWN_TERRAINS_CFG``: flat + inverted pyramid (8 x 8 m patch),
  matched to the stair_down teacher geometry;
- ``FLAT_PLATFORM_STAIR_DOWN_TERRAINS_CFG``: flat + step-platform + inverted
  pyramid, with the platform on its 13-row curriculum schedule;
- ``FLAT_PLATFORM_STAIR_DOWN_PLUM_TERRAINS_CFG``: the same mix plus the plum
  blossom field, with a 10.6 x 10.6 m patch.

The generator runs in curriculum mode, so each sub-terrain gets exactly one
column.  Every env is assigned one column (``terrain_types``) and one
difficulty row (``terrain_levels``).  Teacher routing uses the column index;
a generalist teacher receives the per-env ``terrain_type`` one-hot instead.

This is also the extension point for more terrains: add another key to
``sub_terrains`` and a matching teacher checkpoint; the teacher group will
route it automatically.
"""

from __future__ import annotations

from mjlab.terrains import BoxFlatTerrainCfg, FlatPatchSamplingCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from myrobot_mjlab.config.terrain.plum_blossom_field_terrain import (
    plum_blossom_field_terrain,
)
from myrobot_mjlab.config.terrain.stair_terrain import (
    UpStairsApronTerrainCfg,
    pyramid_stairs_down,
)
from myrobot_mjlab.config.terrain.step_platform_terrain import (
    step_platform_terrain,
)
from myrobot_mjlab.config.terrain.stones_2rows_terrain import (
    stones_2rows_terrain,
)


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


# Flat + down-pyramid stairs.  Use the same 8 x 8 m patch as the stair_down
# teacher so the inverted-pyramid geometry is identical; flat gets an 8 x 8 m
# patch, which is enough for the 3.8 m target.
FLAT_STAIR_DOWN_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=2,  # ignored in curriculum mode
    curriculum=True,
    sub_terrains={
        "flat": BoxFlatTerrainCfg(proportion=0.5),
        "stair_down": pyramid_stairs_down(
            proportion=0.5,
            step_height_range=(0.0, 0.30),
        ),
    },
    add_lights=True,
)


# Flat + down-pyramid stairs + compact stone-2-rows.
#
# The stone-2-rows column is a compact 8 x 8 m version of the specialist
# terrain: 0.8 m center/outer platforms, 0.2 m pillars, 0.1 m gaps,
# 2 rows x 3 pillars.  This keeps the target distance <= 2.7 m and fits the
# mixed patch.  Allocation is flat : stair_down : stones_2rows = 1 : 1 : 1.
FLAT_STAIR_DOWN_STONES_2ROWS_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=13,
    num_cols=3,  # ignored in curriculum mode
    curriculum=True,
    sub_terrains={
        "flat": BoxFlatTerrainCfg(proportion=1.0),
        "stair_down": pyramid_stairs_down(
            proportion=1.0,
            step_height_range=(0.0, 0.30),
        ),
        "stones_2rows": stones_2rows_terrain(
            proportion=1.0,
            platform_size_schedule=(0.8,) * 13,
            pillar_size_schedule=(0.2,) * 13,
            pillar_gap_schedule=(0.1,) * 13,
            num_pillars_per_path=6,
            num_rows=2,
            platform_height=0.2,
            pillar_height=0.2,
            flat_patch_sampling={
                "start": FlatPatchSamplingCfg(
                    num_patches=1, patch_radius=0.5
                ),
                "goal": FlatPatchSamplingCfg(
                    num_patches=1, patch_radius=0.5
                ),
            },
        ),
    },
    add_lights=True,
)


# Flat + step-platform + down-pyramid stairs.
#
# The platform sub-terrain is copied from STEP_PLATFORM_TERRAINS_CFG: 2 fixed
# up-steps, a first platform, 5 mid platforms, and a second platform.  It uses
# 13 curriculum rows (0..12), so the generator uses 13 rows for every column.
# The platform target distance is computed per env from its difficulty; flat
# and stair_down keep a fixed 3.8 m target.
FLAT_PLATFORM_STAIR_DOWN_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=13,
    num_cols=3,  # ignored in curriculum mode
    curriculum=True,
    sub_terrains={
        "flat": BoxFlatTerrainCfg(proportion=1.0),
        "platform": step_platform_terrain(
            # 1:2:1 -> flat 25%, platform 50%, stair_down 25%.
            # curriculum 模式下 proportion 只控制 env 分配数量，不改地形尺寸。
            proportion=2.0,
            # Match STEP_PLATFORM_TERRAINS_CFG: wide platforms across the
            # patch, 5 mid stepping stones, curriculum gap schedule.
            step_length=10.0,
            platform_width=10.0,
            second_platform_width=10.0,
            num_mid_platforms=5,
            mid_platform_length=0.5,
            mid_platform_length_schedule=(
                0.500,
                0.500,
                0.500,
                0.415,
                0.415,
                0.415,
                0.330,
                0.330,
                0.330,
                0.245,
                0.245,
                0.245,
                0.160,
            ),
            mid_platform_width=10.0,
            mid_platform_height=0.36,
            gap_distance_range=(0.01, 0.35),
            gap_distance_schedule=(
                0.0100,
                0.0525,
                0.0950,
                0.0950,
                0.1375,
                0.1800,
                0.1800,
                0.2225,
                0.2650,
                0.2650,
                0.3075,
                0.3500,
                0.3500,
            ),
            vary_gap_with_difficulty=True,
            spawn_on_platform=False,
            spawn_distance_before=1.5,
            flat_patch_sampling={
                "goal": FlatPatchSamplingCfg(
                    num_patches=1, patch_radius=0.5
                ),
            },
        ),
        "stair_down": pyramid_stairs_down(
            proportion=1.0,
            step_height_range=(0.0, 0.30),
        ),
    },
    add_lights=True,
)


# Flat + stair-down + platform + plum blossom field.
#
# All columns share the 10.6 m patch required by the plum field:
#   plum field 6.6 m + 2 m outer flat on each side = 10.6 m.
# The stair-down border is widened so the inverted pyramid keeps 5 steps
# (the same geometry as the standalone stair-down task) at this patch size.
#
# Env allocation is controlled by ``proportion`` in curriculum mode:
#   flat : stair_down : platform : plum = 2 : 2 : 2 : 4
FLAT_PLATFORM_STAIR_DOWN_PLUM_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(10.60, 10.60),
    border_width=20.0,
    num_rows=13,
    num_cols=4,  # ignored in curriculum mode
    curriculum=True,
    sub_terrains={
        "flat": BoxFlatTerrainCfg(proportion=2.0),
        "stair_down": pyramid_stairs_down(
            proportion=2.0,
            step_height_range=(0.0, 0.30),
            # At size 10.6 this border keeps num_steps == 5, so the
            # inverted pyramid height matches the standalone stair_down task.
            border_width=2.2,
        ),
        "platform": step_platform_terrain(
            proportion=2.0,
            # Match the existing flat+platform+stair_down mix.
            step_length=10.0,
            platform_width=10.0,
            second_platform_width=10.0,
            num_mid_platforms=5,
            mid_platform_length=0.5,
            mid_platform_length_schedule=(
                0.500,
                0.500,
                0.500,
                0.415,
                0.415,
                0.415,
                0.330,
                0.330,
                0.330,
                0.245,
                0.245,
                0.245,
                0.160,
            ),
            mid_platform_width=10.0,
            mid_platform_height=0.36,
            gap_distance_range=(0.01, 0.35),
            gap_distance_schedule=(
                0.0100,
                0.0525,
                0.0950,
                0.0950,
                0.1375,
                0.1800,
                0.1800,
                0.2225,
                0.2650,
                0.2650,
                0.3075,
                0.3500,
                0.3500,
            ),
            vary_gap_with_difficulty=True,
            spawn_on_platform=False,
            spawn_distance_before=1.5,
            flat_patch_sampling={
                "goal": FlatPatchSamplingCfg(
                    num_patches=1, patch_radius=0.5
                ),
            },
        ),
        "plum": plum_blossom_field_terrain(
            proportion=4.0,
            field_size=(6.60, 6.60),
            flat_border_width=2.0,
            flat_patch_sampling={
                "goal": FlatPatchSamplingCfg(
                    num_patches=1, patch_radius=0.5
                ),
            },
        ),
    },
    add_lights=True,
)
