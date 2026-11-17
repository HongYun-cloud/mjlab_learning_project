"""Composite half-competition map terrain.

The map is a single narrow patch ordered west to east:

    flat -> 0.2 m step down -> 3 x 0.12 m stairs -> 0.8 m platform
    -> 0.36 m gap -> 0.8 m platform -> 20 deg ramp down
    -> stone-2-rows finish (two 0.8 x 0.8 x 0.1 m platforms,
       6 staggered 0.2 x 0.2 x 0.2 m pillars, 0.1 m platform gaps)

Only the path is solid; the area outside the path is empty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import (
    FlatPatchSamplingCfg,
    SubTerrainCfg,
    TerrainGeneratorCfg,
    TerrainGeometry,
    TerrainOutput,
)


@dataclass(kw_only=True)
class HalfCompetitionMapTerrainCfg(SubTerrainCfg):
    """One-patch half competition map with fixed geometry."""

    # --- Start / lower level ---
    flat_length: float = 1.0
    """West-to-east length of the starting flat platform (m)."""
    path_width: float = 1.0
    """Lateral width of the starting flat and lower ground (m)."""
    stair_down_height: float = 0.2
    """Height drop from the start flat to the lower level (m)."""
    stair_down_run: float = 0.3
    """Flat landing length after the step down (m)."""

    # --- Platform section ---
    step_height: float = 0.12
    num_steps: int = 3
    step_depth: float = 0.32
    """Run length of each up-step (m)."""
    platform_size: float = 0.8
    """Side length of each of the two platforms (m)."""
    platform_gap: float = 0.36
    """Horizontal gap between the two platforms (m)."""
    ramp_angle_deg: float = 20.0
    ramp_thickness: float = 0.1

    # --- Stone-2-rows section ---
    stone_platform_size: float = 0.8
    stone_platform_thickness: float = 0.1
    stone_pillar_size: float = 0.2
    stone_pillar_height: float = 0.2
    stone_platform_gap: float = 0.1
    """Gap from each side platform to the nearest pillar (m)."""
    stone_pillar_gap: float = 0.1
    """Longitudinal gap between pillars within a row (m)."""
    stone_lateral_gap: float = 0.1
    """Lateral gap between the two staggered pillar rows (m)."""
    stone_pillars_per_row: int = 3
    stone_rows: int = 2

    # --- Geometry / collision ---
    base_thickness: float = 0.2
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)

    # --- Colors ---
    flat_color: tuple[float, float, float, float] = (0.45, 0.45, 0.45, 1.0)
    lower_color: tuple[float, float, float, float] = (0.28, 0.28, 0.28, 1.0)
    step_color: tuple[float, float, float, float] = (0.32, 0.55, 0.78, 1.0)
    platform_color: tuple[float, float, float, float] = (0.28, 0.52, 0.72, 1.0)
    ramp_color: tuple[float, float, float, float] = (0.86, 0.52, 0.22, 1.0)
    stone_platform_color: tuple[float, float, float, float] = (
        0.22,
        0.65,
        0.55,
        1.0,
    )
    stone_left_color: tuple[float, float, float, float] = (0.85, 0.35, 0.35, 1.0)
    stone_right_color: tuple[float, float, float, float] = (0.90, 0.60, 0.25, 1.0)

    # --- Derived geometry ---
    @property
    def lower_z(self) -> float:
        return -float(self.stair_down_height)

    @property
    def platform_top_z(self) -> float:
        return self.lower_z + float(self.num_steps) * float(self.step_height)

    @property
    def stone_platform_top_z(self) -> float:
        return self.lower_z + float(self.stone_platform_thickness)

    @property
    def stone_pillar_top_z(self) -> float:
        return self.lower_z + float(self.stone_pillar_height)

    @property
    def step_start_x(self) -> float:
        return float(self.flat_length) + float(self.stair_down_run)

    @property
    def platform1_start_x(self) -> float:
        return self.step_start_x + float(self.num_steps) * float(self.step_depth)

    @property
    def platform1_end_x(self) -> float:
        return self.platform1_start_x + float(self.platform_size)

    @property
    def platform2_start_x(self) -> float:
        return self.platform1_end_x + float(self.platform_gap)

    @property
    def platform2_end_x(self) -> float:
        return self.platform2_start_x + float(self.platform_size)

    @property
    def ramp_horizontal_length(self) -> float:
        drop = self.platform_top_z - self.lower_z
        return drop / math.tan(math.radians(float(self.ramp_angle_deg)))

    @property
    def ramp_end_x(self) -> float:
        return self.platform2_end_x + self.ramp_horizontal_length

    @property
    def stone_west_start_x(self) -> float:
        return self.ramp_end_x

    @property
    def stone_west_end_x(self) -> float:
        return self.stone_west_start_x + float(self.stone_platform_size)

    @property
    def stone_pillar_start_x(self) -> float:
        return self.stone_west_end_x + float(self.stone_platform_gap)

    @property
    def stone_pillar_pitch(self) -> float:
        return float(self.stone_pillar_size) + float(self.stone_pillar_gap)

    @property
    def stone_pillar_stagger(self) -> float:
        return self.stone_pillar_pitch / 2.0

    @property
    def stone_last_pillar_east_x(self) -> float:
        per_row = int(self.stone_pillars_per_row)
        half = float(self.stone_pillar_size) / 2.0
        # The staggered row is the furthest east.
        return (
            self.stone_pillar_start_x
            + half
            + (per_row - 1) * self.stone_pillar_pitch
            + self.stone_pillar_stagger
            + half
        )

    @property
    def stone_east_start_x(self) -> float:
        return self.stone_last_pillar_east_x + float(self.stone_platform_gap)

    @property
    def stone_east_end_x(self) -> float:
        return self.stone_east_start_x + float(self.stone_platform_size)

    @property
    def start_center_x(self) -> float:
        return float(self.flat_length) / 2.0

    @property
    def goal_center_x(self) -> float:
        return self.stone_east_start_x + float(self.stone_platform_size) / 2.0

    @property
    def goal_center_z(self) -> float:
        return self.stone_platform_top_z

    # --- Geometry helpers ---
    def _add_box(
        self,
        geometries: list[TerrainGeometry],
        body: mujoco.MjSpec.body,
        *,
        size: tuple[float, float, float],
        pos: tuple[float, float, float],
        color: tuple[float, float, float, float],
        quat: tuple[float, float, float, float] | None = None,
    ) -> None:
        geom = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=tuple(float(np.maximum(1e-4, s)) for s in size),
            pos=tuple(float(p) for p in pos),
        )
        if quat is not None:
            geom.quat[:] = quat
        geom.contype = 1
        geom.conaffinity = 1
        for idx, value in enumerate(self.friction):
            geom.friction[idx] = value
        geometries.append(TerrainGeometry(geom=geom, color=color))

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del difficulty, rng
        body = spec.body("terrain")
        geometries: list[TerrainGeometry] = []

        cy = self.size[1] / 2.0
        path_half = float(self.path_width) / 2.0
        platform_half = float(self.platform_size) / 2.0

        total_end_x = self.stone_east_end_x
        if total_end_x > self.size[0] + 1e-6:
            raise ValueError(
                f"HalfCompetitionMap needs patch x >= {total_end_x:.3f}, "
                f"got {self.size[0]:.3f}"
            )

        # 1. Starting flat at z=0.
        self._add_box(
            geometries,
            body,
            size=(float(self.flat_length) / 2.0, path_half, self.base_thickness / 2.0),
            pos=(float(self.flat_length) / 2.0, cy, -self.base_thickness / 2.0),
            color=self.flat_color,
        )

        # 2. Lower ground from the 0.2 m step down to the east end.
        lower_length = total_end_x - float(self.flat_length)
        self._add_box(
            geometries,
            body,
            size=(lower_length / 2.0, path_half, self.base_thickness / 2.0),
            pos=(
                float(self.flat_length) + lower_length / 2.0,
                cy,
                self.lower_z - self.base_thickness / 2.0,
            ),
            color=self.lower_color,
        )

        # 3. Three up-steps to the platform level.
        for i in range(int(self.num_steps)):
            top_z = self.lower_z + (i + 1) * float(self.step_height)
            x_start = self.step_start_x + i * float(self.step_depth)
            height = top_z - self.lower_z
            self._add_box(
                geometries,
                body,
                size=(float(self.step_depth) / 2.0, platform_half, height / 2.0),
                pos=(
                    x_start + float(self.step_depth) / 2.0,
                    cy,
                    self.lower_z + height / 2.0,
                ),
                color=self.step_color,
            )

        # 4. Two 0.8 x 0.8 platforms with a 0.36 m gap.
        platform_height = self.platform_top_z - self.lower_z
        for x_start in (self.platform1_start_x, self.platform2_start_x):
            self._add_box(
                geometries,
                body,
                size=(
                    platform_half,
                    platform_half,
                    platform_height / 2.0,
                ),
                pos=(
                    x_start + platform_half,
                    cy,
                    self.lower_z + platform_height / 2.0,
                ),
                color=self.platform_color,
            )

        # 5. 20-degree ramp down from the second platform.
        angle = math.radians(float(self.ramp_angle_deg))
        drop = self.platform_top_z - self.lower_z
        ramp_run = drop / math.tan(angle)
        ramp_slope = drop / math.sin(angle)
        top_mid_x = self.platform2_end_x + ramp_run / 2.0
        top_mid_z = (self.platform_top_z + self.lower_z) / 2.0
        normal = np.array([math.sin(angle), 0.0, math.cos(angle)])
        center = (
            np.array([top_mid_x, cy, top_mid_z])
            - normal * float(self.ramp_thickness) / 2.0
        )
        self._add_box(
            geometries,
            body,
            size=(
                ramp_slope / 2.0,
                platform_half,
                float(self.ramp_thickness) / 2.0,
            ),
            pos=tuple(float(v) for v in center),
            color=self.ramp_color,
            quat=(
                math.cos(angle / 2.0),
                0.0,
                math.sin(angle / 2.0),
                0.0,
            ),
        )

        # 6. Stone-2-rows: west platform, 6 staggered pillars, east platform.
        stone_platform_half = float(self.stone_platform_size) / 2.0
        for x_start in (self.stone_west_start_x, self.stone_east_start_x):
            self._add_box(
                geometries,
                body,
                size=(
                    stone_platform_half,
                    stone_platform_half,
                    float(self.stone_platform_thickness) / 2.0,
                ),
                pos=(
                    x_start + stone_platform_half,
                    cy,
                    self.lower_z + float(self.stone_platform_thickness) / 2.0,
                ),
                color=self.stone_platform_color,
            )

        pillar_half = float(self.stone_pillar_size) / 2.0
        pillar_height = float(self.stone_pillar_height)
        row_offset = (
            float(self.stone_pillar_size) + float(self.stone_lateral_gap)
        ) / 2.0
        for row in range(int(self.stone_rows)):
            row_sign = 1.0 if row == 0 else -1.0
            stagger = 0.0 if row == 0 else self.stone_pillar_stagger
            for idx in range(int(self.stone_pillars_per_row)):
                x_center = (
                    self.stone_pillar_start_x
                    + pillar_half
                    + idx * self.stone_pillar_pitch
                    + stagger
                )
                self._add_box(
                    geometries,
                    body,
                    size=(pillar_half, pillar_half, pillar_height / 2.0),
                    pos=(
                        x_center,
                        cy + row_sign * row_offset,
                        self.lower_z + pillar_height / 2.0,
                    ),
                    color=(
                        self.stone_left_color
                        if row == 0
                        else self.stone_right_color
                    ),
                )

        origin = np.array(
            [self.start_center_x, cy, 0.0],
            dtype=np.float64,
        )
        flat_patches = {
            "start": np.array(
                [[self.start_center_x, cy, 0.0]], dtype=np.float64
            ),
            "goal": np.array(
                [[self.goal_center_x, cy, self.goal_center_z]],
                dtype=np.float64,
            ),
        }
        return TerrainOutput(
            origin=origin,
            geometries=geometries,
            flat_patches=flat_patches,
        )


def half_competition_map_terrain(
    **overrides: Any,
) -> HalfCompetitionMapTerrainCfg:
    return HalfCompetitionMapTerrainCfg(**overrides)


HALF_COMPETITION_MAP_TERRAINS_CFG = TerrainGeneratorCfg(
    # The fixed layout is ~7.96 m long and 1.0 m wide; 8.2 x 2.0 leaves
    # a small margin for the patch and the narrow path.
    size=(8.2, 2.0),
    border_width=20.0,
    num_rows=1,
    num_cols=1,
    curriculum=True,
    sub_terrains={
        "half_competition_map": half_competition_map_terrain(
            proportion=1.0,
            flat_patch_sampling={
                "start": FlatPatchSamplingCfg(num_patches=1, patch_radius=0.5),
                "goal": FlatPatchSamplingCfg(num_patches=1, patch_radius=0.3),
            },
        ),
    },
    add_lights=True,
)
