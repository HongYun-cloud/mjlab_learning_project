"""Multi-terrain student env configs for distillation training.

These environments mix several terrain columns in one terrain generator.
The student actor has no ``terrain_type`` observation; the teacher critic
keeps the privileged one-hot so a generalist teacher checkpoint can be loaded.
Keep each function in sync with the matching teacher-side function in
``task/teacher/config/env_cfg.py``.
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.terrains import TerrainEntityCfg

from myrobot_mjlab.config.command.AllTrackingCommand import (
    GoalDirectionCommandCfg,
    HeightTargetCommandCfg,
)
from myrobot_mjlab.config.event.stair_reset import (
    enforce_terrain_min_level,
    out_of_patch_bounds,
    reset_root_state_from_stair,
    root_below_origin_height,
)
from myrobot_mjlab.config.event.stair_waypoint import (
    reset_root_state_at_stair_top,
    reset_stair_down_height_target,
)
from myrobot_mjlab.config.terrain.half_competition_map_terrain import (
    HALF_COMPETITION_MAP_TERRAINS_CFG,
)
from myrobot_mjlab.config.terrain.mixed_terrain import (
    FLAT_PLATFORM_STAIR_DOWN_PLUM_TERRAINS_CFG,
    FLAT_PLATFORM_STAIR_DOWN_TERRAINS_CFG,
    FLAT_STAIR_DOWN_TERRAINS_CFG,
    FLAT_STAIR_UP_TERRAINS_CFG,
)

from .env_cfg import (
    _make_stair_env_cfg,
    _out_of_patch_bounds_by_terrain,
    _reset_mixed_goal_targets,
    _reset_mixed_height_target,
)


def my_go2_flat_stair_up_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Mixed flat + stair-up environment for two-teacher distillation.

  - Terrain columns: 0 = flat, 1 = stair_up (see mixed_terrain.py).
  - ``terrain_types`` is the per-env column index; the teacher group uses it
    to route each env to the matching teacher.
  - The student actor does NOT receive ``terrain_type``; the teacher critic
    keeps the per-env one-hot as privileged information.
  """
  cfg = _make_stair_env_cfg(
    play=play,
    include_terrain_type_in_actor=False,
  )

  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=replace(FLAT_STAIR_UP_TERRAINS_CFG),
    max_init_terrain_level=0,
  )

  # Both terrain patches are 12 x 12 m; leave a small safety margin.
  cfg.terminations["out_of_patch_bounds"].params = {
    "half_size": 5.8,
    "margin": 0.2,
  }

  # Unified reset:
  # - flat envs: _stair_total_height returns 0 -> spawn at patch center;
  # - stair_up envs: spawn at the pyramid top.
  cfg.events["reset_base"] = EventTermCfg(
    func=reset_root_state_at_stair_top,
    mode="reset",
    params={
      "target_distance": 3.8,
      "nominal_height": 0.28,
    },
  )

  cmd_goal = cfg.commands["Goal_direction"]
  assert isinstance(cmd_goal, GoalDirectionCommandCfg)
  cmd_goal.cardinal_directions = ("E", "N", "W", "S")
  cmd_goal.target_distance = 3.8

  if play:
    # One env per terrain column so the Viser terrain selector can show both.
    cfg.scene.num_envs = 2
    cfg.scene.terrain.max_init_terrain_level = 0

  return cfg


def my_go2_flat_stair_down_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Mixed flat + stair-down environment for two-teacher distillation.

  - Terrain columns: 0 = flat, 1 = stair_down (8 x 8 m patches).
  - Both terrain types start at the patch center: flat at ground level,
    stair_down at the low center of the inverted pyramid.
  - ``reset_stair_down_height_target`` uses the per-env ``_stair_total_height``:
    flat -> 0, stair_down -> pyramid height.
  - ``stair_down`` never starts below level 2, so its geometry and height
    command stay distinguishable from flat even without ``terrain_type`` in
    the student actor.
  - Student actor has no ``terrain_type``; teacher critic keeps the one-hot.
  """
  cfg = _make_stair_env_cfg(
    play=play,
    include_terrain_type_in_actor=False,
  )

  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=replace(FLAT_STAIR_DOWN_TERRAINS_CFG),
    max_init_terrain_level=2,
  )

  # Reset order matters: this clamp must run before reset_base and
  # reset_height_target, which read terrain_levels / env_origins.
  stair_generator = cfg.scene.terrain.terrain_generator
  assert stair_generator is not None
  stair_down_columns = tuple(
    index
    for index, name in enumerate(stair_generator.sub_terrains)
    if name == "stair_down"
  )
  # Clamp inside the curriculum too, so the logged terrain levels reflect the
  # post-clamp values (the reset event alone runs after curriculum logging).
  # Play mode has no curriculum; the reset event still enforces the clamp.
  if "terrain_levels" in cfg.curriculum:
    cfg.curriculum["terrain_levels"].params["min_level"] = 2
    cfg.curriculum["terrain_levels"].params["min_level_columns"] = (
      stair_down_columns
    )
  cfg.events = {
    "enforce_stair_down_min_level": EventTermCfg(
      func=enforce_terrain_min_level,
      mode="reset",
      params={"terrain_columns": stair_down_columns, "min_level": 2},
    ),
    **cfg.events,
  }

  cfg.terminations["out_of_patch_bounds"].params = {
    "half_size": 4.5,
    "margin": 0.2,
  }

  # Same spawn for flat and stair_down: patch center + small jitter.
  cfg.events["reset_base"] = EventTermCfg(
    func=reset_root_state_from_stair,
    mode="reset",
    params={
      "pose_range": {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (0.0, 0.0),
        "yaw": (-3.14, 3.14),
      },
      "velocity_range": {},
      "nominal_height": 0.28,
    },
  )
  # Height command: flat -> 0, stair_down -> top height.
  cfg.events["reset_height_target"] = EventTermCfg(
    func=reset_stair_down_height_target,
    mode="reset",
    params={"height_command_name": "Base_height_target"},
  )

  cmd_goal = cfg.commands["Goal_direction"]
  assert isinstance(cmd_goal, GoalDirectionCommandCfg)
  cmd_goal.cardinal_directions = ("E", "N", "W", "S")
  cmd_goal.target_distance = 3.8

  if play:
    cfg.scene.num_envs = 10

  return cfg


def my_go2_flat_platform_stair_down_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Unlabeled mixed student task: flat + step-platform + stair-down.

  The student actor has no ``terrain_type`` (235 dims); the critic keeps the
  privileged one-hot (275 dims) so the generalist teacher checkpoint can be
  loaded.  Terrain/reset/targets/height/bounds match the teacher-side
  ``FlatPlatformStairDown`` task exactly; keep the two env functions in sync.
  """
  cfg = _make_stair_env_cfg(
    play=play,
    include_terrain_type_in_actor=False,
  )

  # Platform-style reward set: no stair-only speed penalty.
  cfg.rewards["aggressive_motion"].weight = -4.0
  cfg.rewards.pop("stair_speed_tracking", None)

  cmd_height = cfg.commands["Base_height_target"]
  assert isinstance(cmd_height, HeightTargetCommandCfg)
  cmd_height.ground_z_min = None

  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=replace(FLAT_PLATFORM_STAIR_DOWN_TERRAINS_CFG),
    max_init_terrain_level=2,
  )

  terrain_generator = cfg.scene.terrain.terrain_generator
  assert terrain_generator is not None
  sub_terrain_names = list(terrain_generator.sub_terrains)
  stair_down_columns = tuple(
    index
    for index, name in enumerate(sub_terrain_names)
    if name == "stair_down"
  )
  platform_columns = tuple(
    index for index, name in enumerate(sub_terrain_names) if name == "platform"
  )

  if "terrain_levels" in cfg.curriculum:
    params = cfg.curriculum["terrain_levels"].params
    params["max_level"] = 12
    params["randomize_at_max"] = True
    params["min_level"] = 2
    params["min_level_columns"] = stair_down_columns

  cfg.events = {
    "enforce_stair_down_min_level": EventTermCfg(
      func=enforce_terrain_min_level,
      mode="reset",
      params={"terrain_columns": stair_down_columns, "min_level": 2},
    ),
    **cfg.events,
  }

  cfg.events["reset_base"] = EventTermCfg(
    func=reset_root_state_from_stair,
    mode="reset",
    params={
      "pose_range": {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (0.0, 0.0),
        "yaw": (-3.14, 3.14),
      },
      "velocity_range": {},
      "nominal_height": 0.28,
    },
  )
  cfg.events["reset_goal_targets"] = EventTermCfg(
    func=_reset_mixed_goal_targets,
    mode="reset",
    params={
      "command_name": "Goal_direction",
      "platform_columns": platform_columns,
      "flat_stair_distance": 3.8,
    },
  )
  cfg.events["reset_height_target"] = EventTermCfg(
    func=_reset_mixed_height_target,
    mode="reset",
    params={
      "height_command_name": "Base_height_target",
      "platform_columns": platform_columns,
    },
  )

  cfg.terminations["out_of_patch_bounds"] = TerminationTermCfg(
    func=_out_of_patch_bounds_by_terrain,
    time_out=True,
    params={
      "half_size": 4.5,
      "platform_half_size_x": 8.0,
      "platform_half_size_y": 4.0,
      "platform_columns": platform_columns,
      "margin": 0.2,
    },
  )

  cmd_goal = cfg.commands["Goal_direction"]
  assert isinstance(cmd_goal, GoalDirectionCommandCfg)
  cmd_goal.cardinal_directions = ("E", "N", "W", "S")
  cmd_goal.target_distance = 3.8

  if play:
    cfg.scene.num_envs = 3

  return cfg


def my_go2_flat_platform_stair_down_plum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Unlabeled mixed student: flat + stair-down + platform + plum.

  Terrain columns and allocation match the teacher-side
  ``FlatPlatformStairDownPlum`` task:
    flat : stair_down : platform : plum = 2 : 2 : 2 : 4

  The student actor has no ``terrain_type``; the critic keeps the privileged
  one-hot (13 dims) so the generalist teacher checkpoint can be loaded.
  """
  cfg = _make_stair_env_cfg(
    play=play,
    include_terrain_type_in_actor=False,
  )

  # Platform-style reward set: no stair-only speed penalty.
  cfg.rewards["aggressive_motion"].weight = -4.0
  cfg.rewards.pop("stair_speed_tracking", None)

  # stair_down needs negative ground z (pit bottom); platform/plum need
  # zero or positive ground z.
  cmd_height = cfg.commands["Base_height_target"]
  assert isinstance(cmd_height, HeightTargetCommandCfg)
  cmd_height.ground_z_min = None

  # Plum has many small pillars and contacts.
  cfg.sim.njmax = 300
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = 128

  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=replace(FLAT_PLATFORM_STAIR_DOWN_PLUM_TERRAINS_CFG),
    max_init_terrain_level=2,
  )

  terrain_generator = cfg.scene.terrain.terrain_generator
  assert terrain_generator is not None
  sub_terrain_names = list(terrain_generator.sub_terrains)
  stair_down_columns = tuple(
    index
    for index, name in enumerate(sub_terrain_names)
    if name == "stair_down"
  )
  platform_columns = tuple(
    index for index, name in enumerate(sub_terrain_names) if name == "platform"
  )

  if "terrain_levels" in cfg.curriculum:
    params = cfg.curriculum["terrain_levels"].params
    params["max_level"] = 12
    params["randomize_at_max"] = True
    params["min_level"] = 2
    params["min_level_columns"] = stair_down_columns

  cfg.events = {
    "enforce_stair_down_min_level": EventTermCfg(
      func=enforce_terrain_min_level,
      mode="reset",
      params={"terrain_columns": stair_down_columns, "min_level": 2},
    ),
    **cfg.events,
  }

  # Small jitter: the plum center platform shrinks to 0.8 m at level 12.
  cfg.events["reset_base"] = EventTermCfg(
    func=reset_root_state_from_stair,
    mode="reset",
    params={
      "pose_range": {
        "x": (-0.1, 0.1),
        "y": (-0.1, 0.1),
        "z": (0.0, 0.0),
        "yaw": (-3.14, 3.14),
      },
      "velocity_range": {},
      "nominal_height": 0.28,
    },
  )
  cfg.events["reset_goal_targets"] = EventTermCfg(
    func=_reset_mixed_goal_targets,
    mode="reset",
    params={
      "command_name": "Goal_direction",
      "platform_columns": platform_columns,
      "flat_stair_distance": 3.8,
    },
  )
  cfg.events["reset_height_target"] = EventTermCfg(
    func=_reset_mixed_height_target,
    mode="reset",
    params={
      "height_command_name": "Base_height_target",
      "platform_columns": platform_columns,
    },
  )

  cfg.terminations["out_of_patch_bounds"] = TerminationTermCfg(
    func=_out_of_patch_bounds_by_terrain,
    time_out=True,
    params={
      "half_size": 4.5,
      "platform_half_size_x": 8.0,
      "platform_half_size_y": 4.0,
      "platform_columns": platform_columns,
      "margin": 0.2,
    },
  )

  cmd_goal = cfg.commands["Goal_direction"]
  assert isinstance(cmd_goal, GoalDirectionCommandCfg)
  cmd_goal.cardinal_directions = ("E", "N", "W", "S")
  cmd_goal.target_distance = 3.8

  if play:
    # One env per terrain column.
    cfg.scene.num_envs = 4

  return cfg


def my_go2_half_competition_map_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Half-competition map student env.

  Same composite patch as the teacher task, but the student actor has no
  ``terrain_type``; the critic keeps the privileged one-hot.
  """
  cfg = _make_stair_env_cfg(
    play=play,
    include_terrain_type_in_actor=False,
  )

  cfg.rewards["aggressive_motion"].weight = -4.0
  cfg.rewards.pop("stair_speed_tracking", None)

  cmd_height = cfg.commands["Base_height_target"]
  assert isinstance(cmd_height, HeightTargetCommandCfg)
  cmd_height.ground_z_min = None

  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=replace(HALF_COMPETITION_MAP_TERRAINS_CFG),
    max_init_terrain_level=0,
  )

  terrain_generator = cfg.scene.terrain.terrain_generator
  assert terrain_generator is not None
  map_cfg = next(iter(terrain_generator.sub_terrains.values()))
  goal_distance = float(map_cfg.goal_center_x - map_cfg.start_center_x)

  cfg.episode_length_s = 20.0

  cfg.terminations["out_of_patch_bounds"] = TerminationTermCfg(
    func=out_of_patch_bounds,
    time_out=True,
    params={
      "half_size_x": 7.8,
      "half_size_y": 1.2,
      "margin": 0.2,
    },
  )
  cfg.terminations["base_below_map"] = TerminationTermCfg(
    func=root_below_origin_height,
    params={"min_height": -0.5},
  )

  cfg.events["reset_base"] = EventTermCfg(
    func=reset_root_state_from_stair,
    mode="reset",
    params={
      "pose_range": {
        "x": (-0.1, 0.1),
        "y": (-0.1, 0.1),
        "z": (0.0, 0.0),
        "yaw": (-0.2, 0.2),
      },
      "velocity_range": {},
      "nominal_height": 0.28,
    },
  )

  cmd_goal = cfg.commands["Goal_direction"]
  assert isinstance(cmd_goal, GoalDirectionCommandCfg)
  cmd_goal.cardinal_directions = ("E",)
  cmd_goal.target_distance = goal_distance

  if play:
    cfg.scene.num_envs = 1

  return cfg


