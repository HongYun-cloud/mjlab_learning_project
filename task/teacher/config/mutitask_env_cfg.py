"""Multi-terrain teacher env configs for generalist privileged policies.

These environments mix several terrain columns in one terrain generator.
Unlike the student-side functions in
``task/student/config/mutitask_env_cfg.py``, the teacher actor consumes the
full privileged critic observation, including the per-env ``terrain_type``
one-hot inferred from ``terrain_types``.

Keep each function in sync with the matching student-side function.
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
    reset_stair_down_height_target,
)
from myrobot_mjlab.config.terrain.half_competition_map_terrain import (
    HALF_COMPETITION_MAP_TERRAINS_CFG,
)
from myrobot_mjlab.config.terrain.mixed_terrain import (
    FLAT_PLATFORM_STAIR_DOWN_PLUM_TERRAINS_CFG,
    FLAT_PLATFORM_STAIR_DOWN_TERRAINS_CFG,
    FLAT_STAIR_DOWN_STONES_2ROWS_TERRAINS_CFG,
    FLAT_STAIR_DOWN_TERRAINS_CFG,
)

from .env_cfg import (
    _make_stair_env_cfg,
    _out_of_patch_bounds_by_terrain,
    _reset_mixed_goal_targets,
    _reset_mixed_height_target,
)


def my_go2_flat_stair_down_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Mixed flat + stair-down teacher for a generalist privileged policy.

  This is the teacher-side counterpart of the student ``FlatStairDown`` task:
  the MDP (terrain mix, reset, commands, rewards, terminations, episode
  length, curriculum) is the same, but the teacher actor consumes the full
  privileged critic observation, including the per-env ``terrain_type``
  one-hot.  The one-hot is inferred from ``terrain_types`` automatically, so
  do not call ``_set_terrain_type`` here.

  ``stair_down`` starts at level 2 for the same reason as the student task:
  level 0/1 stairs are geometrically flat and cannot be distinguished from the
  flat column without privileged terrain information.
  """
  cfg = _make_stair_env_cfg(play=play)

  # Match the student mixed task exactly: the student env has no hard base
  # speed clamp, only the velocity-tracking / aggressive-motion rewards.
  cfg.events.pop("clamp_base_speed", None)

  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=replace(FLAT_STAIR_DOWN_TERRAINS_CFG),
    max_init_terrain_level=2,
  )

  cfg.terminations["out_of_patch_bounds"].params = {
    "half_size": 4.5,
    "margin": 0.2,
  }

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

  # Same spawn / height target as the student mixed task.
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
    # At least two envs so both terrain columns are instantiated.
    cfg.scene.num_envs = 10

  return cfg


def my_go2_flat_platform_stair_down_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Mixed flat + step-platform + stair-down generalist teacher.

  One privileged policy conditioned on the per-env ``terrain_type`` one-hot:

  - flat: fixed 3.8 m target in a random cardinal direction;
  - stair_down: fixed 3.8 m target, height target = pyramid top height;
  - platform: target on the second/end platform, distance from the platform
    difficulty schedule, height target follows the goal flat patch.

  The platform sub-terrain uses 13 curriculum rows (0..12); ``stair_down``
  starts at level 2 because levels 0/1 are geometrically flat.
  """
  cfg = _make_stair_env_cfg(play=play)

  # Match the student-side mixed task: no hard base-speed clamp.
  cfg.events.pop("clamp_base_speed", None)

  # Platform-style reward tweak.
  cfg.rewards["aggressive_motion"].weight = -4.0
  # The platform task needs precise stepping; the stair-only forward-speed
  # penalty is not part of its reward set.
  cfg.rewards.pop("stair_speed_tracking", None)

  # stair_down needs negative ground z (pit bottom); platform needs positive
  # ground z (0.36 m platform top).
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
    # Platform schedule spans 0..12.
    params["max_level"] = 12
    params["randomize_at_max"] = True
    params["min_level"] = 2
    params["min_level_columns"] = stair_down_columns

  # Reset order: clamp levels/origins -> spawn -> target positions ->
  # height target.  reset_goal_targets must run before reset_height_target
  # so the platform height command can find the goal flat patch.
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
  # Targets are written per column in reset_goal_targets; the cardinal table
  # is only a fallback and is unused once target_positions exists.
  cmd_goal.cardinal_directions = ("E", "N", "W", "S")
  cmd_goal.target_distance = 3.8

  if play:
    # One env per terrain column.
    cfg.scene.num_envs = 3

  return cfg


def my_go2_flat_platform_stair_down_plum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Mixed flat + stair-down + platform + plum generalist teacher.

  Terrain allocation (curriculum columns):
    flat : stair_down : platform : plum = 2 : 2 : 2 : 4

  Reset/target behavior:
  - flat / stair_down / plum: random cardinal direction * 3.8 m;
  - platform: second/end platform, distance from its difficulty schedule;
  - height target: flat/plum ground 0, stair_down pyramid top, platform
    follows the goal flat patch.
  """
  cfg = _make_stair_env_cfg(play=play)

  # Match the platform-style mixed teacher: no hard base-speed clamp.
  cfg.events.pop("clamp_base_speed", None)

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
  """Half-competition map: flat -> step down -> stairs/platform -> stones.

  A single composite patch ordered west to east:
    flat (1 m) -> 0.2 m step down -> 3 x 0.12 m stairs -> 0.8 m platform
    -> 0.36 m gap -> 0.8 m platform -> 20 deg ramp down
    -> stone-2-rows finish.

  The teacher actor keeps the privileged ``terrain_type`` one-hot; the
  student-side counterpart omits it from the actor.
  """
  cfg = _make_stair_env_cfg(play=play)

  # Platform-style reward set: no stair-only speed penalty.
  cfg.rewards["aggressive_motion"].weight = -4.0
  cfg.rewards.pop("stair_speed_tracking", None)

  # Goal is on the stone platform at z=-0.1 (relative to the start origin).
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

  # Spawn on the 1 x 1 m start flat.
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


def my_go2_flat_stair_down_stones_2rows_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Mixed flat + stair-down + compact stones-2-rows teacher.

  Terrain columns and allocation:
    flat : stair_down : stones_2rows = 1 : 1 : 1

  The stones-2-rows column uses an 8 x 8 m compact version: 0.8 m
  center/outer platforms, 0.2 m pillars and 0.1 m gaps, so its target
  distance stays <= 2.7 m.
  """
  cfg = _make_stair_env_cfg(play=play)

  # Match the existing mixed teacher tasks: no hard base-speed clamp.
  cfg.events.pop("clamp_base_speed", None)

  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=replace(FLAT_STAIR_DOWN_STONES_2ROWS_TERRAINS_CFG),
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
  stone_columns = tuple(
    index
    for index, name in enumerate(sub_terrain_names)
    if name == "stones_2rows"
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

  # Small jitter: the stones center platform is only 0.8 m.
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
      "stone_columns": stone_columns,
      "flat_stair_distance": 3.8,
    },
  )
  cfg.events["reset_height_target"] = EventTermCfg(
    func=_reset_mixed_height_target,
    mode="reset",
    params={"height_command_name": "Base_height_target"},
  )

  cfg.terminations["out_of_patch_bounds"] = TerminationTermCfg(
    func=out_of_patch_bounds,
    time_out=True,
    params={"half_size": 4.5, "margin": 0.2},
  )
  cfg.terminations["base_below_map"] = TerminationTermCfg(
    func=root_below_origin_height,
    params={"min_height": -0.5},
  )

  cmd_goal = cfg.commands["Goal_direction"]
  assert isinstance(cmd_goal, GoalDirectionCommandCfg)
  cmd_goal.cardinal_directions = ("E", "N", "W", "S")
  cmd_goal.target_distance = 3.8

  if play:
    cfg.scene.num_envs = 3

  return cfg


