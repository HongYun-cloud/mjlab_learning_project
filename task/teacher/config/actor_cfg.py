"""Actor observation configuration for the teacher task."""

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from myrobot_mjlab.config.observation.Trackingobservation import (
  direction_to_goal,
  heading_to_goal_error,
  height_error,
  target_distance,
  terrain_type,
  time_left,
)


def make_actor_terms(
    height_scan_max_distance: float = 5.0,
) -> dict[str, ObservationTermCfg]:
  """Base actor observation terms."""
  return {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_gyro"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "direction_goal": ObservationTermCfg(
      func=direction_to_goal,
      params={"command_name": "Goal_direction", "in_body_frame": True},
    ),
    "head_error": ObservationTermCfg(
      func=heading_to_goal_error,
      params={"command_name": "Goal_direction"},
    ),
    "target_distance": ObservationTermCfg(
      func=target_distance,
      params={"command_name": "Goal_direction"},
    ),
    "time_left": ObservationTermCfg(
      func=time_left,
      params={"normalize": True},
    ),
    # 地形类型 one-hot；传 None 时按场景地形自动推断。
    "terrain_type": ObservationTermCfg(
      func=terrain_type,
      params={"terrain_type": None},
    ),
    "height_error": ObservationTermCfg(
      func=height_error,
      params={"command_name": "Base_height_target"},
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"biased": True},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      noise=Unoise(n_min=-0.1, n_max=0.1),
      scale=1.0 / max(height_scan_max_distance, 1e-6),
    ),
  }


def make_actor_group(
    terms: dict[str, ObservationTermCfg],
) -> ObservationGroupCfg:
  """Teacher actor uses clean, deterministic observations."""
  return ObservationGroupCfg(
    terms=terms,
    concatenate_terms=True,
    enable_corruption=False,
  )
