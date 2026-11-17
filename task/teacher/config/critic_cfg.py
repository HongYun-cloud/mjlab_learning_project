"""Critic observation configuration for the teacher task."""

from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from myrobot_mjlab.config.observation.Trackingobservation import (
  base_height,
  direction_real,
  friction_coefficient,
)


def make_critic_terms(
    actor_terms: dict[str, ObservationTermCfg],
) -> dict[str, ObservationTermCfg]:
  """Add privileged observations on top of the actor terms."""
  return {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/frame_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "base_height": ObservationTermCfg(func=base_height),
    "direction_real": ObservationTermCfg(func=direction_real),
    "Friction_coefficient": ObservationTermCfg(func=friction_coefficient),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "foot_air_time": ObservationTermCfg(
      func=mdp.foot_air_time,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces,
      params={"sensor_name": "feet_ground_contact"},
    ),
  }


def make_critic_group(
    terms: dict[str, ObservationTermCfg],
) -> ObservationGroupCfg:
  return ObservationGroupCfg(
    terms=terms,
    concatenate_terms=True,
    enable_corruption=False,
  )
