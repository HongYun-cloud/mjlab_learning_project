"""Observation configuration for the student tracking task.

This mirrors the teacher's split-config style.  The student ``actor`` group is
the deployable observation set; the ``critic`` group is privileged (used by
the teacher model during distillation); ``rnd_state`` is kept for PPO/RND
compatibility.
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from myrobot_mjlab.config.observation.Trackingobservation import (
    base_height,
    direction_real,
    direction_to_goal,
    friction_coefficient,
    heading_to_goal_error,
    height_error,
    target_distance,
    terrain_type,
    time_left,
)


def make_actor_terms(height_scan_max_distance: float) -> dict[str, ObservationTermCfg]:
    """Student deployable observation terms (before terrain-type masking)."""
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
        # Terrain one-hot: [flat, stair, tire, platform, beam, plum, stones, ...].
        # None means infer from the scene; each task can override it.
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
            scale=1.0 / height_scan_max_distance,
        ),
    }


def make_critic_terms(
    actor_terms: dict[str, ObservationTermCfg],
) -> dict[str, ObservationTermCfg]:
    """Privileged observation terms; used by the teacher during distillation."""
    return {
        **actor_terms,
        # Critic sees the true (unbiased) joint positions as privileged info.
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


def make_rnd_terms(
    actor_terms: dict[str, ObservationTermCfg],
) -> dict[str, ObservationTermCfg]:
    """Clean (s, a) terms for the PPO RND extension."""

    def _rnd_term(name: str) -> ObservationTermCfg:
        return replace(
            actor_terms[name],
            noise=None,
            delay_min_lag=0,
            delay_max_lag=0,
            history_length=0,
        )

    return {
        name: _rnd_term(name)
        for name in (
            "base_ang_vel",
            "direction_goal",
            "head_error",
            "height_error",
            "projected_gravity",
            "joint_pos",
            "joint_vel",
            "actions",
        )
    }


def make_observations(
    actor_terms: dict[str, ObservationTermCfg],
    critic_terms: dict[str, ObservationTermCfg],
    rnd_terms: dict[str, ObservationTermCfg],
    include_terrain_type_in_actor: bool = True,
) -> dict[str, ObservationGroupCfg]:
    """Build the actor/critic/rnd observation groups."""
    student_terms = (
        actor_terms
        if include_terrain_type_in_actor
        else {k: v for k, v in actor_terms.items() if k != "terrain_type"}
    )

    return {
        # Deployable student observations; corruption on during training.
        "actor": ObservationGroupCfg(
            terms=student_terms,
            concatenate_terms=True,
            enable_corruption=True,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "rnd_state": ObservationGroupCfg(
            terms=rnd_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }
