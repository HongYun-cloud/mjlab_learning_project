"""Base student tracking environment configuration.

The large observation / command / reward blocks live in the split config
modules next to the task registrations:

- ``config/obs_cfg.py``
- ``config/command_cfg.py``
- ``config/rewards_cfg.py``

This file keeps the env assembly: sensors, metrics, actions, events,
terminations, curriculum, scene, viewer and simulation.
"""

import math
import os
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import terminations as velocity_terminations
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.viewer import ViewerConfig

from myrobot_mjlab.config.curriculum.stair_curriculum import (
    terrain_levels_reach_target,
)
from myrobot_mjlab.config.reward.Trackingreward import reach_target
from myrobot_mjlab.task.student.config.command_cfg import make_commands
from myrobot_mjlab.task.student.config.obs_cfg import (
    make_actor_terms,
    make_critic_terms,
    make_observations,
    make_rnd_terms,
)
from myrobot_mjlab.task.student.config.rewards_cfg import make_rewards


def make_velocity_env_cfg(
    include_terrain_type_in_actor: bool = True,
) -> ManagerBasedRlEnvCfg:
    terrain_scan = RayCastSensorCfg(
        name="terrain_scan",
        frame=ObjRef(type="body", name="base_link", entity="robot"),
        ray_alignment="yaw",
        pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
        max_distance=5.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),  # Terrain only.
        debug_vis=True,
    )

    # Observations.
    actor_terms = make_actor_terms(
        height_scan_max_distance=terrain_scan.max_distance
    )
    critic_terms = make_critic_terms(actor_terms)
    rnd_terms = make_rnd_terms(actor_terms)
    observations = make_observations(
        actor_terms,
        critic_terms,
        rnd_terms,
        include_terrain_type_in_actor=include_terrain_type_in_actor,
    )

    # Metrics (training-only, not part of any reward).
    metrics = {
        "mean_action_acc": MetricsTermCfg(func=mdp.mean_action_acc),
        "reach_target_success": MetricsTermCfg(
            func=reach_target,
            reduce="max",
            params={
                "command_name": "Goal_direction",
                "distance_threshold": 0.15,
                "foot_body_names": ("FR_foot", "FL_foot", "RR_foot", "RL_foot"),
                "height_command_name": "Base_height_target",
            },
        ),
    }

    # Actions.
    actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.5,  # Overridden per robot.
            use_default_offset=True,
        )
    }

    commands = make_commands()

    # Events.
    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (3.5, 4.0),
                    "z": (-0.01, -0.05),
                    "yaw": (-1.57, -1.57),
                },
                "velocity_range": {},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
    }

    rewards = make_rewards()

    # Terminations.
    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(70.0)},
        ),
        "reached_edge": TerminationTermCfg(
            func=velocity_terminations.terrain_edge_reached,
            params={"threshold_fraction": 3.5125},
            time_out=True,
        ),
    }

    # Curriculum.
    curriculum_mode = os.getenv(
        "MYROBOT_CURRICULUM_MODE", "curriculum"
    ).strip().lower()
    direct_random = curriculum_mode in ("random", "direct", "1", "true")

    curriculum = {
        "terrain_levels": CurriculumTermCfg(
            func=terrain_levels_reach_target,
            params={
                "command_name": "Goal_direction",
                "distance_threshold": 0.15,
                "required_reach_count": 3,
                "max_level": 6,
                "direct_random": direct_random,
            },
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(
                terrain_type="generator",
                terrain_generator=replace(ROUGH_TERRAINS_CFG),
                max_init_terrain_level=0,
            ),
            sensors=(terrain_scan,),
            num_envs=1,
            extent=2.0,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        metrics=metrics,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="",  # Set per robot.
            distance=3.0,
            elevation=-5.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=35,
            njmax=1500,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
            ),
        ),
        decimation=4,
        episode_length_s=6.0,
    )
