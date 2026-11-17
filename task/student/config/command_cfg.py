"""Command configuration for the student tracking task."""

from __future__ import annotations

from mjlab.managers.command_manager import CommandTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from myrobot_mjlab.config.command.AllTrackingCommand import (
    GoalAlignedVelocityCommandCfg,
    GoalDirectionCommandCfg,
    HeightTargetCommandCfg,
)


def make_commands() -> dict[str, CommandTermCfg]:
    return {
        "Goal_direction": GoalDirectionCommandCfg(
            entity_name="robot",
            target_distance=3.2,
            cardinal_directions=("E", "N", "W", "S"),
            resampling_time_range=(10.0, 10.0),
            resample_on_timer=False,
            debug_vis=False,
        ),
        "Base_height_target": HeightTargetCommandCfg(
            entity_name="robot",
            nominal_height=0.28,
            foot_body_names=("FR_foot", "FL_foot", "RR_foot", "RL_foot"),
            height_smooth_alpha=0.05,
            resampling_time_range=(4.0, 10.0),
            debug_vis=True,
        ),
        # Scalar target speed: direction follows Goal_direction, speed 0.8 m/s.
        "track_base_velocity": GoalAlignedVelocityCommandCfg(
            entity_name="robot",
            goal_command_name="Goal_direction",
            align_speed_range=(0.8, 0.8),
            align_heading=False,
            resampling_time_range=(10.0, 10.0),
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 0.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
            ),
            debug_vis=False,
        ),
    }
