"""Symmetry data augmentation for the Track tasks.

The paper uses left-right / front-back mirror symmetry to enlarge each
``(observation, action)`` transition into up to four pairs.  RSL-RL already
implements the PPO-side bookkeeping (repeating advantages / returns and adding
the optional mirror loss), so this module only has to answer one question:

    given the observation groups and the action vector of a Track environment,
    what is their left-right (and optionally front-back) mirror image?

The functions below are referenced from ``rl_cfg.py`` by a string so that the
runner config stays YAML-serialisable::

    "myrobot_mjlab.task.track.config.symmetry:mirror_obs_actions"

Only observations of the current Track tasks are supported.  If a new
observation term is added, add its mirror rule to ``_mirror_term`` as well.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

try:
    from tensordict import TensorDict
except ImportError:  # pragma: no cover - rsl-rl always depends on tensordict
    TensorDict = None  # type: ignore[assignment,misc]


MirrorMode = Literal["lr", "fb"]

# Sign vectors for the non-joint observation terms.
# lr: reflection across the x-z plane (left <-> right)
# fb: reflection across the y-z plane (front <-> back)
_VEC3_ANG_VEL: dict[str, tuple[float, float, float]] = {
    "lr": (-1.0, 1.0, -1.0),
    "fb": (1.0, -1.0, -1.0),
}
_VEC3_PROJ_GRAV: dict[str, tuple[float, float, float]] = {
    "lr": (1.0, -1.0, 1.0),
    "fb": (-1.0, 1.0, 1.0),
}
_VEC3_LIN_VEL: dict[str, tuple[float, float, float]] = {
    "lr": (1.0, -1.0, 1.0),
    "fb": (-1.0, 1.0, 1.0),
}
_VEC2_DIRECTION: dict[str, tuple[float, float]] = {
    "lr": (1.0, -1.0),
    "fb": (-1.0, 1.0),
}
_VEC3_CONTACT_FORCE: dict[str, tuple[float, float, float]] = {
    "lr": (1.0, -1.0, 1.0),
    "fb": (-1.0, 1.0, 1.0),
}

# Terms that are invariant under both reflections.
_INVARIANT_TERMS = {
    "terrain_type",
    "height_command",
    "base_height",
    "Friction_coefficient",
}

_FOOT_TERMS = {
    "foot_air_time",
    "foot_contact",
    "foot_contact_forces",
}


def _joint_swap_map(
    names: list[str] | tuple[str, ...], mode: MirrorMode
) -> tuple[list[int], list[float]]:
    """Return ``(source_index, sign)`` for mirroring a joint-space vector.

    ``out[i] = vec[source_index[i]] * sign[i]``

    - left-right: FL <-> FR, RL <-> RR; hip abduction is negated.
    - front-back: FL <-> RL, FR <-> RR; thigh and calf are negated.
      (All thigh/calf joint axes are +Y and all hip axes are +X in go2.xml.)
    """
    names = list(names)
    index = {name: i for i, name in enumerate(names)}

    if mode == "lr":
        swap = {"FL": "FR", "FR": "FL", "RL": "RR", "RR": "RL"}
        negate_parts = ("hip",)
    elif mode == "fb":
        swap = {"FL": "RL", "RL": "FL", "FR": "RR", "RR": "FR"}
        negate_parts = ("thigh", "calf")
    else:
        raise ValueError(f"Unsupported mirror mode: {mode}")

    source: list[int] = []
    sign: list[float] = []
    for name in names:
        side, part = name.split("_", 1)
        if side not in swap:
            raise ValueError(
                f"Cannot mirror joint '{name}'. Expected a name like "
                "'FL_hip_joint', 'FR_thigh_joint', ..."
            )
        mirrored_name = f"{swap[side]}_{part}"
        if mirrored_name not in index:
            raise ValueError(
                f"Cannot mirror joint '{name}': '{mirrored_name}' not found in "
                f"{names}"
            )
        source.append(index[mirrored_name])
        sign.append(-1.0 if part.startswith(negate_parts) else 1.0)
    return source, sign


def _foot_swap_map(
    names: list[str] | tuple[str, ...], mode: MirrorMode
) -> list[int]:
    """Return the source index used to permute per-foot values."""
    names = list(names)
    index = {name: i for i, name in enumerate(names)}
    if mode == "lr":
        swap = {"FL": "FR", "FR": "FL", "RL": "RR", "RR": "RL"}
    elif mode == "fb":
        swap = {"FL": "RL", "RL": "FL", "FR": "RR", "RR": "FR"}
    else:
        raise ValueError(f"Unsupported mirror mode: {mode}")

    source: list[int] = []
    for name in names:
        if name not in swap:
            raise ValueError(f"Cannot mirror foot '{name}' with names {names}")
        source.append(index[swap[name]])
    return source


def _term_slices(env) -> tuple[dict[str, dict[str, slice]], dict[str, int]]:
    """Build ``group -> term -> slice`` for concatenated observation groups."""
    manager = env.unwrapped.observation_manager
    layout: dict[str, dict[str, slice]] = {}
    sizes: dict[str, int] = {}
    for group, term_names in manager.active_terms.items():
        dims = manager.group_obs_term_dim[group]
        term_slices: dict[str, slice] = {}
        offset = 0
        for name, dim in zip(term_names, dims, strict=False):
            if len(dim) != 1:
                raise ValueError(
                    f"Symmetry augmentation assumes 1-D observation terms, but "
                    f"'{group}/{name}' has shape {tuple(dim)}. History or "
                    "multi-dimensional observations need a dedicated mirror rule."
                )
            if manager.get_term_cfg(group, name).history_length > 0:
                raise ValueError(
                    f"Symmetry augmentation does not support history for "
                    f"'{group}/{name}'. Disable history or add a dedicated "
                    "mirror rule."
                )
            size = int(np.prod(dim))
            term_slices[name] = slice(offset, offset + size)
            offset += size
        layout[group] = term_slices
        sizes[group] = offset
    return layout, sizes


def _mirror_term(
    value: torch.Tensor,
    name: str,
    mode: MirrorMode,
    joint_source: list[int],
    joint_sign: list[float],
    foot_source: list[int],
    height_scan_shape: tuple[int, int] | None,
) -> torch.Tensor:
    """Apply one reflection to a single observation term."""
    if name == "height_scan":
        if height_scan_shape is None:
            raise ValueError(
                "Cannot mirror 'height_scan': no grid pattern with "
                "size/resolution was found on its sensor."
            )
        ny, nx = height_scan_shape
        if value.shape[1] != ny * nx:
            raise ValueError(
                f"Cannot mirror 'height_scan': expected {ny * nx} values, "
                f"got {value.shape[1]}."
            )
        grid = value.reshape(value.shape[0], ny, nx)
        # Grid order from GridPatternCfg: x is the fast axis, y is the slow axis.
        # lr -> flip lateral y; fb -> flip forward x.
        if mode == "lr":
            grid = torch.flip(grid, dims=(1,))
        elif mode == "fb":
            grid = torch.flip(grid, dims=(2,))
        else:
            raise ValueError(f"Unsupported mirror mode: {mode}")
        return grid.reshape(value.shape[0], -1)
    if name == "base_ang_vel":
        return value * value.new_tensor(_VEC3_ANG_VEL[mode])
    if name == "projected_gravity":
        return value * value.new_tensor(_VEC3_PROJ_GRAV[mode])
    if name == "base_lin_vel":
        return value * value.new_tensor(_VEC3_LIN_VEL[mode])
    if name in ("direction_goal", "direction_real"):
        return value * value.new_tensor(_VEC2_DIRECTION[mode])
    if name == "head_error":
        return -value
    if name in ("joint_pos", "joint_vel", "actions"):
        return value[:, joint_source] * value.new_tensor(joint_sign)
    if name in ("foot_air_time", "foot_contact"):
        return value[:, foot_source]
    if name == "foot_contact_forces":
        # [B, num_feet * 3] -> [B, num_feet, 3]
        forces = value.reshape(value.shape[0], len(foot_source), 3)
        forces = forces[:, foot_source] * value.new_tensor(
            _VEC3_CONTACT_FORCE[mode]
        )
        return forces.reshape(value.shape[0], -1)
    if name in _INVARIANT_TERMS:
        return value
    raise ValueError(
        f"Symmetry augmentation does not know how to mirror observation term "
        f"'{name}'. Add a rule in "
        f"myrobot_mjlab.task.track.config.symmetry._mirror_term()."
    )


def _mirror_group(
    group: str,
    value: torch.Tensor,
    slices: dict[str, slice],
    mode: MirrorMode,
    joint_source: list[int],
    joint_sign: list[float],
    foot_source: list[int],
    height_scan_shape: tuple[int, int] | None,
) -> torch.Tensor:
    out = value.clone()
    for name, sl in slices.items():
        out[:, sl] = _mirror_term(
            value[:, sl],
            name,
            mode,
            joint_source,
            joint_sign,
            foot_source,
            height_scan_shape,
        )
    return out


def _mirror_action(
    actions: torch.Tensor,
    mode: MirrorMode,
    joint_source: list[int],
    joint_sign: list[float],
) -> torch.Tensor:
    return actions[:, joint_source] * actions.new_tensor(joint_sign)


def _concat_obs(obs_list: list) -> object:
    if TensorDict is not None and isinstance(obs_list[0], TensorDict):
        batch_size = sum(int(item.batch_size[0]) for item in obs_list)
        return TensorDict(
            {
                key: torch.cat([item[key] for item in obs_list], dim=0)
                for key in obs_list[0].keys()
            },
            batch_size=[batch_size],
        )
    return {
        key: torch.cat([item[key] for item in obs_list], dim=0)
        for key in obs_list[0]
    }


def _augment(
    env,
    obs,
    actions,
    transforms: tuple[tuple[MirrorMode, ...], ...],
):
    """Return original + mirrored observations/actions.

    ``obs`` is ``None`` when RSL-RL only wants to mirror the action means for
    the optional mirror loss.  In that case the first return value is ``None``.
    """
    manager = env.unwrapped.observation_manager
    layout = _term_slices(env)[0] if obs is not None else {}

    # height_scan is a flattened 2-D grid. Find its (ny, nx) shape from the
    # sensor pattern so it can be flipped on the correct axis.
    height_scan_shape: tuple[int, int] | None = None
    for group, term_names in manager.active_terms.items():
        if "height_scan" not in term_names:
            continue
        term_cfg = manager.get_term_cfg(group, "height_scan")
        sensor_name = term_cfg.params.get("sensor_name")
        if sensor_name is None:
            break
        sensor = env.unwrapped.scene[sensor_name]
        pattern = getattr(sensor.cfg, "pattern", None)
        size = getattr(pattern, "size", None)
        resolution = getattr(pattern, "resolution", None)
        if size is not None and resolution is not None and resolution > 0:
            nx = int(round(float(size[0]) / float(resolution))) + 1
            ny = int(round(float(size[1]) / float(resolution))) + 1
            height_scan_shape = (ny, nx)
        break

    robot = env.unwrapped.scene["robot"]
    joint_names = list(robot.joint_names)

    # All Track tasks have a single joint-position action term.  Build the
    # action map from the actual action-term order instead of assuming it.
    action_names: list[str] = []
    for term_name in env.unwrapped.action_manager.active_terms:
        term = env.unwrapped.action_manager.get_term(term_name)
        action_names.extend(list(term.target_names))
    if action_names != joint_names:
        raise ValueError(
            "Symmetry augmentation currently assumes the policy action order "
            "matches the robot joint order. Got actions "
            f"{action_names} and joints {joint_names}."
        )

    # Find the foot names from one of the contact observation terms.
    foot_names: list[str] | None = None
    for group, term_names in manager.active_terms.items():
        for term_name in term_names:
            if term_name in _FOOT_TERMS:
                term_cfg = manager.get_term_cfg(group, term_name)
                sensor_name = term_cfg.params.get("sensor_name")
                if sensor_name is not None:
                    foot_names = list(env.unwrapped.scene[sensor_name].primary_names)
                    break
        if foot_names is not None:
            break
    if foot_names is None:
        foot_names = ["FL", "FR", "RL", "RR"]

    observations = [obs] if obs is not None else None
    action_list = [actions] if actions is not None else None

    # Compose LR and FB by applying them sequentially.  The order does not
    # matter because the two reflections commute (their composition is a
    # 180-degree rotation about the z axis).
    for mode_seq in transforms:
        obs_m = obs
        act_m = actions
        for mode in mode_seq:
            joint_source, joint_sign = _joint_swap_map(joint_names, mode)
            foot_source = _foot_swap_map(foot_names, mode)
            if obs_m is not None:
                obs_m = _mirror_obs_groups(
                    obs_m,
                    layout,
                    mode,
                    joint_source,
                    joint_sign,
                    foot_source,
                    height_scan_shape,
                )
            if act_m is not None:
                act_m = _mirror_action(act_m, mode, joint_source, joint_sign)
        if observations is not None:
            observations.append(obs_m)
        if action_list is not None:
            action_list.append(act_m)

    obs_aug = _concat_obs(observations) if observations is not None else None
    if action_list is None:
        return obs_aug, None
    return obs_aug, torch.cat(action_list, dim=0)


def _mirror_obs_groups(
    obs,
    layout: dict[str, dict[str, slice]],
    mode: MirrorMode,
    joint_source: list[int],
    joint_sign: list[float],
    foot_source: list[int],
    height_scan_shape: tuple[int, int] | None,
):
    result = {}
    for group in obs.keys():
        if group not in layout:
            # Unknown auxiliary groups are copied through unchanged.
            result[group] = obs[group]
            continue
        result[group] = _mirror_group(
            group,
            obs[group],
            layout[group],
            mode,
            joint_source,
            joint_sign,
            foot_source,
            height_scan_shape,
        )
    if TensorDict is not None and isinstance(obs, TensorDict):
        return TensorDict(result, batch_size=obs.batch_size)
    return result


def mirror_obs_actions(env, obs, actions=None):
    """Left-right mirror: original + 1 mirrored pair (2x data).

    This is the recommended setting for the Track tasks: the platform terrain,
    the robot and the goal-direction observation are all left-right symmetric.
    """
    return _augment(env, obs, actions, transforms=(("lr",),))


def mirror_obs_actions_4way(env, obs, actions=None):
    """Left-right and front-back mirror: original + 3 mirrored pairs (4x data).

    This is the four-way augmentation described in the paper.  Use it only if
    the task itself is (approximately) front-back symmetric.  The Track
    platform terrain has stairs/gaps only on one side, so this option is kept
    off by default.
    """
    return _augment(
        env,
        obs,
        actions,
        transforms=(("lr",), ("fb",), ("lr", "fb")),
    )


__all__ = [
    "mirror_obs_actions",
    "mirror_obs_actions_4way",
]
