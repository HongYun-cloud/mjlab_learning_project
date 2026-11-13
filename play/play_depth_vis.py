"""Launch a Track task in Viser with live raw + processed depth panels.

Usage (from the repository root, /home/hong/myrobot_mjlab):

  # Zero policy (no checkpoint); useful for inspecting the depth pipeline.
  uv run python src/myrobot_mjlab/play/play_depth_vis.py -e flat

  # Trained policy / checkpoint.
  uv run python src/myrobot_mjlab/play/play_depth_vis.py -e stair \
      -c logs/rsl_rl/go2_velocity/<run>/model_5000.pt

The Viser GUI gets an extra "Depth Camera" tab with:
  - raw depth at 270x480 (HxW)
  - processed depth at 18x32 (HxW), nearest-upscaled for display
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mjlab.scripts.play as play_script
from mjlab.scripts.play import PlayConfig
from myrobot_mjlab.viewer.depth_viser import DepthViserPlayViewer

_TASKS = {
    "flat": "Mjlab-Tracking-Flat-Unitree-Go2",
    "stair": "Mjlab-Tracking-Stair-Unitree-Go2",
    "stair_up": "Mjlab-Tracking-StairUp-Unitree-Go2",
    "stair_down": "Mjlab-Tracking-StairDown-Unitree-Go2",
    "platform": "Mjlab-Tracking-Platform-Unitree-Go2",
    "plum": "Mjlab-Tracking-PlumBlossom-Unitree-Go2",
    "stones": "Mjlab-Tracking-Stones2Rows-Unitree-Go2",
    "tire": "Mjlab-Tracking-TireArray-Unitree-Go2",
    "low_clearance_bar": "Mjlab-Tracking-LowClearanceBar-Unitree-Go2",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-e",
        "--terrain",
        choices=sorted(_TASKS),
        default="flat",
        help="Track terrain to visualize (default: flat).",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional policy checkpoint. Without it a zero-action policy is used.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel environments (default: 1).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. cuda:0 or cpu (default: auto).",
    )
    parser.add_argument(
        "--log-root",
        default="logs/rsl_rl",
        help="Checkpoint root passed to the play script (default: logs/rsl_rl).",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    # mjlab.scripts.play resolves ViserPlayViewer from this module global at
    # call time, so replacing it here installs the depth-panel subclass.
    play_script.ViserPlayViewer = DepthViserPlayViewer

    play_script.run_play(
        _TASKS[args.terrain],
        PlayConfig(
            agent="trained" if checkpoint else "zero",
            checkpoint_file=str(checkpoint) if checkpoint else None,
            viewer="viser",
            num_envs=args.num_envs,
            device=args.device,
            log_root=args.log_root,
        ),
    )


if __name__ == "__main__":
    main()
