"""Visualize raw and processed Track depth-camera images.

The Track depth camera renders at 270x480 (HxW).  The observation pipeline
crops/resizes/blurs it into an 18x32 (HxW) depth image.  This script saves a
side-by-side comparison so both stages can be inspected.

Usage (from the repository root, /home/hong/myrobot_mjlab):

  .venv/bin/python src/myrobot_mjlab/test/visualize_depth_camera.py -e flat

  # Capture 3 frames: reset + step 20 + step 40
  .venv/bin/python src/myrobot_mjlab/test/visualize_depth_camera.py \
      -e stair --steps 40 --interval 20

Outputs (default: ``depth_vis/``):
  comparison_000.png       raw 270x480 vs processed 18x32, side by side
  raw_depth_000.png        raw depth only
  processed_depth_000.png  processed depth only (nearest-upscaled)
  raw_depth_000.npy        raw depth in metres (H, W)
  processed_depth_000.npy  processed depth in metres (H, W)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from myrobot_mjlab.task.track.config.env_cfg import (
    my_go2_flat_env_cfg,
    my_go2_low_clearance_bar_env_cfg,
    my_go2_platform_env_cfg,
    my_go2_plum_env_cfg,
    my_go2_stair_down_env_cfg,
    my_go2_stair_env_cfg,
    my_go2_stair_up_env_cfg,
    my_go2_stones_2rows_env_cfg,
    my_go2_tire_env_cfg,
)

_TERRAINS = {
    "flat": my_go2_flat_env_cfg,
    "stair": my_go2_stair_env_cfg,
    "stair_up": my_go2_stair_up_env_cfg,
    "stair_down": my_go2_stair_down_env_cfg,
    "platform": my_go2_platform_env_cfg,
    "plum": my_go2_plum_env_cfg,
    "stones": my_go2_stones_2rows_env_cfg,
    "tire": my_go2_tire_env_cfg,
    "low_clearance_bar": my_go2_low_clearance_bar_env_cfg,
}


def _save_frame(
    output_dir: Path,
    index: int,
    raw_depth: np.ndarray,
    processed_depth: np.ndarray,
    cutoff_distance: float,
) -> None:
    """Save one raw/processed comparison frame and the underlying arrays."""
    # MuJoCo returns 0 for no hit and can return very large values for the sky.
    raw_invalid = (raw_depth <= 0.0) | (raw_depth > cutoff_distance)
    raw_masked = np.ma.masked_where(raw_invalid, raw_depth)

    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("0.75")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
    im_raw = axes[0].imshow(
        raw_masked,
        cmap=cmap,
        vmin=0.0,
        vmax=cutoff_distance,
        interpolation="nearest",
        aspect="auto",
    )
    axes[0].set_title(
        f"raw depth: {raw_depth.shape[0]}x{raw_depth.shape[1]} (HxW), "
        "gray = no return / > cutoff"
    )
    axes[0].set_xlabel("W")
    axes[0].set_ylabel("H")
    fig.colorbar(im_raw, ax=axes[0], label="metres")

    im_proc = axes[1].imshow(
        processed_depth,
        cmap="turbo",
        vmin=0.0,
        vmax=cutoff_distance,
        interpolation="nearest",
        aspect="auto",
    )
    axes[1].set_title(
        f"processed depth: {processed_depth.shape[0]}x{processed_depth.shape[1]} (HxW)"
    )
    axes[1].set_xlabel("W")
    axes[1].set_ylabel("H")
    fig.colorbar(im_proc, ax=axes[1], label="metres")

    fig.savefig(output_dir / f"comparison_{index:03d}.png", dpi=150)
    plt.close(fig)

    # Individual images at native aspect. Processed is nearest-upscaled so the
    # 18x32 grid is still visible when opened directly.
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.imshow(
        raw_masked,
        cmap=cmap,
        vmin=0.0,
        vmax=cutoff_distance,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_title(f"raw depth: {raw_depth.shape[0]}x{raw_depth.shape[1]} (HxW)")
    fig.colorbar(ax.images[0], ax=ax, label="metres")
    fig.savefig(output_dir / f"raw_depth_{index:03d}.png", dpi=150)
    plt.close(fig)

    scale = max(1, 270 // processed_depth.shape[0], 480 // processed_depth.shape[1])
    processed_up = np.repeat(
        np.repeat(processed_depth, scale, axis=0), scale, axis=1
    )
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.imshow(
        processed_up,
        cmap="turbo",
        vmin=0.0,
        vmax=cutoff_distance,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_title(
        "processed depth: "
        f"{processed_depth.shape[0]}x{processed_depth.shape[1]} (HxW), "
        f"shown at {scale}x"
    )
    fig.colorbar(ax.images[0], ax=ax, label="metres")
    fig.savefig(output_dir / f"processed_depth_{index:03d}.png", dpi=150)
    plt.close(fig)

    np.save(output_dir / f"raw_depth_{index:03d}.npy", raw_depth)
    np.save(output_dir / f"processed_depth_{index:03d}.npy", processed_depth)

    print(
        f"[frame {index:03d}] raw={raw_depth.shape} "
        f"min/max={raw_depth.min():.4f}/{raw_depth.max():.4f} m | "
        f"processed={processed_depth.shape} "
        f"min/max={processed_depth.min():.4f}/{processed_depth.max():.4f} m"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-e",
        "--terrain",
        choices=sorted(_TERRAINS),
        default="flat",
        help="Track terrain to visualize (default: flat).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Number of environment steps to run (default: 0, reset only).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=20,
        help="Save a frame every N steps (default: 20).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("depth_vis"),
        help="Directory for PNG/NPY outputs (default: depth_vis).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. cuda:0 or cpu (default: auto).",
    )
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _TERRAINS[args.terrain](play=True)
    cfg.scene.num_envs = 1

    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    try:
        obs, _ = env.reset()

        term_cfg = env.observation_manager.get_term_cfg("depth", "depth_image")
        cutoff_distance = float(term_cfg.params["cutoff_distance"])

        sensor = env.scene["depth_camera"]
        print(
            f"camera raw resolution: {sensor.cfg.width}x{sensor.cfg.height} "
            f"(WxH), cutoff={cutoff_distance} m"
        )

        action = torch.zeros(
            (1, env.action_manager.total_action_dim),
            device=env.device,
        )

        def capture(index: int) -> None:
            raw_depth = sensor.data.depth
            assert raw_depth is not None
            raw_depth = raw_depth[0, ..., 0].detach().cpu().numpy()

            processed = obs["depth"]
            assert processed is not None
            processed_depth = (
                processed[0, 0].detach().cpu().numpy() * cutoff_distance
            )

            _save_frame(
                output_dir,
                index,
                raw_depth,
                processed_depth,
                cutoff_distance,
            )

        capture(0)
        for step in range(1, args.steps + 1):
            obs, *_ = env.step(action)
            if step % max(args.interval, 1) == 0:
                capture(step)

    finally:
        env.close()

    print(f"\nSaved to: {output_dir}")
    print("Open comparison_000.png first.")


if __name__ == "__main__":
    main()
