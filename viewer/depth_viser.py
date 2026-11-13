"""Viser viewer extension: live raw and processed depth-camera panels."""

from __future__ import annotations

import time

import numpy as np
import viser
from matplotlib import colormaps
from mjlab.viewer import ViserPlayViewer

_TURBO = colormaps["turbo"]


def _colorize_depth(
    depth_m: np.ndarray,
    max_depth: float,
    invalid: np.ndarray | None = None,
) -> np.ndarray:
    """Depth in metres -> uint8 RGB using the turbo colormap."""
    normalized = np.clip(depth_m / max(max_depth, 1e-6), 0.0, 1.0)
    rgb = (_TURBO(normalized)[..., :3] * 255.0).astype(np.uint8)
    if invalid is not None:
        rgb[invalid] = (190, 190, 190)
    return rgb


def _nearest_resize(
    image: np.ndarray,
    out_height: int,
    out_width: int,
) -> np.ndarray:
    """Nearest-neighbour resize, used to enlarge the tiny 18x32 depth map."""
    if image.shape[:2] == (out_height, out_width):
        return image
    ys = np.linspace(0, image.shape[0] - 1, out_height).round().astype(np.intp)
    xs = np.linspace(0, image.shape[1] - 1, out_width).round().astype(np.intp)
    return image[ys[:, None], xs[None, :]]


class DepthViserPlayViewer(ViserPlayViewer):
    """Viser play viewer with a live raw/processed depth-camera tab."""

    def setup(self) -> None:
        # Capture the tab group created by ViserPlayViewer.setup() so the depth
        # tab appears in the same top-level tab bar as Controls/Visualization.
        gui = self._server.gui
        original_add_tab_group = gui.add_tab_group
        captured_tabs: list[viser.GuiTabGroupHandle] = []

        def _capture_tab_group(*args, **kwargs):
            handle = original_add_tab_group(*args, **kwargs)
            if not captured_tabs:
                captured_tabs.append(handle)
            return handle

        gui.add_tab_group = _capture_tab_group  # type: ignore[method-assign]
        try:
            super().setup()
        finally:
            gui.add_tab_group = original_add_tab_group  # type: ignore[method-assign]

        env = self.env.unwrapped
        term_cfg = env.observation_manager.get_term_cfg("depth", "depth_image")
        self._depth_cutoff = float(term_cfg.params.get("cutoff_distance", 5.0))

        camera_cfg = env.scene["depth_camera"].cfg
        raw_height = int(camera_cfg.height)
        raw_width = int(camera_cfg.width)
        placeholder = np.zeros((raw_height, raw_width, 3), dtype=np.uint8)

        tabs = captured_tabs[0] if captured_tabs else gui.add_tab_group()
        with tabs.add_tab("Depth Camera", icon=viser.Icon.CAMERA):
            with gui.add_folder("Depth Camera"):
                self._depth_scale_slider = gui.add_slider(
                    "Color scale max (m)",
                    min=0.5,
                    max=10.0,
                    step=0.1,
                    initial_value=self._depth_cutoff,
                )
                self._raw_depth_handle = gui.add_image(
                    placeholder.copy(),
                    label=f"raw {raw_width}x{raw_height}",
                    format="jpeg",
                )
                self._processed_depth_handle = gui.add_image(
                    placeholder.copy(),
                    label="processed 32x18",
                    format="jpeg",
                )

        self._depth_last_update = 0.0
        self._depth_update_period = 1.0 / 15.0
        self._update_depth_images(force=True)

    def sync_env_to_viewer(self) -> None:
        super().sync_env_to_viewer()
        self._update_depth_images()

    def _update_depth_images(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._depth_last_update < self._depth_update_period:
            return
        self._depth_last_update = now

        env = self.env.unwrapped
        processed = env.obs_buf.get("depth") if env.obs_buf else None
        if processed is None:
            return

        env_idx = min(self._scene.env_idx, processed.shape[0] - 1)
        raw = env.scene["depth_camera"].data.depth
        if raw is None:
            return

        raw_np = raw[env_idx, ..., 0].detach().cpu().numpy()
        processed_np = (
            processed[env_idx, 0].detach().cpu().numpy() * self._depth_cutoff
        )

        # The slider controls the colour range only. "No return" and pixels
        # beyond the sensor cutoff stay grey in the raw panel.
        scale = max(float(self._depth_scale_slider.value), 0.1)
        raw_invalid = (raw_np <= 0.0) | (raw_np > self._depth_cutoff)
        processed_up = _nearest_resize(
            processed_np, raw_np.shape[0], raw_np.shape[1]
        )

        self._raw_depth_handle.image = _colorize_depth(
            raw_np, scale, invalid=raw_invalid
        )
        self._processed_depth_handle.image = _colorize_depth(processed_up, scale)


__all__ = ["DepthViserPlayViewer"]
