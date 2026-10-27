"""Tests for the standalone elevation-map UI data projection."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from myrobot_mjlab.task.DTC.tamols.elevation_map_ui import (
    build_elevation_map_snapshot,
)


class TestElevationMapSnapshot(unittest.TestCase):
    def _command(self):
        heights = torch.zeros((1, 1, 4, 4))
        heights.fill_(0.3)
        valid = torch.ones_like(heights)
        return SimpleNamespace(
            h_raw=heights,
            h_valid=valid,
            cell_size=0.1,
            x_min=0.0,
            x_max=0.4,
            y_min=0.0,
            y_max=0.4,
            ground_default=torch.tensor([0.1]),
            footholds_initialized=torch.tensor([True]),
            optimal_footholds=torch.tensor(
                [[[1.0, 2.1, 0.3], [1.0, 1.9, 0.3],
                  [0.8, 2.1, 0.3], [0.8, 1.9, 0.3]]]
            ),
            landing_footholds=torch.tensor(
                [[[1.0, 2.1, 0.3], [1.0, 1.9, 0.3],
                  [0.8, 2.1, 0.3], [0.8, 1.9, 0.3]]]
            ),
            nominal_footholds=torch.tensor(
                [[[1.0, 2.1, 0.3], [1.0, 1.9, 0.3],
                  [0.8, 2.1, 0.3], [0.8, 1.9, 0.3]]]
            ),
            robot=SimpleNamespace(
                data=SimpleNamespace(
                    root_link_pos_w=torch.tensor([[1.0, 2.0, 3.0]]),
                    heading_w=torch.tensor([np.pi / 2]),
                )
            ),
        )

    def test_mesh_and_world_footholds_project_into_body_xy(self):
        snapshot = build_elevation_map_snapshot(self._command(), stride=2)

        self.assertEqual(snapshot.vertices.shape, (4, 3))
        np.testing.assert_allclose(snapshot.vertices[:, 2], 0.2)
        self.assertEqual(snapshot.faces.shape, (6,))
        self.assertEqual(snapshot.observed_cells, 16)
        self.assertEqual(snapshot.total_cells, 16)
        self.assertAlmostEqual(snapshot.min_height, 0.3)
        self.assertAlmostEqual(snapshot.max_height, 0.3)
        self.assertEqual(set(snapshot.footholds), {"optimal", "landing", "nominal"})
        np.testing.assert_allclose(
            snapshot.footholds["optimal"][0], [0.1, 0.0, 0.245], atol=1e-6
        )

    def test_unknown_map_and_uninitialized_plan_are_empty(self):
        cmd = self._command()
        cmd.h_valid.zero_()
        cmd.footholds_initialized.zero_()
        snapshot = build_elevation_map_snapshot(cmd, stride=2)

        self.assertEqual(snapshot.vertices.shape, (4, 3))
        self.assertEqual(snapshot.faces.shape, (0,))
        self.assertEqual(snapshot.footholds, {})
        self.assertIsNone(snapshot.min_height)
        self.assertIsNone(snapshot.max_height)


if __name__ == "__main__":
    unittest.main()
