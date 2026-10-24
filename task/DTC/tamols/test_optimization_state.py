"""optimization_state 的单元测试.

覆盖: 默认布局与尺寸、切片连续性、打包/解包往返、batch 维、字典接口、
梯度和异常路径.

运行:
    python -m unittest myrobot_mjlab.task.DTC.tamols.test_optimization_state -v
"""

from __future__ import annotations

import unittest

import torch

from myrobot_mjlab.task.DTC.tamols.optimization_state import (
    STATE_BLOCK_NAMES,
    OptimizationState,
    OptimizationStateLayout,
)


def make_blocks(layout: OptimizationStateLayout, batch: tuple[int, ...] = ()):
    generator = torch.Generator().manual_seed(0)
    return {
        "spline_coeffs": torch.randn(
            *batch, *layout.block_shapes["spline_coeffs"], generator=generator
        ),
        "landing_footholds": torch.randn(
            *batch, *layout.block_shapes["landing_footholds"], generator=generator
        ),
        "desired_footholds": torch.randn(
            *batch, *layout.block_shapes["desired_footholds"], generator=generator
        ),
        "slack": torch.rand(
            *batch, *layout.block_shapes["slack"], generator=generator
        ),
    }


class TestLayout(unittest.TestCase):
    def test_default_shapes_and_slices(self):
        layout = OptimizationStateLayout()
        self.assertEqual(layout.num_phases, 1)
        self.assertEqual(layout.num_legs, 4)
        self.assertEqual(layout.num_coeffs, 5)
        self.assertEqual(layout.num_slack, 1)

        self.assertEqual(
            layout.block_shapes,
            {
                "spline_coeffs": (1, 6, 5),
                "landing_footholds": (4, 3),
                "desired_footholds": (4, 3),
                "slack": (1,),
            },
        )
        self.assertEqual(
            layout.block_sizes,
            {
                "spline_coeffs": 30,
                "landing_footholds": 12,
                "desired_footholds": 12,
                "slack": 1,
            },
        )
        self.assertEqual(layout.total_size, 55)
        self.assertEqual(
            layout.slices,
            {
                "spline_coeffs": slice(0, 30),
                "landing_footholds": slice(30, 42),
                "desired_footholds": slice(42, 54),
                "slack": slice(54, 55),
            },
        )

    def test_slices_are_contiguous_and_complete(self):
        for layout in (
            OptimizationStateLayout(),
            OptimizationStateLayout(num_phases=3, num_legs=4, num_coeffs=6, num_slack=2),
            OptimizationStateLayout(num_phases=2, num_legs=2, num_coeffs=5, num_slack=0),
        ):
            start = 0
            for name in STATE_BLOCK_NAMES:
                sl = layout.slice_of(name)
                self.assertEqual(sl.start, start, name)
                self.assertEqual(sl.stop - sl.start, layout.block_sizes[name], name)
                start = sl.stop
            self.assertEqual(start, layout.total_size)

    def test_slack_defaults_to_num_phases(self):
        layout = OptimizationStateLayout(num_phases=4)
        self.assertEqual(layout.num_slack, 4)
        self.assertEqual(layout.block_shapes["slack"], (4,))

    def test_custom_layout_size(self):
        layout = OptimizationStateLayout(
            num_phases=2, num_legs=4, num_coeffs=6, num_slack=3
        )
        # 2*6*6 + 4*3 + 4*3 + 3
        self.assertEqual(layout.total_size, 72 + 12 + 12 + 3)

    def test_describe(self):
        text = OptimizationStateLayout().describe()
        for name in STATE_BLOCK_NAMES:
            self.assertIn(name, text)
        self.assertIn("total=55", text)

    def test_invalid_layout(self):
        with self.assertRaises(ValueError):
            OptimizationStateLayout(num_phases=0)
        with self.assertRaises(ValueError):
            OptimizationStateLayout(num_legs=0)
        with self.assertRaises(ValueError):
            OptimizationStateLayout(num_coeffs=0)
        with self.assertRaises(ValueError):
            OptimizationStateLayout(base_dims=5)
        with self.assertRaises(ValueError):
            OptimizationStateLayout(num_slack=-1)
        with self.assertRaises(TypeError):
            OptimizationStateLayout(num_phases=2.0)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            OptimizationStateLayout(num_slack=1.0)  # type: ignore[arg-type]
        with self.assertRaises(KeyError):
            OptimizationStateLayout().slice_of("missing")


class TestPackUnpack(unittest.TestCase):
    def setUp(self):
        self.layout = OptimizationStateLayout()

    def test_round_trip(self):
        blocks = make_blocks(self.layout)
        x = self.layout.pack(**blocks)
        self.assertEqual(tuple(x.shape), (55,))

        state = self.layout.unpack(x)
        self.assertIsInstance(state, OptimizationState)
        for name in STATE_BLOCK_NAMES:
            self.assertTrue(torch.equal(state[name], blocks[name]), name)
            self.assertEqual(
                tuple(state[name].shape), self.layout.block_shapes[name], name
            )

    def test_block_order_in_x(self):
        blocks = make_blocks(self.layout)
        x = self.layout.pack(**blocks)
        sl = self.layout.slices["spline_coeffs"]
        self.assertTrue(
            torch.equal(
                x[sl],
                blocks["spline_coeffs"].reshape(-1),
            )
        )
        sl = self.layout.slices["slack"]
        self.assertTrue(torch.equal(x[sl], blocks["slack"].reshape(-1)))

    def test_batched_round_trip(self):
        for batch in ((2,), (2, 3)):
            blocks = make_blocks(self.layout, batch)
            x = self.layout.pack(**blocks)
            self.assertEqual(tuple(x.shape), (*batch, 55))
            state = self.layout.unpack(x)
            for name in STATE_BLOCK_NAMES:
                self.assertEqual(
                    tuple(state[name].shape),
                    (*batch, *self.layout.block_shapes[name]),
                    name,
                )
                self.assertTrue(torch.equal(state[name], blocks[name]), name)

    def test_pack_state_mapping_and_dataclass(self):
        blocks = make_blocks(self.layout)
        x1 = self.layout.pack_state(blocks)
        x2 = self.layout.pack_state(OptimizationState(**blocks))
        self.assertTrue(torch.equal(x1, x2))

    def test_state_as_dict_and_getitem(self):
        blocks = make_blocks(self.layout)
        state = OptimizationState(**blocks)
        self.assertEqual(set(state.as_dict()), set(STATE_BLOCK_NAMES))
        for name in STATE_BLOCK_NAMES:
            self.assertIs(state[name], blocks[name])
        with self.assertRaises(KeyError):
            _ = state["missing"]

    def test_zeros(self):
        x = self.layout.zeros()
        self.assertEqual(tuple(x.shape), (55,))
        self.assertTrue(bool((x == 0).all()))
        x = self.layout.zeros((2, 3), dtype=torch.float64)
        self.assertEqual(tuple(x.shape), (2, 3, 55))
        self.assertEqual(x.dtype, torch.float64)

    def test_gradients_flow(self):
        blocks = {
            name: torch.zeros(
                *self.layout.block_shapes[name], requires_grad=True
            )
            for name in STATE_BLOCK_NAMES
        }
        x = self.layout.pack(**blocks)
        x.sum().backward()
        for name in STATE_BLOCK_NAMES:
            grad = blocks[name].grad
            self.assertIsNotNone(grad, name)
            self.assertTrue(bool((grad == 1).all()), name)

    def test_pack_shape_errors(self):
        blocks = make_blocks(self.layout)
        bad = dict(blocks)
        bad["landing_footholds"] = torch.zeros(4, 2)
        with self.assertRaises(ValueError):
            self.layout.pack(**bad)

        bad = dict(blocks)
        bad["spline_coeffs"] = torch.zeros(6, 5)  # 缺少相位维
        with self.assertRaises(ValueError):
            self.layout.pack(**bad)

        bad = dict(blocks)
        bad["slack"] = torch.zeros(2)  # S=1
        with self.assertRaises(ValueError):
            self.layout.pack(**bad)

    def test_pack_batch_mismatch(self):
        blocks = make_blocks(self.layout, (2,))
        blocks["slack"] = torch.zeros(3, 1)
        with self.assertRaises(ValueError):
            self.layout.pack(**blocks)

    def test_pack_type_errors(self):
        blocks = make_blocks(self.layout)
        bad = dict(blocks)
        bad["slack"] = [0.0]  # type: ignore[assignment]
        with self.assertRaises(TypeError):
            self.layout.pack(**bad)

        bad = dict(blocks)
        bad["slack"] = torch.zeros(1, dtype=torch.int64)
        with self.assertRaises(TypeError):
            self.layout.pack(**bad)

    def test_unpack_errors(self):
        with self.assertRaises(TypeError):
            self.layout.unpack([0.0] * 55)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            self.layout.unpack(torch.zeros(54))
        with self.assertRaises(ValueError):
            self.layout.unpack(torch.zeros(2, 56))

    def test_custom_layout_round_trip(self):
        layout = OptimizationStateLayout(
            num_phases=3, num_legs=4, num_coeffs=6, num_slack=3
        )
        blocks = make_blocks(layout, (2,))
        x = layout.pack(**blocks)
        self.assertEqual(tuple(x.shape), (2, layout.total_size))
        state = layout.unpack(x)
        for name in STATE_BLOCK_NAMES:
            self.assertTrue(torch.equal(state[name], blocks[name]), name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
