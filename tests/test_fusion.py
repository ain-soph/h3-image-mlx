import unittest

import mlx.core as mx
import numpy as np

from h3_image_mlx import core, dit, gate_norm


class FusedBlockOps(unittest.TestCase):
    """The fused DiT elementwise ops give the same bytes as the composed ops."""

    def setUp(self):
        rng = np.random.default_rng(7)
        self.segments = [(0, 5, 1), (5, 9, 2), (9, 21, 0)]
        self.x = mx.array(rng.standard_normal((21, 64)), dtype=mx.bfloat16)
        self.y = mx.array(rng.standard_normal((21, 64)), dtype=mx.bfloat16)
        self.scale = mx.array(rng.standard_normal((3, 64)), dtype=mx.float32)
        self.shift = mx.array(rng.standard_normal((3, 64)), dtype=mx.float32)

    def test_shifted(self):
        seg = dit._segment_rows(self.segments)
        expected = dit.shifted(self.x, self.scale, self.shift, self.segments)
        self.assertTrue(mx.array_equal(dit.shifted_fused(self.x, self.scale, self.shift, seg), expected))

    def test_gated_add(self):
        seg = dit._segment_rows(self.segments)
        expected = self.x + dit.modulate(self.y, self.scale, self.segments)
        self.assertTrue(mx.array_equal(dit.gated_add(self.x, self.y, self.scale, seg), expected))

    def test_swiglu_and_rope(self):
        rng = np.random.default_rng(11)
        h = mx.array(rng.standard_normal((21, 128)), dtype=mx.bfloat16)
        self.assertTrue(mx.array_equal(dit.swiglu_fused(h), core.swiglu(h)))
        q = mx.array(rng.standard_normal((4, 21, 64)), dtype=mx.bfloat16)
        angles = mx.array(rng.standard_normal((21, 32)), dtype=mx.float32)
        self.assertTrue(mx.array_equal(dit.rope_fused(q, angles), core.rope(q, angles)))


class FusedSeam(unittest.TestCase):
    """gate_norm.norm_modulate gives the same bytes as gated_add -> rms_norm -> shifted_fused."""

    def check(self, rows, width, seed):
        rng = np.random.default_rng(seed)
        a = min(7, rows // 3)
        segments = [(0, a, 1), (a, a + rows // 3, 3), (a + rows // 3, rows, 0)]
        seg = dit._segment_rows(segments)
        x = mx.array(rng.standard_normal((rows, width)) * rng.lognormal(0, 1, (rows, 1)) * 30, dtype=mx.bfloat16)
        y = mx.array(rng.standard_normal((rows, width)) * 30, dtype=mx.bfloat16)
        gate, scale, shift = (mx.array(rng.standard_normal((4, width)) * s, dtype=mx.float32) for s in (0.3, 0.5, 0.2))
        weight = mx.array(rng.standard_normal(width) * 0.3 + 1, dtype=mx.bfloat16)
        expected_x = dit.gated_add(x, y, gate, seg)
        expected = dit.shifted_fused(core.norm(expected_x, weight), scale, shift, seg)
        new_x, out = gate_norm.norm_modulate(x, weight, scale, shift, seg[0], gated=(y, gate))
        self.assertTrue(mx.array_equal(new_x, expected_x))
        self.assertTrue(mx.array_equal(out, expected))
        plain = gate_norm.norm_modulate(x, weight, scale, shift, seg[0])
        self.assertTrue(mx.array_equal(plain, dit.shifted_fused(core.norm(x, weight), scale, shift, seg)))

    def test_looped_width(self):  # the DiT's 5376: MLX's looped rms_norm
        self.check(1046, 5376, 1)
        self.check(37, 5376, 2)

    def test_single_row_width(self):
        self.check(300, 3072, 3)
        self.check(21, 64, 4)


if __name__ == "__main__":
    unittest.main()
