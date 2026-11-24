import os
import re
import unittest
from unittest import mock

import mlx.core as mx
import numpy as np

from h3_image_mlx import core, dit, gate_norm, text


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


class W8A16(unittest.TestCase):
    """core.w8a16 (rotate + TensorOps bfloat16 x int8 GEMM) against float32 math on the same rotated rows, including
    row counts that are not a tile multiple and widths that end in an overlapping last tile."""

    def test_against_float(self):
        if not re.search(r"\bApple M5\b", mx.device_info()["device_name"]):
            self.skipTest("TensorOps W8A16 needs an M5 GPU")
        rng = np.random.default_rng(0)
        token = core.backend.set("w8a8")
        try:
            for rows, k, n in [(5, 512, 200), (37, 768, 384), (70, 512, 130), (64, 256, 128), (129, 512, 191)]:
                x = mx.array(rng.standard_normal((rows, k)), dtype=mx.bfloat16)
                wq = mx.array(rng.integers(-127, 128, (n, k)), dtype=mx.int8)
                s = mx.array(rng.random(n) * 0.01 + 0.001, dtype=mx.float32)
                xr = core.rotate(x)[:rows].astype(mx.float32)
                exact = (x.astype(mx.float32).reshape(-1, 256) @ core.H256).reshape(x.shape)
                self.assertLess(mx.max(mx.abs(xr - exact)).item(), 0.05)
                y = core.w8a16(x, wq, s).astype(mx.float32)
                ref = (xr @ wq.astype(mx.float32).T) * s.reshape(1, -1)
                self.assertEqual(y.shape, (rows, n))
                self.assertLess(mx.max(mx.abs(y - ref) / (mx.abs(ref) + 1e-3)).item(), 2 ** -7)
        finally:
            core.backend.reset(token)


class TextW8A16Classes(unittest.TestCase):
    def test_parse(self):
        token = core.backend.set("w8a8")
        try:
            with mock.patch.dict(os.environ):
                os.environ.pop("H3_TE_W8A16", None)
                self.assertEqual(text.text_w8a16(), {"down"})
                os.environ["H3_TE_W8A16"] = "none"
                self.assertEqual(text.text_w8a16(), set())
                os.environ["H3_TE_W8A16"] = " down, k "
                self.assertEqual(text.text_w8a16(), {"down", "k"})
                os.environ["H3_TE_W8A16"] = "all"
                self.assertEqual(len(text.text_w8a16()), 7)
                os.environ["H3_TE_W8A16"] = "dwon"
                with self.assertRaises(ValueError):
                    text.text_w8a16()
                core.backend.set("portable")
                os.environ.pop("H3_TE_W8A16")
                self.assertEqual(text.text_w8a16(), set())
        finally:
            core.backend.reset(token)


if __name__ == "__main__":
    unittest.main()
