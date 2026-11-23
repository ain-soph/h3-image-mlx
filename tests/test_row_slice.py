"""(ain-soph fork) H3_ROW_SLICE: the row-sliced DiT projections give the same bytes as the whole sequence."""
import re
import unittest

import mlx.core as mx
import numpy as np

from h3_image_mlx import core, dit
from h3_image_mlx.qk_rope import prepare_angles

HIDDEN, FFN = 512, 1024


class FakeWeights:
    """One block of random int8 ConvRot projections (4 heads) behind the Weights interface the DiT uses."""

    def __init__(self):
        rng = np.random.default_rng(3)
        self.tensors = {}
        for name, (k, n) in {"a.qkv_proj": (HIDDEN, 3 * HIDDEN), "a.out_proj": (HIDDEN, HIDDEN),
                             "m.fc1": (HIDDEN, 2 * FFN), "m.fc2": (FFN, HIDDEN)}.items():
            self.tensors[name + ".weight"] = mx.array(rng.integers(-127, 128, (n, k)), dtype=mx.int8)
            self.tensors[name + ".weight_scale"] = mx.array(rng.uniform(1e-4, 1e-3, (n,)), dtype=mx.float32)
        for name in ("a.q_norm.weight", "a.k_norm.weight"):
            self.tensors[name] = mx.array(rng.uniform(0.5, 1.5, (128,)), dtype=mx.bfloat16)
        self.header = dict.fromkeys(self.tensors)
        self.cache = {}
        self.reads = 0

    def __getitem__(self, key):
        self.reads += 1
        return self.cache.get(key, self.tensors[key])

    def linear(self, prefix, x):
        return core.w8a8(x, self[prefix + ".weight"], self[prefix + ".weight_scale"])


@unittest.skipUnless(re.search(r"\bApple M5\b", mx.device_info()["device_name"]), "W8A8 TensorOps need an M5 GPU")
class RowSliceTests(unittest.TestCase):
    def setUp(self):
        core.backend.set("w8a8")
        self.saved = dit.ROW_SLICE, dit.ROW_SLICE_FROM
        rng = np.random.default_rng(5)
        self.rows = 300
        self.x = mx.array(rng.standard_normal((self.rows, HIDDEN)) * 0.5, dtype=mx.bfloat16)
        self.angles = mx.array(rng.standard_normal((self.rows, 32)), dtype=mx.float32)
        self.w = FakeWeights()

    def tearDown(self):
        dit.ROW_SLICE, dit.ROW_SLICE_FROM = self.saved

    def both(self, fn):
        dit.ROW_SLICE = 0
        whole = fn()
        mx.eval(whole)
        dit.ROW_SLICE, dit.ROW_SLICE_FROM = 128, 0      # slices of 128, 128 and 44 rows
        sliced = fn()
        mx.eval(sliced)
        return whole, sliced

    def test_mlp(self):
        whole, sliced = self.both(lambda: dit.mlp(self.w, "m", self.x))
        self.assertTrue(mx.array_equal(whole, sliced))
        self.assertEqual(self.w.cache, {})

    def test_attention_rope(self):
        whole, sliced = self.both(lambda: dit.qkv_attention(self.w, "a", self.x, self.angles))
        self.assertTrue(mx.array_equal(whole, sliced))

    def test_attention_fused_norm_rope(self):
        trig = prepare_angles(self.angles, mx.bfloat16)
        whole, sliced = self.both(lambda: dit.qkv_attention(self.w, "a", self.x, self.angles, trig))
        self.assertTrue(mx.array_equal(whole, sliced))

    def test_only_long_sequences_slice(self):
        dit.ROW_SLICE, dit.ROW_SLICE_FROM = 16384, 35000
        self.assertIsNone(dit._slices(16406))       # 4096^2
        self.assertIsNone(dit._slices(34003))       # 1024^2 + 16 references
        self.assertEqual([(s.start, s.stop) for s in dit._slices(37075)], [(0, 16384), (16384, 32768), (32768, 37075)])
        dit.ROW_SLICE = 0
        self.assertIsNone(dit._slices(100000))


if __name__ == "__main__":
    unittest.main()
