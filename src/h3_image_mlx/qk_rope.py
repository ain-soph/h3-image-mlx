"""Fused 128-wide Q/K normalization and rotary embedding for the H3 DiT."""
from functools import cache

from .core import mx

_SOURCE = """
    const uint lane = thread_index_in_simdgroup;
    const uint item = thread_position_in_grid.x / 32;
    if (item >= rows * HEADS * 2) return;
    const uint row = item % rows;
    const uint pair_head = item / rows;
    const uint head = pair_head % HEADS;
    const bool is_key = pair_head >= HEADS;
    const uint d = lane * 4;
    const device T* x = qkv + row * (3 * HEADS * 128) + pair_head * 128 + d;
    const device T* w = (is_key ? wk : wq) + d;
    // Match MLX RMSNorm's four adjacent values per lane.
    float a[4];
    float sum = 0.0f;
    for (uint j = 0; j < 4; ++j) {
        a[j] = float(x[j]);
        sum += a[j] * a[j];
    }
    float inv = metal::precise::rsqrt(simd_sum(sum) / 128.0f + 1e-5f);
    const uint target = (head * rows + row) * 128 + d;
    const uint other = d < ROT ? lane + ROT / 4 : lane - ROT / 4;
    for (uint j = 0; j < 4; ++j) {
        // Preserve the original BF16 rounding points across RMSNorm and RoPE.
        T value = T(w[j] * T(a[j] * inv));
        T peer = simd_shuffle(value, ushort(other & 31));
        if (d < 2 * ROT) {
            uint angle = row * ROT + (d % ROT) + j;
            T product = T(value * cosines[angle]);
            T rotated = T(peer * sines[angle]);
            value = d < ROT ? T(product - rotated) : T(rotated + product);
        }
        if (is_key) kout[target + j] = value;
        else qout[target + j] = value;
    }
"""


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="h3_qk_norm_rope",
        input_names=["qkv", "wq", "wk", "cosines", "sines", "rows"],
        output_names=["qout", "kout"],
        source=_SOURCE,
    )


def prepare_angles(angles, dtype):
    return mx.cos(angles).astype(dtype), mx.sin(angles).astype(dtype)


def normalize_rotate(qkv, q_weight, k_weight, trigonometry):
    rows = qkv.shape[0]
    heads = qkv.shape[-1] // (3 * 128)
    cosines, sines = trigonometry
    rot = cosines.shape[-1]
    shape = (1, heads, rows, 128)
    simdgroups = rows * heads * 2
    return _kernel()(
        inputs=[qkv, q_weight.astype(qkv.dtype), k_weight.astype(qkv.dtype), cosines, sines, rows],
        template=[("T", qkv.dtype), ("HEADS", heads), ("ROT", rot)],
        grid=(((simdgroups + 3) // 4) * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[shape, shape],
        output_dtypes=[qkv.dtype, qkv.dtype],
    )
