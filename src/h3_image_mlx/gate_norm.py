"""(ain-soph fork) The DiT block seam in one kernel: gated residual + RMSNorm + AdaLN modulation.

After mlx-serve's `msv_gate_norm_mod` (ddalcu/mlx-serve, MIT, src/sdk/blocks/norms/add_norm.zig). Per row r with
table row m = ids[r]:

    x' = x + y * gate[m]                      (only in the gated form; bfloat16 product, then bfloat16 sum)
    h  = rmsnorm(x') * w * (1 + scale[m]) + shift[m]

The composed ops are dit.gated_add, core.norm (mx.fast.rms_norm) and dit.shifted_fused: three kernels and three
passes over the activations. This kernel rounds to bfloat16 exactly where they do and sums the squares in MLX's own
order (four adjacent values per thread, a simdgroup sum, then a sum over the simdgroups; up to 4096 wide one pass of
ceil(D/4) threads, wider MLX's looped kernel with 1024 threads), so it gives the same bytes.
"""
from functools import cache

from .core import mx

_LOAD_GATED = """
            const T hn = T(float(x[at]) + float(T(float(y[at]) * float(gate[m + c]))));
            xout[at] = hn;"""
_LOAD_PLAIN = """
            const T hn = x[at];"""

_SOURCE = """
    const uint t = thread_position_in_threadgroup.x;
    const uint r = threadgroup_position_in_grid.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    constexpr int NC = (D + TN * 4 - 1) / (TN * 4);
    threadgroup float local_sums[32];
    float hv[NC * 4];
    float acc = 0.0f;
    const size_t m = size_t(ids[r]) * D;
    for (int p = 0; p < NC; p++) {
        for (int i = 0; i < 4; i++) {
            const int c = p * TN * 4 + int(t) * 4 + i;
            hv[p * 4 + i] = 0.0f;
            if (c < D) {
                const size_t at = size_t(r) * D + c;%s
                hv[p * 4 + i] = float(hn);
            }
            acc += hv[p * 4 + i] * hv[p * 4 + i];
        }
    }
    acc = simd_sum(acc);
    if (sg == 0) local_sums[lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) local_sums[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = simd_sum(local_sums[lane]);
    const float inv = metal::precise::rsqrt(total / float(D) + 1e-5f);
    for (int p = 0; p < NC; p++) {
        for (int i = 0; i < 4; i++) {
            const int c = p * TN * 4 + int(t) * 4 + i;
            if (c < D) {
                const T n = T(float(w[c]) * float(T(hv[p * 4 + i] * inv)));
                const T sc1 = T(float(scale[m + c]) + 1.0f);
                const T mm = T(float(n) * float(sc1));
                out[size_t(r) * D + c] = T(float(mm) + float(shift[m + c]));
            }
        }
    }
"""

# MLX's rms_norm: one pass for rows up to RMS_LOOPED_LIMIT wide, else the looped kernel at the pipeline's maximum
# threadgroup size (maxTotalThreadsPerThreadgroup: 1024 on the Apple GPUs measured, M5). A GPU with another maximum
# would sum the squares in another order: tests/test_fusion.py FusedSeam (5376 wide) catches that.
_LOOPED_LIMIT = 4096
_LOOPED_THREADS = 1024


@cache
def _kernel(gated):
    inputs = ["x", "y", "gate", "w", "scale", "shift", "ids"] if gated else ["x", "w", "scale", "shift", "ids"]
    outputs = ["xout", "out"] if gated else ["out"]
    return mx.fast.metal_kernel(
        name="h3_gate_norm_mod" if gated else "h3_norm_mod",
        input_names=inputs,
        output_names=outputs,
        source=_SOURCE % (_LOAD_GATED if gated else _LOAD_PLAIN),
    )


def _threads(d):
    if d > _LOOPED_LIMIT:
        return _LOOPED_THREADS
    per_thread = (d + 3) // 4
    return 32 * ((per_thread + 31) // 32)


def supported(x):
    return x.ndim == 2 and x.dtype in (mx.bfloat16, mx.float16) and 0 < x.shape[1] <= 2 * _LOOPED_LIMIT


def norm_modulate(x, weight, scale, shift, ids, gated=None):
    """rmsnorm(x') * weight * (1 + scale[ids]) + shift[ids] with x' = x, or x + y * gate[ids] for gated=(y, gate).

    x: (rows, D); weight: (D,); scale, shift, gate: (table rows, D); ids: (rows,) or (rows, 1) int32.
    Returns the modulated rows, and with gated also x' (first)."""
    rows, d = x.shape
    tn = _threads(d)
    dtype = x.dtype
    tables = [scale.astype(dtype), shift.astype(dtype)]
    common = [weight.astype(dtype), *tables, ids.reshape(-1).astype(mx.int32)]
    inputs = [x, gated[0].astype(dtype), gated[1].astype(dtype), *common] if gated else [x, *common]
    outs = _kernel(bool(gated))(
        inputs=inputs,
        template=[("T", dtype), ("D", d), ("TN", tn)],
        grid=(tn * rows, 1, 1),
        threadgroup=(tn, 1, 1),
        output_shapes=[x.shape] * (2 if gated else 1),
        output_dtypes=[dtype] * (2 if gated else 1),
    )
    return (outs[0], outs[1]) if gated else outs[0]
