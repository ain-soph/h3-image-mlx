"""H3 Image inference primitives. Metal GEMM from ddalcu/mlx-serve @ 9191fbdf."""

import fcntl
import json
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from functools import cache
from pathlib import Path

import mlx.core as mx
import numpy as np

backend = ContextVar("h3_backend", default="portable")
legacy_int8_reads = ContextVar("h3_legacy_int8_reads", default=False)
m5_resident = ContextVar("h3_m5_resident", default=False)


H4 = np.ones((4, 4), dtype=np.float32)
np.fill_diagonal(H4[:, ::-1], -1)
H256 = mx.array(np.kron(np.kron(H4, H4), np.kron(H4, H4)) / 16)
DTYPES = {"I8": "i1", "U8": "u1", "F16": "<f2", "BF16": "<u2", "F32": "<f4", "I64": "<i8"}
# Zero-copy weights: see align.py.
PAGE = 16384


def page_aligned(header, start):
    """True for a file written by align.py: every tensor starts on a 16 KiB page."""
    return (header.get("__metadata__") or {}).get("h3_aligned") == str(PAGE) and start % PAGE == 0


def flag(name):
    """Opt-in tuning knob read from the environment; unset or invalid means off."""
    try:
        return float(os.environ.get(name) or 0)
    except ValueError:
        return 0.0


@mx.compile
def rotate_quantize_reference(x):
    rotated = (x.astype(mx.float32).reshape(-1, 256) @ H256).reshape(x.shape)
    scale = mx.maximum(mx.max(mx.abs(rotated), axis=-1) / 127, 1e-20)
    return mx.clip(mx.round(rotated / scale[:, None]), -127, 127).astype(mx.int8), scale


def rotate_quantize(x, pad_rows=1):
    """Per-row int8 quantization of x H256 (block-diagonal along the last axis) with float32 scales.
    The fast kernel serves the M5 TensorOps path (backend w8a8); the portable path and H3_ROTATE=reference keep the
    float32-matmul formulation. The int8 rows are allocated up to a multiple of pad_rows (the GEMM tile height) so
    that the GEMM never reads past them; the extra rows only feed output rows that are never stored."""
    if backend.get() != "w8a8" or os.environ.get("H3_ROTATE") == "reference" or x.ndim != 2 or x.shape[1] % 256:
        xq, xs = rotate_quantize_reference(x)
        pad = -x.shape[0] % pad_rows if x.ndim == 2 else 0
        return (mx.pad(xq, ((0, pad), (0, 0))) if pad else xq), xs
    # (ain-soph fork) The fused kernel is MLX's (mx.fast.convrot_quantize, ain-soph/mlx serving): one threadgroup
    # per row, Hadamard-256 as base-4 butterflies, rows padded to a multiple of 128 (every GEMM tile height used here).
    return mx.fast.convrot_quantize(x)


def w8a8(x, weight, scale):
    rows = x.shape[0]
    n = weight.shape[0]
    if backend.get() == "portable":
        from ._portable import portable_matmul
        xq, xs = rotate_quantize(x)
        return portable_matmul(xq, weight, xs, scale.reshape(-1), x.dtype)
    if m5_resident.get():
        bm, bn = (32, 64) if rows <= 32 else (64, 128)
        bk = 256
    else:
        # Tile sweep at the DiT shapes on an M5 Ultra: 64x128 output tiles are the fastest or within 3% for every
        # projection (the 1046-row QKV GEMM: 128x64 82 TOPS, 2.94 ms; 64x128 143 TOPS, 1.69 ms), and K steps of 32
        # win from about 2000 rows up. The int32 sums are exact, so every tiling gives the same bytes.
        # (ain-soph fork) H3_GEMM_TILES=bm,bn,bk overrides the tiles (A/B).
        bm, bn, bk = 64, 128, (64 if rows < 2048 else 32)
        if os.environ.get("H3_GEMM_TILES"):
            bm, bn, bk = (int(v) for v in os.environ["H3_GEMM_TILES"].split(","))
    xq, xs = rotate_quantize(x, pad_rows=bm)
    if n < bn:
        # narrower than one column tile, which the kernel cannot place inside the weight: float product
        acc = xq[:rows].astype(mx.float32) @ weight.astype(mx.float32).T
        return (acc * xs.reshape(-1, 1) * scale.reshape(1, -1)).astype(x.dtype)
    # mlx-serve's INT8 TensorOps GEMM (MIT), in MLX (mx.fast.convrot_matmul_quantized) with this sweep's tiles
    return mx.fast.convrot_matmul_quantized(xq, xs, weight, scale.reshape(-1), x.dtype, tiles=(bm, bn, bk))


# (ain-soph fork, 2026-11-18) W8A16 for ConvRot weights: the activations are rotated into the weights' basis but stay
# bfloat16, and the GEMM reads the int8 rows directly (M5 TensorOps bfloat x int8 -> float), so nothing is dequantized
# to memory. Used per projection class (Weights.w8a16): the text encoder runs its MLP down_proj this way by default
# (text.text_w8a16), where int8 activations lose the most (encoder error against BF16 1.2-1.5% -> 0.4-0.9%); the other
# classes stay W8A8, which is faster there at long prompts.
_ROTATE = """
    const uint t = thread_position_in_threadgroup.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint g = threadgroup_position_in_grid.x;
    const uint K = x_shape[1];
    threadgroup float buf[256];
    float a = float(x[(size_t)row * K + g * 256 + t]);
    for (uint s = 1; s < 256; s *= 4) {
        buf[t] = a;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint k = (t / s) % 4, base = t - k * s;
        a = buf[base] + buf[base + s] + buf[base + 2 * s] + buf[base + 3 * s] - 2.0f * buf[base + (3 - k) * s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    xr[(size_t)row * K + g * 256 + t] = T(a * 0.0625f);
"""

_W8A16 = """
const int K = x_shape[1];
const int Mx = x_shape[0];
const int M = rows[0];
const int N = wq_shape[0];
const int m0 = int(threadgroup_position_in_grid.y) * BM;
const int n_lo = int(threadgroup_position_in_grid.x) * BN;
const int n0 = min(n_lo, N - BN);
constexpr auto desc = matmul2d_descriptor(BM, BN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc, execution_simdgroups<4>> op;
tensor<device T, dextents<int, 2>, tensor_inline> A((device T*)x, dextents<int, 2>(K, Mx));
tensor<device int8_t, dextents<int, 2>, tensor_inline> B((device int8_t*)wq, dextents<int, 2>(K, N));
auto a0 = A.template slice<BK, BM>(0, m0);
auto b0 = B.template slice<BK, BN>(0, n0);
auto cT = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
for (uint16_t i = 0; i < cT.get_capacity(); ++i) cT[i] = 0;
for (int k0 = 0; k0 < K; k0 += BK) {
  auto a = A.template slice<BK, BM>(k0, m0);
  auto b = B.template slice<BK, BN>(k0, n0);
  op.run(a, b, cT);
}
for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
  auto ids = cT.get_multidimensional_index(i);
  const int n = n0 + ids[0], m = m0 + ids[1];
  if (m < M && n >= n_lo && n < N) y[(size_t)m * N + n] = T(cT[i] * ws[n]);
}
"""


@cache
def _rotate_kernel():
    return mx.fast.metal_kernel(name="convrot_rotate", input_names=["x"], output_names=["xr"], source=_ROTATE)


@cache
def _w8a16_kernel():
    header = (Path(__file__).resolve().parent / "kernels" / "mpp_tensor_ops.metal").read_text()
    return mx.fast.metal_kernel(name="convrot_w8a16", input_names=["x", "wq", "ws", "rows"], output_names=["y"],
                                header=header, source=_W8A16)


def rotate(x, pad_rows=1):
    """x H256 (block-diagonal along the last axis) in x's dtype, rows allocated up to a multiple of pad_rows (the
    extra rows are left unset: they only feed GEMM output rows that are never stored)."""
    rows, k = x.shape
    (xr,) = _rotate_kernel()(
        inputs=[x], template=[("T", x.dtype)], grid=(k, rows, 1), threadgroup=(256, 1, 1),
        output_shapes=[(-(-rows // pad_rows) * pad_rows, k)], output_dtypes=[x.dtype])
    return xr


def w8a16(x, weight, scale):
    """x @ W.T for a ConvRot int8 weight with bfloat16 (x's dtype) activations: rotate x, then a TensorOps GEMM that
    reads the int8 weight rows as they are and applies the float32 row scales to the float sums. H3_W8A16_TILES=bm,bn,bk
    overrides the tiles (A/B). Without TensorOps (or narrower than one tile) the weight is dequantized in float32."""
    rows, n = x.shape[0], weight.shape[0]
    # (ain-soph fork, 2026-11-24) MLX's own W8A16 (mx.fast.convrot_matmul_w8a16, ain-soph/mlx branch convrot-w8a16):
    # the int8 rows go into MLX's NAX GEMM. split_k=1 keeps one pass over K, so the float sums run in K order as in
    # the kernel below: the same bytes. (Splitting K is faster in isolation but changes the bytes, the encoder's
    # attention-sink row included, and its partial sums cost memory.) Used from 257 rows (the kernel below is as fast
    # or faster for fewer). down_proj, 2026-11-24, ms: 1047 rows 4.35 -> 3.94, 4152 21.0 -> 16.3, 16595 107 -> 63;
    # in the encoder (real activations) e16 down_proj 3.95 -> 3.35 s, text stage -4 %. H3_W8A16_NATIVE=0 (or an
    # H3_W8A16_TILES override) keeps the kernel below.
    if (backend.get() == "w8a8" and rows > 256 and hasattr(mx.fast, "convrot_matmul_w8a16")
            and not os.environ.get("H3_W8A16_TILES") and os.environ.get("H3_W8A16_NATIVE", "1") != "0"):
        return mx.fast.convrot_matmul_w8a16(x, weight, scale.reshape(-1), split_k=1)
    # Tile sweep on the text encoder's down_proj (N 5120, K 25600) on an M5 Ultra, 2026-11-23: 32x128 is the fastest
    # up to 64 rows (32 rows: 0.37 ms against 0.80 ms for 64x128), 64x64 above (176 / 1047 / 4152 rows: 0.68 / 4.48 /
    # 23.0 ms against 1.06 / 6.86 / 27.6 ms). The float sums run over K in the same order for every tiling: same bytes.
    bm, bn, bk = (32, 128, 64) if rows <= 64 else (64, 64, 64)
    if os.environ.get("H3_W8A16_TILES"):
        bm, bn, bk = (int(v) for v in os.environ["H3_W8A16_TILES"].split(","))
    if backend.get() != "w8a8" or x.ndim != 2 or x.shape[1] % 256 or n < bn:
        xr = (x.astype(mx.float32).reshape(-1, 256) @ H256).reshape(x.shape)
        return ((xr @ weight.astype(mx.float32).T) * scale.reshape(1, -1)).astype(x.dtype)
    xr = rotate(x, pad_rows=bm)
    (y,) = _w8a16_kernel()(
        inputs=[xr, weight, scale.reshape(-1), mx.array([rows], dtype=mx.int32)],
        template=[("BM", bm), ("BN", bn), ("BK", bk), ("T", x.dtype)],
        grid=((n + bn - 1) // bn * 128, (rows + bm - 1) // bm, 1), threadgroup=(128, 1, 1),
        output_shapes=[(rows, n)], output_dtypes=[x.dtype])
    return y


def _read_owned(fd, length, offset):
    """Read a tensor into owned storage using complete pages up to EOF."""
    if not length:
        return np.empty(0, dtype=np.uint8)
    page = os.sysconf("SC_PAGESIZE")
    skip = offset % page
    aligned_length = (skip + length + page - 1) // page * page
    allocation = np.empty(aligned_length + page - 1, dtype=np.uint8)
    start = (-allocation.ctypes.data) % page
    data = allocation[start:start + aligned_length]
    done = 0
    while done < skip + length:
        end = min(done + 8 * 1024**2, aligned_length)
        got = os.preadv(fd, [data[done:end]], offset - skip + done)
        if not got:
            raise EOFError("Unexpected end of weight file")
        done += got
    return data[skip:skip + length]


class Weights:
    """Read individual tensors so a streaming stage can release each block."""

    def __init__(self, path, resident=False):
        self.file = open(path, "rb")
        (length,) = struct.unpack("<Q", self.file.read(8))
        self.start = 8 + length
        self.header = json.loads(self.file.read(length))
        self.direct_reads = (m5_resident.get() or legacy_int8_reads.get()) and hasattr(os, "preadv")
        self.resident = resident
        self.cache = {}
        self.int8_calls = 0
        self.w8a16 = frozenset()  # projection names (q, k, v, o, gate, up, down) run as W8A16; see w8a16()
        self.validated_projections = set()
        # Opt-in knobs. None of them changes a single weight byte or the op graph.
        self.pool = ThreadPoolExecutor(1) if flag("H3_PREFETCH") else None
        self.pending = {}
        self.stats = dict(read_s=0.0, read_bytes=0, convert_s=0.0, wait_s=0.0)
        self.lora = None  # optional second Weights holding lora_A / lora_B / alpha per projection
        # H3_ZERO_COPY=1 on a page-aligned file: each tensor is a view of one read-only mapping (MLX's
        # SafetensorsFile, ain-soph/mlx `serving`); Metal reads the page cache directly (no copy, no prefetch
        # thread), and the OS may drop the pages when memory is needed. None when MLX cannot map the file.
        self.mapped = None
        if flag("H3_ZERO_COPY") and page_aligned(self.header, self.start):
            mapped = mx.SafetensorsFile(path, zero_copy=True)
            self.mapped = mapped if mapped.mapped else None
        self.alphas = {}

    def delta(self, prefix, x):
        """LoRA side path for one projection: x @ A.T @ B.T * alpha / rank, or None.

        The int8 ConvRot weights are rotated and quantized, so a LoRA cannot be merged into
        them; it runs beside the quantized GEMM on the unrotated activations instead.
        """
        key = "diffusion_model." + prefix
        if key + ".lora_A.weight" not in self.header:
            return None
        a, b = self[key + ".lora_A.weight"], self[key + ".lora_B.weight"]
        if key not in self.alphas:
            alpha = self[key + ".alpha"].item() if key + ".alpha" in self.header else a.shape[0]
            self.alphas[key] = float(alpha) / a.shape[0]
        low = x.astype(mx.float32) @ a.astype(mx.float32).T
        return (low @ b.astype(mx.float32).T) * self.alphas[key]
        if flag("H3_NOCACHE"):
            fcntl.fcntl(self.file.fileno(), fcntl.F_NOCACHE, 1)

    def release(self):
        self.cache.clear()

    def ahead(self, group, keep=()):
        """Start reading one block's tensors on a worker thread (raw bytes only)."""
        if self.mapped is not None:
            return
        for name in [g for g in self.pending if g != group and g not in keep]:
            del self.pending[name]
        if not self.pool or group in self.pending:
            return
        spans = [
            (key, *spec["data_offsets"])
            for key, spec in self.header.items()
            if key.startswith(group + ".")
            and key not in self.cache
            and "adaln_proj" not in key
            and not key.endswith((".comfy_quant", ".config"))
        ]
        fd, start = self.file.fileno(), self.start

        def job():
            t = time.perf_counter()
            read = _read_owned if self.direct_reads else os.pread
            out = {key: read(fd, b - a, start + a) for key, a, b in spans}
            return out, time.perf_counter() - t

        self.pending[group] = self.pool.submit(job)

    def _read(self, key, row=None):
        spec = self.header[key]
        a, b = spec["data_offsets"]
        shape = spec["shape"]
        if self.mapped is not None and row is None and b > a:
            t = time.perf_counter()
            data = self.mapped[key]
            self.stats["convert_s"] += time.perf_counter() - t
            return data
        if row is not None:
            stride = (b - a) // shape[0]
            a += int(row) * stride
            b = a + stride
            shape = shape[1:]
        buf = None
        t = time.perf_counter()
        future = None
        if self.pending and row is None:
            future = next((f for group, f in self.pending.items() if key.startswith(group + ".")), None)
        if future:
            fetched, seconds = future.result()
            buf = fetched.pop(key, None)
            if "counted" not in fetched:
                fetched["counted"] = True
                self.stats["read_s"] += seconds
            self.stats["wait_s"] += time.perf_counter() - t
        if buf is None:
            read = _read_owned if self.direct_reads and b - a >= 65536 else os.pread
            buf = read(self.file.fileno(), b - a, self.start + a)
            self.stats["read_s"] += time.perf_counter() - t
        self.stats["read_bytes"] += len(buf)
        t = time.perf_counter()
        shaped = np.frombuffer(buf, DTYPES[spec["dtype"]]).reshape(shape)
        data = mx.from_dlpack(shaped) if isinstance(buf, np.ndarray) else mx.array(shaped)
        if spec["dtype"] == "BF16":
            data = data.view(mx.bfloat16)
        self.stats["convert_s"] += time.perf_counter() - t
        return data

    def __getitem__(self, key):
        if key in self.cache:
            return self.cache[key]
        x = self._read(key)
        if self.resident:
            self.cache[key] = x
        return x

    def linear(self, prefix, x):
        w = self[prefix + ".weight"]
        if w.dtype == mx.int8:
            if prefix not in self.validated_projections:
                a, b = self.header[prefix + ".comfy_quant"]["data_offsets"]
                metadata = json.loads(
                    os.pread(self.file.fileno(), b - a, self.start + a)
                )
                if metadata != {
                    "format": "int8_tensorwise",
                    "convrot": True,
                    "convrot_groupsize": 256,
                }:
                    raise ValueError(f"Unsupported quantization: {prefix}: {metadata}")
                self.validated_projections.add(prefix)
            self.int8_calls += 1
            if prefix.rsplit(".", 1)[-1].removesuffix("_proj") in self.w8a16:
                y = w8a16(x, w, self[prefix + ".weight_scale"])
            else:
                y = w8a8(x, w, self[prefix + ".weight_scale"])
        else:
            y = x @ w.astype(x.dtype).T
        if self.lora is not None:
            extra = self.lora.delta(prefix, x)
            if extra is not None:
                y = y + extra.astype(y.dtype)
        if prefix + ".bias" in self.header:
            y = y + self[prefix + ".bias"].astype(y.dtype)
        return y


def norm(x, weight=None, eps=1e-5):
    return mx.fast.rms_norm(x, None if weight is None else weight.astype(x.dtype), eps)


def heads(x, head_dim):
    return x.reshape(1, x.shape[0], -1, head_dim).transpose(0, 2, 1, 3)


def attention(q, k, v, mask=None):
    out = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=q.shape[-1] ** -0.5, mask=mask
    )
    return out.transpose(0, 2, 1, 3).reshape(q.shape[2], -1)


def rope(x, angles):
    n = angles.shape[-1]
    c, s = mx.cos(angles).astype(x.dtype), mx.sin(angles).astype(x.dtype)
    a, b = x[..., :n], x[..., n : 2 * n]
    return mx.concatenate([a * c - b * s, a * s + b * c, x[..., 2 * n :]], axis=-1)


def swiglu(x):
    gate, up = mx.split(x, 2, axis=-1)
    return (gate * mx.sigmoid(gate)) * up
