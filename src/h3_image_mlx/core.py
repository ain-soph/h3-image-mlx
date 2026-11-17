"""H3 Image inference primitives. Metal GEMM from ddalcu/mlx-serve @ 9191fbdf."""

import fcntl
import json
import mmap
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from functools import cache
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent
KERNELS = ROOT / "kernels"
backend = ContextVar("h3_backend", default="portable")
legacy_int8_reads = ContextVar("h3_legacy_int8_reads", default=False)
m5_resident = ContextVar("h3_m5_resident", default=False)


@cache
def tensorops_kernel():
    return mx.fast.metal_kernel(
        name="msv_w8a8_gemm",
        input_names=["xq", "wq", "xs", "ws"],
        output_names=["y"],
        header=(KERNELS / "mpp_tensor_ops.metal").read_text(),
        source=(KERNELS / "w8a8_gemm_body.metal").read_text()
        + "\n"
        + (KERNELS / "w8a8_epi_plain.metal").read_text(),
    )


H4 = np.ones((4, 4), dtype=np.float32)
np.fill_diagonal(H4[:, ::-1], -1)
H256 = mx.array(np.kron(np.kron(H4, H4), np.kron(H4, H4)) / 16)
DTYPES = {"I8": "i1", "U8": "u1", "F16": "<f2", "BF16": "<u2", "F32": "<f4", "I64": "<i8"}
# Zero-copy weights (ain-soph fork): see align.py.
PAGE = 16384
MX_DTYPES = {"I8": mx.int8, "U8": mx.uint8, "F16": mx.float16, "BF16": mx.bfloat16, "F32": mx.float32,
             "I64": mx.int64}


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


# (ain-soph fork) One threadgroup per row, one thread per position in a 256-wide group. H256 = H4 (x) H4 (x) H4 (x) H4
# with H4 = ones - 2 * antidiagonal, so each base-4 digit is a 4-point butterfly y_k = (x_0 + x_1 + x_2 + x_3) - 2 x_(3-k).
# The row is read once, kept in registers and written once as int8: a memory-bound pass instead of a 256x256 float32
# matmul per group. Same kernel as ain-soph/mflux serving (w8a8.py).
_ROTATE_QUANTIZE = """
    const uint t = thread_position_in_threadgroup.x;
    const ushort lane = thread_index_in_simdgroup;
    const uint row = threadgroup_position_in_grid.y;
    const uint K = x_shape[1];
    const uint groups = K / 256;
    threadgroup float buf[256];
    threadgroup float red[8];
    float v[MAX_GROUPS];
    float amax = 0.0f;
    device const T* xr = x + (size_t)row * K;
    for (uint g = 0; g < groups; ++g) {
        float a = float(xr[g * 256 + t]);
        for (uint s = 1; s < 256; s *= 4) {
            buf[t] = a;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint k = (t / s) % 4, base = t - k * s;
            a = buf[base] + buf[base + s] + buf[base + 2 * s] + buf[base + 3 * s] - 2.0f * buf[base + (3 - k) * s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        a *= 0.0625f;
        v[g] = a;
        amax = max(amax, fabs(a));
    }
    amax = simd_max(amax);
    if (lane == 0) red[simdgroup_index_in_threadgroup] = amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    amax = red[0];
    for (uint i = 1; i < 8; ++i) amax = max(amax, red[i]);
    const float scale = max(amax / 127.0f, 1e-20f);
    device int8_t* qr = xq + (size_t)row * K;
    for (uint g = 0; g < groups; ++g) qr[g * 256 + t] = int8_t(clamp(rint(v[g] / scale), -127.0f, 127.0f));
    if (t == 0) xs[row] = scale;
"""


@cache
def _rotate_quantize_kernel():
    return mx.fast.metal_kernel(
        name="convrot_rotate_quantize",
        input_names=["x"],
        output_names=["xq", "xs"],
        source=_ROTATE_QUANTIZE,
    )


def rotate_quantize(x):
    """Per-row int8 quantization of x H256 (block-diagonal along the last axis) with float32 scales.
    The fast kernel serves the M5 TensorOps path (backend w8a8); the portable path and H3_ROTATE=reference keep the
    float32-matmul formulation."""
    if backend.get() != "w8a8" or os.environ.get("H3_ROTATE") == "reference" or x.ndim != 2 or x.shape[1] % 256:
        return rotate_quantize_reference(x)
    rows, k = x.shape
    groups = k // 256
    xq, xs = _rotate_quantize_kernel()(
        inputs=[x],
        template=[("T", x.dtype), ("MAX_GROUPS", 1 << (groups - 1).bit_length())],
        grid=(256, rows, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, k), (rows,)],
        output_dtypes=[mx.int8, mx.float32],
    )
    return xq, xs


def float_activations():
    """(ain-soph fork, 2026-11-17) Projections whose int8 ConvRot weights run on float activations (W8A16) instead of
    int8 ones. Default: the text encoder's down_proj. Measured on a 1024^2 prompt (evals/int8-principle/u5), int8
    activations there carry almost all of the text encoder's activation-quantization error (embedding rel. error 0.0105
    of 0.0106) while the projection is a small share of a one-off stage (and slower than bfloat16 at >= 2048 prompt
    tokens). H3_FLOAT_ACT=<comma-separated projection names>, or "none" for int8 activations everywhere."""
    names = os.environ.get("H3_FLOAT_ACT", "down_proj")
    return set() if names == "none" else set(names.split(","))


def w8a16(x, weight, scale):
    """x @ dequantized int8 ConvRot weight, activations in their own float dtype: rotate x (float32), dequantize the
    weight rows (float32, then the activation dtype) and multiply."""
    shape = x.shape
    xr = (x.astype(mx.float32).reshape(-1, 256) @ H256).reshape(shape).astype(x.dtype)
    w = (weight.astype(mx.float32) * scale.reshape(-1, 1).astype(mx.float32)).astype(x.dtype)
    return xr @ w.T


def w8a8(x, weight, scale):
    rows = x.shape[0]
    n = weight.shape[0]
    xq, xs = rotate_quantize(x)
    if backend.get() == "portable":
        from ._portable import portable_matmul
        return portable_matmul(xq, weight, xs, scale.reshape(-1), x.dtype)
    if m5_resident.get():
        bm, bn = (32, 64) if rows <= 32 else (64, 128)
        bk = 256
    else:
        # (ain-soph fork) Tile sweep at the DiT shapes on an M5 Ultra (opt/k_bench.py gemm, 2026-11-17): 64x128 output
        # tiles are the fastest or within 3% for every projection; the old 128x64 below 8192 rows ran the 1046-row QKV
        # GEMM at 82 TOPS (2.94 ms) against 143 TOPS (1.69 ms). K steps of 32 win from ~2000 rows up. The int32 sums
        # are exact, so every tiling gives the same bytes. H3_GEMM_TILES=bm,bn,bk overrides (A/B).
        bm, bn, bk = 64, 128, (64 if rows < 2048 else 32)
        if os.environ.get("H3_GEMM_TILES"):
            bm, bn, bk = (int(v) for v in os.environ["H3_GEMM_TILES"].split(","))
    (y,) = tensorops_kernel()(
        inputs=[xq, weight, xs, scale.reshape(-1)],
        template=[("BM", bm), ("BN", bn), ("BK", bk), ("T", x.dtype)],
        grid=((n + bn - 1) // bn * 128, (rows + bm - 1) // bm, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[x.dtype],
    )
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
        self.validated_projections = set()
        # Opt-in knobs. None of them changes a single weight byte or the op graph.
        self.pool = ThreadPoolExecutor(1) if flag("H3_PREFETCH") else None
        self.pending = {}
        self.stats = dict(read_s=0.0, read_bytes=0, convert_s=0.0, wait_s=0.0)
        self.lora = None  # optional second Weights holding lora_A / lora_B / alpha per projection
        # H3_ZERO_COPY=1 on a page-aligned file: tensors are views of one read-only mapping; Metal reads the page
        # cache directly (no copy, no prefetch thread), and the OS may drop the pages when memory is needed.
        self.mapped = None
        if flag("H3_ZERO_COPY") and page_aligned(self.header, self.start):
            self.mapped = mmap.mmap(self.file.fileno(), 0, prot=mmap.PROT_READ)
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
            padded = (b - a + PAGE - 1) // PAGE * PAGE
            raw = np.frombuffer(self.mapped, np.uint8, padded, self.start + a)
            data = mx.from_dlpack(raw)[: b - a]
            if spec["dtype"] != "U8":
                data = data.view(MX_DTYPES[spec["dtype"]])
            self.stats["convert_s"] += time.perf_counter() - t
            return data.reshape(shape)
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
            if prefix.rsplit(".", 1)[-1] in float_activations():
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
