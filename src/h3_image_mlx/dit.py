"""Text-to-image forward on the six-step H3 Image grid."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import cache

import itertools
import json
import os
import re
import time

import numpy as np
from .core import Weights, attention, backend, flag, heads, m5_resident, mx, norm, rope, swiglu

# (ain-soph fork) H3_FUSION: comma-separated subset of {modulate, swiglu, rope} for the DiT blocks; unset = all,
# empty = none (the composed ops, for A/B). Each fused form performs the same bfloat16 operations in the same order,
# so the result is the same bytes; only the number of passes over the activations changes.
#   modulate: AdaLN shift/scale and the gated residual over the [text | ... | image] segments in one compiled
#             kernel each (a per-row select between the table rows) instead of per-segment ops + concatenate + add.
#   swiglu:   gate * sigmoid(gate) * up in one compiled kernel.
#   rope:     the rotary embedding of q and k compiled (cos/sin and the four products fused).
_FUSION_ALL = ("modulate", "swiglu", "rope")


def fused(name):
    value = os.environ.get("H3_FUSION")
    if value is None:
        return True
    return name in {part.strip() for part in value.split(",")}


# (ain-soph fork) Row slicing for long edit sequences. A sequence of more than H3_ROW_SLICE_FROM rows (default
# 35000) runs its row-wise work (the QKV projection with Q/K norm and RoPE, the output projection and the MLP) in
# slices of H3_ROW_SLICE rows (default 16384, FreeVideo's h3_dit.ROW_SLICE; 0 = never slice); attention still sees
# the whole sequence. Every slice is the same arithmetic on the same rows (int8 activations carry one scale per row,
# the W8A8 GEMM sums in int32), so the bytes do not change, and the slices' temporaries (the MLP's 28672-wide fc1
# output above all) stop growing with the sequence. Each extra slice streams the block's weights once more (about
# 5.5 s per image on an M5 Ultra), so only sequences that would not fit a 10 GB budget whole are sliced: whole, the
# DiT stage peaks at 9.7 GB for 1024^2 + 16 references (34003 rows) and 10.6 GB for 2048^2 + 16 references (37075
# rows), sliced 8.3 GB (+5% time).
ROW_SLICE = int(os.environ.get("H3_ROW_SLICE") or 16384)
ROW_SLICE_FROM = int(os.environ.get("H3_ROW_SLICE_FROM") or 35000)


def sliced(rows):
    """Whether a sequence of this many rows runs in row slices."""
    return ROW_SLICE > 0 and rows > max(ROW_SLICE, ROW_SLICE_FROM)


def _slices(rows):
    if not sliced(rows):
        return None
    return [slice(start, min(start + ROW_SLICE, rows)) for start in range(0, rows, ROW_SLICE)]


@contextmanager
def _held(w, *prefixes):
    """Read each projection's tensors once for all slices (a streamed Weights without a mapping reads per access)."""
    keys = [prefix + suffix for prefix in prefixes for suffix in (".weight", ".weight_scale", ".bias")]
    keys = [key for key in keys if key in w.header and key not in w.cache]
    for key in keys:
        w.cache[key] = w[key]
    try:
        yield
    finally:
        for key in keys:
            w.cache.pop(key, None)


def by_rows(w, prefixes, x, fn):
    """fn(x) on ROW_SLICE-row slices of x, concatenated; fn(x) itself for a short sequence."""
    parts = _slices(x.shape[0])
    if parts is None:
        return fn(x)
    with _held(w, *prefixes):
        return mx.concatenate([fn(x[part]) for part in parts])


swiglu_fused = mx.compile(swiglu)
rope_fused = mx.compile(rope)


def _segment_rows(segments):
    """(rows, 1) int32 table-row index of every token, and the distinct table rows in use."""
    ids = np.concatenate([np.full(stop - start, row, np.int32) for start, stop, row in segments])
    return mx.array(ids[:, None]), tuple(sorted({row for _, _, row in segments}))


def _select(ids, table, rows):
    value = table[rows[-1]]
    for row in reversed(rows[:-1]):
        value = mx.where(ids == row, table[row], value)
    return value


@cache
def _shift_kernel(rows):
    return mx.compile(lambda x, ids, scale, shift: x * (1 + _select(ids, scale, rows)) + _select(ids, shift, rows))


@cache
def _gated_add_kernel(rows):
    return mx.compile(lambda x, y, ids, gate: x + y * _select(ids, gate, rows))


def shifted_fused(x, scale, shift, seg):
    """shifted() in one pass; seg = _segment_rows(segments)."""
    ids, rows = seg
    return _shift_kernel(rows)(x, ids, scale.astype(x.dtype), shift.astype(x.dtype))


def gated_add(x, y, gate, seg):
    """x + modulate(y, gate, segments) in one pass."""
    ids, rows = seg
    return _gated_add_kernel(rows)(x, y, ids, gate.astype(y.dtype))

# P-H3-Image (PrunaAI) ships no sampling metadata; this is the checkpoint's own 8-step grid.
PH3_SIGMAS = [1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0]


def qkv_project(w, prefix, x, angles=None, trigonometry=None, q_weight=None, k_weight=None):
    qkv = w.linear(prefix + ".qkv_proj", x)
    if q_weight is None:
        q_weight, k_weight = w[prefix + ".q_norm.weight"], w[prefix + ".k_norm.weight"]
    if trigonometry is not None and qkv.dtype == trigonometry[0].dtype:
        from .qk_rope import normalize_rotate
        q, k = normalize_rotate(qkv, q_weight, k_weight, trigonometry)
        v = heads(qkv[:, qkv.shape[-1] * 2 // 3:], 128)
    else:
        q, k, v = [heads(t, 128) for t in mx.split(qkv, 3, -1)]
        q, k = norm(q, q_weight), norm(k, k_weight)
        if angles is not None:
            r = rope_fused if fused("rope") else rope
            q, k = r(q, angles), r(k, angles)
    return q, k, v


def qkv_attention(w, prefix, x, angles=None, trigonometry=None):
    parts = _slices(x.shape[0])
    if parts is None:
        q, k, v = qkv_project(w, prefix, x, angles, trigonometry)
    else:
        q_weight, k_weight = w[prefix + ".q_norm.weight"], w[prefix + ".k_norm.weight"]
        with _held(w, prefix + ".qkv_proj"):
            sliced = [qkv_project(w, prefix, x[part], None if angles is None else angles[part],
                                  None if trigonometry is None else tuple(t[part] for t in trigonometry),
                                  q_weight, k_weight) for part in parts]
        q, k, v = (mx.concatenate([piece[i] for piece in sliced], axis=2) for i in range(3))
    out = attention(q, k, v)
    return by_rows(w, (prefix + ".out_proj",), out, lambda part: w.linear(prefix + ".out_proj", part))


def mlp(w, prefix, x):
    act = swiglu_fused if fused("swiglu") else swiglu

    def run(part):
        return w.linear(prefix + ".fc2", act(w.linear(prefix + ".fc1", part)))

    return by_rows(w, (prefix + ".fc1", prefix + ".fc2"), x, run)


def prepare_text(w, embedding):
    x = w.linear("condition_proj", embedding.astype(mx.bfloat16))
    for i in range(2):
        p = f"token_refiner.blocks.{i}"
        x = x + qkv_attention(w, p + ".attn", norm(x, w[p + ".norm1.weight"]))
        x = x + mlp(w, p + ".mlp", norm(x, w[p + ".norm2.weight"]))
    return norm(x, w["token_refiner.final_norm.weight"])


def image_positions(h, width, cursor):
    area = (h * width) ** 0.5

    def axis(n):
        return (np.arange(n) * (n / area / n) + (1 - n / area) / 2) * 32

    yy, xx = np.meshgrid(axis(h), axis(width), indexing="ij")
    return np.stack([np.full(h * width, cursor), yy.flatten(), xx.flatten()], axis=-1)


def angles_for(w, ntext, grids, audio=0):
    """Rotary angles for the packed sequence [text | reference images | target audio | target image]."""
    text = np.zeros((ntext, 3), np.float32)
    text[:, 0] = np.arange(ntext)
    positions = [text] + [
        image_positions(h, width, ntext + i) for i, (h, width) in enumerate(grids)
    ]
    if audio:
        # Stereo rows, channel-major: time advances per audio latent, width is pinned to the
        # target image's first and last column, height stays 0.
        image = positions[-1]
        sound = np.zeros((2 * audio, 3))
        sound[:, 0] = np.tile(image[0, 0] + np.arange(audio), 2)
        sound[:audio, 2], sound[audio:, 2] = image[0, 2], image[-1, 2]
        positions.insert(-1, sound)
    pos = mx.array(np.concatenate(positions).astype(np.float32))
    return (pos[:, :, None] * w["rope.inv_freq"]).reshape(pos.shape[0], -1)


def curve_coordinates(w, sigmas):
    t = np.clip(1 - np.asarray(sigmas, dtype=np.float32), 0, 1)
    table = np.asarray(w["adaln_t_table"])
    pos = t * (table.shape[0] - 1)
    idx = np.minimum(pos.astype(np.int32), len(table) - 2)
    return (
        table[idx]
        + (table[idx + 1] - table[idx]) * (pos - idx.astype(np.float32))[:, None]
    )


def precompute(w, prefix, coords, modalities, parts):
    p = prefix + ".adaln_proj.linear"
    # Eight basis coordinates per time. CPU float32 preserves the small dot product.
    weights = np.asarray(w[p + ".weight"].astype(mx.float32))
    bias = np.asarray(w[p + ".bias"].astype(mx.float32))
    mod = coords @ weights.T + bias
    return mx.array(mod.reshape(len(coords), modalities, parts, -1))


def modulate(x, table, segments, scale=False):
    buffers = []
    for start, stop, row in segments:
        value = table[row].astype(x.dtype)
        buffers.append(x[start:stop] * (1 + value if scale else value))
    return mx.concatenate(buffers)


def shifted(x, scale, shift, segments):
    return mx.concatenate(
        [
            x[start:stop] * (1 + scale[row].astype(x.dtype))
            + shift[row].astype(x.dtype)
            for start, stop, row in segments
        ]
    )


def head_mix(grid, sigma, sigma_next):
    """Which PDD heads a step from sigma to sigma_next uses, as (head index, weight) offsets on head 0.

    Mirrors ComfyUI's _pdd_head: a step that spans several intervals of the checkpoint's own
    sigma grid takes the dt-weighted mean of their heads. On the stock grid this is one head.
    """
    n = len(grid) - 1
    start, stop = (min(range(n + 1), key=lambda k: abs(grid[k] - s)) for s in (sigma, sigma_next))
    start = min(start, n - 1)
    stop = max(stop, start + 1)
    dt = [grid[k] - grid[k + 1] for k in range(start, stop)]
    return [(k, dt[k - start] / sum(dt)) for k in range(max(start, 1), stop)]


def patchify(latent):
    c, h, width = latent.shape
    return (
        latent.reshape(c, h // 2, 2, width // 2, 2)
        .transpose(1, 3, 0, 2, 4)
        .reshape(-1, c * 4)
    )


def denoise(
    path, text, width, height, seed, resident=True, refs=(), tags=None, progress=None,
    precision=None,
):
    if min(width, height) < 32 or width % 32 or height % 32:
        raise ValueError("Image dimensions must be positive multiples of 32")
    began = time.perf_counter()
    w = Weights(path, resident)
    if precision == "bf16":
        for key, spec in w.header.items():
            if key.endswith(".weight") and spec["dtype"] not in ("BF16", "F16", "F32"):
                if w.pool:
                    w.pool.shutdown()
                w.file.close()
                raise ValueError(f"BF16 requires floating-point weights; {path}: {key} is {spec['dtype']}. Check H3_DIT.")
    # H3_ASYNC: queue each block on the GPU and read the next block's weights meanwhile.
    portable_stream = not resident and precision != "bf16" and backend.get() == "portable"
    native = resident and m5_resident.get()
    if (portable_stream or native) and "H3_PREFETCH" not in os.environ and w.pool is None:
        w.pool = ThreadPoolExecutor(1)
    pipelined = bool(flag("H3_ASYNC")) if "H3_ASYNC" in os.environ else (portable_stream or native)
    profile = dict(steps=[], eval_s=0.0)
    meta = w.header.get("__metadata__") or {}
    grid = sigmas = json.loads(meta["config"])["transformer"]["pdd_sigmas"] if "config" in meta else PH3_SIGMAS
    # A checkpoint with an audio stream (P-H3-Image) is the joint model: a still image spans two audio latents.
    audio = 2 if "audio_patch_proj.weight" in w.header else 0
    # Opt-in few-step mode: H3_LORA=<file> beside the quantized weights, H3_STEPS on a shifted uniform grid.
    if os.environ.get("H3_LORA"):
        w.lora = Weights(os.environ["H3_LORA"])
    steps = int(flag("H3_STEPS"))
    if steps:
        shift = flag("H3_SHIFT") or 12.0
        sigmas = [shift * t / (1 + (shift - 1) * t) for t in (1 - k / steps for k in range(steps + 1))]
    if os.environ.get("H3_SIGMAS"):  # explicit grid, e.g. a subset of the checkpoint's own
        sigmas = [float(v) for v in os.environ["H3_SIGMAS"].split(",")]
        steps = len(sigmas) - 1
    coords = curve_coordinates(w, sigmas[:-1])
    if refs:
        coords = np.concatenate([coords, curve_coordinates(w, [0.001])])
    blocks = [f"blocks.{i}" for i in range(50)]
    mods = [precompute(w, b, coords, 3, 6) for b in blocks]
    final_mod = precompute(w, "final_layer", coords, 1, 2)
    text = prepare_text(w, text)
    mx.eval(text, final_mod, *mods)
    # The text refiner and compressed AdaLN projections are used only during setup.
    w.release()
    mx.clear_cache()
    rows, cols = height // 32, width // 32
    ntext = text.shape[0]
    grids = [(z.shape[1] // 2, z.shape[2] // 2) for z in refs]
    angles = angles_for(w, ntext, [*grids, (rows, cols)], audio)
    fixed = [text]
    text_tags = np.asarray(tags) if tags is not None else np.ones(ntext, int)
    splits = np.r_[0, np.flatnonzero(np.diff(text_tags)) + 1, ntext]
    segments = [(a, b, int(text_tags[a])) for a, b in itertools.pairwise(splits)]
    target_start = ntext
    for ref in refs:
        r = patchify(ref)
        aug_noise = np.random.default_rng(seed).standard_normal(
            r.shape, dtype=np.float32
        )
        r = r * 0.999 + mx.array(aug_noise) * 0.001
        fixed.append(w.linear("video_patch_proj", r).astype(text.dtype))
        segments.append((target_start, target_start + r.shape[0], 3))
        target_start += r.shape[0]
    audio_start = target_start
    if audio:
        segments.append((audio_start, audio_start + 2 * audio, 2))
        target_start += 2 * audio
    segments.append((target_start, target_start + rows * cols, 0))
    seg = _segment_rows(segments) if fused("modulate") else None
    prefix = mx.concatenate(fixed)
    trigonometry = None
    if (
        os.environ.get("H3_QK_NORM_ROPE") != "0"
        and not resident
        and backend.get() == "w8a8"
        and fused("rope")
        and text.dtype == mx.bfloat16
        and angles.shape[0] >= 2048
        and 0 < angles.shape[-1] <= 64
        and angles.shape[-1] % 4 == 0
        and re.search(r"\bApple M5\b", mx.device_info()["device_name"])
    ):
        from .qk_rope import prepare_angles
        trigonometry = prepare_angles(angles, text.dtype)
        mx.eval(*trigonometry)
    mx.eval(prefix)
    rng = np.random.default_rng(seed)
    latent = mx.array(rng.standard_normal((24, rows * 2, cols * 2), dtype=np.float32))
    patches = (
        latent.reshape(24, rows, 2, cols, 2)
        .transpose(1, 3, 0, 2, 4)
        .reshape(rows * cols, 96)
    )
    sound = mx.array(rng.standard_normal((2 * audio, 32), dtype=np.float32)) if audio else None
    profile["setup_s"] = time.perf_counter() - began
    last = len(sigmas) - 2
    for step, (sigma, next_sigma) in enumerate(itertools.pairwise(sigmas)):
        start = time.monotonic()
        previous = None
        resident_batch = native and step > 0 and pipelined
        streams = [w.linear("audio_patch_proj", sound)] if audio else []
        streams.append(w.linear("video_patch_proj", patches))
        x = mx.concatenate([prefix, *(s.astype(text.dtype) for s in streams)])
        for i, b in enumerate(blocks):
            if (i < 49 or step < last) and not (native and step > 0):
                w.ahead(blocks[(i + 1) % 50], keep=(b,))
            tab = mods[i][step]
            if refs:
                tab = mx.concatenate([tab, mods[i][-1, :1]])
            if seg is not None:
                h = shifted_fused(norm(x, w[b + ".norm1.weight"]), tab[:, 1], tab[:, 0], seg)
                x = gated_add(x, qkv_attention(w, b + ".attn", h, angles, trigonometry), tab[:, 2], seg)
                h = shifted_fused(norm(x, w[b + ".norm2.weight"]), tab[:, 4], tab[:, 3], seg)
                x = gated_add(x, mlp(w, b + ".mlp", h), tab[:, 5], seg)
            else:
                h = shifted(norm(x, w[b + ".norm1.weight"]), tab[:, 1], tab[:, 0], segments)
                x = x + modulate(
                    qkv_attention(w, b + ".attn", h, angles, trigonometry), tab[:, 2], segments
                )
                h = shifted(norm(x, w[b + ".norm2.weight"]), tab[:, 4], tab[:, 3], segments)
                x = x + modulate(mlp(w, b + ".mlp", h), tab[:, 5], segments)
            t = time.perf_counter()
            if resident_batch:
                if i % 5 == 4:
                    mx.async_eval(x)
                    if previous is not None:
                        mx.eval(previous)
                    previous = x
            elif pipelined:
                # At most two blocks of weights are alive: one running, one queued.
                mx.async_eval(x)
                if previous is not None:
                    mx.eval(previous)
                previous = x
            else:
                mx.eval(x)
            profile["eval_s"] += time.perf_counter() - t
        shift, scale = final_mod[step, 0]
        x = (
            norm(x[audio_start:], w["final_layer.norm.weight"]).astype(mx.float32)
            * (1 + scale)
            + shift
        )
        head = w["final_layer.video_out.weight"].reshape(-1, 96, x.shape[-1])
        bias = w["final_layer.video_out.bias"].reshape(-1, 96)
        ww, bb = head[0], bias[0]
        # A PDD checkpoint stacks one head per grid interval; P-H3-Image has one plain head.
        if len(head) > 1 and steps:
            for k, part in head_mix(grid, sigma, next_sigma):
                ww, bb = ww + part * head[k], bb + part * bias[k]
        elif len(head) > 1 and step:
            ww, bb = ww + head[step], bb + bias[step]
        patches = patches + (sigma - next_sigma) * (x[2 * audio :] @ ww.T + bb)
        if audio:
            out = x[: 2 * audio] @ w["final_layer.audio_out.weight"].T + w["final_layer.audio_out.bias"]
            sound = sound + (sigma - next_sigma) * out
        mx.eval(patches, *([sound] if audio else []))
        profile["steps"].append(time.monotonic() - start)
        if progress:
            progress(step + 1, time.monotonic() - start)
    latent = (
        patches.reshape(rows, cols, 24, 2, 2)
        .transpose(2, 0, 3, 1, 4)
        .reshape(24, rows * 2, cols * 2)
    )
    if os.environ.get("H3_PROFILE"):
        profile.update(w.stats, total_s=time.perf_counter() - began, tokens=int(x.shape[0]) + audio_start,
                       peak_mlx_gb=mx.get_peak_memory() / 1e9)
        with open(os.environ["H3_PROFILE"], "w") as out:
            json.dump(profile, out)
    return latent, w.int8_calls
