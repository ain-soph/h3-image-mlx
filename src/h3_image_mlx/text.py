import os
from concurrent.futures import ThreadPoolExecutor

from .core import H256, Weights, attention, backend, flag, heads, m5_resident, mx, norm, rope
from tokenizers import Tokenizer

# P-H3-Image conditions on these hidden states of its 4B encoder (0 = embeddings, 36 = final norm output).
PH3_STATES = (13, 19, 25, 31, 36)


_W8A16_CLASSES = ("q", "k", "v", "o", "gate", "up", "down")


def text_w8a16():
    """(ain-soph fork) Text-encoder projection classes that run as W8A16 (bfloat16 activations on the int8 ConvRot
    weights, core.w8a16) instead of W8A8. H3_TE_W8A16=q,k,v,o,gate,up,down picks classes, "all" every one, "none" = W8A8
    everywhere. Default: down on the M5 W8A8 backend (the TensorOps kernel), none elsewhere (the portable backend would
    dequantize the weight in float32). The SwiGLU output that feeds down_proj has the widest activation range, so one
    int8 scale per row loses the most there: against the BF16 encoder down alone cuts the output error from 1.2-1.5% to
    0.4-0.9% (relative RMS)."""
    names = os.environ.get("H3_TE_W8A16")
    if names is None:
        names = "down" if backend.get() == "w8a8" else "none"
    names = names.strip()
    if names == "all":
        return frozenset(_W8A16_CLASSES)
    chosen = frozenset() if names in ("", "none") else frozenset(n.strip() for n in names.split(",") if n.strip())
    if unknown := chosen - set(_W8A16_CLASSES):
        raise ValueError(f"H3_TE_W8A16: unknown projection classes {sorted(unknown)}; use {','.join(_W8A16_CLASSES)}")
    return chosen


def _swiglu_ops(g, u):
    return (g * mx.sigmoid(g)) * u


_swiglu = mx.compile(_swiglu_ops)


def embed(w, key, ids):
    rows = mx.stack([w._read(key, row=i) for i in ids])
    if rows.dtype == mx.int8:
        # ConvRot int8 table: scale each row, then undo the grouped Hadamard rotation (H256 is its own inverse).
        scale = mx.stack([w._read(key + "_scale", row=i) for i in ids])
        rows = ((rows.astype(mx.float32) * scale).reshape(len(ids), -1, 256) @ H256).reshape(len(ids), -1)
        rows = rows.astype(mx.bfloat16)
    return rows


def connect(w, states):
    """P-H3-Image text connector: standardize the stacked hidden states, project, rescale, sink token first."""
    x = mx.concatenate([s.astype(mx.float32) for s in states], axis=-1)
    x = (x - w["text_connector.input_mean"]) / w["text_connector.input_std"]
    x = x @ w["text_connector.proj.weight"].T
    x = x * w["text_connector.output_std"] + w["text_connector.output_mean"]
    return mx.concatenate([w["text_connector.sink_embedding"][None], x[1:]])


def encode(path, tokenizer_path, prompt, refs=()):
    tok = Tokenizer.from_file(str(tokenizer_path))
    ids = tok.encode(prompt, add_special_tokens=False).ids[:4096]
    if not ids:
        raise ValueError("Empty prompt")
    w = Weights(path)
    w.w8a16 = text_w8a16()
    ph3 = "text_connector.proj.weight" in w.header
    lm = "model.language_model." if ph3 else "model."
    layers = sum(key.startswith(lm + "layers.") and key.endswith(".input_layernorm.weight") for key in w.header)
    states = []
    if ph3 and refs:
        raise NotImplementedError("P-H3-Image reference images are not ported yet")
    if refs:
        from .vision import prepare_prompt

        x, angles, rows, deep, tags = prepare_prompt(w, tok, prompt, refs)
    else:
        x = embed(w, lm + "embed_tokens.weight", ids)
        tags = mx.ones((len(ids),), dtype=mx.int32)
    previous = None
    native = m5_resident.get()
    # (ain-soph fork) A zero-copy encoder (mapped weights, nothing to prefetch) also queues each layer while the next
    # one is built when H3_ASYNC is set; the graph is the same.
    pipelined = (native and ("H3_ASYNC" not in os.environ or flag("H3_ASYNC"))) or (
        w.mapped is not None and flag("H3_ASYNC") and os.environ.get("H3_TE_ASYNC") != "0")
    # (ain-soph fork) The SwiGLU as one compiled kernel (same bfloat16 ops in the same order: same bytes).
    swiglu = _swiglu if os.environ.get("H3_TE_FUSION") != "0" else _swiglu_ops
    if native and "H3_PREFETCH" not in os.environ and w.pool is None:
        w.pool = ThreadPoolExecutor(1)
    for i in range(layers):
        if native and i + 1 < layers:
            w.ahead(f"{lm}layers.{i + 1}", keep=(f"{lm}layers.{i}",))
        if ph3 and i in PH3_STATES:
            states.append(x)
        prefix = f"{lm}layers.{i}."
        h = norm(x, w[prefix + "input_layernorm.weight"], 1e-6)
        att = prefix + "self_attn."

        q, k, v = (
            heads(w.linear(att + name + "_proj", h), 128) for name in ("q", "k", "v")
        )
        q, k = (
            norm(value, w[att + name + "_norm.weight"], 1e-6)
            for value, name in ((q, "q"), (k, "k"))
        )
        if refs:
            q, k = rope(q, angles), rope(k, angles)
        else:
            q, k = (
                mx.fast.rope(
                    v, 128, traditional=False, base=5_000_000, scale=1.0, offset=0
                )
                for v in (q, k)
            )
        x = x + w.linear(att + "o_proj", attention(q, k, v, mask="causal"))
        h = norm(x, w[prefix + "post_attention_layernorm.weight"], 1e-6)
        mlp = prefix + "mlp."
        g = w.linear(mlp + "gate_proj", h)
        u = w.linear(mlp + "up_proj", h)
        x = x + w.linear(mlp + "down_proj", swiglu(g, u))
        if refs and i < len(deep):
            x = x.at[rows].add(deep[i].astype(x.dtype))
        if pipelined and i + 1 < layers:
            mx.async_eval(x)
            if previous is not None:
                mx.eval(previous)
            previous = x
        else:
            mx.eval(x)
    if w.pool is not None:
        w.pool.shutdown()
    if ph3:
        states.append(norm(x, w[lm + "norm.weight"], 1e-6))
        x = connect(w, states)
        mx.eval(x)
    return x, tags, w.int8_calls
