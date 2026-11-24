"""Text-to-image and ordered multi-reference editing in isolated MLX stages."""
import hashlib
import importlib.metadata
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from functools import cache
from pathlib import Path
from typing import Sequence

from PIL import Image
from .models import check_models, manifest_for, model_paths, text_file

# (ain-soph fork) Both stage caches below key on this recipe, on a fingerprint of this package's code and of the
# installed MLX build, and on the environment switches that change the stage's numbers. A hit hands the next stage
# the very file a miss would have written, so the image is the same bytes either way.
CACHE_RECIPE = "ain-soph-stage-cache-v1"
NUMERIC_ENV = ("H3_ROTATE", "H3_GEMM_TILES", "H3_TE_W8A16", "H3_W8A16_TILES")
CACHE_MB_DEFAULT = {"H3_EMBED_CACHE_MB": 1024, "H3_REF_CACHE_MB": 256}


@cache
def code_fingerprint():
    """sha256 over this package's source and kernels plus the installed MLX version and build origin."""
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*")):
        if path.suffix in (".py", ".metal", ".json") and "__pycache__" not in path.parts:
            digest.update(str(path.relative_to(package)).encode() + b"\0" + path.read_bytes() + b"\0")
    for dist in ("mlx", "mlx-metal"):
        try:
            found = importlib.metadata.distribution(dist)
        except importlib.metadata.PackageNotFoundError:
            continue
        digest.update(f"{dist}={found.version};{found.read_text('direct_url.json') or ''}".encode())
    return digest.hexdigest()


def _numeric_env():
    return [os.environ.get(name, "") for name in NUMERIC_ENV]


def _cache_root(variable):
    root = os.environ.get(variable)
    if not root:
        return None
    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def embedding_cache(directory, prompt, refs, backend, precision="int8", dit_mode="auto"):
    """Opt-in (H3_EMBED_CACHE=dir): reuse the text-stage output for a repeated prompt + references.
    (ain-soph fork) Least recently used entries go first once the directory exceeds H3_EMBED_CACHE_MB (default 1024)."""
    root = _cache_root("H3_EMBED_CACHE")
    if not root:
        return None
    encoder = Path(os.environ.get("H3_TEXT") or text_file(directory, precision)).resolve()
    stat = encoder.stat()
    tokenizer = Path(os.environ.get("H3_TOKENIZER") or directory / model_paths(precision)[3])
    key = [prompt, backend, precision, str(encoder), stat.st_size, stat.st_mtime_ns,
           hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
           [_file_hash(p) for p in refs], CACHE_RECIPE, code_fingerprint(), _numeric_env(), dit_mode]
    name = hashlib.sha256(json.dumps(key).encode()).hexdigest()
    return root / f"{name}.safetensors"


def reference_cache(directory, refs, backend, precision="int8"):
    """(ain-soph fork) Opt-in (H3_REF_CACHE=dir): one file per reference image with its VAE latent, which depends on
    the image alone (not on its position, the prompt or the output size). Least recently used entries go first once
    the directory exceeds H3_REF_CACHE_MB (default 256). Returns one path per reference, or Nones when off."""
    root = _cache_root("H3_REF_CACHE")
    if not root:
        return [None] * len(refs)
    paths = model_paths(precision)
    vae = (Path(directory) / paths[2]).resolve()      # the references stage reads the manifest's VAE (no H3_VAE)
    stat = vae.stat()
    base = [backend, precision, str(vae), stat.st_size, stat.st_mtime_ns, _file_hash(Path(directory) / paths[4]),
            CACHE_RECIPE, code_fingerprint(), _numeric_env()]
    return [root / (hashlib.sha256(json.dumps(base + [_file_hash(p)]).encode()).hexdigest() + ".safetensors")
            for p in refs]


def _restore(cached, target):
    """Copy a cache entry into the work folder and mark it used; False when it is missing (or was just evicted)."""
    if cached is None:
        return False
    try:
        shutil.copyfile(cached, target)
        os.utime(cached)
    except FileNotFoundError:
        return False
    return True


def _store(source, cached):
    tmp = cached.with_name(f"{cached.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    shutil.copyfile(source, tmp)
    tmp.replace(cached)


def prune_cache(root, limit_mb):
    """Delete the least recently used entries (by mtime, refreshed on every hit) until the folder holds at most
    limit_mb; also leftovers of interrupted writes older than an hour."""
    entries, now = [], time.time()
    for path in root.iterdir():
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        if path.name.endswith(".tmp"):
            if now - stat.st_mtime > 3600:
                path.unlink(missing_ok=True)
        elif path.suffix == ".safetensors":
            entries.append((stat.st_mtime, stat.st_size, path))
    total = sum(size for _, size, _ in entries)
    for _, size, path in sorted(entries):
        if total <= limit_mb << 20:
            break
        path.unlink(missing_ok=True)
        total -= size


def _limit(variable):
    try:
        return int(os.environ.get(variable) or CACHE_MB_DEFAULT[variable])
    except ValueError:
        return CACHE_MB_DEFAULT[variable]


def generate(prompt: str, *, models: str | Path, width: int = 1024, height: int = 1024,
             seed: int = 42, references: Sequence[str | Path] = (),
             backend: str = "auto", dit_mode: str = "auto", precision: str = "int8") -> Image.Image:
    """Render a PIL RGB image. Refer to ordered reference files as <Picture 1>, etc."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Provide a nonempty prompt")
    if not all(type(v) is int and 256 <= v <= 4096 and v % 32 == 0 for v in (width, height)):
        raise ValueError("Width and height must be multiples of 32 from 256 through 4096")
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    if backend not in ("auto", "portable", "w8a8") or dit_mode not in ("auto", "stream", "resident"):
        raise ValueError("Unknown compute backend or DiT mode")
    manifest_for(precision)
    if precision == "bf16" and backend == "w8a8":
        raise ValueError("W8A8 requires INT8 weights; use --backend auto or portable with BF16")
    if isinstance(references, (str, Path)) or len(references) > 16:
        raise ValueError("Supply an ordered list of up to 16 reference files")
    refs = [Path(ref).expanduser().resolve() for ref in references]
    for path in refs:
        with Image.open(path) as im:
            if im.format not in ("PNG", "JPEG", "WEBP"):
                raise ValueError(f"Unsupported reference image format: {path}")
            if max(im.size) / min(im.size) * 1024**2 > 12_845_056:
                raise ValueError(f"Reference exceeds the vision encoder aspect ratio budget: {path}")
    directory = check_models(models, precision=precision)
    with tempfile.TemporaryDirectory(prefix="h3-image-mlx-") as tmp:
        work = Path(tmp)
        stages = ["text", *(["references"] if refs else []), "dit", "vae"]
        cached = embedding_cache(directory, prompt, refs, backend, precision, dit_mode)
        if _restore(cached, work / "embedding.safetensors"):
            stages.remove("text")
        ref_cached = reference_cache(directory, refs, backend, precision)
        missing = [i for i, path in enumerate(ref_cached) if not _restore(path, work / f"reference-{i}.safetensors")]
        if refs and not missing:
            stages.remove("references")
        args = dict(prompt=prompt, models=str(directory), width=width, height=height, seed=seed,
                    references=[str(p) for p in refs], backend=backend, dit_mode=dit_mode, precision=precision,
                    encode_references=missing)
        (work / "request.json").write_text(json.dumps(args))
        for stage in stages:
            proc = subprocess.run([sys.executable, "-m", "h3_image_mlx._stage", stage, tmp],
                                  capture_output=True, text=True)
            if proc.returncode:
                raise RuntimeError(f"H3 {stage} stage exited with {proc.returncode}: {proc.stderr.strip()}")
            if stage == "text" and cached:
                _store(work / "embedding.safetensors", cached)
                prune_cache(cached.parent, _limit("H3_EMBED_CACHE_MB"))
            if stage == "references" and ref_cached[0] is not None:
                for i in missing:
                    _store(work / f"reference-{i}.safetensors", ref_cached[i])
                prune_cache(ref_cached[0].parent, _limit("H3_REF_CACHE_MB"))
        with Image.open(work / "image.png") as image:
            return image.convert("RGB")
