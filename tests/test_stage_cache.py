"""(ain-soph fork) H3_EMBED_CACHE / H3_REF_CACHE: what each stage cache skips, and its size bound."""
import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from h3_image_mlx import generate
from h3_image_mlx.models import model_paths
from h3_image_mlx.pipeline import prune_cache, reference_cache


class StageCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        for index, rel in enumerate(model_paths("int8")):
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}" if index >= 3 else f"weights {index}")
        self.refs = []
        for i in range(3):
            path = self.root / f"ref{i}.png"
            Image.new("RGB", (32, 32), (i * 40, 0, 0)).save(path)
            self.refs.append(path)
        self.env = {"H3_EMBED_CACHE": str(self.root / "text-cache"), "H3_REF_CACHE": str(self.root / "ref-cache")}

    def tearDown(self):
        self.tmp.cleanup()

    def run_generate(self, prompt, refs):
        calls = []

        def run(command, **kwargs):
            stage, folder = command[-2:]
            work = Path(folder)
            request = json.loads((work / "request.json").read_text())
            calls.append((stage, list(request["encode_references"])))
            if stage == "text":
                (work / "embedding.safetensors").write_text(f"text {prompt}")
            elif stage == "references":
                for i in request["encode_references"]:
                    (work / f"reference-{i}.safetensors").write_text(f"latent {Path(refs[i]).name}")
            elif stage == "dit":
                self.assertEqual((work / "embedding.safetensors").read_text(), f"text {prompt}")
                for i, ref in enumerate(refs):
                    self.assertEqual((work / f"reference-{i}.safetensors").read_text(), f"latent {ref.name}")
            elif stage == "vae":
                Image.new("RGB", (256, 256), "blue").save(work / "image.png")
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch.dict(os.environ, self.env), \
                patch("h3_image_mlx.pipeline.check_models", return_value=self.root), \
                patch("h3_image_mlx.pipeline.subprocess.run", side_effect=run):
            generate(prompt, models=self.root, references=refs, width=256, height=256)
        return calls

    def test_hits_skip_stages(self):
        first = self.run_generate("Use <Picture 1>", self.refs[:2])
        self.assertEqual([s for s, _ in first], ["text", "references", "dit", "vae"])
        self.assertEqual(first[1][1], [0, 1])
        again = self.run_generate("Use <Picture 1>", self.refs[:2])
        self.assertEqual([s for s, _ in again], ["dit", "vae"])
        # A new prompt over the same pictures plus a new one: only the new picture is encoded, at its position.
        mixed = self.run_generate("Swap them", [self.refs[2], self.refs[1], self.refs[0]])
        self.assertEqual([s for s, _ in mixed], ["text", "references", "dit", "vae"])
        self.assertEqual(mixed[1][1], [0])

    def test_without_caches_every_stage_runs(self):
        self.env = {}
        calls = self.run_generate("Use <Picture 1>", self.refs[:1])
        self.run_generate("Use <Picture 1>", self.refs[:1])
        self.assertEqual(calls, [("text", [0]), ("references", [0]), ("dit", [0]), ("vae", [0])])

    def test_reference_key_follows_content_not_name(self):
        copy = self.root / "renamed.png"
        copy.write_bytes(self.refs[0].read_bytes())
        with patch.dict(os.environ, self.env):
            a, b, c = reference_cache(self.root, [self.refs[0], copy, self.refs[1]], "auto")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_prune_drops_least_recently_used(self):
        cache = self.root / "cache"
        cache.mkdir()
        now = time.time()
        for i in range(4):
            path = cache / f"{i}.safetensors"
            path.write_bytes(b"x" * (1 << 20))
            os.utime(path, (now - 100 + i, now - 100 + i))
        os.utime(cache / "0.safetensors")             # a hit makes the oldest the newest
        stale = cache / "5.safetensors.1.ab.tmp"
        stale.write_bytes(b"x")
        os.utime(stale, (now - 7200, now - 7200))
        prune_cache(cache, 2)
        self.assertEqual(sorted(p.name for p in cache.iterdir()), ["0.safetensors", "3.safetensors"])


if __name__ == "__main__":
    unittest.main()
