"""Unit tests for the refusal_directions cache layer.

Covers key derivation (same inputs → same hash; perturbed inputs → different
hashes), cache_dir propagation, missing-file detection, and torch tensor
round-trip. Pure CPU; never touches GPU. Safe to run while ai00 PyTorch ROCm
is in its current broken state — `import torch` and `torch.save`/`torch.load`
on CPU tensors do not exercise the hipLaunchKernel path that's hung.

Run:
    /mnt/cephfs/0/Container/systems/ai00/users/builduser/projects/obliteratus-to-sidecar/src/jimbothigpen/obliteratus-to-sidecar/.venv/bin/python -m unittest \\
        discover -s tests -t . -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

# Make the repo root importable when invoked from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from heretic_to_sidecar.cache import (
    CACHE_SCHEMA,
    heretic_commit,
    refusal_directions_cache_path,
)


SETTINGS_A = {
    "model": "google/gemma-4-E2B-it",
    "seed": 1,
    "good_prompts": {"dataset": "tatsu-lab/alpaca", "limit": 400},
    "bad_prompts": {"dataset": "harmful", "limit": 400},
    "orthogonalize_direction": True,
    "torch_dtype": "bfloat16",
}


class TestKeyDerivation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_same_inputs_same_path(self):
        a = refusal_directions_cache_path(SETTINGS_A, "deadbeef", self.cache_dir)
        b = refusal_directions_cache_path(dict(SETTINGS_A), "deadbeef", self.cache_dir)
        self.assertEqual(a, b)

    def test_path_under_cache_dir_with_pt_suffix(self):
        p = refusal_directions_cache_path(SETTINGS_A, "deadbeef", self.cache_dir)
        self.assertEqual(p.parent, self.cache_dir)
        self.assertTrue(p.name.endswith(".pt"))
        # sha256 hex is 64 chars + ".pt"
        self.assertEqual(len(p.name), 64 + len(".pt"))

    def test_settings_perturbation_changes_path(self):
        base = refusal_directions_cache_path(SETTINGS_A, "deadbeef", self.cache_dir)
        for field, mutator in [
            ("model", lambda v: "google/gemma-4-E4B-it"),
            ("seed", lambda v: 2),
            ("good_prompts", lambda v: {"dataset": "other", "limit": 400}),
            ("bad_prompts", lambda v: {"dataset": v["dataset"], "limit": 401}),
            ("orthogonalize_direction", lambda v: not v),
            ("torch_dtype", lambda v: "float16"),
        ]:
            with self.subTest(field=field):
                perturbed = dict(SETTINGS_A)
                perturbed[field] = mutator(perturbed[field])
                self.assertNotEqual(
                    base,
                    refusal_directions_cache_path(perturbed, "deadbeef", self.cache_dir),
                    f"perturbing {field} did not change cache path",
                )

    def test_commit_perturbation_changes_path(self):
        a = refusal_directions_cache_path(SETTINGS_A, "deadbeef", self.cache_dir)
        b = refusal_directions_cache_path(SETTINGS_A, "cafef00d", self.cache_dir)
        c = refusal_directions_cache_path(SETTINGS_A, None, self.cache_dir)
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(b, c)

    def test_cache_dir_propagates(self):
        other = self.cache_dir / "elsewhere"
        a = refusal_directions_cache_path(SETTINGS_A, "deadbeef", self.cache_dir)
        b = refusal_directions_cache_path(SETTINGS_A, "deadbeef", other)
        self.assertEqual(a.name, b.name)
        self.assertEqual(b.parent, other)

    def test_settings_dict_key_order_irrelevant(self):
        # JSON dump uses sort_keys=True; verify a reordered dict produces the
        # same path.
        reordered = {k: SETTINGS_A[k] for k in reversed(list(SETTINGS_A))}
        self.assertEqual(
            refusal_directions_cache_path(SETTINGS_A, "deadbeef", self.cache_dir),
            refusal_directions_cache_path(reordered, "deadbeef", self.cache_dir),
        )

    def test_none_settings_distinct_from_empty(self):
        a = refusal_directions_cache_path(None, "deadbeef", self.cache_dir)
        b = refusal_directions_cache_path({}, "deadbeef", self.cache_dir)
        self.assertNotEqual(a, b)

    def test_schema_bump_changes_path(self):
        # Capture the current schema's path, then monkey-patch CACHE_SCHEMA
        # one higher and recompute. They must differ.
        import heretic_to_sidecar.cache as cache_mod
        before = refusal_directions_cache_path(SETTINGS_A, "deadbeef", self.cache_dir)
        original = cache_mod.CACHE_SCHEMA
        try:
            cache_mod.CACHE_SCHEMA = original + 1
            after = refusal_directions_cache_path(SETTINGS_A, "deadbeef", self.cache_dir)
        finally:
            cache_mod.CACHE_SCHEMA = original
        self.assertNotEqual(before, after)
        self.assertEqual(cache_mod.CACHE_SCHEMA, original)  # restored


class TestFileIO(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_missing_file_path_does_not_exist(self):
        # The recompute path in from_heretic.py is gated on
        # `not args.no_cache and cache_path.is_file()`. The contract is just
        # that an unwritten cache yields a non-existent path under cache_dir.
        p = refusal_directions_cache_path(SETTINGS_A, "deadbeef", self.cache_dir)
        self.assertFalse(p.exists())
        self.assertFalse(p.is_file())
        self.assertEqual(p.parent, self.cache_dir)

    def test_write_then_read_round_trip(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        p = refusal_directions_cache_path(SETTINGS_A, "deadbeef", self.cache_dir)
        # Production code uses parents=True, exist_ok=True before writing.
        p.parent.mkdir(parents=True, exist_ok=True)
        # Shape mirrors a per-layer refusal_directions tensor:
        # [n_layers, hidden_dim], normalised float32 (CPU tensor — no GPU).
        original = torch.randn(32, 2048, dtype=torch.float32)
        torch.save(original, p)
        self.assertTrue(p.is_file())
        loaded = torch.load(p, map_location="cpu", weights_only=True)
        self.assertEqual(original.shape, loaded.shape)
        self.assertEqual(original.dtype, loaded.dtype)
        # Bit-identical, not just close: torch.save preserves exact bytes.
        self.assertTrue(torch.equal(original, loaded))


class TestHereticCommit(unittest.TestCase):
    def test_non_git_path_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(heretic_commit(Path(tmp)))

    def test_nonexistent_path_returns_none(self):
        self.assertIsNone(heretic_commit(Path("/nonexistent/path/for/test")))


if __name__ == "__main__":
    unittest.main()
