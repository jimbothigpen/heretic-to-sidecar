"""SHA-keyed cache helpers for refusal_directions.

Pulled out of scripts/from_heretic.py so the key derivation can be unit-
tested without importing torch/heretic at module load time. The script
re-exports these via a thin import shim.

Cache layout:
    <cache_dir>/<sha256(key)>.pt

The key is a deterministic JSON dump of:
    {
        "settings": journal.settings,           # dict from study user_attr
        "heretic_commit": <git HEAD>|None,      # bumps when Heretic moves
        "schema": <int>,                        # bump on math change here
    }

`schema` is the explicit knob to invalidate every cache entry from a
script-side math change. Bump it whenever the compute block in
from_heretic.py that produces the tensor would yield different numbers
for the same inputs.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

CACHE_SCHEMA = 1


def heretic_commit(path: Path) -> str | None:
    """Return the Heretic checkout's HEAD sha, or None if the path isn't a
    git repo / git is unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def refusal_directions_cache_path(
    journal_settings: dict | None,
    heretic_commit_sha: str | None,
    cache_dir: Path,
) -> Path:
    """Return the cache file path for the given inputs.

    `journal_settings` is the raw dict the original Heretic study was
    launched with — it captures model id, dtype, quant, prompts, seed,
    orthogonalize_direction, etc. Don't drop fields here (e.g. device_map);
    the caller should normalise *before* hashing if it wants two runs to
    share a cache slot.
    """
    key = {
        "settings": journal_settings,
        "heretic_commit": heretic_commit_sha,
        "schema": CACHE_SCHEMA,
    }
    blob = json.dumps(key, sort_keys=True, default=str).encode()
    sha = hashlib.sha256(blob).hexdigest()
    return cache_dir / f"{sha}.pt"
