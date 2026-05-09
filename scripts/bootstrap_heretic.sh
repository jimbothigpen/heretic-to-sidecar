#!/usr/bin/env bash
# Bootstrap a Heretic source checkout for from_heretic.py.
#
# Heretic is loaded at runtime via PYTHONPATH (we don't pip install it),
# so we need its source on disk. The default location is a sibling of
# the other llama-forks repos; HERETIC_PATH overrides it.
#
# Idempotent: if the target exists as a git repo, fetches and resets to
# the pinned commit. If it doesn't exist, clones fresh.
#
# Usage:
#   scripts/bootstrap_heretic.sh
#   HERETIC_PATH=/some/other/path scripts/bootstrap_heretic.sh

set -euo pipefail

HERETIC_REPO="${HERETIC_REPO:-https://github.com/p-e-w/heretic.git}"
# Pinned to a known-good commit. Heretic's API is not versioned, and
# main can break our import patterns (heretic.config.Settings,
# heretic.model.Model/AbliterationParameters, heretic.utils.load_prompts/
# set_seed, heretic.system.empty_cache). Bump deliberately and retest.
HERETIC_PIN="${HERETIC_PIN:-8b5b85bec904aae764aa8d63170e814bd0222a6d}"
HERETIC_PATH="${HERETIC_PATH:-/usr/src/llama-forks/_heretic-vendored}"

if [[ -d "$HERETIC_PATH/.git" ]]; then
    echo "==> existing checkout: $HERETIC_PATH"
    git -C "$HERETIC_PATH" fetch --quiet origin
    git -C "$HERETIC_PATH" -c advice.detachedHead=false \
        checkout --quiet "$HERETIC_PIN"
elif [[ -e "$HERETIC_PATH" ]]; then
    echo "ERROR: $HERETIC_PATH exists but is not a git checkout." >&2
    echo "Remove it or set HERETIC_PATH to a different location." >&2
    exit 1
else
    echo "==> cloning $HERETIC_REPO -> $HERETIC_PATH"
    git clone --quiet "$HERETIC_REPO" "$HERETIC_PATH"
    git -C "$HERETIC_PATH" -c advice.detachedHead=false \
        checkout --quiet "$HERETIC_PIN"
fi

# Sanity check: the import targets from_heretic.py needs.
if [[ ! -f "$HERETIC_PATH/src/heretic/model.py" ]]; then
    echo "ERROR: $HERETIC_PATH/src/heretic/model.py missing after bootstrap." >&2
    echo "Pinned commit may be incompatible with our imports." >&2
    exit 1
fi

echo "==> Heretic ready at $HERETIC_PATH (commit $HERETIC_PIN)"
