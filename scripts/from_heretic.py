#!/usr/bin/env python3
"""Re-derive a Heretic-style abliteration LoRA adapter from the optuna journal.

Skips Heretic's interactive trial-selection menu: takes a chosen trial id
on the command line, replays the abliteration math against the base model,
and saves the resulting peft adapter directory. The directory can then be
fed to llama.cpp's convert_lora_to_gguf.py + the wd-tagging post-processor
to produce a frankenturbo2 weight-delta sidecar (.wd.gguf).

Heretic must be importable. We don't pip-install it; we resolve its
source location at startup and inject `<path>/src` into sys.path. The
obliteratus-to-sidecar venv (torch+rocm, transformers, peft,
bitsandbytes, gguf, optuna) supplies the heavy ML deps for both projects.

Heretic source resolution order:
  1. $HERETIC_PATH (must point to the repo root)
  2. /usr/src/llama-forks/_heretic-vendored/ (scripts/bootstrap_heretic.sh
     default)
  3. /tmp/heretic/ (legacy; kept for backwards compatibility)

If none of those exist, we exit with an installation hint rather than
letting the import-time error confuse the user.

Usage:
    /usr/src/llama-forks/obliteratus-to-sidecar/.venv/bin/python \
    scripts/from_heretic.py \
        --journal /home/builduser/checkpoints/google--gemma-4-E2B-it.jsonl \
        --trial 9 \
        --output /tmp/heretic-trial-9-adapter

Note: this loads the base model + runs ~800 prompts through it to compute
refusal directions. On Gemma-4-E2B-it that takes ~5 minutes of GPU time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Repo-relative import — sibling package
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from heretic_to_sidecar.journal import parse_journal, trial_summary  # noqa: E402


def _resolve_heretic_path() -> Path:
    candidates: list[tuple[Path, str]] = []
    env = os.environ.get("HERETIC_PATH")
    if env:
        candidates.append((Path(env), "$HERETIC_PATH"))
    candidates.append(
        (Path("/usr/src/llama-forks/_heretic-vendored"), "vendored default")
    )
    candidates.append((Path("/tmp/heretic"), "legacy /tmp checkout"))
    for path, label in candidates:
        if (path / "src" / "heretic" / "model.py").is_file():
            print(f"using Heretic source at {path} ({label})", flush=True)
            return path
    sys.exit(
        "ERROR: Heretic source not found.\n"
        "Run scripts/bootstrap_heretic.sh to clone it to "
        "/usr/src/llama-forks/_heretic-vendored/, or set HERETIC_PATH "
        "to an existing checkout root."
    )


sys.path.insert(0, str(_resolve_heretic_path() / "src"))

# Heretic
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from heretic.config import Settings  # noqa: E402
from heretic.model import Model, AbliterationParameters  # noqa: E402
from heretic.utils import load_prompts, set_seed  # noqa: E402
from heretic.system import empty_cache  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--journal", required=True,
                   help="Path to Heretic optuna journal jsonl")
    p.add_argument("--trial", type=int, required=True,
                   help="Trial id to replay (must be COMPLETE in the journal)")
    p.add_argument("--output", required=True,
                   help="Directory to write the peft adapter to")
    p.add_argument("--device-map", default="auto",
                   help="Override Heretic Settings.device_map (default: auto)")
    args = p.parse_args()

    journal = parse_journal(args.journal)
    if args.trial not in journal.trials:
        sys.exit(f"trial {args.trial} not found in journal")
    trial = journal.trials[args.trial]
    if trial.state != "COMPLETE":
        sys.exit(f"trial {args.trial} state={trial.state}, not COMPLETE")

    print(f"replaying {trial_summary(trial)}", flush=True)
    print(f"  params:", flush=True)
    for k, v in sorted(trial.params.items()):
        print(f"    {k} = {v}", flush=True)

    # Reconstruct Heretic settings from the journal's stashed settings blob.
    # Keep the exact same dtypes/quantization/prompts/seed so refusal
    # direction computation is bit-identical to the original study.
    #
    # Heretic's Settings is a pydantic_settings.BaseSettings with an active
    # CliSettingsSource(cli_parse_args=True), which inspects sys.argv at
    # instantiation time. Our own argparse already consumed --journal/--trial
    # /--output, but those would still confuse Heretic's parser, so neuter
    # sys.argv before Settings.model_validate runs.
    if not journal.settings:
        sys.exit("journal lacks the 'settings' user_attr; cannot reconstruct Settings")
    sj = dict(journal.settings)
    sj["device_map"] = args.device_map
    saved_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        settings = Settings.model_validate(sj)
    finally:
        sys.argv = saved_argv
    set_seed(settings.seed)

    print(f"\nloading base model {settings.model}...", flush=True)
    model = Model(settings)

    print(f"\nloading good prompts ({settings.good_prompts.dataset})...", flush=True)
    good_prompts = load_prompts(settings, settings.good_prompts)
    print(f"  {len(good_prompts)} prompts", flush=True)
    print(f"loading bad prompts ({settings.bad_prompts.dataset})...", flush=True)
    bad_prompts = load_prompts(settings, settings.bad_prompts)
    print(f"  {len(bad_prompts)} prompts", flush=True)

    print("\ncomputing per-layer refusal directions...", flush=True)
    good_means = model.get_residuals_mean(good_prompts)
    bad_means = model.get_residuals_mean(bad_prompts)
    refusal_directions = F.normalize(bad_means - good_means, p=2, dim=1)
    if settings.orthogonalize_direction:
        good_directions = F.normalize(good_means, p=2, dim=1)
        proj = torch.sum(refusal_directions * good_directions, dim=1)
        refusal_directions = refusal_directions - proj.unsqueeze(1) * good_directions
        refusal_directions = F.normalize(refusal_directions, p=2, dim=1)
        del good_directions, proj
    del good_means, bad_means
    empty_cache()

    # Build the AbliterationParameters dict from the trial params, reproducing
    # Heretic's `min_weight = min_weight_fraction * max_weight` transform from
    # main.py objective().
    params: dict[str, AbliterationParameters] = {}
    for component in model.get_abliterable_components():
        max_w = float(trial.params[f"{component}.max_weight"])
        max_pos = float(trial.params[f"{component}.max_weight_position"])
        min_w_frac = float(trial.params[f"{component}.min_weight"])
        min_dist = float(trial.params[f"{component}.min_weight_distance"])
        params[component] = AbliterationParameters(
            max_weight=max_w,
            max_weight_position=max_pos,
            min_weight=min_w_frac * max_w,
            min_weight_distance=min_dist,
        )
        print(f"  {component}: max_weight={max_w:.4f} pos={max_pos:.4f} "
              f"min={min_w_frac * max_w:.4f} dist={min_dist:.4f}", flush=True)

    direction_scope = trial.params.get("direction_scope")
    direction_index = trial.params.get("direction_index")
    if direction_scope == "per layer":
        direction_index = None
    elif direction_index is not None:
        direction_index = float(direction_index)
    print(f"  direction_scope={direction_scope} direction_index={direction_index}",
          flush=True)

    print("\nresetting peft adapter to clean state...", flush=True)
    model.reset_model()

    print("running abliterate()...", flush=True)
    model.abliterate(refusal_directions, direction_index, params)

    print(f"\nsaving adapter to {args.output}...", flush=True)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(
        str(out_dir),
        max_shard_size=settings.max_shard_size,
    )
    # Stash trial provenance alongside the adapter so post-processors can
    # populate sidecar.* / wd.* KVs without re-parsing the journal.
    with open(out_dir / "heretic_trial.json", "w") as f:
        json.dump({
            "trial_id": trial.trial_id,
            "kl_divergence": trial.user_attrs.get("kl_divergence"),
            "refusals": trial.user_attrs.get("refusals"),
            "n_bad_prompts": trial.user_attrs.get("n_bad_prompts"),
            "params": trial.params,
            "model": settings.model,
        }, f, indent=2)

    print("done.")


if __name__ == "__main__":
    main()
