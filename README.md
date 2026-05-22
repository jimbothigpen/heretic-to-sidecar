# heretic-to-sidecar

Replays a [Heretic](https://github.com/p-e-w/heretic) abliteration trial
against the base model to produce a peft LoRA adapter, then converts it
to a [frankenturbo2](https://github.com/jimbothigpen/frankenturbo2)
weight-delta sidecar (`.wd.gguf`) that loads via the
`sidecar-weight-delta` plugin.

This tool is **a thin client around Heretic** — Heretic does all the
abliteration math (refusal-direction extraction, per-layer rank-r weight
surgery, optuna study). All credit for that algorithm and implementation
belongs to [@p-e-w](https://github.com/p-e-w) and the Heretic
contributors. We just walk an already-completed optuna journal, replay
the chosen trial's `model.abliterate()` call, save the peft LoRA
adapter, and post-process it into the GGUF format our sidecar plugin
consumes.

## Licensing

This wrapper is licensed under **MIT** (see `LICENSE`). Heretic itself is
licensed **AGPL-3.0** and is loaded at runtime from a local source
checkout via `PYTHONPATH`. We do not redistribute, modify, or bundle
Heretic — anyone running `from_heretic.py` is running unmodified
upstream Heretic.

See [`NOTICE`](NOTICE) for the full runtime-dependency licensing story
(what MIT covers in this repo, what AGPL-3.0 means for your use of
Heretic, and why bundling Heretic was avoided).

The whole point of this repo is to **skip Heretic's interactive
trial-selection menu** — Heretic's optuna journal already contains every
trial's params and outcomes; we just pull the params for a specific
trial and replay the surgery.

## Pipeline

```
                         heretic optuna journal (.jsonl)
                                       │
                          scripts/from_heretic.py --trial N
                                       │
                                       ▼
                             peft adapter directory
                          (adapter_model.safetensors,
                           adapter_config.json,
                           heretic_trial.json)
                                       │
                       scripts/peft_to_wd_gguf.py
                       (wraps llama.cpp convert_lora_to_gguf.py
                        + adds sidecar.type / wd.arch KVs)
                                       │
                                       ▼
                              <trial>.wd.gguf
                       (loaded via sidecar-weight-delta plugin)
```

## Dependencies

This shares the `obliteratus-to-sidecar` venv (torch + transformers +
peft + bitsandbytes + gguf + optuna + scipy). Heretic itself is loaded
at runtime from a local source checkout — no pip install needed.

Use the bootstrap script to clone Heretic at a pinned commit into a
stable location (`/mnt/cephfs/0/Container/systems/ai00/users/builduser/projects/heretic-to-sidecar/src/p-e-w/heretic/`):

```bash
scripts/bootstrap_heretic.sh
```

The script is idempotent: re-running it fetches and resets to the pinned
commit. Override the destination with `HERETIC_PATH=/some/path` if you
want it elsewhere.

`from_heretic.py` resolves the Heretic source at startup in this order:
`$HERETIC_PATH` → `/mnt/cephfs/0/Container/systems/ai00/users/builduser/projects/heretic-to-sidecar/src/p-e-w/heretic/` →
`/tmp/heretic/` (legacy). No `PYTHONPATH=` prefix needed.

## Usage

```bash
# Re-derive the LoRA adapter from trial 9 of an existing Heretic study.
HSA_OVERRIDE_GFX_VERSION=11.0.0 \
/mnt/cephfs/0/Container/systems/ai00/users/builduser/projects/obliteratus-to-sidecar/src/jimbothigpen/obliteratus-to-sidecar/.venv/bin/python \
scripts/from_heretic.py \
    --journal /home/builduser/checkpoints/google--gemma-4-E2B-it.jsonl \
    --trial 9 \
    --output /tmp/heretic-trial-9-adapter

# Convert peft adapter → tagged .wd.gguf.
BASE=/home/builduser/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/<sha>/
/mnt/cephfs/0/Container/systems/ai00/users/builduser/projects/obliteratus-to-sidecar/src/jimbothigpen/obliteratus-to-sidecar/.venv/bin/python \
scripts/peft_to_wd_gguf.py \
    --peft-dir /tmp/heretic-trial-9-adapter \
    --base-model "$BASE" \
    --output /tmp/trial-9.wd.gguf
```

## Notes

- `from_heretic.py` neutralises `sys.argv` before constructing Heretic's
  `Settings` because Heretic uses `pydantic_settings.CliSettingsSource`
  with `cli_parse_args=True`, which would otherwise try (and fail) to
  parse our `--journal/--trial/--output` flags.
- On AMD GPUs without explicit PyTorch ROCm support (e.g. gfx1150 on
  ai00), `HSA_OVERRIDE_GFX_VERSION=11.0.0` makes torch treat the device
  as RDNA3 (gfx1100), which is supported by the standard PyTorch+ROCm
  wheels. Smoke-tested on PyTorch 2.9.1+rocm6.4.
- gfx1103 (ai01 Radeon 780M) is **not viable** for this pipeline.
  PyTorch 2.9.1+rocm6.4 has no native gfx1103 kernels (all matmul/sdpa
  ops fail with `hipErrorInvalidDeviceFunction`). With
  `HSA_OVERRIDE_GFX_VERSION=11.0.0` or `=11.0.2` the basic ops succeed
  in isolation, but real-model workloads (Gemma-4-E2B-it) trigger
  `GPU Hang` aborts mid-load or mid-forward. Run from_heretic.py on a
  gfx1100/1150-class GPU instead.
- `peft_to_wd_gguf.py`'s converter wrapper expects the base model as a
  local path (not an HF id) because llama.cpp's
  `convert_lora_to_gguf.py` does an os.listdir on the path.

## Validation (Gemma-4-E2B-it, trial 9)

End-to-end PPL on `wikitext-2-raw-test.txt` (chunks=32, c=512):

| run                                          | PPL            |
|----------------------------------------------|----------------|
| baseline (no adapter)                        | 146.54 ± 8.55  |
| `--lora trial-9.wd.gguf` (existing path)     | 155.59 ± 9.16  |
| `--sidecar-vectors trial-9.wd.gguf` (plugin) | 155.59 ± 9.16  |

The sidecar plugin path produces bit-identical PPL to the existing
`--lora` path, confirming the engine's apply_to_weights hook correctly
threads the adapter into `ctx->loras` and `build_lora_mm()` for each
forward pass.
