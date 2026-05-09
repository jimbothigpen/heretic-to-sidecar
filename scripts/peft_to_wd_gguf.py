#!/usr/bin/env python3
"""Convert a peft LoRA adapter directory to a frankenturbo2 weight-delta
sidecar (.wd.gguf).

Two-step pipeline:

    1. Run llama.cpp's convert_lora_to_gguf.py on the peft adapter to
       produce a standard LoRA-format GGUF (with `general.type = "adapter"`,
       `adapter.type = "lora"`, and `<base>.lora_a` / `<base>.lora_b`
       tensors per targeted module).

    2. Append the sidecar plugin's required KVs:
         sidecar.type = "weight_delta"
         wd.arch      = <model_architecture>
       The original LoRA-format file is rewritten in place with the new KVs
       appended (existing KVs/tensors preserved verbatim).

The result is dual-schema: passes both `llama_adapter_lora_init`'s checks
(general.type=adapter, adapter.type=lora) AND the weight_delta plugin's
sidecar.type dispatch.

Usage:
    /usr/src/llama-forks/obliteratus-to-sidecar/.venv/bin/python \
    scripts/peft_to_wd_gguf.py \
        --peft-dir /tmp/heretic-trial-9-adapter \
        --base-model google/gemma-4-E2B-it \
        --output /tmp/trial-9.wd.gguf
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import gguf  # noqa: E402


CONVERTER = Path("/usr/src/llama-forks/frankenturbo2/convert_lora_to_gguf.py")
DEFAULT_PYTHON = Path("/usr/src/llama-forks/obliteratus-to-sidecar/.venv/bin/python")


def run_converter(peft_dir: Path, base_model: str, intermediate: Path,
                  python_bin: Path) -> None:
    """Invoke convert_lora_to_gguf.py to produce a LoRA-format GGUF."""
    cmd = [
        str(python_bin),
        str(CONVERTER),
        "--base", base_model,
        "--outfile", str(intermediate),
        "--outtype", "f16",
        str(peft_dir),
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def read_existing(path: Path):
    """Read a LoRA-format GGUF and return (architecture, KVs, tensors)."""
    r = gguf.GGUFReader(str(path))
    arch_field = r.fields.get("general.architecture")
    if arch_field is None:
        sys.exit("missing general.architecture in intermediate GGUF")
    arch = bytes(arch_field.parts[arch_field.data[0]]).decode("utf-8")
    return r, arch


def append_wd_tags(intermediate: Path, output: Path, arch: str) -> None:
    """Read the LoRA GGUF and rewrite it to `output` with sidecar.type and
    wd.arch KVs appended.

    Uses gguf.GGUFWriter to emit a new file rather than mutating in place,
    since GGUFReader is read-only and the writer needs the full schema up
    front. We copy every existing KV (skipping ones we're about to set) and
    every tensor."""
    r, _ = read_existing(intermediate)

    # gguf.GGUFWriter wants the architecture for its `add_architecture`
    # bookkeeping, but since we're already replicating an existing arch,
    # we'll skip the high-level helper and write KVs / tensors directly.
    w = gguf.GGUFWriter(str(output), arch=arch)

    overrides = {
        "sidecar.type": ("string", "weight_delta"),
        "wd.arch": ("string", arch),
    }
    written_keys: set[str] = set()

    for f in r.fields.values():
        key = f.name
        if key in overrides:
            # Skip; we'll write our overridden value below.
            continue
        # The high-level GGUFWriter sets `general.architecture` from its
        # constructor; don't double-write.
        if key == "general.architecture":
            written_keys.add(key)
            continue

        kv_type = gguf.GGUFValueType(f.types[0])
        if kv_type == gguf.GGUFValueType.STRING:
            value = bytes(f.parts[f.data[0]]).decode("utf-8")
            w.add_string(key, value)
        elif kv_type == gguf.GGUFValueType.UINT8:
            w.add_uint8(key, int(f.parts[f.data[0]][0]))
        elif kv_type == gguf.GGUFValueType.INT8:
            w.add_int8(key, int(f.parts[f.data[0]][0]))
        elif kv_type == gguf.GGUFValueType.UINT16:
            w.add_uint16(key, int(f.parts[f.data[0]][0]))
        elif kv_type == gguf.GGUFValueType.INT16:
            w.add_int16(key, int(f.parts[f.data[0]][0]))
        elif kv_type == gguf.GGUFValueType.UINT32:
            w.add_uint32(key, int(f.parts[f.data[0]][0]))
        elif kv_type == gguf.GGUFValueType.INT32:
            w.add_int32(key, int(f.parts[f.data[0]][0]))
        elif kv_type == gguf.GGUFValueType.UINT64:
            w.add_uint64(key, int(f.parts[f.data[0]][0]))
        elif kv_type == gguf.GGUFValueType.INT64:
            w.add_int64(key, int(f.parts[f.data[0]][0]))
        elif kv_type == gguf.GGUFValueType.FLOAT32:
            w.add_float32(key, float(f.parts[f.data[0]][0]))
        elif kv_type == gguf.GGUFValueType.FLOAT64:
            w.add_float64(key, float(f.parts[f.data[0]][0]))
        elif kv_type == gguf.GGUFValueType.BOOL:
            w.add_bool(key, bool(f.parts[f.data[0]][0]))
        elif kv_type == gguf.GGUFValueType.ARRAY:
            # Punt: arrays aren't typically present in LoRA adapter GGUFs
            # other than alora.invocation_tokens (uint32 array). Pass through
            # as raw bytes via the low-level KV writer if present.
            arr_type = gguf.GGUFValueType(f.types[1])
            if arr_type == gguf.GGUFValueType.UINT32:
                values = [int(f.parts[idx][0]) for idx in f.data]
                w.add_array(key, values)
            else:
                print(f"  warning: skipping unsupported array KV {key} "
                      f"of element type {arr_type}", flush=True)
        else:
            print(f"  warning: skipping unsupported KV {key} of type {kv_type}",
                  flush=True)
        written_keys.add(key)

    # Apply overrides.
    for k, (vtype, value) in overrides.items():
        assert vtype == "string"
        w.add_string(k, value)

    # Copy every tensor verbatim. data is a numpy array.
    for t in r.tensors:
        w.add_tensor(
            t.name,
            t.data,
            raw_dtype=t.tensor_type,
        )

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {output}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--peft-dir", required=True,
                   help="Directory containing adapter_model.safetensors + adapter_config.json")
    p.add_argument("--base-model", required=True,
                   help="HF model id or local path to the base model "
                        "(passed to convert_lora_to_gguf.py --base)")
    p.add_argument("--output", required=True,
                   help="Output .wd.gguf path")
    p.add_argument("--python", default=str(DEFAULT_PYTHON),
                   help=f"Python interpreter to run the converter "
                        f"(default: {DEFAULT_PYTHON})")
    p.add_argument("--keep-intermediate", action="store_true",
                   help="Keep the post-converter, pre-tagging intermediate GGUF")
    args = p.parse_args()

    peft_dir = Path(args.peft_dir).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    intermediate = output.with_suffix(".lora_intermediate.gguf")

    run_converter(peft_dir, args.base_model, intermediate, Path(args.python))

    _, arch = read_existing(intermediate)
    print(f"detected architecture = {arch}", flush=True)
    append_wd_tags(intermediate, output, arch)

    if not args.keep_intermediate:
        intermediate.unlink()

    # Echo the trial provenance JSON if the peft dir has it.
    prov = peft_dir / "heretic_trial.json"
    if prov.exists():
        print("\nheretic_trial.json (from peft dir):")
        print(prov.read_text())


if __name__ == "__main__":
    main()
