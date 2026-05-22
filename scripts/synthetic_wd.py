#!/usr/bin/env python3
"""Generate a synthetic .wd.gguf for smoke-testing the engine + plugin path.

Produces a LoRA-format GGUF with the dual-schema KVs the weight_delta
plugin expects. The LoRA tensors are all zeros, so the adapter is a no-op:
loading and running through the plugin should produce identical outputs to
running without any sidecar. Used to verify that the plugin path is wired
correctly (dlopen → register → apply_to_weights → llama_adapter_lora_init →
ctx->loras → build_lora_mm) without depending on Heretic.

Usage:
    /mnt/cephfs/0/Container/systems/ai00/users/builduser/projects/obliteratus-to-sidecar/src/jimbothigpen/obliteratus-to-sidecar/.venv/bin/python \
    scripts/synthetic_wd.py \
        --base /path/to/base_model.gguf \
        --output /tmp/synthetic.wd.gguf \
        --rank 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gguf
import numpy as np


def gguf_int(reader, key: str) -> int:
    f = reader.fields[key]
    return int(f.parts[f.data[0]][0])


def gguf_str(reader, key: str) -> str:
    f = reader.fields[key]
    return bytes(f.parts[f.data[0]]).decode("utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True, help="Path to the base model GGUF")
    p.add_argument("--output", required=True, help="Output .wd.gguf path")
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--targets", nargs="+",
                   default=["attn_output", "ffn_down"],
                   help="Tensor name suffixes to wrap (default: attn_output, ffn_down)")
    p.add_argument("--alpha", type=float, default=1.0)
    args = p.parse_args()

    r = gguf.GGUFReader(args.base)
    arch = gguf_str(r, "general.architecture")
    n_layer = gguf_int(r, f"{arch}.block_count")
    print(f"base arch={arch} n_layer={n_layer}")

    # Find target tensors in the base model. For each `blk.{i}.<suffix>.weight`,
    # we'll emit a paired (.lora_a, .lora_b) so build_lora_mm picks it up.
    targets = []
    for t in r.tensors:
        for suf in args.targets:
            if t.name.startswith("blk.") and t.name.endswith(f".{suf}.weight"):
                # ggml interprets shape as ne[0]=fastest. For a [K, N] weight
                # ggml_mul_mat(W, x [K, M]) returns [N, M], so K = "input dim"
                # and N = "output dim" from the matmul's POV. lora_a is [K, r]
                # (input → rank) and lora_b is [r, N] (rank → output).
                k = int(t.shape[0])
                n = int(t.shape[1])
                targets.append((t.name, k, n))
                break

    if not targets:
        raise SystemExit(f"no tensors matched suffixes {args.targets} in base model")
    print(f"emitting deltas for {len(targets)} tensors at rank={args.rank}")

    w = gguf.GGUFWriter(args.output, arch=arch)

    # LoRA-format required KVs (parsed by llama_adapter_lora_init)
    w.add_string("general.type", "adapter")
    w.add_string("adapter.type", "lora")
    w.add_float32("adapter.lora.alpha", float(args.alpha))

    # weight_delta sidecar plugin KVs
    w.add_string("sidecar.type", "weight_delta")
    w.add_string("wd.arch", arch)

    # All-zero rank-r tensors → zero delta, no-op adapter.
    for name, k, n in targets:
        a = np.zeros((args.rank, k), dtype=np.float32)  # peft convention [r, d_in]
        b = np.zeros((n, args.rank), dtype=np.float32)  # peft convention [d_out, r]
        w.add_tensor(name + ".lora_a", a)
        w.add_tensor(name + ".lora_b", b)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
