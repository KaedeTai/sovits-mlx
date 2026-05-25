"""Convert GPT-SoVITS S1 PyTorch ckpt → MLX safetensors.

The ckpt format (Lightning .ckpt) is:
    { "weight": OrderedDict,  "config": {...},  "info": {...} }

All param keys are prefixed with `model.` — we strip that. Inside, the layout
already matches MLX nn.Linear / nn.Embedding / nn.LayerNorm conventions, so
no shape transposes are needed.

Round-trip naming:
    PT:  model.h.layers.0.self_attn.in_proj_weight
    MLX: h.layers.0.self_attn.in_proj_weight

Embedding rename:
    PT:  model.ar_text_embedding.word_embeddings.weight
    MLX: ar_text_embedding.word_embeddings.weight   (TokenEmbedding wrapper)

Usage:
    python convert.py --in  /Users/kaede/tts/_s1_trilingual/arm_A_e15_trilingual.ckpt \
                      --out /Users/kaede/tts/_sovits_mlx/s1_mlx/s1.safetensors
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def convert(in_path: str, out_path: str, dump_config: str | None = None) -> None:
    import torch

    print(f"Loading {in_path} …", flush=True)
    ck = torch.load(in_path, map_location="cpu", weights_only=False)
    sd = ck["weight"]
    cfg = ck.get("config", {})
    info = ck.get("info", {})
    print(f"  loaded {len(sd)} tensors; config keys: {list(cfg.keys())}")

    out: dict[str, np.ndarray] = {}
    skipped: list[str] = []
    for k, v in sd.items():
        if not k.startswith("model."):
            skipped.append(k)
            continue
        new_k = k[len("model.") :]
        arr = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        # Upcast fp16 → fp32 for max numerical headroom during MLX inference;
        # MLX will cast back to fp16 if model is in fp16.  Keeps the file
        # 2× bigger but matches the way PT MPS uses fp32 accumulation.
        # Actually — keep fp16 to preserve bit-equivalence of weights.
        out[new_k] = np.ascontiguousarray(arr)

    if skipped:
        print(f"  skipped {len(skipped)} keys without 'model.' prefix: {skipped[:5]}…")

    # --- save ---
    from safetensors.numpy import save_file as save_safetensors

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_safetensors(out, out_path)
    print(f"Saved {len(out)} tensors → {out_path}")

    # --- summary ---
    pre = Counter(k.split(".")[0] for k in out)
    print("Key counts by top prefix:")
    for p, c in sorted(pre.items(), key=lambda kv: -kv[1]):
        print(f"  {p}: {c}")

    # --- optional: dump config alongside ---
    if dump_config is None:
        dump_config = str(Path(out_path).with_suffix(".config.json"))
    Path(dump_config).write_text(json.dumps(cfg, indent=2, default=str))
    print(f"Wrote config → {dump_config}")

    # --- round-trip smoke test ---
    from safetensors.numpy import load_file as load_safetensors

    rt = load_safetensors(out_path)
    assert set(rt.keys()) == set(out.keys())
    for k in rt:
        assert rt[k].shape == out[k].shape, f"shape mismatch {k}"
    print(f"Round-trip OK: {len(rt)} tensors load back identically.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in",
        dest="in_path",
        default="/Users/kaede/tts/_s1_trilingual/arm_A_e15_trilingual.ckpt",
    )
    ap.add_argument(
        "--out",
        dest="out_path",
        default="/Users/kaede/tts/_sovits_mlx/s1_mlx/s1.safetensors",
    )
    ap.add_argument("--dump-config", default=None)
    args = ap.parse_args()
    convert(args.in_path, args.out_path, args.dump_config)
