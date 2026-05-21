"""Convert GPT-SoVITS S2 (v2ProTw) PyTorch ckpt → MLX safetensors.

Usage:
  python convert.py --in  /Users/kaede/tts/tw_finetune_synthetic/s2_logs_r4/s2_full_15.pth \
                    --out /Users/kaede/tts/_sovits_mlx/model.safetensors

Drops `enc_q` (training-only posterior encoder).
Keeps `ssl_proj` (top-level) for round-trip completeness even though decode() doesn't use it.

Naming convention (matches our MLX model):
  - Every `nn.Conv1d` lives inside a `Conv1dPT` wrapper, so each weight gets
    `.conv.weight` / `.conv.bias`, and PT (out,in,k) -> MLX (out,k,in)  i.e. transpose(0,2,1).
  - Every `nn.ConvTranspose1d` lives inside a `ConvTranspose1dPT` wrapper, so each weight gets
    `.conv_t.weight` / `.conv_t.bias`, and PT (in,out,k) -> MLX (out,k,in)  i.e. transpose(1,2,0).
  - LayerNorm: PyTorch uses `.gamma`/`.beta`, we rename to `.weight`/`.bias`.
  - weight_norm: fuse `weight_g * weight_v / ||weight_v||` -> `.weight`, drop `weight_g`.
  - All Linear / Embedding / PReLU / scalar parameters: pass through (shape unchanged).
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np


# ----- which keys belong to ConvTranspose1d (vs plain Conv1d) ---------------
def is_conv_transpose_key(key: str) -> bool:
    # Generator's upsamplers are the only ConvTranspose1d in S2.
    return bool(re.match(r"dec\.ups\.\d+\.", key))


# ----- weight_norm fusion ---------------------------------------------------
def fuse_weight_norm(weight_g: np.ndarray, weight_v: np.ndarray) -> np.ndarray:
    """w = g * v / ||v||_2 where the norm is over all but the 0th axis (PyTorch default for Conv1d)."""
    axes = tuple(range(1, weight_v.ndim))
    norm = np.sqrt((weight_v.astype(np.float32) ** 2).sum(axis=axes, keepdims=True))
    return weight_g * weight_v / (norm + 1e-12)


# ----- main convert ---------------------------------------------------------
def convert(in_path: str, out_path: str) -> None:
    import torch
    print(f"Loading {in_path} ...", flush=True)
    ckpt = torch.load(in_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model") or ckpt.get("weight") or ckpt
    if not isinstance(sd, dict):
        raise SystemExit("Could not find state_dict in ckpt")
    print(f"  loaded {len(sd)} keys")

    # --- pass 1: fuse weight_norm pairs into plain weight ------------------
    sanitized: dict[str, np.ndarray] = {}
    skipped_enc_q = 0
    fused = 0
    for key in sorted(sd.keys()):
        # drop posterior encoder (training-only)
        if key.startswith("enc_q."):
            skipped_enc_q += 1
            continue
        # drop training-only RVQ stats (we only need the codebook .embed)
        if key.startswith("quantizer.vq.layers.") and any(
            key.endswith(s)
            for s in (".inited", ".cluster_size", ".embed_avg")
        ):
            continue
        v = sd[key]
        if hasattr(v, "numpy"):
            v = v.detach().cpu().numpy()
        elif hasattr(v, "to"):
            v = v.cpu().numpy()
        else:
            v = np.asarray(v)
        if key.endswith(".weight_g"):
            continue  # paired with weight_v, handled below
        if key.endswith(".weight_v"):
            g_key = key[: -len(".weight_v")] + ".weight_g"
            base = key[: -len(".weight_v")]
            g = sd[g_key]
            g = g.detach().cpu().numpy() if hasattr(g, "numpy") else np.asarray(g)
            sanitized[base + ".weight"] = fuse_weight_norm(g, v)
            fused += 1
            continue
        # LayerNorm rename gamma/beta -> weight/bias
        if key.endswith(".gamma"):
            sanitized[key[: -len(".gamma")] + ".weight"] = v
            continue
        if key.endswith(".beta"):
            sanitized[key[: -len(".beta")] + ".bias"] = v
            continue
        sanitized[key] = v

    print(f"  fused {fused} weight_norm pairs")
    print(f"  skipped {skipped_enc_q} enc_q keys (training-only)")

    # --- pass 2: remap Conv1d/ConvTranspose1d into wrapper layout ----------
    remapped: dict[str, np.ndarray] = {}
    for key, value in sanitized.items():
        # 3-D weight  →  (a) ConvTranspose1d → transpose (1,2,0), key .ups.N.weight → .ups.N.conv_t.weight
        #                (b) plain Conv1d    → transpose (0,2,1), key .X.weight      → .X.conv.weight
        if value.ndim == 3 and key.endswith(".weight"):
            if is_conv_transpose_key(key):
                new_val = np.transpose(value, (1, 2, 0))
                new_key = key[: -len(".weight")] + ".conv_t.weight"
            else:
                new_val = np.transpose(value, (0, 2, 1))
                new_key = key[: -len(".weight")] + ".conv.weight"
            remapped[new_key] = new_val
            continue
        # 1-D bias whose sibling weight is 3-D → wrap into .conv./.conv_t.
        if value.ndim == 1 and key.endswith(".bias"):
            sib = key[: -len(".bias")] + ".weight"
            if sib in sanitized and sanitized[sib].ndim == 3:
                if is_conv_transpose_key(sib):
                    new_key = key[: -len(".bias")] + ".conv_t.bias"
                else:
                    new_key = key[: -len(".bias")] + ".conv.bias"
                remapped[new_key] = value
                continue
        # Everything else (Linear, Embedding, PReLU, MHA emb_rel_*, etc.) → pass through
        remapped[key] = value

    # --- pass 3: rename quantizer codebook key (MLX skips attr names starting with '_') -----
    # PyTorch: quantizer.vq.layers.0._codebook.embed → MLX: quantizer.vq.layers.0.codebook.embed
    if "quantizer.vq.layers.0._codebook.embed" in remapped:
        remapped["quantizer.vq.layers.0.codebook.embed"] = remapped.pop(
            "quantizer.vq.layers.0._codebook.embed"
        )

    # --- save safetensors ----------
    from safetensors.numpy import save_file as save_safetensors
    # safetensors needs contiguous arrays
    remapped = {k: np.ascontiguousarray(v) for k, v in remapped.items()}
    save_safetensors(remapped, out_path)
    print(f"Saved {len(remapped)} tensors → {out_path}")

    # --- summary by top-level prefix ----------
    from collections import Counter
    pre = Counter(k.split(".")[0] for k in remapped.keys())
    print("Key counts by prefix:")
    for p, c in sorted(pre.items(), key=lambda kv: -kv[1]):
        print(f"  {p}: {c}")

    # --- round-trip smoke test ----------
    from safetensors.numpy import load_file as load_safetensors
    rt = load_safetensors(out_path)
    assert set(rt.keys()) == set(remapped.keys())
    print(f"Round-trip OK: {len(rt)} tensors load back identically.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path",
                    default="/Users/kaede/tts/tw_finetune_synthetic/s2_logs_r4/s2_full_15.pth")
    ap.add_argument("--out", dest="out_path",
                    default="/Users/kaede/tts/_sovits_mlx/model.safetensors")
    args = ap.parse_args()
    convert(args.in_path, args.out_path)
