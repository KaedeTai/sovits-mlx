"""Convert GPT-SoVITS S2 (v2ProTw) PyTorch ckpts for TRAINING resume.

Differs from convert.py in two ways:
  1. Reads the "model" field of s2_full_N.pth (containing G + enc_q)
  2. KEEPS enc_q for training resume (convert.py drops it for inference)
  3. Also converts a separate s2_D_N.pth into a discriminator safetensors

Usage:
  python convert_full.py \
      --g  ~/tts/tw_finetune_synthetic/s2_logs_r5/s2_full_16.pth \
      --d  ~/tts/tw_finetune_synthetic/s2_logs_r5/s2_D_16.pth \
      --out-g  ~/tts/_sovits_mlx_train/r6/init_g.safetensors \
      --out-d  ~/tts/_sovits_mlx_train/r6/init_d.safetensors
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np


# Same helpers as convert.py
def is_conv_transpose_key(key: str) -> bool:
    return bool(re.match(r"dec\.ups\.\d+\.", key))


def fuse_weight_norm(weight_g: np.ndarray, weight_v: np.ndarray) -> np.ndarray:
    axes = tuple(range(1, weight_v.ndim))
    norm = np.sqrt((weight_v.astype(np.float32) ** 2).sum(axis=axes, keepdims=True))
    return weight_g * weight_v / (norm + 1e-12)


def _to_np(v):
    if hasattr(v, "numpy"):
        return v.detach().cpu().numpy()
    if hasattr(v, "to"):
        return v.cpu().numpy()
    return np.asarray(v)


def fuse_pass(sd: dict, drop_enc_q: bool) -> tuple[dict, dict]:
    """Pass 1: fuse weight_norm, rename gamma/beta, drop training EMA stats."""
    out: dict[str, np.ndarray] = {}
    stats = {"fused": 0, "skipped_enc_q": 0, "kept_enc_q": 0, "dropped_emas": 0}
    for key in sorted(sd.keys()):
        if drop_enc_q and key.startswith("enc_q."):
            stats["skipped_enc_q"] += 1
            continue
        if not drop_enc_q and key.startswith("enc_q."):
            stats["kept_enc_q"] += 1
        # Drop training-only RVQ EMA stats (only the codebook embed is needed)
        if key.startswith("quantizer.vq.layers.") and any(
            key.endswith(s) for s in (".inited", ".cluster_size", ".embed_avg")
        ):
            stats["dropped_emas"] += 1
            continue
        v = _to_np(sd[key])
        if key.endswith(".weight_g"):
            continue
        if key.endswith(".weight_v"):
            g = _to_np(sd[key[: -len(".weight_v")] + ".weight_g"])
            base = key[: -len(".weight_v")]
            out[base + ".weight"] = fuse_weight_norm(g, v)
            stats["fused"] += 1
            continue
        if key.endswith(".gamma"):
            out[key[: -len(".gamma")] + ".weight"] = v
            continue
        if key.endswith(".beta"):
            out[key[: -len(".beta")] + ".bias"] = v
            continue
        out[key] = v
    return out, stats


def remap_to_wrapper_layout(sanitized: dict) -> dict:
    """Pass 2: PT Conv1d/ConvTranspose1d shape → MLX wrapper layout."""
    remapped: dict[str, np.ndarray] = {}
    for key, value in sanitized.items():
        if value.ndim == 3 and key.endswith(".weight"):
            if is_conv_transpose_key(key):
                new_val = np.transpose(value, (1, 2, 0))
                new_key = key[: -len(".weight")] + ".conv_t.weight"
            else:
                new_val = np.transpose(value, (0, 2, 1))
                new_key = key[: -len(".weight")] + ".conv.weight"
            remapped[new_key] = new_val
            continue
        if value.ndim == 1 and key.endswith(".bias"):
            sib = key[: -len(".bias")] + ".weight"
            if sib in sanitized and sanitized[sib].ndim == 3:
                if is_conv_transpose_key(sib):
                    new_key = key[: -len(".bias")] + ".conv_t.bias"
                else:
                    new_key = key[: -len(".bias")] + ".conv.bias"
                remapped[new_key] = value
                continue
        remapped[key] = value
    return remapped


def rename_codebook(remapped: dict) -> dict:
    """MLX tree_flatten skips attribute names starting with '_'.  We renamed
    `_codebook` → `codebook` in the MLX model; same rename for the key."""
    k = "quantizer.vq.layers.0._codebook.embed"
    if k in remapped:
        remapped["quantizer.vq.layers.0.codebook.embed"] = remapped.pop(k)
    return remapped


def remap_discriminator(sanitized: dict) -> dict:
    """Discriminator-specific remap. The MLX MPD wraps Conv1d/Conv2d in
    Conv1dPT / Conv2dPT (the inference port uses these names).

    Discriminator structure:
        discriminators.{0..6}: DiscriminatorP — uses Conv2dPT
        discriminators.7: DiscriminatorS — uses Conv1dPT

    For Conv2d the PT weight is (out, in, kh, kw).  MLX `nn.Conv2d` takes
    (out, kh, kw, in), so transpose (0, 2, 3, 1).
    """
    out: dict[str, np.ndarray] = {}
    for key, value in sanitized.items():
        if value.ndim == 4 and key.endswith(".weight"):
            # Conv2d → transpose (out,in,kh,kw) → (out,kh,kw,in)
            new_val = np.transpose(value, (0, 2, 3, 1))
            new_key = key[: -len(".weight")] + ".conv.weight"
            out[new_key] = new_val
            continue
        if value.ndim == 3 and key.endswith(".weight"):
            # Conv1d → transpose (out,in,k) → (out,k,in)
            new_val = np.transpose(value, (0, 2, 1))
            new_key = key[: -len(".weight")] + ".conv.weight"
            out[new_key] = new_val
            continue
        if value.ndim == 1 and key.endswith(".bias"):
            sib = key[: -len(".bias")] + ".weight"
            if sib in sanitized and sanitized[sib].ndim in (3, 4):
                new_key = key[: -len(".bias")] + ".conv.bias"
                out[new_key] = value
                continue
        out[key] = value
    return out


def convert_g(in_path: str, out_path: str) -> None:
    import torch
    print(f"[G] Loading {in_path} ...", flush=True)
    ckpt = torch.load(in_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model") or ckpt.get("weight") or ckpt
    epoch = ckpt.get("epoch", "?")
    step = ckpt.get("step", "?")
    print(f"    epoch={epoch}  step={step}  state keys={len(sd)}")
    sanitized, stats = fuse_pass(sd, drop_enc_q=False)
    print(f"    fused={stats['fused']}  kept_enc_q={stats['kept_enc_q']}  "
          f"dropped_emas={stats['dropped_emas']}")
    remapped = remap_to_wrapper_layout(sanitized)
    remapped = rename_codebook(remapped)
    from safetensors.numpy import save_file
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_file({k: np.ascontiguousarray(v) for k, v in remapped.items()}, out_path)
    print(f"[G] saved {len(remapped)} tensors → {out_path}")


def convert_d(in_path: str, out_path: str) -> None:
    import torch
    print(f"[D] Loading {in_path} ...", flush=True)
    ckpt = torch.load(in_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model") or ckpt
    print(f"    state keys={len(sd)}")
    sanitized, stats = fuse_pass(sd, drop_enc_q=False)
    print(f"    fused={stats['fused']}")
    remapped = remap_discriminator(sanitized)
    from safetensors.numpy import save_file
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_file({k: np.ascontiguousarray(v) for k, v in remapped.items()}, out_path)
    print(f"[D] saved {len(remapped)} tensors → {out_path}")


# ============================================================================
#   REVERSE: MLX safetensors → PyTorch .pth  (for inference-fallback round-trip)
# ============================================================================
def export_g_pt(in_st: str, out_pt: str, ref_pth: str | None = None) -> None:
    """Save MLX safetensors back as a PyTorch-loadable .pth.

    Inverts: conv.weight (out,k,in) → (out,in,k)  and  conv_t.weight (out,k,in) → (in,out,k).
    Re-introduces weight_norm `weight_g`/`weight_v` IF ref_pth is supplied (so the
    inference pipeline can load with strict=True). Without ref_pth, we save plain
    `weight` keys and downstream PyTorch code must be patched to load with
    fuse_pretrained=False (or use the inference convert.py path).

    Note: this is the fallback path. For r6 we'll primarily use MLX checkpoints
    for resume, and use this only when we want to run inference via the existing
    PyTorch path.
    """
    from safetensors.numpy import load_file
    import torch
    st = load_file(in_st)
    out: dict = {}
    for k, v in st.items():
        # Reverse the wrapper renames
        if k.endswith(".conv_t.weight"):
            base = k[: -len(".conv_t.weight")]
            # MLX (out, k, in) → PT (in, out, k)
            arr = np.transpose(v, (2, 0, 1))
            out[base + ".weight"] = torch.from_numpy(arr)
            continue
        if k.endswith(".conv_t.bias"):
            base = k[: -len(".conv_t.bias")]
            out[base + ".bias"] = torch.from_numpy(v)
            continue
        if k.endswith(".conv.weight") and v.ndim == 3:
            base = k[: -len(".conv.weight")]
            arr = np.transpose(v, (0, 2, 1))
            out[base + ".weight"] = torch.from_numpy(arr)
            continue
        if k.endswith(".conv.weight") and v.ndim == 4:
            base = k[: -len(".conv.weight")]
            arr = np.transpose(v, (0, 3, 1, 2))
            out[base + ".weight"] = torch.from_numpy(arr)
            continue
        if k.endswith(".conv.bias"):
            base = k[: -len(".conv.bias")]
            out[base + ".bias"] = torch.from_numpy(v)
            continue
        # ChannelLayerNorm weight/bias → PT gamma/beta
        # (the inference convert.py does the opposite; we need to detect by where
        #  the key lives — modules in PT use gamma/beta only inside encoder norm
        #  layers, not LayerNorm everywhere)
        if k.endswith("norm_layers_1.0.weight") or k.endswith("norm_layers_2.0.weight"):
            base = k[: -len(".weight")]
            out[base + ".gamma"] = torch.from_numpy(v)
            continue
        if k.endswith("norm_layers_1.0.bias") or k.endswith("norm_layers_2.0.bias"):
            base = k[: -len(".bias")]
            out[base + ".beta"] = torch.from_numpy(v)
            continue
        # quantizer rename back
        if k == "quantizer.vq.layers.0.codebook.embed":
            out["quantizer.vq.layers.0._codebook.embed"] = torch.from_numpy(v)
            continue
        out[k] = torch.from_numpy(v)
    # Wrap as upstream loaders expect: {"model": ..., "epoch": ..., "step": 0}
    Path(out_pt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": out, "epoch": 0, "step": 0}, out_pt)
    print(f"Exported {len(out)} tensors → {out_pt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("g"); g.add_argument("--in", dest="i", required=True); g.add_argument("--out", dest="o", required=True)
    d = sub.add_parser("d"); d.add_argument("--in", dest="i", required=True); d.add_argument("--out", dest="o", required=True)
    e = sub.add_parser("export-g"); e.add_argument("--in", dest="i", required=True); e.add_argument("--out", dest="o", required=True)
    args = ap.parse_args()
    if args.cmd == "g": convert_g(args.i, args.o)
    elif args.cmd == "d": convert_d(args.i, args.o)
    elif args.cmd == "export-g": export_g_pt(args.i, args.o)
