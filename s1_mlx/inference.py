"""End-to-end MLX-based S1 inference.

Two entry points:

1. `build_mlx_s1(ckpt_path, fp32=True)` — load weights into an MLX T2SModel.
2. `make_mlx_infer_panel(mlx_model, device, dtype)` — return a function with
   the exact same signature as PT's `t2s_model.model.infer_panel` that
   delegates to MLX. Use it to monkey-patch the PyTorch inference path.

Plus a CLI:
    python -m s1_mlx.inference --text "..." --out test.wav
which is the full MLX-S1 + MLX-S2 stack, mirroring the existing
~/tts/_sovits_mlx/inference.py wrapper for S2.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import mlx.core as mx

ROOT = Path("/Users/kaede/tts")
SOVITS_MLX = ROOT / "_sovits_mlx"
S1_MLX_DIR = SOVITS_MLX / "s1_mlx"

# Make local module importable when run as a script or as a package
sys.path.insert(0, str(S1_MLX_DIR))
try:
    from .model import T2SModel
    from .sampling import sample, multinomial_sample_one, logits_to_probs
except ImportError:
    from model import T2SModel
    from sampling import sample, multinomial_sample_one, logits_to_probs


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_mlx_s1(
    safetensors_path: str | os.PathLike,
    config_json: str | os.PathLike | None = None,
    fp32: bool = True,
) -> T2SModel:
    """Construct an MLX T2SModel and load weights from a safetensors file.

    `config_json` defaults to the sibling .config.json (as written by convert.py).
    `fp32=True` upcasts fp16 ckpt weights to fp32 on load (better numerical
    match with PT default).
    """
    safetensors_path = Path(safetensors_path)
    if config_json is None:
        config_json = safetensors_path.with_suffix(".config.json")
    cfg = json.loads(Path(config_json).read_text())
    m = T2SModel(cfg)
    from safetensors.numpy import load_file
    sd = load_file(str(safetensors_path))
    if fp32:
        sd = {k: v.astype(np.float32) for k, v in sd.items()}
    m.load_weights([(k, mx.array(v)) for k, v in sd.items()], strict=True)
    mx.eval(m.parameters())
    return m


# ---------------------------------------------------------------------------
# `infer_panel` drop-in replacement
# ---------------------------------------------------------------------------
def make_mlx_infer_panel(mlx_model: T2SModel, pt_device, pt_dtype, max_tokens: int = 1500):
    """Return a function callable as PT's `t2s_model.model.infer_panel`.

    The PT call site is:
        pred_semantic, idx = t2s_model.model.infer_panel(
            all_phoneme_ids,   # (1, T_t) long
            all_phoneme_len,   # (1,) long  — unused on the single-batch path
            None or prompt,    # (1, T_p) long or None (ref-free)
            bert,              # (1, 1024, T_t) float
            top_k=...,
            top_p=...,
            temperature=...,
            early_stop_num=...,
        )

    Returns: (y, idx) where
      y   — (1, T_p + T_gen) torch long (so caller's `y[:, -idx:].unsqueeze(0)`
            extracts just the generated tokens, matching PT exactly).
      idx — number of generated steps (int), or n-1 if EOS was reached after step n.
    """
    import torch

    def _infer_panel(
        all_phoneme_ids: torch.Tensor,
        all_phoneme_len: torch.Tensor,
        prompts: Optional[torch.Tensor],
        bert_feature: torch.Tensor,
        top_k: int = -100,
        top_p: float = 1.0,
        temperature: float = 1.0,
        early_stop_num: int = -1,
        repetition_penalty: float = 1.35,
        **kwargs,
    ):
        # ---- convert to MLX ----
        phones_mx = mx.array(all_phoneme_ids.detach().cpu().numpy().astype(np.int32))
        bert_mx = mx.array(bert_feature.detach().cpu().numpy().astype(np.float32))
        if prompts is not None:
            prompts_mx = mx.array(prompts.detach().cpu().numpy().astype(np.int32))
            prefix_len = prompts.shape[1]
        else:
            prompts_mx = None
            prefix_len = 0

        # ---- prefill ----
        text_emb = mlx_model.make_text_prefix(phones_mx, bert_mx)
        hidden, kcs, vcs, _ = mlx_model.prefill(text_emb, prompts_mx)

        EOS = mlx_model.EOS

        # ---- generate ----
        # y holds the running audio sequence (including the prompt prefix).
        y = prompts_mx if prompts_mx is not None else mx.zeros((1, 0), dtype=mx.int32)
        y_len_init = prefix_len   # absolute position offset for ar_audio_position
        gen_tokens: List[int] = []  # newly generated tokens (post-prompt)

        idx_out = max_tokens - 1
        for step in range(max_tokens):
            logits = mlx_model.logits(hidden)            # (1, V)
            # PT: "if idx < 11: logits = logits[:, :-1]" — forbid EOS in first 11 steps
            if step < 11:
                logits = logits[:, :-1]
            # sample
            idx_next, _ = sample(
                logits,
                previous_tokens=y if y.shape[1] > 0 else None,
                top_k=top_k if top_k and top_k > 0 else None,
                top_p=top_p if (top_p is not None and top_p < 1.0) else None,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
            )
            tok = int(idx_next[0, 0])
            # also check argmax → if argmax==EOS we also stop (matches PT)
            argmax_tok = int(mx.argmax(logits, axis=-1)[0])

            # append before stopping checks so y length is consistent with PT
            y = mx.concatenate([y, idx_next], axis=1)
            gen_tokens.append(tok)

            stop = False
            if tok == EOS or argmax_tok == EOS:
                # PT trims the last (EOS) token off: y = y[:, :-1]
                y = y[:, :-1]
                gen_tokens.pop()
                idx_out = step
                stop = True
            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                idx_out = step
                stop = True
            if step == max_tokens - 1:
                idx_out = step
                stop = True
            if stop:
                if y.shape[1] == 0:
                    y = mx.zeros((1, 1), dtype=mx.int32)
                    print("[mlx-s1] bad zero prediction")
                break

            # ---- prepare next-step hidden state ----
            pos_idx = y_len_init + step
            last_tok = idx_next                          # (1, 1)
            hidden, kcs, vcs = mlx_model.decode_step(last_tok, pos_idx, kcs, vcs)
            mx.eval(hidden)   # keep latency bounded; small per-step

        mx.eval(y)

        # ---- back to torch ----
        y_np = np.array(y).astype(np.int64)
        y_pt = torch.from_numpy(y_np).to(pt_device)
        return y_pt, idx_out

    return _infer_panel


# ---------------------------------------------------------------------------
# CLI: full MLX S1 + MLX S2 synthesis
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default=str(S1_MLX_DIR / "test_001.mp3"))
    ap.add_argument("--ref-wav", default=str(ROOT / "tw_recording" / "wavs" / "001.wav"))
    ap.add_argument("--ref-text",
                    default="{tw:I hit ê lâng tō sī bô-liōng, bē-kham-tit khuànn lâng hó.}")
    ap.add_argument("--s1-ckpt", default=str(ROOT / "_s1_trilingual/arm_A_e15_trilingual.ckpt"))
    ap.add_argument("--s1-safetensors", default=str(S1_MLX_DIR / "s1.safetensors"))
    ap.add_argument("--s2", default=str(ROOT / "tw_finetune_synthetic/s2_logs_r4/s2_full_15.pth"))
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build MLX S1
    print("[mlx-s1] loading …", flush=True)
    s1 = build_mlx_s1(args.s1_safetensors)
    print(f"[mlx-s1] built: hidden={s1.hidden_dim} heads={s1.num_heads} layers={s1.num_layers}")

    # Build MLX S2 — duplicate the parent's _build_mlx_model but with
    # strict=False (the safetensors lacks training-only enc_q.* keys, which
    # the latest SynthesizerTrn declares for the training path).
    sys.path.insert(0, str(SOVITS_MLX))
    from safetensors.numpy import load_file as _load_sf
    from models import SynthesizerTrn, default_config
    print("[mlx-s2] building model …", flush=True)
    s2 = SynthesizerTrn(**default_config())
    sf = _load_sf(str(SOVITS_MLX / "model.safetensors"))
    s2.load_weights([(k, mx.array(v)) for k, v in sf.items()], strict=False)
    mx.eval(s2.parameters())
    print(f"[mlx-s2]   loaded {len(sf)} tensors (strict=False)", flush=True)

    # Pull the wrapper helper from parent inference.py via importlib.
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("_sovits_mlx_inference", str(SOVITS_MLX / "inference.py"))
    _parent_inf = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_parent_inf)
    _mlx_decode_wrapper = _parent_inf._mlx_decode_wrapper

    # Set up the existing PT pipeline (env + inference_webui import)
    sys.path.insert(0, str(ROOT))
    import tts_long
    s2_path = Path(args.s2)
    if not tts_long._is_inference_compatible_s2(s2_path):
        s2_path = tts_long._convert_s2_to_v2pro(s2_path, Path(tts_long.S2_CONFIG_JSON))
    tts_long._setup_env(Path(args.s1_ckpt), s2_path)
    import GPT_SoVITS.inference_webui as inf

    if not hasattr(inf, "vq_model"):
        next(inf.change_sovits_weights(str(s2_path)))

    # Hook MLX S1 in place of PT S1
    pt_dtype = inf.dtype
    pt_device = inf.device
    inf.t2s_model.model.infer_panel = make_mlx_infer_panel(s1, pt_device, pt_dtype)
    inf.vq_model.decode = _mlx_decode_wrapper(s2, pt_dtype)
    print(f"[mlx] hooked S1 + S2 onto inference_webui (device={pt_device}, dtype={pt_dtype})")

    # Synthesize via tts_long
    wav_path = out_path.with_suffix(".wav")
    t0 = time.time()
    tts_long.synthesize(
        text=args.text,
        out_path=wav_path,
        s1=Path(args.s1_ckpt),
        s2=Path(args.s2),
        ref_wav=Path(args.ref_wav),
        ref_text=args.ref_text,
        max_chars=80,
        also_mp3=True,
        verbose=True,
        phoneticize=None,
        normalize=False,
        min_dur_per_char=0,
        asr_check="off",
        max_retries=0,
    )
    print(f"\n[done] elapsed={time.time()-t0:.1f}s  wav={wav_path}")


if __name__ == "__main__":
    main()
