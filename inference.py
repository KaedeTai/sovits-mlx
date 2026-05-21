"""End-to-end inference: PyTorch S1 + (PyTorch) SV + MLX S2.

Approach
--------
We re-use the existing GPT-SoVITS PyTorch pipeline up through pred_semantic, ref-spec,
and sv_emb computation (everything that happens *before* `vq_model.decode` in
inference_webui.get_tts_wav).  Then we replace that one call with the MLX model.

We do this by hooking `vq_model.decode` after model load.

Run:
    python inference.py --text "{tw:Lí tsia̍h-pá--bē?}" \
                        --out /Users/kaede/tts/_sovits_mlx/test_001.mp3
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import mlx.core as mx

ROOT = Path("/Users/kaede/tts")
SOVITS_MLX = ROOT / "_sovits_mlx"

# Make local module importable
sys.path.insert(0, str(SOVITS_MLX))
sys.path.insert(0, str(ROOT))


def _build_mlx_model():
    from safetensors.numpy import load_file
    from models import SynthesizerTrn, default_config

    print("[mlx] building model …", flush=True)
    m = SynthesizerTrn(**default_config())
    sf = load_file(str(SOVITS_MLX / "model.safetensors"))
    m.load_weights([(k, mx.array(v)) for k, v in sf.items()], strict=True)
    mx.eval(m.parameters())
    print(f"[mlx]   loaded {len(sf)} tensors", flush=True)
    return m


def _mlx_decode_wrapper(mlx_model, pt_dtype):
    """Return a function with the same signature as vq_model.decode for v2pro path."""
    import torch

    def _decode(pred_semantic, phones_long, refers, speed=1.0, sv_emb=None):
        # pred_semantic: (1, 1, T_code) torch long
        # phones_long  : (1, T_t) torch long
        # refers       : list of (1, 1025, T_ref) torch tensors
        # sv_emb       : list of (1, 20480) torch tensors
        assert sv_emb is not None and len(refers) == len(sv_emb), \
            "v2pro path requires sv_emb"

        # Concatenate refers into a list of MLX arrays and combine via mean
        # of ge (matching PT decode which mean-aggregates the ge across refer
        # tensors).  For simplicity we just take the first ref.
        if len(refers) > 1:
            print(f"[mlx-dec] WARNING: {len(refers)} refers, using first only", flush=True)
        refer_pt = refers[0]
        sv_pt = sv_emb[0]

        # Codes: PT shape (1, 1, T) → MLX expects (n_q=1, B=1, T)
        codes_np = pred_semantic.detach().cpu().numpy().astype(np.int32)
        # Drop the singleton n_q dim if present
        if codes_np.shape[0] == 1 and codes_np.shape[1] == 1:
            # Original signature is .decode(codes) where codes is (n_q, B, T).
            # pred_semantic is (1, 1, T) — first dim is "n_q-ish, B-ish".  In the
            # PT _model.decode path, codes are passed straight to quantizer.decode
            # which does `for i, indices in enumerate(q_indices):` so q_indices
            # is iterated over dim 0.  pred_semantic shape (1,1,T) → iterate over 1
            # tensor of shape (1, T) → embedding lookup → ok.
            pass

        codes_mx = mx.array(codes_np)
        text_mx = mx.array(phones_long.detach().cpu().numpy().astype(np.int32))
        refer_mx = mx.array(refer_pt.detach().cpu().numpy().astype(np.float32))
        sv_mx = mx.array(sv_pt.detach().cpu().numpy().astype(np.float32))

        # Run MLX decode
        t0 = time.time()
        audio = mlx_model.decode(codes_mx, text_mx, refer_mx, sv_mx,
                                 noise_scale=0.5)
        mx.eval(audio)
        t1 = time.time()
        print(f"[mlx-dec] decode time {t1 - t0:.2f}s, output shape {tuple(audio.shape)}",
              flush=True)

        # Convert back to torch.  PT decode returns a tensor `o` of shape
        # (1, 1, T_wav); caller does `[0][0]` to get (T_wav,).  Mirror this exactly.
        out_np = np.array(audio).astype(np.float32)
        out_t = torch.from_numpy(out_np)
        # Move to same device/dtype as the rest of inference_webui expects
        out_t = out_t.to(device=pred_semantic.device, dtype=pt_dtype)
        return out_t

    return _decode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default=str(SOVITS_MLX / "test_001.mp3"))
    ap.add_argument("--ref-wav", default=str(ROOT / "tw_recording" / "wavs" / "001.wav"))
    ap.add_argument("--ref-text",
                    default="{tw:I hit ê lâng tō sī bô-liōng, bē-kham-tit khuànn lâng hó.}")
    ap.add_argument("--s1", default=str(ROOT / "tw_finetune_synthetic/s1_ckpt_r2/s1_pathM_r2-e10.ckpt"))
    ap.add_argument("--s2", default=str(ROOT / "tw_finetune_synthetic/s2_logs_r4/s2_full_15.pth"))
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build MLX model first
    mlx_model = _build_mlx_model()

    # Set up env, pre-convert S2 ckpt to v2Pro inference format, import inference_webui
    import tts_long
    s2_path = Path(args.s2)
    if not tts_long._is_inference_compatible_s2(s2_path):
        s2_path = tts_long._convert_s2_to_v2pro(s2_path, Path(tts_long.S2_CONFIG_JSON))
        print(f"[setup] using converted: {s2_path}", flush=True)
    tts_long._setup_env(Path(args.s1), s2_path)
    import GPT_SoVITS.inference_webui as inf

    if not hasattr(inf, "vq_model"):
        # Force model build (module-level next() may have failed silently before
        # our env was applied).
        next(inf.change_sovits_weights(str(s2_path)))

    pt_dtype = inf.dtype
    wrapped_decode = _mlx_decode_wrapper(mlx_model, pt_dtype)
    inf.vq_model.decode = wrapped_decode

    # Now do real synthesis
    wav_path = out_path.with_suffix(".wav")
    t0 = time.time()
    tts_long.synthesize(
        text=args.text,
        out_path=wav_path,
        s1=Path(args.s1),
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
    dur = time.time() - t0
    mp3_path = wav_path.with_suffix(".mp3")
    print(f"\n[done] wav={wav_path.exists()} mp3={mp3_path.exists()} elapsed={dur:.1f}s")


if __name__ == "__main__":
    main()
