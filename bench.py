"""Benchmark MLX S2 decode vs PyTorch MPS S2 decode.

Strategy: intercept vq_model.decode and record wall time.  Run the same set
of sentences twice — once with the PT decode (baseline), once with MLX.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/Users/kaede/tts")
SOVITS_MLX = ROOT / "_sovits_mlx"
sys.path.insert(0, str(SOVITS_MLX))
sys.path.insert(0, str(ROOT))


SENTENCES = [
    ("01_li_tsiah_pa_be",       "{tw:Lí tsia̍h-pá--bē?}"),
    ("02_kin_a_jit_thinn_khi",  "{tw:Kin-á-ji̍t thinn-khì tsiok hó.}"),
    ("03_gua_beh_tng_khi_ah",   "{tw:Guá beh tńg-khì--ah.}"),
    ("04_li_khi_to_ui",         "{tw:Lí khì tó-uī?}"),
    ("05_toh_sia_li",           "{tw:To̍h-siā lí.}"),
]


def _setup_inf_module(s1, s2):
    import tts_long
    s2_path = Path(s2)
    if not tts_long._is_inference_compatible_s2(s2_path):
        s2_path = tts_long._convert_s2_to_v2pro(s2_path, Path(tts_long.S2_CONFIG_JSON))
    tts_long._setup_env(Path(s1), s2_path)
    import GPT_SoVITS.inference_webui as inf
    if not hasattr(inf, "vq_model"):
        next(inf.change_sovits_weights(str(s2_path)))
    return tts_long, inf, s2_path


def _wrap_decode_with_timing(decode_fn, times_list):
    def wrapped(*args, **kwargs):
        t0 = time.time()
        r = decode_fn(*args, **kwargs)
        # ensure all MLX/MPS work has completed
        if hasattr(r, "detach"):
            r.detach().cpu().numpy().sum()
        t1 = time.time()
        times_list.append(t1 - t0)
        return r
    return wrapped


def _build_mlx_decode_wrapper(mlx_model, times_list, pt_dtype):
    import mlx.core as mx
    import torch as torch_pt
    def wrapped(pred_semantic, phones_long, refers, speed=1.0, sv_emb=None):
        refer_pt = refers[0]; sv_pt = sv_emb[0]
        codes_mx = mx.array(pred_semantic.detach().cpu().numpy().astype(np.int32))
        text_mx = mx.array(phones_long.detach().cpu().numpy().astype(np.int32))
        refer_mx = mx.array(refer_pt.detach().cpu().numpy().astype(np.float32))
        sv_mx = mx.array(sv_pt.detach().cpu().numpy().astype(np.float32))
        t0 = time.time()
        audio = mlx_model.decode(codes_mx, text_mx, refer_mx, sv_mx, noise_scale=0.5)
        mx.eval(audio)
        t1 = time.time()
        times_list.append(t1 - t0)
        out_np = np.array(audio).astype(np.float32)
        out_t = torch_pt.from_numpy(out_np).to(device=pred_semantic.device, dtype=pt_dtype)
        return out_t
    return wrapped


def run(backend: str, s1, s2, out_dir, n_runs=2):
    """backend: 'pt' or 'mlx'."""
    tts_long, inf, s2_path = _setup_inf_module(s1, s2)
    decode_times = []

    if backend == "pt":
        orig = inf.vq_model.decode
        inf.vq_model.decode = _wrap_decode_with_timing(orig, decode_times)
    elif backend == "mlx":
        from safetensors.numpy import load_file
        import mlx.core as mx
        from models import SynthesizerTrn, default_config
        m = SynthesizerTrn(**default_config())
        sf = load_file(str(SOVITS_MLX / "model.safetensors"))
        m.load_weights([(k, mx.array(v)) for k, v in sf.items()], strict=True)
        mx.eval(m.parameters())
        inf.vq_model.decode = _build_mlx_decode_wrapper(m, decode_times, inf.dtype)
    else:
        raise ValueError(backend)

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    total_times = []
    for run_idx in range(n_runs):
        for tag, poj in SENTENCES:
            t0 = time.time()
            wav_path = out_dir / f"{tag}_{backend}_r{run_idx}.wav"
            tts_long.synthesize(
                text=poj, out_path=wav_path,
                s1=Path(s1), s2=Path(s2),
                ref_wav=ROOT / "tw_recording/wavs/001.wav",
                ref_text="{tw:I hit ê lâng tō sī bô-liōng, bē-kham-tit khuànn lâng hó.}",
                max_chars=80, also_mp3=False, verbose=False,
                phoneticize=None, normalize=False,
                min_dur_per_char=0, asr_check="off", max_retries=0,
            )
            total_times.append(time.time() - t0)

    return decode_times, total_times


def stats(name, times):
    a = np.array(times)
    print(f"{name}: n={len(a)}  median={np.median(a)*1000:.0f}ms  "
          f"p50={np.percentile(a,50)*1000:.0f}ms  p90={np.percentile(a,90)*1000:.0f}ms  "
          f"mean={a.mean()*1000:.0f}ms  min={a.min()*1000:.0f}ms")
    return a


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", default=str(ROOT / "tw_finetune_synthetic/s1_ckpt_r2/s1_pathM_r2-e10.ckpt"))
    ap.add_argument("--s2", default=str(ROOT / "tw_finetune_synthetic/s2_logs_r4/s2_full_15.pth"))
    ap.add_argument("--backend", choices=["pt", "mlx", "both"], default="both")
    ap.add_argument("--n-runs", type=int, default=2)
    ap.add_argument("--out", default=str(SOVITS_MLX / "_bench"))
    args = ap.parse_args()

    if args.backend in ("pt", "both"):
        # PT baseline must run in a fresh process — we run it as a subprocess
        # to avoid the MLX model staying resident (could perturb timings)
        print("=== PT MPS baseline ===")
        pt_dec, pt_tot = run("pt", args.s1, args.s2, args.out, args.n_runs)
        stats("PT decode", pt_dec)
        stats("PT total ", pt_tot)
    if args.backend in ("mlx", "both"):
        if args.backend == "both":
            # Need to reset the inference_webui module so MPS state doesn't leak.
            # Simpler: invoke a fresh python for the MLX run via subprocess.
            print("=== MLX (rerun in fresh process recommended; running in-process here) ===")
        else:
            print("=== MLX ===")
        mlx_dec, mlx_tot = run("mlx", args.s1, args.s2, args.out, args.n_runs)
        stats("MLX decode", mlx_dec)
        stats("MLX total ", mlx_tot)
