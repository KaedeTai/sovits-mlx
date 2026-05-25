"""S1 (and full-pipeline) speed benchmark: PyTorch MPS vs MLX.

Runs the same 5 canonical TW sentences through both backends, measuring
per-call latency for S1 (`infer_panel`), S2 (`vq_model.decode`), and the
overall `synthesize` call.

Outputs JSON to BENCH_OUT, also prints a human-readable summary.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable, List

import numpy as np

ROOT = Path("/Users/kaede/tts")
SOVITS_MLX = ROOT / "_sovits_mlx"
S1_MLX_DIR = SOVITS_MLX / "s1_mlx"
BENCH_OUT = S1_MLX_DIR / "bench_results.json"

# Hard-pin canonical eval set (same as run_samples.py)
SENTENCES = [
    ("tw_short_01",  "{tw:Lí tsia̍h-pá--bē?}"),
    ("tw_short_02",  "{tw:Kin-á-ji̍t thinn-khì tsiok hó.}"),
    ("tw_short_03",  "{tw:Guá beh tńg-khì--ah.}"),
    ("tw_medium_01", "{tw:Tsa-hng àm-sî guá kah pîng-iú khì tshī-tiûnn bé tshài.}"),
    ("tw_long_01",   "{tw:Sîng-jin ji̍p-tsia̍p lâi sîng-ji̍p siā-huē í-āu, guán tsiah liáu-kái sing-ua̍h pīng bô siūnn-tio̍h ê hiah-nī kán-tan.}"),
]
REF_WAV = ROOT / "tw_recording" / "wavs_norm" / "001.wav"
REF_TEXT = "{tw:I hit ê lâng tō sī bô-liōng, bē-kham-tit khuànn lâng hó.}"
S1_CKPT = ROOT / "_s1_trilingual" / "arm_A_e15_trilingual.ckpt"
S2_CKPT = ROOT / "tw_finetune_synthetic" / "s2_logs_r4" / "s2_full_15.pth"

WARMUP_ITERS = 1
MEASURE_ITERS = 3


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def setup_pipeline():
    """Initialise the PyTorch pipeline (always loaded; MLX patches over it)."""
    sys.path.insert(0, str(ROOT))
    import tts_long
    s2_path = Path(S2_CKPT)
    if not tts_long._is_inference_compatible_s2(s2_path):
        s2_path = tts_long._convert_s2_to_v2pro(s2_path, Path(tts_long.S2_CONFIG_JSON))
    tts_long._setup_env(S1_CKPT, s2_path)
    import GPT_SoVITS.inference_webui as inf
    if not hasattr(inf, "vq_model"):
        next(inf.change_sovits_weights(str(s2_path)))
    # NB: device/dtype is whatever the user's tts_long install defaults to.
    # On this Mac it lands at CPU+fp32 — see bench notes in BENCHMARK.md.
    # Moving to MPS in-place breaks a couple of cnhubert/SV preprocessing
    # paths in tts_long that don't follow inf.device; we leave it alone here.
    print(f"[setup] PT path uses device={inf.device}, dtype={inf.dtype}", flush=True)
    return tts_long, inf, s2_path


# ---------------------------------------------------------------------------
# Timing instrumentation
# ---------------------------------------------------------------------------
class Timer:
    """Captures per-call latencies via monkey-patching `infer_panel` and
    `vq_model.decode`. Resets each invocation."""

    def __init__(self):
        self.s1_times: List[float] = []
        self.s2_times: List[float] = []

    def wrap(self, inf):
        original_s1 = inf.t2s_model.model.infer_panel
        original_s2 = inf.vq_model.decode
        s1_times = self.s1_times
        s2_times = self.s2_times

        def s1_wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            out = original_s1(*args, **kwargs)
            s1_times.append(time.perf_counter() - t0)
            return out

        def s2_wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            out = original_s2(*args, **kwargs)
            s2_times.append(time.perf_counter() - t0)
            return out

        inf.t2s_model.model.infer_panel = s1_wrapped
        inf.vq_model.decode = s2_wrapped
        return original_s1, original_s2

    def reset(self):
        self.s1_times.clear()
        self.s2_times.clear()


# ---------------------------------------------------------------------------
# Run one pipeline (PT baseline or MLX) across the sentence set
# ---------------------------------------------------------------------------
def run_pipeline(name: str, tts_long_mod, inf, s2_path) -> dict:
    """Run all sentences `MEASURE_ITERS` times each, return a summary dict.

    Assumes `inf.t2s_model.model.infer_panel` and `inf.vq_model.decode` are
    already set up for the desired backend.
    """
    print(f"\n=== {name} ===", flush=True)
    timer = Timer()
    # We have to wrap *after* the backend is set, otherwise the original_s1
    # reference will be the original (un-replaced) PT version.
    timer.wrap(inf)

    per_sentence = {}
    for tag, text in SENTENCES:
        print(f"  [{tag}]", flush=True)
        # warmup
        for w in range(WARMUP_ITERS):
            timer.reset()
            wav = Path(f"/tmp/bench_{name}_{tag}_warm{w}.wav")
            tts_long_mod.synthesize(
                text=text, out_path=wav, s1=S1_CKPT, s2=S2_CKPT,
                ref_wav=REF_WAV, ref_text=REF_TEXT,
                max_chars=80, also_mp3=False, verbose=False,
                phoneticize=None, normalize=False, min_dur_per_char=0,
                asr_check="off", max_retries=0,
            )
            if wav.exists():
                wav.unlink()
        # measure
        s1s, s2s, tots = [], [], []
        for it in range(MEASURE_ITERS):
            timer.reset()
            wav = Path(f"/tmp/bench_{name}_{tag}_iter{it}.wav")
            t0 = time.perf_counter()
            tts_long_mod.synthesize(
                text=text, out_path=wav, s1=S1_CKPT, s2=S2_CKPT,
                ref_wav=REF_WAV, ref_text=REF_TEXT,
                max_chars=80, also_mp3=False, verbose=False,
                phoneticize=None, normalize=False, min_dur_per_char=0,
                asr_check="off", max_retries=0,
            )
            tot = time.perf_counter() - t0
            s1 = sum(timer.s1_times); s2 = sum(timer.s2_times)
            s1s.append(s1); s2s.append(s2); tots.append(tot)
            if wav.exists():
                wav.unlink()
            print(f"    iter {it}: s1={s1:.3f}s  s2={s2:.3f}s  total={tot:.3f}s", flush=True)
        per_sentence[tag] = dict(
            s1_mean=mean(s1s), s1_std=pstdev(s1s),
            s2_mean=mean(s2s), s2_std=pstdev(s2s),
            tot_mean=mean(tots), tot_std=pstdev(tots),
            s1_all=s1s, s2_all=s2s, tot_all=tots,
        )

    # aggregate across all sentences
    flat_s1 = [t for r in per_sentence.values() for t in r["s1_all"]]
    flat_s2 = [t for r in per_sentence.values() for t in r["s2_all"]]
    flat_tot = [t for r in per_sentence.values() for t in r["tot_all"]]
    return dict(
        per_sentence=per_sentence,
        s1_mean=mean(flat_s1), s1_std=pstdev(flat_s1),
        s2_mean=mean(flat_s2), s2_std=pstdev(flat_s2),
        tot_mean=mean(flat_tot), tot_std=pstdev(flat_tot),
        n_obs=len(flat_s1),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Building pipeline …", flush=True)
    tts_long_mod, inf, s2_path = setup_pipeline()

    # Save the original PT infer_panel & vq_model.decode so we can restore.
    orig_s1 = inf.t2s_model.model.infer_panel
    orig_s2 = inf.vq_model.decode

    # === Path 1: PyTorch baseline ===
    pt_summary = run_pipeline("PyTorch", tts_long_mod, inf, s2_path)
    # Restore originals (Timer.wrap stored references over them).
    inf.t2s_model.model.infer_panel = orig_s1
    inf.vq_model.decode = orig_s2

    # === Path 2: MLX S1 + MLX S2 ===
    # S1_MLX_DIR must come BEFORE SOVITS_MLX in sys.path so that
    # `import inference` resolves to s1_mlx/inference.py, not the parent.
    sys.path.insert(0, str(SOVITS_MLX))
    sys.path.insert(0, str(S1_MLX_DIR))
    import mlx.core as mx
    from model import T2SModel
    # Use an absolute file load to be safe even if SOVITS_MLX/inference.py
    # was imported earlier.
    import importlib.util as _ilu
    _s1inf_spec = _ilu.spec_from_file_location("_s1_mlx_inference",
                                                str(S1_MLX_DIR / "inference.py"))
    _s1inf = _ilu.module_from_spec(_s1inf_spec); _s1inf_spec.loader.exec_module(_s1inf)
    build_mlx_s1 = _s1inf.build_mlx_s1
    make_mlx_infer_panel = _s1inf.make_mlx_infer_panel
    from safetensors.numpy import load_file
    from models import SynthesizerTrn, default_config

    print("\nLoading MLX S1 …", flush=True)
    s1_mlx = build_mlx_s1(str(S1_MLX_DIR / "s1.safetensors"))
    print("Loading MLX S2 …", flush=True)
    s2_mlx = SynthesizerTrn(**default_config())
    sf = load_file(str(SOVITS_MLX / "model.safetensors"))
    s2_mlx.load_weights([(k, mx.array(v)) for k, v in sf.items()], strict=False)
    mx.eval(s2_mlx.parameters())

    # Hook them in (similar to inference.py)
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("_sovits_mlx_inference",
                                          str(SOVITS_MLX / "inference.py"))
    _parent_inf = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_parent_inf)
    inf.t2s_model.model.infer_panel = make_mlx_infer_panel(s1_mlx, inf.device, inf.dtype)
    inf.vq_model.decode = _parent_inf._mlx_decode_wrapper(s2_mlx, inf.dtype)

    mlx_summary = run_pipeline("MLX", tts_long_mod, inf, s2_path)

    # === Report ===
    print("\n" + "=" * 70)
    print(f"{'Metric':<26} {'PyTorch':>14} {'MLX':>14} {'Speedup':>10}")
    print("=" * 70)
    for m in ("s1_mean", "s2_mean", "tot_mean"):
        pt = pt_summary[m]; mx_ = mlx_summary[m]
        sp = pt / mx_ if mx_ > 0 else float("inf")
        std_key = m.replace("_mean", "_std")
        pt_std = pt_summary[std_key]; mx_std = mlx_summary[std_key]
        label = {"s1_mean": "S1 (infer_panel)",
                 "s2_mean": "S2 (vq_model.decode)",
                 "tot_mean": "Total pipeline"}[m]
        print(f"{label:<26} {pt:>8.3f}±{pt_std:.3f}s  "
              f"{mx_:>8.3f}±{mx_std:.3f}s  {sp:>8.2f}×")
    print("=" * 70)

    BENCH_OUT.write_text(json.dumps(
        dict(pytorch=pt_summary, mlx=mlx_summary,
             warmup_iters=WARMUP_ITERS, measure_iters=MEASURE_ITERS,
             sentences=SENTENCES), indent=2))
    print(f"\nWrote raw numbers to {BENCH_OUT}")


if __name__ == "__main__":
    main()
