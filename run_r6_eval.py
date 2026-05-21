"""Phase E — eval r6 MLX checkpoints with the same Breeze 5-sentence + sandhi v1.

For each epoch ckpt (e17, e18, e19, e20, e21):
  1. Convert MLX safetensors → PyTorch .pth via convert_full.py export-g
  2. Run the existing GPT-SoVITS PyTorch inference path on the 5 test sentences
     (with sandhi v1 preprocessing — same as run_r5_eval_sandhi.py)
  3. Breeze-ASR-26 → CER
  4. Compare against r5 e16 + sandhi v1 and r4 e15 + sandhi v1 (4.44%)

The MLX→PT conversion lets us reuse the existing battle-tested inference
pipeline rather than build a new one. The user explicitly wanted ckpt
round-trip in Phase 3.4 anyway; this is its first use.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path


ROOT = Path("/Users/kaede/tts")
MLX_DIR = ROOT / "_sovits_mlx"
TRAIN_DIR = ROOT / "_sovits_mlx_train/r6"
EVAL_DIR = ROOT / "_sovits_mlx_train/r6_eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

R5_LOGS = ROOT / "tw_finetune_synthetic/s2_logs_r5"  # only for r5 baseline comparison
S1 = ROOT / "tw_finetune_synthetic/s1_ckpt_r2/s1_pathM_r2-e10.ckpt"
REF_WAV = ROOT / "tw_recording/wavs/001.wav"
REF_TEXT = "{tw:I hit ê lâng tō sī bô-liōng, bē-kham-tit khuànn lâng hó.}"

TESTS = [
    (1, "Lí tsia̍h-pá--bē?",                "你吃飽了沒"),
    (2, "Kin-á-ji̍t thinn-khì tsiok hó.",   "今天天氣很好"),
    (3, "Guá beh tńg-khì--ah.",             "我要回家了"),
    (4, "Lí khì tó-uī?",                    "你去哪裡"),
    (5, "To̍h-siā lí.",                     "謝謝你"),
]

EPOCHS = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else "17,18,19,20,21")]
PYTHON = "/Users/kaede/tts/GPT-SoVITS/.venv/bin/python"

# --- sandhi v1 ---
sys.path.insert(0, str(ROOT / "GPT-SoVITS"))
from sandhi_preprocessor import apply_sandhi
SANDHI_KW = dict(t2_target=3, t5_target=7, citation_t3_remap=False)

# --- CER ---
PUNCT_RE = re.compile(r"[\s\.,!?;:'\"()\[\]{}…—\-。，！？；：、「」『』（）《》【】〔〕　]+")
def norm(s):
    if s is None: return ""
    s = unicodedata.normalize("NFKC", str(s)).strip()
    return PUNCT_RE.sub("", s)

def cer(ref, hyp):
    import jiwer
    r, h = norm(ref), norm(hyp)
    if not r: return 0.0 if not h else 1.0
    return jiwer.cer(" ".join(r), " ".join(h))


def convert_mlx_to_pt(mlx_g_path: Path, out_pt: Path) -> Path:
    """Drive convert_full.py export-g as a subprocess (it needs torch import)."""
    if out_pt.exists():
        print(f"  cached {out_pt}", flush=True)
        return out_pt
    cmd = [
        PYTHON, str(MLX_DIR / "convert_full.py"), "export-g",
        "--in", str(mlx_g_path), "--out", str(out_pt),
    ]
    print(f"  converting {mlx_g_path.name} → {out_pt.name} ...", flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"    converted in {time.time()-t0:.1f}s", flush=True)
    return out_pt


sys.path.insert(0, str(ROOT))
import tts_long


_asr_singleton = None
def make_asr():
    import torch
    from transformers import pipeline
    print("Loading Breeze-ASR-26 ...", flush=True)
    t0 = time.time()
    asr = pipeline("automatic-speech-recognition",
                   model="MediaTek-Research/Breeze-ASR-26",
                   device="cpu", torch_dtype=torch.float32)
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
    return asr


def asr_transcribe(wav_path):
    global _asr_singleton
    if _asr_singleton is None:
        _asr_singleton = make_asr()
    out = _asr_singleton(str(wav_path), generate_kwargs={"language": "zh", "task": "transcribe"})
    return out["text"]


def eval_ckpt(epoch: int):
    mlx_g = TRAIN_DIR / f"e{epoch}_g.safetensors"
    if not mlx_g.exists():
        print(f"[e{epoch}] missing MLX ckpt {mlx_g}", flush=True)
        return None
    pt_g = TRAIN_DIR / f"e{epoch}_pt_g.pth"
    convert_mlx_to_pt(mlx_g, pt_g)

    ckpt_out = EVAL_DIR / f"e{epoch}"
    ckpt_out.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, poj, zh_ref in TESTS:
        # sandhi v1 preprocessing
        poj_sandhi = apply_sandhi(poj, **SANDHI_KW)
        wav = ckpt_out / f"{idx:02d}.wav"
        mp3 = wav.with_suffix(".mp3")
        if mp3.exists():
            print(f"[e{epoch}/{idx}] cached", flush=True)
        else:
            t0 = time.time()
            try:
                tts_long.synthesize(
                    text="{tw:" + poj_sandhi + "}",
                    out_path=wav,
                    s1=S1, s2=pt_g,
                    ref_wav=REF_WAV, ref_text=REF_TEXT,
                    max_chars=80, also_mp3=True, verbose=False,
                    phoneticize=None, normalize=False,
                    min_dur_per_char=0, asr_check="off", max_retries=0,
                )
                print(f"[e{epoch}/{idx}] synth ok ({time.time()-t0:.1f}s)", flush=True)
            except Exception as e:
                print(f"[e{epoch}/{idx}] synth FAIL: {type(e).__name__}: {e}", flush=True)
                rows.append({"idx": idx, "poj": poj, "poj_sandhi": poj_sandhi,
                             "ref": zh_ref, "hyp": None, "cer": None,
                             "wav": str(wav), "error": str(e)})
                continue
        try:
            hyp = asr_transcribe(mp3 if mp3.exists() else wav)
        except Exception as e:
            print(f"[e{epoch}/{idx}] asr FAIL: {e}", flush=True)
            hyp = None
        c = cer(zh_ref, hyp) if hyp is not None else None
        cer_pct = f"{c*100:.1f}%" if c is not None else "n/a"
        print(f"[e{epoch}/{idx}] poj={poj_sandhi!r}\n        ref={zh_ref!r}\n        hyp={hyp!r}\n        CER={cer_pct}", flush=True)
        rows.append({"idx": idx, "poj": poj, "poj_sandhi": poj_sandhi,
                     "ref": zh_ref, "hyp": hyp, "cer": c,
                     "wav": str(wav), "mp3": str(mp3)})

    valid = [r["cer"] for r in rows if r["cer"] is not None]
    mean = sum(valid) / len(valid) if valid else None
    summary = {"epoch": epoch, "mean_cer": mean, "n_valid": len(valid), "rows": rows}
    (ckpt_out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    pct = f"{mean*100:.2f}%" if mean is not None else "n/a"
    print(f"[e{epoch}] mean CER = {pct}  ({len(valid)}/{len(TESTS)} valid)", flush=True)
    return summary


all_results = {}
t_start = time.time()
for ep in EPOCHS:
    print(f"\n{'='*70}\n=== r6 e{ep} (sandhi v1) ===\n{'='*70}", flush=True)
    all_results[f"e{ep}"] = eval_ckpt(ep)
print(f"\n=== Total: {time.time()-t_start:.1f}s ===", flush=True)

out_json = EVAL_DIR / "all_results.json"
out_json.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

# Summary table
print(f"\n{'='*70}\n=== R6 SUMMARY (sandhi v1, c_kl=1.0 restored) ===\n{'='*70}", flush=True)
print(f"{'ckpt':<12} {'mean_CER':>10}  {'verdict'}", flush=True)
print("-" * 60, flush=True)
print(f"{'r4_e15+sandhi':<12} {'4.44%':>10}  baseline (current best)", flush=True)
print(f"{'r5_e16+sandhi':<12} {'12.76%':>10}  r5 e17 was 20.23% (KL was disabled)", flush=True)
for ep_name, r in all_results.items():
    if r is None:
        print(f"{ep_name:<12} {'MISSING':>10}"); continue
    m = r["mean_cer"]
    if m is None: verdict = "n/a"
    elif m < 0.0444: verdict = "SUPER WIN (beats r4+sandhi)"
    elif m < 0.05:   verdict = "WIN (KL restoration helped)"
    elif m < 0.08:   verdict = "ok"
    else:            verdict = "regression"
    print(f"{ep_name:<12} {m*100:>9.2f}%  {verdict}", flush=True)
