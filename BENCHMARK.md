# GPT-SoVITS S2 (v2ProTw) — PyTorch MPS vs MLX benchmark

Apple Silicon, r4 e15 checkpoint (`s2_logs_r4/s2_full_15.pth`).
5 Taiwanese sentences × 2 runs = 10 samples per backend.
Same `tts_long.synthesize` pipeline (PyTorch S1 + spec + SV → MLX decode).

## Headline numbers (ms)

| Stage                          | Backend  | median | p50  | p90  | mean | min  |
|--------------------------------|----------|--------|------|------|------|------|
| S2 `vq_model.decode` only      | PT MPS   | 98     | 98   | 156  | 114  | 77   |
| S2 `vq_model.decode` only      | **MLX**  | **32** | 32   | 82   | 41   | 20   |
| End-to-end (S1 + S2 + post)    | PT MPS   | 898    | 898  | 1173 | 1024 | 794  |
| End-to-end (S1 + S2 + post)    | **MLX**  | 868    | 868  | 1050 | 977  | 744  |

## Speedup

- **S2 decode**: **~3.1× faster** on MLX (98ms → 32ms median).
- **End-to-end**: ~1.05× (S1 autoregressive in PyTorch dominates total time;
  S2 was already only ~10% of the pipeline). Porting S1 to MLX is the next big lever.

## Real-time factor (decode-only)

For a typical sentence (`01_li_tsiah_pa_be`, 1.58s of audio at 32kHz):

- PT MPS decode: 98ms / 1580ms = **RTF 0.062** (16× real time).
- MLX decode:   32ms / 1580ms = **RTF 0.020** (49× real time).

## Numerical correctness

- Per-module max-abs diff vs PyTorch reference (random inputs):
  - `ge_to512`:        1.4e-7  (essentially exact)
  - `flow(reverse)`:   2.4e-7  (essentially exact)
  - `quantizer.decode`: 0.0    (exact)
  - `enc_p.m_p/logs_p`: 9e-4 rel
  - `dec(Generator)`:  3e-3 rel
  - `ref_enc`, `ge(prelu)`: 2.6e-3, 2.9e-3 rel
- Full-pipeline outputs differ from PT only because of the independent random
  noise sample inside `z_p = m_p + randn * exp(logs_p) * 0.5` (the model is
  stochastic; PT and MLX draw independent samples).
- Quality: Breeze-ASR-26 transcribes the MLX output identically to the PT
  output (e.g. sentence 01 — both produce `你吃飽了嗎`).

## Methodology

- Each backend run in a separate process (no warm cross-contamination).
- Timing wraps the `vq_model.decode` call with `mx.eval(audio)` (MLX) or
  forced numpy roundtrip (PT MPS) for completion sync.
- Sentences taken from `_eval_r4_e15_driver.py`.
- Hardware: same machine, same thermal state.
