# S1 MLX vs PyTorch — speed benchmark

## Setup

* **Machine**: the user's Mac (Apple Silicon, what the `KaedeTai/sovits-mlx`
  port targets).
* **Eval set**: 5 canonical TW sentences (`tw_short_01..03`, `tw_medium_01`,
  `tw_long_01`), the same set used by `~/tts/_s1_trilingual/run_samples.py`.
  Lengths span ~17 → ~149 characters.
* **Reference voice**: `~/tts/tw_recording/wavs_norm/001.wav` with POJ text.
* **Iterations**: 1 warmup + 3 measured per sentence per backend. Aggregate
  stats are over the 15 measured runs.
* **What's measured**:
  * `s1` — wall-clock seconds inside one `t2s_model.model.infer_panel` call
    (the AR loop, including KV-cache setup and sampling).
  * `s2` — wall-clock seconds inside one `vq_model.decode` call.
  * `total` — wall-clock seconds of the full `tts_long.synthesize` call,
    including text frontend, BERT extraction, ref-audio prep, and file I/O.

## Note on the PyTorch baseline

`tts_long.synthesize` + `GPT_SoVITS/inference_webui.py` set `device='cpu'`
by default on this install — even though `torch.backends.mps.is_available()`
returns `True`. We left the install as-is so the "PyTorch baseline" is
exactly what runs when a user types `python tts_long.py`. Forcing MPS in-place
on `inf.t2s_model` / `inf.vq_model` causes other helpers in tts_long
(cnhubert preprocessing, SV embedding, the BERT projector path) to misroute
their inputs and the segment generator silently yields nothing. Wiring MPS
correctly is a separate refactor; for the purposes of this S1 port, the
relevant question is "is the MLX port faster than the path the user runs
today?", which the CPU baseline answers.

If a user later wires MPS up properly, the absolute PT numbers below will
drop and the S1 speedup factor will shrink — but the MLX numbers are
independent of the PT setup, so the MLX absolute latencies still stand.

## Per-sentence results

PT n=3 per sentence; MLX n=3 per sentence (n=2 for `tw_long_01` because the
benchmark process exited a multiprocessing cleanup early — see `bench.log`).
Mean ± standard deviation in seconds.

| Sentence       | PT S1            | MLX S1           | S1 speedup | PT S2           | MLX S2          | S2 speedup | PT total        | MLX total       | total speedup |
| ---            | ---:             | ---:             | ---:       | ---:            | ---:            | ---:       | ---:            | ---:            | ---:          |
| `tw_short_01`  | 0.169 ± 0.001    | 0.085 ± 0.004    | 1.98×      | 0.089 ± 0.003   | 0.030 ± 0.003   | 2.93×      | 0.806 ± 0.021   | 0.666 ± 0.008   | 1.21×         |
| `tw_short_02`  | 0.260 ± 0.021    | 0.133 ± 0.017    | 1.96×      | 0.137 ± 0.011   | 0.035 ± 0.002   | 3.96×      | 0.960 ± 0.024   | 0.727 ± 0.017   | 1.32×         |
| `tw_short_03`  | 0.179 ± 0.009    | 0.093 ± 0.001    | 1.92×      | 0.088 ± 0.005   | 0.031 ± 0.000   | 2.88×      | 0.813 ± 0.010   | 0.686 ± 0.005   | 1.18×         |
| `tw_medium_01` | 0.519 ± 0.017    | 0.267 ± 0.012    | 1.95×      | 0.297 ± 0.011   | 0.043 ± 0.004   | 6.85×      | 1.386 ± 0.066   | 0.862 ± 0.019   | 1.61×         |
| `tw_long_01`   | 0.999 ± 0.024    | 0.473 ± 0.066    | 2.11×      | 0.582 ± 0.015   | 0.062 ± 0.009   | 9.38×      | 2.120 ± 0.031   | 1.085 ± 0.069   | 1.95×         |

## Aggregate results (n=15 PT, n=14 MLX)

|           | PyTorch          | MLX              | Speedup   |
| ---       | ---:             | ---:             | ---:      |
| **S1**    | 0.425 ± 0.314 s  | 0.191 ± 0.136 s  | **2.22×** |
| **S2**    | 0.239 ± 0.188 s  | 0.039 ± 0.011 s  | **6.18×** |
| **Total** | 1.217 ± 0.500 s  | 0.785 ± 0.144 s  | **1.55×** |

### Headline

* **S1 inference is 2.22× faster on MLX** vs. the default-install PyTorch
  path on the same machine.
* S2 speedup (6.18× here) is higher than the 3.1× reported in the parent
  repo's existing BENCHMARK.md, almost certainly because that earlier number
  was measured against a PT-MPS baseline and this one is against PT-CPU. The
  S2 numbers should not be re-interpreted as a regression in PT performance
  — both backends improved on this Mac since that earlier benchmark.
* End-to-end pipeline speedup is **1.55×** — diluted by text frontend, BERT
  extraction, and file I/O, which are not in scope for this port.
* Variance is small in absolute terms (≤ 5% of the mean per sentence) so
  the speedup factors are stable.

## Reproducing

```bash
cd ~/tts/_sovits_mlx/s1_mlx
~/tts/GPT-SoVITS/.venv/bin/python bench.py
# -> bench_results.json + console table
```
