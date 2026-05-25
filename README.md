# sovits-mlx

An **MLX inference port** of GPT-SoVITS v2 (v2ProTw). Both stages are now
implemented in MLX:

  * **S2** (spectral decoder/vocoder, top-level dir) — drop-in replacement
    for `vq_model.decode(...)`.
  * **S1** (autoregressive text-to-semantic Transformer, `s1_mlx/`) — drop-in
    replacement for `t2s_model.model.infer_panel(...)`.

Both run natively on Apple Silicon via [MLX](https://github.com/ml-explore/mlx)
and coexist with the upstream PyTorch pipeline — you can use either backend
or both, hot-swapped at runtime.

> **Scope:** inference only. No training code here.

Upstream: [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) — all
credit for the model architecture, training recipe, and original PyTorch
implementation goes there.

## Why

S2 decode is the second-largest chunk of GPT-SoVITS latency on Apple Silicon
after the autoregressive S1 Transformer. Running it on MLX instead of PyTorch
MPS gives a clean **~3× speedup on the decode call** with no quality loss.

## Numerical fidelity vs PyTorch reference

Per-module max-abs / relative error against the PyTorch reference on random
inputs (see `tests/test_per_module.py`):

| Module                  | Error vs PT      |
|-------------------------|------------------|
| `ge_to512`              | 1.4e-7 (exact)   |
| `flow(reverse)`         | 2.4e-7 (exact)   |
| `quantizer.decode`      | 0.0 (exact)      |
| `enc_p.m_p / logs_p`    | 9e-4 rel         |
| `dec (Generator)`       | 3e-3 rel         |
| `ref_enc`, `ge(prelu)`  | 2.6e-3 / 2.9e-3 rel |

End-to-end outputs differ from PyTorch only because of the independent random
noise sample inside `z_p = m_p + randn * exp(logs_p) * 0.5` (the model is
stochastic and PT/MLX draw independent samples).

**Audio quality check:** Breeze-ASR-26 transcribes the MLX output identically
to the PyTorch output (e.g. sample 01 — both produce `你吃飽了嗎`). The
included `test_001.mp3` is a sample MLX output.

## Speed (Apple Silicon)

Measured on the same machine, same thermal state, S2 r4 e15 checkpoint, 5
Taiwanese sentences × 2 runs:

| Stage                          | Backend  | median | mean | min  |
|--------------------------------|----------|-------:|-----:|-----:|
| S2 `vq_model.decode` only      | PT MPS   |   98ms | 114  |  77  |
| S2 `vq_model.decode` only      | **MLX**  | **32ms** | 41 |  20  |
| End-to-end (S1 + S2 + post)    | PT MPS   |  898ms | 1024 | 794  |
| End-to-end (S1 + S2 + post)    | **MLX**  |  868ms |  977 | 744  |

- **S2 decode: ~3.1× faster** (98ms → 32ms median).
- **Real-time factor** for a 1.58s utterance: 0.062 (PT) → **0.020 (MLX, ~49× real time)**.
- End-to-end speedup of the S2-only port was small (1.05×) because S1
  still dominated on PyTorch. With the S1 MLX port now in `s1_mlx/`, the
  full MLX pipeline is **1.55× faster end-to-end** (~2.2× on S1 alone) — see
  [`s1_mlx/BENCHMARK.md`](s1_mlx/BENCHMARK.md).

Full numbers in [`BENCHMARK.md`](BENCHMARK.md).

## What's in the box

```
models.py        SynthesizerTrn — top-level S2 model (encoder, MRTE, flow, ref_enc, generator)
modules.py       Core blocks: Conv1dPT wrapper, WN, ResBlock1, Flip, ResidualCouplingLayer, …
mrte.py          MLX_MRTE — multi-reference timbre cross-attention
ref_enc.py       MLX_MelStyleEncoder — speaker style/ge encoder
hifigan.py       HiFi-GAN-style Generator (upsample 10×8×2×2×2 → 32 kHz)
convert.py       PyTorch ckpt → MLX safetensors weight converter
inference.py     End-to-end entry point (PT S1 + PT SV + MLX S2)
bench.py         PT-vs-MLX decode benchmark harness
tests/           Per-module numerical-fidelity tests vs PyTorch reference
MAPPING.md       Architecture mapping doc (PyTorch → MLX, layer by layer)
BENCHMARK.md     Detailed speed numbers + methodology
test_001.mp3     Sample MLX-rendered output

s1_mlx/          S1 (text-to-semantic AR transformer) MLX port
  model.py         T2SModel + transformer blocks + KV-cache decode
  convert.py       PyTorch .ckpt → MLX safetensors
  sampling.py      top-k / top-p / temperature / repetition-penalty (MLX)
  inference.py     Drop-in `infer_panel` + end-to-end CLI (MLX S1 + MLX S2)
  verify_numerical.py   PT-vs-MLX prefill + greedy-rollout equivalence test
  bench.py         PT-vs-MLX timing harness
  MAPPING.md       S1 architecture mapping (PyTorch → MLX)
  BENCHMARK.md     S1 speed numbers
```

## Dependencies

- Python ≥ 3.10
- [`mlx`](https://github.com/ml-explore/mlx) ≥ 0.18
- `numpy`, `torch` (still needed for S1 inference, ref-spec, and SV embedding)
- `safetensors`
- Upstream [`GPT-SoVITS`](https://github.com/RVC-Boss/GPT-SoVITS) on the
  Python path — `inference.py` reuses its S1 + SV + ref-spec pipeline.
- A v2ProTw S2 checkpoint (e.g. one of the official `s2G*.pth` releases or
  your own fine-tune).

## Usage

### 1. Convert PyTorch weights → MLX safetensors

```bash
python convert.py \
  --in  /path/to/s2_full_NN.pth \
  --out model.safetensors
```

This produces ~266 MB of weights (FP32). The converter handles the v2ProTw
key renames (`enc_p.text_embedding`, `sv_emb`, `ge_to512`, `prelu`, etc.) and
flattens nested submodules as needed.

### 2. End-to-end inference (PT S1 + MLX S2)

```bash
python inference.py \
  --text "{tw:Lí tsia̍h-pá--bē?}" \
  --ref-wav /path/to/ref.wav \
  --ref-text "{tw:reference utterance text}" \
  --s1 /path/to/s1.ckpt \
  --s2 /path/to/s2_full_NN.pth \
  --out test_001.mp3
```

`inference.py` boots the upstream GPT-SoVITS PyTorch pipeline, then hot-swaps
`vq_model.decode` for the MLX implementation. Everything else (S1 sampling,
reference spectrogram, SV embedding, post-processing) is unchanged.

### 3. Programmatic use

```python
import mlx.core as mx
from safetensors.numpy import load_file
from models import SynthesizerTrn, default_config

m = SynthesizerTrn(**default_config())
sf = load_file("model.safetensors")
m.load_weights([(k, mx.array(v)) for k, v in sf.items()], strict=True)
mx.eval(m.parameters())

audio = m.decode(codes_mx, text_mx, refer_mx, sv_emb_mx, noise_scale=0.5)
mx.eval(audio)
```

See the docstring of `SynthesizerTrn.decode` in `models.py` for shapes.

### 4. Per-module fidelity tests

```bash
python tests/test_per_module.py --s2 /path/to/s2_full_NN.pth
```

### 5. Benchmark

```bash
python bench.py
```

## Status / non-goals

- **Inference only.** Training and gradient flow are out of scope (both
  S1 and S2 are inference ports).
- **v2ProTw checkpoint format.** This port targets the `version="v2ProTw"`
  variant (1033-row text vocab, `sv_emb` + `ge_to512` + `prelu` heads). Other
  GPT-SoVITS variants (v1, v2, v3) are not supported.
- Single reference audio is used in the decode wrapper (upstream supports
  multi-reference; the MLX path takes only the first reference). PRs welcome.

## Credits

- [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) — original
  model and PyTorch code (MIT-licensed).
- [`mlx-audio`](https://github.com/Blaizzy/mlx-audio) MeloTTS port — used as
  a reference for MLX conventions (Conv1dPT wrapper, weight loading, etc.).
- [Apple MLX](https://github.com/ml-explore/mlx).

## License

MIT — see [`LICENSE`](LICENSE). Inherits and extends the upstream GPT-SoVITS
MIT license.
