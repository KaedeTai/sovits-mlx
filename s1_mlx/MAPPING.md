# S1 PyTorch → MLX architecture map

S1 is the GPT-style autoregressive transformer that converts phoneme tokens to
semantic (audio-code) tokens. It is the slowest stage of GPT-SoVITS inference
and benefits the most from per-token KV-cache decoding. This directory mirrors
the PyTorch implementation in `~/tts/GPT-SoVITS/GPT_SoVITS/AR/` 1:1 in MLX so
that the same checkpoint (`arm_A_e15_trilingual.ckpt`) drives both backends.

## Model dimensions (`config["model"]`)

| key                    | value | notes                                    |
| ---                    | ---:  | ---                                      |
| `hidden_dim`           | 512   | `d_model` for the transformer            |
| `embedding_dim`        | 512   | text & audio embedding dim               |
| `head`                 | 16    | attention heads                          |
| `n_layer`              | 24    | transformer blocks (post-LN)             |
| `linear_units`         | 2048  | FFN inner dim (= 4 × hidden)             |
| `phoneme_vocab_size`   | 1033  | phoneme/grapheme token range             |
| `vocab_size`           | 1025  | audio-code range  (1024 + EOS)           |
| `EOS`                  | 1024  | end-of-sequence id                       |

## Top-level keys (after stripping `model.` prefix)

| PyTorch ckpt key                                       | shape         | MLX module / attr                                       |
| ---                                                    | ---           | ---                                                     |
| `model.bert_proj.{weight,bias}`                        | (512, 1024)   | `bert_proj` (`nn.Linear`)                               |
| `model.ar_text_embedding.word_embeddings.weight`       | (1033, 512)   | `ar_text_embedding.word_embeddings`                     |
| `model.ar_text_position.alpha`                         | (1,)          | `ar_text_position.alpha`                                |
| `model.ar_audio_embedding.word_embeddings.weight`      | (1025, 512)   | `ar_audio_embedding.word_embeddings`                    |
| `model.ar_audio_position.alpha`                        | (1,)          | `ar_audio_position.alpha`                               |
| `model.h.layers.<i>.<…>`                               | per layer     | `h.layers.<i>` — `T2SBlock`                             |
| `model.ar_predict_layer.weight`                        | (1025, 512)   | `ar_predict_layer` (`nn.Linear`, **no bias**)           |

The `model.` prefix is dropped at conversion (`convert.py`). The fixed
sinusoidal table inside `SinePositionalEmbedding` is built on the fly and
named `_pe`; the leading underscore opts it out of MLX's parameter tree so it
does not need a corresponding safetensors entry.

## Per-layer (`h.layers.<i>.*`) keys

| PyTorch                                            | shape           | MLX                                          |
| ---                                                | ---             | ---                                          |
| `self_attn.in_proj_weight`                         | (3*512, 512)    | `self_attn.in_proj_weight` (fused QKV)        |
| `self_attn.in_proj_bias`                           | (3*512,)        | `self_attn.in_proj_bias`                      |
| `self_attn.out_proj.{weight,bias}`                 | (512, 512), (512,) | `self_attn.out_proj` (`nn.Linear`)         |
| `linear1.{weight,bias}`                            | (2048, 512), (2048,) | `linear1` (`nn.Linear`)                  |
| `linear2.{weight,bias}`                            | (512, 2048), (512,) | `linear2` (`nn.Linear`)                   |
| `norm1.{weight,bias}`, `norm2.{weight,bias}`       | (512,)          | `norm1`, `norm2` (`nn.LayerNorm`, eps=1e-5)   |

The conversion is shape-preserving — PyTorch's `nn.Linear.weight` layout `(out, in)` matches
`mlx.nn.Linear`, and `nn.LayerNorm` is identical between the two. No transposes are needed.

## Layer math (post-LN)

```
attn = self_attn(x)                       # multi-head, fused QKV
x = norm1(x + attn)
x = norm2(x + linear2(relu(linear1(x))))
```

This matches the PT `_sa_block` + `_ff_block` pair when `norm_first=False`.

## Attention

PyTorch uses a single `in_proj_weight` of shape `(3·d, d)` for fused QKV
(then `chunk(3, dim=-1)` to split). The MLX side does the same:
`x @ in_proj_weight.T + in_proj_bias`, then `mx.split(., 3, axis=-1)`.

Reshape to heads: `(B, T, H) → (B, T, n_h, head_d) → (B, n_h, T, head_d)`.
Attention itself uses `mx.fast.scaled_dot_product_attention`, which accepts
an additive bias mask (0 keep, −∞ mask).

## KV cache

Per layer, two `(B, T, H)` arrays stored unhead. On each decode step we
`concatenate([k_cache, k_new], axis=1)` and pass to `attend`. This matches
PT's `T2STransformer.decode_next_token`.

## Sampling

`sampling.py` mirrors PT's `AR.models.utils.{logits_to_probs, sample, multinomial_sample_one_no_sync}`:

1. **Repetition penalty.** For tokens that appear in `previous_tokens`,
   negative scores get multiplied by the penalty, non-negative scores divided.
2. **Top-p (nucleus).** Sort descending (via `mx.argsort(-x)` — MLX has no
   `flip`), CDF, keep the smallest set of tokens with cumulative probability
   ≥ top_p, shifted right by one (always keep the first token over the
   threshold).
3. **Temperature.** Divide logits by `max(T, 1e-5)`.
4. **Top-k.** Replace anything below the k-th largest with −∞.
5. **Softmax.**
6. **Multinomial.** Gumbel-max: `argmax(probs / Exp(1))`.

The Exp(1) sample is `-log(U)` with `U ~ Uniform(eps, 1)`.

## Inference flow (`make_mlx_infer_panel` in `inference.py`)

Drop-in replacement for `t2s_model.model.infer_panel`. Steps:

1. **Convert PT tensors → MLX arrays** (phoneme ids, BERT features, optional
   prompt audio codes).
2. **Build text prefix** `x = text_emb + bert_proj(bert.T)` + sinusoidal pos.
3. **Prefill** the full `[text, prompt_audio]` context in one forward pass;
   stash per-layer K/V caches.
4. **Generate** up to 1500 tokens. For each step:
   - Compute logits from `hidden[:, -1, :]`.
   - For steps < 11, mask out the EOS slot (matches PT — prevents premature
     termination on tiny sequences).
   - Sample with the recipe above.
   - Stop on EOS (either via sampled token or argmax of the un-temperatured
     logits, matching PT exactly) or when `(y_len - prefix_len) > early_stop_num`.
   - Otherwise embed the new token, add the `ar_audio_position` contribution
     at absolute position `prefix_len + step`, and run one `decode_step`.
5. **Convert MLX output → torch.LongTensor** on `pt_device` so the caller
   (`inference_webui.get_tts_wav`) sees the same return shape and dtype as
   the original PT path.

## Verified equivalence

`verify_numerical.py` runs two tests against PT:

* **Prefill logits.** Max absolute diff ≈ 0.08, max relative ≈ 1.1% over 1025
  vocab logits. The argmax and the top-10 token set match exactly. The
  residual is from MLX's fast SDPA kernel using different accumulation order
  than PyTorch's, not from a structural error — the rest of the layer math
  is bit-equivalent in our tracing.
* **Greedy autoregressive rollout** (16 steps). PT and MLX produce
  **identical** token sequences with the same `argmax`-greedy decoder.

This is the same numerical-equivalence pattern as the S2 port — close enough
that the AR sampler chooses identical tokens, even though intermediate
floats differ by ~1%.
