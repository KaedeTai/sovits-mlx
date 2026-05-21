# GPT-SoVITS S2 (v2ProTw) → MLX architecture mapping

**Source:** `~/tts/GPT-SoVITS/GPT_SoVITS/module/models.py` + `attentions.py` + `modules.py` + `mrte_model.py`
**Checkpoint:** `tw_finetune_synthetic/s2_logs_r4/s2_full_15.pth` (781 keys, version=v2ProTw)
**Reference MLX port:** `~/tts/mlx_explore/mlx-audio/mlx_audio/tts/models/melotts/`
**Entry point for inference:** `SynthesizerTrn.decode(codes, text, refer, sv_emb=...)`

## Top-level config (from `configs/s2_taiwan.json` + v2Pro overrides)

```
spec_channels       = filter_length // 2 + 1 = 1025  (filter_length=2048)
sampling_rate       = 32000
hop_length          = 640
segment_size        = 20480 / 640 = 32  (samples → frames; only used in training)
inter_channels      = 192
hidden_channels     = 192
filter_channels     = 768
n_heads             = 2
n_layers            = 6
kernel_size         = 3
p_dropout           = 0.0
resblock            = "1"
resblock_kernel_sizes   = [3, 7, 11]
resblock_dilation_sizes = [[1,3,5], [1,3,5], [1,3,5]]
upsample_rates          = [10, 8, 2, 2, 2]
upsample_initial_channel= 512
upsample_kernel_sizes   = [16, 16, 8, 2, 2]
gin_channels        = 1024
semantic_frame_rate = "25hz"        (ssl_proj is Conv1d 768→768 stride=2)
version             = "v2ProTw"     (1033-row text vocab, has sv_emb/ge_to512/prelu)
```

## Component map

PyTorch tensor shapes are **[B, C, T]**.
MLX `nn.Conv1d` consumes **[B, T, C]**, so we use a `Conv1dPT` wrapper (transpose 0,2,1 → conv → transpose back), exactly like the MeloTTS port. All inputs/outputs in the MLX model stay in `[B, C, T]` to mirror the PyTorch reference.

### 1. `ref_enc` — `MelStyleEncoder` (modules.py:672)
Speaker style encoder. **Direct match with PyTorch.**

Pipeline (forward expects [B, 704_or_n_mel, T] spec, drops mask):
- `spectral = nn.Sequential(Linear(704→128), Mish, Dropout, Linear(128→128), Mish, Dropout)`
  → PyTorch `ref_enc.spectral.0.fc.weight: (128,704)`, `spectral.3.fc.weight: (128,128)`
- `temporal = Sequential(Conv1dGLU(128,128,5), Conv1dGLU(128,128,5))`
  → `temporal.N.conv1.conv.weight: (256, 128, 5)` (1d conv producing 2*hid, GLU split, residual)
- `slf_attn = MultiHeadAttention(n_head=2, d_model=128, d_k=64, d_v=64)`
  → `slf_attn.w_qs/w_ks/w_vs.weight: (128,128)`, `fc.weight: (128,128)`. Standard Linear MHA.
- `fc = LinearNorm(128 → 1024)` → `fc.fc.weight: (1024, 128)`
- Output: temporal-avg-pool → `(B, 1024, 1)` (the "ge" speaker embedding)

**MLX module:** `MLX_MelStyleEncoder` — new file `ref_enc.py`. Linear/Conv layers straightforward.
Note: `Conv1dGLU` uses 1d conv expanding to 2*out_channels, then `glu_dim=1` (channel-split + sigmoid gate).

### 2. `sv_emb` / `ge_to512` / `prelu` (v2Pro additions, models.py:932-936)
- `sv_emb = nn.Linear(20480, 1024)` — speaker-verification embedding projection
- `prelu = nn.PReLU(num_parameters=1024)` — per-channel slope on ge (after sv add)
- `ge_to512 = nn.Linear(1024, 512)` — projects ge for enc_p (used inside TextEncoder via MRTE)

Path (decode):
```
ge_raw = ref_enc(refer[:704])           # (B, 1024, 1)
ge = prelu(ge_raw + sv_emb(sv_emb_in).unsqueeze(-1))   # (B, 1024, 1)
ge_512 = ge_to512(ge.transpose(1,2)).transpose(1,2)    # (B, 512, 1)  ← used for MRTE only
# ge (1024) is used as conditioning for flow and dec (g param)
```

**MLX:** `nn.Linear`/`nn.PReLU` map 1:1. Implement as part of `MLX_SynthesizerTrn`.

### 3. `quantizer` — `ResidualVectorQuantizer` (quantize.py)
Only needs `.decode(codes)` at inference. Internally `vq.layers.0._codebook.embed: (1024, 768)`.
Embedding lookup table. **No conversion of `inited`/`cluster_size`/`embed_avg` needed** (only used during training EMA updates).

```python
# decode is essentially: quantized = embed[codes]  (codes: [B,1,T_code] long)
quantized = embed[codes.squeeze(1)]  # (B, T_code, 768)
quantized = quantized.transpose(1,2) # (B, 768, T_code)
```

The real GPT-SoVITS path additionally calls `vq.project_out` etc., but for a single-layer RVQ with `project_in=Identity`/`project_out=Identity` (the default for `dim==codebook_dim`), it reduces to a plain embedding lookup. Confirm via inspection.

**MLX:** Simple `mx.take(embed, codes, axis=0)` + transpose.

### 4. `ssl_proj = nn.Conv1d(768, 768, kernel=2, stride=2)`
Halves the SSL/quantized rate (50→25Hz). NB — this is the **top-level** `ssl_proj` (separate from `enc_p.ssl_proj`).
Weights: `ssl_proj.weight: (768, 768, 2)`, `ssl_proj.bias: (768,)`.

**Wait** — the decode path doesn't use `self.ssl_proj` (only `forward()` does). `decode()` calls `self.quantizer.decode(codes)` then `F.interpolate(..., scale*2, mode="nearest")` (because `semantic_frame_rate=="25hz"`). So at inference, ssl_proj is dead weight.

Action: still convert to keep checkpoint round-trip clean. **Used at inference: no.**

### 5. `enc_p` — `TextEncoder` (models.py:157)
This is the heart. Inputs:
- `quantized` (B, 768, T_y) — interpolated 2x to match y frames
- `y_lengths` (B,)
- `text` (B, T_t) — token ids in 0..1032 (v2ProTw vocab)
- `text_lengths` (B,)
- `ge` for v2pro is **ge_512** (B, 512, 1) — used only by MRTE

Pipeline:
- `ssl_proj = nn.Conv1d(768, 192, 1)` → keys `enc_p.ssl_proj.{weight,bias}` shape (192,768,1)
- `encoder_ssl = attentions.Encoder(192, 768, 2 heads, **n_layers//2 = 3**, k=3, ...)` — 3 transformer layers with relative pos (window_size=4 default)
- `text_embedding = nn.Embedding(1033, 192)` → `text_embedding.weight: (1033, 192)`
- `encoder_text = attentions.Encoder(192, 768, 2, **6 layers**, k=3, ...)`
- `mrte = MRTE()` — cross-attention to fuse ssl_enc with text_enc and ge_512 (see §6)
- `encoder2 = attentions.Encoder(192, 768, 2, **3 layers**, k=3, ...)`
- `proj = nn.Conv1d(192, 2*inter_channels=384, 1)` → `enc_p.proj.weight: (384,192,1)`

forward output: `y, m, logs, y_mask, ...` — m/logs are the (B,192,T) means/log-stds.

`attentions.Encoder` per layer:
- `MultiHeadAttention(192,192,n_heads=2,window_size=4)`
  - conv_q, conv_k, conv_v, conv_o: all `Conv1d(192,192,1)` → keys `conv_q.weight: (192,192,1)`
  - `emb_rel_k, emb_rel_v` each `(1, 9, 96)` (2*window_size+1=9, k_channels=96)
  - Attention computed in `[B, n_heads, T, k_channels]` after reshape.
- `LayerNorm(192)` (channel-first, using gamma/beta names)
- `FFN(192, 192, 768, kernel_size=3)`: conv_1 (192,768,3), conv_2 (768,192,3), ReLU between.
  Padding is **same padding** (pad_l=1, pad_r=1) — not causal, no g passed in encoder.

**MLX:** all match `melotts/attentions.py` patterns: `Conv1dPT`, `LayerNorm` (channel-first), `MultiHeadAttention` with relative-position. Number of layers and channel sizes differ but the building blocks are identical to MeloTTS.

### 6. `MRTE` — Multi-Reference Timbre Encoder (mrte_model.py:9)
```
c_pre = Conv1d(192, 512, 1)        # ssl path
text_pre = Conv1d(192, 512, 1)     # text path
cross_attention = MultiHeadAttention(channels=512, out=512, n_heads=4)
                  # Conv1d(512,512,1) for q,k,v,o; NO relative-position (no window_size)
c_post = Conv1d(512, 192, 1)
```
Forward:
```
attn_mask = text_mask.unsq(2) * ssl_mask.unsq(-1)  # (B,1,T_y,T_t)
ssl_enc = c_pre(ssl_enc)         # (B,512,T_y)
text_enc = text_pre(text)        # (B,512,T_t)
x = cross_attention(ssl_enc, text_enc, attn_mask) + ssl_enc + ge_512   # broadcast ge_512 over T
x = c_post(x)                    # (B,192,T_y)
```

**MLX:** trivial reuse of `MultiHeadAttention` from melotts/attentions.py with `window_size=None`. Need cross-attention variant (q from x, k/v from c). MeloTTS MHA already takes (x, c, mask) — perfect.

### 7. `flow` — `ResidualCouplingBlock` (models.py:290)
4 coupling layers (`flows.0, 2, 4, 6`) interleaved with `Flip` (`flows.1, 3, 5, 7`, no params).
Each `ResidualCouplingLayer`:
- `pre = Conv1d(96, 192, 1)`            # half_channels=96
- `enc = WN(hidden=192, kernel=5, dilation_rate=1, n_layers=4, gin=1024)`
  - cond_layer Conv1d(1024, 2*192*4=1536, 1) — keys `cond_layer.weight_{g,v}` (1536,1024,1)/(1536,1,1)
  - in_layers[0..3] Conv1d(192, 384, 5, dilation=1, pad=2)  (dilation_rate=1, so dilation=1 always)
  - res_skip_layers[0..2] Conv1d(192, 384, 1); [3] Conv1d(192, 192, 1)
- `post = Conv1d(192, 96, 1)`           # mean_only=True → out is 96, not 192

forward (`reverse=True`):
```
x0, x1 = split(x, [96,96], dim=1)
h = pre(x0) * mask
h = enc(h, mask, g=ge)
m = post(h) * mask     # mean_only → no logs
x1 = (x1 - m) * exp(-0) = x1 - m
return concat([x0, x1], dim=1)
```

At inference: `z = flow(z_p, y_mask, g=ge, reverse=True)` runs the 4 layers in reverse order (with Flip in between).

**MLX:** `WN` is straight from MeloTTS modules.py. `ResidualCouplingLayer` (mean_only=True) matches MeloTTS. `Flip` matches.

### 8. `dec` — `Generator` (HiFi-GAN-style, models.py:444)
**Exactly matches MeloTTS Generator** with `gin_channels=1024`.

- `conv_pre = Conv1d(192, 512, 7, pad=3)`
- `cond = Conv1d(1024, 512, 1)`    # gin condition pre-added to x
- 5 upsample stages, each:
  - `ups[i] = ConvTranspose1d(in=512//2^i, out=512//2^(i+1), kernel=upsample_kernel_sizes[i], stride=upsample_rates[i], pad=(k-stride)//2)`
    e.g. ups[0]: ConvTranspose1d(512, 256, 16, 10, pad=3) — but ckpt shows weight_v (512,256,16) which is ConvTranspose1d weight shape `(in_channels, out_channels, kernel)`.
  - 3 resblocks per stage (`ResBlock1` with dilations [1,3,5] × 3 kernels [3,7,11])
- `conv_post = Conv1d(16, 1, 7, pad=3, bias=False)`

forward (g=ge[1024]):
```
x = conv_pre(x)             # (B, 512, T)
x = x + cond(g)             # broadcast over T
for i in range(5):
    x = leaky_relu(x, 0.1)
    x = ups[i](x)
    xs = sum over j (resblocks[i*3 + j](x)) / 3
    x = xs
x = leaky_relu(x); x = conv_post(x); x = tanh(x)   # (B, 1, T_wav)
```

**MLX:** copy `melotts/hifigan.py:Generator` verbatim, only changing `upsample_rates`/`kernel_sizes` to ours. ConvTranspose1d weight transpose is already handled in melotts convert.py (see §weight-conv).

### 9. `enc_q` — `PosteriorEncoder` (models.py:335)
**Only used in training.** Inference path (`decode`) never calls it.
Action: **do not port**. We *do* convert its weights to safetensors (or skip with `enc_q` filter to save space). Skipping reduces converted weight count by 103.

## Discarded for inference

- `enc_q`: posterior encoder, training only
- `ssl_proj` (top-level): used by `forward()`, not `decode()`
- `quantizer.vq.layers.0._codebook.{inited, cluster_size, embed_avg}`: EMA running stats, only `embed` is needed.

## PyTorch ↔ MLX naming/shape conversions (analogous to melotts/convert.py)

1. **weight_norm fusion**: for every key matching `.weight_v`, look up sibling `.weight_g`, compute
   `weight = weight_g * weight_v / ||weight_v||` (norm over axes 1..ndim, keepdims) and emit as
   `.weight`. Drop `.weight_g`. This must run for every `weight_norm`'d module:
   - all `enc_q.enc.in_layers.*` (skipped — enc_q dropped)
   - all `flow.flows.{0,2,4,6}.enc.in_layers.*` and `.res_skip_layers.*` and `.cond_layer`
   - all `dec.ups.*`
   - all `dec.resblocks.*.convs{1,2}.*`
   - (No weight_norm on ref_enc.spectral, ref_enc.slf_attn — those have plain `.weight`.)

2. **LayerNorm renames**: PyTorch `LayerNorm` in `modules.py` uses `gamma`/`beta`. MLX `nn.LayerNorm`
   uses `weight`/`bias`. **But** the MeloTTS port has its own channel-first `LayerNorm` that uses
   `weight`/`bias` (see attentions.py:11). Rename
   `*.gamma` → `*.weight`, `*.beta` → `*.bias`.

3. **Conv1d weight shape**: PyTorch `(out, in, k)` → MLX `nn.Conv1d` `(out, k, in)`. We wrap with
   `Conv1dPT`, so the conversion is `weight.transpose(0, 2, 1)` AND key becomes
   `module.conv.weight` (and `module.conv.bias`).

4. **ConvTranspose1d weight shape**: PyTorch `(in, out, k)` → MLX `nn.ConvTranspose1d` `(out, k, in)`,
   so transpose `(1, 2, 0)`. Key becomes `module.conv_t.weight`. Same as melotts.

5. **PReLU**: PyTorch `prelu.weight` shape `(1024,)`. MLX `nn.PReLU` stores in attribute `weight`
   of shape `(num_parameters,)`. Match by name.

6. **emb_rel_k / emb_rel_v** (relative position embeddings in attentions): plain parameters
   of shape `(1, 9, 96)`. In MeloTTS port they live as attributes of the same name on
   MultiHeadAttention — just save as `attn_layers.N.emb_rel_k`, no shape change.

7. **Embedding**: `text_embedding.weight: (1033, 192)` → MLX `nn.Embedding.weight` same layout.

8. **MelStyleEncoder Linear** (`fc.weight: (out, in)`) — MLX `nn.Linear.weight` same layout.

## Inference plumbing (Phase 4)

Driver:
1. Compute pred_semantic (codes) using PyTorch S1 (unchanged).
2. Compute ref spec [B, 1025, T_ref] via PyTorch `spectrogram_torch` (matches `inference_webui.get_spepc`).
3. Compute `sv_emb` (B, 20480) via PyTorch ERes2NetV2 (separate model — keep in PyTorch).
4. Hand all four (codes, text_ids, spec[:,:704,:], sv_emb) to MLX model.
5. MLX runs `decode()` and returns `(B, 1, T_wav)`. Convert to numpy float32, save WAV → MP3.

## Risks / divergences to validate per-module

- **Relative-position attention** (Encoder MHA with window_size=4): the matmul reshape is delicate.
  MeloTTS port has it, but with `heads_share=True` and 1 head share — confirm shapes by running
  a fixed input through both PT and MLX and diffing.
- **MelStyleEncoder masking**: in our decode path we pass `mask=None` (refer is a single contiguous
  spec), so the `masked_fill` paths in MHA collapse — keep the simpler unmasked code path.
- **`F.interpolate(quantized, scale*2, mode="nearest")`** — MLX has no direct interpolate, but
  nearest-neighbor 2x upsample is just `repeat` along time axis. Same for `enc_p.encoder`
  speed-adjust path (we'll just not pass `speed != 1` initially).
- **`fused_add_tanh_sigmoid_multiply`**: already a plain tanh*sigmoid split in MeloTTS WN port.
- **Random noise**: PyTorch `torch.randn_like` for `z_p = m_p + randn*exp(logs)*noise_scale`.
  Use `mx.random.normal` with a fixed seed (optional) — random noise just adds variance, not
  determinism. For numerical comparison we'll set both to the same noise via numpy `np.random.seed`
  and inject the same noise into both flows.

## Per-module weight count sanity (post-conversion target)

Module          | PyTorch keys (w/ wn) | MLX target keys
----------------|----------------------|------------------
ref_enc         | 18                   | 18 (Linear & Conv unchanged)
sv_emb,ge_to512,prelu | 5             | 5
quantizer.vq.layers.0._codebook.embed | 1 | 1  (drop inited/cluster_size/embed_avg)
ssl_proj        | 2                    | 2  (kept but unused)
enc_p           | 235                  | ~235 (renames only)
flow            | 124 (incl weight_g/v) | 88 (4 cond_layer + 16 in_layers + 16 res_skip + 4 pre + 4 post = 88 fused weights, all with bias separately)
dec             | 290                  | dec convs become .conv./.conv_t. but key count roughly the same after fusion
enc_q (skip)    | 103                  | 0

Expected final safetensors key count: ~700.

## Phase ordering reminder

Phase 2: write `convert.py`. Phase 3: write `models.py` (and `ref_enc.py`, `attentions.py`, `modules.py`, `hifigan.py`). Phase 4: `inference.py` + numerical test.
