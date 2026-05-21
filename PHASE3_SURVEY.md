# Phase 3 build — survey before commit

## Summary

The user's task description lists 6 missing pieces for Phase 3:

> 1. MLX DataLoader  2. Multi-epoch loop  3. LR scheduler
> 4. Checkpoint compatibility  5. KL loss enabled  6. Resume from r5 e16

Each of these is true, but the survey turned up **three additional foundational
gaps** in the Phase 1+2 prototype that have to be built before any of those six
can run end-to-end. They live below the user's stated list because the prototype
deliberately scoped itself to "Generator + Discriminator with random latents."

## What the Phase 1+2 prototype actually exercises

`train_step.py` runs:
- random `z` (B, 192, 32) → `Generator(z, g=cond)` → fake audio
- mel L1 vs random "real audio"
- `MultiPeriodDiscriminator(real, fake)` → LSGAN + feature loss
- 2 AdamW steps (G then D)

It does **not** touch `enc_p` (TextEncoder), `enc_q` (PostEnc — module exists but
never called from `train_step.py`), `flow`, `quantizer`, or `SynthesizerTrn`.

`tests/test_grad.py` exercises `PostEnc`, `MPD`, `Generator+mel L1+LSGAN`
individually — never integrated, never through the flow.

## Foundational gaps the user's list assumes are done (but aren't)

| Gap | Current state | What's needed for training |
|---|---|---|
| `SynthesizerTrn.forward(ssl, spec, lens, text, text_lens, sv_emb)` | Only `.decode()` exists (inference). | Full training forward returning `(y_hat, kl_ssl, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q), quantized)` — see `GPT_SoVITS/module/models.py:938-975`. |
| Training-time RVQ with `commit_loss` (kl_ssl) | `QuantizerDecoder.decode(codes)` is a plain embedding lookup. | Forward RVQ producing `quantized, codes, commit_loss, _` for codebook learning. Upstream uses `vector_quantize_pytorch` with EMA updates — **may need a no-grad EMA workaround in MLX** since MLX doesn't have a built-in EMA-update mechanism that side-steps autograd the same way PyTorch's `torch.no_grad()` does. |
| `commons.rand_slice_segments` | Not in MLX. | Random crop of latent z + audio for the segment_size (20480 audio / 32 latent) sub-batch fed to Generator. Trivial port. |
| `commons.sequence_mask` | `modules.py` has it. | ✓ |
| Flow forward backward | Module exists, code path is `reverse=False`, never autograd-tested. | **Critical gating step.** Needs Phase 3.1 verification. |
| TextEncoder backward | Module exists for inference, never autograd-tested. | Standard ops, expected to work, but verify in Phase 3.1. |

## c_kl default to restore

- Upstream `configs/s2.json` default: **`c_kl = 1.0`**
- r5 `s2_pathM_r5.json` patched: `c_kl = 0.0` (the MPS view-stride bug workaround)
- r6 target: **`c_kl = 1.0`** (the whole point of going to MLX)

Also relevant: upstream uses `kl_ssl * 1` (the commit_loss term from quantizer)
ADDED to loss_gen_all. So the full upstream gen loss is:

    loss_gen_all = loss_gen + loss_fm + loss_mel + kl_ssl * 1 + loss_kl

`loss_mel` = `F.l1_loss(y_mel, y_hat_mel) * c_mel` (c_mel=45)
`loss_kl` = `kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * c_kl` (c_kl=1.0)

## Order of forward ops in training step (upstream)

1. `ge = ref_enc(y[:, :704] * y_mask, y_mask)` — speaker embedding from spec
2. `sv_emb` → projected, added to ge, prelu, also `ge_to512` for MRTE conditioning
3. `ssl = ssl_proj(ssl)` — top-level Conv1d(768→768, k=2, s=2)
4. `quantized, codes, commit_loss, _ = quantizer(ssl, layers=[0])`
5. `quantized = F.interpolate(quantized, *2, "nearest")` (semantic_frame_rate=="25hz")
6. `x, m_p, logs_p, y_mask, _, _ = enc_p(quantized, y_lengths, text, text_lengths, ge512)`
7. `z, m_q, logs_q, y_mask = enc_q(y, y_lengths, g=ge)`
8. `z_p = flow(z, y_mask, g=ge)` ← FORWARD direction (no reverse)
9. `z_slice, ids_slice = rand_slice_segments(z, y_lengths, segment_size)`
10. `y_hat = dec(z_slice, g=ge)` ← Generator on the slice
11. Discriminator step on `(y_real_slice, y_hat.detach())`
12. Generator backward on `loss_gen_all`

The discriminator step happens BEFORE the generator step in upstream (line 420-449
in `s2_train.py`); both use the same `y_hat` (D uses detached, G doesn't).

## text_low_lr_rate

Upstream: text encoder parameters get **0.4× LR**. Implemented via PyTorch
param groups. In MLX we'll need to either:
(a) maintain two AdamW optimizers (one for text-encoder params, one for the rest), or
(b) post-scale text encoder grads by 0.4 before optimizer.update.

Option (b) is easier.

## Honest scope assessment

The user's "6 things missing" implies the foundation is solid. Survey shows
there are **3 more foundational items below that list**. Realistic plan:

1. **Phase 3.0** (this doc) — survey ✓
2. **Phase 3.1 (gating)** — verify MLX autograd through:
   - Flow `forward(x, mask, g, reverse=False)` ← never tested
   - TextEncoder forward+backward ← never tested
   - Full integrated loss with c_kl=1.0 ← never tested
   - **If any of these fails (analog to MPS view-stride bug), STOP and report.**
3. **Phase 3.A** (new, was implicit) — port `SynthesizerTrn.forward()` and a
   training-time RVQ that emits `commit_loss`. ~1 day of work.
4. **Phase 3.B** (new) — port `commons.rand_slice_segments`. Trivial.
5. **Phase 3.2** — MLX DataLoader for r5 manifest.
6. **Phase 3.3** — Multi-epoch loop, LR scheduler (manual ExpLR), logging.
7. **Phase 3.4** — Checkpoint round-trip (G + D + enc_q + optimizer states).
8. **Phase 3.5** — Dry-run smoke (5 steps with KL on, real data).
9. **r6** — Launch + monitor.
10. **Phase E** — Eval with sandhi v1.

## Why this scope is bigger than the user's task description suggests

The Phase 1+2 prototype was deliberately scoped to **isolate the heavy MLX
ops** (Conv1d, ConvTranspose1d, STFT, autograd, AdamW) for a fair speed
comparison vs PyTorch MPS. It was not a "almost ready to train" prototype —
it was a "can we even do these ops" prototype. The speed conclusion was the
deliverable; the missing training-forward path was always the next big step.

The 3 added items below the user's list are real and will take meaningful
time. I'll flag this so they can make an informed decision about whether to
proceed before I burn many hours building.
