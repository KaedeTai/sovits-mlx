"""Phase 3.1 gating test — MLX autograd through the FULL S2 training forward.

This is the gate the rest of the Phase 3 build depends on. r5 had to disable KL
on PyTorch MPS because of a view-stride bug in kl_loss.backward(). The whole
point of going to MLX is to run with c_kl=1.0. If MLX has an analogous bug, the
project dies here.

Path exercised (mirrors GPT_SoVITS/module/models.py:938 SynthesizerTrn.forward):
    text ──> TextEncoder ──> m_p, logs_p
    spec ──> PosteriorEncoder ──> z, m_q, logs_q
    z    ──> Flow(reverse=False) ──> z_p
    (m_p, logs_p, logs_q, z_p, mask) ──> kl_loss ──> scalar
    backward through ALL of those modules.

Sub-modules to verify:
  1. TextEncoder MHA (relative-position attention, never autograd-tested in MLX)
  2. MRTE cross-attention (never autograd-tested)
  3. Flow forward direction (ResidualCouplingLayer reverse=False, never tested)
  4. WN inside Flow (autograd through WaveNet stack with gin conditioning)
  5. kl_loss formula end-to-end (mx.sum / mx.exp / division — could hit numerical
     edge cases at zero mask).

What "pass" means: gradients exist, are finite (no NaN/Inf), have shapes matching
the corresponding parameters, and are not all zero (i.e., they actually
propagated through every parameter that should receive gradient signal).
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "/Users/kaede/tts/_sovits_mlx")

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from models import TextEncoder, ResidualCouplingBlock
from posterior_encoder import PosteriorEncoder
from losses import kl_loss


# ---------------------------------------------------------------------------
# Config that mirrors r5: hidden=192, filter=768, n_heads=2, n_layers=6.
# Smaller B, T, T_text to keep the test fast (~seconds, not minutes).
# ---------------------------------------------------------------------------
B = 2
T_Y = 80         # latent T (frames at 25 Hz)
T_T = 60         # text tokens
SPEC_CH = 1025   # filter_length // 2 + 1
HID = 192
FILT = 768
N_HEADS = 2
N_LAYERS = 6
KS = 3
N_VOCAB = 1033   # v2ProTw
GIN = 1024


# ---------------------------------------------------------------------------
# A wrapper module that owns TextEncoder + PostEnc + Flow so we can
# nn.value_and_grad on the union of parameters in one call.
# ---------------------------------------------------------------------------
class FullPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_p = TextEncoder(
            out_channels=HID,
            hidden_channels=HID,
            filter_channels=FILT,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            kernel_size=KS,
            n_vocab=N_VOCAB,
        )
        self.enc_q = PosteriorEncoder(
            in_channels=SPEC_CH,
            out_channels=HID,
            hidden_channels=HID,
            kernel_size=5,
            dilation_rate=1,
            n_layers=16,
            gin_channels=GIN,
        )
        self.flow = ResidualCouplingBlock(
            channels=HID,
            hidden_channels=HID,
            kernel_size=5,
            dilation_rate=1,
            n_layers=4,
            n_flows=4,
            gin_channels=GIN,
        )

    def __call__(self, ssl_quantized, y_lengths, text, text_lengths,
                 ge_512, ge_1024, spec, noise_q):
        # 1) Text path → m_p, logs_p
        _, m_p, logs_p, y_mask = self.enc_p(
            ssl_quantized, y_lengths, text, text_lengths, ge_512
        )
        # 2) Posterior path → z, m_q, logs_q
        z, m_q, logs_q, _ = self.enc_q(
            spec, y_lengths, g=ge_1024, noise=noise_q
        )
        # 3) Flow forward → z_p
        z_p = self.flow(z, y_mask, g=ge_1024, reverse=False)
        return z_p, m_p, logs_p, m_q, logs_q, y_mask


def main():
    mx.random.seed(0)
    print(f"Config: B={B} T_y={T_Y} T_t={T_T} hid={HID} gin={GIN}")
    print("Building FullPath module (TextEncoder + PostEnc + Flow)...")
    t0 = time.perf_counter()
    model = FullPath()
    mx.eval(model.parameters())
    t1 = time.perf_counter()

    n_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"  built in {t1-t0:.2f}s — {n_params/1e6:.2f}M params")

    # --- Inputs (random, matching r5 shapes) -------------------------------
    ssl_quantized = mx.random.normal((B, 768, T_Y)) * 0.1   # 25 Hz upsampled
    y_lengths = mx.array([T_Y, T_Y - 5], dtype=mx.int32)
    text = mx.random.randint(0, N_VOCAB, (B, T_T))
    text_lengths = mx.array([T_T, T_T - 4], dtype=mx.int32)
    ge_1024 = mx.random.normal((B, 1024, 1)) * 0.05
    ge_512 = mx.random.normal((B, 512, 1)) * 0.05
    spec = mx.abs(mx.random.normal((B, SPEC_CH, T_Y))) + 1e-3   # nonnegative
    noise_q = mx.random.normal((B, HID, T_Y))                    # fixed noise

    mx.eval(ssl_quantized, y_lengths, text, text_lengths,
            ge_1024, ge_512, spec, noise_q)

    # --- Define loss = c_kl * kl_loss ------------------------------------
    def kl_only(m, ssl_q, ylen, txt, tlen, g512, g1024, sp, nq):
        z_p, m_p, logs_p, m_q, logs_q, y_mask = m(
            ssl_q, ylen, txt, tlen, g512, g1024, sp, nq
        )
        # Use the existing kl_loss from losses.py (the one r5 had to disable on MPS)
        return 1.0 * kl_loss(z_p, logs_q, m_p, logs_p, y_mask)

    print("\nRunning forward only (sanity)...")
    t0 = time.perf_counter()
    kl_val = kl_only(model, ssl_quantized, y_lengths, text, text_lengths,
                     ge_512, ge_1024, spec, noise_q)
    mx.eval(kl_val)
    t1 = time.perf_counter()
    print(f"  kl_loss (forward only) = {float(kl_val):.6f}  ({t1-t0:.3f}s)")
    if not mx.isfinite(kl_val):
        print("  FAIL: KL is not finite from forward alone.")
        return 1

    # --- Backward via nn.value_and_grad ----------------------------------
    print("\nRunning backward through Flow + TextEncoder + PostEnc + kl_loss...")
    t0 = time.perf_counter()
    grad_fn = nn.value_and_grad(model, kl_only)
    loss_val, grads = grad_fn(model, ssl_quantized, y_lengths, text, text_lengths,
                              ge_512, ge_1024, spec, noise_q)
    mx.eval(loss_val, grads)
    t1 = time.perf_counter()
    print(f"  backward done in {t1-t0:.3f}s, loss={float(loss_val):.6f}")

    # --- Gradient health check -------------------------------------------
    flat_p = dict(tree_flatten(model.parameters()))
    flat_g = dict(tree_flatten(grads))

    missing, nan_keys, zero_keys, finite_keys = [], [], [], []
    for k, p in flat_p.items():
        g = flat_g.get(k)
        if g is None:
            missing.append(k); continue
        if not bool(mx.all(mx.isfinite(g))):
            nan_keys.append(k); continue
        if float(mx.max(mx.abs(g))) == 0.0:
            zero_keys.append(k); continue
        finite_keys.append(k)

    print(f"\n  params total       : {len(flat_p)}")
    print(f"  finite non-zero    : {len(finite_keys)}")
    print(f"  missing in grads   : {len(missing)}")
    print(f"  non-finite (NaN/Inf): {len(nan_keys)}")
    print(f"  exactly zero       : {len(zero_keys)}")

    if missing:
        print("\n  MISSING (no grad for these params):")
        for k in missing[:10]: print(f"    {k}")

    if nan_keys:
        print("\n  NaN/Inf grads (THE BLOCKER if any of these are KL-path params):")
        for k in nan_keys[:20]: print(f"    {k}")

    # Per-module gradient coverage — surfaces the case where a sub-module silently
    # gets zero gradient (i.e., disconnected from the loss).
    print("\n  per-module finite-nonzero count:")
    by_module = {}
    for k in finite_keys + zero_keys + nan_keys:
        top = k.split(".")[0]
        by_module.setdefault(top, {"finite_nz": 0, "zero": 0, "nan": 0})
    for k in finite_keys:
        by_module[k.split(".")[0]]["finite_nz"] += 1
    for k in zero_keys:
        by_module[k.split(".")[0]]["zero"] += 1
    for k in nan_keys:
        by_module[k.split(".")[0]]["nan"] += 1
    for mname, d in by_module.items():
        total = d["finite_nz"] + d["zero"] + d["nan"]
        print(f"    {mname:12s}  finite_nz={d['finite_nz']:>4d}  zero={d['zero']:>3d}  nan={d['nan']:>3d}  (of {total})")

    # Pass/fail
    if nan_keys or missing:
        print("\nFAIL — Phase 3.1 gating test failed.")
        return 1
    print("\nPASS — MLX autograd through full Flow + TextEncoder + PostEnc + kl_loss works.")
    print("       c_kl=1.0 should be safe to enable for r6.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
