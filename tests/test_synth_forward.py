"""Phase 3.A smoke: SynthesizerTrn.forward (training path) end-to-end.

Builds the full net_g (TextEncoder + PostEnc + Flow + Gen + ref_enc + RVQ),
runs training forward + the full upstream loss formula:

    loss_gen_all = loss_gen + loss_fm + c_mel*mel_l1 + kl_ssl + c_kl*kl_loss

Then runs value_and_grad on the full model and confirms gradients are finite
across all sub-modules. This is the smoke check before we plug into the
multi-epoch loop.
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "/Users/kaede/tts/_sovits_mlx")

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from models import SynthesizerTrn, default_config
from discriminator import MultiPeriodDiscriminator
from losses import generator_loss, feature_loss, kl_loss as kl_loss_fn
import commons as _commons
import mel as mlx_mel


B = 2
T_SSL = 60       # 50 Hz SSL feature length (will be halved to 30 by ssl_proj)
T_Y = 60         # 25 Hz latent T (T_SSL // 2)
T_T = 40
SPEC_CH = 1025
SEG_FRAMES = 32  # latent slice length
HOP = 640
SR = 32000
FFT = 2048
N_MEL = 128
C_MEL = 45.0
C_KL = 1.0


def main():
    mx.random.seed(0)
    cfg = default_config()
    print("Building SynthesizerTrn + MPD...")
    t0 = time.perf_counter()
    net_g = SynthesizerTrn(**cfg)
    net_d = MultiPeriodDiscriminator(version="v2ProTw")
    mx.eval(net_g.parameters(), net_d.parameters())
    n_g = sum(p.size for _, p in tree_flatten(net_g.parameters())) / 1e6
    n_d = sum(p.size for _, p in tree_flatten(net_d.parameters())) / 1e6
    print(f"  built in {time.perf_counter()-t0:.2f}s  net_g={n_g:.1f}M  net_d={n_d:.1f}M")

    # --- random inputs in the shapes of the real dataloader -------------
    ssl = mx.random.normal((B, 768, T_SSL)) * 0.5
    spec = mx.abs(mx.random.normal((B, SPEC_CH, T_Y))) + 1e-3
    y_lengths = mx.array([T_Y, T_Y - 3], dtype=mx.int32)
    text = mx.random.randint(0, 1033, (B, T_T))
    text_lengths = mx.array([T_T, T_T - 4], dtype=mx.int32)
    sv_emb_in = mx.random.normal((B, 20480)) * 0.05
    # Real audio that corresponds to spec — same T_y*hop length.
    wav_full = mx.random.normal((B, 1, T_Y * HOP)) * 0.02
    mx.eval(ssl, spec, y_lengths, text, text_lengths, sv_emb_in, wav_full)

    # --- forward only --------------------------------------------------
    print("\nForward-only smoke...")
    t0 = time.perf_counter()
    slice_key = mx.random.key(42)
    y_hat, kl_ssl, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q), quantized = \
        net_g(ssl, spec, y_lengths, text, text_lengths, sv_emb_in,
              segment_size_frames=SEG_FRAMES, slice_key=slice_key)
    mx.eval(y_hat, kl_ssl, z, z_p, m_p, logs_p, m_q, logs_q, ids_slice)
    print(f"  y_hat shape   : {y_hat.shape}   (expected (B, 1, {SEG_FRAMES*HOP}))")
    print(f"  kl_ssl value  : {float(kl_ssl):.4f}")
    print(f"  z shape       : {z.shape}        (expected (B, 192, T_y))")
    print(f"  z_p shape     : {z_p.shape}")
    print(f"  ids_slice     : {ids_slice.tolist()}")
    print(f"  forward {time.perf_counter()-t0:.2f}s")

    # --- full backward through net_g (training loss) ----------------
    def gen_loss(model_g, model_d, ssl, spec, y_lengths, text, text_lengths,
                 sv_emb_in, wav_full, slice_key):
        y_hat, kl_ssl, ids_slice, x_mask, z_mask, stats, quantized = model_g(
            ssl, spec, y_lengths, text, text_lengths, sv_emb_in,
            segment_size_frames=SEG_FRAMES, slice_key=slice_key,
        )
        z, z_p, m_p, logs_p, m_q, logs_q = stats

        # KL on full latent (matches upstream — no slicing on z_p, m_p, logs_p)
        loss_kl = kl_loss_fn(z_p, logs_q, m_p, logs_p, z_mask) * C_KL

        # Build y_slice (real audio slice corresponding to ids_slice * hop)
        y_real_slice = _commons.slice_segments(
            wav_full, ids_slice * HOP, SEG_FRAMES * HOP
        )

        # mel L1 between mel(y_real_slice) and mel(y_hat)
        mel_real = mlx_mel.mel_spectrogram(
            y_real_slice[:, 0, :], FFT, N_MEL, SR, HOP, FFT, 0.0, None, center=False,
        )
        mel_fake = mlx_mel.mel_spectrogram(
            y_hat[:, 0, :], FFT, N_MEL, SR, HOP, FFT, 0.0, None, center=False,
        )
        Fc = min(mel_real.shape[-1], mel_fake.shape[-1])
        loss_mel = mx.mean(mx.abs(mel_real[..., :Fc] - mel_fake[..., :Fc])) * C_MEL

        # discriminator features
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = model_d(y_real_slice, y_hat)
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, _ = generator_loss(y_d_hat_g)

        loss_all = loss_gen + loss_fm + loss_mel + kl_ssl * 1.0 + loss_kl
        return loss_all, (loss_gen, loss_fm, loss_mel, kl_ssl, loss_kl)

    print("\nFull training-loss backward through net_g (KL on)...")
    t0 = time.perf_counter()
    g_loss_fn = nn.value_and_grad(
        net_g,
        lambda mg, s, sp, yl, t, tl, sv, wav, sk: gen_loss(
            mg, net_d, s, sp, yl, t, tl, sv, wav, sk
        )[0],
    )
    loss_val, grads = g_loss_fn(
        net_g, ssl, spec, y_lengths, text, text_lengths, sv_emb_in, wav_full, slice_key,
    )
    mx.eval(loss_val, grads)
    print(f"  loss_total    : {float(loss_val):.4f}")
    print(f"  backward      : {time.perf_counter()-t0:.2f}s")

    # Component breakdown (forward only, for logging)
    _, comps = gen_loss(net_g, net_d, ssl, spec, y_lengths, text, text_lengths,
                        sv_emb_in, wav_full, slice_key)
    loss_gen_, loss_fm_, loss_mel_, kl_ssl_, loss_kl_ = comps
    print(f"  components    : gen={float(loss_gen_):.3f} fm={float(loss_fm_):.3f} "
          f"mel={float(loss_mel_):.3f} kl_ssl={float(kl_ssl_):.4f} kl={float(loss_kl_):.4f}")

    # --- grad health -----------------------------------------------------
    flat_p = dict(tree_flatten(net_g.parameters()))
    flat_g = dict(tree_flatten(grads))
    nan_keys, zero_keys, finite_nz = [], [], []
    by_mod = {}
    for k, p in flat_p.items():
        g = flat_g.get(k)
        top = k.split(".")[0]
        by_mod.setdefault(top, {"finite_nz": 0, "zero": 0, "nan": 0, "total": 0})
        by_mod[top]["total"] += 1
        if g is None or not bool(mx.all(mx.isfinite(g))):
            nan_keys.append(k); by_mod[top]["nan"] += 1; continue
        if float(mx.max(mx.abs(g))) == 0.0:
            zero_keys.append(k); by_mod[top]["zero"] += 1
        else:
            finite_nz.append(k); by_mod[top]["finite_nz"] += 1

    print(f"\n  total params       : {len(flat_p)}")
    print(f"  finite non-zero    : {len(finite_nz)}")
    print(f"  exactly zero       : {len(zero_keys)}")
    print(f"  NaN/Inf            : {len(nan_keys)}")
    print("  per-module grad coverage:")
    for m, d in by_mod.items():
        print(f"    {m:12s} finite_nz={d['finite_nz']:>4d} "
              f"zero={d['zero']:>3d} nan={d['nan']:>3d} (of {d['total']})")

    if nan_keys:
        print("\nFAIL — NaN/Inf gradients:")
        for k in nan_keys[:10]: print(f"    {k}")
        return 1

    # quantizer is frozen on purpose (stop_gradient) — its 'zero' grads are expected.
    # Anything else should have nonzero grads.
    expected_zero_modules = {"quantizer"}
    unexpected_zero = [k for k in zero_keys if k.split(".")[0] not in expected_zero_modules]
    if unexpected_zero:
        print(f"\nWARN — {len(unexpected_zero)} non-quantizer params have exactly zero grads:")
        for k in unexpected_zero[:10]:
            print(f"    {k}")

    print("\nPASS — SynthesizerTrn.forward + full loss + backward works in MLX.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
