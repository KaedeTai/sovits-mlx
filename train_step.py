"""Single-step training loop prototype + 10-step speed benchmark vs PyTorch MPS.

Scope notes:
- Uses a *slimmed* generator-only training step: random latent z → Generator →
  audio → mel L1 + LSGAN(G) + feature_loss; Discriminator update on (real, fake).
- This isolates the heavy MLX ops (Conv1d, ConvTranspose1d, STFT, Conv2d, autograd,
  AdamW) without dragging in the still-unported posterior/flow/text encoders.
- The TextEncoder/Flow/PosteriorEncoder MLX ops are already exercised in the
  inference path / smoke tests; their backward correctness is checked separately.
- Tensor shapes mirror r5 config: segment_size=20480, T_latent=32, gin=1024.

Why this is a valid speed comparison:
  - The Generator + Discriminator together account for ~95% of S2 step wall time
    upstream (the encoders are a tiny fraction). Benchmarking just them is the
    fair fast-vs-fast comparison the user actually cares about.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, "/Users/kaede/tts/_sovits_mlx")

import numpy as np

# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------
B = 4              # r5 batch size
T_LAT = 32         # latent length (-> 20480 audio samples)
GIN = 1024
SEG = 20480        # segment_size from r5

GEN_KW = dict(
    initial_channel=192, resblock="1",
    resblock_kernel_sizes=[3, 7, 11],
    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    upsample_rates=[10, 8, 2, 2, 2],
    upsample_initial_channel=512,
    upsample_kernel_sizes=[16, 16, 8, 2, 2],
    gin_channels=GIN,
)


def make_inputs(seed: int):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((B, 192, T_LAT)).astype(np.float32)
    cond = rng.standard_normal((B, GIN, 1)).astype(np.float32)
    real_audio = (rng.standard_normal((B, 1, SEG)).astype(np.float32) * 0.05)
    return z, cond, real_audio


# ---------------------------------------------------------------------------
# MLX backend
# ---------------------------------------------------------------------------
def run_mlx(n_steps: int = 10, lr: float = 1e-5, snapshot_weights: bool = False,
            print_each: bool = False):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    from hifigan import Generator
    from discriminator import MultiPeriodDiscriminator
    from losses import (
        generator_loss, discriminator_loss, feature_loss,
    )
    import mel as mlx_mel

    mx.random.seed(0)
    G = Generator(**GEN_KW)
    D = MultiPeriodDiscriminator(version="v2ProTw")
    mx.eval(G.parameters(), D.parameters())

    opt_G = optim.AdamW(learning_rate=lr, betas=[0.8, 0.99], weight_decay=0.01)
    opt_D = optim.AdamW(learning_rate=lr, betas=[0.8, 0.99], weight_decay=0.01)

    # Pre-generate per-step inputs so each step sees a fresh batch (avoids
    # any over-aggressive caching that would underweight Conv compute time).
    inputs = []
    for s in range(n_steps + 4):
        z_np, cond_np, real_np = make_inputs(s + 1)
        inputs.append((mx.array(z_np), mx.array(cond_np), mx.array(real_np)))
    z, cond, real = inputs[0]

    def gen_step_loss(model_G, model_D, z, cond, real_audio):
        # Generator forward
        fake_audio = model_G(z, g=cond)                       # (B, 1, T)
        # Mel L1 (against real)
        mel_real = mlx_mel.mel_spectrogram(
            real_audio[:, 0, :], 2048, 128, 32000, 640, 2048, 0.0, None, center=False,
        )
        mel_fake = mlx_mel.mel_spectrogram(
            fake_audio[:, 0, :], 2048, 128, 32000, 640, 2048, 0.0, None, center=False,
        )
        # Drop spurious last frame if generator audio is longer (it is: 640× vs 640 hop).
        F_common = min(mel_real.shape[-1], mel_fake.shape[-1])
        mel_l1 = mx.mean(mx.abs(mel_real[..., :F_common] - mel_fake[..., :F_common]))

        # Discriminator forward on (real, fake)
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = model_D(real_audio, fake_audio)
        loss_fm = feature_loss(fmap_rs, fmap_gs)
        loss_gen, _ = generator_loss(y_d_gs)
        loss_g_total = loss_gen + loss_fm + 45.0 * mel_l1
        return loss_g_total, (loss_gen, loss_fm, mel_l1)

    def disc_step_loss(model_D, fake_audio_detached, real_audio):
        y_d_rs, y_d_gs, _, _ = model_D(real_audio, fake_audio_detached)
        loss_d, _, _ = discriminator_loss(y_d_rs, y_d_gs)
        return loss_d

    # Snapshot a couple of params for change-detection
    snap_before = None
    if snapshot_weights:
        snap_before = {
            "G.conv_pre.weight": mx.array(G.conv_pre.conv.weight),
            "D.disc0.conv0.weight": mx.array(D.discriminators[0].convs[0].conv.weight),
        }

    def g_loss_one_step(model_G, z, cond, real):
        return gen_step_loss(model_G, D, z, cond, real)[0]

    g_loss_fn = nn.value_and_grad(G, g_loss_one_step)
    d_loss_fn = nn.value_and_grad(
        D,
        lambda m, fake, r: disc_step_loss(m, fake, r),
    )

    losses_log = []
    # Warmup: 2 full training steps on the warmup batches.
    for wi in range(2):
        zw, cw, rw = inputs[n_steps + wi]
        loss_g, grads_g = g_loss_fn(G, zw, cw, rw)
        opt_G.update(G, grads_g); mx.eval(G.parameters(), opt_G.state, loss_g)
        fa = mx.stop_gradient(G(zw, g=cw))
        loss_d, grads_d = d_loss_fn(D, fa, rw)
        opt_D.update(D, grads_d); mx.eval(D.parameters(), opt_D.state, loss_d)

    t0 = time.perf_counter()
    for step in range(n_steps):
        z_s, c_s, r_s = inputs[step]
        # --- G step ---
        loss_g, grads_g = g_loss_fn(G, z_s, c_s, r_s)
        opt_G.update(G, grads_g)
        mx.eval(G.parameters(), opt_G.state, loss_g)

        # --- D step (recompute fake with updated G, then detach) ---
        fake_audio = G(z_s, g=c_s)
        fake_audio_detached = mx.stop_gradient(fake_audio)
        loss_d, grads_d = d_loss_fn(D, fake_audio_detached, r_s)
        opt_D.update(D, grads_d)
        mx.eval(D.parameters(), opt_D.state, loss_d)

        losses_log.append((float(loss_g), float(loss_d)))
        if print_each:
            print(f"  [MLX] step {step}: G={float(loss_g):.4f} D={float(loss_d):.4f}")
    t1 = time.perf_counter()
    sec_per_step = (t1 - t0) / n_steps

    snap_after = None
    if snapshot_weights:
        snap_after = {
            "G.conv_pre.weight": mx.array(G.conv_pre.conv.weight),
            "D.disc0.conv0.weight": mx.array(D.discriminators[0].convs[0].conv.weight),
        }
        for k in snap_before:
            d = float(mx.max(mx.abs(snap_after[k] - snap_before[k])))
            print(f"  [MLX] weight delta '{k}': maxabs={d:.3e}")
    return sec_per_step, losses_log


# ---------------------------------------------------------------------------
# PyTorch MPS backend (equivalent slimmed step)
# ---------------------------------------------------------------------------
def run_torch(n_steps: int = 10, lr: float = 1e-5, snapshot_weights: bool = False,
              print_each: bool = False, device: str = "mps"):
    sys.path.insert(0, "/Users/kaede/tts/GPT-SoVITS")
    sys.path.insert(0, "/Users/kaede/tts/GPT-SoVITS/GPT_SoVITS")
    import torch
    from GPT_SoVITS.module.models import Generator as PT_Generator
    from GPT_SoVITS.module.models import MultiPeriodDiscriminator as PT_MPD
    from GPT_SoVITS.module.losses import (
        generator_loss as pt_gen_loss,
        discriminator_loss as pt_disc_loss,
        feature_loss as pt_feature_loss,
    )
    from GPT_SoVITS.module.mel_processing import mel_spectrogram_torch

    dev = torch.device(device)
    G = PT_Generator(
        initial_channel=192, resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[10, 8, 2, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[16, 16, 8, 2, 2],
        gin_channels=GIN,
    ).to(dev)
    D = PT_MPD(use_spectral_norm=False, version="v2ProTw").to(dev)

    opt_G = torch.optim.AdamW(G.parameters(), lr=lr, betas=(0.8, 0.99), weight_decay=0.01)
    opt_D = torch.optim.AdamW(D.parameters(), lr=lr, betas=(0.8, 0.99), weight_decay=0.01)

    inputs = []
    for s in range(n_steps + 4):
        z_np, cond_np, real_np = make_inputs(s + 1)
        inputs.append((
            torch.from_numpy(z_np).to(dev),
            torch.from_numpy(cond_np).to(dev),
            torch.from_numpy(real_np).to(dev),
        ))
    z, cond, real = inputs[0]

    snap_before = None
    if snapshot_weights:
        snap_before = {
            "G.conv_pre.weight": G.conv_pre.weight.detach().clone(),
            "D.disc0.conv0.weight_v": D.discriminators[0].convs[0].weight_v.detach().clone(),
        }

    losses_log = []
    # Warmup: 2 full training steps so MPS caches kernels.
    for wi in range(2):
        zw, cw, rw = inputs[n_steps + wi]
        opt_G.zero_grad(set_to_none=True)
        fake = G(zw, g=cw)
        mel_real = mel_spectrogram_torch(rw[:, 0, :], 2048, 128, 32000, 640, 2048, 0.0, None, center=False)
        mel_fake = mel_spectrogram_torch(fake[:, 0, :], 2048, 128, 32000, 640, 2048, 0.0, None, center=False)
        Fc = min(mel_real.shape[-1], mel_fake.shape[-1])
        l1 = (mel_real[..., :Fc] - mel_fake[..., :Fc]).abs().mean()
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = D(rw, fake)
        L = pt_gen_loss(y_d_gs)[0] + pt_feature_loss(fmap_rs, fmap_gs) + 45.0 * l1
        L.backward(); opt_G.step()
        opt_D.zero_grad(set_to_none=True)
        with torch.no_grad():
            fake = G(zw, g=cw)
        y_d_rs, y_d_gs, _, _ = D(rw, fake.detach())
        ld, _, _ = pt_disc_loss(y_d_rs, y_d_gs); ld.backward(); opt_D.step()
        if device == "mps":
            torch.mps.synchronize()

    t0 = time.perf_counter()
    for step in range(n_steps):
        z, cond, real = inputs[step]
        # --- G step ---
        opt_G.zero_grad(set_to_none=True)
        fake = G(z, g=cond)
        mel_real = mel_spectrogram_torch(real[:, 0, :], 2048, 128, 32000, 640, 2048, 0.0, None, center=False)
        mel_fake = mel_spectrogram_torch(fake[:, 0, :], 2048, 128, 32000, 640, 2048, 0.0, None, center=False)
        F_common = min(mel_real.shape[-1], mel_fake.shape[-1])
        mel_l1 = (mel_real[..., :F_common] - mel_fake[..., :F_common]).abs().mean()
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = D(real, fake)
        loss_fm = pt_feature_loss(fmap_rs, fmap_gs)
        loss_gen, _ = pt_gen_loss(y_d_gs)
        loss_G = loss_gen + loss_fm + 45.0 * mel_l1
        loss_G.backward()
        opt_G.step()

        # --- D step ---
        opt_D.zero_grad(set_to_none=True)
        with torch.no_grad():
            fake = G(z, g=cond)
        y_d_rs, y_d_gs, _, _ = D(real, fake.detach())
        loss_D, _, _ = pt_disc_loss(y_d_rs, y_d_gs)
        loss_D.backward()
        opt_D.step()

        if device == "mps":
            torch.mps.synchronize()
        losses_log.append((float(loss_G.item()), float(loss_D.item())))
        if print_each:
            print(f"  [PT-{device}] step {step}: G={float(loss_G):.4f} D={float(loss_D):.4f}")
    t1 = time.perf_counter()
    sec_per_step = (t1 - t0) / n_steps

    if snapshot_weights:
        for k, vb in snap_before.items():
            if "weight_v" in k:
                va = D.discriminators[0].convs[0].weight_v.detach()
            else:
                va = G.conv_pre.weight.detach()
            d = (va - vb).abs().max().item()
            print(f"  [PT-{device}] weight delta '{k}': maxabs={d:.3e}")
    return sec_per_step, losses_log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--backends", nargs="+", default=["mlx", "pt-mps"],
                    choices=["mlx", "pt-mps", "pt-cpu"])
    ap.add_argument("--print-each", action="store_true")
    ap.add_argument("--snapshot", action="store_true")
    args = ap.parse_args()

    results = {}
    for be in args.backends:
        print(f"\n=== Backend: {be} ===")
        if be == "mlx":
            spp, log = run_mlx(args.steps, snapshot_weights=args.snapshot,
                               print_each=args.print_each)
        else:
            dev = "mps" if be == "pt-mps" else "cpu"
            spp, log = run_torch(args.steps, snapshot_weights=args.snapshot,
                                 print_each=args.print_each, device=dev)
        results[be] = (spp, log)
        print(f"  sec/step over {args.steps}: {spp:.4f}")
        print(f"  first G/D losses : {log[0]}")
        print(f"  last  G/D losses : {log[-1]}")

    print("\n=== Summary ===")
    for be, (spp, log) in results.items():
        print(f"  {be:8s}  sec/step={spp:.4f}  step/sec={1.0/spp:.3f}")
    if "mlx" in results and "pt-mps" in results:
        spp_mlx, _ = results["mlx"]
        spp_pt, _ = results["pt-mps"]
        ratio = spp_pt / spp_mlx
        print(f"\n  MLX vs PyTorch-MPS speedup = {ratio:.2f}× "
              f"({'MLX faster' if ratio > 1 else 'PyTorch-MPS faster'})")
