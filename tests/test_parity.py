"""PyTorch ↔ MLX numerical-equivalence check (1 step).

Builds Generator+Discriminator in PyTorch (random init), materializes the
parametrized weights, copies them into MLX, then computes the same
G+D losses on the same input batch on both backends.

We expect agreement to within ~1e-3 absolute (large numbers, accumulated
matmuls; the discriminator alone has ~50 convs on float32).
"""

import sys, time
sys.path.insert(0, "/Users/kaede/tts/_sovits_mlx")
sys.path.insert(0, "/Users/kaede/tts/GPT-SoVITS")
sys.path.insert(0, "/Users/kaede/tts/GPT-SoVITS/GPT_SoVITS")

import numpy as np
import torch
import mlx.core as mx
import mlx.nn as nn

from GPT_SoVITS.module.models import (
    Generator as PT_Generator,
    MultiPeriodDiscriminator as PT_MPD,
)
from GPT_SoVITS.module.losses import (
    generator_loss as pt_gen_loss,
    discriminator_loss as pt_disc_loss,
    feature_loss as pt_feature_loss,
)
from GPT_SoVITS.module.mel_processing import mel_spectrogram_torch

from hifigan import Generator as MX_Generator
from discriminator import MultiPeriodDiscriminator as MX_MPD
from losses import (
    generator_loss as mx_gen_loss,
    discriminator_loss as mx_disc_loss,
    feature_loss as mx_feature_loss,
)
import mel as mlx_mel

GEN_KW = dict(
    initial_channel=192, resblock="1",
    resblock_kernel_sizes=[3, 7, 11],
    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    upsample_rates=[10, 8, 2, 2, 2],
    upsample_initial_channel=512,
    upsample_kernel_sizes=[16, 16, 8, 2, 2],
    gin_channels=1024,
)


def materialize_wn(weight_g, weight_v):
    g = weight_g.squeeze()
    norm = torch.norm(weight_v.reshape(weight_v.shape[0], -1), dim=1)
    return weight_v * (g / norm).reshape((-1,) + (1,) * (weight_v.dim() - 1))


def to_mx_conv(w_pt: torch.Tensor) -> mx.array:
    arr = w_pt.detach().cpu().numpy()
    if arr.ndim == 3:
        return mx.array(np.transpose(arr, (0, 2, 1)))
    if arr.ndim == 4:
        return mx.array(np.transpose(arr, (0, 2, 3, 1)))
    raise ValueError(arr.shape)


# ---------------------------------------------------------------------------
# Weight transfer: PyTorch Generator -> MLX Generator
# ---------------------------------------------------------------------------
def transfer_generator(pt_g, mx_g):
    # conv_pre, conv_post, cond: plain Conv1d in upstream Generator
    mx_g.conv_pre.conv.weight = to_mx_conv(pt_g.conv_pre.weight)
    mx_g.conv_pre.conv.bias = mx.array(pt_g.conv_pre.bias.detach().cpu().numpy())
    mx_g.conv_post.conv.weight = to_mx_conv(pt_g.conv_post.weight)
    # conv_post in MLX port uses bias=False; upstream too (we verified)
    # cond (plain Conv1d)
    if hasattr(pt_g, "cond") and pt_g.cond is not None:
        mx_g.cond.conv.weight = to_mx_conv(pt_g.cond.weight)
        mx_g.cond.conv.bias = mx.array(pt_g.cond.bias.detach().cpu().numpy())
    # ups: ConvTranspose1d, weight_norm wrapped
    for i, pt_u in enumerate(pt_g.ups):
        w = materialize_wn(pt_u.weight_g, pt_u.weight_v)        # (in, out, K)
        # MLX ConvTranspose1d weight: (out, K, in)
        arr = w.detach().cpu().numpy()
        mx_g.ups[i].conv_t.weight = mx.array(np.transpose(arr, (1, 2, 0)))
        mx_g.ups[i].conv_t.bias = mx.array(pt_u.bias.detach().cpu().numpy())
    # resblocks
    for ri, pt_rb in enumerate(pt_g.resblocks):
        for ci in range(len(pt_rb.convs1)):
            w1 = materialize_wn(pt_rb.convs1[ci].weight_g, pt_rb.convs1[ci].weight_v)
            mx_g.resblocks[ri].convs1[ci].conv.weight = to_mx_conv(w1)
            mx_g.resblocks[ri].convs1[ci].conv.bias = mx.array(pt_rb.convs1[ci].bias.detach().cpu().numpy())
            w2 = materialize_wn(pt_rb.convs2[ci].weight_g, pt_rb.convs2[ci].weight_v)
            mx_g.resblocks[ri].convs2[ci].conv.weight = to_mx_conv(w2)
            mx_g.resblocks[ri].convs2[ci].conv.bias = mx.array(pt_rb.convs2[ci].bias.detach().cpu().numpy())


def transfer_disc(pt_d, mx_d):
    for pt_sub, mx_sub in zip(pt_d.discriminators, mx_d.discriminators):
        for pt_c, mx_c in zip(pt_sub.convs, mx_sub.convs):
            w = materialize_wn(pt_c.weight_g, pt_c.weight_v)
            mx_c.conv.weight = to_mx_conv(w)
            mx_c.conv.bias = mx.array(pt_c.bias.detach().cpu().numpy())
        w = materialize_wn(pt_sub.conv_post.weight_g, pt_sub.conv_post.weight_v)
        mx_sub.conv_post.conv.weight = to_mx_conv(w)
        mx_sub.conv_post.conv.bias = mx.array(pt_sub.conv_post.bias.detach().cpu().numpy())


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(0)
    pt_G = PT_Generator(**GEN_KW)
    pt_D = PT_MPD(use_spectral_norm=False, version="v2ProTw")
    pt_G.eval(); pt_D.eval()

    mx_G = MX_Generator(**GEN_KW)
    mx_D = MX_MPD(version="v2ProTw")
    mx.eval(mx_G.parameters(), mx_D.parameters())

    transfer_generator(pt_G, mx_G)
    transfer_disc(pt_D, mx_D)
    mx.eval(mx_G.parameters(), mx_D.parameters())

    B, T_LAT, GIN, SEG = 4, 32, 1024, 20480
    rng = np.random.default_rng(0)
    z_np    = rng.standard_normal((B, 192, T_LAT)).astype(np.float32)
    cond_np = rng.standard_normal((B, GIN, 1)).astype(np.float32)
    real_np = rng.standard_normal((B, 1, SEG)).astype(np.float32) * 0.05

    # PyTorch forward (CPU for max parity)
    with torch.no_grad():
        z_pt, cond_pt, real_pt = (torch.from_numpy(x) for x in (z_np, cond_np, real_np))
        fake_pt = pt_G(z_pt, g=cond_pt)
        mel_real_pt = mel_spectrogram_torch(real_pt[:, 0, :], 2048, 128, 32000, 640, 2048, 0.0, None, center=False)
        mel_fake_pt = mel_spectrogram_torch(fake_pt[:, 0, :], 2048, 128, 32000, 640, 2048, 0.0, None, center=False)
        F_c = min(mel_real_pt.shape[-1], mel_fake_pt.shape[-1])
        mel_l1_pt = (mel_real_pt[..., :F_c] - mel_fake_pt[..., :F_c]).abs().mean().item()
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = pt_D(real_pt, fake_pt)
        fm_pt = pt_feature_loss(fmap_rs, fmap_gs).item()
        lg_pt, _ = pt_gen_loss(y_d_gs); lg_pt = lg_pt.item()
        ld_pt, _, _ = pt_disc_loss(y_d_rs, y_d_gs); ld_pt = ld_pt.item()
        total_G_pt = lg_pt + fm_pt + 45.0 * mel_l1_pt

    # MLX forward
    z, cond, real = mx.array(z_np), mx.array(cond_np), mx.array(real_np)
    fake_mx = mx_G(z, g=cond)
    mel_real_mx = mlx_mel.mel_spectrogram(real[:, 0, :], 2048, 128, 32000, 640, 2048, 0.0, None, center=False)
    mel_fake_mx = mlx_mel.mel_spectrogram(fake_mx[:, 0, :], 2048, 128, 32000, 640, 2048, 0.0, None, center=False)
    F_c = min(mel_real_mx.shape[-1], mel_fake_mx.shape[-1])
    mel_l1_mx = float(mx.mean(mx.abs(mel_real_mx[..., :F_c] - mel_fake_mx[..., :F_c])))
    y_d_rs, y_d_gs, fmap_rs, fmap_gs = mx_D(real, fake_mx)
    fm_mx = float(mx_feature_loss(fmap_rs, fmap_gs))
    lg_mx, _ = mx_gen_loss(y_d_gs); lg_mx = float(lg_mx)
    ld_mx, _, _ = mx_disc_loss(y_d_rs, y_d_gs); ld_mx = float(ld_mx)
    total_G_mx = lg_mx + fm_mx + 45.0 * mel_l1_mx

    # Compare
    rows = [
        ("mel_l1", mel_l1_pt, mel_l1_mx),
        ("feature_loss", fm_pt, fm_mx),
        ("gen_loss", lg_pt, lg_mx),
        ("disc_loss", ld_pt, ld_mx),
        ("G_total", total_G_pt, total_G_mx),
    ]
    print(f"{'metric':<14} {'PyTorch':>14} {'MLX':>14} {'abs diff':>12} {'rel diff':>12}")
    print("-" * 72)
    max_rel = 0.0
    for name, pt_v, mx_v in rows:
        ad = abs(pt_v - mx_v)
        rd = ad / (abs(pt_v) + 1e-9)
        max_rel = max(max_rel, rd)
        print(f"{name:<14} {pt_v:>14.6f} {mx_v:>14.6f} {ad:>12.3e} {rd:>12.3e}")
    print(f"\nworst relative diff: {max_rel:.3e}")
    ok = max_rel < 1e-2
    print("PASS" if ok else "WORSE THAN 1e-2 — INVESTIGATE")


if __name__ == "__main__":
    main()
