"""MLX port of module.losses for GPT-SoVITS S2 training.

LSGAN-style discriminator + generator losses, feature-matching, KL flow loss.
All inputs are mx.array; results are scalar mx.array.

PyTorch reference: GPT_SoVITS/module/losses.py
"""

from typing import List, Tuple

import mlx.core as mx


def feature_loss(fmap_r: List[List[mx.array]],
                 fmap_g: List[List[mx.array]]) -> mx.array:
    """L1 feature-matching loss across all discriminator sub-stacks."""
    loss = mx.array(0.0)
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            # PyTorch detaches real-side features; in MLX we mirror that by
            # calling mx.stop_gradient on the real branch.
            rl = mx.stop_gradient(rl)
            loss = loss + mx.mean(mx.abs(rl - gl))
    return loss * 2.0


def discriminator_loss(
    disc_real_outputs: List[mx.array],
    disc_generated_outputs: List[mx.array],
) -> Tuple[mx.array, List[float], List[float]]:
    """LSGAN discriminator loss.  D should output 1 for real, 0 for fake.

    PyTorch returns (loss, r_losses, g_losses) where the sub-lists are .item()
    floats for logging.  In MLX we accumulate scalars and only convert to
    Python float on demand.
    """
    loss = mx.array(0.0)
    r_losses: List[float] = []
    g_losses: List[float] = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = mx.mean((1.0 - dr) ** 2)
        g_loss = mx.mean(dg ** 2)
        loss = loss + r_loss + g_loss
        r_losses.append(r_loss)
        g_losses.append(g_loss)
    return loss, r_losses, g_losses


def generator_loss(
    disc_outputs: List[mx.array],
) -> Tuple[mx.array, List[mx.array]]:
    """LSGAN generator loss.  G wants D to output 1 for its fakes."""
    loss = mx.array(0.0)
    gen_losses: List[mx.array] = []
    for dg in disc_outputs:
        l = mx.mean((1.0 - dg) ** 2)
        gen_losses.append(l)
        loss = loss + l
    return loss, gen_losses


def kl_loss(
    z_p: mx.array,    # (B, H, T)
    logs_q: mx.array, # (B, H, T)
    m_p: mx.array,    # (B, H, T)
    logs_p: mx.array, # (B, H, T)
    z_mask: mx.array, # (B, 1, T)
) -> mx.array:
    """KL divergence between q(z|x) and p(z|c) — VITS-style flow loss."""
    kl = logs_p - logs_q - 0.5
    kl = kl + 0.5 * ((z_p - m_p) ** 2) * mx.exp(-2.0 * logs_p)
    kl = mx.sum(kl * z_mask)
    return kl / mx.sum(z_mask)
