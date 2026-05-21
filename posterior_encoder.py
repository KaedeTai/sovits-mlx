"""MLX PosteriorEncoder (enc_q) for GPT-SoVITS S2 training.

Mirrors module.models.PosteriorEncoder.  Encodes a linear spectrogram into
the posterior latent (z, m_q, logs_q) conditioned on the speaker embedding g.

PyTorch reference:
    pre  = Conv1d(in_channels=spec_channels, out_channels=hidden_channels, k=1)
    enc  = WN(hidden_channels, kernel_size=5, dilation_rate=1, n_layers=16,
              gin_channels=gin_channels)
    proj = Conv1d(hidden_channels, out_channels*2, k=1)

    forward(x, x_lengths, g):
        x_mask = sequence_mask(x_lengths)
        x = pre(x) * x_mask
        x = enc(x, x_mask, g=g)
        stats = proj(x) * x_mask
        m, logs = split(stats, out_channels, dim=1)
        z = (m + randn_like(m) * exp(logs)) * x_mask
        return z, m, logs, x_mask
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from modules import Conv1dPT, WN, sequence_mask


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1025,
        out_channels: int = 192,
        hidden_channels: int = 192,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        n_layers: int = 16,
        gin_channels: int = 1024,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.pre = Conv1dPT(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers,
                      gin_channels=gin_channels)
        self.proj = Conv1dPT(hidden_channels, out_channels * 2, 1)

    def __call__(
        self,
        x: mx.array,                       # (B, in_channels, T)
        x_lengths: mx.array,               # (B,) int
        g: Optional[mx.array] = None,      # (B, gin_channels, 1)
        noise: Optional[mx.array] = None,  # optional injected noise for testing
    ):
        x_mask = sequence_mask(x_lengths, x.shape[2])
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m = stats[:, : self.out_channels, :]
        logs = stats[:, self.out_channels :, :]
        if noise is None:
            noise = mx.random.normal(m.shape)
        z = (m + noise * mx.exp(logs)) * x_mask
        return z, m, logs, x_mask
