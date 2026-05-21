"""MLX port of HiFi-GAN-style multi-period + multi-scale discriminator.

Mirrors module.models {DiscriminatorS, DiscriminatorP, MultiPeriodDiscriminator}.

Note on weight_norm
-------------------
PyTorch wraps each Conv with `nn.utils.weight_norm`, which is a *param-only
re-parameterization*: in the forward pass the conv just sees a regular weight
`w = g * v / ||v||`.  We therefore use plain nn.Conv1d/Conv2d here and the
checkpoint converter must collapse `weight_v, weight_g -> weight` on load.

Note on layout
--------------
PyTorch convs use (B, C, H, W) / (B, C, T); MLX convs use (B, H, W, C) /
(B, T, C).  We accept and return PyTorch-style (B, C, T) at the API boundary
and only transpose internally.

For v2Pro the discriminator periods are [2, 3, 5, 7, 11, 17, 23].
"""

from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn

from modules import LRELU_SLOPE, get_padding


# ---------------------------------------------------------------------------
# Conv1d / Conv2d PT-layout wrappers (no weight_norm; see module docstring)
# ---------------------------------------------------------------------------
class _Conv2dPT(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, padding, groups=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups,
                              bias=bias)

    def __call__(self, x):
        # x : (B, C, H, W) -> (B, H, W, C)
        x = x.transpose(0, 2, 3, 1)
        x = self.conv(x)
        return x.transpose(0, 3, 1, 2)


class _Conv1dPT(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, groups=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c, kernel_size=kernel_size, stride=stride,
                              padding=padding, groups=groups, bias=bias)

    def __call__(self, x):
        x = x.transpose(0, 2, 1)
        x = self.conv(x)
        return x.transpose(0, 2, 1)


def _reflect_pad_last(x: mx.array, pad: int) -> mx.array:
    """Reflect-pad the last axis by `pad` samples on the right."""
    if pad <= 0:
        return x
    # Build a reversed tail of length `pad`, drop the boundary element.
    # x[..., -2:-pad-2:-1] is the inner reflection (skips the last sample).
    # Equivalent to F.pad(x, (0, pad), mode='reflect').
    tail = x[..., -2:-(pad + 2):-1]
    return mx.concatenate([x, tail], axis=-1)


class DiscriminatorP(nn.Module):
    """1D-input, internally reshapes to (B, 1, T/p, p) then 2D convs along T."""

    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3,
                 use_spectral_norm: bool = False):
        super().__init__()
        self.period = period
        # PyTorch padding=(get_padding(kernel_size, 1), 0) -> (2, 0) for k=5
        p_h = get_padding(kernel_size, 1)
        self.convs = [
            _Conv2dPT(1,    32,   (kernel_size, 1), (stride, 1), (p_h, 0)),
            _Conv2dPT(32,   128,  (kernel_size, 1), (stride, 1), (p_h, 0)),
            _Conv2dPT(128,  512,  (kernel_size, 1), (stride, 1), (p_h, 0)),
            _Conv2dPT(512,  1024, (kernel_size, 1), (stride, 1), (p_h, 0)),
            _Conv2dPT(1024, 1024, (kernel_size, 1), 1,           (p_h, 0)),
        ]
        self.conv_post = _Conv2dPT(1024, 1, (3, 1), 1, (1, 0))

    def __call__(self, x: mx.array) -> Tuple[mx.array, List[mx.array]]:
        fmap: List[mx.array] = []
        b, c, t = x.shape
        # Pad time so it's a multiple of period (reflect on the right).
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = _reflect_pad_last(x, n_pad)
            t = t + n_pad
        # (B, C, T) -> (B, C, T/p, p)
        x = x.reshape(b, c, t // self.period, self.period)
        for layer in self.convs:
            x = layer(x)
            x = nn.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = x.reshape(b, -1)
        return x, fmap


class DiscriminatorS(nn.Module):
    """1D-input, 1D-conv scale discriminator."""

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        self.convs = [
            _Conv1dPT(1,    16,   15, 1, padding=7),
            _Conv1dPT(16,   64,   41, 4, groups=4,   padding=20),
            _Conv1dPT(64,   256,  41, 4, groups=16,  padding=20),
            _Conv1dPT(256,  1024, 41, 4, groups=64,  padding=20),
            _Conv1dPT(1024, 1024, 41, 4, groups=256, padding=20),
            _Conv1dPT(1024, 1024, 5,  1, padding=2),
        ]
        self.conv_post = _Conv1dPT(1024, 1, 3, 1, padding=1)

    def __call__(self, x: mx.array) -> Tuple[mx.array, List[mx.array]]:
        fmap: List[mx.array] = []
        for layer in self.convs:
            x = layer(x)
            x = nn.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        b = x.shape[0]
        x = x.reshape(b, -1)
        return x, fmap


_V2PRO_SET = {"v2Pro", "v2ProPlus", "v2ProTw", "v2ProPlusTw"}


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm: bool = False, version: str = "v2ProTw"):
        super().__init__()
        if version in _V2PRO_SET:
            periods = [2, 3, 5, 7, 11, 17, 23]
        else:
            periods = [2, 3, 5, 7, 11]
        self.discriminators = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        for p in periods:
            self.discriminators.append(
                DiscriminatorP(p, use_spectral_norm=use_spectral_norm)
            )

    def __call__(self, y: mx.array, y_hat: mx.array):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
