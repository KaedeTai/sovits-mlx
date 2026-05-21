"""MelStyleEncoder — speaker style encoder.

Port of module.modules.MelStyleEncoder.  Operates on (B, 704, T_ref) spec slice.
Outputs a (B, 1024, 1) speaker embedding ('ge') after a temporal avg-pool.
"""

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from modules import Conv1dPT


class LinearNorm(nn.Module):
    """Thin wrapper around nn.Linear with attribute name `fc` (mirrors PT)."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias=bias)

    def __call__(self, x):
        return self.fc(x)


class Mish(nn.Module):
    def __call__(self, x):
        return x * mx.tanh(nn.softplus(x))


class Conv1dGLU(nn.Module):
    """Conv1d + GLU(Gated Linear Unit) with residual connection.  Input/output (B, C, T)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.out_channels = out_channels
        pad = (kernel_size - 1) // 2
        # ConvNorm wrapper in PT: self.conv1.conv (we matched naming in convert.py)
        # Here we name it `conv1` and inside it a `conv` Conv1dPT to match PT key path.
        class _ConvNorm(nn.Module):
            def __init__(self, in_ch, out_ch, k, p):
                super().__init__()
                self.conv = Conv1dPT(in_ch, out_ch, k, padding=p)
            def __call__(self, x):
                return self.conv(x)
        self.conv1 = _ConvNorm(in_channels, 2 * out_channels, kernel_size, pad)

    def __call__(self, x):
        residual = x
        h = self.conv1(x)
        x1 = h[:, : self.out_channels, :]
        x2 = h[:, self.out_channels :, :]
        return residual + (x1 * mx.sigmoid(x2))


class MHA_Linear(nn.Module):
    """Multi-Head Attention with Linear (not Conv1d) projections, time-major (B, T, C).

    Matches module.modules.MultiHeadAttention (different class from attentions.MultiHeadAttention).
    Used by MelStyleEncoder.
    """

    def __init__(self, n_head: int, d_model: int, d_k: int, d_v: int):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        self.fc = nn.Linear(n_head * d_v, d_model)
        self.temperature = math.sqrt(d_model)

    def __call__(self, x, mask=None):
        # x: (B, T, C)
        b, t, _ = x.shape
        residual = x
        q = self.w_qs(x).reshape(b, t, self.n_head, self.d_k)
        k = self.w_ks(x).reshape(b, t, self.n_head, self.d_k)
        v = self.w_vs(x).reshape(b, t, self.n_head, self.d_v)
        # to (n*b, t, d) via permute(2,0,1,3) + reshape
        q = q.transpose(0, 2, 1, 3).reshape(b * self.n_head, t, self.d_k)
        k = k.transpose(0, 2, 1, 3).reshape(b * self.n_head, t, self.d_k)
        v = v.transpose(0, 2, 1, 3).reshape(b * self.n_head, t, self.d_v)

        # bmm: scores = (q @ k^T) / temperature
        scores = mx.matmul(q, k.transpose(0, 2, 1)) / self.temperature
        if mask is not None:
            # mask shape (n*b, t, t)
            scores = mx.where(mask, mx.array(-1e9, scores.dtype), scores)
        attn = mx.softmax(scores, axis=2)
        out = mx.matmul(attn, v)                                  # (n*b, t, d_v)
        out = out.reshape(self.n_head, b, t, self.d_v).transpose(1, 2, 0, 3).reshape(b, t, -1)
        out = self.fc(out)
        # No residual+dropout in eval mode (dropout disabled) -- match PT inference path
        return out + residual


class MelStyleEncoder(nn.Module):
    def __init__(
        self,
        n_mel_channels: int = 704,
        style_hidden: int = 128,
        style_vector_dim: int = 1024,
        style_kernel_size: int = 5,
        style_head: int = 2,
    ):
        super().__init__()
        self.in_dim = n_mel_channels
        self.hidden_dim = style_hidden
        self.out_dim = style_vector_dim

        # spectral: PyTorch Sequential[LinearNorm(0), Mish(1), Dropout(2),
        #                             LinearNorm(3), Mish(4), Dropout(5)]
        # → MLX flat list, where stateless slots are placeholder _NoParam modules so
        #   load_weights aligns spectral.0/spectral.3 to indices 0/3.
        self.spectral = [
            LinearNorm(n_mel_channels, style_hidden),  # 0
            _NoParam(),                                 # 1 (Mish in PT, but stateless)
            _NoParam(),                                 # 2 (Dropout, stateless)
            LinearNorm(style_hidden, style_hidden),    # 3
            _NoParam(),                                 # 4
            _NoParam(),                                 # 5
        ]

        # temporal: Sequential[Conv1dGLU(0), Conv1dGLU(1)]
        self.temporal = [
            Conv1dGLU(style_hidden, style_hidden, style_kernel_size),
            Conv1dGLU(style_hidden, style_hidden, style_kernel_size),
        ]

        # self-attention
        self.slf_attn = MHA_Linear(
            n_head=style_head,
            d_model=style_hidden,
            d_k=style_hidden // style_head,
            d_v=style_hidden // style_head,
        )

        # fc out
        self.fc = LinearNorm(style_hidden, style_vector_dim)

    def __call__(self, x, mask=None):
        # x: (B, 704, T_ref).  We do not pass a mask in our decode path.
        x = x.transpose(0, 2, 1)            # (B, T_ref, 704)
        x = self.spectral[0](x)             # LinearNorm
        x = x * mx.tanh(nn.softplus(x))     # Mish
        x = self.spectral[3](x)             # LinearNorm
        x = x * mx.tanh(nn.softplus(x))     # Mish
        x = x.transpose(0, 2, 1)            # (B, 128, T_ref)
        x = self.temporal[0](x)
        x = self.temporal[1](x)
        x = x.transpose(0, 2, 1)            # (B, T_ref, 128)
        x = self.slf_attn(x, mask=None)     # (B, T_ref, 128)
        x = self.fc(x)                      # (B, T_ref, 1024)
        w = mx.mean(x, axis=1)              # temporal avg pool
        return mx.expand_dims(w, -1)        # (B, 1024, 1)


# ----- holders that expose list-indexed children to match PT key indices ----
class _NoParam(nn.Module):
    """Placeholder for stateless slots (Mish/Dropout) so list indices align with PT keys."""

    def __call__(self, x):
        return x


class _SpectralHolder(nn.Module):
    """Holds the 'spectral' Sequential block; PT keys are spectral.0.fc.* and spectral.3.fc.*.

    Implemented as a list so load_weights can use integer indexing.
    """

    def __init__(self, n_mel_channels: int, style_hidden: int):
        super().__init__()
        # Indices 0,3: LinearNorm.  Indices 1,2,4,5: Mish/Dropout (no params).
        # We expose all 6 slots so safetensor keys spectral.0.* and spectral.3.* resolve.
        self.children_list = [
            LinearNorm(n_mel_channels, style_hidden),
            _NoParam(),
            _NoParam(),
            LinearNorm(style_hidden, style_hidden),
            _NoParam(),
            _NoParam(),
        ]

    def __call__(self, x):
        x = self.children_list[0](x)        # linear
        x = x * mx.tanh(nn.softplus(x))     # Mish (inline; identical maths to slot 1)
        x = self.children_list[3](x)        # linear
        x = x * mx.tanh(nn.softplus(x))     # Mish
        return x


class _TemporalHolder(nn.Module):
    """Holds 'temporal' Sequential[Conv1dGLU, Conv1dGLU]."""

    def __init__(self, hidden_dim: int, kernel_size: int):
        super().__init__()
        self.children_list = [
            Conv1dGLU(hidden_dim, hidden_dim, kernel_size),
            Conv1dGLU(hidden_dim, hidden_dim, kernel_size),
        ]

    def __call__(self, x):
        x = self.children_list[0](x)
        x = self.children_list[1](x)
        return x
