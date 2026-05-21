"""Core building blocks for the MLX port of GPT-SoVITS S2 (v2ProTw).

Conventions:
  - Inputs/outputs are in PyTorch layout: (B, C, T).  All Conv1d/ConvTranspose1d
    are wrapped to accept this layout via internal transpose.
  - Mask shapes: (B, 1, T) float.  All ops use straightforward elementwise multiply.
  - Stateless modules; everything seedable via mx.random.seed.
"""

import math
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn


LRELU_SLOPE = 0.1


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


# ---------------------------------------------------------------------------
# Conv wrappers that take/produce (B, C, T)
# ---------------------------------------------------------------------------
class Conv1dPT(nn.Module):
    """PyTorch-style Conv1d on (B, C, T) input."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def __call__(self, x):
        # x: (B, C, T) -> (B, T, C) -> conv -> (B, T_out, C_out) -> (B, C_out, T_out)
        x = x.transpose(0, 2, 1)
        x = self.conv(x)
        return x.transpose(0, 2, 1)


class ConvTranspose1dPT(nn.Module):
    """PyTorch-style ConvTranspose1d on (B, C, T) input."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv_t = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )

    def __call__(self, x):
        x = x.transpose(0, 2, 1)
        x = self.conv_t(x)
        return x.transpose(0, 2, 1)


# ---------------------------------------------------------------------------
# Channel-first LayerNorm (matches PyTorch GPT-SoVITS modules.LayerNorm)
# ---------------------------------------------------------------------------
class ChannelLayerNorm(nn.Module):
    """Normalize over channel dim of (B, C, T) input.  Saved as weight/bias."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((channels,))
        self.bias = mx.zeros((channels,))

    def __call__(self, x):
        # x: (B, C, T) — mean/var over axis 1
        mean = mx.mean(x, axis=1, keepdims=True)
        var = mx.var(x, axis=1, keepdims=True)
        x = (x - mean) * mx.rsqrt(var + self.eps)
        return self.weight[None, :, None] * x + self.bias[None, :, None]


# ---------------------------------------------------------------------------
# WN — WaveNet residual block w/ optional gin conditioning (used by flow)
# ---------------------------------------------------------------------------
class WN(nn.Module):
    """WaveNet stack:  n_layers × (dilated conv + gated activation + 1x1 residual/skip).
    Mirrors module.modules.WN; mean_only used by ResidualCouplingLayer in our ckpt.
    """

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        if gin_channels > 0:
            self.cond_layer = Conv1dPT(gin_channels, 2 * hidden_channels * n_layers, 1)

        self.in_layers: List[Conv1dPT] = []
        self.res_skip_layers: List[Conv1dPT] = []
        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = (kernel_size * dilation - dilation) // 2
            self.in_layers.append(
                Conv1dPT(
                    hidden_channels,
                    2 * hidden_channels,
                    kernel_size,
                    dilation=dilation,
                    padding=padding,
                )
            )
            res_skip_ch = 2 * hidden_channels if i < n_layers - 1 else hidden_channels
            self.res_skip_layers.append(Conv1dPT(hidden_channels, res_skip_ch, 1))

    def __call__(self, x, x_mask, g=None):
        output = mx.zeros_like(x)
        if g is not None and self.gin_channels > 0:
            g = self.cond_layer(g)
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                off = i * 2 * self.hidden_channels
                g_l = g[:, off : off + 2 * self.hidden_channels, :]
                in_act = x_in + g_l
            else:
                in_act = x_in
            t_act = mx.tanh(in_act[:, : self.hidden_channels, :])
            s_act = mx.sigmoid(in_act[:, self.hidden_channels :, :])
            acts = t_act * s_act
            res_skip = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res = res_skip[:, : self.hidden_channels, :]
                skip = res_skip[:, self.hidden_channels :, :]
                x = (x + res) * x_mask
                output = output + skip
            else:
                output = output + res_skip
        return output * x_mask


class ResidualCouplingLayer(nn.Module):
    """Affine coupling layer; mean_only matches ResidualCouplingBlock in S2."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        mean_only: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.half_channels = channels // 2
        self.mean_only = mean_only
        self.pre = Conv1dPT(self.half_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels, kernel_size, dilation_rate, n_layers,
            gin_channels=gin_channels,
        )
        post_ch = self.half_channels if mean_only else 2 * self.half_channels
        self.post = Conv1dPT(hidden_channels, post_ch, 1)

    def __call__(self, x, x_mask, g=None, reverse: bool = False):
        x0 = x[:, : self.half_channels, :]
        x1 = x[:, self.half_channels :, :]
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if self.mean_only:
            m = stats
            logs = mx.zeros_like(m)
        else:
            m = stats[:, : self.half_channels, :]
            logs = stats[:, self.half_channels :, :]
        if reverse:
            x1 = (x1 - m) * mx.exp(-logs) * x_mask
        else:
            x1 = m + x1 * mx.exp(logs) * x_mask
        return mx.concatenate([x0, x1], axis=1)


class Flip(nn.Module):
    """Channel-flip (no params)."""

    def __call__(self, x, x_mask=None, g=None, reverse: bool = False):
        return x[:, ::-1, :]


# ---------------------------------------------------------------------------
# Multi-head attention with relative position bias (encoder self-attn)
# and the plainer cross-attention variant used by MRTE.
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        window_size: Optional[int] = None,
        heads_share: bool = True,
    ):
        super().__init__()
        assert channels % n_heads == 0
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.window_size = window_size
        self.heads_share = heads_share
        self.k_channels = channels // n_heads

        self.conv_q = Conv1dPT(channels, channels, 1)
        self.conv_k = Conv1dPT(channels, channels, 1)
        self.conv_v = Conv1dPT(channels, channels, 1)
        self.conv_o = Conv1dPT(channels, out_channels, 1)

        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            # Parameters are stored uninitialized; weights come from loaded ckpt.
            self.emb_rel_k = mx.zeros(
                (n_heads_rel, window_size * 2 + 1, self.k_channels)
            )
            self.emb_rel_v = mx.zeros(
                (n_heads_rel, window_size * 2 + 1, self.k_channels)
            )

    def __call__(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)
        x = self._attention(q, k, v, mask=attn_mask)
        return self.conv_o(x)

    # ---- attention math -------------------------------------------------
    def _attention(self, query, key, value, mask=None):
        b, d, t_s = key.shape
        t_t = query.shape[2]
        # (B, n_heads, k_channels, T)  -> (B, n_heads, T, k_channels)
        query = query.reshape(b, self.n_heads, self.k_channels, t_t).transpose(0, 1, 3, 2)
        key   = key  .reshape(b, self.n_heads, self.k_channels, t_s).transpose(0, 1, 3, 2)
        value = value.reshape(b, self.n_heads, self.k_channels, t_s).transpose(0, 1, 3, 2)

        scale = 1.0 / math.sqrt(self.k_channels)
        scores = mx.matmul(query * scale, key.transpose(0, 1, 3, 2))   # (B, h, t_t, t_s)

        if self.window_size is not None:
            assert t_s == t_t, "Relative attention requires self-attention."
            key_rel = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(query * scale, key_rel)
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + scores_local

        if mask is not None:
            # mask shape (B, 1, t_t, t_s); 0 = pad.  Use a large negative bias.
            scores = mx.where(mask == 0, mx.array(-1e4, scores.dtype), scores)

        p_attn = mx.softmax(scores, axis=-1)
        output = mx.matmul(p_attn, value)                              # (B, h, t_t, k_ch)

        if self.window_size is not None:
            rel_weights = self._absolute_position_to_relative_position(p_attn)
            value_rel = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(rel_weights, value_rel)

        # Back to (B, C, T): (B, h, t_t, k_ch) -> (B, h, k_ch, t_t) -> (B, h*k_ch=C, t_t)
        output = output.transpose(0, 1, 3, 2).reshape(b, d, t_t)
        return output

    # ---- relative-position helpers (mirror module.attentions) ----------
    def _matmul_with_relative_keys(self, x, y):
        # x: (B, h, l, d_k); y: (1, m, d_k) — heads share
        y_sel = y[0] if y.ndim == 3 else y
        return mx.matmul(x, y_sel.T)

    def _matmul_with_relative_values(self, x, y):
        y_sel = y[0] if y.ndim == 3 else y
        return mx.matmul(x, y_sel)

    def _get_relative_embeddings(self, rel_emb, length):
        max_relative_position = 2 * self.window_size + 1
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start = max((self.window_size + 1) - length, 0)
        slice_end = slice_start + 2 * length - 1
        if pad_length > 0:
            rel_emb = mx.pad(rel_emb, [(0, 0), (pad_length, pad_length), (0, 0)])
        return rel_emb[:, slice_start:slice_end]

    def _relative_position_to_absolute_position(self, x):
        b, heads, length, _ = x.shape
        x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (0, 1)])
        x_flat = x.reshape(b, heads, length * 2 * length)
        x_flat = mx.pad(x_flat, [(0, 0), (0, 0), (0, length - 1)])
        x_final = x_flat.reshape(b, heads, length + 1, 2 * length - 1)
        return x_final[:, :, :length, length - 1 :]

    def _absolute_position_to_relative_position(self, x):
        b, heads, length, _ = x.shape
        x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (0, length - 1)])
        x_flat = x.reshape(b, heads, length * length + length * (length - 1))
        x_flat = mx.pad(x_flat, [(0, 0), (0, 0), (length, 0)])
        x_final = x_flat.reshape(b, heads, length, 2 * length)
        return x_final[:, :, :, 1:]


# ---------------------------------------------------------------------------
# FFN — Conv1d → relu → Conv1d with same-padding
# ---------------------------------------------------------------------------
class FFN(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        pad = (kernel_size - 1) // 2
        # same-padding (pad_l = (k-1)//2, pad_r = k//2) — we use the symmetric
        # version: kernel_size=3 -> pad=1, which matches GPT-SoVITS FFN at k=3.
        self.conv_1 = Conv1dPT(in_channels, filter_channels, kernel_size, padding=pad)
        self.conv_2 = Conv1dPT(filter_channels, out_channels, kernel_size, padding=pad)

    def __call__(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = nn.relu(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask


# ---------------------------------------------------------------------------
# Encoder — transformer encoder used in TextEncoder ({encoder_ssl, encoder_text, encoder2})
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        window_size: int = 4,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.attn_layers: List[MultiHeadAttention] = []
        self.norm_layers_1: List[ChannelLayerNorm] = []
        self.ffn_layers: List[FFN] = []
        self.norm_layers_2: List[ChannelLayerNorm] = []
        for _ in range(n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels, hidden_channels, n_heads, window_size=window_size,
                )
            )
            self.norm_layers_1.append(ChannelLayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(hidden_channels, hidden_channels, filter_channels, kernel_size)
            )
            self.norm_layers_2.append(ChannelLayerNorm(hidden_channels))

    def __call__(self, x, x_mask):
        # attn_mask: (B, 1, T, T) from (B,1,1,T) * (B,1,T,1)
        attn_mask = mx.expand_dims(x_mask, 2) * mx.expand_dims(x_mask, -1)
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.attn_layers[i](x, x, attn_mask)
            x = self.norm_layers_1[i](x + y)
            y = self.ffn_layers[i](x, x_mask)
            x = self.norm_layers_2[i](x + y)
        return x * x_mask


# ---------------------------------------------------------------------------
# Helper: sequence_mask
# ---------------------------------------------------------------------------
def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = int(length.max().item())
    x = mx.arange(max_length, dtype=mx.int32)
    return mx.expand_dims((x[None, :] < length[:, None]).astype(mx.float32), 1)
