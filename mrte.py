"""MRTE — Multi-Reference Timbre Encoder (cross-attention to inject text + ge into ssl features).

Direct port of module.mrte_model.MRTE.
"""

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from modules import Conv1dPT, MultiHeadAttention


class MRTE(nn.Module):
    def __init__(
        self,
        content_enc_channels: int = 192,
        hidden_size: int = 512,
        out_channels: int = 192,
        n_heads: int = 4,
    ):
        super().__init__()
        # Cross-attention — no relative-position window
        self.cross_attention = MultiHeadAttention(
            hidden_size, hidden_size, n_heads, window_size=None
        )
        self.c_pre = Conv1dPT(content_enc_channels, hidden_size, 1)
        self.text_pre = Conv1dPT(content_enc_channels, hidden_size, 1)
        self.c_post = Conv1dPT(hidden_size, out_channels, 1)

    def __call__(self, ssl_enc, ssl_mask, text, text_mask, ge):
        # ssl_enc: (B, 192, T_y); text: (B, 192, T_t); ge: (B, 512, 1)
        # attn_mask: (B, 1, T_y, T_t) — query=ssl positions, key=text positions
        attn_mask = mx.expand_dims(ssl_mask, -1) * mx.expand_dims(text_mask, 2)

        ssl_h = self.c_pre(ssl_enc * ssl_mask)       # (B, 512, T_y)
        text_h = self.text_pre(text * text_mask)     # (B, 512, T_t)

        # MultiHeadAttention.__call__(x=query_src, c=key/value_src, mask)
        x = self.cross_attention(ssl_h * ssl_mask, text_h * text_mask, attn_mask)
        x = x + ssl_h
        if ge is not None:
            x = x + ge                                # broadcast (B,512,1) over T_y
        return self.c_post(x * ssl_mask)
