"""MLX port of GPT-SoVITS S1 (Text-to-Semantic AR transformer).

Mirrors the PyTorch implementation in
  GPT-SoVITS/GPT_SoVITS/AR/models/t2s_model.py
  GPT-SoVITS/GPT_SoVITS/AR/modules/embedding.py
  GPT-SoVITS/GPT_SoVITS/AR/modules/transformer.py

Architecture (config arm_A_e15_trilingual):
  - 24-layer post-LN transformer
  - hidden_dim = embedding_dim = 512
  - num_heads = 16
  - linear_units (FFN) = 2048
  - phoneme_vocab_size = 1033
  - vocab_size (audio codes) = 1025  (1024 + EOS)
  - EOS = 1024

Use:
    from s1_mlx.model import T2SModel
    from safetensors.numpy import load_file
    import mlx.core as mx

    cfg = {"hidden_dim": 512, "embedding_dim": 512, "head": 16,
           "n_layer": 24, "vocab_size": 1025, "phoneme_vocab_size": 1033,
           "EOS": 1024}
    m = T2SModel(cfg)
    sd = load_file("s1.safetensors")
    m.load_weights([(k, mx.array(v)) for k, v in sd.items()], strict=True)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
class TokenEmbedding(nn.Module):
    """Mirror of AR.modules.embedding.TokenEmbedding.

    PT keys:  word_embeddings.weight  → MLX key: word_embeddings.weight
    """

    def __init__(self, embedding_dim: int, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

    @property
    def weight(self) -> mx.array:
        return self.word_embeddings.weight

    def __call__(self, x: mx.array) -> mx.array:
        return self.word_embeddings(x)


class SinePositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding with a learnable scalar `alpha`.

    PT keys: alpha (shape (1,))  → MLX key: alpha (shape (1,))

    Forward semantics (from PT):
        pe is a fixed sinusoidal table of length >= seq_len
        output = x * x_scale + alpha * pe[:, :seq_len]
        x_scale = sqrt(d) if scale else 1.0
        alpha is a Parameter (scalar)
    """

    # `pe` is a frozen buffer; declare it as non-Parameter via the MLX
    # state-dict opt-out (any attribute starting with "_" is treated as
    # internal by mx.utils.tree_flatten).
    def __init__(
        self,
        embedding_dim: int,
        scale: bool = False,
        initial_size: int = 4000,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.x_scale = math.sqrt(embedding_dim) if scale else 1.0
        # learnable scalar — name matches PT key
        self.alpha = mx.ones((1,))
        # frozen position table; underscored so MLX skips it during weight
        # loading & state-dict flattening.
        self._pe = self._build_pe(initial_size)

    def _build_pe(self, max_len: int) -> mx.array:
        position = mx.arange(0, max_len, dtype=mx.float32).reshape(-1, 1)
        div_term = mx.exp(
            mx.arange(0, self.embedding_dim, 2, dtype=mx.float32)
            * -(math.log(10000.0) / self.embedding_dim)
        )
        # pe: (max_len, d)
        pe = mx.zeros((max_len, self.embedding_dim), dtype=mx.float32)
        # MLX has no in-place slice assign on a freshly created array, so
        # build sin / cos and interleave.
        sin_part = mx.sin(position * div_term)            # (L, d/2)
        cos_part = mx.cos(position * div_term)            # (L, d/2)
        # Interleave: pe[:, 0::2] = sin, pe[:, 1::2] = cos
        stacked = mx.stack([sin_part, cos_part], axis=-1)  # (L, d/2, 2)
        pe = stacked.reshape(max_len, self.embedding_dim)
        return pe[None, :, :]                             # (1, L, d)

    def extend(self, seq_len: int, dtype) -> None:
        """Grow the cached pe to at least `seq_len`. Called lazily."""
        if self._pe.shape[1] >= seq_len:
            if self._pe.dtype != dtype:
                self._pe = self._pe.astype(dtype)
            return
        new_size = max(seq_len, self._pe.shape[1] * 2)
        self._pe = self._build_pe(new_size).astype(dtype)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, d)
        self.extend(x.shape[1], x.dtype)
        return x * self.x_scale + self.alpha * self._pe[:, : x.shape[1], :]

    def lookup(self, pos: int, dtype) -> mx.array:
        """Return self.alpha * pe[:, pos:pos+1, :] cast to `dtype`."""
        self.extend(pos + 1, dtype)
        return self.alpha * self._pe[:, pos : pos + 1, :]


# ---------------------------------------------------------------------------
# T2SBlock (single transformer encoder layer, post-LN, fused QKV)
# ---------------------------------------------------------------------------
class T2SBlock(nn.Module):
    """One transformer block — post-LN, fused QKV.

    PT keys (per layer i):
        self_attn.in_proj_weight   (3*H, H)
        self_attn.in_proj_bias     (3*H,)
        self_attn.out_proj.weight  (H, H)
        self_attn.out_proj.bias    (H,)
        linear1.weight             (FF, H)
        linear1.bias               (FF,)
        linear2.weight             (H, FF)
        linear2.bias               (H,)
        norm1.weight, norm1.bias   (H,)
        norm2.weight, norm2.bias   (H,)
    Layout matches MLX nn.Linear / nn.LayerNorm directly.
    """

    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, eps: float = 1e-5):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Fused QKV — single (3*H, H) linear
        # PT names: in_proj_weight / in_proj_bias  (NOT a Linear object)
        # We expose them as plain mx.array attributes so the safetensors
        # key 'self_attn.in_proj_weight' maps 1:1.
        self.self_attn = _MHA(hidden_dim, num_heads)
        self.linear1 = nn.Linear(hidden_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps)

    def _mlp(self, x: mx.array) -> mx.array:
        return self.linear2(nn.relu(self.linear1(x)))

    def process_prompt(
        self,
        x: mx.array,
        attn_bias: Optional[mx.array],
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Full (prefill) attention. Returns (out, k_cache, v_cache).

        attn_bias: (T, T) float — additive (0 for keep, -inf for mask), or None.
        """
        q, k, v = self.self_attn.qkv(x)
        # Save K/V caches in unhead layout: (B, T, H_dim) — easier to extend.
        k_cache = k
        v_cache = v
        attn = self.self_attn.attend(q, k, v, attn_bias)
        attn = self.self_attn.o(attn)
        x = self.norm1(x + attn)
        x = self.norm2(x + self._mlp(x))
        return x, k_cache, v_cache

    def decode_next_token(
        self,
        x: mx.array,
        k_cache: mx.array,
        v_cache: mx.array,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Step-wise decode. x is a single token (B, 1, H).
        Appends new K/V to caches and returns updated caches.
        """
        q, k, v = self.self_attn.qkv(x)
        k_cache = mx.concatenate([k_cache, k], axis=1)
        v_cache = mx.concatenate([v_cache, v], axis=1)
        attn = self.self_attn.attend(q, k_cache, v_cache, None)
        attn = self.self_attn.o(attn)
        x = self.norm1(x + attn)
        x = self.norm2(x + self._mlp(x))
        return x, k_cache, v_cache


class _MHA(nn.Module):
    """Holder for self_attn.{in_proj_weight,in_proj_bias,out_proj.weight,out_proj.bias}.

    We use explicit attributes for the fused-QKV projection because PyTorch's
    nn.MultiheadAttention stores qkv as a single `in_proj_weight` (not a Linear
    sublayer), and we want the safetensors key names to round-trip 1:1.
    """

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        # raw weight/bias arrays — names match PT keys
        self.in_proj_weight = mx.zeros((3 * hidden_dim, hidden_dim))
        self.in_proj_bias = mx.zeros((3 * hidden_dim,))
        # out_proj is a proper Linear so out_proj.weight / out_proj.bias work
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def qkv(self, x: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """x: (B, T, H) → q,k,v each (B, T, H)."""
        # MLX matmul: (B,T,H) @ (H, 3H) → (B, T, 3H)
        proj = x @ self.in_proj_weight.T + self.in_proj_bias
        return mx.split(proj, 3, axis=-1)

    def attend(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        attn_bias: Optional[mx.array],
    ) -> mx.array:
        """Scaled dot-product attention. All inputs (B, T, H) → output (B, T_q, H)."""
        B, Tq, _ = q.shape
        Tk = k.shape[1]
        # Reshape into heads: (B, T, H) → (B, T, n_h, head_d) → (B, n_h, T, head_d)
        q = q.reshape(B, Tq, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, Tk, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, Tk, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # mx.fast.scaled_dot_product_attention takes (B, n_h, T, d) and a mask
        # that is added to the logits (0 keep, -inf mask).
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=attn_bias)
        # back to (B, T, H)
        out = out.transpose(0, 2, 1, 3).reshape(B, Tq, self.hidden_dim)
        return out

    def o(self, x: mx.array) -> mx.array:
        return self.out_proj(x)


# ---------------------------------------------------------------------------
# Full T2S decoder
# ---------------------------------------------------------------------------
class T2SModel(nn.Module):
    """Top-level model. Key prefix matches the PT ckpt: top-level keys are
    `bert_proj.*`, `ar_text_embedding.*`, `ar_text_position.alpha`,
    `ar_audio_embedding.*`, `ar_audio_position.alpha`,
    `h.layers.{i}.{...}`, and `ar_predict_layer.weight`.

    The PT ckpt has all of these prefixed with `model.` — strip that prefix
    during conversion (see convert.py).
    """

    def __init__(self, config: dict):
        super().__init__()
        m = config.get("model", config)
        self.hidden_dim = int(m["hidden_dim"])
        self.embedding_dim = int(m["embedding_dim"])
        self.num_heads = int(m.get("head", m.get("num_head", 16)))
        self.num_layers = int(m.get("n_layer", m.get("num_layers", 24)))
        self.vocab_size = int(m["vocab_size"])
        self.phoneme_vocab_size = int(m["phoneme_vocab_size"])
        self.EOS = int(m.get("EOS", self.vocab_size - 1))
        self.ff_dim = int(m.get("linear_units", self.hidden_dim * 4))

        self.bert_proj = nn.Linear(1024, self.embedding_dim)
        self.ar_text_embedding = TokenEmbedding(self.embedding_dim, self.phoneme_vocab_size)
        self.ar_text_position = SinePositionalEmbedding(self.embedding_dim, scale=False)
        self.ar_audio_embedding = TokenEmbedding(self.embedding_dim, self.vocab_size)
        self.ar_audio_position = SinePositionalEmbedding(self.embedding_dim, scale=False)

        # the 24 transformer layers — name `h.layers.<i>` matches PT
        self.h = _LayerStack(
            [T2SBlock(self.hidden_dim, self.num_heads, self.ff_dim) for _ in range(self.num_layers)]
        )

        # output head — no bias
        self.ar_predict_layer = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)

    # ---- Convenience: build the text prefix (text emb + bert_proj + pos) -----
    def make_text_prefix(self, phoneme_ids: mx.array, bert: mx.array) -> mx.array:
        """phoneme_ids: (B, T_t) int.  bert: (B, 1024, T_t) float.
        Returns: x of shape (B, T_t, d).
        """
        x = self.ar_text_embedding(phoneme_ids)              # (B, T_t, d)
        x = x + self.bert_proj(bert.transpose(0, 2, 1))      # bert.T → (B, T_t, 1024) → (B, T_t, d)
        x = self.ar_text_position(x)
        return x

    def embed_audio_with_pos(self, audio_ids: mx.array) -> mx.array:
        y = self.ar_audio_embedding(audio_ids)
        y = self.ar_audio_position(y)
        return y

    # ---- Prefill ----
    def prefill(
        self,
        text_emb: mx.array,
        prompt_audio_ids: Optional[mx.array],
    ) -> Tuple[mx.array, List[mx.array], List[mx.array], int]:
        """Run a full forward over [text, prompt_audio]. Returns:
          (last_token_hidden,  k_cache_list,  v_cache_list,  context_len)

        text_emb:        (B, T_t, d) — already positional / bert-mixed
        prompt_audio_ids:(B, T_p) int — optional. None → ref-free (audio side empty).
        """
        B, T_t, d = text_emb.shape
        if prompt_audio_ids is not None and prompt_audio_ids.shape[1] > 0:
            y_emb = self.embed_audio_with_pos(prompt_audio_ids)        # (B, T_p, d)
            T_p = y_emb.shape[1]
            xy = mx.concatenate([text_emb, y_emb], axis=1)             # (B, T_t+T_p, d)
        else:
            T_p = 0
            xy = text_emb

        total = T_t + T_p
        # Build attention bias:
        #   - positions in [0, T_t) attend to all of [0, T_t) (full) — block y
        #   - positions in [T_t, T_t+T_p) attend causally over the whole context
        # In PT: x_mask = pad((T_t,T_t), (0, T_p), True)  — i.e. text rows mask out audio
        #        y_mask = pad(triu(T_p, T_p), (T_t, 0), False) — y rows causal
        # We replicate by building a (total, total) bool mask, then convert.
        text_rows = mx.concatenate(
            [mx.zeros((T_t, T_t), dtype=mx.bool_), mx.ones((T_t, T_p), dtype=mx.bool_)],
            axis=1,
        )
        if T_p > 0:
            # triu(ones(T_p,T_p), diagonal=1)  →  True above diagonal
            ar = mx.arange(T_p)
            triu = ar[None, :] > ar[:, None]            # (T_p, T_p) bool
            y_rows = mx.concatenate(
                [mx.zeros((T_p, T_t), dtype=mx.bool_), triu],
                axis=1,
            )
            mask = mx.concatenate([text_rows, y_rows], axis=0)         # (total, total)
        else:
            mask = text_rows[:T_t, :T_t]
        # Convert bool → additive bias (-inf where masked)
        attn_bias = mx.where(mask, mx.array(-1e9, dtype=xy.dtype), mx.array(0.0, dtype=xy.dtype))

        k_caches: List[mx.array] = []
        v_caches: List[mx.array] = []
        h = xy
        for blk in self.h.layers:
            h, kc, vc = blk.process_prompt(h, attn_bias)
            k_caches.append(kc)
            v_caches.append(vc)
        return h, k_caches, v_caches, total

    # ---- Single-step decode ----
    def decode_step(
        self,
        last_token: mx.array,
        pos_idx: int,
        k_caches: List[mx.array],
        v_caches: List[mx.array],
    ) -> Tuple[mx.array, List[mx.array], List[mx.array]]:
        """One AR step. Returns (hidden_last, updated_kcs, updated_vcs).

        last_token: (B, 1) int — newly sampled token
        pos_idx:    int — absolute position in the audio sequence (== current y_len)
        """
        # Embed and add positional contribution at pos_idx.
        y_emb = self.ar_audio_embedding(last_token)                 # (B,1,d)
        # scale=False → x_scale = 1.0
        pos = self.ar_audio_position.lookup(pos_idx, y_emb.dtype)   # (1,1,d)
        h = y_emb + pos                                              # (B,1,d)
        new_kcs: List[mx.array] = []
        new_vcs: List[mx.array] = []
        for i, blk in enumerate(self.h.layers):
            h, kc, vc = blk.decode_next_token(h, k_caches[i], v_caches[i])
            new_kcs.append(kc)
            new_vcs.append(vc)
        return h, new_kcs, new_vcs

    def logits(self, hidden: mx.array) -> mx.array:
        """hidden: (B, T, d) — apply ar_predict_layer to the last position.
        Returns (B, vocab_size)."""
        return self.ar_predict_layer(hidden[:, -1, :])


class _LayerStack(nn.Module):
    """Holds a list of submodules under .layers — matches PT key prefix `h.layers.<i>`."""
    def __init__(self, layers: List[nn.Module]):
        super().__init__()
        self.layers = layers
