"""MLX port of GPT-SoVITS S2 (v2ProTw).

Top-level class: SynthesizerTrn — supports both inference (.decode) and
training (.forward).  Training path mirrors GPT_SoVITS/module/models.py:938.

Use:
    from models import SynthesizerTrn, default_config
    model = SynthesizerTrn(**default_config())
    model.load_weights("/path/to/model.safetensors", strict=True)

    # inference:
    audio = model.decode(codes, text, refer, sv_emb=sv_emb_in)

    # training:
    (y_hat, kl_ssl, ids_slice, x_mask, z_mask,
     (z, z_p, m_p, logs_p, m_q, logs_q),
     quantized) = model(ssl, spec, spec_lens, text, text_lens, sv_emb_in,
                        segment_size_frames=32)
"""

from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from modules import (
    Conv1dPT,
    Encoder,
    Flip,
    ResidualCouplingLayer,
    sequence_mask,
)
from mrte import MRTE
from ref_enc import MelStyleEncoder
from hifigan import Generator
from posterior_encoder import PosteriorEncoder
import commons as _commons


# ---------------------------------------------------------------------------
# Defaults — match configs/s2_taiwan.json + v2Pro promotion
# ---------------------------------------------------------------------------
def default_config() -> dict:
    return dict(
        spec_channels=1025,             # filter_length//2+1
        inter_channels=192,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[10, 8, 2, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[16, 16, 8, 2, 2],
        gin_channels=1024,
        semantic_frame_rate="25hz",
        version="v2ProTw",
        ssl_dim=768,
        n_vocab=1033,
        codebook_size=1024,
    )


# ---------------------------------------------------------------------------
# TextEncoder — enc_p (the meaty 235-key block)
# ---------------------------------------------------------------------------
class TextEncoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        n_vocab: int,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.ssl_proj = Conv1dPT(768, hidden_channels, 1)

        # n_layers//2 transformer for SSL, full n_layers for text, n_layers//2 for fusion (encoder2)
        self.encoder_ssl = Encoder(
            hidden_channels, filter_channels, n_heads, n_layers // 2, kernel_size,
        )
        self.text_embedding = nn.Embedding(n_vocab, hidden_channels)
        self.encoder_text = Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size,
        )
        self.mrte = MRTE(
            content_enc_channels=hidden_channels, hidden_size=512, out_channels=hidden_channels,
        )
        self.encoder2 = Encoder(
            hidden_channels, filter_channels, n_heads, n_layers // 2, kernel_size,
        )
        self.proj = Conv1dPT(hidden_channels, out_channels * 2, 1)

    def __call__(self, y, y_lengths, text, text_lengths, ge_512):
        # y       : (B, 768, T_y) — quantized SSL feats (already 2× upsampled)
        # y_lengths : (B,) int
        # text    : (B, T_t) — token ids
        # text_lengths : (B,)
        # ge_512  : (B, 512, 1) — v2Pro projected speaker embedding (input to MRTE)
        y_mask = sequence_mask(y_lengths, y.shape[2])              # (B,1,T_y)
        text_mask = sequence_mask(text_lengths, text.shape[1])     # (B,1,T_t)

        y = self.ssl_proj(y * y_mask) * y_mask                     # (B, 192, T_y)
        y = self.encoder_ssl(y * y_mask, y_mask)                   # (B, 192, T_y)

        t_emb = self.text_embedding(text)                          # (B, T_t, 192)
        t_emb = t_emb.transpose(0, 2, 1)                           # (B, 192, T_t)
        t_emb = self.encoder_text(t_emb * text_mask, text_mask)    # (B, 192, T_t)

        y = self.mrte(y, y_mask, t_emb, text_mask, ge_512)         # (B, 192, T_y)
        y = self.encoder2(y * y_mask, y_mask)                      # (B, 192, T_y)

        stats = self.proj(y) * y_mask                              # (B, 384, T_y)
        m = stats[:, : self.out_channels, :]
        logs = stats[:, self.out_channels :, :]
        return y, m, logs, y_mask


# ---------------------------------------------------------------------------
# ResidualCouplingBlock — flow (4 coupling layers interleaved with flip)
# ---------------------------------------------------------------------------
class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels: int = 192,
        hidden_channels: int = 192,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        n_layers: int = 4,
        n_flows: int = 4,
        gin_channels: int = 1024,
    ):
        super().__init__()
        # In the ckpt, indices 0,2,4,6 are coupling layers; 1,3,5,7 are Flip (no params).
        # We store the same indexing so load_weights sees `flows.0.pre.weight` etc.
        self.flows = []
        for _ in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels, hidden_channels, kernel_size, dilation_rate, n_layers,
                    gin_channels=gin_channels, mean_only=True,
                )
            )
            self.flows.append(Flip())

    def __call__(self, x, x_mask, g=None, reverse: bool = False):
        if reverse:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=True)
        else:
            for flow in self.flows:
                x = flow(x, x_mask, g=g, reverse=False)
        return x


# ---------------------------------------------------------------------------
# Quantizer decoder — single codebook lookup, identity projection
# ---------------------------------------------------------------------------
class QuantizerDecoder(nn.Module):
    """Holds `vq.layers.0._codebook.embed` and supports both inference decode
    and training-time forward (with commit_loss, frozen codebook).

    Replicates ResidualVectorQuantization for n_q=1, dim==codebook_dim==768.

    Training note: we treat the codebook as **frozen** for r6 (no EMA updates).
    The r5 e16 codebook is already learned; the user's 5-epoch r6 fine-tune
    has decay=0.99 EMA which would barely move it anyway. Skipping EMA is
    equivalent to the upstream `freeze_quantizer=True` mode (which freezes
    only the codebook; ssl_proj and downstream still train freely via the
    straight-through estimator + commit_loss).
    """

    def __init__(self, n_codes: int = 1024, dim: int = 768):
        super().__init__()
        self.vq = _VQ(n_codes, dim)

    def decode(self, codes):
        # codes: (n_q, B, T_code) — here n_q=1.  We take layer 0.
        idx = codes[0]                                # (B, T_code)
        embed = self.vq.layers[0].codebook.embed      # (n_codes, dim)
        # F.embedding behavior: gather rows
        quantized = embed[idx]                        # (B, T_code, dim)
        quantized = quantized.transpose(0, 2, 1)      # (B, dim, T_code)
        return quantized

    def __call__(self, x: mx.array, layers=None):
        """Training-time RVQ forward, n_q=1.

        x        : (B, dim, T) — input to be quantized (from ssl_proj output)
        layers   : ignored (only layer 0 supported; matches upstream contract)

        Returns: (quantized, codes, commit_loss, [quantized])
          - quantized: (B, dim, T) with straight-through estimator
          - codes:     (B, T) int — codebook indices for each frame
          - commit_loss: scalar mx.array — F.mse_loss(stop_gradient(quantized), x)
          - quantized_list: [quantized] for compatibility with upstream signature

        Codebook is frozen (treated as stop_gradient); only the residual path
        carries gradient back to the input x via the straight-through trick.
        """
        # x: (B, D, T) → (B, T, D)
        B, D, T = x.shape
        x_bt = x.transpose(0, 2, 1)                       # (B, T, D)
        x_flat = x_bt.reshape(B * T, D)                   # (B*T, D)

        # Codebook (frozen for r6): stop_gradient prevents any backprop into it.
        embed = mx.stop_gradient(self.vq.layers[0].codebook.embed)   # (n_codes, D)

        # Squared distance: ||x||^2 - 2 x . e + ||e||^2 — we want argmin so we can
        # drop ||x||^2 (constant across codes). Compute (-2*x.eT + ||e||^2).
        # That's argmin_e ||x-e||^2 == argmax_e (2 x.e - ||e||^2).
        emb_norm2 = mx.sum(embed * embed, axis=-1)                    # (n_codes,)
        scores = 2.0 * (x_flat @ embed.T) - mx.expand_dims(emb_norm2, 0)  # (B*T, n_codes)
        codes_flat = mx.argmax(scores, axis=-1).astype(mx.int32)      # (B*T,)
        codes = codes_flat.reshape(B, T)                              # (B, T)

        # Dequantize
        quantize_flat = embed[codes_flat]                              # (B*T, D)
        quantize = quantize_flat.reshape(B, T, D).transpose(0, 2, 1)   # (B, D, T)

        # Straight-through estimator: quantize = x + (quantize - x).detach
        # so the gradient that flows into `quantize` flows directly into `x`.
        quantize_st = x + mx.stop_gradient(quantize - x)

        # Commit loss: F.mse_loss(quantize.detach(), x) * commitment_weight
        # upstream commitment_weight=1.0 (default in VectorQuantization).
        commit_loss = mx.mean((mx.stop_gradient(quantize) - x) ** 2)

        return quantize_st, codes, commit_loss, [quantize_st]


class _VQ(nn.Module):
    def __init__(self, n_codes, dim):
        super().__init__()
        self.layers = [_VQLayer(n_codes, dim)]


class _VQLayer(nn.Module):
    def __init__(self, n_codes, dim):
        super().__init__()
        # NOTE: PyTorch used `_codebook` (private), but MLX tree_flatten skips
        # attribute names starting with '_'.  We rename to `codebook` and the
        # conversion utility renames the corresponding safetensors key.
        self.codebook = _CB(n_codes, dim)


class _CB(nn.Module):
    def __init__(self, n_codes, dim):
        super().__init__()
        self.embed = mx.zeros((n_codes, dim))


# ---------------------------------------------------------------------------
# Top-level SynthesizerTrn (inference path only)
# ---------------------------------------------------------------------------
class SynthesizerTrn(nn.Module):
    def __init__(
        self,
        spec_channels: int = 1025,
        inter_channels: int = 192,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        resblock: str = "1",
        resblock_kernel_sizes: List[int] = None,
        resblock_dilation_sizes: List[List[int]] = None,
        upsample_rates: List[int] = None,
        upsample_initial_channel: int = 512,
        upsample_kernel_sizes: List[int] = None,
        gin_channels: int = 1024,
        semantic_frame_rate: str = "25hz",
        version: str = "v2ProTw",
        ssl_dim: int = 768,
        n_vocab: int = 1033,
        codebook_size: int = 1024,
        **_kwargs,
    ):
        super().__init__()
        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = [3, 7, 11]
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        if upsample_rates is None:
            upsample_rates = [10, 8, 2, 2, 2]
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = [16, 16, 8, 2, 2]

        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.gin_channels = gin_channels
        self.semantic_frame_rate = semantic_frame_rate
        self.version = version

        self.enc_p = TextEncoder(
            inter_channels, hidden_channels, filter_channels,
            n_heads, n_layers, kernel_size, n_vocab,
        )
        self.dec = Generator(
            inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes,
            upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            channels=inter_channels, hidden_channels=hidden_channels,
            kernel_size=5, dilation_rate=1, n_layers=4, n_flows=4,
            gin_channels=gin_channels,
        )

        # MelStyleEncoder — for v2ProTw, n_mel_channels=704
        self.ref_enc = MelStyleEncoder(
            n_mel_channels=704, style_hidden=128, style_vector_dim=gin_channels,
            style_kernel_size=5, style_head=2,
        )

        # top-level ssl_proj (Conv1d 768→768 k=2 s=2, halves 50Hz → 25Hz at training)
        self.ssl_proj = Conv1dPT(ssl_dim, ssl_dim, 2, stride=2)

        # quantizer (supports both inference decode AND training forward with commit_loss)
        self.quantizer = QuantizerDecoder(n_codes=codebook_size, dim=ssl_dim)

        # Posterior encoder (training only) — encodes linear spectrogram → z, m_q, logs_q
        self.enc_q = PosteriorEncoder(
            in_channels=spec_channels,
            out_channels=inter_channels,
            hidden_channels=hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=16,
            gin_channels=gin_channels,
        )

        # v2Pro additions
        self.sv_emb = nn.Linear(20480, gin_channels)
        self.ge_to512 = nn.Linear(gin_channels, 512)
        self.prelu = nn.PReLU(num_parameters=gin_channels)

    # ---- helpers ----
    def _compute_ge(self, refer, sv_emb_in):
        """refer: (B, spec_channels, T_ref); sv_emb_in: (B, 20480).  Returns (B, 1024, 1)."""
        ge = self.ref_enc(refer[:, :704, :], mask=None)        # (B, 1024, 1)
        sv = self.sv_emb(sv_emb_in)                            # (B, 1024)
        ge = ge + mx.expand_dims(sv, -1)                       # (B, 1024, 1)
        # MLX nn.PReLU expects channel-last layout (broadcasts last dim against weight).
        # Our ge is (B, C, T) PT-style → transpose to (B, T, C) for the activation.
        ge_tc = ge.transpose(0, 2, 1)                          # (B, 1, 1024)
        ge_tc = self.prelu(ge_tc)
        ge = ge_tc.transpose(0, 2, 1)                          # (B, 1024, 1)
        return ge

    def _nearest_upsample_2x(self, x):
        """Mirror F.interpolate(scale_factor=2, mode='nearest') along the last (T) axis."""
        # x: (B, C, T) → (B, C, T, 1) → repeat dim=-1 → (B, C, T, 2) → (B, C, 2T)
        b, c, t = x.shape
        x = mx.expand_dims(x, -1)                              # (B, C, T, 1)
        x = mx.broadcast_to(x, (b, c, t, 2))
        return x.reshape(b, c, t * 2)

    def decode(self, codes, text, refer, sv_emb_in, noise_scale: float = 0.5,
               noise: Optional[mx.array] = None):
        """
        codes  : (n_q, B, T_code) int — typically n_q=1
        text   : (B, T_t) int
        refer  : (B, 1025, T_ref) float — spec
        sv_emb_in : (B, 20480) float — pre-computed SV embedding
        noise_scale : float — applied to z_p sampling
        noise  : (B, 192, T_y) optional injected noise (for numerical comparison).
                 If None, noise is sampled internally.

        Returns: (B, 1, T_wav) float
        """
        ge = self._compute_ge(refer, sv_emb_in)                 # (B, 1024, 1)
        ge_512 = self.ge_to512(ge.transpose(0, 2, 1)).transpose(0, 2, 1)  # (B, 512, 1)

        # quantizer.decode → (B, 768, T_code)
        quantized = self.quantizer.decode(codes)
        # 25hz: nearest-up 2x to match y frame rate
        if self.semantic_frame_rate == "25hz":
            quantized = self._nearest_upsample_2x(quantized)

        b = quantized.shape[0]
        T_y = quantized.shape[2]
        y_lengths = mx.array([T_y] * b, dtype=mx.int32)
        T_t = text.shape[1]
        text_lengths = mx.array([T_t] * b, dtype=mx.int32)

        # enc_p → m, logs of shape (B, 192, T_y)
        _, m_p, logs_p, y_mask = self.enc_p(
            quantized, y_lengths, text, text_lengths, ge_512,
        )

        # sample z_p (optionally with injected noise for reproducibility)
        if noise is None:
            noise = mx.random.normal(m_p.shape)
        z_p = m_p + noise * mx.exp(logs_p) * noise_scale

        # reverse flow conditioned on ge (1024 dim)
        z = self.flow(z_p, y_mask, g=ge, reverse=True)

        # decoder → waveform
        audio = self.dec(z * y_mask, g=ge)
        return audio

    # ---- training forward (mirrors upstream SynthesizerTrn.forward) -------
    def __call__(
        self,
        ssl: mx.array,            # (B, 768, T_ssl)   raw SSL features (50 Hz)
        y: mx.array,              # (B, spec_ch, T_y) linear spectrogram (25 Hz)
        y_lengths: mx.array,      # (B,)              true T_y per row
        text: mx.array,           # (B, T_t)          token ids
        text_lengths: mx.array,   # (B,)              true T_t per row
        sv_emb_in: mx.array,      # (B, 20480)        speaker-verification embedding
        segment_size_frames: int = 32,   # latent-frame slice length (audio = 32*hop = 20480)
        slice_key: Optional[mx.array] = None,  # mx.random.key for slice idx
    ):
        """Training forward pass.

        Returns:
            y_hat         : (B, 1, segment_size_audio)   generated waveform on the slice
            kl_ssl        : scalar mx.array              RVQ commit_loss (kl_ssl term in losses)
            ids_slice     : (B,) int                     per-batch slice start (in z frames)
            x_mask        : (B, 1, T_y)                  alias for y_mask (upstream returns same twice)
            z_mask        : (B, 1, T_y)                  same mask (post-flow)
            (z, z_p, m_p, logs_p, m_q, logs_q)            posterior + flow + prior stats for KL
            quantized     : (B, 768, T_y)                quantized SSL features after ssl_proj
        """
        # 1) Speaker embedding from spec (use first 704 channels per v2Pro convention)
        # Compute y_mask first so we mask before ref_enc to match upstream.
        y_mask = _commons.sequence_mask(y_lengths, y.shape[2]).astype(y.dtype)   # (B,1,T_y)
        ge_raw = self.ref_enc(y[:, :704, :] * y_mask, mask=None)                 # (B, 1024, 1)
        sv = self.sv_emb(sv_emb_in)                                              # (B, 1024)
        ge = ge_raw + mx.expand_dims(sv, -1)
        ge_tc = ge.transpose(0, 2, 1)                                            # (B, 1, 1024)
        ge_tc = self.prelu(ge_tc)
        ge = ge_tc.transpose(0, 2, 1)                                            # (B, 1024, 1)
        ge_512 = self.ge_to512(ge.transpose(0, 2, 1)).transpose(0, 2, 1)          # (B, 512, 1)

        # 2) ssl → ssl_proj (50→25 Hz, k=2 s=2) → quantizer (with commit_loss)
        ssl_proj_out = self.ssl_proj(ssl)                                         # (B, 768, T_ssl/2)
        quantized, codes, commit_loss, _ = self.quantizer(ssl_proj_out, layers=[0])

        # 3) 25Hz → upsample to match y frame rate (semantic_frame_rate == "25hz")
        if self.semantic_frame_rate == "25hz":
            quantized = self._nearest_upsample_2x(quantized)                       # (B, 768, T_y_pred)

        # Adjust the time dim if the upsample produced extra frames vs y_lengths.
        # Upstream uses F.interpolate with size=2*T which is exact; we get the
        # same via repeat. Truncate to T_y for safety.
        if quantized.shape[2] != y.shape[2]:
            T_target = y.shape[2]
            if quantized.shape[2] > T_target:
                quantized = quantized[:, :, :T_target]
            else:
                # pad with last-frame repeat
                pad_len = T_target - quantized.shape[2]
                last = quantized[:, :, -1:]
                pad = mx.broadcast_to(last, (quantized.shape[0], quantized.shape[1], pad_len))
                quantized = mx.concatenate([quantized, pad], axis=2)

        # 4) TextEncoder → m_p, logs_p (the prior)
        _, m_p, logs_p, x_mask = self.enc_p(
            quantized, y_lengths, text, text_lengths, ge_512,
        )

        # 5) PosteriorEncoder → z, m_q, logs_q  (z is the latent we'll slice)
        z, m_q, logs_q, z_mask = self.enc_q(y, y_lengths, g=ge)

        # 6) Flow forward → z_p (used in KL closed form vs m_p, logs_p)
        z_p = self.flow(z, z_mask, g=ge, reverse=False)

        # 7) Random slice of z + decode to audio for the generator+disc step
        z_slice, ids_slice = _commons.rand_slice_segments(
            z, y_lengths, segment_size_frames, key=slice_key,
        )
        y_hat = self.dec(z_slice, g=ge)

        return (
            y_hat,
            commit_loss,                                  # kl_ssl
            ids_slice,
            x_mask,
            z_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            quantized,
        )
