"""MLX port of module.mel_processing.

We implement STFT manually since MLX has no torch.stft equivalent (it ships
rfft only).  Framing + Hann window + rfft is straightforward and matches
PyTorch's `torch.stft(center=False, normalized=False, onesided=True,
return_complex=True)` semantics.

The mel filterbank is loaded once from librosa (it's a constant matrix).

API matches the PyTorch side:
    spectrogram_torch(y, n_fft, hop, win, sr=...)        ->  (B, n_fft//2+1, T)
    spec_to_mel_torch(spec, n_fft, num_mels, sr, fmin, fmax)
    mel_spectrogram_torch(y, n_fft, num_mels, sr, hop, win, fmin, fmax)
"""

from functools import lru_cache
from typing import Optional

import mlx.core as mx
import numpy as np
from librosa.filters import mel as librosa_mel_fn


@lru_cache(maxsize=8)
def _hann_window(win_size: int) -> mx.array:
    # torch.hann_window(periodic=True) uses 2π n / N (the periodic version,
    # which is the default; the symmetric version divides by N-1).
    n = np.arange(win_size, dtype=np.float32)
    w = 0.5 - 0.5 * np.cos(2 * np.pi * n / win_size)
    return mx.array(w)


@lru_cache(maxsize=8)
def _mel_basis(n_fft: int, num_mels: int, sr: int, fmin: float,
               fmax: Optional[float]) -> mx.array:
    mel = librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=num_mels,
                        fmin=fmin, fmax=fmax)
    return mx.array(mel.astype(np.float32))


def stft(y: mx.array, n_fft: int, hop_size: int, win_size: int,
        center: bool = False) -> mx.array:
    """Real STFT.

    y         : (B, T)
    returns   : (B, n_fft//2+1, n_frames) complex amplitude
                actually we return magnitude as (B, n_fft//2+1, n_frames) below
                via spectrogram(); here we return the complex result.

    Mirrors `torch.stft(center=False, normalized=False, onesided=True,
    return_complex=True)`.  The upstream code pads y by (n_fft-hop)/2 on each
    side with reflect mode *before* calling stft, so we don't pad here.

    NOTE: reflect-padding step is done in spectrogram() to mirror the
    PyTorch pipeline precisely.
    """
    assert win_size == n_fft, "we don't support window-padding the FFT window yet"
    window = _hann_window(win_size)            # (win_size,)
    B, T = y.shape
    n_frames = 1 + (T - n_fft) // hop_size
    # Build (B, n_frames, n_fft) windowed frames.
    # Use mx.arange + broadcasting to gather indices.
    frame_idx = mx.arange(n_frames) * hop_size            # (n_frames,)
    sample_idx = mx.arange(n_fft)                          # (n_fft,)
    # (n_frames, n_fft) = frame_idx[:, None] + sample_idx[None, :]
    indices = frame_idx[:, None] + sample_idx[None, :]
    # Gather: y[:, indices] -> (B, n_frames, n_fft)
    frames = y[:, indices]
    frames = frames * window[None, None, :]
    # rfft along last axis -> (B, n_frames, n_fft//2+1) complex
    spec = mx.fft.rfft(frames, axis=-1)
    # Match torch (B, F, T) layout
    spec = spec.transpose(0, 2, 1)
    return spec


def spectrogram(y: mx.array, n_fft: int, sampling_rate: int, hop_size: int,
                win_size: int, center: bool = False) -> mx.array:
    """Linear magnitude spectrogram, with the same reflect-pad as upstream."""
    # reflect pad: (n_fft - hop) // 2 each side
    pad = (n_fft - hop_size) // 2
    # mx.pad with reflect mode
    y_p = mx.pad(y, [(0, 0), (pad, pad)], mode="edge") \
        if not hasattr(mx, "pad") else mx.pad(y, [(0, 0), (pad, pad)], mode="edge")
    # MLX mx.pad supports mode="constant" by default; reflect added in 0.21+.
    # Try reflect; fall back to numpy if unavailable.
    try:
        y_p = mx.pad(y, [(0, 0), (pad, pad)], mode="reflect")
    except Exception:
        y_np = np.array(y)
        y_p_np = np.pad(y_np, [(0, 0), (pad, pad)], mode="reflect")
        y_p = mx.array(y_p_np)
    spec = stft(y_p, n_fft=n_fft, hop_size=hop_size, win_size=win_size, center=center)
    # |z| = sqrt(re^2 + im^2 + eps)
    # MLX complex magnitude:
    re = mx.real(spec)
    im = mx.imag(spec)
    mag = mx.sqrt(re * re + im * im + 1e-8)
    return mag


def spec_to_mel(spec: mx.array, n_fft: int, num_mels: int, sampling_rate: int,
                fmin: float, fmax: Optional[float]) -> mx.array:
    """spec: (B, F, T) linear magnitude  -> mel: (B, num_mels, T) log-magnitude."""
    mb = _mel_basis(n_fft, num_mels, sampling_rate, fmin, fmax)   # (M, F)
    mel = mx.matmul(mb, spec)                                      # (B, M, T)
    mel = mx.log(mx.maximum(mel, mx.array(1e-5, mel.dtype)))
    return mel


def mel_spectrogram(y: mx.array, n_fft: int, num_mels: int, sampling_rate: int,
                    hop_size: int, win_size: int, fmin: float,
                    fmax: Optional[float], center: bool = False) -> mx.array:
    spec = spectrogram(y, n_fft, sampling_rate, hop_size, win_size, center=center)
    return spec_to_mel(spec, n_fft, num_mels, sampling_rate, fmin, fmax)
