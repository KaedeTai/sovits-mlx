"""MLX equivalents of GPT_SoVITS/module/commons.py utilities used in training.

Only what the S2 training loop actually calls is ported:
  - rand_slice_segments(x, x_lengths, segment_size) -> (slice, ids_str)
  - slice_segments(x, ids_str, segment_size)
  - clip_grad_value (PyTorch's clip_grad_value_ but returns norm; with clip=None
    it's used purely for logging the global gradient norm)
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import math
import mlx.core as mx


def slice_segments(x: mx.array, ids_str: mx.array, segment_size: int) -> mx.array:
    """Per-batch slice of length `segment_size` from x[b, :, ids_str[b]:ids_str[b]+segment_size].

    x        : (B, C, T)
    ids_str  : (B,) int — start indices, one per batch row
    returns  : (B, C, segment_size)

    Implementation: gather via per-row arange index. MLX `mx.take_along_axis`
    handles the variable-start case cleanly.
    """
    B, C, T = x.shape
    # Build a (B, segment_size) index array: ids_str[b] + arange(segment_size)
    base = mx.arange(segment_size, dtype=mx.int32)                          # (S,)
    idx = mx.expand_dims(ids_str.astype(mx.int32), -1) + mx.expand_dims(base, 0)  # (B, S)
    # Broadcast to (B, C, S) so we can take_along_axis on the time dimension
    idx = mx.broadcast_to(mx.expand_dims(idx, 1), (B, C, segment_size))
    return mx.take_along_axis(x, idx, axis=2)


def rand_slice_segments(
    x: mx.array,
    x_lengths: mx.array,
    segment_size: int,
    key: mx.array = None,
) -> Tuple[mx.array, mx.array]:
    """Randomly slice `segment_size` frames from each row in batch.

    x        : (B, C, T)
    x_lengths: (B,) int — true length per row (slice start can be at most x_lengths-segment_size)
    segment_size : int

    Returns (sliced, ids_str) where sliced has shape (B, C, segment_size).
    """
    B = x.shape[0]
    if key is None:
        key = mx.random.key(0)
    # ids_str_max = max(x_lengths - segment_size + 1, 1) — safe clamp for very short rows.
    ids_str_max = mx.maximum(x_lengths - segment_size + 1, mx.array(1, dtype=x_lengths.dtype))
    r = mx.random.uniform(low=0.0, high=1.0, shape=(B,), key=key)
    ids_str = (r * ids_str_max.astype(mx.float32)).astype(mx.int32)
    sliced = slice_segments(x, ids_str, segment_size)
    return sliced, ids_str


def sequence_mask(length: mx.array, max_length: int = None) -> mx.array:
    """Boolean → float mask of shape (B, 1, max_length).

    length      : (B,) int
    max_length  : int (defaults to length.max())
    returns     : (B, 1, max_length) float mask
    """
    if max_length is None:
        max_length = int(mx.max(length).item())
    rng = mx.arange(max_length, dtype=length.dtype)                # (T,)
    mask = mx.expand_dims(rng, 0) < mx.expand_dims(length, 1)       # (B, T) bool
    mask = mask.astype(mx.float32)
    return mx.expand_dims(mask, 1)                                  # (B, 1, T)


def grad_global_norm(grads_tree, eps: float = 1e-12) -> mx.array:
    """Global L2 norm over a tree of gradients (for logging, not clipping).

    Mirrors PyTorch `clip_grad_value_(..., clip_value=None)` which in the
    upstream s2_train.py is used purely for the grad_norm_g / grad_norm_d log
    fields.
    """
    from mlx.utils import tree_flatten
    total = mx.array(0.0)
    for _, g in tree_flatten(grads_tree):
        total = total + mx.sum(g * g)
    return mx.sqrt(total + eps)
