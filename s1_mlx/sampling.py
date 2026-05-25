"""MLX equivalents of the PT sampling routines used by S1.

The PT functions live in AR/models/utils.py.  We replicate `sample`
(repetition penalty → top-p → temperature → top-k → softmax → Gumbel-max
multinomial).

Notes on MLX-isms:
  * MLX has no `mx.flip`. To sort descending we use `mx.argsort(-x)` which
    returns indices that put the largest values first.
  * Scatter via `mx.array.at[idx].add(values)` is the supported in-place form.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx


def _argsort_desc(x: mx.array, axis: int = -1) -> mx.array:
    """Return indices that sort `x` in descending order along `axis`."""
    return mx.argsort(-x, axis=axis)


def logits_to_probs(
    logits: mx.array,
    previous_tokens: Optional[mx.array] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
) -> mx.array:
    """Mirror of PT logits_to_probs.

    logits:           (B, V) float
    previous_tokens:  (B, L) int — for repetition penalty
    Returns:          (B, V) probabilities (post all filters, after softmax)
    """
    NEG_INF = mx.array(-1e9, dtype=logits.dtype)

    # ---- repetition penalty ----
    if previous_tokens is not None and repetition_penalty != 1.0:
        B, V = logits.shape
        prev = previous_tokens.astype(mx.int32)
        # Build a (B, V) bool mask: True where token id is in previous_tokens row.
        # Scatter-add on a flattened (B*V,) buffer.
        b_idx = mx.broadcast_to(mx.arange(B)[:, None], prev.shape)
        flat_idx = (b_idx * V + prev).reshape(-1)
        flat_oh = mx.zeros((B * V,), dtype=mx.int32)
        flat_oh = flat_oh.at[flat_idx].add(mx.ones(flat_idx.shape, dtype=mx.int32))
        mask = flat_oh.reshape(B, V) > 0
        # Apply penalty: scores<0 → ×p, scores≥0 → /p
        penalised_neg = logits * repetition_penalty
        penalised_pos = logits / repetition_penalty
        penalised = mx.where(logits < 0, penalised_neg, penalised_pos)
        logits = mx.where(mask, penalised, logits)

    # ---- top-p ----
    if top_p is not None and top_p < 1.0:
        sorted_idx = _argsort_desc(logits, axis=-1)              # (B, V) int
        sorted_logits = mx.take_along_axis(logits, sorted_idx, axis=-1)
        probs = mx.softmax(sorted_logits, axis=-1)
        cum = mx.cumsum(probs, axis=-1)
        remove = cum > top_p
        # Shift right by one column — keep the first token that crosses threshold.
        first_col = mx.zeros((logits.shape[0], 1), dtype=remove.dtype)
        remove = mx.concatenate([first_col, remove[:, :-1]], axis=1)
        # Inverse permutation to map `remove` back into original token order.
        order_back = mx.argsort(sorted_idx, axis=-1)
        remove_unsorted = mx.take_along_axis(remove, order_back, axis=-1)
        logits = mx.where(remove_unsorted, NEG_INF, logits)

    # ---- temperature ----
    logits = logits / max(temperature, 1e-5)

    # ---- top-k ----
    if top_k is not None and top_k > 0 and top_k < logits.shape[-1]:
        # mx.topk returns the top-k VALUES along the last axis (unsorted).
        # We need the k-th largest (the minimum of those values) per row.
        top_vals = mx.topk(logits, top_k, axis=-1)               # (B, k) — unsorted
        pivot = mx.min(top_vals, axis=-1, keepdims=True)         # (B, 1) — k-th largest
        logits = mx.where(logits < pivot, NEG_INF, logits)

    return mx.softmax(logits, axis=-1)


def multinomial_sample_one(probs: mx.array, key: Optional[mx.array] = None) -> mx.array:
    """Gumbel-max trick: argmax(probs / Exp(1)).

    Mirror of PT `multinomial_sample_one_no_sync`.

    probs: (B, V)  →  returns (B, 1) int
    """
    if key is None:
        u = mx.random.uniform(shape=probs.shape, low=1e-12, high=1.0)
    else:
        u = mx.random.uniform(shape=probs.shape, low=1e-12, high=1.0, key=key)
    q = -mx.log(u)
    return mx.argmax(probs / q, axis=-1, keepdims=True).astype(mx.int32)


def sample(
    logits: mx.array,
    previous_tokens: Optional[mx.array] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = 1.0,
    repetition_penalty: float = 1.0,
    temperature: float = 1.0,
    key: Optional[mx.array] = None,
):
    """Mirror of PT `sample` — returns (idx_next, probs)."""
    probs = logits_to_probs(
        logits,
        previous_tokens=previous_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p if top_p is not None and top_p < 1.0 else None,
        repetition_penalty=repetition_penalty,
    )
    idx = multinomial_sample_one(probs, key=key)
    return idx, probs
