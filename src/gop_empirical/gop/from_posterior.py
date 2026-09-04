"""Witt & Young GOP from frame-level phone log-posteriors (Group C SSL).

C2/C3: mean log P(canonical) on a Kaldi span after blank tokens are removed.
C4: max log P(canonical) on the same span. C5 (Cao GOP-S) lives in ``cao.py``.
"""

from __future__ import annotations

import numpy as np


def log_softmax_over_ids(logits: np.ndarray, phone_ids: np.ndarray) -> np.ndarray:
    """Log-softmax of ``logits`` restricted to ``phone_ids`` (blank stripped).

    Parameters
    ----------
    logits:
        ``[T, V]`` raw CTC logits (or a single frame ``[V]``).
    phone_ids:
        Indices of the scored phones in the CTC vocab (length 39 for Group C).

    Returns
    -------
    log_probs:
        ``[T, n_phones]`` (or ``[n_phones]``) log P over the scored phones only.
        Each row sums to 1 in probability space.
    """
    logits = np.asarray(logits, dtype=np.float64)
    phone_ids = np.asarray(phone_ids, dtype=np.int64).reshape(-1)
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
        squeeze = True
    elif logits.ndim == 2:
        squeeze = False
    else:
        raise ValueError(f"expected 1-D or 2-D logits, got shape {logits.shape}")
    if phone_ids.size < 1:
        raise ValueError("need at least one scored phone id")
    if np.any(phone_ids < 0) or np.any(phone_ids >= logits.shape[1]):
        raise IndexError(
            f"phone_ids out of range for vocab size {logits.shape[1]}: {phone_ids}"
        )
    selected = logits[:, phone_ids]
    # logsumexp along phones
    m = selected.max(axis=1, keepdims=True)
    log_z = m + np.log(np.exp(selected - m).sum(axis=1, keepdims=True))
    log_probs = selected - log_z
    if squeeze:
        return log_probs.reshape(-1)
    return log_probs


def log_softmax_rows(logits: np.ndarray) -> np.ndarray:
    """Row-wise log-softmax over the last axis. ``logits`` is ``[T, V]``."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError(f"expected 2-D logits, got {logits.shape}")
    m = logits.max(axis=1, keepdims=True)
    log_z = m + np.log(np.exp(logits - m).sum(axis=1, keepdims=True))
    return logits - log_z


def _canonical_span(
    log_probs: np.ndarray, canonical_idx: int, t0: int, t1: int
) -> np.ndarray | None:
    log_probs = np.asarray(log_probs, dtype=np.float64)
    if log_probs.ndim != 2:
        raise ValueError(f"expected [T, n_phones] log_probs, got {log_probs.shape}")
    n_phones = log_probs.shape[1]
    if canonical_idx < 0 or canonical_idx >= n_phones:
        raise IndexError(f"canonical_idx={canonical_idx} out of [0, {n_phones})")
    t0 = int(t0)
    t1 = int(t1)
    if t1 <= t0:
        return None
    span = log_probs[t0:t1, canonical_idx]
    if span.size == 0:
        return None
    return span


def gop_from_log_probs(
    log_probs: np.ndarray,
    canonical_idx: int,
    t0: int,
    t1: int,
) -> tuple[float, int]:
    """Mean log P(canonical) on ``log_probs[t0:t1]``.

    Returns ``(gop, n_frames)``. ``n_frames == 0`` if the span is empty; GOP is
    then NaN and the caller should drop the phone (``n_empty_segment``).
    """
    span = _canonical_span(log_probs, canonical_idx, t0, t1)
    if span is None:
        return float("nan"), 0
    return float(np.mean(span)), int(span.size)


def gop_max_from_log_probs(
    log_probs: np.ndarray,
    canonical_idx: int,
    t0: int,
    t1: int,
) -> tuple[float, int]:
    """Max log P(canonical) on the Kaldi span (C4 / CTC spike on forced alignment)."""
    span = _canonical_span(log_probs, canonical_idx, t0, t1)
    if span is None:
        return float("nan"), 0
    return float(np.max(span)), int(span.size)


def gop_from_span(span_log_probs: np.ndarray, canonical_idx: int) -> tuple[float, int]:
    """Mean log P(canonical) over an already-sliced ``[T_p, n_phones]`` span."""
    span_log_probs = np.asarray(span_log_probs, dtype=np.float64)
    if span_log_probs.ndim != 2:
        raise ValueError(f"expected 2-D span, got {span_log_probs.shape}")
    return gop_from_log_probs(span_log_probs, canonical_idx, 0, span_log_probs.shape[0])
