"""CTC Viterbi forced alignment (same graph as GOP-S; no Kaldi start/end)."""

from __future__ import annotations

import numpy as np

_NEG = -1.0e30


def _log_emit(log_probs: np.ndarray, seq: np.ndarray, blank: int, state: int, t: int) -> float:
    if state % 2 == 0:
        return float(log_probs[t, blank])
    return float(log_probs[t, seq[(state - 1) // 2]])


def ctc_viterbi_states(
    frame_probs: np.ndarray,
    seq: np.ndarray,
    *,
    blank: int = 0,
) -> np.ndarray | None:
    """Return ``[T]`` CTC graph-state ids, or ``None`` if no finite path."""
    probs = np.asarray(frame_probs, dtype=np.float64)
    seq = np.asarray(seq, dtype=np.int64).reshape(-1)
    if probs.ndim != 2:
        raise ValueError(f"expected [T, V] frame_probs, got {probs.shape}")
    if seq.size < 1:
        return None
    vocab = int(probs.shape[1])
    if np.any(seq < 0) or np.any(seq >= vocab):
        raise IndexError(f"seq ids out of vocab size {vocab}")
    if blank < 0 or blank >= vocab:
        raise IndexError(f"blank={blank} out of vocab size {vocab}")
    log_p = np.log(np.clip(probs, 1e-30, 1.0))
    t_len = int(log_p.shape[0])
    n_state = 2 * int(seq.size) + 1
    dp = np.full((t_len, n_state), _NEG, dtype=np.float64)
    bp = np.full((t_len, n_state), -1, dtype=np.int32)

    dp[0, 0] = _log_emit(log_p, seq, blank, 0, 0)
    if n_state > 1:
        dp[0, 1] = _log_emit(log_p, seq, blank, 1, 0)

    for t in range(1, t_len):
        for state in range(n_state):
            emit = _log_emit(log_p, seq, blank, state, t)
            best_score = dp[t - 1, state]
            best_prev = state
            if state - 1 >= 0 and dp[t - 1, state - 1] > best_score:
                best_score = dp[t - 1, state - 1]
                best_prev = state - 1
            if state >= 2 and state % 2 == 1:
                label_i = (state - 1) // 2
                if label_i > 0 and int(seq[label_i]) != int(seq[label_i - 1]):
                    if dp[t - 1, state - 2] > best_score:
                        best_score = dp[t - 1, state - 2]
                        best_prev = state - 2
            dp[t, state] = best_score + emit
            bp[t, state] = best_prev

    end_a = n_state - 1
    end_b = n_state - 2
    end_state = end_a if dp[t_len - 1, end_a] >= dp[t_len - 1, end_b] else end_b
    if dp[t_len - 1, end_state] <= _NEG / 2:
        return None

    states = np.empty(t_len, dtype=np.int32)
    state = int(end_state)
    for t in range(t_len - 1, -1, -1):
        states[t] = state
        prev = int(bp[t, state])
        if t == 0:
            break
        if prev < 0:
            return None
        state = prev
    return states


def ctc_label_frames(
    frame_probs: np.ndarray,
    seq: np.ndarray,
    *,
    blank: int = 0,
) -> list[np.ndarray]:
    """Non-blank Viterbi frames for each canonical phone (CTC self-alignment)."""
    seq = np.asarray(seq, dtype=np.int64).reshape(-1)
    states = ctc_viterbi_states(frame_probs, seq, blank=blank)
    out: list[list[int]] = [[] for _ in range(int(seq.size))]
    if states is None:
        return [np.zeros(0, dtype=np.int64) for _ in range(int(seq.size))]
    for t, state in enumerate(states.tolist()):
        if int(state) % 2 == 0:
            continue
        label_i = (int(state) - 1) // 2
        if 0 <= label_i < len(out):
            out[label_i].append(t)
    return [np.asarray(frames, dtype=np.int64) for frames in out]
