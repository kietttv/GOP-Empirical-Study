"""Cao et al. alignment-free CTC GOP (Interspeech 2024).

Uses the canonical phone *sequence* only — not Kaldi start/end times.

GOP at position i is the scaled CTC log-likelihood ratio

    GOP(i) = log P(y | x) - log P(y with y_i free | x)

GOP-S (AF-S): denominator is substitution wildcard only.
GOP-SD (AF-SD): denominator also allows deletion (skip) of y_i.

Blank stays in the CTC graph; this is not the 39-phone renormalized LPP of C2/C3.
Both variants use per-frame ``alpha_bar`` scaling (same as GOP-S) so S vs SD
isolates the graph, not the CTC numeric convention.
"""

from __future__ import annotations

import numpy as np


def _check_params(params: np.ndarray, seq: np.ndarray, blank: int) -> tuple[np.ndarray, np.ndarray]:
    params = np.asarray(params, dtype=np.float64)
    seq = np.asarray(seq, dtype=np.int64).reshape(-1)
    if params.ndim != 2:
        raise ValueError(f"expected [V, T] params, got {params.shape}")
    vocab, _t = params.shape
    if seq.size < 1:
        raise ValueError("empty canonical sequence")
    if np.any(seq < 0) or np.any(seq >= vocab):
        raise IndexError(f"seq ids out of vocab size {vocab}: {seq}")
    if blank < 0 or blank >= vocab:
        raise IndexError(f"blank={blank} out of vocab size {vocab}")
    if np.any(params < 0) or not np.isfinite(params).all():
        raise ValueError("CTC params must be finite non-negative probabilities")
    return params, seq


def ctc_nll(params: np.ndarray, seq: np.ndarray, blank: int = 0) -> float:
    """Scaled CTC negative log-likelihood (Cao ``ctc_loss``). ``params`` is ``[V, T]``."""
    params, seq = _check_params(params, seq, blank)
    seq_len = int(seq.shape[0])
    t_len = int(params.shape[1])
    graph_len = 2 * seq_len + 1
    alphas = np.zeros((graph_len, t_len), dtype=np.float64)
    alpha_bar = np.zeros(t_len, dtype=np.float64)

    alphas[0, 0] = params[blank, 0]
    alphas[1, 0] = params[seq[0], 0]
    alpha_bar[0] = alphas[:, 0].sum()
    if alpha_bar[0] <= 0:
        return float("inf")
    alphas[:, 0] /= alpha_bar[0]

    for t in range(1, t_len):
        start = max(0, graph_len - 2 * (t_len - t))
        for s in range(start, graph_len):
            label_i = (s - 1) // 2
            if s % 2 == 0:
                if s == 0:
                    alphas[s, t] = alphas[s, t - 1] * params[blank, t]
                else:
                    alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1]) * params[blank, t]
            elif s == 1 or seq[label_i] == seq[label_i - 1]:
                alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1]) * params[seq[label_i], t]
            else:
                alphas[s, t] = (
                    alphas[s, t - 1] + alphas[s - 1, t - 1] + alphas[s - 2, t - 1]
                ) * params[seq[label_i], t]
        alpha_bar[t] = alphas[:, t].sum()
        if alpha_bar[t] <= 0:
            return float("inf")
        alphas[:, t] /= alpha_bar[t]
    return float(-np.log(alpha_bar).sum())


def _arbitrary_mass(alphas: np.ndarray, s: int, t: int, zero_pos: list[int]) -> float | None:
    row = alphas[s, t]
    if np.count_nonzero(row) <= 1:
        return None
    if not zero_pos:
        return float(row.sum())
    mask = np.ones(row.shape[0], dtype=bool)
    mask[np.asarray(zero_pos, dtype=np.int64)] = False
    return float(row[mask].sum())


def _alpha_bar_wildcard(alphas: np.ndarray, t: int, blank: int, pos: int) -> float:
    arbitrary = 2 * pos + 1
    mask = np.ones(alphas.shape[2], dtype=bool)
    mask[blank] = False
    return float(
        alphas[:arbitrary, t, 0].sum()
        + alphas[arbitrary + 1 :, t, 0].sum()
        + alphas[arbitrary, t, mask].sum()
    )


def ctc_nll_wildcard(params: np.ndarray, seq: np.ndarray, pos: int, blank: int = 0) -> float:
    """CTC NLL with label ``pos`` replaced by Cao's substitution wildcard."""
    params, seq = _check_params(params, seq, blank)
    pos = int(pos)
    seq_len = int(seq.shape[0])
    if pos < 0 or pos >= seq_len:
        raise IndexError(f"pos={pos} out of sequence length {seq_len}")
    vocab, t_len = int(params.shape[0]), int(params.shape[1])
    graph_len = 2 * seq_len + 1
    alphas = np.zeros((graph_len, t_len, vocab), dtype=np.float64)
    alpha_bar = np.zeros(t_len, dtype=np.float64)

    if pos == 0:
        alphas[0, 0, 0] = params[blank, 0]
        alphas[1, 0, :] = params[:, 0]
        alphas[1, 0, 0] = 0.0
    else:
        alphas[0, 0, 0] = params[blank, 0]
        alphas[1, 0, 0] = params[seq[0], 0]
    alpha_bar[0] = _alpha_bar_wildcard(alphas, 0, blank, pos)
    if alpha_bar[0] <= 0:
        return float("inf")
    alphas[:, 0, :] /= alpha_bar[0]

    for t in range(1, t_len):
        start = max(0, graph_len - 2 * (t_len - t))
        for s in range(start, graph_len):
            label_i = (s - 1) // 2
            if s % 2 == 0:
                if s == 0:
                    alphas[s, t, 0] = alphas[s, t - 1, 0] * params[blank, t]
                else:
                    collected = _arbitrary_mass(alphas, s - 1, t - 1, [blank])
                    if collected is not None:
                        alphas[s, t, 0] = (alphas[s, t - 1, 0] + collected) * params[blank, t]
                    else:
                        alphas[s, t, 0] = (alphas[s, t - 1, 0] + alphas[s - 1, t - 1, 0]) * params[
                            blank, t
                        ]
            elif pos != label_i and pos != label_i - 1:
                if s == 1 or seq[label_i] == seq[label_i - 1]:
                    alphas[s, t, 0] = (alphas[s, t - 1, 0] + alphas[s - 1, t - 1, 0]) * params[
                        seq[label_i], t
                    ]
                else:
                    alphas[s, t, 0] = (
                        alphas[s, t - 1, 0] + alphas[s - 1, t - 1, 0] + alphas[s - 2, t - 1, 0]
                    ) * params[seq[label_i], t]
            elif pos == label_i - 1:
                collected = _arbitrary_mass(alphas, s - 2, t - 1, [blank, int(seq[label_i])])
                extra = 0.0 if collected is None else collected
                alphas[s, t, 0] = (alphas[s, t - 1, 0] + alphas[s - 1, t - 1, 0] + extra) * params[
                    seq[label_i], t
                ]
            elif s == 1:
                empty_prob = alphas[s - 1, t - 1, 0] * params[:, t]
                empty_prob[blank] = 0.0
                stay = alphas[s, t - 1, :] * params[:, t]
                alphas[s, t, :] = stay + empty_prob
            else:
                skip_prob = alphas[s - 2, t - 1, 0] * params[:, t]
                skip_prob[int(seq[label_i - 1])] = 0.0
                skip_prob[blank] = 0.0
                empty_prob = alphas[s - 1, t - 1, 0] * params[:, t]
                empty_prob[blank] = 0.0
                stay = alphas[s, t - 1, :] * params[:, t]
                alphas[s, t, :] = stay + skip_prob + empty_prob
        alpha_bar[t] = _alpha_bar_wildcard(alphas, t, blank, pos)
        if alpha_bar[t] <= 0:
            return float("inf")
        alphas[:, t, :] /= alpha_bar[t]
    return float(-np.log(alpha_bar).sum())


def ctc_nll_wildcard_sd(params: np.ndarray, seq: np.ndarray, pos: int, blank: int = 0) -> float:
    """CTC NLL with label ``pos`` as substitution-or-deletion wildcard (AF-SD).

    Skip paths follow Cao ``gop-ctc-af-SD.py``; scaling matches ``ctc_nll_wildcard``.
    """
    params, seq = _check_params(params, seq, blank)
    pos = int(pos)
    seq_len = int(seq.shape[0])
    if pos < 0 or pos >= seq_len:
        raise IndexError(f"pos={pos} out of sequence length {seq_len}")
    vocab, t_len = int(params.shape[0]), int(params.shape[1])
    graph_len = 2 * seq_len + 1
    alphas = np.zeros((graph_len, t_len, vocab), dtype=np.float64)
    alpha_bar = np.zeros(t_len, dtype=np.float64)

    if pos == 0:
        alphas[0, 0, 0] = params[blank, 0]
        alphas[1, 0, :] = params[:, 0]
        alphas[1, 0, blank] = 0.0
        if seq_len > 1:
            alphas[3, 0, 0] = params[seq[1], 0]
    else:
        alphas[0, 0, 0] = params[blank, 0]
        alphas[1, 0, 0] = params[seq[0], 0]
    alpha_bar[0] = _alpha_bar_wildcard(alphas, 0, blank, pos)
    if alpha_bar[0] <= 0:
        return float("inf")
    alphas[:, 0, :] /= alpha_bar[0]

    for t in range(1, t_len):
        if pos == seq_len - 1:
            lowest = graph_len - 2 * (t_len - t + 1)
        else:
            lowest = graph_len - 2 * (t_len - t)
        start = max(0, lowest)
        for s in range(start, graph_len):
            label_i = (s - 1) // 2
            if s % 2 == 0:
                if s == 0:
                    alphas[s, t, 0] = alphas[s, t - 1, 0] * params[blank, t]
                else:
                    collected = _arbitrary_mass(alphas, s - 1, t - 1, [blank])
                    if collected is not None:
                        alphas[s, t, 0] = (alphas[s, t - 1, 0] + collected) * params[blank, t]
                    else:
                        alphas[s, t, 0] = (alphas[s, t - 1, 0] + alphas[s - 1, t - 1, 0]) * params[
                            blank, t
                        ]
            elif pos != label_i and pos != label_i - 1:
                if s == 1 or seq[label_i] == seq[label_i - 1]:
                    alphas[s, t, 0] = (alphas[s, t - 1, 0] + alphas[s - 1, t - 1, 0]) * params[
                        seq[label_i], t
                    ]
                else:
                    alphas[s, t, 0] = (
                        alphas[s, t - 1, 0] + alphas[s - 1, t - 1, 0] + alphas[s - 2, t - 1, 0]
                    ) * params[seq[label_i], t]
            elif pos == label_i - 1:
                collected = _arbitrary_mass(alphas, s - 2, t - 1, [blank, int(seq[label_i])])
                extra = 0.0 if collected is None else collected
                skip_token = 0.0
                if (
                    label_i - 2 >= 0
                    and int(seq[label_i - 2]) != int(seq[label_i])
                    and s >= 4
                ):
                    skip_token = alphas[s - 4, t - 1, 0] * params[seq[label_i], t]
                skip_empty = 0.0 if s < 3 else alphas[s - 3, t - 1, 0] * params[seq[label_i], t]
                alphas[s, t, 0] = (
                    alphas[s, t - 1, 0] + alphas[s - 1, t - 1, 0] + extra
                ) * params[seq[label_i], t] + skip_empty + skip_token
            elif s == 1:
                empty_prob = alphas[s - 1, t - 1, 0] * params[:, t]
                empty_prob[blank] = 0.0
                stay = alphas[s, t - 1, :] * params[:, t]
                alphas[s, t, :] = stay + empty_prob
            else:
                skip_prob = alphas[s - 2, t - 1, 0] * params[:, t]
                skip_prob[int(seq[label_i - 1])] = 0.0
                skip_prob[blank] = 0.0
                empty_prob = alphas[s - 1, t - 1, 0] * params[:, t]
                empty_prob[blank] = 0.0
                stay = alphas[s, t - 1, :] * params[:, t]
                alphas[s, t, :] = stay + skip_prob + empty_prob
        alpha_bar[t] = _alpha_bar_wildcard(alphas, t, blank, pos)
        if alpha_bar[t] <= 0:
            return float("inf")
        alphas[:, t, :] /= alpha_bar[t]
    return float(-np.log(alpha_bar).sum())


def _cao_gop(
    frame_probs: np.ndarray,
    seq: np.ndarray,
    denom_fn,
    *,
    blank: int,
) -> np.ndarray:
    frame_probs = np.asarray(frame_probs, dtype=np.float64)
    if frame_probs.ndim != 2:
        raise ValueError(f"expected [T, V] frame_probs, got {frame_probs.shape}")
    params = frame_probs.T
    seq = np.asarray(seq, dtype=np.int64).reshape(-1)
    nll_num = ctc_nll(params, seq, blank=blank)
    out = np.empty(seq.shape[0], dtype=np.float64)
    for i in range(seq.shape[0]):
        nll_den = denom_fn(params, seq, i, blank=blank)
        gop = -nll_num + nll_den
        out[i] = gop if np.isfinite(gop) else np.nan
    return out


def cao_gop_s(
    frame_probs: np.ndarray,
    seq: np.ndarray,
    *,
    blank: int = 0,
) -> np.ndarray:
    """Phone-level GOP-S (AF-S). ``frame_probs`` is ``[T, V]`` CTC softmax (blank included)."""
    return _cao_gop(frame_probs, seq, ctc_nll_wildcard, blank=blank)


def cao_gop_sd(
    frame_probs: np.ndarray,
    seq: np.ndarray,
    *,
    blank: int = 0,
) -> np.ndarray:
    """Phone-level GOP-SD (AF-SD). Same numerator as GOP-S; deletion skip in the denom."""
    return _cao_gop(frame_probs, seq, ctc_nll_wildcard_sd, blank=blank)
