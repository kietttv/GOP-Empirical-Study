"""Group B GOP representations from the Xiaomi/GOPT Kaldi extract.

Layout (same as traditional GOP):

    feat = [phone_id, LPP_0 .. LPP_{n-1}, LPR_0 .. LPR_{n-1}]

On this extract the precomputed LPR slot is

    LPR[q] = LPP[canonical] - LPP[q]

so LPR[canonical] is identically 0 and must not be used as B3.

B1 / B2  canonical LPP = LPP[phone_id]          (identical on this extract)
B3       LPR vs best competitor = LPP[p] - max_{q≠p} LPP[q]
B4       GOP-only vector [LPP[p], max competitor LPP, LPR_vs_best]
         Rank 2: LPR_vs_best = LPP[p] - max competitor, so OLS uses the first two.
B5       84-d GOP feature = [LPP_0..41, LPR_0..41] (GOPT paper naming; OLS direct)
"""

from __future__ import annotations

import numpy as np

from gop_empirical.gop.traditional import traditional_gop_batch

B4_FEATURE_NAMES = ("lpp_canonical", "lpp_max_competitor", "lpr_vs_best")
B4_OLS_FEATURE_NAMES = ("lpp_canonical", "lpp_max_competitor")
B5_N_PHONES = 42
B5_N_FEATURES = 84
SSL_LPP_LPR_N_PHONES = 39
SSL_LPP_LPR_N_FEATURES = 78


def _lpp_matrix(
    feats: np.ndarray,
    phone_index_base: int = 0,
    expected_n_phones: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(idx [N], lpp [N, n], n_phones)`` after validating layout."""
    feats = np.asarray(feats, dtype=np.float64)
    if feats.ndim != 2 or feats.shape[1] < 3:
        raise ValueError(f"expected 2-D GOP matrix, got shape {feats.shape}")
    rest = feats.shape[1] - 1
    if rest % 2 != 0:
        raise ValueError(f"expected even LPP+LPR tail, got {rest}")
    n_phones = rest // 2
    if expected_n_phones is not None and n_phones != expected_n_phones:
        raise ValueError(
            f"expected n_phones={expected_n_phones}, got {n_phones} from feat dim"
        )
    phone_ids = feats[:, 0].astype(np.int64)
    lpp = feats[:, 1 : 1 + n_phones]
    idx = phone_ids - int(phone_index_base)
    bad = (idx < 0) | (idx >= n_phones)
    if np.any(bad):
        i = int(np.flatnonzero(bad)[0])
        raise IndexError(
            f"row {i}: phone_id={int(phone_ids[i])} (base={phone_index_base}) "
            f"out of LPP range [0, {n_phones})"
        )
    return idx, lpp, n_phones


def canonical_lpp(
    feats: np.ndarray,
    phone_index_base: int = 0,
    expected_n_phones: int | None = None,
) -> np.ndarray:
    """B1 / B2: canonical-phone log phone posterior. Same scalar as traditional GOP."""
    return traditional_gop_batch(
        feats,
        phone_index_base=phone_index_base,
        expected_n_phones=expected_n_phones,
    )


def lpp_max_competitor(
    feats: np.ndarray,
    phone_index_base: int = 0,
    expected_n_phones: int | None = None,
) -> np.ndarray:
    """Max LPP among phones other than the canonical one."""
    idx, lpp, n_phones = _lpp_matrix(
        feats, phone_index_base=phone_index_base, expected_n_phones=expected_n_phones
    )
    if n_phones < 2:
        raise ValueError("need at least 2 phone slots to pick a competitor")
    masked = lpp.copy()
    masked[np.arange(lpp.shape[0]), idx] = -np.inf
    return masked.max(axis=1)


def lpr_vs_best_competitor(
    feats: np.ndarray,
    phone_index_base: int = 0,
    expected_n_phones: int | None = None,
) -> np.ndarray:
    """B3: log posterior ratio vs the strongest competing phone.

    LPR(p) = LPP[p] - max_{q≠p} LPP[q] = min_{q≠p} of the precomputed LPR vector.
    """
    canonical = canonical_lpp(
        feats,
        phone_index_base=phone_index_base,
        expected_n_phones=expected_n_phones,
    )
    competitor = lpp_max_competitor(
        feats,
        phone_index_base=phone_index_base,
        expected_n_phones=expected_n_phones,
    )
    return canonical - competitor


def gop_feature_vector(
    feats: np.ndarray,
    phone_index_base: int = 0,
    expected_n_phones: int | None = None,
) -> np.ndarray:
    """B4 GOP-only matrix ``[N, 3]``: canonical LPP, max competitor LPP, LPR vs best.

    The third column is linearly determined by the first two. OLS scoring uses
    columns 0 and 1 (see ``B4_OLS_FEATURE_NAMES``).
    """
    canonical = canonical_lpp(
        feats,
        phone_index_base=phone_index_base,
        expected_n_phones=expected_n_phones,
    )
    competitor = lpp_max_competitor(
        feats,
        phone_index_base=phone_index_base,
        expected_n_phones=expected_n_phones,
    )
    lpr = canonical - competitor
    return np.column_stack([canonical, competitor, lpr])


def gop_feature_vector_ols(
    feats: np.ndarray,
    phone_index_base: int = 0,
    expected_n_phones: int | None = None,
) -> np.ndarray:
    """Rank-2 slice of ``gop_feature_vector`` used for B4 direct (linear) scoring."""
    vec = gop_feature_vector(
        feats,
        phone_index_base=phone_index_base,
        expected_n_phones=expected_n_phones,
    )
    return vec[:, :2]


def gopt_gop_feature_84(
    feats: np.ndarray,
    *,
    expected_n_phones: int = B5_N_PHONES,
) -> np.ndarray:
    """B5: GOPT-paper 84-d GOP feature ``[LPP_0..n-1, LPR_0..n-1]``.

    Xiaomi extract stores ``LPR[q] = LPP[canonical] - LPP[q]`` (sign-flipped vs
    paper ``LPR(pj|p) = LPP(pj) - LPP(p)``); OLS absorbs the sign. Does not flip.
    """
    feats = np.asarray(feats, dtype=np.float64)
    if feats.ndim != 2 or feats.shape[1] < 3:
        raise ValueError(f"expected 2-D GOP matrix, got shape {feats.shape}")
    rest = feats.shape[1] - 1
    if rest % 2 != 0:
        raise ValueError(f"expected even LPP+LPR tail, got {rest}")
    n_phones = rest // 2
    if n_phones != int(expected_n_phones):
        raise ValueError(
            f"B5 expects n_phones={expected_n_phones}, got {n_phones} from feat dim"
        )
    out = feats[:, 1 : 1 + 2 * n_phones]
    if out.shape[1] != B5_N_FEATURES:
        raise ValueError(f"expected {B5_N_FEATURES}-d GOP feature, got {out.shape[1]}")
    return np.asarray(out, dtype=np.float64)


def lpp_lpr_concat(
    lpp: np.ndarray,
    canonical_idx: np.ndarray | int,
) -> np.ndarray:
    """GOPT-style ``[LPP, LPR]`` with ``LPR[q] = LPP[canonical] - LPP[q]``.

    Kaldi B5 uses 42 phones (84-d). C8/C9 analog uses 39 scored CMU-mapped
    IPA tokens (78-d). ``LPR[canonical]`` is 0.
    """
    lpp = np.asarray(lpp, dtype=np.float64)
    if lpp.ndim == 1:
        lpp = lpp.reshape(1, -1)
    if lpp.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D LPP, got shape {lpp.shape}")
    idx = np.asarray(canonical_idx, dtype=np.int64).reshape(-1)
    if idx.size == 1 and lpp.shape[0] > 1:
        idx = np.full(lpp.shape[0], int(idx[0]), dtype=np.int64)
    if idx.size != lpp.shape[0]:
        raise ValueError(f"canonical_idx length {idx.size} != n_rows {lpp.shape[0]}")
    n_phones = lpp.shape[1]
    bad = (idx < 0) | (idx >= n_phones)
    if np.any(bad):
        i = int(np.flatnonzero(bad)[0])
        raise IndexError(f"row {i}: canonical_idx={int(idx[i])} out of [0, {n_phones})")
    can = lpp[np.arange(lpp.shape[0]), idx]
    lpr = can[:, None] - lpp
    return np.concatenate([lpp, lpr], axis=1)


def mean_lpp_on_span(log_probs: np.ndarray, t0: int, t1: int) -> np.ndarray | None:
    """Mean log-posterior vector on ``log_probs[t0:t1]``. ``None`` if empty."""
    log_probs = np.asarray(log_probs, dtype=np.float64)
    if log_probs.ndim != 2:
        raise ValueError(f"expected [T, n_phones] log_probs, got {log_probs.shape}")
    t0 = int(t0)
    t1 = int(t1)
    if t1 <= t0:
        return None
    span = log_probs[t0:t1]
    if span.size == 0:
        return None
    return np.asarray(span.mean(axis=0), dtype=np.float64)


def mean_lpp_on_frames(log_probs: np.ndarray, frame_idx: np.ndarray | None) -> np.ndarray | None:
    """Mean log-posterior on an arbitrary set of frames. ``None`` if empty."""
    log_probs = np.asarray(log_probs, dtype=np.float64)
    if log_probs.ndim != 2:
        raise ValueError(f"expected [T, n_phones] log_probs, got {log_probs.shape}")
    if frame_idx is None:
        return None
    idx = np.asarray(frame_idx, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        return None
    if np.any(idx < 0) or np.any(idx >= log_probs.shape[0]):
        raise IndexError(f"frame_idx out of [0, {log_probs.shape[0]})")
    return np.asarray(log_probs[idx].mean(axis=0), dtype=np.float64)
