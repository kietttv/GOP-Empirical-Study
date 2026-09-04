"""Traditional GOP = canonical-phone log phone posterior (Witt & Young / Xiaomi LPP).

Kaldi Librispeech GOP vector per aligned phone:

    feat = [phone_id, LPP_0 .. LPP_{n-1}, LPR_0 .. LPR_{n-1}]

GOP(p) = LPP[phone_id - phone_index_base]

LPR is kept only so later Group B can reuse the same parser; A1 does not use it.
"""

from __future__ import annotations

import numpy as np


def split_lpp_lpr(feat: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """Split a raw Kaldi GOP row into phone id, LPP vector, LPR vector."""
    feat = np.asarray(feat, dtype=np.float64).reshape(-1)
    if feat.size < 3:
        raise ValueError(f"GOP vector too short: dim={feat.size}")
    rest = feat.size - 1
    if rest % 2 != 0:
        raise ValueError(f"expected even LPP+LPR tail, got {rest} dims after phone_id")
    n_phones = rest // 2
    phone_id = int(feat[0])
    lpp = feat[1 : 1 + n_phones]
    lpr = feat[1 + n_phones :]
    return phone_id, lpp, lpr


def traditional_gop(feat: np.ndarray, phone_index_base: int = 0) -> float:
    """Scalar GOP: LPP of the canonical phone in ``feat``.

    Parameters
    ----------
    feat:
        Raw Kaldi row ``[phone_id, LPP..., LPR...]``.
    phone_index_base:
        0 if ``phone_id`` is already an index into LPP (Xiaomi/GOPT Librispeech).
        1 if ids are 1-based Kaldi symbols.
    """
    phone_id, lpp, _lpr = split_lpp_lpr(feat)
    idx = phone_id - phone_index_base
    if idx < 0 or idx >= lpp.size:
        raise IndexError(
            f"canonical phone_id={phone_id} (base={phone_index_base}) "
            f"out of LPP range [0, {lpp.size})"
        )
    return float(lpp[idx])


def traditional_gop_batch(
    feats: np.ndarray,
    phone_index_base: int = 0,
    expected_n_phones: int | None = None,
) -> np.ndarray:
    """Vectorized ``traditional_gop`` over a ``[N, 1+2n]`` feature matrix."""
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
    return lpp[np.arange(feats.shape[0]), idx]
