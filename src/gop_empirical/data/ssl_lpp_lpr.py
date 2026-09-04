"""GOPT-style LPP+LPR concat for C8/C9 (39 scored phones → 78-d)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from gop_empirical.gop.representation import SSL_LPP_LPR_N_FEATURES

_SPLIT_FILES = {
    "train": "tr_ssl_lpp_lpr.npz",
    "test": "te_ssl_lpp_lpr.npz",
}


def ssl_lpp_lpr_split_path(ssl_dir: str | Path, split: str) -> Path:
    if split not in _SPLIT_FILES:
        raise ValueError(f"unknown split {split!r}; expected train or test")
    return Path(ssl_dir) / _SPLIT_FILES[split]


def ssl_lpp_lpr_dir_ready(ssl_dir: str | Path) -> bool:
    return all(ssl_lpp_lpr_split_path(ssl_dir, split).is_file() for split in ("train", "test"))


def write_ssl_lpp_lpr_split(
    rows: list[dict[str, Any]],
    ssl_dir: str | Path,
    split: str,
) -> Path:
    path = ssl_lpp_lpr_split_path(ssl_dir, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no LPP+LPR rows to write")
    keys = np.asarray([str(r["key"]) for r in rows], dtype=object)
    phones = np.asarray([str(r["phone"]) for r in rows], dtype=object)
    n_frames = np.asarray([int(r["n_frames"]) for r in rows], dtype=np.int32)
    feats = np.vstack([np.asarray(r["features"], dtype=np.float64).reshape(1, -1) for r in rows])
    if feats.shape[1] != SSL_LPP_LPR_N_FEATURES:
        raise ValueError(f"expected {SSL_LPP_LPR_N_FEATURES}-d features, got {feats.shape[1]}")
    np.savez_compressed(path, keys=keys, phones=phones, n_frames=n_frames, features=feats)
    return path


def load_ssl_lpp_lpr_split(ssl_dir: str | Path, split: str) -> dict[str, dict[str, Any]]:
    path = ssl_lpp_lpr_split_path(ssl_dir, split)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = np.load(path, allow_pickle=True)
    keys = payload["keys"]
    phones = payload["phones"]
    n_frames = payload["n_frames"]
    features = np.asarray(payload["features"], dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != SSL_LPP_LPR_N_FEATURES:
        raise ValueError(f"{path}: expected [N, {SSL_LPP_LPR_N_FEATURES}] features, got {features.shape}")
    if not (len(keys) == len(phones) == len(n_frames) == features.shape[0]):
        raise ValueError(f"{path}: mismatched array lengths")
    out: dict[str, dict[str, Any]] = {}
    for i, key in enumerate(keys):
        out[str(key)] = {
            "phone": str(phones[i]),
            "n_frames": int(n_frames[i]),
            "features": features[i],
        }
    return out


def load_ssl_lpp_lpr_dir(ssl_dir: str | Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for split in ("train", "test"):
        out.update(load_ssl_lpp_lpr_split(ssl_dir, split))
    return out
