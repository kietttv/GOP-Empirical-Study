"""Load Xiaomi/GOPT raw Kaldi GOP CSVs (no Kaldi runtime)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_SPLIT_FILES = {
    "train": ("tr_feats.csv", "tr_keys_phn.csv"),
    "test": ("te_feats.csv", "te_keys_phn.csv"),
}


def list_kaldi_splits(kaldi_dir: str | Path) -> list[str]:
    kaldi_dir = Path(kaldi_dir)
    found = []
    for split, (feat_name, key_name) in _SPLIT_FILES.items():
        if (kaldi_dir / feat_name).is_file() and (kaldi_dir / key_name).is_file():
            found.append(split)
    return found


def load_kaldi_gop_split(kaldi_dir: str | Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(feats [N, 85], keys [N] of 'utt_id.phn_idx')``."""
    if split not in _SPLIT_FILES:
        raise ValueError(f"unknown split {split!r}; expected train or test")
    kaldi_dir = Path(kaldi_dir)
    feat_name, key_name = _SPLIT_FILES[split]
    feat_path = kaldi_dir / feat_name
    key_path = kaldi_dir / key_name
    if not feat_path.is_file():
        raise FileNotFoundError(feat_path)
    if not key_path.is_file():
        raise FileNotFoundError(key_path)

    feats = np.loadtxt(feat_path, delimiter=",", dtype=np.float64)
    if feats.ndim == 1:
        feats = feats.reshape(1, -1)
    keys = np.loadtxt(key_path, delimiter=",", dtype=str)
    if keys.ndim == 0:
        keys = np.array([str(keys)])
    keys = np.asarray(keys, dtype=str).reshape(-1)
    if feats.shape[0] != keys.shape[0]:
        raise ValueError(
            f"{split}: feat rows {feats.shape[0]} != key rows {keys.shape[0]}"
        )
    return feats, keys
