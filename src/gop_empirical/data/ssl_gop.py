"""Load SSL GOP CSVs produced by scripts/extract_ssl_gop.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

_SPLIT_FILES = {
    "train": "tr_ssl_gop.csv",
    "test": "te_ssl_gop.csv",
}

SSL_GOP_COLUMNS = ("key", "phone", "gop", "n_frames")


def ssl_gop_split_path(ssl_dir: str | Path, split: str) -> Path:
    if split not in _SPLIT_FILES:
        raise ValueError(f"unknown split {split!r}; expected train or test")
    return Path(ssl_dir) / _SPLIT_FILES[split]


def list_ssl_gop_splits(ssl_dir: str | Path) -> list[str]:
    found = []
    for split in ("train", "test"):
        if ssl_gop_split_path(ssl_dir, split).is_file():
            found.append(split)
    return found


def ssl_gop_dir_ready(ssl_dir: str | Path) -> bool:
    return list_ssl_gop_splits(ssl_dir) == ["train", "test"]


def load_ssl_gop_split(ssl_dir: str | Path, split: str) -> dict[str, dict[str, Any]]:
    """Map ``utt_id.phn_idx`` -> phone, gop, n_frames."""
    path = ssl_gop_split_path(ssl_dir, split)
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype={"key": str, "phone": str})
    missing = [c for c in SSL_GOP_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    out: dict[str, dict[str, Any]] = {}
    for row in df.itertuples(index=False):
        key = str(row.key)
        out[key] = {
            "phone": str(row.phone),
            "gop": float(row.gop),
            "n_frames": int(row.n_frames),
        }
    return out


def write_ssl_gop_split(
    rows: list[dict[str, Any]],
    ssl_dir: str | Path,
    split: str,
) -> Path:
    path = ssl_gop_split_path(ssl_dir, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    missing = [c for c in SSL_GOP_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"SSL GOP rows missing columns: {missing}")
    df.loc[:, list(SSL_GOP_COLUMNS)].to_csv(path, index=False)
    return path
