"""Write experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

PREDICTION_COLUMNS = [
    "utt_id",
    "split",
    "word_id",
    "phone_id",
    "phone",
    "gop",
    "human_score",
]

GROUP_B_PREDICTION_COLUMNS = [
    "utt_id",
    "split",
    "word_id",
    "phone_id",
    "phone",
    "human_score",
    "b1_gop",
    "b2_lpp",
    "b3_lpr",
    "b4_lpp_canonical",
    "b4_lpp_max_competitor",
    "b4_lpr",
    "b5_pred",
]

GROUP_C_PREDICTION_COLUMNS = [
    "utt_id",
    "split",
    "word_id",
    "phone_id",
    "phone",
    "human_score",
    "gop_c1",
    "gop_c2",
    "gop_c3",
    "gop_c4",
    "gop_c5",
    "gop_c6",
    "gop_c7",
    "gop_c8",
    "gop_c9",
    "gop_c10",
    "gop_c11",
    "n_frames_c2",
    "n_frames_c3",
    "n_frames_c4",
    "n_frames_c5",
    "n_frames_c6",
    "n_frames_c7",
    "n_frames_c8",
    "n_frames_c9",
    "n_frames_c10",
    "n_frames_c11",
]

GROUP_E_BASE_COLUMNS = [
    "utt_id",
    "split",
    "role",
    "word_id",
    "phone_id",
    "phone",
    "speaker",
    "human_score",
    "pred_e1",
    "pred_e2",
]

GROUP_E_B4_FEATURE_COLUMNS = [
    "feat_lpp_canonical",
    "feat_lpp_max_competitor",
    "feat_lpr",
]

GROUP_E_A1_FEATURE_COLUMNS = [
    "feat_gop",
]

GROUP_E_C8_FEATURE_COLUMNS = [
    "feat_gop_c8",
]

GROUP_E_C9_FEATURE_COLUMNS = [
    "feat_gop_c9",
]

GROUP_E_C10_FEATURE_COLUMNS = [
    "feat_gop_c10",
]

GROUP_E_C11_FEATURE_COLUMNS = [
    "feat_gop_c11",
]


def write_predictions(
    df: pd.DataFrame,
    path: str | Path,
    columns: Sequence[str] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(columns) if columns is not None else PREDICTION_COLUMNS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"predictions missing columns: {missing}")
    out = df.loc[:, cols].copy()
    out.to_csv(path, index=False)
    return path


def write_results(payload: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path
