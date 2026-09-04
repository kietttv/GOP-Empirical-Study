"""Group F2 residual tables and heuristic error taxonomy."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

ERROR_TYPES = ("T1_alignment", "T2_confusion", "T3_accent", "T4_context", "T5_speaker")


def _finite(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)


def attach_residuals(
    df: pd.DataFrame,
    *,
    pred_col: str,
    human_col: str = "human_score",
) -> pd.DataFrame:
    out = df.copy()
    pred = _finite(out[pred_col])
    human = _finite(out[human_col])
    out["pred"] = pred
    out["residual"] = pred - human
    out["abs_err"] = np.abs(out["residual"])
    return out


def assign_primary_type(flags: dict[str, bool]) -> str:
    """Priority: confusion > accent > context > speaker > alignment > other."""
    order = (
        "T2_confusion",
        "T3_accent",
        "T4_context",
        "T5_speaker",
        "T1_alignment",
    )
    for name in order:
        if flags.get(name):
            return name
    return "other"


def classify_errors(
    df: pd.DataFrame,
    *,
    pred_col: str,
    human_col: str = "human_score",
    competitor_wins_col: str | None = "competitor_wins",
    n_frames_col: str | None = "n_frames_c8",
    abs_err_self_col: str | None = None,
    abs_err_other_col: str | None = None,
    # Backward-compatible aliases (C8 vs E16 fixed direction — prefer self/other).
    abs_err_c8_col: str | None = "abs_err_c8",
    abs_err_e16_col: str | None = "abs_err_e16",
    extreme_speaker_col: str | None = "extreme_speaker",
    context_err_gap: float = 0.5,
    accent_pred_min: float = 1.5,
    alignment_frame_lo: float = 2.0,
    alignment_frame_hi: float = 80.0,
) -> pd.DataFrame:
    """Add boolean type columns + primary_type for one model's residuals.

    T4 (context): ``|err_self| − |err_other| ≥ gap`` — the *other* scorer is
    clearly better on this phone. For C8, other=E16 (sequence helps); for E16,
    other=C8 (sequence hurts relative to direct).
    """
    out = attach_residuals(df, pred_col=pred_col, human_col=human_col)
    human = _finite(out[human_col])
    pred = out["pred"].to_numpy(dtype=np.float64)
    abs_err = out["abs_err"].to_numpy(dtype=np.float64)

    any_incorrect = out["any_incorrect"].fillna(False).astype(bool).to_numpy()
    any_accent = out["any_accent"].fillna(False).astype(bool).to_numpy()
    any_insertion = out["any_insertion"].fillna(False).astype(bool).to_numpy()

    competitor_wins = np.zeros(len(out), dtype=bool)
    if competitor_wins_col and competitor_wins_col in out.columns:
        competitor_wins = out[competitor_wins_col].fillna(False).astype(bool).to_numpy()

    extreme_speaker = np.zeros(len(out), dtype=bool)
    if extreme_speaker_col and extreme_speaker_col in out.columns:
        extreme_speaker = out[extreme_speaker_col].fillna(False).astype(bool).to_numpy()

    t2 = (human <= 1.0 + 1e-9) & (competitor_wins | any_incorrect | any_insertion)
    t3 = any_accent | ((np.abs(human - 1.0) <= 0.35) & (pred >= float(accent_pred_min)))

    if abs_err_self_col is None and abs_err_other_col is None:
        # Legacy: treat as C8-self / E16-other if both columns exist.
        abs_err_self_col = abs_err_c8_col
        abs_err_other_col = abs_err_e16_col

    t4 = np.zeros(len(out), dtype=bool)
    if (
        abs_err_self_col
        and abs_err_other_col
        and abs_err_self_col in out.columns
        and abs_err_other_col in out.columns
    ):
        err_self = _finite(out[abs_err_self_col])
        err_other = _finite(out[abs_err_other_col])
        t4 = (err_self - err_other) >= float(context_err_gap)

    t1 = np.zeros(len(out), dtype=bool)
    if n_frames_col and n_frames_col in out.columns:
        nf = _finite(out[n_frames_col])
        unusual = (~np.isfinite(nf)) | (nf <= float(alignment_frame_lo)) | (nf >= float(alignment_frame_hi))
        t1 = unusual & (abs_err >= 0.75)

    t5 = extreme_speaker & (abs_err >= 0.5)

    out["T1_alignment"] = t1
    out["T2_confusion"] = t2
    out["T3_accent"] = t3
    out["T4_context"] = t4
    out["T5_speaker"] = t5
    out["primary_type"] = [
        assign_primary_type(
            {
                "T1_alignment": bool(t1[i]),
                "T2_confusion": bool(t2[i]),
                "T3_accent": bool(t3[i]),
                "T4_context": bool(t4[i]),
                "T5_speaker": bool(t5[i]),
            }
        )
        for i in range(len(out))
    ]
    return out


def top_errors(df: pd.DataFrame, *, top_k: int = 100) -> pd.DataFrame:
    if "abs_err" not in df.columns:
        raise KeyError("abs_err missing; call classify_errors / attach_residuals first")
    return df.sort_values("abs_err", ascending=False, kind="mergesort").head(int(top_k)).reset_index(drop=True)


def type_counts(df: pd.DataFrame) -> dict[str, Any]:
    primary = df["primary_type"].value_counts(dropna=False).to_dict() if "primary_type" in df.columns else {}
    flags = {name: int(df[name].sum()) for name in ERROR_TYPES if name in df.columns}
    return {
        "n": int(len(df)),
        "primary": {str(k): int(v) for k, v in primary.items()},
        "flag_counts": flags,
        "mean_abs_err": float(df["abs_err"].mean()) if len(df) else None,
        "median_abs_err": float(df["abs_err"].median()) if len(df) else None,
    }


def stratified_error_summary(df: pd.DataFrame, human_col: str = "human_score") -> dict[str, Any]:
    out: dict[str, Any] = {}
    human = pd.to_numeric(df[human_col], errors="coerce")
    for label, lo, hi in (("human_0", -0.1, 0.5), ("human_1", 0.5, 1.5), ("human_2", 1.5, 2.1)):
        mask = (human > lo) & (human <= hi)
        sub = df.loc[mask]
        out[label] = {
            "n": int(len(sub)),
            "mean_abs_err": float(sub["abs_err"].mean()) if len(sub) else None,
        }
    return out
