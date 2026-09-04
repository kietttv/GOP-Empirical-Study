"""Group D stratified GOP behavior tables (phone / speaker / score stratum)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from gop_empirical.eval.metrics import apply_linear_map, correlation_metrics, error_metrics

# CMU 39-phone vowels used for D1 rollup (stress already stripped).
CMU_VOWELS = frozenset(
    {
        "AA",
        "AE",
        "AH",
        "AO",
        "AW",
        "AY",
        "EH",
        "ER",
        "EY",
        "IH",
        "IY",
        "OW",
        "OY",
        "UH",
        "UW",
    }
)

STRATUM_LABELS = ("Low", "Mid", "High")
GROUP_D_PREDICTION_COLUMNS = [
    "utt_id",
    "split",
    "word_id",
    "phone_id",
    "phone",
    "gop",
    "human_score",
    "speaker",
    "age",
    "sentence_accuracy",
    "score_stratum",
]


def phone_class(phone: str) -> str:
    return "vowel" if str(phone) in CMU_VOWELS else "consonant"


def _finite_std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def subset_metrics(
    gop: np.ndarray,
    human: np.ndarray,
    mapping: dict[str, float],
    *,
    min_n: int,
    clip: tuple[float, float] | None = (0.0, 2.0),
) -> dict[str, Any]:
    """PCC/SCC on raw GOP; MAE/MSE from a **frozen** train mapping.

    ``pcc`` / ``scc`` are None when ``n < min_n`` or either series has zero variance.
    MAE/MSE are still computed whenever ``n >= 1``.
    """
    gop = np.asarray(gop, dtype=np.float64).reshape(-1)
    human = np.asarray(human, dtype=np.float64).reshape(-1)
    if gop.size != human.size:
        raise ValueError(f"length mismatch: gop={gop.size} human={human.size}")
    n = int(gop.size)
    out: dict[str, Any] = {
        "n": n,
        "mean_gop": float(np.mean(gop)) if n else None,
        "std_gop": _finite_std(gop) if n else None,
        "mean_human": float(np.mean(human)) if n else None,
        "pcc": None,
        "scc": None,
        "mae": None,
        "mse": None,
        "reported": bool(n >= min_n),
    }
    if n >= 1:
        pred = apply_linear_map(gop, mapping, clip=clip)
        out.update(error_metrics(pred, human))
    if n >= min_n and n >= 2 and _finite_std(gop) > 0.0 and _finite_std(human) > 0.0:
        corr = correlation_metrics(gop, human)
        out["pcc"] = corr["pcc"]
        out["scc"] = corr["scc"]
    return out


def group_metrics_table(
    df: pd.DataFrame,
    group_col: str,
    mapping: dict[str, float],
    *,
    min_n: int,
    clip: tuple[float, float] | None = (0.0, 2.0),
    extra_fields: Sequence[str] = (),
) -> pd.DataFrame:
    if group_col not in df.columns:
        raise KeyError(group_col)
    rows: list[dict[str, Any]] = []
    for key, sub in df.groupby(group_col, sort=True, dropna=False):
        row = {group_col: key}
        row.update(
            subset_metrics(
                sub["gop"].to_numpy(),
                sub["human_score"].to_numpy(),
                mapping,
                min_n=min_n,
                clip=clip,
            )
        )
        for field in extra_fields:
            if field in sub.columns:
                row[field] = sub[field].iloc[0]
        if "utt_id" in sub.columns:
            row["n_utt"] = int(sub["utt_id"].nunique())
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def phone_class_metrics(
    df: pd.DataFrame,
    mapping: dict[str, float],
    *,
    min_n: int,
    clip: tuple[float, float] | None = (0.0, 2.0),
) -> pd.DataFrame:
    work = df.copy()
    work["phone_class"] = work["phone"].map(phone_class)
    table = group_metrics_table(
        work, "phone_class", mapping, min_n=min_n, clip=clip
    )
    table = table.rename(columns={"phone_class": "phone"})
    table["aggregate"] = True
    table["phone"] = table["phone"].str.upper()
    return table


def speaker_mean_sentence_accuracy(df: pd.DataFrame) -> pd.Series:
    """One sentence-accuracy value per utterance, then mean over utterances/speaker."""
    if df.empty:
        return pd.Series(dtype=np.float64)
    utt = df.drop_duplicates(["speaker", "utt_id"])
    return utt.groupby("speaker", sort=True)["sentence_accuracy"].mean()


def tertile_cutpoints(values: pd.Series) -> tuple[float, float]:
    if values.empty:
        raise ValueError("cannot compute tertiles on an empty series")
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    q33, q66 = (float(v) for v in np.quantile(arr, [1.0 / 3.0, 2.0 / 3.0]))
    return q33, q66


def assign_score_stratum(mean_accuracy: float | np.floating, q33: float, q66: float) -> str:
    value = float(mean_accuracy)
    if value <= q33:
        return "Low"
    if value <= q66:
        return "Mid"
    return "High"


def attach_score_strata(df: pd.DataFrame, *, split: str = "test") -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign Low/Mid/High on ``split`` speakers; other rows keep empty stratum."""
    out = df.copy()
    if "score_stratum" not in out.columns:
        out["score_stratum"] = ""
    else:
        out["score_stratum"] = out["score_stratum"].fillna("").astype(str)
    target = out[out["split"] == split]
    means = speaker_mean_sentence_accuracy(target)
    q33, q66 = tertile_cutpoints(means)
    speaker_to_stratum = {
        str(spk): assign_score_stratum(acc, q33, q66) for spk, acc in means.items()
    }
    mask = out["split"] == split
    out.loc[mask, "score_stratum"] = out.loc[mask, "speaker"].map(speaker_to_stratum).fillna("")
    counts = {label: int(sum(1 for v in speaker_to_stratum.values() if v == label)) for label in STRATUM_LABELS}
    meta = {
        "q33": q33,
        "q66": q66,
        "n_speakers": int(len(means)),
        "n_speakers_per_stratum": counts,
    }
    return out, meta


def load_a1_predictions(path: str | Path) -> pd.DataFrame:
    """Read A1 CSV with utterance ids kept as 9-digit strings."""
    csv_path = Path(path)
    df = pd.read_csv(csv_path, dtype={"utt_id": str, "phone": str, "split": str})
    required = {"utt_id", "split", "word_id", "phone_id", "phone", "gop", "human_score"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}")
    df["utt_id"] = df["utt_id"].map(lambda x: str(x).zfill(9))
    return df


def enrich_predictions(
    df: pd.DataFrame,
    *,
    utt2spk: dict[str, str],
    spk2age: dict[str, int],
    utterance_scores: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Attach speaker, age, and sentence accuracy. Stratum is filled later."""
    out = df.copy()
    out["speaker"] = out["utt_id"].map(utt2spk)
    n_missing_spk = int(out["speaker"].isna().sum())
    if n_missing_spk:
        raise ValueError(f"{n_missing_spk} phones have no utt2spk mapping")
    out["age"] = out["speaker"].map(spk2age)
    acc = {utt: rec["accuracy"] for utt, rec in utterance_scores.items()}
    out["sentence_accuracy"] = out["utt_id"].map(acc)
    n_missing_acc = int(out["sentence_accuracy"].isna().sum())
    if n_missing_acc:
        raise ValueError(f"{n_missing_acc} phones have no sentence accuracy")
    out["score_stratum"] = ""
    return out


def jsonable_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in df.to_dict(orient="records"):
        row: dict[str, Any] = {}
        for key, value in raw.items():
            if value is None or (not isinstance(value, (str, bool)) and pd.isna(value)):
                row[key] = None
            elif isinstance(value, (np.bool_, bool)):
                row[key] = bool(value)
            elif isinstance(value, (np.floating, float)):
                number = float(value)
                row[key] = None if not np.isfinite(number) else number
            elif isinstance(value, (np.integer, int)):
                row[key] = int(value)
            else:
                row[key] = value
        records.append(row)
    return records


def speaker_pcc_summary(speaker_table: pd.DataFrame) -> dict[str, Any]:
    reported = speaker_table[speaker_table["reported"] & speaker_table["pcc"].notna()]
    pcc = reported["pcc"].to_numpy(dtype=np.float64) if not reported.empty else np.array([], dtype=np.float64)
    return {
        "n_speakers": int(len(speaker_table)),
        "n_speakers_reported": int(len(reported)),
        "pcc_mean": float(np.mean(pcc)) if pcc.size else None,
        "pcc_std": float(np.std(pcc, ddof=1)) if pcc.size >= 2 else None,
        "pcc_min": float(np.min(pcc)) if pcc.size else None,
        "pcc_max": float(np.max(pcc)) if pcc.size else None,
        "mae_mean": float(reported["mae"].mean()) if not reported.empty else None,
        "mean_gop_mean": float(reported["mean_gop"].mean()) if not reported.empty else None,
    }
