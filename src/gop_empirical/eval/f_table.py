"""Build the locked phone-level score table for Group F."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gop_empirical.eval.metrics import apply_linear_map, apply_linear_map_multi

JOIN_KEYS = ["utt_id", "word_id", "phone_id"]


def _read_pred_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={"utt_id": str, "phone": str, "split": str},
    )
    if "role" in df.columns:
        df["role"] = df["role"].astype(str)
    for col in ("word_id", "phone_id"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def _require(path: Path, hint: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; {hint}")
    return path


def _mapping_from_c(payload: dict[str, Any], key: str) -> dict[str, float]:
    block = payload[key]
    mapping = block["mapping"]
    return {"slope": float(mapping["slope"]), "intercept": float(mapping["intercept"])}


def build_group_f_score_table(
    paths: dict[str, Path],
    *,
    clip: tuple[float, float] = (0.0, 2.0),
) -> pd.DataFrame:
    """Join C/B/E test phones into one table with score columns for F models."""
    c_path = _require(paths["c_predictions"], "run Group C first")
    b_path = _require(paths["b_predictions"], "run Group B first")
    b_res = _require(paths["b_results"], "run Group B first")
    c_res = _require(paths["c_results"], "run Group C first")
    e_path = _require(paths["e_predictions"], "run Group E (b4) first")
    e_b5 = _require(paths["e_b5_predictions"], "run Group E --features b5 / b5_embed")
    e_c8 = _require(
        paths["e_c8_lpp_lpr_predictions"],
        "run Group E --features c8_lpp_lpr / c8_lpp_lpr_embed",
    )
    e_c9 = _require(
        paths["e_c9_lpp_lpr_predictions"],
        "run Group E --features c9_lpp_lpr / c9_lpp_lpr_embed",
    )

    c = _read_pred_csv(c_path)
    c = c[c["split"] == "test"].copy()
    b = _read_pred_csv(b_path)
    b = b[b["split"] == "test"].copy()
    e = _read_pred_csv(e_path)
    e = e[e["role"] == "test"].copy() if "role" in e.columns else e[e["split"] == "test"].copy()
    eb5 = _read_pred_csv(e_b5)
    eb5 = eb5[eb5["role"] == "test"].copy() if "role" in eb5.columns else eb5[eb5["split"] == "test"].copy()
    ec8 = _read_pred_csv(e_c8)
    ec8 = ec8[ec8["role"] == "test"].copy() if "role" in ec8.columns else ec8[ec8["split"] == "test"].copy()
    ec9 = _read_pred_csv(e_c9)
    ec9 = ec9[ec9["role"] == "test"].copy() if "role" in ec9.columns else ec9[ec9["split"] == "test"].copy()

    c_payload = json.loads(c_res.read_text(encoding="utf-8"))
    b_payload = json.loads(b_res.read_text(encoding="utf-8"))

    base = c[JOIN_KEYS + ["phone", "human_score", "gop_c1", "gop_c8", "gop_c9", "n_frames_c8"]].copy()
    for model, gop_col in (("C1", "gop_c1"), ("C8", "gop_c8"), ("C9", "gop_c9")):
        mapping = _mapping_from_c(c_payload, model)
        base[f"score_{model}"] = apply_linear_map(base[gop_col].to_numpy(), mapping, clip=clip)

    b_cols = JOIN_KEYS + ["b4_lpp_canonical", "b4_lpp_max_competitor"]
    base = base.merge(b[b_cols], on=JOIN_KEYS, how="inner", validate="one_to_one")
    b4_map = b_payload["B4"]["mapping"]
    x = base[["b4_lpp_canonical", "b4_lpp_max_competitor"]].to_numpy(dtype=np.float64)
    base["score_B4_OLS"] = apply_linear_map_multi(x, b4_map, clip=clip)
    base["competitor_wins"] = base["b4_lpp_max_competitor"] > base["b4_lpp_canonical"]

    e_cols = JOIN_KEYS + (["speaker"] if "speaker" in e.columns else []) + ["pred_e2"]
    base = base.merge(e[e_cols], on=JOIN_KEYS, how="inner", validate="one_to_one")
    base["score_E2"] = np.clip(base["pred_e2"].to_numpy(dtype=np.float64), clip[0], clip[1])

    for src, cols, rename in (
        (eb5, ["pred_e7", "pred_e8", "pred_e14"], {"pred_e7": "score_E7", "pred_e8": "score_E8", "pred_e14": "score_E14"}),
        (ec8, ["pred_e10", "pred_e16"], {"pred_e10": "score_E10", "pred_e16": "score_E16"}),
        (ec9, ["pred_e12", "pred_e18"], {"pred_e12": "score_E12", "pred_e18": "score_E18"}),
    ):
        piece = src[JOIN_KEYS + cols].rename(columns=rename)
        for new_col in rename.values():
            piece[new_col] = np.clip(piece[new_col].to_numpy(dtype=np.float64), clip[0], clip[1])
        base = base.merge(piece, on=JOIN_KEYS, how="inner", validate="one_to_one")

    if "speaker" not in base.columns:
        base["speaker"] = ""
    base = base.sort_values(JOIN_KEYS, kind="mergesort").reset_index(drop=True)
    return base


SCORE_COLUMN = {
    "C1": "score_C1",
    "C8": "score_C8",
    "C9": "score_C9",
    "B4_OLS": "score_B4_OLS",
    "E2": "score_E2",
    "E7": "score_E7",
    "E8": "score_E8",
    "E10": "score_E10",
    "E12": "score_E12",
    "E14": "score_E14",
    "E16": "score_E16",
    "E18": "score_E18",
}
