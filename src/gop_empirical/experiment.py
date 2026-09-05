"""Run GOP empirical experiments from a YAML config (Group A–F)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

from gop_empirical.data.kaldi import list_kaldi_splits, load_kaldi_gop_split
from gop_empirical.data.learned import (
    CANONICAL_PHONE_COL,
    DUMP_FEATURE_COLUMNS,
    FEATURE_ACOUSTIC_MODEL,
    FEATURE_GOP_TYPE,
    FEATURE_MISSING_HINT,
    FeatureScaler,
    architecture_for_experiment,
    assign_roles,
    attach_canonical_phone_index,
    attach_speakers,
    choose_val_speakers,
    experiment_ids_for_features,
    feature_stored_columns,
    group_e_artifact_path,
    load_group_e_feature_table,
    matrix_from_table,
    normalize_group_e_features,
    pack_utterances,
    phone_embed_spec,
    scoring_ids,
    scoring_pred_columns,
    uses_phone_embed,
)
from gop_empirical.data.scores import load_human_scores, load_utterance_scores
from gop_empirical.data.speakers import load_speaker_metadata
from gop_empirical.data.ssl_gop import load_ssl_gop_split, ssl_gop_dir_ready, ssl_gop_split_path
from gop_empirical.data.scores_detail import load_scores_detail, phone_markup_table
from gop_empirical.eval.behavior import (
    GROUP_D_PREDICTION_COLUMNS,
    STRATUM_LABELS,
    attach_score_strata,
    enrich_predictions,
    group_metrics_table,
    jsonable_records,
    load_a1_predictions,
    phone_class_metrics,
    speaker_pcc_summary,
)
from gop_empirical.eval.errors import (
    classify_errors,
    stratified_error_summary,
    top_errors,
    type_counts,
)
from gop_empirical.eval.f_table import SCORE_COLUMN, build_group_f_score_table
from gop_empirical.eval.io import (
    GROUP_B_PREDICTION_COLUMNS,
    GROUP_C_PREDICTION_COLUMNS,
    GROUP_E_BASE_COLUMNS,
    write_predictions,
    write_results,
)
from gop_empirical.eval.metrics import (
    apply_linear_map,
    apply_linear_map_multi,
    evaluate_predictions,
    evaluate_split,
    evaluate_split_vector,
    fit_linear_score_map,
    fit_linear_score_map_multi,
)
from gop_empirical.eval.stats import bootstrap_model_metrics, paired_delta_bootstrap
from gop_empirical.gop.representation import (
    B4_FEATURE_NAMES,
    B4_OLS_FEATURE_NAMES,
    B5_N_FEATURES,
    gop_feature_vector,
    gopt_gop_feature_84,
)
from gop_empirical.gop.traditional import traditional_gop_batch

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return cfg


def resolve_path(raw: str | Path, *, base: Path) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


def rel_path(path: str | Path, *, base: Path) -> str:
    """Store paths relative to project root (portable across machines)."""
    p = Path(path)
    try:
        return p.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def _clip_tuple(cfg: dict[str, Any]) -> tuple[float, float]:
    clip = cfg.get("score_clip", [0.0, 2.0])
    return float(clip[0]), float(clip[1])


def _load_join_inputs(
    cfg: dict[str, Any],
    *,
    package_root: Path,
) -> tuple[Path, dict[str, Any], set[str], int, int | None]:
    paths = cfg["paths"]
    kaldi_dir = resolve_path(paths["kaldi_dir"], base=package_root)
    scores_path = resolve_path(paths["scores_json"], base=package_root)
    floor = float(cfg.get("score_floor", 0.1))
    skip = {str(p) for p in cfg.get("skip_phones", [])}
    phone_index_base = int(cfg.get("phone_index_base", 0))
    expected_n = cfg.get("expected_n_phones")
    expected_n_phones = int(expected_n) if expected_n is not None else None
    human = load_human_scores(scores_path, floor=floor)
    splits = list_kaldi_splits(kaldi_dir)
    if "train" not in splits or "test" not in splits:
        raise FileNotFoundError(f"need train and test CSVs under {kaldi_dir}")
    return kaldi_dir, human, skip, phone_index_base, expected_n_phones


def build_predictions(
    cfg: dict[str, Any],
    *,
    package_root: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """A1: canonical LPP → one GOP scalar per phone, joined to human scores."""
    package_root = package_root or PACKAGE_ROOT
    kaldi_dir, human, skip, phone_index_base, expected_n_phones = _load_join_inputs(
        cfg, package_root=package_root
    )

    rows: list[dict[str, Any]] = []
    n_missing = 0
    n_skipped_sil = 0
    for split in ("train", "test"):
        feats, keys = load_kaldi_gop_split(kaldi_dir, split)
        gops = traditional_gop_batch(
            feats,
            phone_index_base=phone_index_base,
            expected_n_phones=expected_n_phones,
        )
        for key, gop in zip(keys, gops, strict=True):
            rec = human.get(str(key))
            if rec is None:
                n_missing += 1
                continue
            if rec["phone"] in skip:
                n_skipped_sil += 1
                continue
            rows.append(
                {
                    "utt_id": rec["utt_id"],
                    "split": split,
                    "word_id": rec["word_id"],
                    "phone_id": rec["phone_id"],
                    "phone": rec["phone"],
                    "gop": float(gop),
                    "human_score": float(rec["human_score"]),
                }
            )
    if not rows:
        raise RuntimeError("no phones joined; check Kaldi keys vs scores.json")
    df = pd.DataFrame(rows)
    stats = {
        "n_missing_human": int(n_missing),
        "n_skipped_silence": int(n_skipped_sil),
        "n_train": int((df["split"] == "train").sum()),
        "n_test": int((df["split"] == "test").sum()),
    }
    return df, stats


def build_group_b_predictions(
    cfg: dict[str, Any],
    *,
    package_root: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, int], np.ndarray]:
    """B1–B5 representations in one pass over the Kaldi CSVs, joined to human scores.

    Returns ``(df, stats, x_b5)`` where ``x_b5`` is ``[N, 84]`` aligned with ``df``
    (not written to CSV).
    """
    package_root = package_root or PACKAGE_ROOT
    kaldi_dir, human, skip, phone_index_base, expected_n_phones = _load_join_inputs(
        cfg, package_root=package_root
    )
    n_phones = int(expected_n_phones) if expected_n_phones is not None else 42

    rows: list[dict[str, Any]] = []
    b5_rows: list[np.ndarray] = []
    n_missing = 0
    n_skipped_sil = 0
    for split in ("train", "test"):
        feats, keys = load_kaldi_gop_split(kaldi_dir, split)
        vec = gop_feature_vector(
            feats,
            phone_index_base=phone_index_base,
            expected_n_phones=expected_n_phones,
        )
        x84 = gopt_gop_feature_84(feats, expected_n_phones=n_phones)
        for key, feat_row, feat84 in zip(keys, vec, x84, strict=True):
            rec = human.get(str(key))
            if rec is None:
                n_missing += 1
                continue
            if rec["phone"] in skip:
                n_skipped_sil += 1
                continue
            lpp_can, lpp_comp, lpr = (float(feat_row[0]), float(feat_row[1]), float(feat_row[2]))
            rows.append(
                {
                    "utt_id": rec["utt_id"],
                    "split": split,
                    "word_id": rec["word_id"],
                    "phone_id": rec["phone_id"],
                    "phone": rec["phone"],
                    "human_score": float(rec["human_score"]),
                    "b1_gop": lpp_can,
                    "b2_lpp": lpp_can,
                    "b3_lpr": lpr,
                    "b4_lpp_canonical": lpp_can,
                    "b4_lpp_max_competitor": lpp_comp,
                    "b4_lpr": lpr,
                }
            )
            b5_rows.append(np.asarray(feat84, dtype=np.float64))
    if not rows:
        raise RuntimeError("no phones joined; check Kaldi keys vs scores.json")
    df = pd.DataFrame(rows)
    x_b5 = np.vstack(b5_rows)
    if x_b5.shape != (len(df), B5_N_FEATURES):
        raise RuntimeError(
            f"B5 feature matrix shape {x_b5.shape} != ({len(df)}, {B5_N_FEATURES})"
        )
    stats = {
        "n_missing_human": int(n_missing),
        "n_skipped_silence": int(n_skipped_sil),
        "n_train": int((df["split"] == "train").sum()),
        "n_test": int((df["split"] == "test").sum()),
    }
    return df, stats, x_b5


def _scalar_metrics(
    train: pd.DataFrame,
    test: pd.DataFrame,
    column: str,
    clip: tuple[float, float],
) -> dict[str, Any]:
    mapping = fit_linear_score_map(train[column].to_numpy(), train["human_score"].to_numpy())
    train_m = evaluate_split(
        train[column].to_numpy(), train["human_score"].to_numpy(), mapping, clip=clip
    )
    test_m = evaluate_split(
        test[column].to_numpy(), test["human_score"].to_numpy(), mapping, clip=clip
    )
    return {
        "pcc": test_m["pcc"],
        "scc": test_m["scc"],
        "mae": test_m["mae"],
        "mse": test_m["mse"],
        "n": test_m["n"],
        "train": train_m,
        "mapping": mapping,
    }


def _metric_subset(block: dict[str, Any]) -> dict[str, Any]:
    return {
        "pcc": block["pcc"],
        "scc": block["scc"],
        "mae": block["mae"],
        "mse": block["mse"],
        "n": block["n"],
    }


def _print_metrics(tag: str, m: dict[str, Any]) -> None:
    """Log PCC/SCC/MAE/MSE (shared by Groups A–E)."""
    def _fmt(key: str, digits: int) -> str:
        val = m.get(key)
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            return "nan"
        return f"{float(val):.{digits}f}"

    n_val = m.get("n")
    n_str = str(int(n_val)) if n_val is not None else "?"
    print(
        f"{tag}  "
        f"PCC={_fmt('pcc', 4)}  SCC={_fmt('scc', 4)}  "
        f"MAE={_fmt('mae', 4)}  MSE={_fmt('mse', 4)}  "
        f"n={n_str}",
        flush=True,
    )


def run_group_a(
    cfg: dict[str, Any],
    *,
    package_root: Path | None = None,
) -> dict[str, Any]:
    """A1 compute GOP, A2 correlate / map on official test split."""
    package_root = package_root or PACKAGE_ROOT
    print("A1 joining Kaldi GOP + scores.json ...", flush=True)
    df, stats = build_predictions(cfg, package_root=package_root)
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    if train.empty or test.empty:
        raise RuntimeError("empty train or test after join")
    print(
        f"A1 done  n_train={stats['n_train']}  n_test={stats['n_test']}  "
        f"missing={stats['n_missing_human']}  skipped_sil={stats['n_skipped_silence']}",
        flush=True,
    )

    clip = _clip_tuple(cfg)
    mapping = fit_linear_score_map(train["gop"].to_numpy(), train["human_score"].to_numpy())
    train_m = evaluate_split(
        train["gop"].to_numpy(), train["human_score"].to_numpy(), mapping, clip=clip
    )
    test_m = evaluate_split(
        test["gop"].to_numpy(), test["human_score"].to_numpy(), mapping, clip=clip
    )
    if train_m["pcc"] < 0.05:
        raise RuntimeError(
            f"train PCC={train_m['pcc']:.4f} is too weak for canonical LPP; "
            "try phone_index_base: 1 in the config (Kaldi 1-based ids)"
        )
    print(
        f"A2 map  slope={mapping['slope']:.4f}  intercept={mapping['intercept']:.4f}",
        flush=True,
    )
    _print_metrics("A2 train", train_m)
    _print_metrics("A2 test ", test_m)

    out_dir = resolve_path(cfg["paths"]["output_dir"], base=package_root)
    pred_path = write_predictions(df, out_dir / "a1_predictions.csv")
    results = {
        "group": cfg.get("group", "A"),
        "experiments": cfg.get("experiments", ["A1", "A2"]),
        "A1": {
            "description": "Traditional GOP = canonical LPP scalar",
            "gop_type": cfg.get("gop_type", "standard"),
            "acoustic_model": cfg.get("acoustic_model"),
            "alignment": cfg.get("alignment"),
            "dataset": cfg.get("dataset"),
            "level": cfg.get("level", "phoneme"),
            "scoring": cfg.get("scoring", "direct"),
            "phone_index_base": int(cfg.get("phone_index_base", 0)),
            "n_train": stats["n_train"],
            "n_test": stats["n_test"],
            "n_missing_human": stats["n_missing_human"],
            "n_skipped_silence": stats["n_skipped_silence"],
            "predictions_path": rel_path(pred_path, base=package_root),
        },
        "A2": {
            "description": "GOP vs human phoneme score on official test split",
            "pcc": test_m["pcc"],
            "scc": test_m["scc"],
            "mae": test_m["mae"],
            "mse": test_m["mse"],
            "n": test_m["n"],
            "train": train_m,
            "mapping": mapping,
        },
    }
    results_path = write_results(results, out_dir / "a2_results.json")
    results["results_path"] = rel_path(results_path, base=package_root)
    results["predictions_path"] = rel_path(pred_path, base=package_root)
    return results


def run_group_b(
    cfg: dict[str, Any],
    *,
    package_root: Path | None = None,
) -> dict[str, Any]:
    """B1–B5 GOP representations vs human scores on the official test split."""
    package_root = package_root or PACKAGE_ROOT
    print("B joining Kaldi GOP representations + scores.json ...", flush=True)
    df, stats, x_b5 = build_group_b_predictions(cfg, package_root=package_root)
    train_mask = (df["split"] == "train").to_numpy()
    test_mask = (df["split"] == "test").to_numpy()
    train = df.loc[train_mask]
    test = df.loc[test_mask]
    if train.empty or test.empty:
        raise RuntimeError("empty train or test after join")
    print(
        f"B joined  n_train={stats['n_train']}  n_test={stats['n_test']}",
        flush=True,
    )

    clip = _clip_tuple(cfg)
    b1_equals_b2 = bool(np.allclose(df["b1_gop"].to_numpy(), df["b2_lpp"].to_numpy()))
    b1 = _scalar_metrics(train, test, "b1_gop", clip)
    if b1["train"]["pcc"] < 0.05:
        raise RuntimeError(
            f"train PCC={b1['train']['pcc']:.4f} is too weak for canonical LPP; "
            "try phone_index_base: 1 in the config (Kaldi 1-based ids)"
        )
    b2 = _scalar_metrics(train, test, "b2_lpp", clip)
    b3 = _scalar_metrics(train, test, "b3_lpr", clip)
    _print_metrics("B1 test", b1)
    _print_metrics("B2 test", b2)
    _print_metrics("B3 test", b3)

    x_cols = list(B4_OLS_FEATURE_NAMES)
    # DataFrame columns are prefixed with b4_
    x_train = train[["b4_lpp_canonical", "b4_lpp_max_competitor"]].to_numpy()
    x_test = test[["b4_lpp_canonical", "b4_lpp_max_competitor"]].to_numpy()
    y_train = train["human_score"].to_numpy()
    y_test = test["human_score"].to_numpy()
    b4_mapping = fit_linear_score_map_multi(x_train, y_train)
    b4_train = evaluate_split_vector(x_train, y_train, b4_mapping, clip=clip)
    b4_test = evaluate_split_vector(x_test, y_test, b4_mapping, clip=clip)
    b4 = {
        "pcc": b4_test["pcc"],
        "scc": b4_test["scc"],
        "mae": b4_test["mae"],
        "mse": b4_test["mse"],
        "n": b4_test["n"],
        "train": b4_train,
        "mapping": b4_mapping,
        "n_features": int(b4_mapping["n_features"]),
        "feature_names": list(x_cols),
        "stored_feature_names": list(B4_FEATURE_NAMES),
    }
    _print_metrics("B4 test", b4)

    x5_train = x_b5[train_mask]
    x5_test = x_b5[test_mask]
    b5_mapping = fit_linear_score_map_multi(x5_train, y_train)
    b5_train = evaluate_split_vector(x5_train, y_train, b5_mapping, clip=clip)
    b5_test = evaluate_split_vector(x5_test, y_test, b5_mapping, clip=clip)
    df = df.copy()
    df["b5_pred"] = apply_linear_map_multi(x_b5, b5_mapping, clip=clip)
    b5 = {
        "pcc": b5_test["pcc"],
        "scc": b5_test["scc"],
        "mae": b5_test["mae"],
        "mse": b5_test["mse"],
        "n": b5_test["n"],
        "train": b5_train,
        "mapping": b5_mapping,
        "n_features": int(b5_mapping["n_features"]),
        "feature_names": (
            [f"lpp_{i}" for i in range(42)] + [f"lpr_{i}" for i in range(42)]
        ),
    }
    _print_metrics("B5 test", b5)

    out_dir = resolve_path(cfg["paths"]["output_dir"], base=package_root)
    pred_path = write_predictions(df, out_dir / "b_predictions.csv", columns=GROUP_B_PREDICTION_COLUMNS)

    protocol = {
        "dataset": cfg.get("dataset"),
        "acoustic_model": cfg.get("acoustic_model"),
        "alignment": cfg.get("alignment"),
        "level": cfg.get("level", "phoneme"),
        "scoring": cfg.get("scoring", "direct"),
        "phone_index_base": int(cfg.get("phone_index_base", 0)),
        "n_train": stats["n_train"],
        "n_test": stats["n_test"],
        "n_missing_human": stats["n_missing_human"],
        "n_skipped_silence": stats["n_skipped_silence"],
        "b1_identical_to_b2": b1_equals_b2,
        "predictions_path": rel_path(pred_path, base=package_root),
    }
    results = {
        "group": cfg.get("group", "B"),
        "experiments": cfg.get("experiments", ["B1", "B2", "B3", "B4", "B5"]),
        "protocol": protocol,
        "B1": {
            "description": "Standard GOP = canonical LPP scalar (same as A1)",
            "gop_type": "standard",
            **b1,
        },
        "B2": {
            "description": "LPP = canonical log phone posterior",
            "gop_type": "lpp",
            "identical_to_B1": b1_equals_b2,
            **b2,
        },
        "B3": {
            "description": "LPR vs best competitor = LPP[p] - max_{q≠p} LPP[q]",
            "gop_type": "lpr",
            **b3,
        },
        "B4": {
            "description": (
                "GOP-only vector [LPP canonical, max competitor LPP, LPR]; "
                "OLS on the rank-2 pair [LPP canonical, max competitor]"
            ),
            "gop_type": "gop_feature_vector",
            **b4,
        },
        "B5": {
            "description": (
                "84-dimensional GOP feature [LPP_0..41, LPR_0..41] "
                "(GOPT paper naming); OLS train-only; not full GOPT model"
            ),
            "gop_type": "gopt_gop_feature_84",
            **b5,
        },
        "comparison": {
            "test": {
                "B1": _metric_subset(b1),
                "B2": _metric_subset(b2),
                "B3": _metric_subset(b3),
                "B4": _metric_subset(b4),
                "B5": _metric_subset(b5),
            }
        },
    }
    results_path = write_results(results, out_dir / "b_results.json")
    results["results_path"] = rel_path(results_path, base=package_root)
    results["predictions_path"] = rel_path(pred_path, base=package_root)
    return results


_GROUP_C_MODELS = ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11")
_SSL_MODEL_PATH_KEYS = {
    "C2": "wav2vec2_gop_dir",
    "C3": "hubert_gop_dir",
    "C4": "hubert_gop_max_dir",
    "C5": "hubert_gop_cao_dir",
    "C6": "wav2vec2_gop_max_dir",
    "C7": "wav2vec2_gop_cao_dir",
    "C8": "xlsr_espeak_gop_cao_dir",
    "C9": "lv60_espeak_gop_cao_dir",
    "C10": "xlsr_espeak_gop_cao_sd_dir",
    "C11": "lv60_espeak_gop_cao_sd_dir",
}
_SSL_GOP_COL = {
    "C2": "gop_c2",
    "C3": "gop_c3",
    "C4": "gop_c4",
    "C5": "gop_c5",
    "C6": "gop_c6",
    "C7": "gop_c7",
    "C8": "gop_c8",
    "C9": "gop_c9",
    "C10": "gop_c10",
    "C11": "gop_c11",
}
_SSL_NFRAMES_COL = {
    "C2": "n_frames_c2",
    "C3": "n_frames_c3",
    "C4": "n_frames_c4",
    "C5": "n_frames_c5",
    "C6": "n_frames_c6",
    "C7": "n_frames_c7",
    "C8": "n_frames_c8",
    "C9": "n_frames_c9",
    "C10": "n_frames_c10",
    "C11": "n_frames_c11",
}
_SSL_EXTRACT_HINT = {
    "C2": "wav2vec2",
    "C3": "hubert",
    "C4": "hubert --gop max",
    "C5": "hubert --gop cao_s",
    "C6": "wav2vec2 --gop max",
    "C7": "wav2vec2 --gop cao_s",
    "C8": "xlsr_espeak --gop cao_s",
    "C9": "lv60_espeak --gop cao_s",
    "C10": "xlsr_espeak --gop cao_sd",
    "C11": "lv60_espeak --gop cao_sd",
}


def normalize_group_c_models(models: list[str] | tuple[str, ...] | None) -> list[str]:
    if not models:
        return list(_GROUP_C_MODELS)
    out: list[str] = []
    for raw in models:
        m = str(raw).strip().upper()
        if m not in _GROUP_C_MODELS:
            raise ValueError(f"unknown Group C model {raw!r}; expected C1–C11")
        if m not in out:
            out.append(m)
    if "C1" not in out:
        out.insert(0, "C1")
    return out


def _ssl_dir_for_model(cfg: dict[str, Any], model: str, *, package_root: Path) -> Path:
    key = _SSL_MODEL_PATH_KEYS[model]
    paths = cfg.get("paths") or {}
    if key not in paths:
        raise FileNotFoundError(
            f"config paths.{key} is missing; needed for {model}. "
            "Run scripts/extract_ssl_gop.py or pass --models C1"
        )
    return resolve_path(paths[key], base=package_root)


def _require_ssl_extract(ssl_dir: Path, model: str) -> None:
    if ssl_gop_dir_ready(ssl_dir):
        return
    train_p = ssl_gop_split_path(ssl_dir, "train")
    test_p = ssl_gop_split_path(ssl_dir, "test")
    raise FileNotFoundError(
        f"{model} SSL GOP CSVs are missing under {ssl_dir} "
        f"(need {train_p.name} and {test_p.name}). "
        "Build them with: python scripts/extract_ssl_gop.py "
        "--config configs/c_acoustic_model.yaml --model "
        f"{_SSL_EXTRACT_HINT[model]} "
        "or run C1 only: --models C1"
    )


def build_group_c_predictions(
    cfg: dict[str, Any],
    *,
    package_root: Path | None = None,
    models: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """C1 Kaldi LPP plus optional C2–C11 SSL GOP CSVs, joined to human scores."""
    package_root = package_root or PACKAGE_ROOT
    requested = normalize_group_c_models(models or cfg.get("experiments"))
    kaldi_dir, human, skip, phone_index_base, expected_n_phones = _load_join_inputs(
        cfg, package_root=package_root
    )

    ssl_tables: dict[str, dict[str, dict[str, Any]]] = {}
    for model in requested:
        if model == "C1":
            continue
        ssl_dir = _ssl_dir_for_model(cfg, model, package_root=package_root)
        _require_ssl_extract(ssl_dir, model)
        ssl_tables[model] = {
            "train": load_ssl_gop_split(ssl_dir, "train"),
            "test": load_ssl_gop_split(ssl_dir, "test"),
        }

    rows: list[dict[str, Any]] = []
    n_missing = 0
    n_skipped_sil = 0
    n_empty_segment = 0
    n_missing_ssl = {m: 0 for m in ssl_tables}
    for split in ("train", "test"):
        feats, keys = load_kaldi_gop_split(kaldi_dir, split)
        gops = traditional_gop_batch(
            feats,
            phone_index_base=phone_index_base,
            expected_n_phones=expected_n_phones,
        )
        for key, gop in zip(keys, gops, strict=True):
            rec = human.get(str(key))
            if rec is None:
                n_missing += 1
                continue
            if rec["phone"] in skip:
                n_skipped_sil += 1
                continue
            row: dict[str, Any] = {
                "utt_id": rec["utt_id"],
                "split": split,
                "word_id": rec["word_id"],
                "phone_id": rec["phone_id"],
                "phone": rec["phone"],
                "human_score": float(rec["human_score"]),
                "gop_c1": float(gop),
            }
            for ssl_model, gop_col in _SSL_GOP_COL.items():
                row[gop_col] = np.nan
                row[_SSL_NFRAMES_COL[ssl_model]] = np.nan
            for model, tables in ssl_tables.items():
                ssl_rec = tables[split].get(str(key))
                gop_col = _SSL_GOP_COL[model]
                n_col = _SSL_NFRAMES_COL[model]
                if ssl_rec is None:
                    n_missing_ssl[model] += 1
                    continue
                n_frames = int(ssl_rec["n_frames"])
                row[n_col] = n_frames
                if n_frames <= 0 or not np.isfinite(ssl_rec["gop"]):
                    n_empty_segment += 1
                    continue
                row[gop_col] = float(ssl_rec["gop"])
            rows.append(row)
    if not rows:
        raise RuntimeError("no phones joined; check Kaldi keys vs scores.json")
    df = pd.DataFrame(rows)
    stats = {
        "n_missing_human": int(n_missing),
        "n_skipped_silence": int(n_skipped_sil),
        "n_empty_segment": int(n_empty_segment),
        "n_missing_ssl": {k: int(v) for k, v in n_missing_ssl.items()},
        "n_train": int((df["split"] == "train").sum()),
        "n_test": int((df["split"] == "test").sum()),
        "requested_models": requested,
    }
    return df, stats


def _paired_mask(df: pd.DataFrame, models: list[str]) -> pd.Series:
    mask = df["gop_c1"].notna()
    for model in models:
        if model == "C1":
            continue
        mask = mask & df[_SSL_GOP_COL[model]].notna()
    return mask


def _c1_matches_a2(test_pcc: float, package_root: Path, atol: float = 1e-6) -> bool | None:
    a2_path = package_root / "outputs" / "A" / "a2_results.json"
    if not a2_path.is_file():
        return None
    import json

    a2 = json.loads(a2_path.read_text(encoding="utf-8"))
    a2_pcc = float(a2["A2"]["pcc"])
    return bool(abs(test_pcc - a2_pcc) < atol)


def merge_group_c_results(
    existing: dict[str, Any] | None,
    new: dict[str, Any],
) -> dict[str, Any]:
    """Keep locked C1–C9 blocks when a later run adds C10/C11."""
    if not existing:
        return new
    old_experiments = [str(x) for x in existing.get("experiments") or []]
    new_experiments = [str(x) for x in new.get("experiments") or []]
    experiments = list(dict.fromkeys(old_experiments + new_experiments))

    old_comp = dict(existing.get("comparison") or {})
    new_comp = dict(new.get("comparison") or {})
    test = dict(old_comp.get("test") or {})
    test.update(new_comp.get("test") or {})
    paired = dict(old_comp.get("test_paired") or {})
    paired.update(new_comp.get("test_paired") or {})
    comparison = dict(old_comp)
    comparison["test"] = test
    comparison["test_paired"] = paired

    old_proto = dict(existing.get("protocol") or {})
    new_proto = dict(new.get("protocol") or {})
    gop_by = dict(old_proto.get("gop_type_by_model") or {})
    gop_by.update(new_proto.get("gop_type_by_model") or {})
    requested = list(
        dict.fromkeys(
            [str(x) for x in old_proto.get("requested_models") or []]
            + [str(x) for x in new_proto.get("requested_models") or []]
        )
    )
    protocol = dict(old_proto)
    protocol.update(new_proto)
    protocol["gop_type_by_model"] = gop_by
    protocol["requested_models"] = requested

    merged = dict(existing)
    merged["group"] = new.get("group", existing.get("group", "C"))
    merged["experiments"] = experiments
    merged["comparison"] = comparison
    merged["protocol"] = protocol
    for model in _GROUP_C_MODELS:
        if model in new:
            merged[model] = new[model]
    return merged


def _merge_c_prediction_columns(
    df: pd.DataFrame,
    existing_path: Path,
    requested: list[str],
) -> pd.DataFrame:
    """Copy locked SSL GOP columns from a previous c_predictions.csv."""
    if not existing_path.is_file():
        return df
    old = pd.read_csv(existing_path, dtype={"utt_id": str})
    keys = ["utt_id", "split", "word_id", "phone_id"]
    keep: list[str] = []
    for model, gop_col in _SSL_GOP_COL.items():
        if model in requested:
            continue
        n_col = _SSL_NFRAMES_COL[model]
        if gop_col in old.columns:
            keep.append(gop_col)
        if n_col in old.columns:
            keep.append(n_col)
    if not keep:
        return df
    missing_keys = [c for c in keys if c not in old.columns]
    if missing_keys:
        return df
    merged = df.merge(old.loc[:, keys + keep], on=keys, how="left", suffixes=("", "_locked"))
    for col in keep:
        locked = f"{col}_locked"
        if locked in merged.columns:
            merged[col] = merged[locked]
            merged = merged.drop(columns=[locked])
    return merged


def run_group_c(
    cfg: dict[str, Any],
    *,
    package_root: Path | None = None,
    models: list[str] | None = None,
) -> dict[str, Any]:
    """C1–C11 GOP vs human scores on the official test split."""
    package_root = package_root or PACKAGE_ROOT
    requested = normalize_group_c_models(models or cfg.get("experiments"))
    df, stats = build_group_c_predictions(cfg, package_root=package_root, models=requested)
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    if train.empty or test.empty:
        raise RuntimeError("empty train or test after join")

    clip = _clip_tuple(cfg)
    acoustic = cfg.get("acoustic_models") or {}
    descriptions = {
        "C1": "Kaldi Librispeech M13 canonical LPP (same extract as A1)",
        "C2": "Wav2Vec2 phoneme-CTC mean log P(canonical) on Kaldi segments",
        "C3": "HuBERT phoneme-CTC mean log P(canonical) on Kaldi segments",
        "C4": "HuBERT phoneme-CTC max log P(canonical) on Kaldi segments",
        "C5": "HuBERT Cao GOP-S on the CTM phone sequence (no Kaldi times)",
        "C6": "Wav2Vec2 phoneme-CTC max log P(canonical) on Kaldi segments",
        "C7": "Wav2Vec2 Cao GOP-S on the CTM phone sequence (no Kaldi times)",
        "C8": "XLSR-53 espeak CTC (off-the-shelf) Cao GOP-S / AF-S; CMU→IPA map; no fine-tune",
        "C9": "wav2vec2-large-lv60 espeak CTC (off-the-shelf) Cao GOP-S / AF-S; same IPA map as C8",
        "C10": "XLSR-53 espeak CTC (same AM as C8) Cao GOP-SD / AF-SD; deletion skip in denom",
        "C11": "lv60 espeak CTC (same AM as C9) Cao GOP-SD / AF-SD; same graph as C10",
    }
    gop_types = {
        "C1": "standard",
        "C2": "standard",
        "C3": "standard",
        "C4": "max_logp",
        "C5": "cao_gop_s",
        "C6": "max_logp",
        "C7": "cao_gop_s",
        "C8": "cao_gop_s",
        "C9": "cao_gop_s",
        "C10": "cao_gop_sd",
        "C11": "cao_gop_sd",
    }
    gop_cols = {
        "C1": "gop_c1",
        "C2": "gop_c2",
        "C3": "gop_c3",
        "C4": "gop_c4",
        "C5": "gop_c5",
        "C6": "gop_c6",
        "C7": "gop_c7",
        "C8": "gop_c8",
        "C9": "gop_c9",
        "C10": "gop_c10",
        "C11": "gop_c11",
    }

    per_model: dict[str, Any] = {}
    for model in requested:
        col = gop_cols[model]
        tr = train[train[col].notna()]
        te = test[test[col].notna()]
        if tr.empty or te.empty:
            raise RuntimeError(f"{model}: empty train or test after dropping missing GOP")
        block = _scalar_metrics(tr, te, col, clip)
        if model == "C1" and block["train"]["pcc"] < 0.05:
            raise RuntimeError(
                f"train PCC={block['train']['pcc']:.4f} is too weak for canonical LPP; "
                "try phone_index_base: 1 in the config (Kaldi 1-based ids)"
            )
        per_model[model] = {
            "description": descriptions[model],
            "gop_type": gop_types[model],
            "acoustic_model": acoustic.get(model),
            **block,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
        }
        _print_metrics(f"{model} test", block)

    ssl_requested = [m for m in requested if m != "C1"]
    paired_train = train[_paired_mask(train, requested)]
    paired_test = test[_paired_mask(test, requested)]
    n_test_c1 = int(test["gop_c1"].notna().sum())
    n_test_paired = int(len(paired_test))
    min_frac = float(cfg.get("min_paired_frac", 0.95))
    if ssl_requested:
        if n_test_c1 <= 0:
            raise RuntimeError("C1 test set is empty; cannot check paired coverage")
        if n_test_paired / n_test_c1 < min_frac:
            raise RuntimeError(
                f"paired test n={n_test_paired} is below {min_frac:.0%} of "
                f"C1 test n={n_test_c1}; SSL extract dropped too many phones"
            )

    paired_metrics: dict[str, Any] = {}
    if ssl_requested and not paired_train.empty and not paired_test.empty:
        for model in requested:
            col = gop_cols[model]
            paired_metrics[model] = _metric_subset(
                _scalar_metrics(paired_train, paired_test, col, clip)
            )

    c1_pcc = per_model["C1"]["pcc"]
    c1_matches = _c1_matches_a2(c1_pcc, package_root)

    out_dir = resolve_path(cfg["paths"]["output_dir"], base=package_root)
    pred_path = out_dir / "c_predictions.csv"
    df = _merge_c_prediction_columns(df, pred_path, requested)
    pred_path = write_predictions(df, pred_path, columns=GROUP_C_PREDICTION_COLUMNS)

    protocol = {
        "dataset": cfg.get("dataset"),
        "alignment": cfg.get("alignment"),
        "gop_type": cfg.get("gop_type", "standard"),
        "gop_type_by_model": {m: gop_types[m] for m in requested},
        "level": cfg.get("level", "phoneme"),
        "scoring": cfg.get("scoring", "direct"),
        "phone_index_base": int(cfg.get("phone_index_base", 0)),
        "requested_models": requested,
        "n_train": stats["n_train"],
        "n_test": stats["n_test"],
        "n_train_c1": per_model["C1"]["n_train"],
        "n_test_c1": per_model["C1"]["n_test"],
        "n_test_paired": n_test_paired,
        "n_missing_human": stats["n_missing_human"],
        "n_skipped_silence": stats["n_skipped_silence"],
        "n_empty_segment": stats["n_empty_segment"],
        "n_missing_ssl": stats["n_missing_ssl"],
        "c1_matches_a2": c1_matches,
        "predictions_path": rel_path(pred_path, base=package_root),
    }
    results: dict[str, Any] = {
        "group": cfg.get("group", "C"),
        "experiments": requested,
        "protocol": protocol,
        "comparison": {
            "test": {m: _metric_subset(per_model[m]) for m in requested},
            "test_paired": paired_metrics,
        },
    }
    for model in requested:
        results[model] = per_model[model]
    results_file = out_dir / "c_results.json"
    existing: dict[str, Any] | None = None
    if results_file.is_file():
        existing = json.loads(results_file.read_text(encoding="utf-8"))
    results = merge_group_c_results(existing, results)
    results_path = write_results(results, results_file)
    results["results_path"] = rel_path(results_path, base=package_root)
    results["predictions_path"] = rel_path(pred_path, base=package_root)
    return results


def _metrics_block(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "n": row.get("n"),
        "mean_gop": row.get("mean_gop"),
        "std_gop": row.get("std_gop"),
        "mean_human": row.get("mean_human"),
        "pcc": row.get("pcc"),
        "scc": row.get("scc"),
        "mae": row.get("mae"),
        "mse": row.get("mse"),
        "reported": bool(row.get("reported")),
    }


def run_group_d(
    cfg: dict[str, Any],
    *,
    package_root: Path | None = None,
) -> dict[str, Any]:
    """D1–D3: stratify locked A1 GOP by phone, speaker, and score stratum."""
    package_root = package_root or PACKAGE_ROOT
    paths = cfg["paths"]
    pred_src = resolve_path(paths["a1_predictions"], base=package_root)
    a2_src = resolve_path(paths["a2_results"], base=package_root)
    scores_path = resolve_path(paths["scores_json"], base=package_root)
    so_dir = resolve_path(paths["speechocean_dir"], base=package_root)
    if not pred_src.is_file():
        raise FileNotFoundError(
            f"missing {pred_src}; run Group A first: "
            "python scripts/run_experiment.py --config configs/a_traditional_gop.yaml"
        )
    if not a2_src.is_file():
        raise FileNotFoundError(f"missing {a2_src}; run Group A first")

    a2_payload = json.loads(a2_src.read_text(encoding="utf-8"))
    mapping = {
        "slope": float(a2_payload["A2"]["mapping"]["slope"]),
        "intercept": float(a2_payload["A2"]["mapping"]["intercept"]),
    }
    a2_pcc = float(a2_payload["A2"]["pcc"])
    a2_n = int(a2_payload["A2"]["n"])

    df = load_a1_predictions(pred_src)
    speaker_meta = load_speaker_metadata(so_dir)
    utterance_scores = load_utterance_scores(scores_path)
    df = enrich_predictions(
        df,
        utt2spk=speaker_meta["utt2spk"],
        spk2age=speaker_meta["spk2age"],
        utterance_scores=utterance_scores,
    )
    df, stratum_meta = attach_score_strata(df, split="test")

    train = df[df["split"] == "train"].copy()
    test = df[df["split"] == "test"].copy()
    if train.empty or test.empty:
        raise RuntimeError("empty train or test after joining Group D metadata")

    clip = _clip_tuple(cfg)
    min_n_phone = int(cfg.get("min_n_phone", 100))
    min_n_speaker = int(cfg.get("min_n_speaker", 50))
    min_n_stratum = int(cfg.get("min_n_stratum", 50))
    atol = float(cfg.get("a2_pcc_atol", 1e-6))

    sanity = evaluate_split(
        test["gop"].to_numpy(), test["human_score"].to_numpy(), mapping, clip=clip
    )
    if abs(float(sanity["pcc"]) - a2_pcc) >= atol:
        raise RuntimeError(
            f"Group D test PCC={sanity['pcc']:.6f} does not match A2 PCC={a2_pcc:.6f}"
        )
    if int(sanity["n"]) != a2_n:
        raise RuntimeError(f"Group D test n={sanity['n']} does not match A2 n={a2_n}")
    print("D sanity vs A2 ok", flush=True)
    _print_metrics("D pooled test", sanity)

    phone_table = group_metrics_table(test, "phone", mapping, min_n=min_n_phone, clip=clip)
    phone_table["aggregate"] = False
    class_table = phone_class_metrics(test, mapping, min_n=min_n_phone, clip=clip)
    d1_table = pd.concat([phone_table, class_table], ignore_index=True)
    for row in jsonable_records(class_table):
        _print_metrics(f"D1 {row['phone']}", row)

    speaker_table = group_metrics_table(
        test, "speaker", mapping, min_n=min_n_speaker, clip=clip, extra_fields=("age",)
    )
    speaker_table["age"] = pd.to_numeric(speaker_table.get("age"), errors="coerce")

    stratum_table = group_metrics_table(
        test, "score_stratum", mapping, min_n=min_n_stratum, clip=clip
    )
    n_spk = test.groupby("score_stratum")["speaker"].nunique()
    stratum_table["n_speakers"] = stratum_table["score_stratum"].map(n_spk).astype(int)
    stratum_table["score_stratum"] = pd.Categorical(
        stratum_table["score_stratum"], categories=list(STRATUM_LABELS), ordered=True
    )
    stratum_table = stratum_table.sort_values("score_stratum").reset_index(drop=True)

    overlap = speaker_meta["speaker_overlap"]
    d2_summary = speaker_pcc_summary(speaker_table)
    d2_summary["n_speakers_train"] = int(len(speaker_meta["speakers_train"]))
    d2_summary["n_speakers_test"] = int(len(speaker_meta["speakers_test"]))
    d2_summary["speaker_overlap"] = int(len(overlap))
    d2_summary["n_missing_age"] = int(speaker_table["age"].isna().sum())
    print(
        f"D2 speakers  n={d2_summary['n_speakers_reported']}  "
        f"pcc_mean={d2_summary['pcc_mean'] if d2_summary['pcc_mean'] is not None else float('nan'):.4f}  "
        f"pcc_std={d2_summary['pcc_std'] if d2_summary['pcc_std'] is not None else float('nan'):.4f}  "
        f"overlap={d2_summary['speaker_overlap']}",
        flush=True,
    )
    for row in jsonable_records(stratum_table):
        _print_metrics(f"D3 {row['score_stratum']}", row)
    reported_phones = phone_table[phone_table["reported"] & phone_table["pcc"].notna()].copy()
    top = reported_phones.sort_values("pcc", ascending=False).head(5)
    bottom = reported_phones.sort_values("pcc", ascending=True).head(5)

    out_dir = resolve_path(paths["output_dir"], base=package_root)
    pred_path = write_predictions(df, out_dir / "d_predictions.csv", columns=GROUP_D_PREDICTION_COLUMNS)
    d1_path = out_dir / "d1_phone_metrics.csv"
    d2_path = out_dir / "d2_speaker_metrics.csv"
    d3_path = out_dir / "d3_stratum_metrics.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    d1_table.to_csv(d1_path, index=False)
    speaker_table.to_csv(d2_path, index=False)
    stratum_table.to_csv(d3_path, index=False)

    class_records = {str(row["phone"]): _metrics_block(row) for row in jsonable_records(class_table)}
    stratum_records = {
        str(row["score_stratum"]): {
            **_metrics_block(row),
            "n_utt": row.get("n_utt"),
            "n_speakers": row.get("n_speakers"),
        }
        for row in jsonable_records(stratum_table)
    }
    protocol = {
        "dataset": cfg.get("dataset"),
        "acoustic_model": cfg.get("acoustic_model"),
        "alignment": cfg.get("alignment"),
        "gop_type": cfg.get("gop_type", "standard"),
        "level": cfg.get("level", "phoneme"),
        "scoring": cfg.get("scoring", "direct"),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_speakers_train": d2_summary["n_speakers_train"],
        "n_speakers_test": d2_summary["n_speakers_test"],
        "speaker_overlap": d2_summary["speaker_overlap"],
        "min_n_phone": min_n_phone,
        "min_n_speaker": min_n_speaker,
        "min_n_stratum": min_n_stratum,
        "mapping": mapping,
        "mapping_source": rel_path(a2_src, base=package_root),
        "gop_source": rel_path(pred_src, base=package_root),
        "sanity_matches_a2": True,
        "sanity_pcc": float(sanity["pcc"]),
        "sanity_scc": float(sanity["scc"]),
        "sanity_mae": float(sanity["mae"]),
        "sanity_mse": float(sanity["mse"]),
        "predictions_path": rel_path(pred_path, base=package_root),
    }
    results: dict[str, Any] = {
        "group": cfg.get("group", "D"),
        "experiments": cfg.get("experiments", ["D1", "D2", "D3"]),
        "protocol": protocol,
        "D1": {
            "description": "Phone-level GOP vs human on official test split",
            "min_n": min_n_phone,
            "n_phones": int(len(phone_table)),
            "n_reported": int(phone_table["reported"].sum()),
            "phone_class": class_records,
            "top_pcc": jsonable_records(top[["phone", "n", "pcc", "scc", "mean_gop", "mean_human"]]),
            "bottom_pcc": jsonable_records(
                bottom[["phone", "n", "pcc", "scc", "mean_gop", "mean_human"]]
            ),
            "metrics_path": rel_path(d1_path, base=package_root),
        },
        "D2": {
            "description": "Speaker-level GOP vs human on official test split",
            "min_n": min_n_speaker,
            **d2_summary,
            "metrics_path": rel_path(d2_path, base=package_root),
        },
        "D3": {
            "description": "Speaker-mean sentence-accuracy tertiles (not proficiency)",
            "min_n": min_n_stratum,
            "q33": stratum_meta["q33"],
            "q66": stratum_meta["q66"],
            "n_speakers_per_stratum": stratum_meta["n_speakers_per_stratum"],
            "strata": stratum_records,
            "metrics_path": rel_path(d3_path, base=package_root),
        },
    }
    results_path = write_results(results, out_dir / "d_results.json")
    results["results_path"] = rel_path(results_path, base=package_root)
    results["predictions_path"] = rel_path(pred_path, base=package_root)
    return results


def _group_e_prediction_columns(feature_set: str, df: pd.DataFrame | None = None) -> list[str]:
    stored = list(feature_stored_columns(feature_set))
    pred_mlp, pred_tf = scoring_pred_columns(feature_set)
    cols = [name for name in GROUP_E_BASE_COLUMNS if not name.startswith("pred_")]
    if feature_set in DUMP_FEATURE_COLUMNS:
        cols.extend(stored)
    if feature_set in _LOCKED_PRED_COLUMNS:
        for locked in _LOCKED_PRED_COLUMNS[feature_set]:
            if df is not None and locked in df.columns:
                cols.append(locked)
        cols.extend([pred_mlp, pred_tf])
        return cols
    cols.extend([pred_mlp, pred_tf])
    return cols


_MLP_DESCRIPTION = {
    "E1": "MLP Linear-ReLU-Linear-ReLU-Linear on GOP features",
    "E3": "MLP Linear-ReLU-Linear-ReLU-Linear on C8 Cao GOP-S",
    "E5": "MLP Linear-ReLU-Linear-ReLU-Linear on C9 Cao GOP-S",
    "E19": "Poly2 per-phoneme regression on C10 Cao GOP-SD (paper-like mode)",
    "E21": "Poly2 per-phoneme regression on C11 Cao GOP-SD (paper-like mode)",
    "E7": "MLP Linear-ReLU-Linear-ReLU-Linear on Kaldi 84-d LPP+LPR",
    "E9": "MLP Linear-ReLU-Linear-ReLU-Linear on C8 78-d LPP+LPR",
    "E11": "MLP Linear-ReLU-Linear-ReLU-Linear on C9 78-d LPP+LPR",
    "E13": "MLP on Kaldi 84-d LPP+LPR plus canonical phone embedding",
    "E15": "MLP on C8 78-d LPP+LPR plus canonical phone embedding",
    "E17": "MLP on C9 78-d LPP+LPR plus canonical phone embedding",
}
_TF_DESCRIPTION = {
    "E2": "Transformer encoder, per-phone head, same GOP features as E1",
    "E4": "Transformer encoder, per-phone head, same GOP-S as E3",
    "E6": "Transformer encoder, per-phone head, same GOP-S as E5",
    "E20": "SVR per-phoneme on C10 Cao GOP-SD (paper-like mode)",
    "E22": "SVR per-phoneme on C11 Cao GOP-SD (paper-like mode)",
    "E8": "Transformer encoder, per-phone head, same 84-d LPP+LPR as E7",
    "E10": "Transformer encoder, per-phone head, same 78-d LPP+LPR as E9",
    "E12": "Transformer encoder, per-phone head, same 78-d LPP+LPR as E11",
    "E14": "Transformer encoder, same 84-d as E13, plus canonical phone embedding",
    "E16": "Transformer encoder, same C8 78-d as E15, plus canonical phone embedding",
    "E18": "Transformer encoder, same C9 78-d as E17, plus canonical phone embedding",
}

_KALDI_PRIMARY_FEATURE_SETS = frozenset({"b4", "a1"})
_FOLLOWUP_FEATURE_SETS = frozenset(
    {
        "c8",
        "c9",
        "c10",
        "c11",
        "b5",
        "b5_embed",
        "c8_lpp_lpr",
        "c9_lpp_lpr",
        "c8_lpp_lpr_embed",
        "c9_lpp_lpr_embed",
    }
)
_SSL_FEATURE_SETS = frozenset(
    {
        "c8",
        "c9",
        "c10",
        "c11",
        "c8_lpp_lpr",
        "c9_lpp_lpr",
        "c8_lpp_lpr_embed",
        "c9_lpp_lpr_embed",
    }
)
_LOCKED_PRED_COLUMNS = {
    "b5_embed": ("pred_e7", "pred_e8"),
    "c8_lpp_lpr_embed": ("pred_e9", "pred_e10"),
    "c9_lpp_lpr_embed": ("pred_e11", "pred_e12"),
}
_LOCKED_PRED_HINT = {
    "b5_embed": "run --features b5 first",
    "c8_lpp_lpr_embed": "run --features c8_lpp_lpr first",
    "c9_lpp_lpr_embed": "run --features c9_lpp_lpr first",
}


def _group_e_pred_filename(feature_set: str, requested: list[str]) -> str:
    if feature_set in _FOLLOWUP_FEATURE_SETS:
        slug = feature_set.replace("_", "_")
        if feature_set in ("b5", "b5_embed"):
            return "e_b5_predictions.csv"
        if feature_set in ("c8_lpp_lpr", "c8_lpp_lpr_embed"):
            return "e_c8_lpp_lpr_predictions.csv"
        if feature_set in ("c9_lpp_lpr", "c9_lpp_lpr_embed"):
            return "e_c9_lpp_lpr_predictions.csv"
        return f"e_{feature_set}_predictions.csv"
    if feature_set == "b4":
        return "e_predictions.csv" if requested == ["b4"] else "e_b4_predictions.csv"
    if feature_set == "a1":
        return "e_predictions.csv" if requested == ["a1"] else "e_a1_predictions.csv"
    raise ValueError(f"unknown feature set {feature_set!r}")


def _group_e_feature_path(feature_set: str, paths: dict[str, Any], package_root: Path) -> Path:
    return group_e_artifact_path(feature_set, paths, package_root)


def _group_e_checkpoint_dir(cfg: dict[str, Any], package_root: Path) -> Path:
    raw = (cfg.get("paths") or {}).get("checkpoint_dir", "checkpoints")
    return resolve_path(raw, base=package_root)


def _write_group_e_checkpoint(
    checkpoint_dir: Path,
    package_root: Path,
    *,
    experiment_id: str,
    architecture: str,
    model: Any,
    scaler: FeatureScaler,
    model_kwargs: dict[str, Any],
    feature_set: str,
    fit: dict[str, Any],
    protocol: dict[str, Any],
    metrics: dict[str, Any],
) -> str:
    from gop_empirical.scoring.checkpoint import default_checkpoint_path, save_checkpoint

    path = default_checkpoint_path(checkpoint_dir, experiment_id, architecture)
    save_checkpoint(
        path,
        experiment_id=experiment_id,
        architecture=architecture,
        model=model,
        scaler=scaler,
        model_kwargs=model_kwargs,
        feature_set=feature_set,
        fit=fit,
        protocol=protocol,
        metrics=metrics,
    )
    rel = rel_path(path, base=package_root)
    print(f"  saved {experiment_id} -> {rel}", flush=True)
    return rel


def _maybe_save_group_e_scorer(
    *,
    save_checkpoints: bool,
    checkpoint_dir: Path | None,
    package_root: Path,
    experiment_id: str,
    architecture: str,
    model: Any,
    scaler: FeatureScaler,
    model_kwargs: dict[str, Any],
    feature_set: str,
    fit: dict[str, Any],
    protocol: dict[str, Any],
    role_metrics: dict[str, Any],
) -> str | None:
    if not save_checkpoints or checkpoint_dir is None:
        return None
    return _write_group_e_checkpoint(
        checkpoint_dir,
        package_root,
        experiment_id=experiment_id,
        architecture=architecture,
        model=model,
        scaler=scaler,
        model_kwargs=model_kwargs,
        feature_set=feature_set,
        fit=fit,
        protocol=protocol,
        metrics={
            "test": _metric_subset(role_metrics["test"]),
            "val": _metric_subset(role_metrics["val"]),
            "train": _metric_subset(role_metrics["train"]),
        },
    )


def merge_group_e_results(
    existing: dict[str, Any] | None,
    new: dict[str, Any],
) -> dict[str, Any]:
    """Keep locked E1/E2 (and other feature-set blocks) when a later run adds c8/c9."""
    if not existing:
        return new
    old_experiments = [str(x) for x in existing.get("experiments") or []]
    new_experiments = [str(x) for x in new.get("experiments") or []]
    experiments = list(dict.fromkeys(old_experiments + new_experiments))

    comparison = dict(existing.get("comparison") or {})
    comparison.update(new.get("comparison") or {})

    old_proto = dict(existing.get("protocol") or {})
    new_proto = dict(new.get("protocol") or {})
    pred_paths = dict(old_proto.get("prediction_paths") or {})
    pred_paths.update(new_proto.get("prediction_paths") or {})
    am_by = dict(old_proto.get("acoustic_model_by_features") or {})
    am_by.update(new_proto.get("acoustic_model_by_features") or {})
    gop_by = dict(old_proto.get("gop_type_by_features") or {})
    gop_by.update(new_proto.get("gop_type_by_features") or {})

    requested = [str(x) for x in new_proto.get("features_requested") or []]
    followup_only = bool(requested) and all(name in _FOLLOWUP_FEATURE_SETS for name in requested)

    protocol = dict(old_proto)
    protocol.update(new_proto)
    protocol["prediction_paths"] = pred_paths
    protocol["acoustic_model_by_features"] = am_by
    protocol["gop_type_by_features"] = gop_by
    protocol["features_requested"] = requested
    if followup_only:
        if old_proto.get("acoustic_model"):
            protocol["acoustic_model"] = old_proto["acoustic_model"]
        if old_proto.get("gop_type"):
            protocol["gop_type"] = old_proto["gop_type"]
        if old_proto.get("predictions_path"):
            protocol["predictions_path"] = old_proto["predictions_path"]

    merged = dict(existing)
    merged["group"] = new.get("group", existing.get("group", "E"))
    merged["experiments"] = experiments
    merged["comparison"] = comparison
    merged["protocol"] = protocol
    for key in (
        "b4",
        "a1",
        "c8",
        "c9",
        "c10",
        "c11",
        "b5",
        "b5_embed",
        "c8_lpp_lpr",
        "c9_lpp_lpr",
        "c8_lpp_lpr_embed",
        "c9_lpp_lpr_embed",
    ):
        if key in new:
            merged[key] = new[key]
    for eid in (
        "E1",
        "E2",
        "E3",
        "E4",
        "E5",
        "E6",
        "E7",
        "E8",
        "E9",
        "E10",
        "E11",
        "E12",
        "E13",
        "E14",
        "E15",
        "E16",
        "E17",
        "E18",
        "E19",
        "E20",
        "E21",
        "E22",
    ):
        if eid in new:
            merged[eid] = new[eid]
    if followup_only and existing.get("feature_set") in ("b4", "a1"):
        merged["feature_set"] = existing["feature_set"]
    else:
        merged["feature_set"] = new.get("feature_set", existing.get("feature_set"))
    merged.pop("results_path", None)
    merged.pop("predictions_path", None)
    return merged


def _ols_official_split_baseline(
    table: pd.DataFrame,
    stored: Sequence[str],
    *,
    clip: tuple[float, float],
    n_test: int,
    name: str,
    description: str,
) -> dict[str, Any]:
    train = table[table["split"] == "train"]
    test = table[table["split"] == "test"]
    if train.empty or test.empty:
        raise RuntimeError("empty official train or test for OLS baseline")
    mapping = fit_linear_score_map_multi(
        matrix_from_table(train, stored), train["human_score"].to_numpy()
    )
    metrics = evaluate_split_vector(
        matrix_from_table(test, stored),
        test["human_score"].to_numpy(),
        mapping,
        clip=clip,
    )
    if int(metrics["n"]) != int(n_test):
        raise RuntimeError(
            f"{name} OLS n={metrics['n']} does not match Group E test n={n_test}"
        )
    return {
        "name": name,
        "description": description,
        "pcc": float(metrics["pcc"]),
        "scc": float(metrics["scc"]),
        "mae": float(metrics["mae"]),
        "mse": float(metrics["mse"]),
        "n": int(metrics["n"]),
        "source": "ols_official_train",
        "mapping": mapping,
    }


def _frozen_linear_baseline(
    feature_set: str,
    cfg: dict[str, Any],
    *,
    package_root: Path,
    n_test: int,
    table: pd.DataFrame | None = None,
    clip: tuple[float, float] | None = None,
) -> dict[str, Any]:
    paths = cfg["paths"]
    if feature_set == "a1":
        src = resolve_path(paths["a2_results"], base=package_root)
        if not src.is_file():
            raise FileNotFoundError(f"missing {src}; run Group A first")
        payload = json.loads(src.read_text(encoding="utf-8"))
        block = payload["A2"]
        name = "direct_gop_a2"
        description = "Frozen A2 Direct GOP (canonical LPP vs human on official test)"
    elif feature_set == "b4":
        src = resolve_path(paths["b_results"], base=package_root)
        if not src.is_file():
            raise FileNotFoundError(f"missing {src}; run Group B first")
        payload = json.loads(src.read_text(encoding="utf-8"))
        block = payload["B4"]
        name = "b4_ols"
        description = "Frozen B4 OLS on [LPP, max competitor] (official test)"
    elif feature_set in ("c8", "c9", "c10", "c11"):
        if "c_results" not in paths:
            raise FileNotFoundError("config paths.c_results is required for --features c8/c9/c10/c11")
        src = resolve_path(paths["c_results"], base=package_root)
        if not src.is_file():
            raise FileNotFoundError(
                f"missing {src}; run Group C first: "
                "python scripts/run_experiment.py --config configs/c_acoustic_model.yaml"
            )
        payload = json.loads(src.read_text(encoding="utf-8"))
        key = {"c8": "C8", "c9": "C9", "c10": "C10", "c11": "C11"}[feature_set]
        if key not in payload:
            raise FileNotFoundError(f"{src}: missing {key} block")
        block = payload[key]
        if feature_set == "c8":
            name = "direct_c8"
            description = (
                "Frozen C8 direct Cao GOP-S (XLSR-53 espeak vs human on official test)"
            )
        elif feature_set == "c9":
            name = "direct_c9"
            description = (
                "Frozen C9 direct Cao GOP-S (lv60 espeak vs human on official test)"
            )
        elif feature_set == "c10":
            name = "direct_c10"
            description = (
                "Frozen C10 direct Cao GOP-SD (XLSR-53 espeak vs human on official test)"
            )
        else:
            name = "direct_c11"
            description = (
                "Frozen C11 direct Cao GOP-SD (lv60 espeak vs human on official test)"
            )
    elif feature_set in ("b5", "b5_embed"):
        src = resolve_path(paths["b_results"], base=package_root)
        if not src.is_file():
            raise FileNotFoundError(f"missing {src}; run Group B first")
        payload = json.loads(src.read_text(encoding="utf-8"))
        if "B5" not in payload:
            raise FileNotFoundError(f"{src}: missing B5 block")
        block = payload["B5"]
        name = "b5_ols"
        description = "Frozen B5 OLS on 84-d LPP+LPR (official test)"
    elif feature_set in ("c8_lpp_lpr", "c9_lpp_lpr", "c8_lpp_lpr_embed", "c9_lpp_lpr_embed"):
        if table is None or clip is None:
            raise ValueError("78-d OLS baseline needs the feature table and clip")
        stored = list(feature_stored_columns(feature_set))
        label = "C8" if "c8" in feature_set else "C9"
        return _ols_official_split_baseline(
            table,
            stored,
            clip=clip,
            n_test=n_test,
            name=f"{feature_set}_ols",
            description=f"Train-only OLS on {label} 78-d LPP+LPR (official test)",
        )
    else:
        raise ValueError(f"unknown feature set {feature_set!r}")
    n_ref = int(block["n"])
    if n_ref != int(n_test):
        raise RuntimeError(
            f"{feature_set} baseline n={n_ref} does not match Group E test n={n_test}"
        )
    return {
        "name": name,
        "description": description,
        "pcc": float(block["pcc"]),
        "scc": float(block["scc"]),
        "mae": float(block["mae"]),
        "mse": float(block["mse"]),
        "n": n_ref,
        "source": rel_path(src, base=package_root),
    }


def _role_prediction_metrics(
    df: pd.DataFrame,
    column: str,
    clip: tuple[float, float],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for role in ("train", "val", "test"):
        sub = df[df["role"] == role]
        if sub.empty:
            raise RuntimeError(f"no phones with role={role!r} after Group E split")
        out[role] = evaluate_predictions(
            sub[column].to_numpy(), sub["human_score"].to_numpy(), clip=clip
        )
    return out


def _role_prediction_metrics_from_array(
    df: pd.DataFrame,
    pred: np.ndarray,
    clip: tuple[float, float],
) -> dict[str, Any]:
    work = df.loc[:, ["role", "human_score"]].copy()
    work["_pred"] = np.clip(np.asarray(pred, dtype=np.float64), clip[0], clip[1])
    return _role_prediction_metrics(work, "_pred", clip)


def _is_paper_like_c10_c11_mode(cfg: dict[str, Any], feature_set: str) -> bool:
    """Use Cao IS2024-style scorers for E19–E22 on scalar GOP-SD."""
    if feature_set not in ("c10", "c11"):
        return False
    # Default ON for E19–E22 after user request.
    return bool(cfg.get("paper_like_c10_c11", True))


def _train_predict_poly2_per_phone(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    feature_col: str,
    *,
    clip: tuple[float, float],
) -> np.ndarray:
    """Paper scalar scorer: polynomial regression (order 2) per phoneme."""
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    out = np.empty(len(pred_df), dtype=np.float64)
    out.fill(np.nan)
    poly = PolynomialFeatures(degree=2, include_bias=True)
    grouped_train = train_df.groupby("phone", sort=False)
    global_x = train_df[[feature_col]].to_numpy(dtype=np.float64)
    global_y = train_df["human_score"].to_numpy(dtype=np.float64)
    global_model = LinearRegression().fit(poly.fit_transform(global_x), global_y)
    for phone, idx in pred_df.groupby("phone", sort=False).groups.items():
        sub = grouped_train.get_group(phone) if phone in grouped_train.groups else None
        if sub is None or len(sub) < 3:
            x = pred_df.loc[idx, [feature_col]].to_numpy(dtype=np.float64)
            out[idx] = global_model.predict(poly.transform(x))
            continue
        x_tr = sub[[feature_col]].to_numpy(dtype=np.float64)
        y_tr = sub["human_score"].to_numpy(dtype=np.float64)
        model = LinearRegression().fit(poly.fit_transform(x_tr), y_tr)
        x_te = pred_df.loc[idx, [feature_col]].to_numpy(dtype=np.float64)
        out[idx] = model.predict(poly.transform(x_te))
    return np.clip(out, clip[0], clip[1])


def _train_predict_svr_per_phone(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    feature_col: str,
    *,
    clip: tuple[float, float],
) -> np.ndarray:
    """Paper vector scorer family: SVR per phoneme (applied to scalar GOP-SD)."""
    from sklearn.svm import SVR

    out = np.empty(len(pred_df), dtype=np.float64)
    out.fill(np.nan)
    grouped_train = train_df.groupby("phone", sort=False)
    global_x = train_df[[feature_col]].to_numpy(dtype=np.float64)
    global_y = train_df["human_score"].to_numpy(dtype=np.float64)
    global_model = SVR().fit(global_x, global_y)
    for phone, idx in pred_df.groupby("phone", sort=False).groups.items():
        sub = grouped_train.get_group(phone) if phone in grouped_train.groups else None
        if sub is None or len(sub) < 6:
            x = pred_df.loc[idx, [feature_col]].to_numpy(dtype=np.float64)
            out[idx] = global_model.predict(x)
            continue
        x_tr = sub[[feature_col]].to_numpy(dtype=np.float64)
        y_tr = sub["human_score"].to_numpy(dtype=np.float64)
        model = SVR().fit(x_tr, y_tr)
        x_te = pred_df.loc[idx, [feature_col]].to_numpy(dtype=np.float64)
        out[idx] = model.predict(x_te)
    return np.clip(out, clip[0], clip[1])


def _run_feature_set(
    df: pd.DataFrame,
    feature_set: str,
    cfg: dict[str, Any],
    *,
    seed: int,
    clip: tuple[float, float],
    device: Any,
    checkpoint_dir: Path | None = None,
    save_checkpoints: bool = False,
    package_root: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from gop_empirical.scoring.mlp import PhoneMLP
    from gop_empirical.scoring.train import (
        phone_loader,
        predict_mlp,
        predict_transformer,
        sequence_loader,
        train_regressor,
    )
    from gop_empirical.scoring.transformer import PhoneTransformer

    stored = list(feature_stored_columns(feature_set))
    cap = int(cfg.get("max_seq_len", 50))
    df = df.sort_values(["utt_id", "word_id", "phone_id"], kind="mergesort").reset_index(drop=True)
    n_phones = None
    pad_phone_id = None
    pack_phone_kw: dict[str, Any] = {}
    if uses_phone_embed(feature_set):
        spec = phone_embed_spec(feature_set)
        n_phones = int(spec["n_phones"])
        df = attach_canonical_phone_index(
            df, n_phones=n_phones, space=str(spec["space"])
        )
        pad_phone_id = n_phones
        pack_phone_kw = {
            "phone_idx_col": CANONICAL_PHONE_COL,
            "pad_phone_id": pad_phone_id,
        }
    observed = int(df.groupby("utt_id").size().max()) if len(df) else 1
    # Pad to at least the config cap; never drop phones (keeps n_test = A2/B4).
    max_seq_len = max(cap, observed)
    n_truncated = 0
    mlp_cfg = cfg.get("mlp") or {}
    tf_cfg = cfg.get("transformer") or {}
    train_cfg = cfg.get("train") or {}
    hidden_dim = int(mlp_cfg.get("hidden_dim", 32))
    batch_e1 = int(train_cfg.get("batch_size_e1", 256))
    batch_e2 = int(train_cfg.get("batch_size_e2", 32))
    lr = float(train_cfg.get("lr", 1e-3))
    max_epochs = int(train_cfg.get("max_epochs", 40))
    patience = int(train_cfg.get("patience", 8))

    train_df = df[df["role"] == "train"]
    val_df = df[df["role"] == "val"]
    if train_df.empty or val_df.empty:
        raise RuntimeError("empty train or val after speaker split / truncation")
    scaler = FeatureScaler.fit(matrix_from_table(train_df, stored))
    scaled = scaler.transform(matrix_from_table(df, stored))
    y_all = df["human_score"].to_numpy(dtype=np.float64)
    train_mask = (df["role"] == "train").to_numpy()
    val_mask = (df["role"] == "val").to_numpy()
    phone_idx = (
        None
        if n_phones is None
        else df[CANONICAL_PHONE_COL].to_numpy(dtype=np.int64)
    )

    if _is_paper_like_c10_c11_mode(cfg, feature_set):
        if len(stored) != 1:
            raise RuntimeError(
                f"paper_like_c10_c11 expects scalar feature, got {feature_set} with {len(stored)} dims"
            )
        feat_col = stored[0]
        work = df.copy()
        train_fit = work[work["role"] == "train"]
        if train_fit.empty:
            raise RuntimeError("empty train split for paper-like c10/c11 scorer")
        pred_poly = _train_predict_poly2_per_phone(train_fit, work, feat_col, clip=clip)
        pred_svr = _train_predict_svr_per_phone(train_fit, work, feat_col, clip=clip)
        out = work.copy()
        if CANONICAL_PHONE_COL in out.columns:
            out = out.drop(columns=[CANONICAL_PHONE_COL])
        mlp_id, tf_id = scoring_ids(feature_set)
        pred_mlp, pred_tf = scoring_pred_columns(feature_set)
        out[pred_mlp] = pred_poly
        out[pred_tf] = pred_svr
        e1_metrics = _role_prediction_metrics(out, pred_mlp, clip)
        e2_metrics = _role_prediction_metrics(out, pred_tf, clip)
        _print_metrics(f"{mlp_id} test", e1_metrics["test"])
        _print_metrics(f"{tf_id} test", e2_metrics["test"])
        block = {
            "feature_set": feature_set,
            "feature_columns": stored,
            "n_features": int(len(stored)),
            "n_truncated_phones": int(n_truncated),
            "observed_max_phones": int(observed),
            "pad_len": int(max_seq_len),
            "scaler": scaler.to_dict(),
            "n_train": int((out["role"] == "train").sum()),
            "n_val": int((out["role"] == "val").sum()),
            "n_test": int((out["role"] == "test").sum()),
            "acoustic_model": FEATURE_ACOUSTIC_MODEL[feature_set],
            "gop_type": FEATURE_GOP_TYPE[feature_set],
            "phone_embed": False,
            "n_phone_embed": None,
            "paper_like": True,
            mlp_id: {
                "description": "Polynomial regression (order 2) per phoneme on scalar GOP-SD",
                "method": "poly2_per_phone",
                "phone_embed": False,
                **e1_metrics["test"],
                "train": e1_metrics["train"],
                "val": e1_metrics["val"],
                "fit": {"best_epoch": None, "best_val_mse": None, "epochs_ran": None},
            },
            tf_id: {
                "description": "SVR per phoneme on scalar GOP-SD",
                "method": "svr_per_phone",
                "phone_embed": False,
                **e2_metrics["test"],
                "train": e2_metrics["train"],
                "val": e2_metrics["val"],
                "fit": {"best_epoch": None, "best_val_mse": None, "epochs_ran": None},
            },
            "comparison": {
                "test": {
                    mlp_id: _metric_subset(e1_metrics["test"]),
                    tf_id: _metric_subset(e2_metrics["test"]),
                }
            },
            "checkpoint_paths": {},
        }
        return out, block

    mlp_id, tf_id = scoring_ids(feature_set)
    model_e1 = PhoneMLP(input_dim=len(stored), hidden_dim=hidden_dim, n_phones=n_phones)
    hist_e1 = train_regressor(
        model_e1,
        phone_loader(
            scaled[train_mask],
            y_all[train_mask],
            batch_size=batch_e1,
            shuffle=True,
            seed=seed,
            phone_ids=None if phone_idx is None else phone_idx[train_mask],
        ),
        phone_loader(
            scaled[val_mask],
            y_all[val_mask],
            batch_size=batch_e1,
            shuffle=False,
            seed=seed,
            phone_ids=None if phone_idx is None else phone_idx[val_mask],
        ),
        lr=lr,
        max_epochs=max_epochs,
        patience=patience,
        seed=seed,
        device=device,
        sequence=False,
    )
    pred_e1 = predict_mlp(
        model_e1, scaled, batch_size=batch_e1, device=device, phone_ids=phone_idx
    )
    e1_metrics = _role_prediction_metrics_from_array(df, pred_e1, clip)
    _print_metrics(f"{mlp_id} test", e1_metrics["test"])
    ckpt_root = package_root or PACKAGE_ROOT
    ckpt_protocol = {
        "dataset": cfg.get("dataset"),
        "feature_set": feature_set,
        "acoustic_model": FEATURE_ACOUSTIC_MODEL[feature_set],
        "gop_type": FEATURE_GOP_TYPE[feature_set],
        "seed": seed,
        "score_clip": [float(clip[0]), float(clip[1])],
        "device": str(device),
        "phone_embed": n_phones is not None,
        "n_phone_embed": n_phones,
        "max_seq_len": int(max_seq_len),
    }
    fit_e1 = {
        "best_epoch": hist_e1["best_epoch"],
        "best_val_mse": hist_e1["best_val_mse"],
        "epochs_ran": hist_e1["epochs_ran"],
    }
    ckpt_paths: dict[str, str] = {}
    saved_mlp = _maybe_save_group_e_scorer(
        save_checkpoints=save_checkpoints,
        checkpoint_dir=checkpoint_dir,
        package_root=ckpt_root,
        experiment_id=mlp_id,
        architecture="mlp",
        model=model_e1,
        scaler=scaler,
        model_kwargs={
            "input_dim": int(len(stored)),
            "hidden_dim": hidden_dim,
            "n_phones": n_phones,
        },
        feature_set=feature_set,
        fit=fit_e1,
        protocol=ckpt_protocol,
        role_metrics=e1_metrics,
    )
    if saved_mlp is not None:
        ckpt_paths[mlp_id] = saved_mlp

    scaled_df = df.copy()
    for i, col in enumerate(stored):
        scaled_df[col] = scaled[:, i]
    packed_train = pack_utterances(
        scaled_df.loc[train_mask].reset_index(drop=True),
        stored,
        max_seq_len=max_seq_len,
        **pack_phone_kw,
    )
    packed_val = pack_utterances(
        scaled_df.loc[val_mask].reset_index(drop=True),
        stored,
        max_seq_len=max_seq_len,
        **pack_phone_kw,
    )
    packed_all = pack_utterances(
        scaled_df, stored, max_seq_len=max_seq_len, **pack_phone_kw
    )

    model_e2 = PhoneTransformer(
        input_dim=len(stored),
        d_model=int(tf_cfg.get("d_model", 32)),
        nhead=int(tf_cfg.get("nhead", 4)),
        nlayers=int(tf_cfg.get("nlayers", 2)),
        dim_feedforward=int(tf_cfg.get("dim_feedforward", 64)),
        dropout=float(tf_cfg.get("dropout", 0.1)),
        max_len=max_seq_len,
        n_phones=n_phones,
    )
    hist_e2 = train_regressor(
        model_e2,
        sequence_loader(
            packed_train["x"],
            packed_train["y"],
            packed_train["pad_mask"],
            batch_size=batch_e2,
            shuffle=True,
            seed=seed,
            phone_ids=packed_train.get("phone_ids"),
        ),
        sequence_loader(
            packed_val["x"],
            packed_val["y"],
            packed_val["pad_mask"],
            batch_size=batch_e2,
            shuffle=False,
            seed=seed,
            phone_ids=packed_val.get("phone_ids"),
        ),
        lr=lr,
        max_epochs=max_epochs,
        patience=patience,
        seed=seed,
        device=device,
        sequence=True,
    )
    pred_e2 = predict_transformer(
        model_e2, packed_all, batch_size=batch_e2, device=device, n_rows=len(df)
    )
    e2_metrics = _role_prediction_metrics_from_array(df, pred_e2, clip)
    _print_metrics(f"{tf_id} test", e2_metrics["test"])
    fit_e2 = {
        "best_epoch": hist_e2["best_epoch"],
        "best_val_mse": hist_e2["best_val_mse"],
        "epochs_ran": hist_e2["epochs_ran"],
    }
    saved_tf = _maybe_save_group_e_scorer(
        save_checkpoints=save_checkpoints,
        checkpoint_dir=checkpoint_dir,
        package_root=ckpt_root,
        experiment_id=tf_id,
        architecture="transformer",
        model=model_e2,
        scaler=scaler,
        model_kwargs={
            "input_dim": int(len(stored)),
            "d_model": int(tf_cfg.get("d_model", 32)),
            "nhead": int(tf_cfg.get("nhead", 4)),
            "nlayers": int(tf_cfg.get("nlayers", 2)),
            "dim_feedforward": int(tf_cfg.get("dim_feedforward", 64)),
            "dropout": float(tf_cfg.get("dropout", 0.1)),
            "max_len": int(max_seq_len),
            "n_phones": n_phones,
        },
        feature_set=feature_set,
        fit=fit_e2,
        protocol=ckpt_protocol,
        role_metrics=e2_metrics,
    )
    if saved_tf is not None:
        ckpt_paths[tf_id] = saved_tf

    out = df.copy()
    if CANONICAL_PHONE_COL in out.columns:
        out = out.drop(columns=[CANONICAL_PHONE_COL])
    pred_mlp, pred_tf = scoring_pred_columns(feature_set)
    out[pred_mlp] = np.clip(pred_e1, clip[0], clip[1])
    out[pred_tf] = np.clip(pred_e2, clip[0], clip[1])
    block = {
        "feature_set": feature_set,
        "feature_columns": stored,
        "n_features": int(len(stored)),
        "n_truncated_phones": int(n_truncated),
        "observed_max_phones": int(observed),
        "pad_len": int(max_seq_len),
        "scaler": scaler.to_dict(),
        "n_train": int((out["role"] == "train").sum()),
        "n_val": int((out["role"] == "val").sum()),
        "n_test": int((out["role"] == "test").sum()),
        "acoustic_model": FEATURE_ACOUSTIC_MODEL[feature_set],
        "gop_type": FEATURE_GOP_TYPE[feature_set],
        "phone_embed": n_phones is not None,
        "n_phone_embed": None if n_phones is None else int(n_phones),
        mlp_id: {
            "description": _MLP_DESCRIPTION[mlp_id],
            "hidden_dim": hidden_dim,
            "phone_embed": n_phones is not None,
            **e1_metrics["test"],
            "train": e1_metrics["train"],
            "val": e1_metrics["val"],
            "fit": fit_e1,
        },
        tf_id: {
            "description": _TF_DESCRIPTION[tf_id],
            "d_model": int(tf_cfg.get("d_model", 32)),
            "nhead": int(tf_cfg.get("nhead", 4)),
            "nlayers": int(tf_cfg.get("nlayers", 2)),
            "phone_embed": n_phones is not None,
            **e2_metrics["test"],
            "train": e2_metrics["train"],
            "val": e2_metrics["val"],
            "fit": fit_e2,
        },
        "comparison": {
            "test": {
                mlp_id: _metric_subset(e1_metrics["test"]),
                tf_id: _metric_subset(e2_metrics["test"]),
            }
        },
        "checkpoint_paths": ckpt_paths,
    }
    return out, block


def _merge_locked_predictions(df: pd.DataFrame, path: Path, feature_set: str) -> pd.DataFrame:
    """Keep locked pred columns when writing embed follow-ups into the same CSV."""
    keep = list(_LOCKED_PRED_COLUMNS.get(feature_set, ()))
    if not keep or not path.is_file():
        return df
    old = pd.read_csv(
        path, dtype={"utt_id": str, "phone": str, "split": str, "role": str}
    )
    keep = [c for c in keep if c in old.columns]
    if not keep:
        return df
    keys = ["utt_id", "split", "word_id", "phone_id"]
    extra = old.loc[:, keys + keep].copy()
    extra["utt_id"] = extra["utt_id"].map(str)
    extra["word_id"] = extra["word_id"].astype(np.int64)
    extra["phone_id"] = extra["phone_id"].astype(np.int64)
    work = df.copy()
    work["utt_id"] = work["utt_id"].map(str)
    work["word_id"] = work["word_id"].astype(np.int64)
    work["phone_id"] = work["phone_id"].astype(np.int64)
    merged = work.merge(extra, on=keys, how="left")
    n_miss = int(merged[keep[0]].isna().sum())
    if n_miss:
        hint = _LOCKED_PRED_HINT.get(feature_set, "run the no-embed feature set first")
        raise RuntimeError(
            f"{path} is missing locked {keep} predictions for {n_miss} phones; {hint}"
        )
    return merged


def _resolve_group_e_experiment_slot(
    experiment_id: str,
    feature_set: str,
) -> tuple[str, str]:
    """Return ``(architecture, experiment_id)`` for a Group E neural scorer."""
    eid = str(experiment_id).strip().upper()
    fs = str(feature_set).strip().lower()
    mlp_id, tf_id = scoring_ids(fs)
    if eid == mlp_id:
        return "mlp", eid
    if eid == tf_id:
        return "transformer", eid
    raise ValueError(
        f"experiment {eid!r} is not trained on feature set {fs!r}; "
        f"expected {mlp_id} (mlp) or {tf_id} (transformer)"
    )


def export_group_e_checkpoint(
    cfg: dict[str, Any],
    *,
    experiment_id: str,
    feature_set: str | None = None,
    out_path: str | Path | None = None,
    package_root: Path | None = None,
    force: bool = False,
    device_override: str | None = None,
) -> dict[str, Any]:
    """Train one Group E MLP/Transformer scorer and save a reusable ``.pt`` checkpoint.

    Does not rewrite Group E prediction CSVs. Reusable for E1–E18 neural scorers
    (not paper-like poly/SVR E19–E22).
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Group E requires PyTorch. Install a CPU wheel in the conda env 'gop', "
            "e.g. pip install torch"
        ) from exc

    from gop_empirical.scoring.checkpoint import (
        default_checkpoint_path,
        save_checkpoint,
    )
    from gop_empirical.scoring.mlp import PhoneMLP
    from gop_empirical.scoring.train import (
        phone_loader,
        predict_mlp,
        predict_transformer,
        sequence_loader,
        train_regressor,
    )
    from gop_empirical.scoring.transformer import PhoneTransformer

    package_root = package_root or PACKAGE_ROOT
    if feature_set is None:
        requested = normalize_group_e_features(cfg.get("features"))
        if len(requested) != 1:
            raise ValueError(
                "pass --features explicitly when config lists multiple feature sets"
            )
        feature_set = requested[0]
    else:
        feature_set = normalize_group_e_features([feature_set])[0]

    if _is_paper_like_c10_c11_mode(cfg, feature_set):
        raise ValueError(
            f"feature set {feature_set!r} uses paper-like poly/SVR scorers; "
            "torch checkpoints are only for MLP/Transformer (E1–E18 neural path)"
        )

    architecture, eid = _resolve_group_e_experiment_slot(experiment_id, feature_set)
    paths = cfg["paths"]
    src = _group_e_feature_path(feature_set, paths, package_root)
    if not src.is_file():
        group_name, cfg_name = FEATURE_MISSING_HINT[feature_set]
        if group_name == "extract":
            raise FileNotFoundError(f"missing {src}; {cfg_name}")
        raise FileNotFoundError(
            f"missing {src}; run Group {group_name} first: "
            f"python scripts/run_experiment.py --config configs/{cfg_name}.yaml"
        )

    ckpt_dir = _group_e_checkpoint_dir(cfg, package_root)
    ckpt_path = (
        resolve_path(out_path, base=package_root)
        if out_path is not None
        else default_checkpoint_path(ckpt_dir, eid, architecture)
    )
    if ckpt_path.is_file() and not force:
        raise FileExistsError(
            f"{ckpt_path} already exists; pass --force to overwrite"
        )

    so_dir = resolve_path(paths["speechocean_dir"], base=package_root)
    speaker_meta = load_speaker_metadata(so_dir)
    seed = int(cfg.get("seed", 0))
    val_frac = float(cfg.get("val_speaker_frac", 0.2))
    val_speakers = choose_val_speakers(
        speaker_meta["speakers_train"], frac=val_frac, seed=seed
    )
    clip = _clip_tuple(cfg)
    train_cfg = cfg.get("train") or {}
    device_name = str(
        device_override if device_override is not None else train_cfg.get("device", "cpu")
    ).lower()
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    np.random.seed(seed)
    torch.manual_seed(seed)

    table = load_group_e_feature_table(feature_set, cfg, package_root)
    table = attach_speakers(table, speaker_meta["utt2spk"])
    df = assign_roles(
        table,
        val_speakers=val_speakers,
        train_speakers=speaker_meta["speakers_train"],
        test_speakers=speaker_meta["speakers_test"],
    )

    stored = list(feature_stored_columns(feature_set))
    cap = int(cfg.get("max_seq_len", 50))
    df = df.sort_values(["utt_id", "word_id", "phone_id"], kind="mergesort").reset_index(
        drop=True
    )
    n_phones = None
    pack_phone_kw: dict[str, Any] = {}
    if uses_phone_embed(feature_set):
        spec = phone_embed_spec(feature_set)
        n_phones = int(spec["n_phones"])
        df = attach_canonical_phone_index(
            df, n_phones=n_phones, space=str(spec["space"])
        )
        pack_phone_kw = {
            "phone_idx_col": CANONICAL_PHONE_COL,
            "pad_phone_id": n_phones,
        }
    observed = int(df.groupby("utt_id").size().max()) if len(df) else 1
    max_seq_len = max(cap, observed)
    mlp_cfg = cfg.get("mlp") or {}
    tf_cfg = cfg.get("transformer") or {}
    hidden_dim = int(mlp_cfg.get("hidden_dim", 32))
    batch_e1 = int(train_cfg.get("batch_size_e1", 256))
    batch_e2 = int(train_cfg.get("batch_size_e2", 32))
    lr = float(train_cfg.get("lr", 1e-3))
    max_epochs = int(train_cfg.get("max_epochs", 40))
    patience = int(train_cfg.get("patience", 8))

    train_mask = (df["role"] == "train").to_numpy()
    val_mask = (df["role"] == "val").to_numpy()
    if not train_mask.any() or not val_mask.any():
        raise RuntimeError("empty train or val after speaker split")
    scaler = FeatureScaler.fit(matrix_from_table(df.loc[train_mask], stored))
    scaled = scaler.transform(matrix_from_table(df, stored))
    y_all = df["human_score"].to_numpy(dtype=np.float64)
    phone_idx = (
        None
        if n_phones is None
        else df[CANONICAL_PHONE_COL].to_numpy(dtype=np.int64)
    )

    if architecture == "mlp":
        model_kwargs: dict[str, Any] = {
            "input_dim": int(len(stored)),
            "hidden_dim": hidden_dim,
            "n_phones": n_phones,
        }
        model = PhoneMLP(**model_kwargs)
        fit = train_regressor(
            model,
            phone_loader(
                scaled[train_mask],
                y_all[train_mask],
                batch_size=batch_e1,
                shuffle=True,
                seed=seed,
                phone_ids=None if phone_idx is None else phone_idx[train_mask],
            ),
            phone_loader(
                scaled[val_mask],
                y_all[val_mask],
                batch_size=batch_e1,
                shuffle=False,
                seed=seed,
                phone_ids=None if phone_idx is None else phone_idx[val_mask],
            ),
            lr=lr,
            max_epochs=max_epochs,
            patience=patience,
            seed=seed,
            device=device,
            sequence=False,
        )
        pred = np.clip(
            predict_mlp(
                model, scaled, batch_size=batch_e1, device=device, phone_ids=phone_idx
            ),
            clip[0],
            clip[1],
        )
    else:
        scaled_df = df.copy()
        for i, col in enumerate(stored):
            scaled_df[col] = scaled[:, i]
        packed_train = pack_utterances(
            scaled_df.loc[train_mask].reset_index(drop=True),
            stored,
            max_seq_len=max_seq_len,
            **pack_phone_kw,
        )
        packed_val = pack_utterances(
            scaled_df.loc[val_mask].reset_index(drop=True),
            stored,
            max_seq_len=max_seq_len,
            **pack_phone_kw,
        )
        packed_all = pack_utterances(
            scaled_df, stored, max_seq_len=max_seq_len, **pack_phone_kw
        )
        model_kwargs = {
            "input_dim": int(len(stored)),
            "d_model": int(tf_cfg.get("d_model", 32)),
            "nhead": int(tf_cfg.get("nhead", 4)),
            "nlayers": int(tf_cfg.get("nlayers", 2)),
            "dim_feedforward": int(tf_cfg.get("dim_feedforward", 64)),
            "dropout": float(tf_cfg.get("dropout", 0.1)),
            "max_len": int(max_seq_len),
            "n_phones": n_phones,
        }
        model = PhoneTransformer(**model_kwargs)
        fit = train_regressor(
            model,
            sequence_loader(
                packed_train["x"],
                packed_train["y"],
                packed_train["pad_mask"],
                batch_size=batch_e2,
                shuffle=True,
                seed=seed,
                phone_ids=packed_train.get("phone_ids"),
            ),
            sequence_loader(
                packed_val["x"],
                packed_val["y"],
                packed_val["pad_mask"],
                batch_size=batch_e2,
                shuffle=False,
                seed=seed,
                phone_ids=packed_val.get("phone_ids"),
            ),
            lr=lr,
            max_epochs=max_epochs,
            patience=patience,
            seed=seed,
            device=device,
            sequence=True,
        )
        pred = np.clip(
            predict_transformer(
                model, packed_all, batch_size=batch_e2, device=device, n_rows=len(df)
            ),
            clip[0],
            clip[1],
        )

    scored = df.copy()
    if CANONICAL_PHONE_COL in scored.columns:
        scored = scored.drop(columns=[CANONICAL_PHONE_COL])
    pred_col = "_ckpt_pred"
    scored[pred_col] = pred
    role_metrics = _role_prediction_metrics(scored, pred_col, clip)
    fit_meta = {
        "best_epoch": fit["best_epoch"],
        "best_val_mse": fit["best_val_mse"],
        "epochs_ran": fit["epochs_ran"],
    }
    protocol = {
        "dataset": cfg.get("dataset"),
        "feature_set": feature_set,
        "acoustic_model": FEATURE_ACOUSTIC_MODEL[feature_set],
        "gop_type": FEATURE_GOP_TYPE[feature_set],
        "seed": seed,
        "val_speaker_frac": val_frac,
        "n_speakers_val": int(len(val_speakers)),
        "max_seq_len": int(max_seq_len),
        "score_clip": [float(clip[0]), float(clip[1])],
        "device": str(device),
        "phone_embed": bool(n_phones is not None),
        "n_phone_embed": n_phones,
    }
    saved = save_checkpoint(
        ckpt_path,
        experiment_id=eid,
        architecture=architecture,
        model=model,
        scaler=scaler,
        model_kwargs=model_kwargs,
        feature_set=feature_set,
        fit=fit_meta,
        protocol=protocol,
        metrics={
            "test": _metric_subset(role_metrics["test"]),
            "val": _metric_subset(role_metrics["val"]),
            "train": _metric_subset(role_metrics["train"]),
        },
    )
    _print_metrics(f"{eid} test", role_metrics["test"])
    return {
        "experiment_id": eid,
        "architecture": architecture,
        "feature_set": feature_set,
        "checkpoint_path": rel_path(saved, base=package_root),
        "fit": fit_meta,
        "metrics": {
            "test": role_metrics["test"],
            "val": role_metrics["val"],
            "train": role_metrics["train"],
        },
        "protocol": protocol,
        "n_features": int(len(stored)),
        "pad_len": int(max_seq_len),
    }


def eval_group_e_checkpoint(
    cfg: dict[str, Any],
    *,
    experiment_id: str,
    checkpoint_path: str | Path | None = None,
    package_root: Path | None = None,
    device_override: str | None = None,
    split: str = "test",
) -> dict[str, Any]:
    """Load a Group E ``.pt`` and evaluate on official ``split`` (default test). Does not train."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Group E requires PyTorch. Install a CPU wheel in the conda env 'gop', "
            "e.g. pip install torch"
        ) from exc

    from gop_empirical.scoring.checkpoint import default_checkpoint_path, load_checkpoint
    from gop_empirical.scoring.eval_checkpoint import score_group_e_table

    package_root = package_root or PACKAGE_ROOT
    eid = str(experiment_id).strip().upper()
    architecture = architecture_for_experiment(eid)
    ckpt_dir = _group_e_checkpoint_dir(cfg, package_root)
    ckpt_path = (
        resolve_path(checkpoint_path, base=package_root)
        if checkpoint_path is not None
        else default_checkpoint_path(ckpt_dir, eid, architecture)
    )
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"missing {ckpt_path}; run Group E first, e.g. "
            "python scripts/run_experiment.py --config configs/e_learned_scoring.yaml"
        )

    train_cfg = cfg.get("train") or {}
    device_name = str(
        device_override if device_override is not None else train_cfg.get("device", "cpu")
    ).lower()
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ckpt = load_checkpoint(ckpt_path, device=device)
    loaded_id = str(ckpt["experiment_id"]).strip().upper()
    if loaded_id != eid:
        raise ValueError(f"checkpoint {ckpt_path} is {loaded_id}, not {eid}")
    feature_set = str(ckpt["feature_set"]).strip().lower()
    mlp_id, tf_id = scoring_ids(feature_set)
    if eid not in (mlp_id, tf_id):
        raise ValueError(
            f"experiment {eid} is not trained on feature set {feature_set!r}; "
            f"expected {mlp_id} or {tf_id}"
        )

    src = _group_e_feature_path(feature_set, cfg["paths"], package_root)
    if not src.is_file():
        group_name, cfg_name = FEATURE_MISSING_HINT[feature_set]
        if group_name == "extract":
            raise FileNotFoundError(f"missing {src}; {cfg_name}")
        raise FileNotFoundError(
            f"missing {src}; run Group {group_name} first: "
            f"python scripts/run_experiment.py --config configs/{cfg_name}.yaml"
        )

    table = load_group_e_feature_table(feature_set, cfg, package_root)
    split_name = str(split).strip().lower()
    subset = table[table["split"].astype(str) == split_name].copy()
    if subset.empty:
        raise RuntimeError(f"no phones with split={split_name!r} in {feature_set} table")

    clip = _clip_tuple(cfg)
    batch_mlp = int(train_cfg.get("batch_size_e1", 256))
    batch_tf = int(train_cfg.get("batch_size_e2", 32))
    scored, pred = score_group_e_table(
        ckpt,
        subset,
        clip=clip,
        device=device,
        batch_size_mlp=batch_mlp,
        batch_size_tf=batch_tf,
    )
    metrics = evaluate_predictions(
        pred, scored["human_score"].to_numpy(), clip=clip
    )
    _print_metrics(f"{eid} {split_name}", metrics)
    return {
        "experiment_id": eid,
        "architecture": str(ckpt["architecture"]),
        "feature_set": feature_set,
        "checkpoint_path": rel_path(ckpt_path, base=package_root),
        "split": split_name,
        "device": str(device),
        "metrics": metrics,
        "n": int(metrics["n"]),
    }


def eval_group_e_checkpoints(
    cfg: dict[str, Any],
    experiment_ids: Sequence[str],
    *,
    checkpoint_path: str | Path | None = None,
    package_root: Path | None = None,
    device_override: str | None = None,
    split: str = "test",
) -> list[dict[str, Any]]:
    ids = [str(x).strip().upper() for x in experiment_ids if str(x).strip()]
    if not ids:
        raise ValueError("pass at least one --experiment id (e.g. E15, E1 E2)")
    if checkpoint_path is not None and len(ids) != 1:
        raise ValueError("--checkpoint can only be used with a single --experiment")
    return [
        eval_group_e_checkpoint(
            cfg,
            experiment_id=eid,
            checkpoint_path=checkpoint_path,
            package_root=package_root,
            device_override=device_override,
            split=split,
        )
        for eid in ids
    ]


def run_group_e(
    cfg: dict[str, Any],
    *,
    package_root: Path | None = None,
    features: list[str] | None = None,
) -> dict[str, Any]:
    """Learned scoring: E1–E6 plus E7–E18 GOPT-style LPP+LPR and E19–E22 GOP-SD follow-ups."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Group E requires PyTorch. Install a CPU wheel in the conda env 'gop', "
            "e.g. pip install torch"
        ) from exc

    package_root = package_root or PACKAGE_ROOT
    requested = normalize_group_e_features(features or cfg.get("features"))
    paths = cfg["paths"]
    so_dir = resolve_path(paths["speechocean_dir"], base=package_root)
    feature_paths = {name: _group_e_feature_path(name, paths, package_root) for name in requested}
    for name in requested:
        src = feature_paths[name]
        if not src.is_file():
            group_name, cfg_name = FEATURE_MISSING_HINT[name]
            if group_name == "extract":
                raise FileNotFoundError(f"missing {src}; {cfg_name}")
            raise FileNotFoundError(
                f"missing {src}; run Group {group_name} first: "
                f"python scripts/run_experiment.py --config configs/{cfg_name}.yaml"
            )

    speaker_meta = load_speaker_metadata(so_dir)
    seed = int(cfg.get("seed", 0))
    val_frac = float(cfg.get("val_speaker_frac", 0.2))
    val_speakers = choose_val_speakers(
        speaker_meta["speakers_train"], frac=val_frac, seed=seed
    )
    train_fit_speakers = sorted(set(speaker_meta["speakers_train"]) - set(val_speakers))
    clip = _clip_tuple(cfg)
    train_cfg = cfg.get("train") or {}
    device_name = str(train_cfg.get("device", "cpu")).lower()
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    np.random.seed(seed)
    torch.manual_seed(seed)

    out_dir = resolve_path(paths["output_dir"], base=package_root)
    ckpt_dir = _group_e_checkpoint_dir(cfg, package_root)
    save_ckpts = bool(cfg.get("save_checkpoints", True))
    per_set: dict[str, Any] = {}
    pred_paths: dict[str, str] = {}
    for feature_set in requested:
        table = load_group_e_feature_table(feature_set, cfg, package_root)
        table = attach_speakers(table, speaker_meta["utt2spk"])
        table = assign_roles(
            table,
            val_speakers=val_speakers,
            train_speakers=speaker_meta["speakers_train"],
            test_speakers=speaker_meta["speakers_test"],
        )
        scored, block = _run_feature_set(
            table,
            feature_set,
            cfg,
            seed=seed,
            clip=clip,
            device=device,
            checkpoint_dir=ckpt_dir,
            save_checkpoints=save_ckpts,
            package_root=package_root,
        )
        n_test = int((scored["role"] == "test").sum())
        block["baseline"] = _frozen_linear_baseline(
            feature_set,
            cfg,
            package_root=package_root,
            n_test=n_test,
            table=table,
            clip=clip,
        )
        pred_file = _group_e_pred_filename(feature_set, requested)
        pred_path = out_dir / pred_file
        if feature_set in _LOCKED_PRED_COLUMNS:
            scored = _merge_locked_predictions(scored, pred_path, feature_set)
        pred_path = write_predictions(
            scored, pred_path, columns=_group_e_prediction_columns(feature_set, scored)
        )
        pred_paths[feature_set] = rel_path(pred_path, base=package_root)
        block["predictions_path"] = pred_paths[feature_set]
        per_set[feature_set] = block

    kaldi_requested = [name for name in requested if name in _KALDI_PRIMARY_FEATURE_SETS]
    ssl_only = not kaldi_requested
    primary_pred: Path | None = None
    if kaldi_requested:
        primary_name = "b4" if "b4" in requested else "a1"
        if len(requested) > 1 and _group_e_pred_filename(primary_name, requested) != "e_predictions.csv":
            src_primary = resolve_path(pred_paths[primary_name], base=package_root)
            primary_df = pd.read_csv(
                src_primary, dtype={"utt_id": str, "phone": str, "split": str, "role": str}
            )
            primary_pred = write_predictions(
                primary_df,
                out_dir / "e_predictions.csv",
                columns=_group_e_prediction_columns(primary_name),
            )
        else:
            primary_pred = resolve_path(pred_paths[primary_name], base=package_root)
    else:
        primary_name = requested[0]

    am_by_feat = {name: FEATURE_ACOUSTIC_MODEL[name] for name in requested}
    gop_by_feat = {name: FEATURE_GOP_TYPE[name] for name in requested}
    if ssl_only:
        protocol_am = FEATURE_ACOUSTIC_MODEL[requested[0]]
        protocol_gop = FEATURE_GOP_TYPE[requested[0]]
    else:
        protocol_am = cfg.get("acoustic_model")
        protocol_gop = cfg.get("gop_type", "b4_gop_vector")

    protocol = {
        "dataset": cfg.get("dataset"),
        "acoustic_model": protocol_am,
        "alignment": cfg.get("alignment"),
        "gop_type": protocol_gop,
        "level": cfg.get("level", "phoneme"),
        "scoring": cfg.get("scoring", "learned"),
        "features_requested": requested,
        "acoustic_model_by_features": am_by_feat,
        "gop_type_by_features": gop_by_feat,
        "val_speaker_frac": val_frac,
        "val_speakers": val_speakers,
        "n_speakers_train_fit": int(len(train_fit_speakers)),
        "n_speakers_val": int(len(val_speakers)),
        "n_speakers_test": int(len(speaker_meta["speakers_test"])),
        "speaker_overlap_train_test": int(len(speaker_meta["speaker_overlap"])),
        "max_seq_len": int(cfg.get("max_seq_len", 50)),
        "seed": seed,
        "device": str(device),
        "predictions_path": rel_path(primary_pred, base=package_root) if primary_pred else None,
        "prediction_paths": pred_paths,
    }
    this_run: dict[str, Any] = {
        "group": cfg.get("group", "E"),
        "experiments": experiment_ids_for_features(requested),
        "protocol": protocol,
        "comparison": {
            name: {
                "baseline": per_set[name]["baseline"],
                **per_set[name]["comparison"]["test"],
            }
            for name in requested
        },
    }
    for name in requested:
        this_run[name] = per_set[name]
        mlp_id, tf_id = scoring_ids(name)
        this_run[mlp_id] = per_set[name][mlp_id]
        this_run[tf_id] = per_set[name][tf_id]
    this_run["feature_set"] = primary_name if kaldi_requested else requested[0]

    results_file = out_dir / "e_results.json"
    existing: dict[str, Any] | None = None
    if results_file.is_file():
        existing = json.loads(results_file.read_text(encoding="utf-8"))
    results = merge_group_e_results(existing, this_run)
    results_path = write_results(results, results_file)
    results["results_path"] = rel_path(results_path, base=package_root)
    results["predictions_path"] = results["protocol"].get("predictions_path")
    return results


_MULTISEED_FEATURE = {
    "E2": "b4",
    "E16": "c8_lpp_lpr_embed",
}


def _group_f_paths(cfg: dict[str, Any], package_root: Path) -> dict[str, Path]:
    paths = cfg["paths"]
    keys = [
        "c_predictions",
        "c_results",
        "b_predictions",
        "b_results",
        "e_predictions",
        "e_b5_predictions",
        "e_c8_lpp_lpr_predictions",
        "e_c9_lpp_lpr_predictions",
        "e_results",
        "e_config",
        "scores_detail",
        "speechocean_dir",
        "output_dir",
        "d2_speaker_metrics",
    ]
    return {k: resolve_path(paths[k], base=package_root) for k in keys if k in paths}


def _extreme_speakers(d2_path: Path | None, quantile: float) -> set[str]:
    if d2_path is None or not d2_path.is_file():
        return set()
    d2 = pd.read_csv(d2_path, dtype={"speaker": str})
    if "mae" not in d2.columns or d2.empty:
        return set()
    thr = float(d2["mae"].quantile(float(quantile)))
    return set(d2.loc[d2["mae"] >= thr, "speaker"].astype(str))


def _run_multiseed_model(
    model_id: str,
    seed: int,
    *,
    e_cfg: dict[str, Any],
    package_root: Path,
    val_speakers: list[str],
    clip: tuple[float, float],
    device: Any,
    out_dir: Path,
) -> dict[str, Any]:
    feature_set = _MULTISEED_FEATURE[model_id]
    table = load_group_e_feature_table(feature_set, e_cfg, package_root)
    speaker_meta = load_speaker_metadata(resolve_path(e_cfg["paths"]["speechocean_dir"], base=package_root))
    table = attach_speakers(table, speaker_meta["utt2spk"])
    table = assign_roles(
        table,
        val_speakers=val_speakers,
        train_speakers=speaker_meta["speakers_train"],
        test_speakers=speaker_meta["speakers_test"],
    )
    scored, block = _run_feature_set(
        table, feature_set, e_cfg, seed=int(seed), clip=clip, device=device
    )
    mlp_id, tf_id = scoring_ids(feature_set)
    target_id = tf_id if model_id in ("E2", "E16") else mlp_id
    metrics = block[target_id]
    pred_col = scoring_pred_columns(feature_set)[1]  # transformer column
    test = scored[scored["role"] == "test"].copy()
    keep = [
        c
        for c in (
            "utt_id",
            "split",
            "role",
            "word_id",
            "phone_id",
            "phone",
            "speaker",
            "human_score",
            pred_col,
        )
        if c in test.columns
    ]
    pred_path = out_dir / f"f1_multiseed_{model_id.lower()}_seed{seed}.csv"
    write_predictions(test[keep], pred_path, columns=keep)
    return {
        "model": model_id,
        "feature_set": feature_set,
        "seed": int(seed),
        "pcc": float(metrics["pcc"]),
        "scc": float(metrics["scc"]),
        "mae": float(metrics["mae"]),
        "mse": float(metrics["mse"]),
        "n": int(metrics["n"]),
        "predictions_path": rel_path(pred_path, base=package_root),
    }


def run_group_f(
    cfg: dict[str, Any],
    *,
    package_root: Path | None = None,
    skip_multiseed: bool = False,
) -> dict[str, Any]:
    """F1 bootstrap / paired Δ / multi-seed; F2 residual taxonomy on locked scores."""
    package_root = package_root or PACKAGE_ROOT
    path_map = _group_f_paths(cfg, package_root)
    clip = _clip_tuple(cfg)
    seed = int(cfg.get("seed", 0))
    n_boot = int(cfg.get("n_bootstrap", 1000))
    ci_level = float(cfg.get("ci_level", 0.95))
    model_ids = [str(m) for m in cfg.get("models", list(SCORE_COLUMN))]
    contrasts = [tuple(str(x) for x in pair) for pair in cfg.get("contrasts", [])]

    table = build_group_f_score_table(path_map, clip=clip)
    expected_n = 47369
    if len(table) != expected_n:
        raise RuntimeError(
            f"Group F join n_test={len(table)} does not match A2 n={expected_n}"
        )
    human = table["human_score"].to_numpy(dtype=np.float64)

    f1_models: dict[str, Any] = {}
    for mid in model_ids:
        col = SCORE_COLUMN[mid]
        if col not in table.columns:
            raise KeyError(f"missing score column {col} for model {mid}")
        f1_models[mid] = bootstrap_model_metrics(
            table[col].to_numpy(dtype=np.float64),
            human,
            n_boot=n_boot,
            seed=seed,
            ci_level=ci_level,
        )

    f1_contrasts: dict[str, Any] = {}
    for a, b in contrasts:
        key = f"{a}-{b}"
        f1_contrasts[key] = {
            "a": a,
            "b": b,
            **paired_delta_bootstrap(
                table[SCORE_COLUMN[a]].to_numpy(dtype=np.float64),
                table[SCORE_COLUMN[b]].to_numpy(dtype=np.float64),
                human,
                n_boot=n_boot,
                seed=seed,
                ci_level=ci_level,
            ),
        }

    out_dir = path_map["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    base_cols = ["utt_id", "word_id", "phone_id", "phone", "speaker", "human_score"]
    score_cols = [SCORE_COLUMN[m] for m in model_ids]
    extra = [
        c
        for c in (
            "gop_c1",
            "gop_c8",
            "gop_c9",
            "n_frames_c8",
            "b4_lpp_canonical",
            "b4_lpp_max_competitor",
            "competitor_wins",
        )
        if c in table.columns
    ]
    pred_path = write_predictions(
        table,
        out_dir / "f_predictions.csv",
        columns=base_cols + score_cols + extra,
    )

    # --- F2 ---
    err_cfg = cfg.get("error") or {}
    top_k = int(err_cfg.get("top_k", 100))
    context_gap = float(err_cfg.get("context_err_gap", 0.5))
    accent_pred_min = float(err_cfg.get("accent_pred_min", 1.5))
    spk_q = float(err_cfg.get("speaker_mae_quantile", 0.9))
    extreme = _extreme_speakers(path_map.get("d2_speaker_metrics"), spk_q)

    detail_path = path_map["scores_detail"]
    if not detail_path.is_file():
        raise FileNotFoundError(f"missing {detail_path}")
    markup = phone_markup_table(load_scores_detail(detail_path))
    markup = markup.drop_duplicates(subset=["utt_id", "word_id", "phone_id"], keep="first")
    f2_base = table.merge(
        markup[
            [
                "utt_id",
                "word_id",
                "phone_id",
                "any_accent",
                "any_incorrect",
                "any_insertion",
                "n_experts_accent",
                "n_experts_incorrect",
            ]
        ],
        on=["utt_id", "word_id", "phone_id"],
        how="left",
    )
    f2_base["extreme_speaker"] = f2_base["speaker"].astype(str).isin(extreme)
    f2_base["abs_err_c8"] = np.abs(
        f2_base["score_C8"].to_numpy(dtype=np.float64)
        - f2_base["human_score"].to_numpy(dtype=np.float64)
    )
    f2_base["abs_err_e16"] = np.abs(
        f2_base["score_E16"].to_numpy(dtype=np.float64)
        - f2_base["human_score"].to_numpy(dtype=np.float64)
    )

    f2_blocks: dict[str, Any] = {}
    top_frames: list[pd.DataFrame] = []
    context_pair = {
        "C8": ("abs_err_c8", "abs_err_e16"),
        "E16": ("abs_err_e16", "abs_err_c8"),
    }
    for mid in [str(m) for m in err_cfg.get("models", ["C8", "E16"])]:
        self_col, other_col = context_pair.get(mid, ("abs_err_c8", "abs_err_e16"))
        classified = classify_errors(
            f2_base,
            pred_col=SCORE_COLUMN[mid],
            abs_err_self_col=self_col,
            abs_err_other_col=other_col,
            context_err_gap=context_gap,
            accent_pred_min=accent_pred_min,
        )
        residual_path = out_dir / f"f2_residuals_{mid.lower()}.csv"
        keep_cols = [
            c
            for c in classified.columns
            if c
            in {
                "utt_id",
                "word_id",
                "phone_id",
                "phone",
                "speaker",
                "human_score",
                "pred",
                "residual",
                "abs_err",
                "any_accent",
                "any_incorrect",
                "any_insertion",
                "competitor_wins",
                "n_frames_c8",
                "extreme_speaker",
                "T1_alignment",
                "T2_confusion",
                "T3_accent",
                "T4_context",
                "T5_speaker",
                "primary_type",
            }
        ]
        classified[keep_cols].to_csv(residual_path, index=False)
        top = top_errors(classified, top_k=top_k).copy()
        top.insert(0, "model", mid)
        export_cols = ["model"] + [c for c in keep_cols if c in top.columns]
        top_frames.append(top[export_cols])
        f2_blocks[mid] = {
            "predictions_path": rel_path(residual_path, base=package_root),
            "counts": type_counts(classified),
            "by_human": stratified_error_summary(classified),
            "n_extreme_speakers": int(len(extreme)),
        }

    top_all = pd.concat(top_frames, ignore_index=True) if top_frames else pd.DataFrame()
    top_path = out_dir / "f2_top_errors.csv"
    top_all.to_csv(top_path, index=False)

    # --- F1c multi-seed ---
    multiseed_cfg = cfg.get("multiseed") or {}
    prev_ms: dict[str, Any] | None = None
    existing_results_path = out_dir / "f_results.json"
    if existing_results_path.is_file():
        try:
            prev = json.loads(existing_results_path.read_text(encoding="utf-8"))
            prev_ms = prev.get("F1", {}).get("multiseed")
        except (json.JSONDecodeError, OSError):
            prev_ms = None
    multiseed_block: dict[str, Any] = {"skipped": bool(skip_multiseed), "models": {}}
    if skip_multiseed and isinstance(prev_ms, dict) and prev_ms.get("models"):
        # Keep prior F1c when re-running F1a/F1b/F2 only.
        multiseed_block = prev_ms
        multiseed_block["skipped"] = False
        multiseed_block["preserved_from_prior_run"] = True
    elif not skip_multiseed:
        e_cfg_path = path_map["e_config"]
        if not e_cfg_path.is_file():
            raise FileNotFoundError(f"missing {e_cfg_path}")
        e_cfg = load_config(e_cfg_path)
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "Group F multi-seed requires PyTorch (same as Group E)."
            ) from exc
        lock_val_seed = int(multiseed_cfg.get("lock_val_seed", 0))
        speaker_meta = load_speaker_metadata(path_map["speechocean_dir"])
        val_speakers = choose_val_speakers(
            speaker_meta["speakers_train"],
            frac=float(e_cfg.get("val_speaker_frac", 0.2)),
            seed=lock_val_seed,
        )
        train_cfg = e_cfg.get("train") or {}
        device_name = str(train_cfg.get("device", "cpu")).lower()
        device = (
            torch.device("cuda")
            if device_name == "cuda" and torch.cuda.is_available()
            else torch.device("cpu")
        )
        for mid in [str(m) for m in multiseed_cfg.get("models", ["E2", "E16"])]:
            runs = []
            for s in [int(x) for x in multiseed_cfg.get("seeds", [0, 1, 2, 3, 4])]:
                runs.append(
                    _run_multiseed_model(
                        mid,
                        s,
                        e_cfg=e_cfg,
                        package_root=package_root,
                        val_speakers=val_speakers,
                        clip=clip,
                        device=device,
                        out_dir=out_dir,
                    )
                )
            pccs = np.asarray([r["pcc"] for r in runs], dtype=np.float64)
            multiseed_block["models"][mid] = {
                "val_speakers_locked_seed": lock_val_seed,
                "n_speakers_val": int(len(val_speakers)),
                "runs": runs,
                "pcc_mean": float(pccs.mean()),
                "pcc_std": float(pccs.std(ddof=1)) if len(pccs) > 1 else 0.0,
                "pcc_min": float(pccs.min()),
                "pcc_max": float(pccs.max()),
            }

    protocol = {
        "dataset": cfg.get("dataset"),
        "level": cfg.get("level", "phoneme"),
        "n_test": int(len(table)),
        "n_bootstrap": n_boot,
        "ci_level": ci_level,
        "seed": seed,
        "models": model_ids,
        "contrasts": [f"{a}-{b}" for a, b in contrasts],
        "skip_multiseed": bool(skip_multiseed),
        "predictions_path": rel_path(pred_path, base=package_root),
    }
    results: dict[str, Any] = {
        "group": cfg.get("group", "F"),
        "experiments": cfg.get("experiments", ["F1", "F2"]),
        "protocol": protocol,
        "F1": {
            "description": "Phone-level bootstrap CI + paired Δ + multi-seed E2/E16",
            "models": f1_models,
            "contrasts": f1_contrasts,
            "multiseed": multiseed_block,
        },
        "F2": {
            "description": "Residual taxonomy for C8 and E16 using scores-detail markup",
            "top_k": top_k,
            "top_errors_path": rel_path(top_path, base=package_root),
            "models": f2_blocks,
        },
    }
    results_path = write_results(results, out_dir / "f_results.json")
    results["results_path"] = rel_path(results_path, base=package_root)
    results["predictions_path"] = protocol["predictions_path"]
    return results


def run_from_config(
    config_path: str | Path,
    *,
    models: list[str] | None = None,
    features: list[str] | None = None,
    skip_multiseed: bool = False,
    device: str | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path)
    cfg = load_config(config_path)
    np.random.seed(int(cfg.get("seed", 0)))
    group = str(cfg.get("group", "A")).upper()
    if group == "A":
        return run_group_a(cfg, package_root=PACKAGE_ROOT)
    if group == "B":
        return run_group_b(cfg, package_root=PACKAGE_ROOT)
    if group == "C":
        return run_group_c(cfg, package_root=PACKAGE_ROOT, models=models)
    if group == "D":
        return run_group_d(cfg, package_root=PACKAGE_ROOT)
    if group == "E":
        if device:
            cfg = dict(cfg)
            train = dict(cfg.get("train") or {})
            train["device"] = str(device)
            cfg["train"] = train
        return run_group_e(cfg, package_root=PACKAGE_ROOT, features=features)
    if group == "F":
        return run_group_f(
            cfg, package_root=PACKAGE_ROOT, skip_multiseed=skip_multiseed
        )
    raise ValueError(f"unsupported group {group!r}; expected A, B, C, D, E, or F")
