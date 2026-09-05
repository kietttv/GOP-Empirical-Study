"""Group E inputs: locked A/B/C GOP features, speaker-independent val, sequence packing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from gop_empirical.acoustic.phones import load_phone_inventory
from gop_empirical.data.kaldi import list_kaldi_splits, load_kaldi_gop_split
from gop_empirical.data.scores import load_human_scores
from gop_empirical.data.ssl_lpp_lpr import (
    load_ssl_lpp_lpr_split,
    ssl_lpp_lpr_split_path,
)
from gop_empirical.gop.representation import (
    B5_N_PHONES,
    SSL_LPP_LPR_N_PHONES,
    gopt_gop_feature_84,
)

_GROUP_E_FEATURES = (
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
)
_FEATURE_HELP = (
    "b4, a1, c8, c9, c10, c11, b5, b5_embed, c8_lpp_lpr, c9_lpp_lpr, "
    "c8_lpp_lpr_embed, or c9_lpp_lpr_embed"
)
CANONICAL_PHONE_COL = "canonical_phone_idx"

B4_SOURCE_COLUMNS = ("b4_lpp_canonical", "b4_lpp_max_competitor", "b4_lpr")
B4_STORED_COLUMNS = ("feat_lpp_canonical", "feat_lpp_max_competitor", "feat_lpr")
A1_SOURCE_COLUMNS = ("gop",)
A1_STORED_COLUMNS = ("feat_gop",)
C8_SOURCE_COLUMNS = ("gop_c8",)
C8_STORED_COLUMNS = ("feat_gop_c8",)
C9_SOURCE_COLUMNS = ("gop_c9",)
C9_STORED_COLUMNS = ("feat_gop_c9",)
C10_SOURCE_COLUMNS = ("gop_c10",)
C10_STORED_COLUMNS = ("feat_gop_c10",)
C11_SOURCE_COLUMNS = ("gop_c11",)
C11_STORED_COLUMNS = ("feat_gop_c11",)

_BASE_COLUMNS = ("utt_id", "split", "word_id", "phone_id", "phone", "human_score")

B5_STORED_COLUMNS = tuple(f"feat_lpp_{i}" for i in range(B5_N_PHONES)) + tuple(
    f"feat_lpr_{i}" for i in range(B5_N_PHONES)
)
SSL_LPP_LPR_STORED_COLUMNS = tuple(f"feat_lpp_{i}" for i in range(SSL_LPP_LPR_N_PHONES)) + tuple(
    f"feat_lpr_{i}" for i in range(SSL_LPP_LPR_N_PHONES)
)

SCORING_IDS = {
    "b4": ("E1", "E2"),
    "a1": ("E1", "E2"),
    "c8": ("E3", "E4"),
    "c9": ("E5", "E6"),
    "c10": ("E19", "E20"),
    "c11": ("E21", "E22"),
    "b5": ("E7", "E8"),
    "b5_embed": ("E13", "E14"),
    "c8_lpp_lpr": ("E9", "E10"),
    "c9_lpp_lpr": ("E11", "E12"),
    "c8_lpp_lpr_embed": ("E15", "E16"),
    "c9_lpp_lpr_embed": ("E17", "E18"),
}

PHONE_EMBED_SPEC = {
    "b5_embed": {"n_phones": B5_N_PHONES, "space": "kaldi"},
    "c8_lpp_lpr_embed": {"n_phones": SSL_LPP_LPR_N_PHONES, "space": "ssl"},
    "c9_lpp_lpr_embed": {"n_phones": SSL_LPP_LPR_N_PHONES, "space": "ssl"},
}

FEATURE_ACOUSTIC_MODEL = {
    "b4": "kaldi_librispeech_m13",
    "a1": "kaldi_librispeech_m13",
    "b5": "kaldi_librispeech_m13",
    "b5_embed": "kaldi_librispeech_m13",
    "c8": "wav2vec2_xlsr53_espeak_ctc",
    "c9": "wav2vec2_lv60_espeak_ctc",
    "c10": "wav2vec2_xlsr53_espeak_ctc",
    "c11": "wav2vec2_lv60_espeak_ctc",
    "c8_lpp_lpr": "wav2vec2_xlsr53_espeak_ctc",
    "c9_lpp_lpr": "wav2vec2_lv60_espeak_ctc",
    "c8_lpp_lpr_embed": "wav2vec2_xlsr53_espeak_ctc",
    "c9_lpp_lpr_embed": "wav2vec2_lv60_espeak_ctc",
}

FEATURE_GOP_TYPE = {
    "b4": "b4_gop_vector",
    "a1": "canonical_lpp",
    "b5": "gopt_gop_feature_84",
    "b5_embed": "gopt_gop_feature_84_phone_embed",
    "c8": "cao_gop_s",
    "c9": "cao_gop_s",
    "c10": "cao_gop_sd",
    "c11": "cao_gop_sd",
    "c8_lpp_lpr": "gopt_style_lpp_lpr_78_ctc_align",
    "c9_lpp_lpr": "gopt_style_lpp_lpr_78_ctc_align",
    "c8_lpp_lpr_embed": "gopt_style_lpp_lpr_78_ctc_align_phone_embed",
    "c9_lpp_lpr_embed": "gopt_style_lpp_lpr_78_ctc_align_phone_embed",
}

FEATURE_MISSING_HINT = {
    "b4": ("B", "b_gop_representation"),
    "a1": ("A", "a_traditional_gop"),
    "b5": ("B", "b_gop_representation"),
    "b5_embed": ("B", "b_gop_representation"),
    "c8": ("C", "c_acoustic_model"),
    "c9": ("C", "c_acoustic_model"),
    "c10": ("C", "c_acoustic_model"),
    "c11": ("C", "c_acoustic_model"),
    "c8_lpp_lpr": (
        "extract",
        "python scripts/extract_ssl_gop.py --config configs/c_acoustic_model.yaml --model xlsr_espeak --gop lpp_lpr",
    ),
    "c9_lpp_lpr": (
        "extract",
        "python scripts/extract_ssl_gop.py --config configs/c_acoustic_model.yaml --model lv60_espeak --gop lpp_lpr",
    ),
    "c8_lpp_lpr_embed": (
        "extract",
        "python scripts/extract_ssl_gop.py --config configs/c_acoustic_model.yaml --model xlsr_espeak --gop lpp_lpr",
    ),
    "c9_lpp_lpr_embed": (
        "extract",
        "python scripts/extract_ssl_gop.py --config configs/c_acoustic_model.yaml --model lv60_espeak --gop lpp_lpr",
    ),
}

DUMP_FEATURE_COLUMNS = frozenset({"b4", "a1", "c8", "c9", "c10", "c11"})


def _resolve_under(raw: str | Path, package_root: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return package_root / path


def normalize_group_e_features(features: Sequence[str] | str | None) -> list[str]:
    if features is None:
        return ["b4"]
    if isinstance(features, str):
        raw_list = [features]
    else:
        raw_list = list(features)
    out: list[str] = []
    for raw in raw_list:
        name = str(raw).strip().lower()
        if name not in _GROUP_E_FEATURES:
            raise ValueError(f"unknown Group E features {raw!r}; expected {_FEATURE_HELP}")
        if name not in out:
            out.append(name)
    if not out:
        return ["b4"]
    return out


def _require_feature_set(feature_set: str) -> str:
    name = str(feature_set).strip().lower()
    if name not in _GROUP_E_FEATURES:
        raise ValueError(f"unknown Group E features {feature_set!r}; expected {_FEATURE_HELP}")
    return name


def scoring_ids(feature_set: str) -> tuple[str, str]:
    return SCORING_IDS[_require_feature_set(feature_set)]


def architecture_for_experiment(experiment_id: str) -> str:
    """Return ``mlp`` or ``transformer`` for a Group E experiment id."""
    eid = str(experiment_id).strip().upper()
    mlp_ids = {pair[0] for pair in SCORING_IDS.values()}
    tf_ids = {pair[1] for pair in SCORING_IDS.values()}
    if eid in mlp_ids:
        return "mlp"
    if eid in tf_ids:
        return "transformer"
    raise ValueError(f"unknown Group E experiment {eid!r}")


def scoring_pred_columns(feature_set: str) -> tuple[str, str]:
    mlp_id, tf_id = scoring_ids(feature_set)
    return (f"pred_{mlp_id.lower()}", f"pred_{tf_id.lower()}")


def experiment_ids_for_features(features: Sequence[str]) -> list[str]:
    out: list[str] = []
    for name in features:
        for eid in scoring_ids(name):
            if eid not in out:
                out.append(eid)
    return out


def feature_source_columns(feature_set: str) -> tuple[str, ...]:
    name = _require_feature_set(feature_set)
    if name == "b4":
        return B4_SOURCE_COLUMNS
    if name == "a1":
        return A1_SOURCE_COLUMNS
    if name == "c8":
        return C8_SOURCE_COLUMNS
    if name == "c9":
        return C9_SOURCE_COLUMNS
    if name == "c10":
        return C10_SOURCE_COLUMNS
    if name == "c11":
        return C11_SOURCE_COLUMNS
    if name in ("b5", "b5_embed"):
        return B5_STORED_COLUMNS
    return SSL_LPP_LPR_STORED_COLUMNS


def feature_stored_columns(feature_set: str) -> tuple[str, ...]:
    name = _require_feature_set(feature_set)
    if name == "b4":
        return B4_STORED_COLUMNS
    if name == "a1":
        return A1_STORED_COLUMNS
    if name == "c8":
        return C8_STORED_COLUMNS
    if name == "c9":
        return C9_STORED_COLUMNS
    if name == "c10":
        return C10_STORED_COLUMNS
    if name == "c11":
        return C11_STORED_COLUMNS
    if name in ("b5", "b5_embed"):
        return B5_STORED_COLUMNS
    return SSL_LPP_LPR_STORED_COLUMNS


def uses_phone_embed(feature_set: str) -> bool:
    return _require_feature_set(feature_set) in PHONE_EMBED_SPEC


def phone_embed_spec(feature_set: str) -> dict[str, Any]:
    name = _require_feature_set(feature_set)
    if name not in PHONE_EMBED_SPEC:
        raise ValueError(f"{feature_set!r} has no phone embedding")
    return dict(PHONE_EMBED_SPEC[name])


def _normalize_utt_id(utt_id: object) -> str:
    text = str(utt_id).strip()
    if text.isdigit():
        return text.zfill(9)
    return text


def _read_prediction_csv(path: str | Any, required: set[str]) -> pd.DataFrame:
    csv_path = Path(path)
    df = pd.read_csv(csv_path, dtype={"utt_id": str, "phone": str, "split": str})
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}")
    df["utt_id"] = df["utt_id"].map(_normalize_utt_id)
    return df


def load_b4_predictions(path: str | Any) -> pd.DataFrame:
    required = set(_BASE_COLUMNS) | set(B4_SOURCE_COLUMNS)
    df = _read_prediction_csv(path, required)
    out = df.loc[:, list(_BASE_COLUMNS)].copy()
    for src, dst in zip(B4_SOURCE_COLUMNS, B4_STORED_COLUMNS, strict=True):
        out[dst] = df[src].astype(np.float64)
    return out


def load_a1_feature_table(path: str | Any) -> pd.DataFrame:
    required = set(_BASE_COLUMNS) | set(A1_SOURCE_COLUMNS)
    df = _read_prediction_csv(path, required)
    out = df.loc[:, list(_BASE_COLUMNS)].copy()
    out[A1_STORED_COLUMNS[0]] = df["gop"].astype(np.float64)
    return out


def load_c_gop_feature_table(path: str | Any, feature_set: str) -> pd.DataFrame:
    """Scalar Cao GOP from Group C predictions (C8/C9 GOP-S; C10/C11 GOP-SD)."""
    name = _require_feature_set(feature_set)
    if name not in ("c8", "c9", "c10", "c11"):
        raise ValueError(f"expected c8, c9, c10, or c11, got {feature_set!r}")
    src = feature_source_columns(name)[0]
    dst = feature_stored_columns(name)[0]
    required = set(_BASE_COLUMNS) | {src}
    df = _read_prediction_csv(path, required)
    out = df.loc[:, list(_BASE_COLUMNS)].copy()
    out[dst] = pd.to_numeric(df[src], errors="coerce")
    n_before = len(out)
    out = out.loc[out[dst].notna()].copy()
    out[dst] = out[dst].astype(np.float64)
    if out.empty:
        raise ValueError(f"{path}: no finite {src} values")
    n_drop = n_before - len(out)
    if n_drop:
        out.attrs["n_dropped_nan"] = int(n_drop)
    return out.reset_index(drop=True)


def load_b5_feature_table(cfg: dict[str, Any], package_root: Path) -> pd.DataFrame:
    """Kaldi 84-d LPP+LPR (B5 / GOPT paper), joined to human scores."""
    paths = cfg["paths"]
    kaldi_dir = _resolve_under(paths["kaldi_dir"], package_root)
    scores_path = _resolve_under(paths["scores_json"], package_root)
    skip = {str(p) for p in cfg.get("skip_phones", [])}
    phone_index_base = int(cfg.get("phone_index_base", 0))
    expected_n = int(cfg.get("expected_n_phones", B5_N_PHONES))
    floor = float(cfg.get("score_floor", 0.1))
    if phone_index_base != 0:
        raise ValueError("B5 loader expects phone_index_base 0")
    splits = list_kaldi_splits(kaldi_dir)
    if "train" not in splits or "test" not in splits:
        raise FileNotFoundError(f"need train and test CSVs under {kaldi_dir}")
    human = load_human_scores(scores_path, floor=floor)
    stored = feature_stored_columns("b5")
    base_rows: list[dict[str, Any]] = []
    feat_rows: list[np.ndarray] = []
    for split in ("train", "test"):
        feats, keys = load_kaldi_gop_split(kaldi_dir, split)
        x84 = gopt_gop_feature_84(feats, expected_n_phones=expected_n)
        for key, feat84 in zip(keys, x84, strict=True):
            rec = human.get(str(key))
            if rec is None or rec["phone"] in skip:
                continue
            base_rows.append(
                {
                    "utt_id": rec["utt_id"],
                    "split": split,
                    "word_id": rec["word_id"],
                    "phone_id": rec["phone_id"],
                    "phone": rec["phone"],
                    "human_score": float(rec["human_score"]),
                }
            )
            feat_rows.append(np.asarray(feat84, dtype=np.float64))
    if not base_rows:
        raise RuntimeError("no phones joined for B5; check Kaldi keys vs scores.json")
    out = pd.DataFrame(base_rows)
    feat_mat = np.vstack(feat_rows)
    if feat_mat.shape != (len(out), len(stored)):
        raise RuntimeError(f"B5 feature shape {feat_mat.shape} != ({len(out)}, {len(stored)})")
    for i, col in enumerate(stored):
        out[col] = feat_mat[:, i]
    out["utt_id"] = out["utt_id"].map(_normalize_utt_id)
    return out


def attach_canonical_phone_index(
    df: pd.DataFrame,
    *,
    n_phones: int = B5_N_PHONES,
    inventory_path: str | Path | None = None,
    space: str = "kaldi",
) -> pd.DataFrame:
    """Map ``phone`` symbols to inventory ids (not word-level ``phone_id``).

    ``space='kaldi'`` uses 42 Kaldi slots. ``space='ssl'`` uses the 39 scored
    CMU-mapped IPA indices (same order as the 78-d LPP+LPR vector).
    """
    inv = load_phone_inventory(inventory_path)
    space_name = str(space).strip().lower()
    ids: list[int] = []
    for symbol in df["phone"].astype(str).tolist():
        if space_name == "ssl":
            kid = int(inv.ssl_index(symbol))
        elif space_name == "kaldi":
            if symbol not in inv.kaldi_symbol_to_id:
                raise KeyError(f"phone {symbol!r} is not in the Kaldi inventory")
            kid = int(inv.kaldi_symbol_to_id[symbol])
        else:
            raise ValueError(f"unknown phone space {space!r}; expected kaldi or ssl")
        if kid < 0 or kid >= int(n_phones):
            raise ValueError(f"{space_name} id {kid} for {symbol!r} outside [0, {n_phones})")
        ids.append(kid)
    out = df.copy()
    out[CANONICAL_PHONE_COL] = np.asarray(ids, dtype=np.int64)
    return out


def _utt_level_phone_index(df: pd.DataFrame) -> pd.Series:
    """Kaldi / scores.json key index: 0..n-1 across the utterance, not per word.

    ``c_predictions.phone_id`` is the phone index *inside a word*. Extract npz keys
    are ``utt_id.phn_idx`` with ``phn_idx`` counting every scored phone in the
    utterance (same as ``load_human_scores``).
    """
    order = df.sort_values(["split", "utt_id", "word_id", "phone_id"], kind="mergesort")
    idx = order.groupby(["split", "utt_id"], sort=False).cumcount()
    return idx.reindex(df.index)


def load_ssl_lpp_lpr_feature_table(
    c_predictions_path: str | Path,
    npz_dir: str | Path,
) -> pd.DataFrame:
    """Join C predictions (human/keys) to 78-d GOPT-style LPP+LPR."""
    df = _read_prediction_csv(c_predictions_path, set(_BASE_COLUMNS))
    utt_phn_idx = _utt_level_phone_index(df)
    by_split = {
        "train": load_ssl_lpp_lpr_split(npz_dir, "train"),
        "test": load_ssl_lpp_lpr_split(npz_dir, "test"),
    }
    stored = SSL_LPP_LPR_STORED_COLUMNS
    keep: list[bool] = []
    feats: list[np.ndarray] = []
    n_phone_mismatch = 0
    for split, utt, phn_idx, phone in zip(
        df["split"].tolist(),
        df["utt_id"].tolist(),
        utt_phn_idx.tolist(),
        df["phone"].tolist(),
        strict=True,
    ):
        rec = by_split.get(str(split), {}).get(f"{utt}.{int(phn_idx)}")
        if rec is None or not np.isfinite(rec["features"]).all():
            keep.append(False)
            continue
        if rec["phone"] != str(phone):
            n_phone_mismatch += 1
        keep.append(True)
        feats.append(np.asarray(rec["features"], dtype=np.float64))
    mask = np.asarray(keep, dtype=bool)
    out = df.loc[mask, list(_BASE_COLUMNS)].copy().reset_index(drop=True)
    if out.empty:
        raise ValueError(f"{npz_dir}: no finite 78-d LPP+LPR rows joined to {c_predictions_path}")
    if n_phone_mismatch:
        raise RuntimeError(
            f"{npz_dir}: {n_phone_mismatch} joined rows have a different phone than "
            "c_predictions (utt.phn_idx / word-level phone_id mix-up)"
        )
    feat_mat = np.vstack(feats)
    if feat_mat.shape != (len(out), len(stored)):
        raise RuntimeError(
            f"SSL LPP+LPR shape {feat_mat.shape} != ({len(out)}, {len(stored)})"
        )
    for i, col in enumerate(stored):
        out[col] = feat_mat[:, i]
    n_drop = int((~mask).sum())
    if n_drop:
        out.attrs["n_dropped_nan"] = n_drop
    return out


def group_e_artifact_path(
    feature_set: str,
    paths: dict[str, Any],
    package_root: Path,
) -> Path:
    name = _require_feature_set(feature_set)
    if name == "b4":
        return _resolve_under(paths["b_predictions"], package_root)
    if name == "a1":
        return _resolve_under(paths["a1_predictions"], package_root)
    if name in ("b5", "b5_embed"):
        if "b_predictions" not in paths:
            raise FileNotFoundError("config paths.b_predictions is required for --features b5")
        return _resolve_under(paths["b_predictions"], package_root)
    if name in ("c8", "c9", "c10", "c11"):
        if "c_predictions" not in paths:
            raise FileNotFoundError(
                "config paths.c_predictions is required for --features c8/c9/c10/c11"
            )
        return _resolve_under(paths["c_predictions"], package_root)
    if name in ("c8_lpp_lpr", "c8_lpp_lpr_embed"):
        key = "xlsr_espeak_lpp_lpr_dir"
        if key not in paths:
            raise FileNotFoundError(f"config paths.{key} is required for --features {name}")
        return ssl_lpp_lpr_split_path(_resolve_under(paths[key], package_root), "test")
    if name in ("c9_lpp_lpr", "c9_lpp_lpr_embed"):
        key = "lv60_espeak_lpp_lpr_dir"
        if key not in paths:
            raise FileNotFoundError(f"config paths.{key} is required for --features {name}")
        return ssl_lpp_lpr_split_path(_resolve_under(paths[key], package_root), "test")
    raise ValueError(f"unknown Group E features {feature_set!r}; expected {_FEATURE_HELP}")


def load_group_e_feature_table(
    feature_set: str,
    cfg: dict[str, Any],
    package_root: Path,
) -> pd.DataFrame:
    name = _require_feature_set(feature_set)
    paths = cfg["paths"]
    if name in ("b5", "b5_embed"):
        return load_b5_feature_table(cfg, package_root)
    if name in ("c8_lpp_lpr", "c8_lpp_lpr_embed"):
        return load_ssl_lpp_lpr_feature_table(
            _resolve_under(paths["c_predictions"], package_root),
            _resolve_under(paths["xlsr_espeak_lpp_lpr_dir"], package_root),
        )
    if name in ("c9_lpp_lpr", "c9_lpp_lpr_embed"):
        return load_ssl_lpp_lpr_feature_table(
            _resolve_under(paths["c_predictions"], package_root),
            _resolve_under(paths["lv60_espeak_lpp_lpr_dir"], package_root),
        )
    return load_feature_table(group_e_artifact_path(name, paths, package_root), name)


def load_feature_table(path: str | Any, feature_set: str) -> pd.DataFrame:
    name = _require_feature_set(feature_set)
    if name == "b4":
        return load_b4_predictions(path)
    if name == "a1":
        return load_a1_feature_table(path)
    if name in ("c8", "c9", "c10", "c11"):
        return load_c_gop_feature_table(path, name)
    raise ValueError(
        f"load_feature_table does not support {name!r}; use load_group_e_feature_table"
    )


def attach_speakers(df: pd.DataFrame, utt2spk: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    out["speaker"] = out["utt_id"].map(utt2spk)
    n_missing = int(out["speaker"].isna().sum())
    if n_missing:
        raise ValueError(f"{n_missing} phones have no utt2spk mapping")
    return out


def choose_val_speakers(
    train_speakers: Sequence[str],
    *,
    frac: float,
    seed: int,
) -> list[str]:
    speakers = sorted({str(s) for s in train_speakers})
    if not speakers:
        raise ValueError("no train speakers to split for validation")
    if not 0.0 < float(frac) < 1.0:
        raise ValueError(f"val_speaker_frac must be in (0, 1), got {frac}")
    n_val = int(round(len(speakers) * float(frac)))
    n_val = min(max(n_val, 1), len(speakers) - 1) if len(speakers) > 1 else 1
    rng = np.random.RandomState(int(seed))
    chosen = rng.choice(np.asarray(speakers, dtype=object), size=n_val, replace=False)
    return sorted(str(s) for s in chosen)


def assign_roles(
    df: pd.DataFrame,
    *,
    val_speakers: Sequence[str],
    train_speakers: set[str],
    test_speakers: set[str],
) -> pd.DataFrame:
    """Label each phone train/val/test. Official test is never used as val."""
    out = df.copy()
    val_set = {str(s) for s in val_speakers}
    overlap = val_set & {str(s) for s in test_speakers}
    if overlap:
        raise ValueError(f"val speakers leak into official test: {sorted(overlap)[:5]}")
    extra = val_set - {str(s) for s in train_speakers}
    if extra:
        raise ValueError(f"val speakers are not official train speakers: {sorted(extra)[:5]}")

    roles = np.full(len(out), "", dtype=object)
    split = out["split"].astype(str)
    speaker = out["speaker"].astype(str)
    roles[split == "test"] = "test"
    train_mask = split == "train"
    roles[train_mask & speaker.isin(val_set)] = "val"
    roles[train_mask & ~speaker.isin(val_set)] = "train"
    unknown = [s for s in pd.unique(split) if s not in {"train", "test"}]
    if unknown:
        raise ValueError(f"unexpected split values: {unknown}")
    if np.any(roles == ""):
        raise RuntimeError("failed to assign a role to every phone")
    out["role"] = roles
    return out


class FeatureScaler:
    """Train-only z-score. Never fit on val or test."""

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float64).reshape(-1)
        self.std = np.asarray(std, dtype=np.float64).reshape(-1)
        if self.mean.size != self.std.size:
            raise ValueError("mean/std length mismatch")
        if np.any(self.std <= 0):
            raise ValueError("std must be positive")

    @classmethod
    def fit(cls, x_train: np.ndarray, *, eps: float = 1e-6) -> FeatureScaler:
        x_train = np.asarray(x_train, dtype=np.float64)
        if x_train.ndim == 1:
            x_train = x_train.reshape(-1, 1)
        if x_train.ndim != 2:
            raise ValueError(f"expected 2-D train features, got {x_train.shape}")
        if x_train.shape[0] < 2:
            raise ValueError("need at least 2 train phones to fit a scaler")
        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0, ddof=0)
        std = np.maximum(std, float(eps))
        return cls(mean, std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.shape[1] != self.mean.size:
            raise ValueError(f"feature dim {x.shape[1]} != scaler dim {self.mean.size}")
        return (x - self.mean) / self.std

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": [float(v) for v in self.mean],
            "std": [float(v) for v in self.std],
            "n_features": int(self.mean.size),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureScaler:
        if "mean" not in payload or "std" not in payload:
            raise ValueError("scaler dict requires mean and std")
        return cls(
            np.asarray(payload["mean"], dtype=np.float64),
            np.asarray(payload["std"], dtype=np.float64),
        )


def matrix_from_table(df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"table missing feature columns: {missing}")
    return df.loc[:, list(columns)].to_numpy(dtype=np.float64)


def keep_first_phones(df: pd.DataFrame, max_seq_len: int) -> tuple[pd.DataFrame, int]:
    """Keep the first ``max_seq_len`` phones per utterance (word_id, phone_id order)."""
    if int(max_seq_len) < 1:
        raise ValueError("max_seq_len must be >= 1")
    ordered = df.sort_values(["utt_id", "word_id", "phone_id"], kind="mergesort").reset_index(
        drop=True
    )
    rank = ordered.groupby("utt_id", sort=False).cumcount()
    keep = rank < int(max_seq_len)
    n_drop = int((~keep).sum())
    return ordered.loc[keep].reset_index(drop=True), n_drop


def pack_utterances(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    max_seq_len: int,
    phone_idx_col: str | None = None,
    pad_phone_id: int | None = None,
) -> dict[str, Any]:
    """Pad phone sequences. ``pad_mask`` is True on pad slots (ignored by the loss)."""
    ordered = df.sort_values(["utt_id", "word_id", "phone_id"], kind="mergesort")
    utt_ids = list(dict.fromkeys(ordered["utt_id"].astype(str).tolist()))
    n_utt = len(utt_ids)
    d_feat = len(feature_cols)
    t_max = int(max_seq_len)
    x = np.zeros((n_utt, t_max, d_feat), dtype=np.float64)
    y = np.zeros((n_utt, t_max), dtype=np.float64)
    pad_mask = np.ones((n_utt, t_max), dtype=bool)
    lengths = np.zeros(n_utt, dtype=np.int64)
    row_indices: list[np.ndarray] = []
    phone_ids: np.ndarray | None = None
    if phone_idx_col is not None:
        if pad_phone_id is None:
            raise ValueError("pad_phone_id is required when packing phone ids")
        if phone_idx_col not in ordered.columns:
            raise ValueError(f"table missing {phone_idx_col}")
        phone_ids = np.full((n_utt, t_max), int(pad_phone_id), dtype=np.int64)
    grouped = {utt: grp for utt, grp in ordered.groupby("utt_id", sort=False)}
    for i, utt in enumerate(utt_ids):
        grp = grouped[utt]
        if len(grp) > t_max:
            raise ValueError(
                f"utterance {utt} has {len(grp)} phones; call keep_first_phones first"
            )
        idx = grp.index.to_numpy()
        feat = matrix_from_table(grp, feature_cols)
        n = int(feat.shape[0])
        x[i, :n, :] = feat
        y[i, :n] = grp["human_score"].to_numpy(dtype=np.float64)
        pad_mask[i, :n] = False
        lengths[i] = n
        row_indices.append(idx)
        if phone_ids is not None:
            phone_ids[i, :n] = grp[phone_idx_col].to_numpy(dtype=np.int64)
    packed = {
        "utt_ids": utt_ids,
        "x": x,
        "y": y,
        "pad_mask": pad_mask,
        "lengths": lengths,
        "row_indices": row_indices,
    }
    if phone_ids is not None:
        packed["phone_ids"] = phone_ids
        packed["pad_phone_id"] = int(pad_phone_id)
    return packed
