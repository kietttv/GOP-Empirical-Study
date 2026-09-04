"""A2 metrics: correlation on raw GOP, error after train-only linear map."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression


def correlation_metrics(gop: np.ndarray, human: np.ndarray) -> dict[str, float]:
    gop = np.asarray(gop, dtype=np.float64).reshape(-1)
    human = np.asarray(human, dtype=np.float64).reshape(-1)
    if gop.size != human.size:
        raise ValueError(f"length mismatch: gop={gop.size} human={human.size}")
    if gop.size < 2:
        raise ValueError("need at least 2 phones for correlation")
    pcc = float(pearsonr(gop, human).statistic)
    scc = float(spearmanr(gop, human).statistic)
    return {"pcc": pcc, "scc": scc}


def fit_linear_score_map(gop_train: np.ndarray, y_train: np.ndarray) -> dict[str, float]:
    """Fit GOP → human score on **train only** (no test leakage)."""
    gop_train = np.asarray(gop_train, dtype=np.float64).reshape(-1, 1)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    if gop_train.shape[0] != y_train.shape[0]:
        raise ValueError("train GOP / label length mismatch")
    model = LinearRegression()
    model.fit(gop_train, y_train)
    return {
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
    }


def apply_linear_map(
    gop: np.ndarray,
    mapping: dict[str, float],
    clip: tuple[float, float] | None = (0.0, 2.0),
) -> np.ndarray:
    gop = np.asarray(gop, dtype=np.float64).reshape(-1)
    pred = mapping["slope"] * gop + mapping["intercept"]
    if clip is not None:
        pred = np.clip(pred, clip[0], clip[1])
    return pred


def error_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "mse": float(np.mean(err**2)),
    }


def evaluate_split(
    gop: np.ndarray,
    human: np.ndarray,
    mapping: dict[str, float],
    clip: tuple[float, float] | None = (0.0, 2.0),
) -> dict[str, Any]:
    out = correlation_metrics(gop, human)
    pred = apply_linear_map(gop, mapping, clip=clip)
    out.update(error_metrics(pred, human))
    out["n"] = int(np.asarray(gop).size)
    return out


def fit_linear_score_map_multi(x_train: np.ndarray, y_train: np.ndarray) -> dict[str, Any]:
    """Fit X → human score on **train only** (no test leakage)."""
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    if x_train.ndim == 1:
        x_train = x_train.reshape(-1, 1)
    if x_train.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D train features, got shape {x_train.shape}")
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("train feature / label length mismatch")
    model = LinearRegression()
    model.fit(x_train, y_train)
    return {
        "coef": [float(c) for c in np.asarray(model.coef_).reshape(-1)],
        "intercept": float(model.intercept_),
        "n_features": int(x_train.shape[1]),
    }


def apply_linear_map_multi(
    x: np.ndarray,
    mapping: dict[str, Any],
    clip: tuple[float, float] | None = (0.0, 2.0),
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    coef = np.asarray(mapping["coef"], dtype=np.float64).reshape(-1)
    if x.shape[1] != coef.size:
        raise ValueError(f"feature dim {x.shape[1]} != mapping coef dim {coef.size}")
    pred = x @ coef + float(mapping["intercept"])
    if clip is not None:
        pred = np.clip(pred, clip[0], clip[1])
    return pred


def evaluate_split_vector(
    x: np.ndarray,
    human: np.ndarray,
    mapping: dict[str, Any],
    clip: tuple[float, float] | None = (0.0, 2.0),
) -> dict[str, Any]:
    """PCC/SCC/MAE/MSE of a train-only linear combination vs human scores."""
    pred = apply_linear_map_multi(x, mapping, clip=clip)
    out = correlation_metrics(pred, human)
    out.update(error_metrics(pred, human))
    out["n"] = int(np.asarray(human).reshape(-1).size)
    return out


def evaluate_predictions(
    pred: np.ndarray,
    human: np.ndarray,
    clip: tuple[float, float] | None = (0.0, 2.0),
) -> dict[str, Any]:
    """PCC/SCC/MAE/MSE of model scores vs human (Group E; clip at eval only)."""
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    human = np.asarray(human, dtype=np.float64).reshape(-1)
    if pred.size != human.size:
        raise ValueError(f"length mismatch: pred={pred.size} human={human.size}")
    if clip is not None:
        pred = np.clip(pred, clip[0], clip[1])
    out = correlation_metrics(pred, human)
    out.update(error_metrics(pred, human))
    out["n"] = int(pred.size)
    return out
