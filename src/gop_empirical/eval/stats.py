"""Group F statistical helpers: phone-level bootstrap CI and paired deltas."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from scipy.stats import rankdata

from gop_empirical.eval.metrics import error_metrics


def _as_1d(a: np.ndarray | Sequence[float]) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).reshape(-1)


def _corr_pearson(score: np.ndarray, human: np.ndarray) -> float:
    score = score - score.mean()
    human = human - human.mean()
    denom = float(np.sqrt(np.dot(score, score) * np.dot(human, human)))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(score, human) / denom)


def pearson_pcc(score: np.ndarray, human: np.ndarray) -> float:
    score = _as_1d(score)
    human = _as_1d(human)
    if score.size != human.size:
        raise ValueError(f"length mismatch: score={score.size} human={human.size}")
    if score.size < 2:
        return float("nan")
    return _corr_pearson(score, human)


def spearman_scc(score: np.ndarray, human: np.ndarray) -> float:
    score = _as_1d(score)
    human = _as_1d(human)
    if score.size != human.size:
        raise ValueError(f"length mismatch: score={score.size} human={human.size}")
    if score.size < 2:
        return float("nan")
    return _corr_pearson(rankdata(score), rankdata(human))


def mae_metric(score: np.ndarray, human: np.ndarray) -> float:
    return float(error_metrics(score, human)["mae"])


def mse_metric(score: np.ndarray, human: np.ndarray) -> float:
    return float(error_metrics(score, human)["mse"])


METRIC_FNS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "pcc": pearson_pcc,
    "scc": spearman_scc,
    "mae": mae_metric,
    "mse": mse_metric,
}


def bootstrap_metric(
    score: np.ndarray,
    human: np.ndarray,
    metric: str | Callable[[np.ndarray, np.ndarray], float],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    """Phone-level percentile bootstrap for a bivariate metric."""
    score = _as_1d(score)
    human = _as_1d(human)
    if score.size != human.size:
        raise ValueError(f"length mismatch: score={score.size} human={human.size}")
    n = int(score.size)
    if n < 2:
        raise ValueError("need at least 2 phones for bootstrap")
    fn = METRIC_FNS[metric] if isinstance(metric, str) else metric
    point = float(fn(score, human))
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        samples[i] = float(fn(score[idx], human[idx]))
    alpha = (1.0 - float(ci_level)) / 2.0
    lo = float(np.nanquantile(samples, alpha))
    hi = float(np.nanquantile(samples, 1.0 - alpha))
    finite = samples[np.isfinite(samples)]
    return {
        "point": point,
        "mean": float(np.nanmean(samples)) if finite.size else float("nan"),
        "std": float(np.nanstd(samples, ddof=1)) if finite.size > 1 else float("nan"),
        "ci_low": lo,
        "ci_high": hi,
        "ci_level": float(ci_level),
        "n_boot": int(n_boot),
        "n": n,
        "seed": int(seed),
        "ci_contains_point": bool(np.isfinite(point) and lo <= point <= hi),
    }


def bootstrap_model_metrics(
    score: np.ndarray,
    human: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    ci_level: float = 0.95,
    metrics: Sequence[str] = ("pcc", "scc", "mae", "mse"),
) -> dict[str, Any]:
    out: dict[str, Any] = {"n": int(_as_1d(score).size)}
    for name in metrics:
        out[name] = bootstrap_metric(
            score, human, name, n_boot=n_boot, seed=seed, ci_level=ci_level
        )
    return out


def paired_delta_bootstrap(
    score_a: np.ndarray,
    score_b: np.ndarray,
    human: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    ci_level: float = 0.95,
    metrics: Sequence[str] = ("pcc", "scc"),
) -> dict[str, Any]:
    """Bootstrap CI for metric(A) − metric(B) on the same phone indices."""
    score_a = _as_1d(score_a)
    score_b = _as_1d(score_b)
    human = _as_1d(human)
    if not (score_a.size == score_b.size == human.size):
        raise ValueError(
            f"length mismatch: a={score_a.size} b={score_b.size} human={human.size}"
        )
    n = int(human.size)
    if n < 2:
        raise ValueError("need at least 2 phones for paired bootstrap")
    rng = np.random.default_rng(int(seed))
    alpha = (1.0 - float(ci_level)) / 2.0
    out: dict[str, Any] = {
        "n": n,
        "n_boot": int(n_boot),
        "ci_level": float(ci_level),
        "seed": int(seed),
    }
    for name in metrics:
        fn = METRIC_FNS[name]
        point = float(fn(score_a, human) - fn(score_b, human))
        samples = np.empty(int(n_boot), dtype=np.float64)
        for i in range(int(n_boot)):
            idx = rng.integers(0, n, size=n)
            samples[i] = float(fn(score_a[idx], human[idx]) - fn(score_b[idx], human[idx]))
        lo = float(np.nanquantile(samples, alpha))
        hi = float(np.nanquantile(samples, 1.0 - alpha))
        out[name] = {
            "delta": point,
            "mean": float(np.nanmean(samples)),
            "std": float(np.nanstd(samples, ddof=1)),
            "ci_low": lo,
            "ci_high": hi,
            "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
        }
    return out
