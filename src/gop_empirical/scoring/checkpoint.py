"""Save/load Group E scorer checkpoints (MLP / Transformer)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from gop_empirical.data.learned import FeatureScaler
from gop_empirical.scoring.mlp import PhoneMLP
from gop_empirical.scoring.transformer import PhoneTransformer

ARCHITECTURES = frozenset({"mlp", "transformer"})


def default_checkpoint_path(
    checkpoint_dir: str | Path,
    experiment_id: str,
    architecture: str,
) -> Path:
    """e.g. ``checkpoints/e16/transformer_ckpt.pt``."""
    arch = str(architecture).strip().lower()
    if arch not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {sorted(ARCHITECTURES)}, got {architecture!r}")
    eid = str(experiment_id).strip().lower()
    if not eid:
        raise ValueError("experiment_id must be non-empty")
    return Path(checkpoint_dir) / eid / f"{arch}_ckpt.pt"


def save_checkpoint(
    path: str | Path,
    *,
    experiment_id: str,
    architecture: str,
    model: nn.Module,
    scaler: FeatureScaler,
    model_kwargs: dict[str, Any],
    feature_set: str,
    fit: dict[str, Any] | None = None,
    protocol: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """Persist best weights + scaler + rebuild metadata."""
    arch = str(architecture).strip().lower()
    if arch not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {sorted(ARCHITECTURES)}, got {architecture!r}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    scaler_dict = scaler.to_dict()
    payload: dict[str, Any] = {
        "experiment_id": str(experiment_id).strip().upper(),
        "architecture": arch,
        "model_kwargs": dict(model_kwargs),
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "scaler_mean": [float(v) for v in scaler.mean],
        "scaler_std": [float(v) for v in scaler.std],
        "scaler": scaler_dict,
        "feature_set": str(feature_set).strip().lower(),
        "fit": dict(fit or {}),
        "protocol": dict(protocol or {}),
        "metrics": dict(metrics or {}),
    }
    torch.save(payload, out)
    return out


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device | None = None,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Load a Group E checkpoint; rebuild model on ``device`` if given."""
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    loc = map_location if map_location is not None else "cpu"
    try:
        payload = torch.load(ckpt_path, map_location=loc, weights_only=False)
    except TypeError:
        payload = torch.load(ckpt_path, map_location=loc)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint must be a dict: {ckpt_path}")
    required = ("experiment_id", "architecture", "model_kwargs", "state_dict", "feature_set")
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"checkpoint missing keys {missing}: {ckpt_path}")
    arch = str(payload["architecture"]).strip().lower()
    if arch not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {arch!r} in {ckpt_path}")

    if "scaler_mean" in payload and "scaler_std" in payload:
        scaler = FeatureScaler(
            np.asarray(payload["scaler_mean"], dtype=np.float64),
            np.asarray(payload["scaler_std"], dtype=np.float64),
        )
    elif isinstance(payload.get("scaler"), dict):
        scaler = FeatureScaler.from_dict(payload["scaler"])
    else:
        raise ValueError(f"checkpoint missing scaler: {ckpt_path}")

    model_kwargs = dict(payload["model_kwargs"])
    model = build_model(arch, model_kwargs)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    if device is not None:
        model.to(torch.device(device))

    return {
        "path": str(ckpt_path),
        "experiment_id": str(payload["experiment_id"]).upper(),
        "architecture": arch,
        "model_kwargs": model_kwargs,
        "state_dict": payload["state_dict"],
        "model": model,
        "scaler": scaler,
        "feature_set": str(payload["feature_set"]).strip().lower(),
        "fit": dict(payload.get("fit") or {}),
        "protocol": dict(payload.get("protocol") or {}),
        "metrics": dict(payload.get("metrics") or {}),
    }


def build_model(architecture: str, model_kwargs: dict[str, Any]) -> nn.Module:
    arch = str(architecture).strip().lower()
    kwargs = dict(model_kwargs)
    if arch == "mlp":
        return PhoneMLP(**kwargs)
    if arch == "transformer":
        return PhoneTransformer(**kwargs)
    raise ValueError(f"architecture must be one of {sorted(ARCHITECTURES)}, got {architecture!r}")
