"""Unit tests for Group E checkpoint save/load."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gop_empirical.data.learned import FeatureScaler
from gop_empirical.scoring.checkpoint import (
    build_model,
    default_checkpoint_path,
    load_checkpoint,
    save_checkpoint,
)
from gop_empirical.scoring.mlp import PhoneMLP
from gop_empirical.scoring.transformer import PhoneTransformer


def test_default_checkpoint_path() -> None:
    p = default_checkpoint_path("checkpoints", "E16", "transformer")
    assert p == Path("checkpoints/e16/transformer_ckpt.pt")
    p2 = default_checkpoint_path("checkpoints", "e15", "mlp")
    assert p2 == Path("checkpoints/e15/mlp_ckpt.pt")
    assert default_checkpoint_path("checkpoints", "E1", "mlp") == Path(
        "checkpoints/e1/mlp_ckpt.pt"
    )


def test_save_load_transformer_roundtrip(tmp_path: Path) -> None:
    model = PhoneTransformer(
        input_dim=78,
        d_model=32,
        nhead=4,
        nlayers=2,
        dim_feedforward=64,
        dropout=0.1,
        max_len=50,
        n_phones=39,
    )
    scaler = FeatureScaler(np.zeros(78), np.ones(78))
    path = tmp_path / "e16" / "transformer_ckpt.pt"
    save_checkpoint(
        path,
        experiment_id="E16",
        architecture="transformer",
        model=model,
        scaler=scaler,
        model_kwargs={
            "input_dim": 78,
            "d_model": 32,
            "nhead": 4,
            "nlayers": 2,
            "dim_feedforward": 64,
            "dropout": 0.1,
            "max_len": 50,
            "n_phones": 39,
        },
        feature_set="c8_lpp_lpr_embed",
        fit={"best_epoch": 14, "epochs_ran": 22, "best_val_mse": 0.05},
        protocol={"seed": 0},
    )
    loaded = load_checkpoint(path, device="cpu")
    assert loaded["experiment_id"] == "E16"
    assert loaded["architecture"] == "transformer"
    assert loaded["feature_set"] == "c8_lpp_lpr_embed"
    assert isinstance(loaded["model"], PhoneTransformer)
    assert loaded["fit"]["best_epoch"] == 14
    model.eval()
    x = torch.randn(1, 5, 78)
    pad = torch.zeros(1, 5, dtype=torch.bool)
    phones = torch.zeros(1, 5, dtype=torch.long)
    with torch.no_grad():
        y0 = model(x, pad, phones)
        y1 = loaded["model"](x, pad, phones)
    assert torch.allclose(y0, y1)


def test_save_load_mlp_roundtrip(tmp_path: Path) -> None:
    model = PhoneMLP(input_dim=3, hidden_dim=32, n_phones=None)
    scaler = FeatureScaler(np.zeros(3), np.ones(3))
    path = default_checkpoint_path(tmp_path, "E1", "mlp")
    save_checkpoint(
        path,
        experiment_id="E1",
        architecture="mlp",
        model=model,
        scaler=scaler,
        model_kwargs={"input_dim": 3, "hidden_dim": 32, "n_phones": None},
        feature_set="b4",
    )
    loaded = load_checkpoint(path)
    assert isinstance(loaded["model"], PhoneMLP)
    rebuilt = build_model("mlp", loaded["model_kwargs"])
    assert isinstance(rebuilt, PhoneMLP)
    assert path == tmp_path / "e1" / "mlp_ckpt.pt"


def test_group_e_checkpoint_dir(tmp_path: Path) -> None:
    from gop_empirical.experiment import _group_e_checkpoint_dir

    cfg = {"paths": {"checkpoint_dir": "checkpoints"}}
    assert _group_e_checkpoint_dir(cfg, tmp_path) == (tmp_path / "checkpoints").resolve()
    assert _group_e_checkpoint_dir({}, tmp_path) == (tmp_path / "checkpoints").resolve()
