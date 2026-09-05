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


def test_architecture_for_experiment() -> None:
    from gop_empirical.data.learned import architecture_for_experiment

    assert architecture_for_experiment("E1") == "mlp"
    assert architecture_for_experiment("e2") == "transformer"
    assert architecture_for_experiment("E15") == "mlp"
    assert architecture_for_experiment("E16") == "transformer"


def test_score_group_e_table_mlp(tmp_path: Path) -> None:
    import pandas as pd
    from gop_empirical.eval.metrics import evaluate_predictions
    from gop_empirical.scoring.eval_checkpoint import score_group_e_table

    rng = np.random.RandomState(0)
    model = PhoneMLP(input_dim=3, hidden_dim=8, n_phones=None)
    scaler = FeatureScaler(np.zeros(3), np.ones(3))
    path = default_checkpoint_path(tmp_path, "E1", "mlp")
    save_checkpoint(
        path,
        experiment_id="E1",
        architecture="mlp",
        model=model,
        scaler=scaler,
        model_kwargs={"input_dim": 3, "hidden_dim": 8, "n_phones": None},
        feature_set="b4",
    )
    ckpt = load_checkpoint(path, device="cpu")
    n = 12
    table = pd.DataFrame(
        {
            "utt_id": ["u1"] * 6 + ["u2"] * 6,
            "split": ["test"] * n,
            "word_id": np.zeros(n, dtype=np.int64),
            "phone_id": list(range(6)) * 2,
            "phone": ["AH"] * n,
            "human_score": rng.uniform(0.0, 2.0, n),
            "feat_lpp_canonical": rng.randn(n),
            "feat_lpp_max_competitor": rng.randn(n),
            "feat_lpr": rng.randn(n),
        }
    )
    scored, pred = score_group_e_table(
        ckpt, table, clip=(0.0, 2.0), device=torch.device("cpu")
    )
    assert len(scored) == n
    assert pred.shape == (n,)
    metrics = evaluate_predictions(pred, scored["human_score"].to_numpy(), clip=(0.0, 2.0))
    assert metrics["n"] == n
    assert np.isfinite(metrics["mse"])


def test_score_group_e_table_transformer(tmp_path: Path) -> None:
    import pandas as pd
    from gop_empirical.scoring.eval_checkpoint import score_group_e_table

    rng = np.random.RandomState(1)
    model = PhoneTransformer(
        input_dim=3,
        d_model=8,
        nhead=2,
        nlayers=1,
        dim_feedforward=16,
        dropout=0.0,
        max_len=8,
        n_phones=None,
    )
    scaler = FeatureScaler(np.zeros(3), np.ones(3))
    path = default_checkpoint_path(tmp_path, "E2", "transformer")
    save_checkpoint(
        path,
        experiment_id="E2",
        architecture="transformer",
        model=model,
        scaler=scaler,
        model_kwargs={
            "input_dim": 3,
            "d_model": 8,
            "nhead": 2,
            "nlayers": 1,
            "dim_feedforward": 16,
            "dropout": 0.0,
            "max_len": 8,
            "n_phones": None,
        },
        feature_set="b4",
    )
    ckpt = load_checkpoint(path, device="cpu")
    n = 8
    table = pd.DataFrame(
        {
            "utt_id": ["u1"] * 4 + ["u2"] * 4,
            "split": ["test"] * n,
            "word_id": np.zeros(n, dtype=np.int64),
            "phone_id": list(range(4)) * 2,
            "phone": ["AH"] * n,
            "human_score": rng.uniform(0.0, 2.0, n),
            "feat_lpp_canonical": rng.randn(n),
            "feat_lpp_max_competitor": rng.randn(n),
            "feat_lpr": rng.randn(n),
        }
    )
    scored, pred = score_group_e_table(
        ckpt, table, clip=(0.0, 2.0), device=torch.device("cpu")
    )
    assert pred.shape == (n,)
    assert np.isfinite(pred).all()
    assert len(scored) == n


def test_eval_rejects_checkpoint_with_multiple_ids() -> None:
    from gop_empirical.experiment import eval_group_e_checkpoints

    try:
        eval_group_e_checkpoints({}, ["E1", "E2"], checkpoint_path="x.pt")
    except ValueError as exc:
        assert "--checkpoint" in str(exc)
    else:
        raise AssertionError("expected ValueError")
