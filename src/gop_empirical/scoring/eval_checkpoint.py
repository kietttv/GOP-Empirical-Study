"""Score a Group E feature table from a saved MLP/Transformer checkpoint."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from gop_empirical.data.learned import (
    CANONICAL_PHONE_COL,
    attach_canonical_phone_index,
    feature_stored_columns,
    matrix_from_table,
    pack_utterances,
    phone_embed_spec,
    uses_phone_embed,
)
from gop_empirical.scoring.train import predict_mlp, predict_transformer


def score_group_e_table(
    checkpoint: dict[str, Any],
    table: pd.DataFrame,
    *,
    clip: tuple[float, float],
    device: torch.device,
    batch_size_mlp: int = 256,
    batch_size_tf: int = 32,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Score phones; return ``(ordered_table, clipped_pred)`` aligned row-wise."""
    fs = str(checkpoint["feature_set"]).strip().lower()
    arch = str(checkpoint["architecture"]).strip().lower()
    model = checkpoint["model"]
    scaler = checkpoint["scaler"]
    kwargs = dict(checkpoint.get("model_kwargs") or {})
    stored = list(feature_stored_columns(fs))
    work = table.sort_values(["utt_id", "word_id", "phone_id"], kind="mergesort").reset_index(
        drop=True
    )
    n_phones = None
    pack_phone_kw: dict[str, Any] = {}
    phone_idx = None
    if uses_phone_embed(fs):
        spec = phone_embed_spec(fs)
        n_phones = int(spec["n_phones"])
        work = attach_canonical_phone_index(
            work, n_phones=n_phones, space=str(spec["space"])
        )
        phone_idx = work[CANONICAL_PHONE_COL].to_numpy(dtype=np.int64)
        pack_phone_kw = {
            "phone_idx_col": CANONICAL_PHONE_COL,
            "pad_phone_id": n_phones,
        }
    scaled = scaler.transform(matrix_from_table(work, stored))
    model.to(device)
    model.eval()
    if arch == "mlp":
        pred = predict_mlp(
            model,
            scaled,
            batch_size=int(batch_size_mlp),
            device=device,
            phone_ids=phone_idx,
        )
    elif arch == "transformer":
        max_len = int(kwargs.get("max_len", 50))
        observed = int(work.groupby("utt_id").size().max()) if len(work) else 1
        if observed > max_len:
            raise ValueError(
                f"longest utterance has {observed} phones > checkpoint max_len={max_len}"
            )
        scaled_df = work.copy()
        for i, col in enumerate(stored):
            scaled_df[col] = scaled[:, i]
        packed = pack_utterances(
            scaled_df, stored, max_seq_len=max_len, **pack_phone_kw
        )
        pred = predict_transformer(
            model,
            packed,
            batch_size=int(batch_size_tf),
            device=device,
            n_rows=len(work),
        )
    else:
        raise ValueError(f"unknown architecture {arch!r}")
    return work, np.clip(pred, float(clip[0]), float(clip[1]))
