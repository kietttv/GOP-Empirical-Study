"""E2: per-phone Transformer encoder on a GOP feature sequence."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoid_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """[1, max_len, d_model] sinusoid table (GOPT-style positions, not GOPT input)."""
    pe = torch.zeros(int(max_len), int(d_model))
    position = torch.arange(0, int(max_len), dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, int(d_model), 2, dtype=torch.float32)
        * (-math.log(10000.0) / float(d_model))
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe.unsqueeze(0)


class PhoneTransformer(nn.Module):
    """Project GOP features → Transformer encoder → per-phone score.

    Same per-phone feature as E1. No CLS. E2/E8 use GOP-only features.
    E14/E16/E18 may add a canonical-phone embedding after ``in_proj`` (GOPT-style).
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 32,
        nhead: int = 4,
        nlayers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
        max_len: int = 50,
        n_phones: int | None = None,
    ) -> None:
        super().__init__()
        if int(input_dim) < 1:
            raise ValueError("input_dim must be >= 1")
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.n_phones = None if n_phones is None else int(n_phones)
        self.pad_phone_id = None if self.n_phones is None else self.n_phones
        self.in_proj = nn.Linear(self.input_dim, self.d_model)
        self.phone_embed: nn.Embedding | None
        if self.n_phones is not None:
            if self.n_phones < 1:
                raise ValueError("n_phones must be >= 1")
            self.phone_embed = nn.Embedding(
                self.n_phones + 1, self.d_model, padding_idx=self.pad_phone_id
            )
        else:
            self.phone_embed = None
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            batch_first=True,
        )
        try:
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=int(nlayers), enable_nested_tensor=False
            )
        except TypeError:
            self.encoder = nn.TransformerEncoder(layer, num_layers=int(nlayers))
        self.head = nn.Linear(self.d_model, 1)
        self.register_buffer("pos", sinusoid_encoding(int(max_len), self.d_model), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        phone_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``x`` is [B, T, D]; ``pad_mask`` is True on pad slots."""
        seq_len = x.size(1)
        h = self.in_proj(x)
        if self.phone_embed is not None:
            if phone_ids is None:
                raise ValueError("PhoneTransformer with n_phones requires phone_ids")
            h = h + self.phone_embed(phone_ids.long())
        elif phone_ids is not None:
            raise ValueError("phone_ids passed to PhoneTransformer without n_phones")
        h = h + self.pos[:, :seq_len, :]
        encoded = self.encoder(h, src_key_padding_mask=pad_mask)
        return self.head(encoded).squeeze(-1)
