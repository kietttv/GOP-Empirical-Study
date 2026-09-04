"""E1: phone-independent MLP on GOP features."""

from __future__ import annotations

import torch
import torch.nn as nn


class PhoneMLP(nn.Module):
    """Linear → ReLU → Linear → ReLU → Linear (Experimental List E1).

    Optional canonical-phone embedding (E13/E15/E17) is added after the first linear,
    GOPT-style: projected GOP + phone identity, then the same ReLU stack.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        n_phones: int | None = None,
    ) -> None:
        super().__init__()
        if int(input_dim) < 1:
            raise ValueError("input_dim must be >= 1")
        hidden = int(hidden_dim)
        self.n_phones = None if n_phones is None else int(n_phones)
        self.fc1 = nn.Linear(int(input_dim), hidden)
        self.phone_embed: nn.Embedding | None
        if self.n_phones is not None:
            if self.n_phones < 1:
                raise ValueError("n_phones must be >= 1")
            self.phone_embed = nn.Embedding(self.n_phones, hidden)
        else:
            self.phone_embed = None
        self.rest = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, phone_ids: torch.Tensor | None = None) -> torch.Tensor:
        h = self.fc1(x)
        if self.phone_embed is not None:
            if phone_ids is None:
                raise ValueError("PhoneMLP with n_phones requires phone_ids")
            h = h + self.phone_embed(phone_ids.long())
        elif phone_ids is not None:
            raise ValueError("phone_ids passed to PhoneMLP without n_phones")
        return self.rest(h).squeeze(-1)
