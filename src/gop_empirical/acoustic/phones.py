"""Phone inventory: 42 Kaldi slots <-> 39 CMU scored phones + SSL CTC ids."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_INVENTORY = Path(__file__).resolve().parents[3] / "data" / "phone_inventory.json"


class PhoneInventory:
    """Frozen mapping used by Group C so Kaldi and SSL GOP share one phone set."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.raw = payload
        self.skip_phones = {str(p) for p in payload.get("skip_phones", [])}
        self.ctc_blank_tokens = {str(p) for p in payload.get("ctc_blank_tokens", [])}

        self.kaldi_by_id: dict[int, dict[str, Any]] = {}
        self.kaldi_symbol_to_id: dict[str, int] = {}
        for row in payload["kaldi"]:
            kid = int(row["id"])
            sym = str(row["symbol"])
            self.kaldi_by_id[kid] = dict(row)
            self.kaldi_symbol_to_id[sym] = kid

        self.ssl_by_id: dict[int, dict[str, Any]] = {}
        self.ssl_symbol_to_id: dict[str, int] = {}
        for row in payload["ssl"]:
            sid = int(row["id"])
            sym = str(row["symbol"])
            self.ssl_by_id[sid] = dict(row)
            self.ssl_symbol_to_id[sym] = sid

        scored = [r for r in payload["ssl"] if not r.get("skip") and not r.get("blank")]
        self.scored_symbols = [str(r["symbol"]) for r in scored]
        self.ssl_scored_ids = [int(r["id"]) for r in scored]
        self.n_kaldi_slots = int(payload.get("n_kaldi_slots", len(payload["kaldi"])))
        self.n_scored_phones = int(payload.get("n_scored_phones", len(self.scored_symbols)))

    def is_skip(self, symbol: str) -> bool:
        return str(symbol) in self.skip_phones

    def kaldi_symbol(self, kaldi_id: int) -> str:
        row = self.kaldi_by_id.get(int(kaldi_id))
        if row is None:
            raise KeyError(f"unknown Kaldi phone id {kaldi_id}")
        return str(row["symbol"])

    def ssl_index(self, symbol: str) -> int:
        """Index into the 39-d scored SSL posterior (not the raw CTC id)."""
        if symbol not in self.ssl_symbol_to_id:
            raise KeyError(f"unknown SSL phone symbol {symbol!r}")
        raw_id = self.ssl_symbol_to_id[symbol]
        try:
            return self.ssl_scored_ids.index(raw_id)
        except ValueError as exc:
            raise KeyError(f"{symbol!r} is not a scored SSL phone") from exc

    def ssl_ctc_id(self, symbol: str) -> int:
        if symbol not in self.ssl_symbol_to_id:
            raise KeyError(f"unknown SSL phone symbol {symbol!r}")
        return int(self.ssl_symbol_to_id[symbol])

    def blank_ctc_ids(self) -> list[int]:
        ids = []
        for sid, row in self.ssl_by_id.items():
            if row.get("blank") or row.get("skip") or str(row["symbol"]) in self.ctc_blank_tokens:
                ids.append(int(sid))
        return ids


def load_phone_inventory(path: str | Path | None = None) -> PhoneInventory:
    path = Path(path) if path is not None else DEFAULT_INVENTORY
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"phone inventory must be an object: {path}")
    inv = PhoneInventory(payload)
    if inv.n_kaldi_slots != 42:
        raise ValueError(f"expected 42 Kaldi slots, got {inv.n_kaldi_slots}")
    if inv.n_scored_phones != 39 or len(inv.scored_symbols) != 39:
        raise ValueError(f"expected 39 scored phones, got {inv.n_scored_phones}")
    return inv
