"""CMU 39-phone inventory → Facebook espeak CTC token ids (Group C C8/C9).

Off-the-shelf ``wav2vec2-*-espeak-cv-ft`` models emit IPA/espeak symbols, not
CMU ids. Cao GOP-S needs a 1-1 sequence id per canonical phone. This module
loads the frozen map in ``data/cmu_to_espeak.json`` and resolves ids against a
tokenizer vocab. Missing tokens fail loudly — no silent fallback.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_MAP = Path(__file__).resolve().parents[3] / "data" / "cmu_to_espeak.json"
DEFAULT_VOCAB = Path(__file__).resolve().parents[3] / "data" / "espeak_ctc_vocab.json"


def load_cmu_to_espeak(
    path: str | Path | None = None,
    *,
    scored_symbols: Sequence[str] | None = None,
) -> dict[str, str]:
    """Return ``{CMU: espeak_token}`` for exactly the 39 scored phones."""
    path = Path(path) if path is not None else DEFAULT_MAP
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "map" in payload:
        raw = payload["map"]
    elif isinstance(payload, dict):
        raw = {k: v for k, v in payload.items() if not str(k).startswith("_") and k not in {"description", "notes", "vocab_file"}}
    else:
        raise ValueError(f"cmu_to_espeak must be an object: {path}")
    mapping = {str(k): str(v) for k, v in raw.items()}
    if scored_symbols is not None:
        scored = [str(s) for s in scored_symbols]
        missing = [s for s in scored if s not in mapping]
        extra = sorted(set(mapping) - set(scored))
        if missing:
            raise ValueError(f"{path}: map missing scored CMU phones {missing}")
        if extra:
            raise ValueError(f"{path}: map has extra CMU phones {extra}")
        mapping = {s: mapping[s] for s in scored}
    if len(mapping) != 39:
        raise ValueError(f"{path}: expected 39 CMU phones, got {len(mapping)}")
    empty = [k for k, v in mapping.items() if not v]
    if empty:
        raise ValueError(f"{path}: empty espeak token for {empty}")
    ipas = list(mapping.values())
    if len(ipas) != len(set(ipas)):
        seen: dict[str, str] = {}
        dupes = []
        for cmu, ipa in mapping.items():
            if ipa in seen:
                dupes.append(f"{seen[ipa]}/{cmu}->{ipa}")
            else:
                seen[ipa] = cmu
        raise ValueError(f"{path}: espeak tokens are not 1-1: {dupes}")
    return mapping


def load_espeak_vocab(path: str | Path | None = None) -> dict[str, int]:
    path = Path(path) if path is not None else DEFAULT_VOCAB
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"espeak vocab must be an object: {path}")
    return {str(k): int(v) for k, v in payload.items()}


def ctc_ids_from_vocab(
    cmu_to_ipa: Mapping[str, str],
    vocab: Mapping[str, int],
) -> dict[str, int]:
    """Map CMU symbols to CTC ids. Raises if any IPA token is absent."""
    missing = [f"{cmu}->{ipa}" for cmu, ipa in cmu_to_ipa.items() if ipa not in vocab]
    if missing:
        raise KeyError(
            "espeak CTC vocab is missing mapped tokens (no silent fallback): "
            + ", ".join(missing)
        )
    return {cmu: int(vocab[ipa]) for cmu, ipa in cmu_to_ipa.items()}


def ctc_ids_from_tokenizer(
    cmu_to_ipa: Mapping[str, str],
    tokenizer: Any,
) -> dict[str, int]:
    """Resolve ids via ``tokenizer.get_vocab()`` (HuggingFace)."""
    if not hasattr(tokenizer, "get_vocab"):
        raise TypeError("tokenizer must provide get_vocab()")
    vocab = {str(k): int(v) for k, v in tokenizer.get_vocab().items()}
    unk_id = getattr(tokenizer, "unk_token_id", None)
    ids = ctc_ids_from_vocab(cmu_to_ipa, vocab)
    if unk_id is not None:
        collisions = [cmu for cmu, tid in ids.items() if int(tid) == int(unk_id) and cmu_to_ipa[cmu] != getattr(tokenizer, "unk_token", None)]
        if collisions:
            raise KeyError(
                "mapped espeak tokens resolved to unk_token_id: "
                + ", ".join(f"{c}->{cmu_to_ipa[c]}" for c in collisions)
            )
    return ids


def blank_id_from_tokenizer(tokenizer: Any) -> int:
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is not None:
        return int(pad)
    vocab = tokenizer.get_vocab()
    if "<pad>" in vocab:
        return int(vocab["<pad>"])
    raise ValueError("espeak tokenizer has no pad/blank id")


def blank_id_from_vocab(vocab: Mapping[str, int]) -> int:
    if "<pad>" not in vocab:
        raise KeyError("espeak vocab has no <pad> token")
    return int(vocab["<pad>"])
