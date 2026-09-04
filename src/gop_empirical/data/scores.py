"""Speechocean762 human phoneme scores (same key scheme as GOPT / Xiaomi)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def round_score(score: float, floor: float = 0.1, min_val: float = 0.0, max_val: float = 2.0) -> float:
    score = max(min(max_val, float(score)), min_val)
    if floor <= 0:
        return score
    return round(score / floor) * floor


def load_human_scores(path: str | Path, floor: float = 0.1) -> dict[str, dict[str, Any]]:
    """Map ``utt_id.phn_idx`` → phone, human score, word_id, phone_id.

    Phone symbols drop stress / diacritics the same way as Xiaomi
    ``load_human_scores``: ``IY0`` → ``IY``.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        info = json.load(f)

    records: dict[str, dict[str, Any]] = {}
    for utt, utt_info in info.items():
        phone_num = 0
        for word_id, word in enumerate(utt_info["words"]):
            phones = word["phones"]
            accuracies = word["phones-accuracy"]
            if len(phones) != len(accuracies):
                raise ValueError(
                    f"{utt} word {word_id}: phones ({len(phones)}) != "
                    f"phones-accuracy ({len(accuracies)})"
                )
            for phone_in_word, (phone, acc) in enumerate(zip(phones, accuracies)):
                key = f"{utt}.{phone_num}"
                records[key] = {
                    "utt_id": utt,
                    "word_id": int(word_id),
                    "phone_id": int(phone_in_word),
                    "phone": re.sub(r"[_\d].*", "", str(phone)),
                    "human_score": round_score(acc, floor=floor),
                }
                phone_num += 1
    return records


def load_utterance_scores(path: str | Path) -> dict[str, dict[str, Any]]:
    """Map utterance id → sentence-level scores (accuracy 0–10, etc.).

    Does not touch phoneme scores; Group D uses ``accuracy`` for speaker strata.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        info = json.load(f)

    records: dict[str, dict[str, Any]] = {}
    for utt, utt_info in info.items():
        if not isinstance(utt_info, dict):
            raise ValueError(f"{utt}: expected a mapping of sentence scores")
        if "accuracy" not in utt_info:
            raise ValueError(f"{utt}: missing sentence-level accuracy")
        records[str(utt)] = {
            "utt_id": str(utt),
            "accuracy": float(utt_info["accuracy"]),
            "fluency": float(utt_info["fluency"]) if "fluency" in utt_info else None,
            "prosodic": float(utt_info["prosodic"]) if "prosodic" in utt_info else None,
            "completeness": float(utt_info["completeness"]) if "completeness" in utt_info else None,
        }
    return records
