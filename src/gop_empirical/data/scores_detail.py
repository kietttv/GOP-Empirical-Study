"""Parse Speechocean762 scores-detail.json expert phone markup."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

# Token examples: EH0, {AO0}, (R), [L], AH
_TOKEN_RE = re.compile(
    r"\{([A-Z]{1,3}\d?)\}|\(([A-Z]{1,3}\d?)\)|\[([A-Z]{1,3}\d?)\]|([A-Z]{1,3}\d?)"
)


def strip_stress(phone: str) -> str:
    phone = str(phone).strip().upper()
    if phone and phone[-1].isdigit():
        return phone[:-1]
    return phone


def parse_expert_phone_string(text: str) -> list[dict[str, Any]]:
    """Parse one expert's phone string into canonical-aligned tokens + insertions.

    Returns a list of events in order. Canonical slots have ``kind`` in
    ``{correct, accent, incorrect}``. Insertions have ``kind == "insertion"``.
    """
    events: list[dict[str, Any]] = []
    for match in _TOKEN_RE.finditer(str(text)):
        accent, incorrect, insertion, correct = match.groups()
        if accent is not None:
            events.append({"phone": strip_stress(accent), "kind": "accent", "raw": match.group(0)})
        elif incorrect is not None:
            events.append(
                {"phone": strip_stress(incorrect), "kind": "incorrect", "raw": match.group(0)}
            )
        elif insertion is not None:
            events.append(
                {"phone": strip_stress(insertion), "kind": "insertion", "raw": match.group(0)}
            )
        else:
            events.append(
                {"phone": strip_stress(correct), "kind": "correct", "raw": match.group(0)}
            )
    return events


def canonical_slot_flags(events: list[dict[str, Any]]) -> list[dict[str, bool]]:
    """Map parse events onto canonical phone slots (insertions do not consume a slot).

    Leading ``[PH]`` insertions attach to the *next* canonical phone via
    ``any_insertion_before`` — they must not invent a fake slot (that shifts
    accent/incorrect flags off-by-one).
    """
    slots: list[dict[str, bool]] = []
    pending_before = False
    for ev in events:
        if ev["kind"] == "insertion":
            if slots:
                slots[-1]["any_insertion_after"] = True
            else:
                pending_before = True
            continue
        slots.append(
            {
                "phone": ev["phone"],
                "any_accent": ev["kind"] == "accent",
                "any_incorrect": ev["kind"] == "incorrect",
                "any_insertion_after": False,
                "any_insertion_before": pending_before,
            }
        )
        pending_before = False
    return slots


def aggregate_expert_slots(
    expert_strings: list[str],
    *,
    n_canonical: int | None = None,
) -> list[dict[str, Any]]:
    """OR accent/incorrect/insertion flags across experts for each canonical index."""
    per_expert = [canonical_slot_flags(parse_expert_phone_string(s)) for s in expert_strings]
    if n_canonical is None:
        n_canonical = max((len(s) for s in per_expert), default=0)
    out: list[dict[str, Any]] = []
    for i in range(int(n_canonical)):
        accent = False
        incorrect = False
        insertion = False
        phone = None
        n_accent = 0
        n_incorrect = 0
        for slots in per_expert:
            if i >= len(slots):
                continue
            slot = slots[i]
            phone = phone or slot.get("phone")
            if slot.get("any_accent"):
                accent = True
                n_accent += 1
            if slot.get("any_incorrect"):
                incorrect = True
                n_incorrect += 1
            if slot.get("any_insertion_after") or slot.get("any_insertion_before"):
                insertion = True
        out.append(
            {
                "phone": phone,
                "any_accent": accent,
                "any_incorrect": incorrect,
                "any_insertion": insertion,
                "n_experts_accent": n_accent,
                "n_experts_incorrect": n_incorrect,
            }
        )
    return out


def load_scores_detail(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def phone_markup_table(scores_detail: dict[str, Any]) -> pd.DataFrame:
    """Flatten scores-detail to one row per (utt_id, word_id, phone_id)."""
    rows: list[dict[str, Any]] = []
    for utt_id, utt in scores_detail.items():
        words = utt.get("words") or []
        for word_id, word in enumerate(words):
            ref = str(word.get("ref-phones") or word.get("phones") or "")
            if isinstance(word.get("phones"), list):
                expert_strings = [str(x) for x in word["phones"]]
            else:
                expert_strings = [ref]
            ref_phones = [strip_stress(p) for p in ref.split() if p]
            slots = aggregate_expert_slots(expert_strings, n_canonical=len(ref_phones))
            for phone_id, (ref_ph, slot) in enumerate(zip(ref_phones, slots)):
                rows.append(
                    {
                        "utt_id": str(utt_id),
                        "word_id": int(word_id),
                        "phone_id": int(phone_id),
                        "phone": ref_ph,
                        "any_accent": bool(slot["any_accent"]),
                        "any_incorrect": bool(slot["any_incorrect"]),
                        "any_insertion": bool(slot["any_insertion"]),
                        "n_experts_accent": int(slot["n_experts_accent"]),
                        "n_experts_incorrect": int(slot["n_experts_incorrect"]),
                    }
                )
    return pd.DataFrame(rows)
