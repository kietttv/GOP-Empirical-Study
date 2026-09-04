"""Speechocean762 speaker metadata (Kaldi-style utt2spk / spk2age / spk2gender)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _normalize_utt_id(utt_id: str) -> str:
    text = str(utt_id).strip()
    if text.isdigit():
        return text.zfill(9)
    return text


def _normalize_speaker_id(speaker: str) -> str:
    text = str(speaker).strip()
    if text.isdigit():
        return text.zfill(4)
    return text


def _parse_kaldi_map(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"{path}:{line_no}: expected 2 fields, got {parts!r}")
            mapping[parts[0]] = parts[1]
    return mapping


def load_split_utt2spk(split_dir: str | Path) -> dict[str, str]:
    path = Path(split_dir) / "utt2spk"
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = _parse_kaldi_map(path)
    return {_normalize_utt_id(utt): _normalize_speaker_id(spk) for utt, spk in raw.items()}


def load_split_spk2field(split_dir: str | Path, filename: str) -> dict[str, str]:
    path = Path(split_dir) / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = _parse_kaldi_map(path)
    return {_normalize_speaker_id(spk): value for spk, value in raw.items()}


def load_speaker_metadata(speechocean_dir: str | Path) -> dict[str, Any]:
    """Load train+test utt2spk, spk2age, spk2gender.

    Speaker ids are zero-padded to 4 digits; utterance ids to 9 digits.
    """
    root = Path(speechocean_dir)
    utt2spk: dict[str, str] = {}
    spk2age: dict[str, int] = {}
    spk2gender: dict[str, str] = {}
    speakers_by_split: dict[str, set[str]] = {}
    for split in ("train", "test"):
        split_dir = root / split
        split_utt2spk = load_split_utt2spk(split_dir)
        overlap = set(utt2spk).intersection(split_utt2spk)
        if overlap:
            raise ValueError(f"utt_id overlap between splits: {sorted(overlap)[:5]}")
        utt2spk.update(split_utt2spk)
        speakers_by_split[split] = set(split_utt2spk.values())
        for spk, age_raw in load_split_spk2field(split_dir, "spk2age").items():
            spk2age[spk] = int(age_raw)
        for spk, gender in load_split_spk2field(split_dir, "spk2gender").items():
            spk2gender[spk] = str(gender).strip().lower()
    return {
        "utt2spk": utt2spk,
        "spk2age": spk2age,
        "spk2gender": spk2gender,
        "speakers_train": speakers_by_split["train"],
        "speakers_test": speakers_by_split["test"],
        "speaker_overlap": speakers_by_split["train"] & speakers_by_split["test"],
    }
