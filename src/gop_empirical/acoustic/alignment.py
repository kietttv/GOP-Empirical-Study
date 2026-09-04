"""Kaldi CTM alignment helpers for Group C SSL GOP (same segments as A/B keys)."""

from __future__ import annotations

import re
import wave
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import numpy as np

from gop_empirical.acoustic.phones import PhoneInventory

# Speechocean762 wavs are often shorter than Kaldi CTM (GOPT enrich script).
CTM_TOL_S = 0.02
CTM_UNDER_RATIO = 0.8
_POS_MARKER_RE = re.compile(r"_(B|E|I|S)$")
_STRESS_RE = re.compile(r"[_\d].*")


def parse_ctm(path: str | Path) -> OrderedDict[str, list[tuple[float, float, str]]]:
    """Kaldi CTM: utt channel start dur phone -> OrderedDict[utt] = [(start, dur, phone)]."""
    segs: OrderedDict[str, list[tuple[float, float, str]]] = OrderedDict()
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 5:
                raise ValueError(f"{path}:{line_no}: expected >=5 CTM fields, got {parts!r}")
            utt, _ch, start, dur, phone = parts[0], parts[1], float(parts[2]), float(parts[3]), parts[4]
            segs.setdefault(utt, []).append((start, dur, phone))
    return segs


def clean_phone(symbol: str, inventory: PhoneInventory | None = None) -> str | None:
    """Strip Kaldi position markers and stress; return None for silence."""
    base = _POS_MARKER_RE.sub("", str(symbol))
    stripped = _STRESS_RE.sub("", base)
    if inventory is not None and inventory.is_skip(stripped):
        return None
    skip = {"SIL", "SPN", "NSN", "<eps>", "<UNK>", "sil", "spn", "nsn"}
    if stripped in skip or base in skip:
        return None
    return stripped


def drop_silence(
    segs: Sequence[tuple[float, float, str]],
    inventory: PhoneInventory | None = None,
) -> list[tuple[float, float, str]]:
    out: list[tuple[float, float, str]] = []
    for start, dur, phone in segs:
        cleaned = clean_phone(phone, inventory)
        if cleaned is None:
            continue
        out.append((float(start), float(dur), cleaned))
    return out


def scale_segments(
    segs: Sequence[tuple[float, float, str]],
    dur_s: float,
    *,
    tol: float = CTM_TOL_S,
    under_ratio: float = CTM_UNDER_RATIO,
) -> tuple[list[tuple[float, float, str]], float]:
    """Map CTM (start, dur, phone) onto wav duration when clocks disagree."""
    if not segs or dur_s <= 0:
        return list(segs), 1.0
    ctm_end = max(s + d for s, d, _ in segs)
    if ctm_end <= 0:
        return list(segs), 1.0
    if ctm_end > dur_s + tol or ctm_end < under_ratio * dur_s:
        scale = float(dur_s / ctm_end)
        return [(s * scale, d * scale, p) for s, d, p in segs], scale
    return list(segs), 1.0


def wav_duration_s(path: str | Path) -> float:
    with wave.open(str(path), "rb") as w:
        n, sr = w.getnframes(), w.getframerate()
        if sr <= 0:
            raise ValueError(f"bad sample rate in {path}")
        return float(n) / float(sr)


def find_wav(utt_id: str, wav_dir: str | Path) -> Path:
    wav_dir = Path(wav_dir)
    for ext in (".wav", ".WAV"):
        p = wav_dir / f"{utt_id}{ext}"
        if p.is_file():
            return p
    matches = list(wav_dir.rglob(f"{utt_id}.wav")) + list(wav_dir.rglob(f"{utt_id}.WAV"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"wav not found for utt_id={utt_id} under {wav_dir}")


def time_to_frame_span(
    start_s: float,
    end_s: float,
    *,
    hop_s: float,
    n_frames: int,
) -> tuple[int, int]:
    """Half-open frame span ``[i0, i1)`` clipped to ``[0, n_frames]``.

    Empty span (i1 <= i0) means the segment did not cover any frame after rescale.
    """
    if hop_s <= 0:
        raise ValueError(f"hop_s must be positive, got {hop_s}")
    if n_frames < 0:
        raise ValueError(f"n_frames must be >= 0, got {n_frames}")
    i0 = int(np.floor(start_s / hop_s))
    i1 = int(np.ceil(end_s / hop_s))
    i0 = max(0, min(i0, n_frames))
    i1 = max(0, min(i1, n_frames))
    return i0, i1


def encoder_hop_samples(conv_stride: Sequence[int]) -> int:
    hop = 1
    for s in conv_stride:
        hop *= int(s)
    return hop


def keys_to_utt_order(keys: Sequence[str]) -> tuple[list[str], OrderedDict[str, int]]:
    """``utt_id.phn_idx`` keys -> first-seen utt order and phone counts."""
    order: list[str] = []
    counts: OrderedDict[str, int] = OrderedDict()
    for key in keys:
        utt, idx_s = str(key).rsplit(".", 1)
        idx = int(idx_s)
        if utt not in counts:
            order.append(utt)
            counts[utt] = 0
        if idx != counts[utt]:
            raise ValueError(
                f"non-contiguous phone index for {utt}: got {idx}, expected {counts[utt]}"
            )
        counts[utt] += 1
    return order, counts


def align_ctm_to_keys(
    ctm_segs: OrderedDict[str, list[tuple[float, float, str]]],
    keys: Sequence[str],
    *,
    inventory: PhoneInventory | None = None,
    wav_dir: str | Path | None = None,
    durations: dict[str, float] | None = None,
) -> tuple[list[tuple[str, int, str, float, float]], dict[str, int]]:
    """One (key, phone, start, end) per ``utt_id.phn_idx``, silence dropped, CTM rescaled.

    Returns rows ``(utt_id, phn_idx, phone, start_s, end_s)`` and stats
    ``n_scaled``, ``n_empty_after_scale`` (always 0 here; empty frames are counted later).
    """
    order, counts = keys_to_utt_order(keys)
    durations = dict(durations or {})
    rows: list[tuple[str, int, str, float, float]] = []
    n_scaled = 0
    for utt in order:
        if utt not in ctm_segs:
            raise KeyError(f"CTM missing utterance {utt}")
        cleaned = drop_silence(ctm_segs[utt], inventory)
        if len(cleaned) != counts[utt]:
            raise ValueError(
                f"{utt}: CTM has {len(cleaned)} non-silence phones, keys expect {counts[utt]}"
            )
        dur_s = durations.get(utt)
        if dur_s is None and wav_dir is not None:
            dur_s = wav_duration_s(find_wav(utt, wav_dir))
            durations[utt] = dur_s
        if dur_s is None:
            scaled, scale = list(cleaned), 1.0
        else:
            scaled, scale = scale_segments(cleaned, float(dur_s))
        if abs(scale - 1.0) > 1e-6:
            n_scaled += 1
        for phn_idx, (start, dur, phone) in enumerate(scaled):
            rows.append((utt, phn_idx, phone, float(start), float(start + dur)))
    stats = {"n_scaled": int(n_scaled), "n_utts": int(len(order))}
    return rows, stats
