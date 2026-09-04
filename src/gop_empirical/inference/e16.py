"""E16 single-utterance inference: wav + transcript → phoneme scores."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from gop_empirical.acoustic.espeak_map import (
    blank_id_from_vocab,
    ctc_ids_from_vocab,
    load_cmu_to_espeak,
    load_espeak_vocab,
)
from gop_empirical.acoustic.phones import load_phone_inventory
from gop_empirical.acoustic.posterior import CtcPhonemePosterior
from gop_empirical.data.learned import SSL_LPP_LPR_STORED_COLUMNS, pack_utterances
from gop_empirical.gop.ctc_align import ctc_label_frames
from gop_empirical.gop.from_posterior import log_softmax_over_ids, log_softmax_rows
from gop_empirical.gop.representation import (
    SSL_LPP_LPR_N_FEATURES,
    lpp_lpr_concat,
    mean_lpp_on_frames,
)
from gop_empirical.scoring.checkpoint import load_checkpoint
from gop_empirical.scoring.train import predict_transformer

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
_STRESS_RE = re.compile(r"[0-2]$")
_SKIP = frozenset({"SIL", "SPN", "NSN", "<EPS>", ""})


def ensure_repo_hf_cache(package_root: Path | None = None) -> Path:
    root = package_root or PACKAGE_ROOT
    hf_home = root / "models" / ".hf_home"
    os.environ.setdefault("NO_PROXY", "*")
    os.environ.setdefault("no_proxy", "*")
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(hf_home / "transformers")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    nltk_dir = root / "models" / "_scratch" / "nltk"
    if nltk_dir.is_dir():
        os.environ["NLTK_DATA"] = str(nltk_dir)
    return hf_home


def _ensure_nltk_data(package_root: Path | None = None) -> None:
    """Point NLTK at repo-local data used by g2p_en (cmudict / tagger)."""
    root = package_root or PACKAGE_ROOT
    nltk_dir = root / "models" / "_scratch" / "nltk"
    if nltk_dir.is_dir():
        os.environ["NLTK_DATA"] = str(nltk_dir)
        try:
            import nltk

            if str(nltk_dir) not in nltk.data.path:
                nltk.data.path.insert(0, str(nltk_dir))
        except ImportError:
            pass


def strip_stress(symbol: str) -> str:
    s = str(symbol).strip().upper()
    return _STRESS_RE.sub("", s)


def text_to_cmu_phones(text: str, *, g2p: Any | None = None) -> list[str]:
    """Grapheme → CMU ARPAbet (no stress), filtered to inventory-compatible tokens.

    Requires ``g2p_en`` (``pip install g2p-en``).
    """
    _ensure_nltk_data()
    if g2p is None:
        try:
            from g2p_en import G2p
        except ImportError as exc:
            raise ImportError(
                "text_to_cmu_phones needs g2p-en. Install with: pip install g2p-en"
            ) from exc
        g2p = G2p()
    raw = g2p(str(text))
    phones: list[str] = []
    for tok in raw:
        if not isinstance(tok, str):
            continue
        t = tok.strip()
        if not t or t in {" ", ""}:
            continue
        # g2p_en emits spaces between words as " "
        if re.fullmatch(r"[^A-Za-z0-9]+", t):
            continue
        phone = strip_stress(t)
        if phone in _SKIP:
            continue
        phones.append(phone)
    return phones


def load_wav(path: str | Path, sampling_rate: int = 16000) -> np.ndarray:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ImportError("load_wav needs soundfile") from exc
    wav, sr = sf.read(str(path), always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if int(sr) != int(sampling_rate):
        try:
            import librosa
        except ImportError as exc:
            raise ImportError(
                f"wav is {sr} Hz; install librosa to resample to {sampling_rate}"
            ) from exc
        wav = librosa.resample(wav, orig_sr=int(sr), target_sr=int(sampling_rate))
    return np.asarray(wav, dtype=np.float32).reshape(-1)


def extract_78d_lpp_lpr(
    wav_path: str | Path,
    phones: list[str],
    *,
    package_root: Path | None = None,
    device: str | None = None,
    sampling_rate: int = 16000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return ``(feat[T,78], ssl_ids[T], n_frames[T], kept_phones)`` via CTC Viterbi.

    Uses XLSR-53 espeak CTC (C8 / E16 AM). Phones outside the 39 scored set are dropped.
    """
    root = package_root or PACKAGE_ROOT
    ensure_repo_hf_cache(root)
    inventory = load_phone_inventory(root / "data" / "phone_inventory.json")
    cmu_map = load_cmu_to_espeak(
        root / "data" / "cmu_to_espeak.json",
        scored_symbols=inventory.scored_symbols,
    )
    vocab = load_espeak_vocab(root / "data" / "espeak_ctc_vocab.json")
    ctc_lookup = ctc_ids_from_vocab(cmu_map, vocab)
    blank_id = blank_id_from_vocab(vocab)
    espeak_scored_ids = np.asarray(
        [ctc_lookup[sym] for sym in inventory.scored_symbols], dtype=np.int64
    )

    ckpt = root / "models" / "wav2vec2-xlsr-53-espeak-cv-ft"
    if not ckpt.is_dir():
        raise FileNotFoundError(
            f"missing AM {ckpt}; run: python scripts/download_espeak_ctc.py"
        )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    posterior = CtcPhonemePosterior(
        ckpt,
        inventory,
        processor_dir=ckpt,
        device=device,
        sampling_rate=int(sampling_rate),
        restrict_to_inventory=False,
    )

    kept: list[str] = []
    seq_ids: list[int] = []
    ssl_ids: list[int] = []
    for phone in phones:
        p = strip_stress(phone)
        if p in _SKIP:
            continue
        try:
            ssl_ids.append(int(inventory.ssl_index(p)))
            seq_ids.append(int(ctc_lookup[p]))
            kept.append(p)
        except KeyError:
            continue
    if not kept:
        raise ValueError("no scored CMU phones left after G2P / inventory filter")

    waveform = load_wav(wav_path, sampling_rate=sampling_rate)
    logits = posterior.frame_logits(waveform)
    ctc_probs = np.exp(log_softmax_rows(logits))
    log_p39 = log_softmax_over_ids(logits, espeak_scored_ids)
    aligned = ctc_label_frames(
        ctc_probs, np.asarray(seq_ids, dtype=np.int64), blank=int(blank_id)
    )

    feats = np.full((len(kept), SSL_LPP_LPR_N_FEATURES), np.nan, dtype=np.float64)
    n_frames = np.zeros(len(kept), dtype=np.int64)
    for i, can_idx in enumerate(ssl_ids):
        frames = aligned[i] if i < len(aligned) else np.zeros(0, dtype=np.int64)
        lpp_vec = mean_lpp_on_frames(log_p39, frames)
        if lpp_vec is None:
            continue
        feats[i] = lpp_lpr_concat(lpp_vec, can_idx).reshape(-1)
        n_frames[i] = int(frames.size)
    if not np.isfinite(feats).all():
        bad = [kept[i] for i in range(len(kept)) if not np.isfinite(feats[i]).all()]
        raise RuntimeError(f"empty CTC align / LPP for phones: {bad}")
    return feats, np.asarray(ssl_ids, dtype=np.int64), n_frames, kept


def score_e16(
    feat: np.ndarray,
    ssl_ids: np.ndarray,
    checkpoint: dict[str, Any] | str | Path,
    *,
    device: str | torch.device | None = None,
    clip: tuple[float, float] = (0.0, 2.0),
) -> np.ndarray:
    """Apply scaler + E16 Transformer; return per-phone scores clipped to ``clip``."""
    if not isinstance(checkpoint, dict):
        checkpoint = load_checkpoint(checkpoint, device=device or "cpu")
    model = checkpoint["model"]
    scaler = checkpoint["scaler"]
    kwargs = dict(checkpoint.get("model_kwargs") or {})
    max_len = int(kwargs.get("max_len", 50))
    n_phones = kwargs.get("n_phones")
    if n_phones is None:
        raise ValueError("E16 checkpoint must include n_phones for SSL embed")
    feat = np.asarray(feat, dtype=np.float64)
    ssl_ids = np.asarray(ssl_ids, dtype=np.int64).reshape(-1)
    if feat.ndim != 2 or feat.shape[1] != scaler.mean.size:
        raise ValueError(
            f"expected [T, {scaler.mean.size}] features, got {feat.shape}"
        )
    if feat.shape[0] != ssl_ids.size:
        raise ValueError("feat / ssl_ids length mismatch")
    if feat.shape[0] > max_len:
        raise ValueError(
            f"utterance has {feat.shape[0]} phones > max_len={max_len}; "
            "truncate or retrain with a larger pad"
        )

    scaled = scaler.transform(feat)
    cols = list(SSL_LPP_LPR_STORED_COLUMNS)
    rows = []
    for i in range(scaled.shape[0]):
        row = {
            "utt_id": "utt0",
            "word_id": 0,
            "phone_id": int(i),
            "human_score": 0.0,
            "canonical_phone_idx": int(ssl_ids[i]),
        }
        for j, col in enumerate(cols):
            row[col] = float(scaled[i, j])
        rows.append(row)
    table = pd.DataFrame(rows)
    packed = pack_utterances(
        table,
        cols,
        max_seq_len=max_len,
        phone_idx_col="canonical_phone_idx",
        pad_phone_id=int(n_phones),
    )
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    model.to(dev)
    pred = predict_transformer(
        model, packed, batch_size=1, device=dev, n_rows=len(table)
    )
    return np.clip(pred, float(clip[0]), float(clip[1]))


def run_e16_utterance(
    wav_path: str | Path,
    transcript: str,
    *,
    checkpoint_path: str | Path | None = None,
    package_root: Path | None = None,
    device: str | None = None,
    g2p: Any | None = None,
) -> dict[str, Any]:
    """Full pipeline: G2P → 78-d LPP+LPR → E16 → JSON-ready dict."""
    root = package_root or PACKAGE_ROOT
    ckpt_path = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else root / "outputs" / "E" / "e16_phone_transformer.pt"
    )
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"missing {ckpt_path}; run: "
            "python scripts/export_group_e_checkpoint.py "
            "--config configs/e_learned_scoring.yaml "
            "--experiment E16 --features c8_lpp_lpr_embed"
        )
    phones = text_to_cmu_phones(transcript, g2p=g2p)
    feat, ssl_ids, n_frames, kept = extract_78d_lpp_lpr(
        wav_path,
        phones,
        package_root=root,
        device=device,
    )
    ckpt = load_checkpoint(ckpt_path, device=device or "cpu")
    scores = score_e16(feat, ssl_ids, ckpt, device=device or "cpu")
    return {
        "model": "E16",
        "acoustic_model": "wav2vec2-xlsr-53-espeak-cv-ft",
        "transcript": str(transcript),
        "wav_path": str(Path(wav_path)),
        "checkpoint": str(ckpt_path),
        "g2p_phones": phones,
        "phones": [
            {
                "phone_id": int(i),
                "phone": kept[i],
                "ssl_index": int(ssl_ids[i]),
                "n_frames": int(n_frames[i]),
                "score": float(scores[i]),
            }
            for i in range(len(kept))
        ],
    }


def dumps_result(result: dict[str, Any], *, indent: int = 2) -> str:
    return json.dumps(result, indent=indent, ensure_ascii=False)
