#!/usr/bin/env python3
"""Extract SSL GOP variants from a phoneme-CTC model on Speechocean762.

``mean`` (C2/C3): blank-stripped mean log P(canonical) on Kaldi spans.
``max``  (C4):    blank-stripped max log P(canonical) on the same spans.
``lpp_lpr`` (E9–E18): 39-d GOPT-style LPP+LPR pooled on **CTC Viterbi**
frames for the canonical sequence (same graph as GOP-S; not Kaldi start/end).
``cao_sd`` (C10/C11): Cao GOP-SD (AF-SD) on the same C8/C9 espeak AMs.

Examples:
    python scripts/extract_ssl_gop.py --config configs/c_acoustic_model.yaml --model wav2vec2
    python scripts/extract_ssl_gop.py --config configs/c_acoustic_model.yaml --model hubert --gop max cao_s
    python scripts/extract_ssl_gop.py --config configs/c_acoustic_model.yaml --model xlsr_espeak --gop cao_s
    python scripts/extract_ssl_gop.py --config configs/c_acoustic_model.yaml --model lv60_espeak --gop cao_s
    python scripts/extract_ssl_gop.py --config configs/c_acoustic_model.yaml --model xlsr_espeak --gop cao_sd
    python scripts/extract_ssl_gop.py --config configs/c_acoustic_model.yaml --model lv60_espeak --gop cao_sd
    python scripts/extract_ssl_gop.py --config configs/c_acoustic_model.yaml --model xlsr_espeak --gop lpp_lpr
    python scripts/extract_ssl_gop.py --config configs/c_acoustic_model.yaml --model lv60_espeak --gop lpp_lpr
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from gop_empirical.acoustic.alignment import (  # noqa: E402
    align_ctm_to_keys,
    find_wav,
    parse_ctm,
    time_to_frame_span,
)
from gop_empirical.acoustic.espeak_map import (  # noqa: E402
    blank_id_from_vocab,
    ctc_ids_from_vocab,
    load_cmu_to_espeak,
    load_espeak_vocab,
)
from gop_empirical.acoustic.phones import load_phone_inventory  # noqa: E402
from gop_empirical.acoustic.posterior import CtcPhonemePosterior  # noqa: E402
from gop_empirical.data.kaldi import load_kaldi_gop_split  # noqa: E402
from gop_empirical.data.ssl_gop import write_ssl_gop_split  # noqa: E402
from gop_empirical.data.ssl_lpp_lpr import write_ssl_lpp_lpr_split  # noqa: E402
from gop_empirical.experiment import load_config, resolve_path  # noqa: E402
from gop_empirical.gop.cao import cao_gop_s, cao_gop_sd  # noqa: E402
from gop_empirical.gop.ctc_align import ctc_label_frames  # noqa: E402
from gop_empirical.gop.from_posterior import (  # noqa: E402
    gop_from_log_probs,
    gop_max_from_log_probs,
    log_softmax_over_ids,
    log_softmax_rows,
)
from gop_empirical.gop.representation import (  # noqa: E402
    SSL_LPP_LPR_N_FEATURES,
    lpp_lpr_concat,
    mean_lpp_on_frames,
    mean_lpp_on_span,
)

GOP_VARIANTS = ("mean", "max", "cao_s", "cao_sd", "lpp_lpr")
ESPEAK_GOP_VARIANTS = ("cao_s", "cao_sd", "lpp_lpr")
LOCAL_MODELS = ("wav2vec2", "hubert")
ESPEAK_MODELS = ("xlsr_espeak", "lv60_espeak")
ALL_MODELS = LOCAL_MODELS + ESPEAK_MODELS


def _load_wav(path: Path, sampling_rate: int) -> np.ndarray:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ImportError(
            "extract_ssl_gop.py needs soundfile. pip install -r requirements-ssl.txt"
        ) from exc
    wav, sr = sf.read(str(path), always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if int(sr) != int(sampling_rate):
        try:
            import librosa
        except ImportError as exc:
            raise ImportError(
                f"wav {path} is {sr} Hz; need librosa to resample to {sampling_rate}"
            ) from exc
        wav = librosa.resample(wav, orig_sr=int(sr), target_sr=int(sampling_rate))
    return np.asarray(wav, dtype=np.float32).reshape(-1)


def _variant_out_dir(cfg: dict, model: str, variant: str, package_root: Path) -> Path:
    paths = cfg["paths"]
    if model == "wav2vec2":
        key = {
            "mean": "wav2vec2_gop_dir",
            "max": "wav2vec2_gop_max_dir",
            "cao_s": "wav2vec2_gop_cao_dir",
        }[variant]
    elif model == "hubert":
        key = {
            "mean": "hubert_gop_dir",
            "max": "hubert_gop_max_dir",
            "cao_s": "hubert_gop_cao_dir",
        }[variant]
    elif model == "xlsr_espeak":
        if variant == "cao_s":
            key = "xlsr_espeak_gop_cao_dir"
        elif variant == "cao_sd":
            key = "xlsr_espeak_gop_cao_sd_dir"
        elif variant == "lpp_lpr":
            key = "xlsr_espeak_lpp_lpr_dir"
        else:
            raise ValueError(
                "xlsr_espeak only supports GOP variants cao_s (C8), cao_sd (C10), lpp_lpr (E9/E10)"
            )
    elif model == "lv60_espeak":
        if variant == "cao_s":
            key = "lv60_espeak_gop_cao_dir"
        elif variant == "cao_sd":
            key = "lv60_espeak_gop_cao_sd_dir"
        elif variant == "lpp_lpr":
            key = "lv60_espeak_lpp_lpr_dir"
        else:
            raise ValueError(
                "lv60_espeak only supports GOP variants cao_s (C9), cao_sd (C11), lpp_lpr (E11/E12)"
            )
    else:
        raise ValueError(f"model must be one of {ALL_MODELS}, got {model!r}")
    if key not in paths:
        raise FileNotFoundError(
            f"config paths.{key} is missing; needed for {model} GOP variant {variant}"
        )
    return resolve_path(paths[key], base=package_root)


def _redirect_hf_cache_to_repo(package_root: Path) -> Path:
    """Keep any HuggingFace I/O on the repo drive, never ``%USERPROFILE%`` (C:)."""
    hf_home = package_root / "models" / ".hf_home"
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(hf_home / "transformers")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    return hf_home


def _checkpoint_paths(cfg: dict, model: str, package_root: Path) -> tuple[str, str, bool]:
    """Return ``(checkpoint, processor, is_local)``. C8/C9 are local folders under ``models/``."""
    paths = cfg["paths"]
    if model == "wav2vec2":
        ckpt = resolve_path(paths["wav2vec2_checkpoint"], base=package_root)
        proc = resolve_path(paths.get("wav2vec2_processor", paths["wav2vec2_checkpoint"]), base=package_root)
        return str(ckpt), str(proc), True
    if model == "hubert":
        ckpt = resolve_path(paths["hubert_checkpoint"], base=package_root)
        proc = resolve_path(paths.get("hubert_processor", paths["hubert_checkpoint"]), base=package_root)
        return str(ckpt), str(proc), True
    if model == "xlsr_espeak":
        ckpt = resolve_path(paths["xlsr_espeak_checkpoint"], base=package_root)
        proc = resolve_path(paths.get("xlsr_espeak_processor", paths["xlsr_espeak_checkpoint"]), base=package_root)
        return str(ckpt), str(proc), True
    if model == "lv60_espeak":
        ckpt = resolve_path(paths["lv60_espeak_checkpoint"], base=package_root)
        proc = resolve_path(paths.get("lv60_espeak_processor", paths["lv60_espeak_checkpoint"]), base=package_root)
        return str(ckpt), str(proc), True
    raise ValueError(f"model must be one of {ALL_MODELS}, got {model!r}")


def extract_split(
    *,
    cfg: dict,
    package_root: Path,
    split: str,
    posterior: CtcPhonemePosterior,
    inventory,
    ctm_path: Path,
    wav_dir: Path,
    variants: list[str],
    ctc_id_lookup: dict[str, int] | None = None,
    blank_id: int = 0,
    espeak_scored_ids: np.ndarray | None = None,
) -> dict[str, list[dict]]:
    kaldi_dir = resolve_path(cfg["paths"]["kaldi_dir"], base=package_root)
    _feats, keys = load_kaldi_gop_split(kaldi_dir, split)
    ctm = parse_ctm(ctm_path)
    durations = {}
    rows_align, _stats = align_ctm_to_keys(
        ctm,
        keys,
        inventory=inventory,
        wav_dir=wav_dir,
        durations=durations,
    )
    by_utt: dict[str, list[tuple[int, str, float, float]]] = {}
    for utt, phn_idx, phone, start, end in rows_align:
        by_utt.setdefault(utt, []).append((phn_idx, phone, start, end))

    need_span = any(v in variants for v in ("mean", "max"))
    need_cao_s = "cao_s" in variants
    need_cao_sd = "cao_sd" in variants
    need_lpp_lpr = "lpp_lpr" in variants
    need_ctc_probs = need_cao_s or need_cao_sd or need_lpp_lpr
    if need_lpp_lpr and (espeak_scored_ids is None or ctc_id_lookup is None):
        raise ValueError("lpp_lpr extract needs espeak_scored_ids and ctc_id_lookup")
    out_rows: dict[str, list[dict]] = {v: [] for v in variants}
    hop_s = posterior.hop_s
    n_utt = len(by_utt)
    for i, (utt, phones) in enumerate(by_utt.items(), start=1):
        if i == 1 or i % 100 == 0 or i == n_utt:
            print(f"{split} {i}/{n_utt}", flush=True)
        wav_path = find_wav(utt, wav_dir)
        waveform = _load_wav(wav_path, int(cfg.get("sampling_rate_hz", 16000)))
        logits = posterior.frame_logits(waveform)
        n_frames = int(logits.shape[0])
        ctc_probs = np.exp(log_softmax_rows(logits)) if need_ctc_probs else None
        log_p = None
        log_p39 = None
        if need_span:
            if posterior.phone_ids.size < 1:
                raise RuntimeError("mean/max GOP needs inventory-restricted CTC ids")
            log_p = log_softmax_over_ids(logits, posterior.phone_ids)
        if need_lpp_lpr:
            if espeak_scored_ids is None:
                raise ValueError("lpp_lpr extract needs espeak_scored_ids")
            log_p39 = log_softmax_over_ids(logits, espeak_scored_ids)
        seq_ids: list[int] = []
        span_rows: list[tuple[int, str, int, int]] = []
        lpp_meta: list[tuple[int, str, int]] = []
        for phn_idx, phone, start, end in phones:
            try:
                if ctc_id_lookup is not None:
                    ctc_id = ctc_id_lookup[phone]
                    can_idx = None
                else:
                    can_idx = inventory.ssl_index(phone)
                    ctc_id = inventory.ssl_ctc_id(phone)
            except KeyError:
                for variant in variants:
                    if variant == "lpp_lpr":
                        out_rows[variant].append(
                            {
                                "key": f"{utt}.{phn_idx}",
                                "phone": phone,
                                "n_frames": 0,
                                "features": np.full(
                                    SSL_LPP_LPR_N_FEATURES, np.nan, dtype=np.float64
                                ),
                            }
                        )
                    else:
                        out_rows[variant].append(
                            {
                                "key": f"{utt}.{phn_idx}",
                                "phone": phone,
                                "gop": float("nan"),
                                "n_frames": 0,
                            }
                        )
                continue
            t0, t1 = time_to_frame_span(start, end, hop_s=hop_s, n_frames=n_frames)
            seq_ids.append(int(ctc_id))
            span_rows.append((phn_idx, phone, t0, t1))
            if need_lpp_lpr:
                lpp_meta.append((phn_idx, phone, inventory.ssl_index(phone)))
            if need_span:
                if can_idx is None:
                    raise RuntimeError(
                        "mean/max GOP needs inventory ssl_index; espeak extract is cao_s/cao_sd/lpp_lpr"
                    )
                if "mean" in out_rows:
                    gop, n = gop_from_log_probs(log_p, can_idx, t0, t1)
                    out_rows["mean"].append(
                        {"key": f"{utt}.{phn_idx}", "phone": phone, "gop": gop, "n_frames": n}
                    )
                if "max" in out_rows:
                    gop, n = gop_max_from_log_probs(log_p, can_idx, t0, t1)
                    out_rows["max"].append(
                        {"key": f"{utt}.{phn_idx}", "phone": phone, "gop": gop, "n_frames": n}
                    )
        cao_jobs = []
        if need_cao_s:
            cao_jobs.append(("cao_s", cao_gop_s))
        if need_cao_sd:
            cao_jobs.append(("cao_sd", cao_gop_sd))
        for variant, fn in cao_jobs:
            if seq_ids:
                gops = fn(ctc_probs, np.asarray(seq_ids, dtype=np.int64), blank=int(blank_id))
            else:
                gops = np.asarray([], dtype=np.float64)
            gi = 0
            for phn_idx, phone, _t0, _t1 in span_rows:
                gop = float(gops[gi]) if gi < gops.size else float("nan")
                n_ok = int(n_frames) if np.isfinite(gop) else 0
                out_rows[variant].append(
                    {"key": f"{utt}.{phn_idx}", "phone": phone, "gop": gop, "n_frames": n_ok}
                )
                gi += 1
        if need_lpp_lpr:
            if seq_ids and ctc_probs is not None and log_p39 is not None:
                aligned = ctc_label_frames(
                    ctc_probs,
                    np.asarray(seq_ids, dtype=np.int64),
                    blank=int(blank_id),
                )
            else:
                aligned = []
            for i, (phn_idx, phone, can_idx_39) in enumerate(lpp_meta):
                frames = aligned[i] if i < len(aligned) else np.zeros(0, dtype=np.int64)
                lpp_vec = mean_lpp_on_frames(log_p39, frames)
                if lpp_vec is None:
                    feat = np.full(SSL_LPP_LPR_N_FEATURES, np.nan, dtype=np.float64)
                    n_ok = 0
                else:
                    feat = lpp_lpr_concat(lpp_vec, can_idx_39).reshape(-1)
                    n_ok = int(frames.size)
                out_rows["lpp_lpr"].append(
                    {
                        "key": f"{utt}.{phn_idx}",
                        "phone": phone,
                        "n_frames": n_ok,
                        "features": feat,
                    }
                )
    return out_rows


def main() -> int:
    # HuggingFace/httpx cannot use the Windows SOCKS system proxy (socks4://).
    os.environ.setdefault("NO_PROXY", "*")
    os.environ.setdefault("no_proxy", "*")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=ALL_MODELS)
    parser.add_argument(
        "--gop",
        nargs="+",
        default=["mean"],
        choices=GOP_VARIANTS,
        help="GOP variants to write (default: mean). C4/C6=max, C5/C7/C8/C9=cao_s, C10/C11=cao_sd, E9–E12=lpp_lpr.",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    variants = list(dict.fromkeys(args.gop))
    espeak = args.model in ESPEAK_MODELS
    if not espeak and "cao_sd" in variants:
        raise ValueError("cao_sd (C10/C11) is only implemented for xlsr_espeak and lv60_espeak")
    if espeak:
        extra = set(variants) - set(ESPEAK_GOP_VARIANTS)
        if extra:
            raise ValueError(
                "C8/C9/C10/C11 (xlsr_espeak, lv60_espeak) only support "
                "--gop cao_s, cao_sd, and/or lpp_lpr"
            )

    cfg_path = args.config
    if not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path).resolve()
        if not cfg_path.is_file():
            cfg_path = (PACKAGE_ROOT / args.config).resolve()
    cfg = load_config(cfg_path)
    paths = cfg["paths"]
    inv_path = resolve_path(paths.get("phone_inventory", "data/phone_inventory.json"), base=PACKAGE_ROOT)
    inventory = load_phone_inventory(inv_path)
    wav_dir = resolve_path(paths["wav_dir"], base=PACKAGE_ROOT)
    if not wav_dir.is_dir():
        raise FileNotFoundError(
            f"wav_dir not found: {wav_dir}. Place Speechocean762 16 kHz wavs there "
            "(see data/SOURCE.txt) before extracting SSL GOP."
        )
    _redirect_hf_cache_to_repo(PACKAGE_ROOT)
    ckpt, proc, is_local = _checkpoint_paths(cfg, args.model, PACKAGE_ROOT)
    if is_local and not Path(ckpt).exists():
        if espeak:
            raise FileNotFoundError(
                f"{args.model} checkpoint not found: {ckpt}. "
                "Download into models/ with: python scripts/download_espeak_ctc.py"
            )
        raise FileNotFoundError(
            f"{args.model} checkpoint not found: {ckpt}. "
            "For HuBERT, fine-tune first with scripts/finetune_phoneme_ctc.py "
            "--backbone hubert"
        )
    if Path(ckpt).is_dir():
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    out_dirs = {v: _variant_out_dir(cfg, args.model, v, PACKAGE_ROOT) for v in variants}

    posterior = CtcPhonemePosterior(
        ckpt,
        inventory,
        processor_dir=proc,
        device=args.device,
        sampling_rate=int(cfg.get("sampling_rate_hz", 16000)),
        restrict_to_inventory=not espeak,
    )
    ctc_id_lookup = None
    blank_id = 0
    espeak_scored_ids = None
    if espeak:
        map_path = resolve_path(paths.get("cmu_to_espeak", "data/cmu_to_espeak.json"), base=PACKAGE_ROOT)
        cmu_map = load_cmu_to_espeak(map_path, scored_symbols=inventory.scored_symbols)
        vocab_path = resolve_path(
            paths.get("espeak_ctc_vocab", "data/espeak_ctc_vocab.json"),
            base=PACKAGE_ROOT,
        )
        vocab = load_espeak_vocab(vocab_path)
        ctc_id_lookup = ctc_ids_from_vocab(cmu_map, vocab)
        blank_id = blank_id_from_vocab(vocab)
        espeak_scored_ids = np.asarray(
            [ctc_id_lookup[sym] for sym in inventory.scored_symbols],
            dtype=np.int64,
        )
        print(
            f"{args.model}: local={ckpt}  blank={blank_id}  mapped={len(ctc_id_lookup)} phones",
            flush=True,
        )
    ctm_map = {
        "train": resolve_path(paths["ctm_train"], base=PACKAGE_ROOT),
        "test": resolve_path(paths["ctm_test"], base=PACKAGE_ROOT),
    }
    for split in ("train", "test"):
        ctm_path = ctm_map[split]
        if not ctm_path.is_file():
            raise FileNotFoundError(
                f"Kaldi CTM missing: {ctm_path}. Export the same alignment used "
                "for the GOPT extract (see data/SOURCE.txt)."
            )
        rows_by_variant = extract_split(
            cfg=cfg,
            package_root=PACKAGE_ROOT,
            split=split,
            posterior=posterior,
            inventory=inventory,
            ctm_path=ctm_path,
            wav_dir=wav_dir,
            variants=variants,
            ctc_id_lookup=ctc_id_lookup,
            blank_id=blank_id,
            espeak_scored_ids=espeak_scored_ids,
        )
        for variant, rows in rows_by_variant.items():
            if variant == "lpp_lpr":
                written = write_ssl_lpp_lpr_split(rows, out_dirs[variant], split)
            else:
                written = write_ssl_gop_split(rows, out_dirs[variant], split)
            n_ok = sum(1 for r in rows if r["n_frames"] > 0)
            print(f"{split} {variant}: wrote {written}  n={len(rows)} scored={n_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
