"""HuggingFace CTC frame posteriors for Group C (Wav2Vec2 / HuBERT).

Requires torch + transformers (see requirements-ssl.txt). Evaluation of GOP
from saved CSVs does not import this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from gop_empirical.acoustic.alignment import encoder_hop_samples
from gop_empirical.acoustic.phones import PhoneInventory
from gop_empirical.gop.from_posterior import log_softmax_over_ids


def _require_torch():
    try:
        import torch
        from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForCTC, AutoProcessor
    except ImportError as exc:
        raise ImportError(
            "C2/C3 extract needs torch and transformers. "
            "Install with: pip install -r requirements-ssl.txt"
        ) from exc
    return torch, AutoConfig, AutoFeatureExtractor, AutoModelForCTC, AutoProcessor


def _pretrained_kwargs(src: str) -> dict[str, Any]:
    """Load a local folder without touching the HuggingFace hub cache."""
    if Path(src).is_dir():
        return {"local_files_only": True}
    return {}


def hop_seconds_from_config(config: Any, sampling_rate: int = 16000) -> float:
    stride = getattr(config, "conv_stride", None)
    if stride is None:
        raise ValueError("model config has no conv_stride; cannot infer encoder hop")
    hop = encoder_hop_samples(list(stride))
    if sampling_rate <= 0:
        raise ValueError(f"sampling_rate must be positive, got {sampling_rate}")
    return float(hop) / float(sampling_rate)


class CtcPhonemePosterior:
    """Frame log-posteriors over the 39 scored CMU phones (blank stripped)."""

    def __init__(
        self,
        checkpoint: str | Path,
        inventory: PhoneInventory,
        *,
        processor_dir: str | Path | None = None,
        device: str | None = None,
        sampling_rate: int = 16000,
        restrict_to_inventory: bool = True,
    ) -> None:
        torch, AutoConfig, AutoFeatureExtractor, AutoModelForCTC, AutoProcessor = _require_torch()
        self.torch = torch
        self.inventory = inventory
        self.sampling_rate = int(sampling_rate)
        self.restrict_to_inventory = bool(restrict_to_inventory)
        ckpt = str(checkpoint)
        proc_src = str(processor_dir) if processor_dir is not None else ckpt
        ckpt_kw = _pretrained_kwargs(ckpt)
        proc_kw = _pretrained_kwargs(proc_src)
        self.config = AutoConfig.from_pretrained(ckpt, **ckpt_kw)
        self.hop_s = hop_seconds_from_config(self.config, sampling_rate=self.sampling_rate)
        if self.restrict_to_inventory:
            self.processor = AutoProcessor.from_pretrained(proc_src, **proc_kw)
        else:
            # C8/C9 Wav2Vec2PhonemeCTCTokenizer requires phonemizer even when
            # we never phonemize text. Feature extractor is enough for logits.
            self.processor = AutoFeatureExtractor.from_pretrained(proc_src, **proc_kw)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = AutoModelForCTC.from_pretrained(ckpt, **ckpt_kw)
        self.model.to(device)
        self.model.eval()
        if self.restrict_to_inventory:
            self.phone_ids = np.asarray(inventory.ssl_scored_ids, dtype=np.int64)
        else:
            # C8/C9: full espeak CTC vocab; Cao GOP-S maps CMU→IPA ids separately.
            self.phone_ids = np.zeros((0,), dtype=np.int64)

    def _frame_logits(self, waveform: np.ndarray) -> np.ndarray:
        torch = self.torch
        wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
        inputs = self.processor(
            wav,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=False,
        )
        input_values = inputs.input_values.to(self.device)
        with torch.no_grad():
            logits = self.model(input_values).logits
        return logits.squeeze(0).detach().cpu().numpy()

    def frame_logits(self, waveform: np.ndarray) -> np.ndarray:
        """Raw CTC logits ``[T, V]``. One encoder forward."""
        return self._frame_logits(waveform)

    def log_phone_posteriors(self, waveform: np.ndarray) -> np.ndarray:
        """Return ``[T, 39]`` log P over scored phones. Hidden states are unused."""
        log_phone, _ctc = self.scored_and_ctc_probs(waveform)
        return log_phone

    def ctc_frame_probs(self, waveform: np.ndarray) -> np.ndarray:
        """Full-vocab CTC softmax ``[T, V]`` including blank (Cao GOP-S)."""
        _log_phone, ctc = self.scored_and_ctc_probs(waveform)
        return ctc

    def scored_and_ctc_probs(self, waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One encoder forward: blank-stripped log P ``[T, 39]`` and full CTC softmax ``[T, V]``."""
        from gop_empirical.gop.from_posterior import log_softmax_rows

        logits = self._frame_logits(waveform)
        ctc_probs = np.exp(log_softmax_rows(logits))
        if self.phone_ids.size:
            log_phone = log_softmax_over_ids(logits, self.phone_ids)
        else:
            log_phone = np.empty((logits.shape[0], 0), dtype=np.float64)
        return log_phone, ctc_probs
