#!/usr/bin/env python3
"""Download C8/C9 Facebook espeak CTC checkpoints into ``models/`` on this drive.

Does not use ``%USERPROFILE%\\.cache\\huggingface`` (C:). A temporary Hub cache
under ``models/.hf_home`` is deleted after the local folders are complete.

    python scripts/download_espeak_ctc.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MODELS = PACKAGE_ROOT / "models"
HF_HOME = MODELS / ".hf_home"

REPOS = (
    ("facebook/wav2vec2-xlsr-53-espeak-cv-ft", MODELS / "wav2vec2-xlsr-53-espeak-cv-ft"),
    ("facebook/wav2vec2-lv-60-espeak-cv-ft", MODELS / "wav2vec2-lv-60-espeak-cv-ft"),
)


def _is_complete(dest: Path) -> bool:
    if not dest.is_dir():
        return False
    has_config = (dest / "config.json").is_file()
    has_weights = (dest / "pytorch_model.bin").is_file() or (dest / "model.safetensors").is_file()
    has_fe = (dest / "preprocessor_config.json").is_file()
    return has_config and has_weights and has_fe


def main() -> int:
    os.environ.setdefault("NO_PROXY", "*")
    os.environ.setdefault("no_proxy", "*")
    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HF_HUB_CACHE"] = str(HF_HOME / "hub")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    MODELS.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    for repo_id, dest in REPOS:
        if _is_complete(dest):
            print(f"skip (already local): {dest}", flush=True)
            continue
        print(f"download {repo_id} -> {dest}", flush=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest),
            cache_dir=str(HF_HOME / "hub"),
        )
        if not _is_complete(dest):
            raise RuntimeError(f"download incomplete: {dest}")
        print(f"ok {dest}", flush=True)

    if HF_HOME.exists():
        shutil.rmtree(HF_HOME, ignore_errors=True)
        print(f"removed temp cache {HF_HOME}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
