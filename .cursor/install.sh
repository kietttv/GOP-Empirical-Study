#!/usr/bin/env bash
# Idempotent Cloud Agent setup for the GOP Empirical Study repo.
# Installs Miniforge, creates the pinned conda env `gop`, installs pip
# requirements, and downloads the Hugging Face dataset into data/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_DIR="$HOME/miniforge3"
DATASET_REPO="tkiet1877/gop-empirical-study-data"

# 1. Miniforge (pins the conda toolchain; provides Python 3.11 for env `gop`).
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  echo "[install] Installing Miniforge3 ..."
  curl -fsSL -o /tmp/miniforge.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash /tmp/miniforge.sh -b -p "$CONDA_DIR"
  rm -f /tmp/miniforge.sh
fi
CONDA="$CONDA_DIR/bin/conda"

# Make `conda activate gop` work in interactive shells (idempotent).
"$CONDA" init bash >/dev/null 2>&1 || true

# 2. Create or update the pinned conda env.
if "$CONDA" env list | grep -qE '^\s*gop\s'; then
  echo "[install] Updating conda env 'gop' ..."
  "$CONDA" env update -n gop -f environment.yml --prune
else
  echo "[install] Creating conda env 'gop' ..."
  "$CONDA" env create -f environment.yml
fi

# 3. Pip requirements (torch/transformers/soundfile/hf + core stack) + pytest.
"$CONDA" run -n gop pip install -r requirements.txt -r requirements-ssl.txt
"$CONDA" run -n gop pip install pytest

# 4. Dataset (~1.1 GB). Guarded so re-runs skip the download.
if [ ! -f data/scores.json ] || [ ! -d data/kaldi_gop_librispeech ]; then
  echo "[install] Downloading dataset from Hugging Face ($DATASET_REPO) ..."
  "$CONDA" run -n gop hf download "$DATASET_REPO" data.zip \
    --repo-type dataset --local-dir data/_hf
  "$CONDA" run -n gop python -c \
    "import zipfile; zipfile.ZipFile('data/_hf/data.zip').extractall('data')"
  echo "[install] Dataset extracted into data/."
else
  echo "[install] Dataset already present; skipping download."
fi

echo "[install] Done. Activate with: conda activate gop"
