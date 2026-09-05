# GOP Empirical Study

---

## 1. Setup

```powershell
# From repo root
conda env create -f environment.yml
conda activate gop
pip install -r requirements.txt
pip install -r requirements-ssl.txt   # torch/transformers/soundfile/huggingface_hub — SSL + HF download
```

Run all commands below from the repo root with `conda activate gop`.

Unit tests (optional):

```powershell
python -m pytest tests/
```

---

## 2. Get data from Hugging Face

### Option 1 — Download from browser

Access: [https://huggingface.co/datasets/tkiet1877/gop-empirical-study-data](https://huggingface.co/datasets/tkiet1877/gop-empirical-study-data)

Dowload the "data.zip" to /data

Unzip "data.zip"

### Option 2 — CLI (`hf`)

```powershell
# From repo root

# Download data.zip into a temp folder under the repo
hf download tkiet1877/gop-empirical-study-data data.zip --repo-type dataset --local-dir data\_hf

# Unzip into data/ (overwrites same names if already present)
Expand-Archive -Path data\_hf\data.zip -DestinationPath data -Force
```

After unzip, you should see under `data/`:

- `kaldi_gop_librispeech/`
- `so762_inputs/`
- `speechocean762/`
- `scores.json`, `phone_inventory.json`, `cmu_to_espeak.json`, `espeak_ctc_vocab.json`, `SOURCE.txt`

Quick check:

```powershell
Test-Path data\scores.json
Test-Path data\kaldi_gop_librispeech
Test-Path data\so762_inputs\wavs
```

---



## 3. Extract SSL features (before C8–C11 / E9–E18)

Requires: `data/so762_inputs/wavs/`, `data/so762_inputs/segments/{train,test}.ctm`.

### 3.1 Download 2 Wav2Vec2 espeak CTC models (C8 / C9)

Off-the-shelf Facebook phoneme CTC checkpoints from Hugging Face Hub.  
Needed before any `extract_ssl_gop.py` run that uses `--model xlsr_espeak` or `lv60_espeak`.


| Experiment AM       | Hugging Face repo                                                                                         | Local folder                            |
| ------------------- | --------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| C8 / C10 / E (XLSR) | `[facebook/wav2vec2-xlsr-53-espeak-cv-ft](https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft)` | `models/wav2vec2-xlsr-53-espeak-cv-ft/` |
| C9 / C11 / E (lv60) | `[facebook/wav2vec2-lv-60-espeak-cv-ft](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft)`     | `models/wav2vec2-lv-60-espeak-cv-ft/`   |


Prerequisite: `pip install -r requirements-ssl.txt` (pulls `transformers` / `huggingface_hub`).

```powershell
# From repo root, conda env gop active, network OK
python scripts/download_espeak_ctc.py
```

What the script does:

- Downloads both repos into `models/` on this drive (not `%USERPROFILE%\.cache\huggingface`).
- Skips a model if that folder already has `config.json`, weights (`pytorch_model.bin` or `model.safetensors`), and `preprocessor_config.json`.
- Uses a temporary cache under `models/.hf_home`, then deletes it after a successful download.

Expected console:

```text
download facebook/wav2vec2-xlsr-53-espeak-cv-ft -> ...\models\wav2vec2-xlsr-53-espeak-cv-ft
ok ...
download facebook/wav2vec2-lv-60-espeak-cv-ft -> ...\models\wav2vec2-lv-60-espeak-cv-ft
ok ...
removed temp cache ...\models\.hf_home
```

Or `skip (already local): ...` if both folders are complete.

Quick check:

```powershell
Test-Path models\wav2vec2-xlsr-53-espeak-cv-ft\config.json
Test-Path models\wav2vec2-lv-60-espeak-cv-ft\config.json
```

Both should return `True`. Do **not** fine-tune these for the thesis C8/C9 path — use them as published.

### 3.2 Extract Cao GOP-S / GOP-SD and 78-d LPP+LPR

```powershell
# C8 / E3–E4
python scripts/extract_ssl_gop.py --config configs/c_acoustic_model.yaml --model xlsr_espeak --gop cao_s --device cuda

# C9 / E5–E6
python scripts/extract_ssl_gop.py --config configs/c_acoustic_model.yaml --model lv60_espeak --gop cao_s --device cuda

# C10
python scripts/extract_ssl_gop.py --config configs/c_acoustic_model.yaml --model xlsr_espeak --gop cao_sd --device cuda

# C11
python scripts/extract_ssl_gop.py --config configs/c_acoustic_model.yaml --model lv60_espeak --gop cao_sd --device cuda

# E9/E10/E15/E16 (78-d)
python scripts/extract_ssl_gop.py --config configs/c_acoustic_model.yaml --model xlsr_espeak --gop lpp_lpr --device cuda

# E11/E12/E17/E18 (78-d)
python scripts/extract_ssl_gop.py --config configs/c_acoustic_model.yaml --model lv60_espeak --gop lpp_lpr --device cuda
```

Omit `--device cuda` to use CPU (slower).  
`run_experiment.py` only **reads** these CSVs/NPZs; it does not extract.

---



## 4. Run experiments

Dependency: **A → B** (parallel OK) → **C** (needs SSL CSV for C8–C11) → **D** (needs A) → **E** (needs A/B/C + NPZ for 78-d) → **F** (needs A/B/C/D/E locked).

Artifacts go to `outputs/{A,B,C,D,E,F}/`. Terminal prints metrics (not a full JSON dump).

### 4.1 Group A — Traditional GOP (A1–A2)

```powershell
python scripts/run_experiment.py --config configs/a_traditional_gop.yaml
```

Uses: `data/kaldi_gop_librispeech/`, `data/scores.json`.

### 4.2 Group B — Feature representation (B1–B5)

```powershell
python scripts/run_experiment.py --config configs/b_gop_representation.yaml
```

Same Kaldi extract as A.

### 4.3 Group C — Acoustic model (C1, C8–C11)

```powershell
# Thesis set
python scripts/run_experiment.py --config configs/c_acoustic_model.yaml --models C1 C8 C9 C10 C11

# C1 only (no SSL CSV)
python scripts/run_experiment.py --config configs/c_acoustic_model.yaml --models C1
```

Default yaml also omits C6/C7. Needs SSL extracts for any model other than C1.

### 4.4 Group D — Behavior analysis (D1–D3)

```powershell
python scripts/run_experiment.py --config configs/d_gop_behavior.yaml
```

Needs `outputs/A/` (`a1_predictions.csv`, `a2_results.json`).

### 4.5 Group E — Learned scoring (E1–E18)

Each `--features` trains a pair (MLP + Transformer). Follow-ups merge into `e_results.json` without wiping earlier blocks.

```powershell
# E1 / E2 — Kaldi B4
python scripts/run_experiment.py --config configs/e_learned_scoring.yaml --features b4

# E3–E6 — Cao GOP-S from C8/C9
python scripts/run_experiment.py --config configs/e_learned_scoring.yaml --features c8 c9

# E7 / E8 — Kaldi 84-d
python scripts/run_experiment.py --config configs/e_learned_scoring.yaml --features b5

# E13 / E14 — 84-d + phone embed (keeps E7/E8 preds)
python scripts/run_experiment.py --config configs/e_learned_scoring.yaml --features b5_embed

# E9–E12 — 78-d LPP+LPR (needs NPZ)
python scripts/run_experiment.py --config configs/e_learned_scoring.yaml --features c8_lpp_lpr c9_lpp_lpr

# E15–E18 — 78-d + SSL embed (keeps E9–E12 preds)
python scripts/run_experiment.py --config configs/e_learned_scoring.yaml --features c8_lpp_lpr_embed c9_lpp_lpr_embed
```


| `--features`                          | Experiments | Needs                                              |
| ------------------------------------- | ----------- | -------------------------------------------------- |
| `b4`                                  | E1, E2      | A, B                                               |
| `c8` `c9`                             | E3–E6       | C predictions                                      |
| `b5`                                  | E7, E8      | Kaldi 84-d                                         |
| `b5_embed`                            | E13, E14    | prior `b5` CSV preferred                           |
| `c8_lpp_lpr` `c9_lpp_lpr`             | E9–E12      | `*_lpp_lpr` NPZ + C                                |
| `c8_lpp_lpr_embed` `c9_lpp_lpr_embed` | E15–E18     | same NPZ; run `c8_lpp_lpr` first if locking E9–E12 |


Default device in yaml is `cpu`. Override with `--device cuda` if desired.

Each run writes scorer weights to `checkpoints/{eid}/{mlp|transformer}_ckpt.pt` (overwrites on re-run). Group F multi-seed does not write here. One-off export without rewriting CSVs:

```powershell
python scripts/export_group_e_checkpoint.py --config configs/e_learned_scoring.yaml --experiment E16 --features c8_lpp_lpr_embed
```

Evaluate a saved checkpoint on official test (no training):

```powershell
python scripts/eval_group_e_checkpoint.py --experiment E15
python scripts/eval_group_e_checkpoint.py --experiment E1 E2
```



### 4.6 Group F — Validation & error analysis (F1–F2)

```powershell
python scripts/run_experiment.py --config configs/f_validation.yaml
```

Needs locked A/B/C/D/E predictions. Multi-seed retrain of E2/E16 writes under `outputs/F/` only.

Bootstrap only (skip multi-seed):

```powershell
python scripts/run_experiment.py --config configs/f_validation.yaml --skip-multiseed
python scripts/run_f_multiseed.py
```

---



## 5. Minimal “eval only” path (extracts already on disk)

```powershell
conda activate gop
cd # repo root

python scripts/run_experiment.py --config configs/a_traditional_gop.yaml
python scripts/run_experiment.py --config configs/b_gop_representation.yaml
python scripts/run_experiment.py --config configs/c_acoustic_model.yaml --models C1 C8 C9 C10 C11
python scripts/run_experiment.py --config configs/d_gop_behavior.yaml

python scripts/run_experiment.py --config configs/e_learned_scoring.yaml --features b4
python scripts/run_experiment.py --config configs/e_learned_scoring.yaml --features c8 c9
python scripts/run_experiment.py --config configs/e_learned_scoring.yaml --features b5
python scripts/run_experiment.py --config configs/e_learned_scoring.yaml --features b5_embed
python scripts/run_experiment.py --config configs/e_learned_scoring.yaml --features c8_lpp_lpr c9_lpp_lpr
python scripts/run_experiment.py --config configs/e_learned_scoring.yaml --features c8_lpp_lpr_embed c9_lpp_lpr_embed

python scripts/run_experiment.py --config configs/f_validation.yaml
```

---



## 6. Notebooks

`notebooks/A2_*.ipynb`, `B_*.ipynb`, `C_*.ipynb`, `D_*.ipynb`, `E_*.ipynb`, `F_*.ipynb` **read** `outputs/` only — they do not recompute GOP.

---

