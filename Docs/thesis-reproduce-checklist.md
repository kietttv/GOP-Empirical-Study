# Thesis experiment checklist — re-run

Dataset: Speechocean762 · phoneme · official test.  
Nguồn danh sách: `Docs/summary.md` (trong thesis). Đã loại: C2–C7, E19–E22 (`Docs/remove.md`).

Đánh dấu `[x]` sau khi re-run và khớp số với `outputs/*/`.

---

## 0. Prerequisites (extract / AM)

- [x] Kaldi GOP extract sẵn (`data/kaldi_gop_librispeech/`) — không re-run Kaldi
- [x] Download C8/C9 espeak CTC (`python scripts/download_espeak_ctc.py`)
- [x] Extract C8 Cao GOP-S (`--model xlsr_espeak --gop cao_s`)
- [x] Extract C9 Cao GOP-S (`--model lv60_espeak --gop cao_s`)
- [x] Extract C10 Cao GOP-SD (`--model xlsr_espeak --gop cao_sd`)
- [x] Extract C11 Cao GOP-SD (`--model lv60_espeak --gop cao_sd`)
- [x] Extract C8 78-d LPP+LPR (`--model xlsr_espeak --gop lpp_lpr`) → E9/E10/E15/E16
- [x] Extract C9 78-d LPP+LPR (`--model lv60_espeak --gop lpp_lpr`) → E11/E12/E17/E18

---



## 1. A — Traditional GOP

```text
python scripts/run_experiment.py --config configs/a_traditional_gop.yaml
```

- [x] A1 Traditional GOP (canonical LPP)
- [x] A2 GOP vs Human (eval A1 trên official test)

---



## 2. B — GOP Feature Representation

```text
python scripts/run_experiment.py --config configs/b_gop_representation.yaml
```

- [x] B1 Standard GOP (= LPP[canonical])
- [x] B2 LPP (≡ B1 trên extract này)
- [x] B3 LPR vs best competitor
- [x] B4 GOP-only vector OLS `[LPP, max competitor, LPR]`
- [x] B5 84-d LPP+LPR OLS (GOPT-style)

---



## 3. C — Acoustic Model Dependency

```text
python scripts/run_experiment.py --config configs/c_acoustic_model.yaml --models C1 C8 C9 C10 C11
```

- [x] C1 Kaldi M13 · LPP (≡ A2)
- [x] C8 XLSR-53 espeak · Cao GOP-S (AF-S)
- [x] C9 lv60 espeak · Cao GOP-S (AF-S)
- [x] C10 cùng AM C8 · Cao GOP-SD (AF-SD)
- [x] C11 cùng AM C9 · Cao GOP-SD (AF-SD)

---



## 4. D — GOP Behavior Analysis

```text
python scripts/run_experiment.py --config configs/d_gop_behavior.yaml
```

- [ ] D1 Phone-level (CONSONANT / VOWEL rollup)
- [ ] D2 Speaker-level (per-speaker PCC summary)
- [ ] D3 Score strata (Low / Mid / High by speaker-mean sentence accuracy)

---



## 5. E — GOP Learned Scoring



### 5.1 E1–E2 · Kaldi B4 (3-d)

```text
python scripts/run_experiment.py --config configs/e_learned_scoring.yaml --features b4
```

- [x] E1 B4 + MLP
- [x] E2 B4 + Transformer



### 5.2 E3–E6 · Cao GOP-S (C8/C9)

```text
python scripts/run_experiment.py --config configs/e_learned_scoring.yaml --features c8 c9
```

- [x] E3 C8 GOP-S + MLP
- [x] E4 C8 GOP-S + Transformer
- [ ] E5 C9 GOP-S + MLP
- [ ] E6 C9 GOP-S + Transformer



### 5.3 E7–E8 · Kaldi 84-d

```text
python scripts/run_experiment.py --config configs/e_learned_scoring.yaml --features b5
```

- [x] E7 Kaldi 84-d + MLP
- [x] E8 Kaldi 84-d + Transformer



### 5.4 E13–E14 · Kaldi 84-d + phone embed

```text
python scripts/run_experiment.py --config configs/e_learned_scoring.yaml --features b5_embed
```

- [x] E13 Kaldi 84-d + MLP + phone embed
- [x] E14 Kaldi 84-d + Transformer + phone embed



### 5.5 E9–E12 · C8/C9 78-d LPP+LPR

```text
python scripts/run_experiment.py --config configs/e_learned_scoring.yaml --features c8_lpp_lpr c9_lpp_lpr
```

- [x] E9 C8 78-d + MLP
- [x] E10 C8 78-d + Transformer
- [x] E11 C9 78-d + MLP
- [x] E12 C9 78-d + Transformer



### 5.6 E15–E18 · C8/C9 78-d + SSL phone embed

```text
python scripts/run_experiment.py --config configs/e_learned_scoring.yaml --features c8_lpp_lpr_embed c9_lpp_lpr_embed
```

- [x] E15 C8 78-d + MLP + SSL embed
- [x] E16 C8 78-d + Transformer + SSL embed
- [x] E17 C9 78-d + MLP + SSL embed
- [x] E18 C9 78-d + Transformer + SSL embed

---



## 6. F — Statistical Validation & Error Analysis

```text
python scripts/run_experiment.py --config configs/f_validation.yaml
```

- [ ] F1a Bootstrap CI (headline models)
- [ ] F1b Paired ΔPCC contrasts
- [ ] F1c Multi-seed E2 / E16 (seeds 0–4)
- [ ] F2 Error taxonomy (C8 vs E16)

---



## Out of thesis (không cần re-run cho bảng chính)

- [ ] ~~C2–C7~~ — negative control / in-house AM (`Docs/remove.md`)
- [ ] ~~E19–E22~~ — learned scoring trên GOP-SD (`Docs/remove.md`)