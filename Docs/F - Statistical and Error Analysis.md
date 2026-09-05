# F — Statistical Validation and Error Analysis

Group F trả lời:

> **Các finding A–E đáng tin cậy đến mức nào, và GOP / learned scoring fail ở đâu?**

Không phải pipeline GOP mới. F **không** tính lại posterior. Independent variable không đổi AM / representation / scoring architecture — F chỉ bootstrap, paired Δ, multi-seed, và taxonomy lỗi trên prediction đã khóa.

| ID | Tên | Việc làm |
| -- | --- | -------- |
| **F1** | Statistical analysis | Phone-level bootstrap CI; paired ΔPCC; multi-seed E2/E16 |
| **F2** | Error analysis | Residual + Speechocean762 expert markup taxonomy |

**Phạm vi model (headline):**

| Vai trò | Model | Score |
| ------- | ----- | ----- |
| Direct baseline | **C1** (≡ A2) | map train-only trên `gop_c1` |
| Best direct SSL | **C8**, **C9** | map trên `gop_c8` / `gop_c9` |
| B4 linear | **B4_OLS** | OLS `[LPP, max competitor]` từ B |
| Learned Kaldi seq | **E2** | `pred_e2` |
| Learned 84-d | **E7**, **E8**, **E14** | pred E |
| Best learned | **E10**, **E12**, **E16**, **E18** | pred E |

Cặp paired: C8−C1, C9−C1, C8−C9, E2−B4_OLS, E8−E7, E14−E8, E16−E10, E18−E12, E16−C8.

Multi-seed chỉ **E2** và **E16**, seeds `{0,1,2,3,4}` → `outputs/F/` (**không** ghi đè `outputs/E/` seed-0, **không** ghi `checkpoints/`). Val speakers **cố định** theo seed 0; chỉ RNG train đổi.

Câu hỏi nghiên cứu:

> Khi cố định dataset và official test phones, correlation / ΔPCC giữa các headline models có CI ổn định không, và residual lớn gắn với confusion / accent markup thế nào?

---

## 1. Protocol lock

| Thành phần | Giá trị Group F |
| ---------- | --------------- |
| Dataset | Speechocean762 |
| Level | phoneme |
| Test split | official test |
| Metrics | PCC, SCC, MAE, MSE (+ bootstrap CI) |
| Bootstrap | phone-level, `n_boot=1000`, seed 0 |
| CI | percentile 95% |
| Score range | human / pred clip [0, 2] |
| Seed (F1a/F1b) | 0 |
| Multi-seed | E2, E16; seeds 0–4; val speakers locked |

Config: `configs/f_validation.yaml`.

F **không** dùng C2–C7 (negative control). Không claim F thay Group D.

---

## 2. Input đã khóa

| File | Vai trò |
| ---- | ------- |
| `outputs/C/c_predictions.csv` + `c_results.json` | C1/C8/C9 GOP + maps |
| `outputs/B/b_predictions.csv` + `b_results.json` | B4 features + OLS coef |
| `outputs/E/e_*.csv` + `e_results.json` | pred E2/E7/E8/E10/E12/E14/E16/E18 |
| `outputs/D/d2_speaker_metrics.csv` | extreme-MAE speakers (T5) |
| `data/speechocean762/resource/scores-detail.json` | expert `{}` / `()` / `[]` markup |
| `configs/e_learned_scoring.yaml` | hyperparams multi-seed |

Sanity: `n_test` join = 47 369 (cùng A2).

---

## 3. Experiment cards

```text
Experiment ID: F1
Research Question: How uncertain are headline PCC/SCC, and which model deltas
are distinguishable from zero under paired phone bootstrap?
Hypothesis: Large locked deltas (e.g. E16 vs C8, C8 vs C1) have 95% CIs
excluding zero; multi-seed std(E2/E16) << cross-model gaps.
Independent Variable: none (validation of locked scores)
Controlled: official test phones, frozen maps / seed-0 preds for F1a/F1b
```

```text
Experiment ID: F2
Research Question: What failure modes dominate largest residuals for C8 and E16?
Hypothesis: Large errors concentrate on expert () / {} markup and acoustic
competitor wins, not uniform noise; sequence (E16) reduces some C8 context errors.
Independent Variable: none (descriptive taxonomy)
Models: C8 (best direct), E16 (best learned on plan scope)
```

---

## 4. Pipeline

```text
Locked A/B/C/E artifacts
        │
        ▼
Join test phones → outputs/F/f_predictions.csv
        │
        ├── F1a  bootstrap CI per model
        ├── F1b  paired Δ bootstrap
        ├── F1c  multi-seed E2/E16 → outputs/F/ only
        └── F2   residuals + scores-detail taxonomy
                │
                ▼
        outputs/F/f_results.json
```

```text
python scripts/run_experiment.py --config configs/f_validation.yaml
python scripts/run_experiment.py --config configs/f_validation.yaml --skip-multiseed
```

Notebook: `notebooks/F_validation.ipynb` — **chỉ đọc artifact**.

---

## 5. F2 taxonomy

Speechocean762 `scores-detail` phone strings:

| Markup | Meaning |
| ------ | ------- |
| bare | score ≈ 2 (correct) |
| `{PH}` | score ≈ 1 (accent) |
| `(PH)` | score ≈ 0 (incorrect/missing) |
| `[PH]` | insertion |

Heuristic `primary_type` (có thể multi-label):

| Type | Rule |
| ---- | ---- |
| T2 acoustic confusion | human ≤ 1 và (competitor LPP > canonical **hoặc** expert `()` / `[]`) |
| T3 accent | expert `{}` **hoặc** (human ≈ 1 và pred ≥ 1.5) |
| T4 context | `|err_self| − |err_other| ≥ gap` (C8: other=E16; E16: other=C8) |
| T5 speaker | speaker trong top MAE quantile (D2) |
| T1 alignment | residual lớn + `n_frames_c8` bất thường; không claim chắc nếu chưa nghe audio |

---

## 6. Code map

| File | Vai trò |
| ---- | ------- |
| `configs/f_validation.yaml` | protocol |
| `src/gop_empirical/eval/stats.py` | bootstrap / paired Δ |
| `src/gop_empirical/eval/errors.py` | residuals + taxonomy |
| `src/gop_empirical/data/scores_detail.py` | parse expert markup |
| `src/gop_empirical/experiment.py` | `run_group_f` |
| `tests/test_validation.py` | unit + smoke |
| `notebooks/F_validation.ipynb` | plots từ artifact |

---

## 7. Artifact

```text
outputs/F/
  f_predictions.csv
  f_results.json
  f2_residuals_c8.csv
  f2_residuals_e16.csv
  f2_top_errors.csv
  f1_multiseed_e2_seed{k}.csv
  f1_multiseed_e16_seed{k}.csv
```

---

## 8. Kết quả (run hiện tại)

Nguồn: `outputs/F/f_results.json`. Official test n = 47 369. Bootstrap phone-level, n_boot = 1000, seed 0. **PCC/SCC trên score-space** (direct = map train-only rồi clip [0, 2]; learned = pred clip). Vì clip, C1 PCC score-space (0.342) hơi khác A2 raw-GOP PCC (0.323).

### F1a — Bootstrap CI (PCC)

| Model | PCC | 95% CI | SCC |
| ----- | --: | ------ | --: |
| C1 | 0.342 | [0.333, 0.351] | 0.313 |
| B4_OLS | 0.351 | [0.342, 0.361] | 0.342 |
| C9 | 0.430 | [0.420, 0.440] | 0.334 |
| C8 | 0.463 | [0.453, 0.473] | 0.355 |
| E7 | 0.446 | [0.433, 0.460] | 0.350 |
| E2 | 0.510 | [0.495, 0.523] | 0.368 |
| E8 | 0.530 | [0.516, 0.543] | 0.379 |
| E12 | 0.624 | [0.611, 0.637] | 0.380 |
| E14 | 0.625 | [0.612, 0.637] | 0.403 |
| E10 | 0.639 | [0.626, 0.651] | 0.415 |
| E18 | 0.653 | [0.639, 0.664] | 0.427 |
| **E16** | **0.671** | **[0.660, 0.683]** | 0.412 |

### F1b — Paired ΔPCC

Tất cả 9 cặp có `ci_excludes_zero = true`.

| Contrast | ΔPCC | 95% CI |
| -------- | ---: | ------ |
| C8 − C1 | +0.121 | [0.111, 0.131] |
| C9 − C1 | +0.088 | [0.077, 0.098] |
| C8 − C9 | +0.033 | [0.027, 0.040] |
| E2 − B4_OLS | +0.158 | [0.147, 0.168] |
| E8 − E7 | +0.084 | [0.072, 0.096] |
| E14 − E8 | +0.095 | [0.086, 0.104] |
| E16 − E10 | +0.033 | [0.026, 0.039] |
| E18 − E12 | +0.028 | [0.020, 0.038] |
| **E16 − C8** | **+0.208** | **[0.198, 0.217]** |

### F1c — Multi-seed (val speakers khóa seed 0)

Ghi `outputs/F/f1_multiseed_*.csv` — **không** overwrite `outputs/E/` hay `checkpoints/`.

| Model | PCC mean ± std | min / max |
| ----- | -------------: | --------- |
| E2 | 0.509 ± 0.001 | 0.507 / 0.510 |
| E16 | 0.665 ± 0.005 | 0.659 / 0.670 |

Std multi-seed ≪ khoảng cách model (vd. E16−C8 ≈ 0.21).

### F2 — Error taxonomy (primary_type)

| Type | C8 | E16 |
| ---- | -: | --: |
| other | 40 025 | 40 205 |
| T3 accent | 4 516 | 4 545 |
| T2 confusion | 2 149 | 2 149 |
| T4 context | 402 | 176 |
| T5 speaker | 202 | 186 |
| T1 alignment | 75 | 108 |

T4 = model đang xét xấu hơn model kia ≥ 0.5 abs-err (C8: sequence E16 giúp; E16: sequence làm xấu hơn direct C8). Mean \|err\| theo human bin: human≈0 cao hơn rõ (C8 1.31 → E16 0.91); human≈2 thấp (C8 0.122 → E16 0.089).

Finding:

> Headline model rankings are stable under phone bootstrap: every planned paired ΔPCC excludes zero. Training-seed variance for E2/E16 is small relative to cross-model gaps. Large residuals concentrate on accent/confusion markup more than uniform noise; E16 reduces error especially on low human scores.
