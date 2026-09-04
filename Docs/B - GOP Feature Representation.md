# B — GOP Feature Representation

Group B trả lời:

> **GOP nên được biểu diễn như thế nào?**

Không phải pipeline mới, và **không** đổi acoustic model / alignment / split / scoring. Chỉ đổi representation trên cùng extract Kaldi mà Group A đã dùng.

| ID | Tên | Feature trên extract này |
| -- | --- | ------------------------ |
| **B1** | Standard GOP | `LPP[phone_id]` — cùng scalar A1 |
| **B2** | LPP | `LPP[phone_id]` |
| **B3** | LPR vs best competitor | `LPP[p] − max_{q≠p} LPP[q]` |
| **B4** | GOP-only vector | `[LPP[p], max competitor LPP, LPR]`; OLS trên cặp rank-2 `[LPP[p], max competitor]` |
| **B5** | 84-d GOP feature | `[LPP_0..41, LPR_0..41]` (GOPT paper naming); OLS train-only |

**B1 ≡ B2 là finding, không phải bug.** Extract Xiaomi đã lưu canonical LPP như GOP scalar; không có posterior từng frame để tách “GOP aggregation” khỏi raw LPP.

Scoring vẫn `direct` (không MLP / Transformer GOPT). B1–B3: PCC/SCC trên scalar thô, MAE/MSE sau map tuyến tính train-only. B4/B5: OLS đa biến train-only, rồi PCC/SCC/MAE/MSE trên điểm dự đoán (clip [0, 2]).

Câu hỏi nghiên cứu:

> Khi cố định dataset, alignment, acoustic model và cách scoring, representation GOP nào (scalar LPP, LPR vs competitor, vector GOP-only, hay 84-d LPP+LPR) tương quan tốt hơn với human phoneme score?

---

## 1. Protocol lock

Giống Group A. Chỉ `gop_type` / representation thay đổi.

| Thành phần | Giá trị Group B |
| ---------- | --------------- |
| Dataset | Speechocean762 |
| Sampling rate | 16 kHz |
| Level | phoneme |
| Alignment | Kaldi (đã có trong extract GOP) |
| Acoustic model | Kaldi Librispeech M13 |
| Scoring | `direct` (B4/B5 = OLS tuyến tính, không MLP) |
| Test split | official train / test Speechocean762 |
| Metrics | PCC, SCC; MAE, MSE sau map train-only |
| Seed | 0 |
| Score range | human 0–2; floor 0.1 |
| Phone index | `phone_index_base: 0` |

Config: `configs/b_gop_representation.yaml`.

---

## 2. Công thức trên extract này

Layout Xiaomi (85 cột), parser dùng chung với A1:

```text
feat = [phone_id, LPP_0 … LPP_41, LPR_0 … LPR_41]
```

Precomputed:

\[
\mathrm{LPR}[q] = \mathrm{LPP}[\mathrm{canonical}] - \mathrm{LPP}[q]
\]

nên `LPR[canonical] = 0`. **Không dùng slot LPR canonical làm B3** (PCC ≈ 0).

| ID | Công thức |
| -- | --------- |
| B1 / B2 | \(\mathrm{GOP}(p)=\mathrm{LPP}[p]\) |
| B3 | \(\mathrm{LPR}(p)=\mathrm{LPP}[p]-\max_{q\neq p}\mathrm{LPP}[q]=\min_{q\neq p}\mathrm{LPR}[q]\) |
| B4 stored | \([\mathrm{LPP}[p],\;\max_{q\neq p}\mathrm{LPP}[q],\;\mathrm{LPR}(p)]\) |
| B4 OLS | hai cột đầu (cột 3 = cột 1 − cột 2) |
| B5 | \([\mathrm{LPP}_0..\mathrm{LPP}_{41},\;\mathrm{LPR}_0..\mathrm{LPR}_{41}]\) (84-d; OLS) |

GOPT paper gọi vector này **84-dimensional GOP feature**. Extract Xiaomi lưu LPR đối dấu so với `LPR(p_j|p)=\mathrm{LPP}(p_j)-\mathrm{LPP}(p)`; OLS hấp thụ dấu. Design matrix rank-deficient (LPR tuyến tính theo LPP khi biết canonical) — vẫn OLS như B4 (sklearn lstsq).

Không thêm duration / entropy. MLP là Group E. B5 **không** phải full GOPT (không phone embed / Transformer / multi-task).

---

## 3. Dữ liệu

Cùng `data/` với Group A. Join key `utt_id.phn_idx`. Skip silence giống A: `SIL`, `SPN`, `NSN`, `<eps>`.

---

## 4. Pipeline

```text
Kaldi GOP CSV (feats + keys)
        │
        ▼
One pass: LPP vector → B1/B2/B3/B4; 84-d → B5 (in memory)
        │
        ▼
Join scores.json  (utt_id.phn_idx)
        │
        ├── skip silence
        └── write outputs/B/b_predictions.csv  (+ b5_pred; không dump 84 cột)
                │
                ▼
B1–B3  test PCC/SCC on raw scalar
       train-only univariate map → MAE/MSE
B4     train-only OLS on [LPP, max competitor]
B5     train-only OLS on 84-d LPP+LPR
       PCC/SCC/MAE/MSE on clipped predictions
                │
                ▼
        outputs/B/b_results.json
```

Một lần chạy cho cả B1–B5:

```text
python scripts/run_experiment.py --config configs/b_gop_representation.yaml
```

Notebook: `notebooks/B_gop_representation.ipynb` — **chỉ đọc artifact**, không tính lại GOP.

---

## 5. Experiment cards

```text
Experiment ID: B1
Research Question: Canonical LPP (standard GOP) vs human score, same protocol as A1?
Hypothesis: B1 reproduces A2 (test PCC ≈ 0.323).
Independent Variable: GOP representation = canonical LPP
Controlled: dataset, alignment, AM, split, direct scoring
```

```text
Experiment ID: B2
Research Question: Raw canonical LPP vs standard GOP on this extract?
Hypothesis: B2 ≡ B1 because the extract stores GOP as canonical LPP.
Independent Variable: GOP representation = LPP
```

```text
Experiment ID: B3
Research Question: Does beating the strongest competing phone predict human scores better than canonical LPP alone?
Hypothesis: LPR vs best competitor captures a different (competition) signal than raw LPP.
Independent Variable: GOP representation = LPR vs max competitor
```

```text
Experiment ID: B4
Research Question: Does a GOP-only feature vector improve the linear mapping to human scores?
Hypothesis: Linear combination of canonical LPP and max-competitor LPP can match or beat the best scalar.
Independent Variable: GOP representation = rank-2 GOP-only vector
Scoring: still direct (multivariate OLS, train-only)
```

```text
Experiment ID: B5
Research Question: Does the GOPT-paper 84-d GOP feature (LPP+LPR concat)
improve linear mapping to human phoneme scores vs B4?
Hypothesis: B5 test PCC ≥ B4 (0.351), because full confusion profile
beats the rank-2 competitor summary.
Independent Variable: GOP representation = 84-d LPP+LPR
Controlled: dataset, AM, alignment, split, direct OLS scoring
Not claimed: full GOPT (no phone embed / Transformer / multi-task)
```

---

## 6. Code map

| File | Vai trò |
| ---- | ------- |
| `configs/b_gop_representation.yaml` | protocol + đường dẫn |
| `src/gop_empirical/gop/representation.py` | B1–B4 + `gopt_gop_feature_84` (B5) |
| `src/gop_empirical/eval/metrics.py` | OLS đa biến train-only |
| `src/gop_empirical/experiment.py` | `run_group_b` |
| `scripts/run_experiment.py` | entry point |
| `tests/test_gop_representation.py` | công thức, collinearity, B5 shape, no-leak |
| `notebooks/B_gop_representation.ipynb` | so sánh B1–B5 vs A2 |

---

## 7. Kết quả (run hiện tại)

Nguồn: `outputs/B/b_results.json`. Test n = 47 369 (cùng A2). `b1_identical_to_b2: true`. B1 khớp A2 (PCC 0.323).

| | PCC | SCC | MAE | MSE |
| - | --: | --: | --: | --: |
| **B1** Standard GOP | 0.323 | 0.313 | 0.199 | 0.120 |
| **B2** LPP | 0.323 | 0.313 | 0.199 | 0.120 |
| **B3** LPR vs competitor | 0.328 | **0.342** | 0.198 | 0.121 |
| **B4** GOP-only vector (OLS) | 0.351 | 0.342 | 0.197 | 0.120 |
| **B5** 84-d LPP+LPR (OLS) | **0.361** | 0.332 | **0.197** | **0.119** |

- B4 mapping train-only: `score ≈ 0.223 · LPP[p] − 0.147 · max_competitor + 1.355`, clip [0, 2].
- B5: OLS trên 84-d; `n_features: 84`; cột `b5_pred` trong CSV (không dump 84-d).
- Finding: trên extract này LPP raw không khác standard GOP; LPR cạnh tranh cải thiện SCC; vector GOP-only (B4) cải thiện PCC so với scalar; **84-d GOPT-style (B5) tăng PCC nhẹ so với B4** (+0.010) nhưng **SCC thấp hơn** B3/B4 — full confusion profile giúp linear correlation hơn rank correlation dưới OLS.

---

## 8. Artifact

```text
outputs/B/
  b_predictions.csv          (+ b5_pred)
  b_results.json
  b_comparison_pcc.png
  b_scatter_b3_lpr_vs_human.png
```

`b_results.json` gồm `protocol` (n_train, n_test, `b1_identical_to_b2`) và `comparison.test` (PCC/SCC/MAE/MSE từng B1–B5).
