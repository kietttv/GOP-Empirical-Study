# D — GOP Behavior Analysis

Group D trả lời:

> **GOP hành xử thế nào theo phoneme, speaker, và mức điểm câu?**

Không phải pipeline GOP mới, và **không** đợi Group C. D đọc artifact A đã khóa rồi chỉ đổi cách group. Independent variable là **stratum**, không phải acoustic model hay công thức GOP.

| ID | Tên | Grouping |
| -- | --- | -------- |
| **D1** | Phone-level | `phone` (39 CMU) |
| **D2** | Speaker-level | `speaker` (`utt2spk`) |
| **D3** | Score strata | tertile speaker-mean sentence `accuracy` |

Speechocean762 **không** có nhãn proficiency / CEFR. D3 **không** viết Beginner / Intermediate / Advanced. Protocol: không suy ra proficiency từ điểm rồi gọi đó là ground truth.

Câu hỏi nghiên cứu:

> Khi cố định dataset, alignment Kaldi, AM Kaldi M13, GOP chuẩn và direct scoring, correlation GOP–human có ổn định giữa các phoneme, speakers, và speaker-score strata không?

---

## 1. Protocol lock

Giống Group A. Chỉ cách stratify thay đổi. GOP lấy từ A1; map MAE/MSE đóng băng từ A2 (train-only).

| Thành phần | Giá trị Group D |
| ---------- | --------------- |
| Dataset | Speechocean762 |
| Sampling rate | 16 kHz |
| Level | phoneme |
| Alignment | Kaldi (cùng A1) |
| Acoustic model | Kaldi Librispeech M13 |
| GOP type | `standard` (canonical LPP từ A1) |
| Scoring | `direct` (map A2, không fit lại từng group) |
| Test split | official train / test Speechocean762 |
| Metrics | PCC, SCC trên GOP thô; MAE, MSE sau map A2 |
| Seed | 0 |
| Score range | human 0–2; floor 0.1 |
| Min N (phone) | 100 |
| Min N (speaker) | 50 |
| Min N (stratum) | 50 |

Config: `configs/d_gop_behavior.yaml`.

Báo cáo trên **official test** (giống A2). Train chỉ để (1) lấy slope/intercept A2 và (2) sanity PCC tổng.

Nếu `n < min_n` hoặc variance GOP/human = 0 → `pcc`/`scc` = null; vẫn ghi N / mean GOP / MAE. Hàng dưới ngưỡng vẫn có trong CSV, không đưa vào finding (`reported = false`).

Không dùng GOP C2/C3. Lặp D theo AM là việc sau khi C khóa.

---

## 2. Input đã khóa

D **không** tính lại GOP.

| File | Vai trò |
| ---- | ------- |
| `outputs/A/a1_predictions.csv` | phone, GOP, human, split |
| `outputs/A/a2_results.json` | mapping train-only + PCC sanity |
| `data/speechocean762/{train,test}/utt2spk` | speaker |
| `data/speechocean762/{train,test}/spk2age` | age (mô tả, không phải proficiency) |
| `data/scores.json` | sentence-level `accuracy` 0–10 |

Sanity: test PCC của D trên toàn bộ A1 phải khớp A2 (≈ 0.323, n = 47 369), `abs(Δ) < 1e-6`.

---

## 3. Experiment cards

```text
Experiment ID: D1
Research Question: Does GOP–human correlation differ across English phonemes?
Hypothesis: Vowels correlate more strongly than some consonants (e.g. /TH/, /R/).
Independent Variable: phone identity
Controlled: dataset, alignment, AM, standard GOP, split, A2 mapping
Min N: 100 phones / symbol
```

```text
Experiment ID: D2
Research Question: Is GOP–human correlation stable across speakers?
Hypothesis: Per-speaker PCC has non-trivial spread; GOP is not equally calibrated for every learner.
Independent Variable: speaker id
Controlled: same as D1
Min N: 50 phones / speaker
Note: official SO762 split is speaker-independent.
```

```text
Experiment ID: D3
Research Question: Does GOP–human correlation change across speaker sentence-accuracy strata?
Hypothesis: PCC differs between Low / Mid / High speaker-mean sentence accuracy.
Independent Variable: score_stratum (tertiles of speaker-mean sentence accuracy on test speakers)
Controlled: same as D1
Not proficiency: no CEFR label; do not name groups Beginner/Advanced.
```

---

## 4. Pipeline

```text
outputs/A/a1_predictions.csv
outputs/A/a2_results.json   (frozen mapping)
        │
        ▼
Join utt2spk / spk2age / sentence accuracy
        │
        ├── skip if A1 missing
        └── write outputs/D/d_predictions.csv
                │
                ▼
D1  groupby phone          (test; min_n=100)
    + vowel vs consonant rollup
D2  groupby speaker        (test; min_n=50)
    + train/test speaker overlap check
D3  tertile test speakers by mean sentence accuracy
    then GOP ↔ human within Low / Mid / High
                │
                ▼
        outputs/D/d_results.json
```

Một lần chạy cho cả D1–D3:

```text
python scripts/run_experiment.py --config configs/d_gop_behavior.yaml
```

Môi trường: conda `gop`. Notebook: `notebooks/D_gop_behavior.ipynb` — **chỉ đọc artifact**, không tính lại GOP.

MAE/MSE mọi stratum dùng **cùng** map A2:

\[
\widehat{y}=\mathrm{clip}(a\cdot\mathrm{GOP}+b,\,0,\,2)
\]

\(a,b\) lấy từ A2, **không** fit lại trên phone / speaker / stratum.

---

## 5. D3 cutpoints

Trên **test speakers** (không dùng test phone rows để định nghĩa speaker mean — trung bình `accuracy` theo utterance, rồi trung bình các utterance của speaker):

```text
q33, q66 = quantiles of speaker-mean sentence accuracy
Low  : mean_acc <= q33
Mid  : q33 < mean_acc <= q66
High : mean_acc > q66
```

Human phoneme score trong mỗi stratum vẫn biến thiên 0–2, nên PCC có nghĩa. Không group theo phoneme score 0/1/2 (variance human ≈ 0 → PCC rác).

---

## 6. Code map

| File | Vai trò |
| ---- | ------- |
| `configs/d_gop_behavior.yaml` | protocol + đường dẫn |
| `src/gop_empirical/data/speakers.py` | `utt2spk` / `spk2age` |
| `src/gop_empirical/data/scores.py` | `load_utterance_scores` |
| `src/gop_empirical/eval/behavior.py` | D1/D2/D3 tables, tertiles, vowel/consonant |
| `src/gop_empirical/experiment.py` | `run_group_d` |
| `scripts/run_experiment.py` | entry point |
| `tests/test_gop_behavior.py` | min_n, constant-human, tertiles, no remapping |
| `notebooks/D_gop_behavior.ipynb` | bar / histogram từ artifact |

---

## 7. Artifact

```text
outputs/D/
  d_predictions.csv
  d1_phone_metrics.csv
  d2_speaker_metrics.csv
  d3_stratum_metrics.csv
  d_results.json
  d1_phone_pcc.png              notebook
  d2_speaker_pcc_hist.png       notebook
  d3_stratum_pcc.png            notebook
```

`d_predictions.csv` = A1 + `speaker` + `age` + `sentence_accuracy` + `score_stratum`.

`d_results.json` gồm `protocol` (n, speaker overlap, `sanity_matches_a2`, mapping A2) và block D1/D2/D3.

---

## 8. Kết quả (run hiện tại)

Nguồn: `outputs/D/d_results.json`. Sanity vs A2: test PCC **0.323**, n = 47 369, speaker overlap train/test = 0. Map A2 đóng băng: slope ≈ 0.219, intercept ≈ 0.444.

### D1 — Phone

37 / 39 phones đạt `N ≥ 100`. Không báo cáo **/OY/** (n = 20) và **/ZH/** (n = 19).

| | N | PCC | SCC | MAE |
| - | --: | --: | --: | --: |
| **CONSONANT** | 28 386 | **0.343** | 0.314 | 0.184 |
| **VOWEL** | 18 983 | 0.307 | 0.310 | 0.223 |
| All test (A2) | 47 369 | 0.323 | 0.313 | 0.199 |

PCC cao nhất (reported): **SH 0.626**, **CH 0.616**, **EY 0.580**, M 0.480, AO 0.474.

PCC thấp nhất (reported): **AE 0.273**, HH 0.288, AH 0.288, B 0.292, Y 0.293.

/TH/ PCC = 0.335 (n = 348), /R/ = 0.384 (n = 1200) — không phải phoneme yếu nhất trên extract này. Hypothesis “vowel tốt hơn consonant” **không** được hỗ trợ: consonant rollup PCC cao hơn vowel.

### D2 — Speaker

125 test speakers, tất cả `N ≥ 50` (0 missing age).

| | |
| - | --: |
| n speakers (train / test) | 125 / 125 |
| speaker overlap | **0** |
| mean per-speaker PCC | **0.252** |
| std / min / max PCC | 0.084 / 0.078 / 0.539 |
| mean per-speaker MAE | 0.200 |

PCC gộp A2 (0.323) cao hơn trung bình PCC từng speaker (0.252). GOP không calibrate đều: khoảng PCC speakers trải 0.08–0.54.

### D3 — Score strata (không phải proficiency)

Tertile speaker-mean sentence `accuracy` trên test speakers: q33 ≈ **7.72**, q66 ≈ **8.35** (thang 0–10).

| Stratum | Speakers | N phones | Mean human | PCC | SCC | MAE |
| ------- | -------: | -------: | ---------: | --: | --: | --: |
| **Low** | 42 | 16 002 | 1.730 | **0.363** | 0.351 | 0.319 |
| **Mid** | 46 | 16 537 | 1.937 | 0.228 | 0.238 | 0.157 |
| **High** | 37 | 14 830 | 1.961 | 0.207 | 0.206 | 0.117 |

PCC cao nhất ở stratum Low (human còn biến thiên). High gần trần điểm 2 → PCC/MAE thấp hơn vì variance human nhỏ, không vì “GOP tốt hơn với learner giỏi.” Không gọi các nhóm này là Beginner/Advanced.

Finding:

> GOP–human correlation is phoneme-dependent and speaker-dependent. It is strongest among lower sentence-accuracy speakers, where human scores still vary, and weaker where scores pile up at ceiling.

