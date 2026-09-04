# A — Traditional GOP

Group A là **baseline** của thesis *An Empirical Study of English Pronunciation Assessment Using Goodness of Pronunciation Scoring*. Hai thí nghiệm chạy **cùng một pipeline**:

| ID | Tên | Việc làm |
| -- | --- | -------- |
| **A1** | Traditional GOP | Lấy scalar GOP = log phone posterior (LPP) của phoneme canonical |
| **A2** | GOP vs Human Score | Đo mức GOP phản ánh điểm chuyên gia (PCC, SCC, MAE, MSE) |

A2 không phải pipeline mới. A2 đọc scalar A1 và so với human phoneme score trên **official test split**.

Câu hỏi nghiên cứu:

> GOP truyền thống (canonical LPP) phản ánh human pronunciation score ở mức độ nào, khi cố định dataset, alignment, acoustic model và cách scoring?

---

## 1. Protocol lock

Các trường dưới đây **không đổi** giữa Group A và các group sau, trừ khi thí nghiệm đó *đang nghiên cứu đúng nhân tố đó*.

| Thành phần | Giá trị Group A |
| ---------- | --------------- |
| Dataset | Speechocean762 |
| Sampling rate | 16 kHz |
| Level | phoneme |
| Alignment | Kaldi (đã có trong extract GOP) |
| Acoustic model | Kaldi Librispeech M13 |
| GOP type | `standard` (canonical LPP) |
| Scoring | `direct` (không MLP / GOPT) |
| Test split | official train / test của Speechocean762 |
| Metrics | PCC, SCC trên GOP thô; MAE, MSE sau map tuyến tính train-only |
| Seed | 0 |
| Score range | human 0–2; floor 0.1 |
| Phone index | `phone_index_base: 0` |

Config: `configs/a_traditional_gop.yaml`.

---

## 2. Công thức và những gì repo thực sự tính

Witt & Young GOP kinh điển là trung bình log-posterior trên các frame của phoneme canonical:

\[
\mathrm{GOP}(p)=\frac{1}{T_p}\sum_{t \in p}\log P(p \mid x_t)
\]

**Repo này không tính GOP từ waveform hay từ posterior từng frame.** Group A đọc vector GOP **đã extract** bởi recipe Kaldi / GOPT (`extract_gop_feats.py`), layout Xiaomi:

```text
feat = [phone_id, LPP_0 … LPP_41, LPR_0 … LPR_41]     # 85 cột
GOP  = LPP[phone_id]                                   # phone_index_base = 0
```

- LPP = log phone posterior của từng slot trong 42-phone set.
- LPR được parse nhưng **A1 không dùng** (dành cho Group B).
- `phone_id` Kaldi trên extract này đã là index 0-based vào vector LPP (ids thường 3…41). `phone_index_base: 1` làm PCC train ≈ 0 — pipeline sẽ fail nếu indexing sai.
- Giá trị GOP trên Speechocean762 extract này **dương, khoảng 3–9**. Đó là LPP precomputed, không phải `mean log P(p|x_t)` âm như công thức textbook.

**Cách viết trong thesis:** Kaldi precomputed canonical LPP. Không viết “pipeline này tính GOP frame-level”, và không viết “Kaldi trực tiếp output phoneme posterior” nếu không nêu mapping senone → phone (việc đó nằm trong recipe extract, ngoài repo này).

---

## 3. Dữ liệu

Nằm trong `GOP-Empirical-Study/data/` (copy từ GOPT `kaggle_upload_raw_kaldi_gop`; xem `data/SOURCE.txt`).

| File | Vai trò |
| ---- | ------- |
| `data/scores.json` | điểm chuyên gia Speechocean762 |
| `data/kaldi_gop_librispeech/tr_feats.csv` | GOP train, 85 cột |
| `data/kaldi_gop_librispeech/te_feats.csv` | GOP test |
| `data/kaldi_gop_librispeech/tr_keys_phn.csv` | khóa `utt_id.phn_idx` train |
| `data/kaldi_gop_librispeech/te_keys_phn.csv` | khóa test |

Join key: `utt_id.phn_idx` (ví dụ `000010011.0`).

Human phoneme score Speechocean762: **0–2** (2 = đúng, 1 = đúng nhưng accent nặng, 0 = sai / mất phoneme), làm tròn theo `score_floor: 0.1`. Tên phone bỏ stress giống Xiaomi: `IY0` → `IY`.

Bỏ qua silence / noise nếu có: `SIL`, `SPN`, `NSN`, `<eps>` (không phân biệt hoa thường). Run A hiện tại: **0** phone bị skip, **0** key thiếu human score.

Không dùng (Group A không load): `{tr,te}_labels_*.csv`, `{tr,te}_keys_word.csv`.

---

## 4. Pipeline

```text
Kaldi GOP CSV (feats + keys)
        │
        ▼
A1  GOP = LPP[canonical phone_id]
        │
        ▼
Join scores.json  (utt_id.phn_idx)
        │
        ├── skip silence
        └── write outputs/A/a1_predictions.csv
                │
                ▼
A2  test:  PCC / SCC  trên GOP thô
    train: fit  human ≈ slope · GOP + intercept
    test:  MAE / MSE  sau map, clip [0, 2]
                │
                ▼
        outputs/A/a2_results.json
```

Một lần chạy cho cả A1 và A2:

```text
python scripts/run_experiment.py --config configs/a_traditional_gop.yaml
```

Môi trường: conda `gop` (xem `environment.yml`). Notebook phân tích: `notebooks/A2_gop_vs_human.ipynb` — **chỉ đọc artifact**, không tính lại GOP.

---

## 5. A1 — Traditional GOP

**Input:** feats + keys + `scores.json`.  
**Output:** một hàng mỗi phoneme đã align:

```csv
utt_id,split,word_id,phone_id,phone,gop,human_score
000010011,train,0,0,W,5.81848,2.0
```

`word_id` / `phone_id` lấy từ `scores.json` (thứ tự word trong câu, thứ tự phone trong word). `split` là official train/test.

**Kiểm tra indexing:** LPR[canonical] ≈ 0 trên extract này; PCC train với `phone_index_base: 0` ≈ 0.34. Nếu base = 1 thì PCC ≈ 0 và `run_group_a` raise.

Unit test indexing + no-leak mapping: `tests/test_traditional_gop.py`.

---

## 6. A2 — GOP vs human

Trên **test** (báo cáo chính). Train chỉ dùng để (1) fit map MAE/MSE và (2) sanity-check PCC.

| Metric | Ý nghĩa | Input |
| ------ | ------- | ----- |
| **PCC** | tương quan tuyến tính | GOP thô vs human |
| **SCC** | tương quan thứ hạng | GOP thô vs human |
| **MAE / MSE** | sai số nếu coi GOP như bộ dự đoán điểm | \(\widehat{y} = \mathrm{clip}(a\cdot\mathrm{GOP}+b,\,0,\,2)\), \(a,b\) **fit trên train only** |

Không fit map trên test. MLP / GOPT không vào Group A.

Notebook vẽ scatter GOP–human và histogram GOP theo human score → `outputs/A/a2_scatter_gop_vs_human.png`, `outputs/A/a2_gop_histogram.png`.

---

## 7. Kết quả (run hiện tại)

Nguồn: `outputs/A/a2_results.json`.

| | Train | Test (báo cáo) |
| - | ----: | -------------: |
| n phones | 47 076 | 47 369 |
| PCC | 0.342 | **0.323** |
| SCC | 0.329 | **0.313** |
| MAE | 0.219 | **0.199** |
| MSE | 0.152 | **0.120** |

- Utterances: 2500 / 2500 (train / test), 0 overlap, 0 missing join, 0 sil skipped, 39 phone symbols sau khi bỏ stress.
- Map train-only: slope ≈ 0.219, intercept ≈ 0.444.
- Mean GOP tăng đơn điệu theo human score; khoảng 81% phone test có human = 2.0 (class lệch mạnh → MAE thấp một phần vì phần lớn điểm đã ở trần 2).

Đây là **baseline reference** cho B–F. So sánh group sau với cùng split, cùng human scores, cùng script eval.

---

## 8. Code map

| File | Vai trò |
| ---- | ------- |
| `configs/a_traditional_gop.yaml` | protocol + đường dẫn |
| `src/gop_empirical/gop/traditional.py` | `GOP = LPP[phone_id]` |
| `src/gop_empirical/data/kaldi.py` | load CSV 85 cột |
| `src/gop_empirical/data/scores.py` | load / làm tròn human score |
| `src/gop_empirical/eval/metrics.py` | PCC, SCC, map tuyến tính, MAE/MSE |
| `src/gop_empirical/experiment.py` | A1 join → A2 metrics |
| `scripts/run_experiment.py` | entry point |
| `notebooks/A2_gop_vs_human.ipynb` | scatter / histogram |

---

## 9. Artifact

```text
outputs/A/
  a1_predictions.csv              A1
  a2_results.json                 A2
  a2_scatter_gop_vs_human.png     notebook
  a2_gop_histogram.png            notebook
```

`a2_results.json` gồm protocol A1 (`n_train`, `n_test`, `n_missing_human`, `n_skipped_silence`) và số A2 (test + train + mapping).
