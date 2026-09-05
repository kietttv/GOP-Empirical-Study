# E — GOP Learned Scoring

Group E trả lời:

> **GOP có thể được học lại thành pronunciation score tốt hơn direct / linear scoring không?**

Không phải pipeline GOP mới. E **không** tính lại posterior. E1/E2 **không** dùng 84-d GOPT paper. Follow-up **E3–E6** đọc GOP-S C8/C9. Follow-up **E19–E22** đọc GOP-SD C10/C11 (cùng AM, graph AF-S → AF-SD). Follow-up **E7–E12** = 1 representation (LPP+LPR concat) × 3 AM × 2 scorer. Follow-up **E13/E14** = cùng 84-d Kaldi với E7/E8, thêm canonical phone embedding (không overwrite E7/E8). Follow-up **E15–E18** = cùng 78-d C8/C9 với E9–E12, thêm SSL 39-way phone embed (không overwrite E9–E12).

| ID | Tên | Model | Input |
| -- | --- | ----- | ----- |
| **E1** | GOP + MLP | MLP phone-independent | cùng X với E2 (Kaldi B4) |
| **E2** | GOP + Transformer | Transformer trên chuỗi phone trong câu | cùng X với E1 |
| **E3** | C8 GOP-S + MLP | MLP (cùng E1) | scalar Cao GOP-S `gop_c8` |
| **E4** | C8 GOP-S + Transformer | Transformer (cùng E2) | cùng X với E3 |
| **E5** | C9 GOP-S + MLP | MLP | scalar Cao GOP-S `gop_c9` |
| **E6** | C9 GOP-S + Transformer | Transformer | cùng X với E5 |
| **E19** | C10 GOP-SD + MLP | MLP (cùng E1) | scalar Cao GOP-SD `gop_c10` (AF-SD) |
| **E20** | C10 GOP-SD + Transformer | Transformer (cùng E2) | cùng X với E19 |
| **E21** | C11 GOP-SD + MLP | MLP | scalar Cao GOP-SD `gop_c11` |
| **E22** | C11 GOP-SD + Transformer | Transformer | cùng X với E21 |
| **E7** | Kaldi 84-d + MLP | MLP (cùng E1) | B5 `[LPP_0..41, LPR_0..41]` |
| **E8** | Kaldi 84-d + Transformer | Transformer (cùng E2) | cùng X với E7 |
| **E9** | C8 LPP+LPR + MLP | MLP | 78-d (39 CMU-mapped IPA) |
| **E10** | C8 LPP+LPR + Transformer | Transformer | cùng X với E9 |
| **E11** | C9 LPP+LPR + MLP | MLP | 78-d (cùng recipe E9) |
| **E12** | C9 LPP+LPR + Transformer | Transformer | cùng X với E11 |
| **E13** | Kaldi 84-d + MLP + phone embed | MLP (cùng E7) + embed | cùng 84-d với E7/E8 |
| **E14** | Kaldi 84-d + Transformer + phone embed | Transformer (cùng E8) + embed | cùng X với E13 |
| **E15** | C8 78-d + MLP + phone embed | MLP (cùng E9) + embed | cùng 78-d với E9/E10 |
| **E16** | C8 78-d + Transformer + phone embed | Transformer (cùng E10) + embed | cùng X với E15 |
| **E17** | C9 78-d + MLP + phone embed | MLP (cùng E11) + embed | cùng 78-d với E11/E12 |
| **E18** | C9 78-d + Transformer + phone embed | Transformer (cùng E12) + embed | cùng X với E17 |

E1/E2: independent variable = kiến trúc scoring. Controlled: dataset, alignment, AM Kaldi M13, official test, target phoneme, **cùng GOP features**.

E3–E6: follow-up (cùng tinh thần C4–C9). So sánh công bằng: **E3 vs E4**, **E5 vs E6**. E19–E22: **E19 vs E20**, **E21 vs E22** trên GOP-SD; ablation graph **E3 vs E19** / **E4 vs E20** (cùng XLSR), **E5 vs E21** / **E6 vs E22** (cùng lv60). E7–E12: **E7 vs E8**, **E9 vs E10**, **E11 vs E12**. E13/E14: **E13 vs E14** trên cùng 84-d + embed; so với E7/E8 là ablation *có/không phone embed*. E15/E16 (C8) và E17/E18 (C9): cùng 78-d + SSL embed; ablation vs E9–E12. Không so PCC xuyên X (B4 ≠ 84-d ≠ 78-d ≠ GOP-S ≠ GOP-SD).

Câu hỏi nghiên cứu:

> Khi cố định representation GOP, nonlinear mapping (MLP) và sequence modeling (Transformer) có map GOP → human phoneme score tốt hơn direct / OLS không?

Finding đúng:

> A learned nonlinear scoring function improves the mapping from GOP features to human pronunciation scores.

Không viết: “MLP improved GOP itself.” E2 là **kiến trúc kiểu GOPT** trên feature GOP-only. E7/E8 dùng đúng *84-dimensional GOP feature* của paper, **không** phone embed. E13/E14 thêm embed trên cùng 84-d — **vẫn không** claim full GOPT (không multi-task word/utt). E15–E18 thêm SSL 39-way embed trên cùng 78-d (không dùng Kaldi 42-slot). E9–E12 analog 78-d. C8/C9 **không** gọi là 84-d.

---

## 1. Protocol lock

Giống Group A. Chỉ scoring thay đổi. Feature mặc định là B4 (Group B). Có thể đổi sang scalar A1 để so thẳng Direct GOP.

| Thành phần | Giá trị Group E |
| ---------- | --------------- |
| Dataset | Speechocean762 |
| Sampling rate | 16 kHz |
| Level | phoneme |
| Alignment | Kaldi (cùng A1) |
| Acoustic model | Kaldi Librispeech M13 |
| Default features | B4 `[LPP[p], max competitor LPP, LPR vs best]` |
| Optional features | A1 scalar; `c8`/`c9` GOP-S; `c10`/`c11` GOP-SD; `b5` 84-d; `c8_lpp_lpr`/`c9_lpp_lpr` 78-d; `*_embed` + phone embed |
| Scoring | learned (MLP / Transformer) |
| Test split | official Speechocean762 **test** (không đụng) |
| Validation | 20% **train speakers**, seed 0, speaker-independent |
| Metrics | PCC, SCC, MAE, MSE trên điểm **dự đoán** (clip [0, 2]) |
| Seed | 0 |
| Score range | human 0–2 |

Config: `configs/e_learned_scoring.yaml`.

**Không dùng (E1/E2):** 84-d LPP+LPR, phone embedding, duration, entropy, GOP C2/C3, word/utterance multi-task. E7/E8 **có** dùng 84-d, **không** embed. E13/E14 **có** 84-d + Kaldi canonical embed (42 slot). E9–E12 dùng analog 78-d (không dump 392-d full espeak vocab). E15–E18 dùng cùng 78-d + SSL scored-phone embed (39 slot, `inventory.ssl_index`). Still no word/utt heads.

B4 cột 3 thừa tuyến tính: \(\mathrm{LPR}=\mathrm{LPP}[p]-\max_{q\neq p}\mathrm{LPP}[q]\). Default **vẫn feed đủ 3 cột** (không drop). OLS Group B chỉ dùng 2 cột đầu; neural net nhận cả 3.

---

## 2. Input đã khóa

E1–E6 **không** đọc Kaldi CSV 85 cột. E7/E8 cắt 84-d từ extract Xiaomi (cùng B5, không tính lại posterior). E9–E12 đọc npz 78-d đã extract.

| File | Vai trò |
| ---- | ------- |
| `outputs/B/b_predictions.csv` | B4 vector + human (default); tồn tại = B đã chạy (E7/E8) |
| `outputs/B/b_results.json` | baseline B4 OLS và B5 OLS |
| `outputs/A/a1_predictions.csv` | scalar GOP khi `--features a1` |
| `outputs/A/a2_results.json` | baseline Direct GOP (không fit lại) |
| `outputs/C/c_predictions.csv` | keys + human; scalar `gop_c8`/`gop_c9`/`gop_c10`/`gop_c11` |
| `outputs/C/c_results.json` | baseline direct C8/C9/C10/C11 |
| `data/kaldi_gop_librispeech/` | 84-d LPP+LPR khi `--features b5` |
| `data/xlsr_espeak_lpp_lpr/` | 78-d C8 (CTC Viterbi frames; join `utt.phn_idx`) |
| `data/lv60_espeak_lpp_lpr/` | 78-d C9 (cùng recipe) |
| `data/speechocean762/{train,test}/utt2spk` | speaker id cho val split |

---

## 3. Feature sets

E1 và E2 **luôn cùng X**. E3/E4 cùng `gop_c8`. E5/E6 cùng `gop_c9`. E19/E20 cùng `gop_c10`. E21/E22 cùng `gop_c11`. E7/E8 cùng 84-d. E9/E10 cùng 78-d C8. E11/E12 cùng 78-d C9. E15/E16 cùng 78-d C8 + SSL embed. E17/E18 cùng 78-d C9 + SSL embed.

| Flag | X mỗi phone | dim | Linear baseline (frozen) |
| ---- | ----------- | --: | ------------------------ |
| `b4` (default) | `[LPP, max competitor, LPR]` | 3 | B4 OLS, test PCC ≈ 0.351 |
| `a1` | `[GOP]` canonical LPP | 1 | Direct GOP A2, test PCC ≈ 0.323 |
| `c8` | Cao GOP-S C8 (XLSR-53 espeak) | 1 | Direct C8, test PCC ≈ 0.463 |
| `c9` | Cao GOP-S C9 (lv60 espeak) | 1 | Direct C9, test PCC ≈ 0.430 |
| `c10` | Cao GOP-SD C10 (cùng AM C8, AF-SD) | 1 | Direct C10, test PCC ≈ 0.498 |
| `c11` | Cao GOP-SD C11 (cùng AM C9, AF-SD) | 1 | Direct C11, test PCC ≈ 0.463 |
| `b5` | `[LPP_0..41, LPR_0..41]` | 84 | B5 OLS, test PCC ≈ 0.361 |
| `b5_embed` | cùng 84-d + phone embed (E13/E14) | 84 | cùng B5 OLS (frozen) |
| `c8_lpp_lpr` | GOPT-style LPP+LPR, CTC-aligned (39 IPA) | 78 | OLS train-only trên cùng 78-d |
| `c9_lpp_lpr` | cùng recipe C8, AM lv60 | 78 | OLS train-only trên cùng 78-d |
| `c8_lpp_lpr_embed` | cùng 78-d C8 + SSL phone embed (E15/E16) | 78 | cùng C8 78-d OLS (frozen) |
| `c9_lpp_lpr_embed` | cùng 78-d C9 + SSL phone embed (E17/E18) | 78 | cùng C9 78-d OLS (frozen) |

Z-score fit trên phone **role=train** only, apply val/test.

```text
python scripts/run_experiment.py --config configs/e_learned_scoring.yaml
python scripts/run_experiment.py --config configs/e_learned_scoring.yaml --features a1
python scripts/run_experiment.py --config configs/e_learned_scoring.yaml --features b4 a1
python scripts/run_experiment.py --config configs/e_learned_scoring.yaml --features c8 c9
python scripts/run_experiment.py --config configs/e_learned_scoring.yaml --features c10 c11 --device cuda
python scripts/run_experiment.py --config configs/e_learned_scoring.yaml --features b5
python scripts/run_experiment.py --config configs/e_learned_scoring.yaml --features b5_embed
python scripts/extract_ssl_gop.py --config configs/c_acoustic_model.yaml --model xlsr_espeak --gop lpp_lpr
python scripts/extract_ssl_gop.py --config configs/c_acoustic_model.yaml --model lv60_espeak --gop lpp_lpr
python scripts/run_experiment.py --config configs/e_learned_scoring.yaml --features c8_lpp_lpr c9_lpp_lpr
python scripts/run_experiment.py --config configs/e_learned_scoring.yaml --features c8_lpp_lpr_embed c9_lpp_lpr_embed
```

`--features c8 c9` / `c10 c11` / `b5` / `b5_embed` / `c8_lpp_lpr` / `*_embed` **không** ghi đè `e_predictions.csv`. Merge E3–E22 vào `e_results.json`. `--features b5_embed` **không** train lại E7/E8. `--features c8_lpp_lpr_embed c9_lpp_lpr_embed` **không** train lại E9–E12. Embed SSL dùng 39 scored IPA (`ssl_index`), không Kaldi 42-slot. E19–E22 train CUDA (`--device cuda`); yaml default `device: cpu` giữ cho E1–E18.

---

## 4. Experiment cards

```text
Experiment ID: E1
Research Question: Does a nonlinear MLP map GOP features to human phoneme
scores better than the locked linear baseline?
Hypothesis: MLP PCC on test > B4 OLS (when features=b4) or > A2 (when features=a1).
Independent Variable: scoring model = MLP
Controlled: dataset, alignment, AM, split, target, GOP features (shared with E2)
Input: B4 vector (default) or A1 scalar
Output: predicted phoneme score, clip [0, 2]
```

```text
Experiment ID: E2
Research Question: Does sequence context improve GOP scoring beyond a
phone-independent MLP, given the same features?
Hypothesis: Transformer PCC on test ≥ MLP, because neighbouring phones carry context.
Independent Variable: scoring model = Transformer encoder (per-phone head)
Controlled: same as E1, including identical X
Not GOPT-paper input: no 84-d LPP+LPR, no phone embedding, no word/utt heads
```

```text
Experiment ID: E3
Research Question: Does a nonlinear MLP map C8 Cao GOP-S to human phoneme
scores better than frozen direct C8?
Hypothesis: E3 test PCC > C8 direct (PCC ≈ 0.463).
Independent Variable: scoring model = MLP
Controlled: dataset, split, target, AM = wav2vec2-xlsr-53-espeak-cv-ft,
            GOP = Cao GOP-S (shared with E4)
Input: scalar gop_c8
Output: predicted phoneme score, clip [0, 2]
```

```text
Experiment ID: E4
Research Question: Does sequence context improve C8 GOP-S scoring beyond MLP?
Hypothesis: E4 test PCC ≥ E3 on the same gop_c8 sequence.
Independent Variable: scoring model = Transformer encoder (per-phone head)
Controlled: same as E3, including identical X
```

```text
Experiment ID: E5
Research Question: Same as E3, with C9 lv60 espeak Cao GOP-S.
Hypothesis: E5 test PCC > C9 direct (PCC ≈ 0.430).
Independent Variable: scoring model = MLP
Controlled: dataset, split, target, AM = wav2vec2-lv-60-espeak-cv-ft,
            GOP = Cao GOP-S (shared with E6)
Input: scalar gop_c9
```

```text
Experiment ID: E6
Research Question: Same as E4, with C9 GOP-S (same X as E5).
Hypothesis: E6 test PCC ≥ E5.
Independent Variable: scoring model = Transformer encoder (per-phone head)
Controlled: same as E5, including identical X
```

```text
Experiment ID: E19
Research Question: Does a nonlinear MLP map C10 Cao GOP-SD to human phoneme
scores better than frozen direct C10?
Hypothesis: E19 test PCC > C10 direct (PCC ≈ 0.498).
Independent Variable: scoring model = MLP
Controlled: dataset, split, target, AM = wav2vec2-xlsr-53-espeak-cv-ft,
            GOP = Cao GOP-SD / AF-SD (shared with E20); phoneme-level
Input: scalar gop_c10
Output: predicted phoneme score, clip [0, 2]
```

```text
Experiment ID: E20
Research Question: Does sequence context improve C10 GOP-SD scoring beyond MLP?
Hypothesis: E20 test PCC ≥ E19 on the same gop_c10 sequence.
Independent Variable: scoring model = Transformer encoder (per-phone head)
Controlled: same as E19, including identical X
```

```text
Experiment ID: E21
Research Question: Same as E19, with C11 lv60 espeak Cao GOP-SD.
Hypothesis: E21 test PCC > C11 direct (PCC ≈ 0.463).
Independent Variable: scoring model = MLP
Controlled: dataset, split, target, AM = wav2vec2-lv-60-espeak-cv-ft,
            GOP = Cao GOP-SD (shared with E22)
Input: scalar gop_c11
```

```text
Experiment ID: E22
Research Question: Same as E20, with C11 GOP-SD (same X as E21).
Hypothesis: E22 test PCC ≥ E21.
Independent Variable: scoring model = Transformer encoder (per-phone head)
Controlled: same as E21, including identical X
```

```text
Experiment ID: E7
Research Question: Does a nonlinear MLP map the GOPT-paper 84-d GOP feature
to human phoneme scores better than frozen B5 OLS?
Hypothesis: E7 test PCC > B5 OLS (PCC ≈ 0.361).
Independent Variable: scoring model = MLP
Controlled: dataset, AM Kaldi M13, split, target, 84-d X (shared with E8)
Not full GOPT: no phone embedding / multi-task
```

```text
Experiment ID: E8
Research Question: Does sequence context improve 84-d GOP scoring beyond MLP?
Hypothesis: E8 test PCC ≥ E7 on the same 84-d sequence.
Independent Variable: scoring model = Transformer encoder (per-phone head)
Controlled: same as E7, including identical X
Not full GOPT: no phone embedding / multi-task
```

```text
Experiment ID: E13
Research Question: Does adding a canonical phone embedding to the same 84-d GOP
feature improve MLP scoring beyond E7?
Hypothesis: E13 test PCC > E7 on the same 84-d + same protocol.
Independent Variable: phone embedding (Kaldi slot id, 42-way)
Controlled: dataset, AM Kaldi M13, split, target, 84-d X, MLP architecture
Not full GOPT: no word/utt multi-task
```

```text
Experiment ID: E14
Research Question: Does sequence context plus phone embedding improve 84-d
scoring beyond E13 (MLP+embed) and beyond E8 (Transformer, no embed)?
Hypothesis: E14 test PCC ≥ E13 on the same 84-d+embed; E14 vs E8 isolates embed.
Independent Variable: Transformer + phone embedding
Controlled: same 84-d as E7–E13; embed shared with E13
Not full GOPT: no word/utt multi-task; val is train-speaker holdout (not test MSE)
```

```text
Experiment ID: E9
Research Question: Does MLP map C8 GOPT-style 78-d LPP+LPR better than OLS
on the same 78-d (CTC-aligned 39 mapped IPA LPP+LPR, not Kaldi span)?
Hypothesis: E9 test PCC > 78-d OLS.
Independent Variable: scoring model = MLP
Controlled: AM = wav2vec2-xlsr-53-espeak-cv-ft, 78-d X (shared with E10)
```

```text
Experiment ID: E10
Research Question: Does sequence context improve C8 78-d scoring beyond MLP?
Hypothesis: E10 test PCC ≥ E9 on the same 78-d sequence.
Independent Variable: scoring model = Transformer
Controlled: same as E9
```

```text
Experiment ID: E11
Research Question: Same as E9, with C9 lv60 espeak 78-d LPP+LPR.
Hypothesis: E11 test PCC > 78-d OLS on C9.
Independent Variable: scoring model = MLP
Controlled: AM = wav2vec2-lv-60-espeak-cv-ft, 78-d X (shared with E12)
```

```text
Experiment ID: E12
Research Question: Same as E10, with C9 78-d (same X as E11).
Hypothesis: E12 test PCC ≥ E11.
Independent Variable: scoring model = Transformer
Controlled: same as E11
```

```text
Experiment ID: E15
Research Question: Does adding an SSL 39-way canonical phone embedding to the
same C8 78-d LPP+LPR improve MLP scoring beyond E9?
Hypothesis: E15 test PCC > E9 on the same 78-d + same protocol.
Independent Variable: phone embedding (ssl_index, 39-way; not Kaldi 42-slot)
Controlled: AM = wav2vec2-xlsr-53-espeak-cv-ft, 78-d X, MLP architecture
Not full GOPT: no word/utt multi-task
```

```text
Experiment ID: E16
Research Question: Does sequence context plus SSL phone embedding improve C8
78-d scoring beyond E15 (MLP+embed) and beyond E10 (Transformer, no embed)?
Hypothesis: E16 test PCC ≥ E15 on the same 78-d+embed; E16 vs E10 isolates embed.
Independent Variable: Transformer + SSL phone embedding
Controlled: same 78-d as E9–E15; embed shared with E15
Not full GOPT: no word/utt multi-task; val is train-speaker holdout
```

```text
Experiment ID: E17
Research Question: Same as E15, with C9 lv60 espeak 78-d + SSL phone embed.
Hypothesis: E17 test PCC > E11 on the same 78-d.
Independent Variable: phone embedding (ssl_index, 39-way)
Controlled: AM = wav2vec2-lv-60-espeak-cv-ft, 78-d X, MLP architecture
Not full GOPT: no word/utt multi-task
```

```text
Experiment ID: E18
Research Question: Same as E16, with C9 78-d + SSL embed (same X as E17).
Hypothesis: E18 test PCC ≥ E17; E18 vs E12 isolates embed.
Independent Variable: Transformer + SSL phone embedding
Controlled: same 78-d as E11–E17; embed shared with E17
Not full GOPT: no word/utt multi-task
```

---

## 5. Split

```text
official test speakers  →  role=test   (125 speakers, overlap 0)
official train speakers
        │
        ├── 20% (seed 0)  →  role=val    early stopping
        └── 80%           →  role=train  fit scaler + weights
```

Val speakers ⊂ official train, ∩ official test = ∅. Danh sách val được ghi trong `e_results.json`.

---

## 6. Pipeline

```text
b_predictions.csv  (or a1_predictions.csv / c_predictions.csv)
        │
        ▼
Join utt2spk → carve val from train speakers
        │
        ▼
Train-only z-score
        │
        ├── E1 / E3 / E5 / E7 / E9 / E11 / E13 / E15 / E17 / E19 / E21  MLP on independent phones
        └── E2 / E4 / E6 / E8 / E10 / E12 / E14 / E16 / E18 / E20 / E22 Transformer on padded phone sequences (cap 50)
                per-phone regression head, pad masked in loss
        │
        ▼
clip [0, 2] → PCC / SCC / MAE / MSE on official test
        │
        ▼
outputs/E/e_predictions.csv               (B4; not overwritten)
outputs/E/e_c8_predictions.csv            (--features c8)
outputs/E/e_c9_predictions.csv            (--features c9)
outputs/E/e_c10_predictions.csv           (--features c10; pred_e19/e20)
outputs/E/e_c11_predictions.csv           (--features c11; pred_e21/e22)
outputs/E/e_b5_predictions.csv            (--features b5; pred_e7/e8; --features b5_embed adds pred_e13/e14)
outputs/E/e_c8_lpp_lpr_predictions.csv    (--features c8_lpp_lpr; pred_e9/e10; embed adds pred_e15/e16)
outputs/E/e_c9_lpp_lpr_predictions.csv    (--features c9_lpp_lpr; pred_e11/e12; embed adds pred_e17/e18)
outputs/E/e_results.json                  (merge E3–E22 into locked E1/E2)
checkpoints/e1/mlp_ckpt.pt                (and e2/transformer_ckpt.pt, …)
```

Môi trường: conda `gop` + PyTorch. Notebook: `notebooks/E_learned_scoring.ipynb` — **chỉ đọc artifact**.

Cả hai model: Adam, MSE, early stop **val MSE**, seed 0. Yaml default `device: cpu` (E1–E18). E19–E22: `--device cuda`.

- E1: `Linear → ReLU → Linear → ReLU → Linear` (hidden 32).
- E2: Transformer encoder 2 layer, d_model 32, 4 heads, sinusoidal position, **không** CLS. E8 không phone embed; E14 cộng `Embedding(42)` sau `in_proj` (GOPT-style; pad id = 42). E16/E18 cộng `Embedding(39)` SSL (`ssl_index`; pad id = 39).

---

## 7. Metrics và baseline

PCC/SCC/MAE/MSE trên \(\widehat{y}\) đã clip, giống B4 (không phải correlation GOP thô).

| So sánh | Khi |
| -------- | --- |
| Direct GOP (A2) | `features=a1` |
| B4 OLS | `features=b4` |
| Direct C8 / C9 | `features=c8` / `c9` (PCC/SCC on raw GOP-S; MAE/MSE C là map univariate) |
| Direct C10 / C11 | `features=c10` / `c11` (PCC/SCC on raw GOP-SD) |
| E1 MLP vs E2 Transformer | `b4` / `a1` (cùng X) |
| E3 vs E4 | `c8` (cùng GOP-S) |
| E5 vs E6 | `c9` (cùng GOP-S) |
| E19 vs E20 | `c10` (cùng GOP-SD) |
| E21 vs E22 | `c11` (cùng GOP-SD) |
| E3 vs E19 / E4 vs E20 | graph S vs SD, cùng AM XLSR |
| E5 vs E21 / E6 vs E22 | graph S vs SD, cùng AM lv60 |
| E7 vs E8 | `b5` (cùng 84-d, không embed) |
| E13 vs E14 | `b5_embed` (cùng 84-d + embed) |
| E7 vs E13 / E8 vs E14 | ablation phone embed |
| E9 vs E10 | `c8_lpp_lpr` (cùng 78-d) |
| E11 vs E12 | `c9_lpp_lpr` (cùng 78-d) |
| E15 vs E16 | `c8_lpp_lpr_embed` (cùng 78-d C8 + SSL embed) |
| E17 vs E18 | `c9_lpp_lpr_embed` (cùng 78-d C9 + SSL embed) |
| E9 vs E15 / E10 vs E16 | ablation SSL phone embed (C8) |
| E11 vs E17 / E12 vs E18 | ablation SSL phone embed (C9) |

Baseline copy từ JSON đã khóa, **không** fit lại.

---

## 8. Code map

| File | Vai trò |
| ---- | ------- |
| `configs/e_learned_scoring.yaml` | protocol + hyperparams |
| `src/gop_empirical/data/learned.py` | load A/B/C, val speakers, scaler, pack sequences |
| `src/gop_empirical/scoring/mlp.py` | E1 / E3 / E5 / E7 / E9 / E11 / E13 / E15 / E17 / E19 / E21 |
| `src/gop_empirical/scoring/transformer.py` | E2 / E4 / E6 / E8 / E10 / E12 / E14 / E16 / E18 / E20 / E22 |
| `src/gop_empirical/scoring/train.py` | Adam + early stop |
| `src/gop_empirical/scoring/checkpoint.py` | `checkpoints/{eid}/{arch}_ckpt.pt` |
| `scripts/eval_group_e_checkpoint.py` | eval official test from `.pt` (no train) |
| `src/gop_empirical/eval/metrics.py` | `evaluate_predictions` |
| `src/gop_empirical/experiment.py` | `run_group_e` |
| `scripts/run_experiment.py` | `--features` |
| `tests/test_learned_scoring.py` | no-leak split / scaler / dims / pad mask |
| `notebooks/E_learned_scoring.ipynb` | bar PCC từ artifact |

---

## 9. Artifact

```text
outputs/E/
  e_predictions.csv
  e_a1_predictions.csv      only if --features includes a1 and b4
  e_b4_predictions.csv      only if both Kaldi feature sets run
  e_c8_predictions.csv           --features c8
  e_c9_predictions.csv           --features c9
  e_c10_predictions.csv          --features c10
  e_c11_predictions.csv          --features c11
  e_b5_predictions.csv           --features b5 (keys + human + pred_e7/e8)
  e_c8_lpp_lpr_predictions.csv   --features c8_lpp_lpr; embed adds pred_e15/e16
  e_c9_lpp_lpr_predictions.csv   --features c9_lpp_lpr; embed adds pred_e17/e18
  e_results.json
  e_comparison_pcc.png      notebook
  e_scatter_pred_vs_human.png

checkpoints/
  e1/mlp_ckpt.pt
  e2/transformer_ckpt.pt
  e15/mlp_ckpt.pt
  e16/transformer_ckpt.pt
  … (one folder per E1–E18 neural scorer; overwritten on re-run)
```

`e_predictions.csv` gồm `role` (train/val/test), feature columns, `pred_e1`, `pred_e2`.

`e_c8_predictions.csv` / `e_c9_predictions.csv` gồm `pred_e3`/`pred_e4` và `pred_e5`/`pred_e6`. `e_c10_predictions.csv` / `e_c11_predictions.csv` gồm `pred_e19`/`pred_e20` và `pred_e21`/`pred_e22`.

`e_b5_predictions.csv` / `e_c8_lpp_lpr_predictions.csv` / `e_c9_lpp_lpr_predictions.csv` gồm `pred_e7`/`pred_e8`, `pred_e9`/`pred_e10`, `pred_e11`/`pred_e12`; sau embed: `pred_e13`/`pred_e14`, `pred_e15`/`pred_e16`, `pred_e17`/`pred_e18`. **Không** dump 84/78 feature columns.

`e_results.json` gồm `protocol` (val speaker ids, scaler, n), block theo từng feature set, baseline frozen. Run `c8`/`c9`/`c10`/`c11`/`b5`/`b5_embed`/`c8_lpp_lpr`/`c8_lpp_lpr_embed` merge E3–E22, **không** xóa E1/E2, E7/E8, hay E9–E12.

---

## 10. Kết quả (run hiện tại)

Nguồn: `outputs/E/e_results.json`. Feature set **b4**. Official test n = 47 369 (khớp A2/B4). Val = 25 / 125 train speakers (seed 0), overlap train/test = 0.

| Model | PCC | SCC | MAE | MSE |
| ----- | --: | --: | --: | --: |
| B4 OLS (frozen) | 0.351 | 0.342 | 0.197 | 0.120 |
| **E1** MLP | 0.362 | 0.327 | 0.192 | 0.118 |
| **E2** Transformer | **0.510** | **0.368** | **0.158** | **0.100** |

MLP cải thiện nhẹ so với OLS tuyến tính trên cùng B4. Transformer (cùng X, thêm sequence) tăng PCC rõ. Đây là finding về *scoring model*, không phải GOP mới.

Run `--features a1` khi cần so thẳng Direct GOP (A2 PCC 0.323).

### E3–E6 (C8/C9 Cao GOP-S)

Nguồn: `outputs/E/e_results.json` (merge sau `--features c8 c9`). Official test n = 47 369. E1/E2 B4 không bị ghi đè. Cặp so sánh: E3 vs E4 trên `gop_c8`; E5 vs E6 trên `gop_c9`. Không so PCC với E2 (B4 3-d). PCC/SCC vs direct C8/C9 là metric chính; MAE/MSE của C8/C9 frozen là sau map univariate (giống A2).

```text
python scripts/run_experiment.py --config configs/e_learned_scoring.yaml --features c8 c9
```

| Model | PCC | SCC | MAE | MSE |
| ----- | --: | --: | --: | --: |
| C8 direct (frozen) | 0.463 | 0.355 | 0.173 | 0.108 |
| **E3** MLP (c8) | 0.496 | 0.355 | 0.167 | 0.103 |
| **E4** Transformer (c8) | **0.578** | **0.361** | **0.154** | **0.090** |
| C9 direct (frozen) | 0.430 | 0.334 | 0.179 | 0.112 |
| **E5** MLP (c9) | 0.447 | 0.334 | 0.162 | 0.110 |
| **E6** Transformer (c9) | **0.483** | 0.319 | 0.157 | 0.105 |

Trên C8, MLP tăng PCC so với direct; Transformer tăng thêm (0.463 → 0.496 → 0.578). Trên C9, MLP và Transformer cũng tăng PCC (0.430 → 0.447 → 0.483), nhưng E6 **SCC** thấp hơn direct/E5 — sequence giúp linear correlation hơn rank correlation trên GOP-S lv60. Finding vẫn là *scoring model* trên posterior đã khóa, không phải GOP mới.

### E19–E22 (C10/C11 Cao GOP-SD)

Nguồn: `outputs/E/e_results.json` (merge sau `--features c10 c11 --device cuda`). Official test n = 47 369. E1–E18 không bị ghi đè. Cặp so sánh: E19 vs E20 trên `gop_c10`; E21 vs E22 trên `gop_c11`. Ablation graph: E3 vs E19, E4 vs E20, E5 vs E21, E6 vs E22. Không so PCC với E2/E16. Device CUDA (E3–E6 đã train CPU).

```text
python scripts/run_experiment.py --config configs/e_learned_scoring.yaml --features c10 c11 --device cuda
```

| Model | PCC | SCC | MAE | MSE |
| ----- | --: | --: | --: | --: |
| C10 direct (frozen) | 0.498 | 0.366 | 0.170 | 0.103 |
| **E19** MLP (c10) | 0.546 | 0.366 | 0.163 | 0.096 |
| **E20** Transformer (c10) | **0.614** | 0.367 | **0.144** | **0.084** |
| C11 direct (frozen) | 0.463 | 0.346 | 0.174 | 0.108 |
| **E21** MLP (c11) | 0.492 | 0.342 | 0.175 | 0.105 |
| **E22** Transformer (c11) | **0.526** | 0.314 | 0.169 | 0.098 |

Trên C10, MLP tăng PCC so với direct; Transformer tăng thêm (0.498 → 0.546 → 0.614). Trên C11 cùng hướng PCC (0.463 → 0.492 → 0.526). E21 MAE hơi cao hơn direct (0.175 vs 0.174); E22 **SCC** thấp hơn direct/E21 — giống pattern E6 trên GOP-S lv60. Ablation graph (cùng AM): E3 0.496 → E19 0.546; E4 0.578 → E20 0.614; E5 0.447 → E21 0.492; E6 0.483 → E22 0.526. Finding vẫn là *scoring model* trên GOP-SD đã khóa, không phải GOP mới. Device CUDA; E3–E6 đã train CPU.

### E7–E18 (GOPT-style LPP+LPR × 3 AM × 2 scorer; E13–E18 = + phone embed)

Nguồn: `outputs/E/e_results.json`. E1–E8 và E13/E14 không bị ghi đè. E9–E12 / E15–E18 được train lại sau khi sửa extract 78-d (CTC Viterbi, không Kaldi span). Cặp so sánh: **E7 vs E8** (84-d), **E13 vs E14** (84-d + Kaldi embed), **E9 vs E10**, **E15 vs E16** (C8 78-d + SSL embed), **E11 vs E12**, **E17 vs E18** (C9 78-d + SSL embed). Ablation embed: E7 vs E13, E8 vs E14, E9 vs E15, E10 vs E16, E11 vs E17, E12 vs E18. Không so PCC xuyên dim / AM. C8/C9 **không** gọi 84-d. Linear/OLS là baseline frozen, không phải E ID. E13–E18 **không** phải full GOPT paper (không multi-task; không chọn epoch trên test). SSL embed = 39 scored IPA (`ssl_index`), không Kaldi 42-slot.

Join 78-d phải dùng `utt.phn_idx` toàn câu (giống Kaldi / `scores.json`). `c_predictions.phone_id` là index **trong từ** — join nhầm khiến nhiều phone dính vector của phone đầu câu. Đã sửa trong `load_ssl_lpp_lpr_feature_table`.

**Protocol 78-d (đã sửa):** pool mean LPP trên frame **CTC Viterbi** của chuỗi canonical (cùng graph GOP-S), không phải start/end Kaldi. Trước đó (mean trên span Kaldi) là analog C2: `argmax(LPP)==canonical` ~6%, scalar LPP test PCC ~0.016, n_test = 47 308 (61 span rỗng). Sau CTC align: argmax ~72% (C8) / 72% (C9); scalar LPP test PCC ~0.470 / 0.447; n_test = 47 369. `LPR[canonical]` vẫn 0. Không đổi sang scalar GOP-S hay 392-d vocab.

| Model | PCC | SCC | MAE | MSE |
| ----- | --: | --: | --: | --: |
| B5 OLS (frozen) | 0.361 | 0.332 | 0.197 | 0.119 |
| **E7** MLP (b5) | 0.446 | 0.350 | 0.168 | 0.109 |
| **E8** Transformer (b5) | **0.530** | **0.379** | **0.154** | **0.097** |
| **E13** MLP + embed (b5_embed) | **0.552** | 0.359 | 0.159 | 0.094 |
| **E14** Transformer + embed (b5_embed) | **0.625** | **0.403** | **0.148** | **0.082** |
| C8 78-d OLS (train-only) | 0.507 | 0.358 | 0.172 | 0.102 |
| **E9** MLP (c8_lpp_lpr) | 0.573 | 0.392 | 0.142 | 0.091 |
| **E10** Transformer (c8_lpp_lpr) | **0.639** | **0.415** | 0.150 | 0.080 |
| **E15** MLP + embed (c8_lpp_lpr_embed) | 0.637 | 0.401 | 0.136 | 0.080 |
| **E16** Transformer + embed (c8_lpp_lpr_embed) | **0.671** | 0.412 | **0.133** | **0.074** |
| C9 78-d OLS (train-only) | 0.482 | 0.332 | 0.177 | 0.105 |
| **E11** MLP (c9_lpp_lpr) | 0.548 | 0.365 | 0.158 | 0.095 |
| **E12** Transformer (c9_lpp_lpr) | **0.624** | 0.380 | 0.134 | 0.083 |
| **E17** MLP + embed (c9_lpp_lpr_embed) | 0.618 | 0.379 | 0.153 | 0.084 |
| **E18** Transformer + embed (c9_lpp_lpr_embed) | **0.653** | **0.427** | **0.115** | **0.079** |

E7–E18 n = 47 369.

Phone embed trên cùng 84-d: E7 0.446 → E13 **0.552**; E8 0.530 → E14 **0.625**. Sequence vẫn giúp khi đã có embed (E13 vs E14). E14 nằm cùng vùng PCC với GOPT paper (~0.612 mean / 0.616 pretrained) nhưng **không** phải reproduction: không multi-task word/utt, val là train-speaker holdout (không chọn epoch trên test), encoder Group E.

Sau CTC self-align, 78-d **có** confusion profile: C8 OLS 0.224→**0.507**, E9 0.247→**0.573**, E10 0.290→**0.639**. Embed lúc này giúp như Kaldi: E9 0.573 → E15 **0.637**; E10 0.639 → E16 **0.671**. C9 tương tự (E11 0.548 → E17 **0.618**; E12 0.624 → E18 **0.653**). Scalar LPP[canonical] test PCC 0.470 (C8) / 0.447 (C9) cùng vùng C8/C9 GOP-S (0.463 / 0.430) — estimator đã khớp AM. C2 (mean trên span Kaldi) vẫn là negative control, không phải analog GOPT. E16/E18 **không** phải full GOPT paper.

