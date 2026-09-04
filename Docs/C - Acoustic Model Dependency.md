# C — Acoustic Model Dependency

Group C trả lời:

> **GOP có phụ thuộc vào acoustic model không?**

Không đổi dataset, alignment, phone set, công thức GOP, split, hay scoring. Chỉ đổi model tạo phone posterior.

| ID | AM | Posterior | GOP |
| -- | -- | --------- | --- |
| **C1** | Kaldi Librispeech M13 | Xiaomi LPP extract (cùng A1) | canonical LPP |
| **C2** | Wav2Vec2 + phoneme CTC | frame softmax, bỏ blank | mean log P(canonical) trên đoạn Kaldi |
| **C3** | HuBERT + phoneme CTC | giống C2 | cùng công thức |
| **C4** | cùng HuBERT C3 | giống C3 | **max** log P(canonical) trên đoạn Kaldi |
| **C5** | cùng HuBERT C3 | CTC softmax đủ vocab (có blank) | Cao GOP-S trên **chuỗi** phone CTM, không dùng thời gian Kaldi |
| **C6** | cùng Wav2Vec2 C2 | giống C2 | **max** log P(canonical) trên đoạn Kaldi |
| **C7** | cùng Wav2Vec2 C2 | CTC softmax đủ vocab (có blank) | Cao GOP-S trên **chuỗi** phone CTM, không dùng thời gian Kaldi |
| **C8** | `facebook/wav2vec2-xlsr-53-espeak-cv-ft` | espeak IPA CTC (off-the-shelf) | Cao GOP-S / **GOP-CTC-AF-S** sau map 1-1 CMU→IPA |
| **C9** | `facebook/wav2vec2-lv-60-espeak-cv-ft` | cùng IPA CTC, encoder lv60 English | Cao GOP-S / **GOP-CTC-AF-S**, cùng map với C8 |
| **C10** | cùng AM C8 | cùng posterior C8 (V = 392 + blank) | Cao GOP-SD / **GOP-CTC-AF-SD** (thêm deletion) |
| **C11** | cùng AM C9 | cùng posterior C9 | Cao GOP-SD / **GOP-CTC-AF-SD**, cùng graph C10 |

C1–C3 khóa AM. C4–C5 khóa HuBERT (C3); C6–C7 khóa Wav2Vec2 (C2). Hai cặp đó đổi công thức GOP (follow-up, không còn protocol lock “chỉ đổi AM”). **C8/C9 khóa Cao GOP-S (AF-S)** và đổi AM sang checkpoint espeak công khai — **không** fine-tune. **C10/C11 khóa AM C8/C9** và đổi graph S → SD. C2–C4 giữ negative control (mean/max trên span Kaldi).

**C1 phải khớp A2** (test PCC ≈ 0.323, n = 47 369). Đó là sanity check, giống B1 ≡ A2.

Embedding HuBERT / Wav2Vec2 **không phải GOP**. Cần phoneme CTC head.

Câu hỏi nghiên cứu:

> Khi cố định dataset, alignment Kaldi, phone inventory 39 CMU, GOP chuẩn và direct scoring, correlation GOP–human thay đổi bao nhiêu giữa Kaldi, Wav2Vec2-CTC và HuBERT-CTC?

Finding là **ΔPCC/SCC trên phone đã ghép cặp**, không phải “SSL là GOP tốt hơn.”

Ba AM **không** cùng kiến trúc / dữ liệu huấn luyện (Kaldi M13 vs XLSR-53+LS-100 CTC vs HuBERT+LS-100 CTC). Thesis không được viết như một bake-off matched-architecture.

---

## 1. Protocol lock

Giống Group A. Chỉ `acoustic_model` thay đổi.

| Thành phần | Giá trị Group C |
| ---------- | --------------- |
| Dataset | Speechocean762 |
| Sampling rate | 16 kHz |
| Level | phoneme |
| Alignment | Kaldi CTM (cùng `utt_id.phn_idx` với A/B) |
| GOP type | `standard` (canonical mean log-posterior) |
| Scoring | `direct` (không MLP / GOPT / LPR / B4) |
| Phone set | 39 CMU (bỏ SIL / SPN / NSN) |
| Test split | official train / test Speechocean762 |
| Metrics | PCC, SCC trên GOP thô; MAE, MSE sau map train-only |
| Seed | 0 |
| Score range | human 0–2; floor 0.1 |
| Phone index | `phone_index_base: 0` |

Config: `configs/c_acoustic_model.yaml`.

C1–C3 **không** dùng segmentation-free CTC GOP. C4/C5 là follow-up: cùng HuBERT, đổi aggregation (C4) hoặc bỏ span Kaldi (C5, Cao et al. IS2024 GOP-S / AF-S). C10/C11 là follow-up trên C8/C9: cùng AM espeak, đổi graph AF-S → AF-SD.

---

## 2. Công thức

C1 (extract sẵn):

\[
\mathrm{GOP}(p)=\mathrm{LPP}[p]
\]

C2 / C3 (Witt & Young trên posterior CTC, blank đã bỏ):

\[
\mathrm{GOP}(p)=\frac{1}{T_p}\sum_{t\in p}\log P(p\mid x_t)
\]

\(P(\cdot\mid x_t)\) = softmax **chỉ trên 39 CMU phones** (renormalize sau khi bỏ `<pad>` / blank). Kaldi không có blank; giữ blank trong mẫu số sẽ làm C2/C3 không so sánh được với C1.

Canonical phone = symbol bỏ stress từ `scores.json`, map qua `data/phone_inventory.json`.

Nếu \(T_p=0\) sau khi rescale CTM → wav: bỏ phone, đếm `n_empty_segment`. `n_frames` ghi ra để audit, không phải feature.

C4 (cùng span, đổi aggregation):

\[
\mathrm{GOP}_{\mathrm{C4}}(p)=\max_{t\in p}\log P(p\mid x_t)
\]

C5 (Cao GOP-S; \(y\) = chuỗi phone CTM đã bỏ silence; không dùng start/end):

\[
\mathrm{GOP}_{\mathrm{C5}}(y_i)=\log P(y\mid x)-\log P(y\text{ với }y_i\text{ wildcard}\mid x)
\]

C5 softmax **giữ blank** (đúng CTC). Không renormalize 39 phone như C2–C4.

C10 / C11 (cùng tỷ số; mẫu số thêm skip/deletion):

\[
\mathrm{GOP}_{\mathrm{C10}}(y_i)=\log P(y\mid x)-\log P(y\text{ với }y_i\text{ substitution hoặc deletion}\mid x)
\]

Cùng scaled `alpha_bar` với GOP-S (không copy CTC unscaled của official AF-SD). Wildcard vẫn full vocab espeak (V = 392). **Không** phải reproduction Table 2 paper.

---

## 3. Hai tầng: extract vs evaluate

Evaluate chạy CPU trong conda `gop`, giống A/B. SSL extract là script GPU riêng, ghi CSV; `run_experiment.py` chỉ đọc CSV.

**C1** đã có: `data/kaldi_gop_librispeech/`. Không chạy lại Kaldi.

**C2 / C3** cần wav + CTM + checkpoint (xem `data/SOURCE.txt`). Speechocean762 wav thường ngắn hơn CTM (median scale ~0.385); extract rescale CTM lên duration wav trước khi map frame.

```text
python scripts/extract_ssl_gop.py --config configs/c_acoustic_model.yaml --model wav2vec2
python scripts/extract_ssl_gop.py --config configs/c_acoustic_model.yaml --model hubert
python scripts/extract_ssl_gop.py --config configs/c_acoustic_model.yaml --model hubert --gop max cao_s
python scripts/extract_ssl_gop.py --config configs/c_acoustic_model.yaml --model wav2vec2 --gop max cao_s
python scripts/extract_ssl_gop.py --config configs/c_acoustic_model.yaml --model xlsr_espeak --gop cao_s
python scripts/extract_ssl_gop.py --config configs/c_acoustic_model.yaml --model lv60_espeak --gop cao_s
python scripts/extract_ssl_gop.py --config configs/c_acoustic_model.yaml --model xlsr_espeak --gop cao_sd
python scripts/extract_ssl_gop.py --config configs/c_acoustic_model.yaml --model lv60_espeak --gop cao_sd
python scripts/run_experiment.py --config configs/c_acoustic_model.yaml --models C1 C2 C3 C4 C5 C6 C7 C8 C9
python scripts/run_experiment.py --config configs/c_acoustic_model.yaml --models C1 C2 C3 C4 C5 C8 C9 C10 C11
```

C8/C9 không fine-tune; checkpoint local `models/wav2vec2-*-espeak-cv-ft/` (`python scripts/download_espeak_ctc.py`, không cache `%USERPROFILE%`). Extract load `AutoFeatureExtractor` + `data/espeak_ctc_vocab.json` (không cần `phonemizer`). Softmax giữ full vocab espeak + blank (V = 392), không renormalize về 39 CMU. C10/C11 dùng **cùng** checkpoint và map; ghi CSV mới (`*_gop_cao_sd`), **không** ghi đè `*_gop_cao`. Default yaml gồm C10/C11 (CSV đã extract); vẫn **không** gồm C6/C7. Eval merge C10/C11 vào `c_results.json` / `c_predictions.csv`, không xóa C1–C9.

Chạy chỉ C1 (không cần CSV SSL):

```text
python scripts/run_experiment.py --config configs/c_acoustic_model.yaml --models C1
```

Wav2Vec2 / HuBERT phoneme CTC là **dựng AM**, không phải thí nghiệm GOP. Cùng 39-phone vocab, train trên Librispeech train-clean-100 (không SO762):

- C2: `notebooks/C2_finetune_wav2vec2_kaggle.ipynb` (`facebook/wav2vec2-large-xlsr-53`)
- C3: `notebooks/C3_finetune_hubert_kaggle.ipynb` (`facebook/hubert-base-ls960`)

Hoặc `scripts/finetune_phoneme_ctc.py --backbone wav2vec2|hubert`.

Nhãn CTC local: `data/LibriSpeech ASR corpus/{train,dev}.ctm` (lexicon CMUdict, **không** force-align Kaldi). Tạo lại bằng `scripts/build_ls_phoneme_ctm.py`.

---

## 4. Experiment cards

```text
Experiment ID: C1
Research Question: Does Kaldi M13 canonical LPP still match A2 when scored as Group C?
Hypothesis: C1 reproduces A2 (test PCC ≈ 0.323).
Independent Variable: acoustic model = Kaldi Librispeech M13
Controlled: dataset, Kaldi alignment, phone set, standard GOP, split, direct scoring
```

```text
Experiment ID: C2
Research Question: How does Wav2Vec2 phoneme-CTC GOP correlate with human scores on the same segments?
Hypothesis: GOP–human correlation will differ from C1 because the posterior comes from a different AM.
Independent Variable: acoustic model = Wav2Vec2 + phoneme CTC
```

```text
Experiment ID: C3
Research Question: How does HuBERT phoneme-CTC GOP correlate with human scores on the same segments?
Hypothesis: HuBERT-CTC GOP is another AM-dependent posterior, not an embedding baseline.
Independent Variable: acoustic model = HuBERT + phoneme CTC
```

```text
Experiment ID: C4
Research Question: If HuBERT CTC mean-on-Kaldi-span GOP fails, does max/spike on the same span recover correlation?
Hypothesis: Peak log P(canonical) is closer to a CTC spike than the span mean; PCC should rise vs C3 if mean-on-CTM was the failure mode.
Independent Variable: GOP aggregation = max (AM locked to C3 HuBERT)
```

```text
Experiment ID: C5
Research Question: Does Cao GOP-S (no Kaldi times) correlate with human scores on the same HuBERT AM?
Hypothesis: Alignment-free substitution GOP uses the CTC graph instead of Kaldi frames; PCC will differ from C3/C4. Report as a follow-up, not an AM bake-off.
Independent Variable: GOP = Cao GOP-S; AM locked to C3 HuBERT
```

```text
Experiment ID: C6
Research Question: Same as C4, with the Wav2Vec2 AM from C2.
Hypothesis: Max-on-Kaldi-span should behave like C4 (weak positive), not like C2 mean (near zero).
Independent Variable: GOP aggregation = max (AM locked to C2 Wav2Vec2)
```

```text
Experiment ID: C7
Research Question: Same as C5, with the Wav2Vec2 AM from C2.
Hypothesis: If GOP formula was the failure mode, Cao GOP-S should recover correlation even on the undertrained XLSR checkpoint.
Independent Variable: GOP = Cao GOP-S; AM locked to C2 Wav2Vec2
```

```text
Experiment ID: C8
Research Question: Does an off-the-shelf XLSR-53 espeak CTC AM, scored with Cao GOP-S (GOP-CTC-AF-S) after a frozen CMU→IPA map, correlate with human scores?
Hypothesis: Public IPA CTC + GOP-S is a different posterior source than Kaldi LPP or in-house 39-phone HuBERT. Report ΔPCC vs C1/C5; do not claim a matched-vocab bake-off.
Independent Variable: acoustic model = facebook/wav2vec2-xlsr-53-espeak-cv-ft (no fine-tune)
Controlled: dataset, CTM sequence, Cao GOP-S, split, direct scoring
```

```text
Experiment ID: C9
Research Question: Same GOP-S / AF-S estimator as C8, with wav2vec2-large-lv60 espeak CTC.
Hypothesis: The English lv60 encoder may yield a stronger English posterior than multilingual XLSR-53; same IPA map as C8.
Independent Variable: acoustic model = facebook/wav2vec2-lv-60-espeak-cv-ft (no fine-tune)
```

```text
Experiment ID: C10
Research Question: On the locked C8 XLSR-53 espeak AM, does allowing
deletion in the CTC GOP graph (AF-SD) change GOP–human correlation
versus frozen C8 AF-S?
Hypothesis: C10 test PCC > C8, because SO762 score 0 includes missing
phones and AF-S cannot represent deletion.
Independent Variable: GOP graph = substitution+deletion
Controlled: AM C8, CMU→IPA map, CTM sequence, split, direct scoring
Not a paper reproduction: vocab 392 espeak, not 39-phone XLSR CTC
```

```text
Experiment ID: C11
Research Question: Same as C10, with the locked C9 lv60 espeak AM.
Hypothesis: C11 test PCC > C9 on the same deletion graph as C10.
Independent Variable: GOP graph = substitution+deletion
Controlled: AM C9, same IPA map as C8/C10, CTM sequence, split, direct scoring
```

Fair pairs: **C8 vs C10**, **C9 vs C11**. C10 vs C11 không phải so sánh chính (lẫn AM).

**Không làm (out of scope):** GOP-CTC-AF-SDI; GOP-feature-CTC-AF 41-d; C7 39-CMU; CMU-kids; overwrite CSV C8/C9; so PCC với 0.433 paper như reproduction. Learned scoring trên C10/C11 = Group E **E19–E22**.

---

## 5. Pipeline

```text
C1  Kaldi GOP CSV
C2  wav2vec2_gop/{tr,te}_ssl_gop.csv
C3  hubert_gop/{tr,te}_ssl_gop.csv
C4  hubert_gop_max/{tr,te}_ssl_gop.csv
C5  hubert_gop_cao/{tr,te}_ssl_gop.csv
C6  wav2vec2_gop_max/{tr,te}_ssl_gop.csv
C7  wav2vec2_gop_cao/{tr,te}_ssl_gop.csv
C8  xlsr_espeak_gop_cao/{tr,te}_ssl_gop.csv
C9  lv60_espeak_gop_cao/{tr,te}_ssl_gop.csv
C10 xlsr_espeak_gop_cao_sd/{tr,te}_ssl_gop.csv
C11 lv60_espeak_gop_cao_sd/{tr,te}_ssl_gop.csv
        │
        ▼
Join scores.json  (utt_id.phn_idx)
        │
        ├── skip silence
        └── write outputs/C/c_predictions.csv
                │
                ▼
Each model: test PCC/SCC on raw GOP
            train-only univariate map → MAE/MSE
Paired: inner-join requested models; n_test_paired ≥ 95% of C1 test n
                │
                ▼
        outputs/C/c_results.json
```

---

## 6. Code map

| File | Vai trò |
| ---- | ------- |
| `configs/c_acoustic_model.yaml` | protocol + đường dẫn |
| `data/phone_inventory.json` | 42 Kaldi slots ↔ 39 CMU + SSL ids |
| `data/cmu_to_espeak.json` | 1-1 CMU → Facebook espeak IPA tokens (C8/C9) |
| `data/espeak_ctc_vocab.json` | frozen C8/C9 tokenizer vocab |
| `src/gop_empirical/acoustic/espeak_map.py` | resolve CMU→CTC ids; fail if token missing |
| `src/gop_empirical/acoustic/phones.py` | load inventory / map symbol |
| `src/gop_empirical/acoustic/alignment.py` | CTM, rescale, frame span |
| `src/gop_empirical/acoustic/posterior.py` | HF CTC → log P over 39 phones |
| `src/gop_empirical/gop/from_posterior.py` | mean / max log P(canonical) |
| `src/gop_empirical/gop/cao.py` | Cao GOP-S (C5/C8/C9) và GOP-SD (C10/C11) |
| `src/gop_empirical/data/ssl_gop.py` | load SSL GOP CSV |
| `src/gop_empirical/experiment.py` | `run_group_c` (C1–C11; merge C10/C11) |
| `scripts/download_espeak_ctc.py` | tải C8/C9 vào `models/` (không cache user trên C:) |
| `scripts/extract_ssl_gop.py` | GPU extract C2–C11 (`--gop mean|max|cao_s|cao_sd`; C8/C9 = `cao_s`, C10/C11 = `cao_sd`) |
| `scripts/build_ls_phoneme_ctm.py` | CTM chuỗi phone LS-100 (lexicon, không ali Kaldi) |
| `scripts/finetune_phoneme_ctc.py` | dựng HuBERT/Wav2Vec2 phoneme CTC |
| `notebooks/C2_finetune_wav2vec2_kaggle.ipynb` | Kaggle: XLSR-53 phoneme CTC (C2 AM) |
| `notebooks/C3_finetune_hubert_kaggle.ipynb` | Kaggle: HuBERT phoneme CTC (C3 AM) |
| `tests/test_acoustic_gop.py` | công thức, blank-strip, CTM, C1 ≡ A2 |
| `notebooks/C_acoustic_model.ipynb` | so sánh C1–C3 vs A2 |

---

## 7. Artifact

```text
outputs/C/
  c_predictions.csv
  c_results.json
  c_comparison_pcc.png
  c_scatter_c1_vs_human.png
  c_scatter_c2_vs_human.png   (after C2 extract)
  c_scatter_c3_vs_human.png   (after C3 extract)
```

`c_results.json` gồm `protocol` (`n_train` / `n_test` từng model, `n_test_paired`, `n_empty_segment`, `c1_matches_a2`) và `comparison.test` (full + paired).

---

## 8. Kết quả (run hiện tại)

Nguồn: `outputs/C/c_results.json`. `c1_matches_a2: true`. Test paired n = 47 308.

| | AM | GOP | Test PCC | Test SCC | n |
| - | -- | --- | -------: | -------: | -: |
| C1 | Kaldi M13 | LPP | **0.323** | 0.313 | 47 369 |
| C2 | Wav2Vec2 CTC | mean on Kaldi span | −0.038 | −0.007 | 47 308 |
| C3 | HuBERT CTC | mean on Kaldi span | −0.060 | −0.029 | 47 308 |
| C4 | HuBERT CTC | max on Kaldi span | 0.055 | 0.074 | 47 308 |
| C5 | HuBERT CTC | Cao GOP-S (no span) | 0.346 | 0.340 | 47 369 |
| C6 | Wav2Vec2 CTC | max on Kaldi span | 0.074 | 0.094 | 47 308 |
| C7 | Wav2Vec2 CTC (in-house 39-CMU) | Cao GOP-S (no span) | 0.378 | 0.353 | 47 369 |
| C8 | XLSR-53 espeak CTC (off-the-shelf) | Cao GOP-S / AF-S + CMU→IPA | 0.463 | 0.355 | 47 369 |
| C9 | lv60 espeak CTC (off-the-shelf) | Cao GOP-S / AF-S, cùng map C8 | 0.430 | 0.334 | 47 369 |
| C10 | cùng AM C8 | Cao GOP-SD / AF-SD | **0.498** | 0.366 | 47 369 |
| C11 | cùng AM C9 | Cao GOP-SD / AF-SD | 0.463 | 0.346 | 47 369 |

Paired (n = 47 308): C1 0.320, C5 0.344, C7 0.375, C9 0.428, C8 0.461, C11 0.460, C10 **0.495**. Max-on-span (C4/C6) chỉ ~0.05–0.07. Cao GOP-S phục hồi trên **cả hai** AM in-house, kể cả Wav2Vec2 undertrained. Chỗ gãy C2/C3 là công thức GOP + span Kaldi, không phải AM.

C8/C9 (cùng Cao GOP-S / AF-S, **không** fine-tune) vượt C1 và C5 trên PCC. Đó là so sánh **nguồn posterior** (Kaldi LPP / 39-CMU CTC / espeak IPA CTC), không phải bake-off matched-vocab. C8 (XLSR-53 đa ngữ) > C9 (lv60 English) trên PCC; SCC C8 ≈ C7 (0.355 vs 0.353) — khoảng PCC lớn hơn khoảng rank correlation. Hypothesis C9 (encoder English mạnh hơn XLSR) **không** đúng trên metric này.

Cặp khóa graph: C8→C10 ΔPCC **+0.035** (0.463 → 0.498); C9→C11 ΔPCC **+0.033** (0.430 → 0.463). Cả hai hypothesis C10/C11 đúng trên official test: thêm deletion trên **cùng** posterior làm tăng PCC (và giảm MAE: C8 0.173→C10 0.170, C9 0.179→C11 0.174). Finding: *cho phép deletion trên cùng posterior C8/C9 thay đổi ΔPCC*. Không viết “AF-SD tốt hơn GOP”; không so 0.498 với paper 0.433 như reproduction (vocab 392 espeak, không phải 39-phone XLSR CTC). C10 vs C11 không phải so sánh chính (lẫn AM).

---