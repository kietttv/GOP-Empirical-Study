Triển khai thesis theo kiểu **từng tầng một**, và quan trọng nhất là **không để các experiment phụ thuộc lẫn nhau một cách không kiểm soát**.

Với đề tài:

> **An Empirical Study of English Pronunciation Assessment Using Goodness of Pronunciation Scoring**

**GOP sẽ là đối tượng nghiên cứu chính**, còn MLP/GOPT/acoustic model là các công cụ để trả lời từng research question.

Dataset **Speechocean762** rất phù hợp cho thiết kế này vì có 5.000 câu tiếng Anh của 250 người học không bản ngữ, có annotation của 5 chuyên gia ở **phoneme-, word- và sentence-level**. ([arXiv][1])

---

# 1. Trước hết: khóa Experimental Protocol

Trước khi chạy Experiment A1, bạn cần cố định một số thứ.

```text
                    Experimental Protocol
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
      Dataset           Alignment          Acoustic
        │                  │                Model
        ▼                  ▼                  ▼
   Speechocean762        MFA/Kaldi        Model A/B/C
        │
        ▼
   Ground Truth
        │
        ▼
  Human Scores
```

## Những thứ phải cố định

| Thành phần           | Quyết định           |
| -------------------- | -------------------- |
| Dataset              | Speechocean762       |
| Sampling rate        | 16 kHz               |
| Phone representation | Chốt một phone set   |
| Alignment            | Chốt một phương pháp |
| Test split           | Cố định              |
| Metrics              | PCC, SCC, MSE/MAE    |
| Random seed          | Cố định cho từng run |
| Preprocessing        | Giống nhau           |
| Normalization        | Giống nhau           |
| Evaluation script    | Một script duy nhất  |

Speechocean762 có sẵn sentence-, word- và phoneme-level annotations; phoneme score nằm trong khoảng 0–2, trong đó 2 là phát âm đúng, 1 là đúng nhưng có accent nặng, và 0 là sai/mất phoneme. ([Hugging Face][2])

---

# 2. Pipeline tổng thể

Tôi đề xuất xây **một pipeline chung**, sau đó tất cả experiment chỉ thay đổi đúng thứ cần nghiên cứu.

```text
                    AUDIO
                      │
                      ▼
               Preprocessing
                      │
                      ▼
               Forced Alignment
                      │
                      ▼
                 Phone Segment
                      │
                      ▼
                Acoustic Model
                      │
                      ▼
               Phone Posterior
                      │
                      ▼
                    GOP
                      │
       ┌──────────────┼───────────────┐
       ▼              ▼               ▼
   Experiment B   Experiment C    Experiment E
   GOP features   Acoustic model  Learned model
       │              │               │
       └──────────────┼───────────────┘
                      ▼
                  Prediction
                      │
                      ▼
             Human Ground Truth
                      │
                      ▼
              PCC / SCC / MSE
                      │
                      ▼
                   Analysis
```

**Điểm quan trọng:** đừng viết 10 notebook riêng biệt. Hãy viết một pipeline có thể thay đổi configuration.

Ví dụ:

```python
EXPERIMENT = "A1"

ACOUSTIC_MODEL = "kaldi_m13"

GOP_TYPE = "standard"

SCORING_MODEL = "direct"

LEVEL = "phoneme"
```

Sau đó:

```python
EXPERIMENT = "B1"
GOP_TYPE = "lpp"
```

hoặc:

```python
EXPERIMENT = "C2"
ACOUSTIC_MODEL = "hubert"
```

---

# 3. Group A — Establish GOP Baseline

Đây là **experiment quan trọng nhất**.

---

## A1 — Traditional GOP

### Mục tiêu

Trả lời:

> **GOP truyền thống có thể phản ánh human pronunciation score ở mức độ nào?**

### Input

```text
Audio
+
Canonical transcript
```

### Pipeline

```text
Audio
 │
 ▼
Forced Alignment
 │
 ▼
Canonical phones
 │
 ▼
Acoustic Model
 │
 ▼
Frame-level posterior
 │
 ▼
Phone-level posterior
 │
 ▼
GOP
 │
 ▼
Phone score
```

---

## Bước A1.1 — Forced Alignment

Ví dụ câu:

> "THE CAT IS BIG"

Alignment:

```text
THE       CAT       IS       BIG

DH AH     K AE T    IH Z     B IH G
│ │       │ │ │     │ │      │ │ │
```

Bạn cần có:

```text
phone
start
end
```

Ví dụ:

| phone | start |  end |
| ----- | ----: | ---: |
| DH    |  0.00 | 0.18 |
| AH    |  0.18 | 0.31 |
| K     |  0.31 | 0.44 |
| AE    |  0.44 | 0.61 |
| T     |  0.61 | 0.72 |

**Mục đích:** biết frame nào thuộc phoneme nào.

---

# 4. Bước A1.2 — Acoustic Model

Bạn cần acoustic model cung cấp:

[
P(q|x_t)
]

hoặc posterior tương ứng cho phone/senone.

Ví dụ:

```text
Frame t
 │
 ├── /AA/ : 0.05
 ├── /AE/ : 0.72
 ├── /AH/ : 0.08
 ├── /EH/ : 0.10
 └── ...
```

Nếu dùng Kaldi M13, cần đặc biệt chú ý model có thể hoạt động ở **senone/state level**, vì vậy pipeline:

```text
Senone posterior
       ↓
Phone posterior
       ↓
GOP
```

phải được định nghĩa rõ ràng.

Đừng viết trong thesis rằng:

> "Kaldi directly outputs phoneme posterior"

nếu thực tế model đang output posterior trên senones.

Bạn cần document mapping:

```text
Senones
   ↓
Phone states
   ↓
Phone posterior
```

---

# 5. Bước A1.3 — Tính GOP

Một dạng GOP kinh điển có thể biểu diễn khái quát:

[
GOP(p)=
\frac{1}{T_p}
\sum_{t \in p}
\log P(p|x_t)
]

Trong đó:

* (p): canonical phoneme
* (x_t): acoustic frame
* (T_p): số frame của phoneme

Ý tưởng:

> Nếu acoustic model cho xác suất cao rằng đoạn âm thanh thực sự là phoneme canonical → GOP cao.

Nếu:

```text
Canonical = /TH/

Posterior:
TH = 0.80
S  = 0.10
T  = 0.05
...
```

→ GOP tương đối tốt.

Nếu:

```text
Canonical = /TH/

Posterior:
TH = 0.10
S  = 0.65
T  = 0.15
...
```

→ GOP thấp.

---

# 6. Bước A1.4 — Map GOP với human score

Speechocean762 cho human phoneme score:

```text
0
1
2
```

([Hugging Face][2])

Bạn tạo:

```text
phone_id | canonical | GOP | human_score
------------------------------------------------
0001     | TH        | -1.82 | 0
0002     | R         | -0.61 | 1
0003     | AE        | -0.12 | 2
```

Sau đó:

```text
GOP
 │
 ▼
Correlation
 │
 ├── PCC
 ├── SCC
 └── MSE/MAE
```

---

# 7. A2 — GOP vs Human Score

A2 thực ra **không phải một pipeline mới**.

Nó là bước phân tích của A1.

Bạn lấy:

```text
X = GOP
Y = Human phoneme score
```

và tính:

### PCC

Đo linear correlation.

### SCC

Đo rank correlation.

### MSE / MAE

Đo prediction error nếu bạn normalize/map GOP thành score.

---

## Nhưng nên thêm scatter plot

Ví dụ:

```text
Human
score
  2 |                • • •
    |          • • •
  1 |      • •
    |   • •
  0 | • •
    +-------------------------
       Low GOP       High GOP
```

Nếu GOP thực sự phản ánh pronunciation:

> điểm càng cao → human score có xu hướng càng cao.

---

# 8. A1/A2 cần output những gì?

Tôi khuyên lưu một file:

```text
experiment_A1_predictions.csv
```

Ví dụ:

```csv
utt_id,word_id,phone_id,phone,gop,human_score
000010011,0,0,W,-0.32,2
000010011,0,1,IY,-0.51,2
000010011,1,0,K,-0.62,2
000010011,1,1,AO,-1.91,1
```

Và:

```text
results_A1.json
```

```json
{
  "pcc": 0.48,
  "scc": 0.43,
  "mae": 0.51,
  "mse": 0.38
}
```

**Đây sẽ là baseline reference cho toàn bộ thesis.**

---

# 9. Group B — GOP Feature Representation

Đây là experiment rất quan trọng.

Bạn muốn trả lời:

> **Không phải "GOP có tốt không?" mà là "GOP nên được biểu diễn như thế nào?"**

---

# 10. B1 — Standard GOP

```text
Posterior
   ↓
GOP
   ↓
Score
```

Ví dụ:

[
GOP(p) = \log P(p|X)
]

---

# 11. B2 — LPP

LPP = **Log Phone Posterior**.

Bạn lấy:

[
LPP(p)=\log P(p|X)
]

Về thực tế, nó rất gần với cách bạn xây GOP tùy formulation.

Mục tiêu của B2 là xác định:

> **Raw log posterior có predictive information tương đương/khác GOP aggregation như thế nào?**

---

# 12. B3 — LPR

LPR = **Log Posterior Ratio**.

Ý tưởng:

```text
Probability canonical phone
          vs
Probability competing phones
```

Một dạng:

[
LPR(p)=
\log
\frac{P(p|X)}
{\max_{q\neq p}P(q|X)}
]

Ý nghĩa:

Nếu:

```text
P(canonical) = 0.8
P(best competitor) = 0.1
```

thì:

```text
LPR = log(8)
```

→ khá tốt.

Nhưng:

```text
P(canonical) = 0.35
P(competitor) = 0.40
```

→ LPR thấp/âm.

Nó trả lời một câu hỏi khác với LPP:

> **Canonical phone có thắng các phone cạnh tranh hay không?**

---

# 13. B4 — GOP feature vector

Thay vì:

```text
phone → 1 scalar
```

bạn giữ:

```text
phone
 ↓
posterior vector
 ↓
GOP-related features
```

Ví dụ:

```text
[canonical_logprob,
 max_competitor_logprob,
 LPR,
 duration,
 posterior_entropy]
```

Nhưng ở đây cần cẩn thận:

**Nếu thesis của bạn muốn nghiên cứu GOP thuần**, đừng thêm duration/entropy ngay vào B4.

Hãy giữ:

```text
GOP-only feature vector
```

trước.

---

# 14. Cách thực hiện Group B

Bạn phải giữ nguyên:

```text
Dataset
Alignment
Acoustic Model
Train/test split
Scoring model
```

Chỉ thay:

```text
GOP representation
```

Ví dụ:

| Experiment | Feature            |
| ---------- | ------------------ |
| B1         | Standard GOP       |
| B2         | LPP                |
| B3         | LPR                |
| B4         | GOP feature vector |

Sau đó:

```text
B1 → PCC
B2 → PCC
B3 → PCC
B4 → PCC
```

**Không được đổi acoustic model giữa B1–B5.**

Nếu không, bạn không biết improvement đến từ feature hay acoustic model.

---

# 15. Group C — Acoustic Model Dependency

Đây là một trong những experiment **có giá trị nhất**.

Câu hỏi:

> **GOP có phụ thuộc vào acoustic model không?**

Rất có thể câu trả lời là **có**, vì GOP được tính từ posterior do acoustic model tạo ra.

---

# 16. C1 — Kaldi

```text
Audio
 ↓
Kaldi
 ↓
Phone posterior
 ↓
GOP
 ↓
Human score
```

Đây là baseline.

---

# 17. C2 — Wav2Vec2

```text
Audio
 ↓
Wav2Vec2
 ↓
Phone posterior
 ↓
GOP
 ↓
Human score
```

Nhưng có một vấn đề quan trọng:

**Wav2Vec2 pretrained không tự nhiên output English phoneme posterior theo cách Kaldi acoustic model làm.**

Bạn cần một model Wav2Vec2-CTC đã có phoneme vocabulary hoặc fine-tuned phoneme recognizer.

Ví dụ:

```text
Wav2Vec2 encoder
       ↓
CTC phoneme head
       ↓
phoneme posterior
       ↓
GOP
```

Không nên lấy hidden embedding rồi gọi đó là GOP.

---

# 18. C3 — HuBERT

Tương tự:

```text
HuBERT
 ↓
phoneme recognition head
 ↓
phone posterior
 ↓
GOP
```

Nếu chỉ:

```text
Audio
 ↓
HuBERT
 ↓
768-d embedding
```

thì **đó không phải GOP**.

Nó là acoustic representation.

Đây là distinction cực kỳ quan trọng cho thesis.

---

# 19. Cần thiết kế C rất cẩn thận

Bạn muốn so sánh:

```text
Kaldi-GOP
Wav2Vec2-GOP
HuBERT-GOP
```

thì tất cả phải tạo ra **cùng phone set**.

Ví dụ:

```text
               Same phone inventory
                       │
       ┌───────────────┼────────────────┐
       ▼               ▼                ▼
     Kaldi          Wav2Vec2          HuBERT
       │               │                │
       ▼               ▼                ▼
    Posterior        Posterior        Posterior
       │               │                │
       ▼               ▼                ▼
      GOP             GOP              GOP
```

Nếu Kaldi dùng 39 phones nhưng Wav2Vec2 dùng 41 phones thì comparison không còn clean.

---

# 20. Group D — Phone-level Analysis

Đây là nơi bạn bắt đầu **khám phá hành vi của GOP**.

Sau khi có:

```text
phone
GOP
human score
```

group theo phone.

Ví dụ:

```python
groupby("phone")
```

Tính:

```text
Phone
N
Mean GOP
Std GOP
Mean human score
PCC
SCC
```

Kết quả:

| Phone |    N | Mean GOP | PCC |
| ----- | ---: | -------: | --: |
| /AE/  | 1200 |    -0.45 | .71 |
| /IH/  | 1100 |    -0.52 | .68 |
| /TH/  |  800 |    -1.21 | .39 |
| /R/   |  700 |    -0.94 | .42 |

Bạn có thể phát hiện:

> GOP có correlation cao với vowels nhưng thấp hơn với một số consonants.

---

# 21. Nhưng phải kiểm soát sample size

Không được thấy:

```text
/ZZ/ → PCC = 0.90
```

rồi kết luận /ZZ/ tốt nhất nếu chỉ có 10 samples.

Bạn cần:

```text
N >= threshold
```

Ví dụ nghiên cứu chỉ report phone có:

```text
N ≥ 50
```

hoặc:

```text
N ≥ 100
```

và ghi rõ threshold.

---

# 22. Group D2 — Speaker-level Analysis

Bạn group:

```text
speaker
```

Ví dụ:

| Speaker |   N | PCC | Mean GOP |
| ------- | --: | --: | -------: |
| S001    | 100 | .61 |    -0.72 |
| S002    | 120 | .54 |    -0.91 |
| S003    |  98 | .39 |    -1.14 |

Mục đích:

> GOP có ổn định giữa các speakers không?

Nhưng **không nên tính PCC trên quá ít samples/speaker**.

Nếu mỗi speaker chỉ có vài utterances thì nên phân tích:

* Mean absolute error
* mean GOP
* score distribution
* calibration

thay vì cố tính correlation cho từng speaker.

---

# 23. Group D3 — Proficiency Analysis

Nếu dataset có proficiency level rõ ràng, chia:

```text
Beginner
Intermediate
Advanced
```

Sau đó:

```text
GOP ↔ Human score
```

theo từng nhóm.

Ví dụ:

| Level        | PCC |
| ------------ | --: |
| Beginner     | .52 |
| Intermediate | .61 |
| Advanced     | .48 |

Điều này có thể dẫn đến finding:

> GOP performs differently across proficiency levels.

Nếu dataset không có proficiency label đáng tin cậy thì **không tự suy ra proficiency từ score rồi gọi đó là ground truth**. Bạn có thể thay experiment này bằng analysis theo human-score bins.

---

# 24. Group E — GOP + MLP

Đến đây mới bắt đầu dùng machine learning.

Mục đích:

> **GOP có thể được học lại thành pronunciation score tốt hơn direct GOP không?**

Pipeline:

```text
GOP feature
    │
    ▼
  MLP
    │
    ▼
Predicted human score
```

Ví dụ:

```text
Input:
[GOP]

MLP:
Linear
ReLU
Linear
ReLU
Linear

Output:
score
```

---

# 25. Cực kỳ quan trọng: MLP không được leak test set

Chia:

```text
Train
Validation
Test
```

Ví dụ:

```text
Train: 70%
Validation: 15%
Test: 15%
```

Nhưng với pronunciation dataset, **speaker-independent split** càng tốt:

```text
Train speakers
      ≠
Test speakers
```

Speechocean762 có speaker IDs và demographic metadata, nên bạn có thể kiểm soát split theo speaker. ([Hugging Face][3])

Nếu dataset split gốc đã được cung cấp, bạn cần kiểm tra speaker overlap trước khi quyết định dùng trực tiếp.

---

# 26. E1 — GOP + MLP

```text
GOP
 │
 ▼
MLP
 │
 ▼
Score
```

Kết quả:

```text
Direct GOP     PCC = X
MLP + GOP      PCC = Y
```

Nếu:

```text
Y > X
```

thì finding:

> Nonlinear mapping between GOP and human score may improve prediction.

Nhưng đây **không phải**:

> "MLP improved GOP itself."

Chính xác hơn:

> "A learned nonlinear scoring function improves the mapping from GOP features to human pronunciation scores."

---

# 27. E2 — GOP + GOPT

Pipeline:

```text
Phone GOP sequence
       │
       ▼
GOPT / Transformer
       │
       ▼
Regression Head
       │
       ▼
Score
```

Ví dụ:

```text
Sentence

/TH/ /AH/ /K/ /AE/ /T/
 │    │    │    │    │
 ▼    ▼    ▼    ▼    ▼
GOP  GOP  GOP  GOP  GOP
 │    │    │    │    │
 └────┴────┴────┴────┘
          │
          ▼
     Transformer
          │
          ▼
      Regression
```

Mục tiêu:

> Sequence context có giúp pronunciation scoring không?

Đây là câu hỏi empirical rất đẹp.

---

# 28. E1 và E2 phải so sánh công bằng

```text
Same GOP features
Same dataset
Same split
Same target
Same evaluation
```

Chỉ thay:

```text
MLP
vs
Transformer
```

Ví dụ:

| Model      | PCC | SCC | MSE |
| ---------- | --: | --: | --: |
| Direct GOP | .48 | .44 | .42 |
| MLP        | .53 | .49 | .37 |
| GOPT       | .57 | .53 | .34 |

Finding:

> Sequence-aware modeling may better exploit GOP features than independent phone-level regression.

---

# 29. Group F — Statistical Analysis

Đây không phải một model experiment. Đây là **validation layer** trên prediction A–E đã khóa.

Protocol khóa: [`F - Statistical and Error Analysis.md`](F%20-%20Statistical%20and%20Error%20Analysis.md), config `configs/f_validation.yaml`.

**F1a — Bootstrap CI (phone-level, n_boot=1000):** headline C1, C8, C9, B4_OLS, E2, E7, E8, E10, E12, E14, E16, E18.

Báo cáo:

```text
PCC = 0.671, 95% CI = [lo, hi]
```

**F1b — Paired Δ:** cùng phone test; CI của ΔPCC (A−B). Cặp: C8−C1, C9−C1, C8−C9, E2−B4_OLS, E8−E7, E14−E8, E16−E10, E18−E12, E16−C8.

**F1c — Multi-seed:** chỉ E2 và E16, seeds 0–4; ghi `outputs/F/` (không overwrite E seed-0). Val speakers cố định seed 0.

```text
python scripts/run_experiment.py --config configs/f_validation.yaml
```

---

# 30. So sánh model phải có confidence interval

Bootstrap percentile CI trên correlation / Δ. Không kết luận “better” chỉ vì điểm PCC cao hơn nếu CI của Δ chứa 0.

---

# 31. So sánh hai model không chỉ nhìn PCC

Giữ **prediction từng phone** (`outputs/F/f_predictions.csv`) để paired/bootstrap. Xem Δ, CI, multi-seed std.

---

# 32. Group F2 — Error Analysis

Largest `|pred − human|` trên **C8** (best direct) và **E16** (best learned trong scope F). Join Speechocean762 `scores-detail.json` markup.

Inspect:

```text
Canonical phone
Mapped / predicted score
Human score
Expert {} / () / []
Competitor LPP (B4)
```

---

# 33. Phân loại error

Taxonomy heuristic (F2):

### Type 1 — Alignment

Residual lớn + `n_frames` bất thường (không claim chắc nếu chưa nghe audio).

### Type 2 — Acoustic confusion

Human ≤ 1 và (competitor LPP thắng canonical **hoặc** expert `()` / `[]`).

### Type 3 — Accent

Expert `{}` hoặc human ≈ 1 với pred cao.

### Type 4 — Context effect

`|err_C8|` lớn nhưng `|err_E16|` nhỏ (sequence sửa) — hoặc ngược.

### Type 5 — Speaker variation

Speaker thuộc extreme MAE (D2).

---

# 34. Một điểm rất hay với Speechocean762

Dataset có cả thông tin **mispronunciation**, trong đó với một số phoneme có score thấp, annotation có thể chỉ ra phoneme được cho là phát âm thay thế. ([Hugging Face][2])

Ví dụ:

```text
Canonical: /L/
Human: incorrect
Pronounced-phone: /D/
```

Bạn có thể kiểm tra:

```text
GOP says:
L = low
D = high
```

Nếu đúng:

> GOP không chỉ cho biết "phoneme này xấu", mà còn phản ánh acoustic confusion.

Đây là một **error analysis rất đẹp** cho thesis.

---

# 35. Tôi sẽ tổ chức toàn bộ experiments thành 3 tầng

Thay vì nhìn 11 experiment riêng lẻ, hãy nghĩ:

## Tầng 1 — Does GOP work?

```text
A1 Traditional GOP
A2 GOP vs Human
```

↓

## Tầng 2 — What affects GOP?

```text
B: GOP representation

C: Acoustic model (C1–C9; C10/C11 = GOP-CTC-AF-SD on C8/C9)

D: Phone / Speaker / Proficiency
```

↓

## Tầng 3 — Can GOP information be modeled better?

```text
E1: GOP + MLP
E2: GOP + GOPT
E3: C8 GOP-S + MLP
E4: C8 GOP-S + Transformer
E5: C9 GOP-S + MLP
E6: C9 GOP-S + Transformer
E19: C10 GOP-SD + MLP
E20: C10 GOP-SD + Transformer
E21: C11 GOP-SD + MLP
E22: C11 GOP-SD + Transformer
E7: Kaldi 84-d + MLP
E8: Kaldi 84-d + Transformer
E9: C8 78-d LPP+LPR + MLP
E10: C8 78-d LPP+LPR + Transformer
E11: C9 78-d LPP+LPR + MLP
E12: C9 78-d LPP+LPR + Transformer
E13: Kaldi 84-d + MLP + phone embed
E14: Kaldi 84-d + Transformer + phone embed
E15: C8 78-d + MLP + phone embed
E16: C8 78-d + Transformer + phone embed
E17: C9 78-d + MLP + phone embed
E18: C9 78-d + Transformer + phone embed
```

↓

## Validation

```text
F1: Statistical analysis
F2: Error analysis
```

Câu chuyện nghiên cứu lúc này cực kỳ rõ:

```text
               DOES GOP WORK?
                      │
                      ▼
              WHAT AFFECTS GOP?
                      │
       ┌──────────────┼──────────────┐
       ▼              ▼              ▼
    Feature        Acoustic        Speaker/
    format          model          phoneme
       │              │              │
       └──────────────┼──────────────┘
                      ▼
             CAN WE MODEL GOP
                 BETTER?
                      │
                ┌─────┴─────┐
                ▼           ▼
               MLP         GOPT
                │           │
                └─────┬─────┘
                      ▼
                 VALIDATION
                      │
              ┌───────┴───────┐
              ▼               ▼
          Statistics        Errors
              │               │
              └───────┬───────┘
                      ▼
               EMPIRICAL FINDINGS
```

---

# 36. Thứ tự triển khai thực tế

**Đừng code tất cả cùng lúc.**

Tôi đề xuất thứ tự:

### Phase 0 — Infrastructure

* Dataset loader
* Phone mapping
* Alignment parser
* Posterior extractor
* GOP calculator
* Evaluation module
* Visualization
* Experiment config

↓

### Phase 1

**A1 → A2**

Mục tiêu:

> Có được GOP baseline đáng tin cậy.

↓

### Phase 2

**B1 → B4**

Mục tiêu:

> Hiểu GOP representation.

↓

### Phase 3

**C1 → C3**

**C4 → C9** (follow-up GOP / espeak AM)

**C10 → C11** (GOP-CTC-AF-SD trên AM C8/C9; không overwrite C8/C9)

Mục tiêu:

> Hiểu acoustic-model dependency. C10/C11 khóa AM, đổi graph AF-S → AF-SD.

↓

### Phase 4

**D1 → D3**

Mục tiêu:

> Hiểu GOP behavior.

↓

### Phase 5

**E1 → E2** (Kaldi B4)

**E3 → E6** (C8/C9 Cao GOP-S; follow-up)

**E19 → E22** (C10/C11 Cao GOP-SD + MLP/Transformer; không overwrite E3–E6)

**E7 → E12** (GOPT-style LPP+LPR × Kaldi / C8 / C9 × MLP/Transformer)

**E13 → E14** (cùng 84-d Kaldi + canonical phone embed; không overwrite E7/E8)

**E15 → E18** (cùng 78-d C8/C9 + SSL 39-way phone embed; không overwrite E9–E12)

Mục tiêu:

> Kiểm tra learned scoring.

↓

### Phase 6

**F1 → F2**

Mục tiêu:

> Statistical validation + error analysis.

---

# 37. Quan trọng nhất: mỗi experiment phải có 5 thứ

Từ giờ, với mỗi experiment trong thesis, bạn nên viết một "experiment card":

```text
Experiment ID:
Research Question:
Hypothesis:
Independent Variable:
Controlled Variables:
Dataset:
Input:
Output:
Metrics:
Expected Analysis:
```

Ví dụ A1:

```text
Experiment ID:
A1

Research Question:
How effectively does traditional GOP
reflect human phoneme pronunciation scores?

Hypothesis:
Higher GOP scores should correspond
to higher human pronunciation scores.

Independent Variable:
GOP score

Controlled Variables:
Dataset
Alignment
Acoustic model
Phone set
Test split

Input:
Audio + transcript

Output:
Phone-level GOP

Metrics:
PCC
SCC
MAE
MSE

Analysis:
GOP vs human score correlation
```

Nếu một experiment **không điền được 5–10 dòng kiểu này**, có khả năng nó chưa phải một experiment tốt.

---

# 38. Một lưu ý rất quan trọng về thesis của bạn

Vì bạn chọn **Empirical Study thuần**, tôi khuyên **chưa đưa HuBERT vào Experimental Matrix chính**.

HuBERT có thể xuất hiện trong **Group C nếu bạn có một HuBERT-based phoneme posterior model**, nhưng:

```text
HuBERT embedding
```

không nên tự động gọi là:

```text
GOP
```

Hai thứ khác nhau:

```text
GOP
= pronunciation-oriented score
derived from phone posterior

HuBERT embedding
= pretrained acoustic representation
```

Nếu sau này bạn muốn nghiên cứu:

```text
GOP + HuBERT embedding
```

thì đó sẽ là một **nghiên cứu mở rộng**, và thực chất bắt đầu chuyển sang hướng "enhancement".

Với **Hướng A thuần**, tôi sẽ giữ thesis trung tâm là:

> **What can we empirically learn about GOP?**

chứ không phải:

> **How can we build a better GOP model?**

---

## 39. Và đây là kết quả cuối cùng mà thesis nên hướng tới

Không phải:

> "Model X đạt PCC = 0.63."

Mà là một tập các findings kiểu:

> **Finding 1:** GOP has a measurable correlation with expert pronunciation ratings.

> **Finding 2:** GOP performance varies across phoneme categories.

> **Finding 3:** GOP performance is sensitive to the acoustic model used to generate phone posteriors.

> **Finding 4:** Different GOP representations exhibit different relationships with human ratings.

> **Finding 5:** Sequence-aware models can exploit GOP features more effectively than direct scoring.

> **Finding 6:** GOP has identifiable failure modes related to phoneme confusion, alignment and speaker variability.

**Đó mới chính là "sản phẩm" của một Empirical Study thuần.**

Và nếu thực hiện theo hướng này, tôi khuyên **Experiment A1 phải được làm thật chắc trước**. A1 là nền móng: nếu pipeline **Forced Alignment → Acoustic Model → Phone Posterior → GOP → Human Score** chưa đáng tin cậy thì tất cả B–F phía sau đều không có ý nghĩa.

[1]: https://arxiv.org/abs/2104.01378?utm_source=chatgpt.com "speechocean762: An Open-Source Non-native English Speech Corpus For Pronunciation Assessment"
[2]: https://huggingface.co/datasets/mispeech/speechocean762?utm_source=chatgpt.com "mispeech/speechocean762 · Datasets at Hugging Face"
[3]: https://huggingface.co/datasets/seba3y/speechocean762?utm_source=chatgpt.com "seba3y/speechocean762 · Datasets at Hugging Face"
