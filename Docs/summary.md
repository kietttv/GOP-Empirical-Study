# Tóm tắt kết quả thí nghiệm — GOP Empirical Study

Dataset: **Speechocean762** · Level: **phoneme** · Official test **n = 47 369** (trừ khi ghi khác) · Metrics: PCC, SCC, MAE, MSE · Score human 0–2.

Nguồn số: `outputs/{A,B,C,D,E,F}/*_results.json` (run hiện tại).

---

## A. Traditional GOP

Baseline Kaldi M13 · canonical LPP · direct scoring.


| ID  | Phương pháp                      | PCC   | SCC   | MAE   | MSE   |
| --- | -------------------------------- | ----- | ----- | ----- | ----- |
| A1  | Traditional GOP (canonical LPP)  | —     | —     | —     | —     |
| A2  | GOP vs Human (eval A1 trên test) | 0.323 | 0.313 | 0.199 | 0.120 |


A1 chỉ extract scalar GOP; metric báo cáo ở A2 (train PCC 0.342). Map train-only: slope ≈ 0.219, intercept ≈ 0.444.

---

## B. GOP Feature Representation

Cùng extract Kaldi A; chỉ đổi representation · direct / OLS.


| ID  | Phương pháp                                 | PCC   | SCC   | MAE   | MSE   |
| --- | ------------------------------------------- | ----- | ----- | ----- | ----- |
| B1  | Standard GOP (= LPP[canonical])             | 0.323 | 0.313 | 0.199 | 0.120 |
| B2  | LPP (≡ B1 trên extract này)                 | 0.323 | 0.313 | 0.199 | 0.120 |
| B3  | LPR vs best competitor                      | 0.328 | 0.342 | 0.198 | 0.121 |
| B4  | GOP-only vector OLS `[LPP, max competitor]` | 0.351 | 0.342 | 0.197 | 0.120 |
| B5  | 84-d LPP+LPR OLS (GOPT-style)               | 0.361 | 0.332 | 0.197 | 0.119 |


---

## C. Acoustic Model Dependency

Cố định dataset / CTM / direct scoring; đổi AM hoặc công thức GOP.


| ID      | Phương pháp                       | PCC   | SCC   | MAE   | MSE   |
| ------- | --------------------------------- | ----- | ----- | ----- | ----- |
| C1      | Kaldi M13 · LPP (≡ A2)            | 0.323 | 0.313 | 0.199 | 0.120 |
| C8      | XLSR-53 espeak · Cao GOP-S (AF-S) | 0.463 | 0.355 | 0.173 | 0.108 |
| C9      | lv60 espeak · Cao GOP-S (AF-S)    | 0.430 | 0.334 | 0.179 | 0.112 |
| C10 (x) | cùng AM C8 · Cao GOP-SD (AF-SD)   | 0.498 | 0.366 | 0.170 | 0.103 |
| C11 (x) | cùng AM C9 · Cao GOP-SD (AF-SD)   | 0.463 | 0.346 | 0.174 | 0.108 |


Cặp khóa graph: C8→C10 ΔPCC +0.035; C9→C11 ΔPCC +0.033.

---

## D. GOP Behavior Analysis

GOP = A1 (Kaldi LPP); map A2 đóng băng; chỉ đổi stratum.


| ID  | Phương pháp                            | PCC   | SCC   | MAE   | MSE   |
| --- | -------------------------------------- | ----- | ----- | ----- | ----- |
| D1  | Phone-level · CONSONANT (n=28 386)     | 0.343 | 0.314 | 0.184 | 0.103 |
| D1  | Phone-level · VOWEL (n=18 983)         | 0.307 | 0.310 | 0.223 | 0.147 |
| D2  | Speaker-level (mean trên 125 speakers) | 0.252 | —     | 0.200 | —     |
| D3  | Score strata · Low (42 spk)            | 0.363 | 0.351 | 0.319 | 0.275 |
| D3  | Score strata · Mid (46 spk)            | 0.228 | 0.238 | 0.157 | 0.050 |
| D3  | Score strata · High (37 spk)           | 0.207 | 0.206 | 0.117 | 0.031 |


D2: per-speaker PCC std 0.084, min 0.078, max 0.539 · speaker overlap train/test = 0. D3 = tertile speaker-mean sentence accuracy (không phải CEFR).

---

## E. GOP Learned Scoring

Learned MLP / Transformer trên feature đã khóa; pred clip [0, 2].

### E1–E2 · Kaldi B4 (3-d)


| ID  | Phương pháp      | PCC   | SCC   | MAE   | MSE   |
| --- | ---------------- | ----- | ----- | ----- | ----- |
| E1  | B4 + MLP         | 0.362 | 0.327 | 0.192 | 0.118 |
| E2  | B4 + Transformer | 0.510 | 0.368 | 0.158 | 0.100 |


### E3–E6 · Cao GOP-S (C8/C9)


| ID   | Phương pháp            | PCC   | SCC   | MAE   | MSE   |
| ---- | ---------------------- | ----- | ----- | ----- | ----- |
| E3 x | C8 GOP-S + MLP         | 0.496 | 0.355 | 0.167 | 0.103 |
| E4   | C8 GOP-S + Transformer | 0.578 | 0.361 | 0.154 | 0.090 |
| E5 x | C9 GOP-S + MLP         | 0.447 | 0.334 | 0.162 | 0.110 |
| E6   | C9 GOP-S + Transformer | 0.483 | 0.319 | 0.157 | 0.105 |


### E7–E18 · LPP+LPR × MLP/Transformer (± phone embed)


| ID   | Phương pháp                            | PCC       | SCC       | MAE       | MSE       |
| ---- | -------------------------------------- | --------- | --------- | --------- | --------- |
| E7 x | Kaldi 84-d + MLP                       | 0.446     | 0.350     | 0.168     | 0.109     |
| E8   | Kaldi 84-d + Transformer               | 0.530     | 0.379     | 0.154     | 0.097     |
| E13  | Kaldi 84-d + MLP + phone embed         | 0.552     | 0.359     | 0.159     | 0.094     |
| E14  | Kaldi 84-d + Transformer + phone embed | 0.625     | 0.403     | 0.148     | 0.082     |
| E9   | C8 78-d + MLP                          | 0.573     | 0.392     | 0.142     | 0.091     |
| E10  | C8 78-d + Transformer                  | 0.639     | 0.415     | 0.150     | 0.080     |
| E15  | C8 78-d + MLP + SSL embed              | 0.637     | 0.401     | 0.136     | 0.080     |
| E16  | C8 78-d + Transformer + SSL embed      | **0.671** | 0.412     | **0.133** | **0.074** |
| E11  | C9 78-d + MLP                          | 0.548     | 0.365     | 0.158     | 0.095     |
| E12  | C9 78-d + Transformer                  | 0.624     | 0.380     | 0.134     | 0.083     |
| E17  | C9 78-d + MLP + SSL embed              | 0.618     | 0.379     | 0.153     | 0.084     |
| E18  | C9 78-d + Transformer + SSL embed      | 0.653     | **0.427** | **0.115** | 0.079     |


Best PCC trong scope: **E16** (0.671).

---

## F. Statistical Validation & Error Analysis

Không đổi model; bootstrap / paired Δ / multi-seed / taxonomy trên prediction đã khóa.

### F1a — Bootstrap CI (PCC, n_boot=1000)


| ID  | Phương pháp                 | PCC   | 95% CI         | SCC   |
| --- | --------------------------- | ----- | -------------- | ----- |
| C1  | Direct Kaldi (score-space)  | 0.342 | [0.333, 0.351] | 0.313 |
| B4  | B4 OLS                      | 0.351 | [0.342, 0.361] | 0.342 |
| C9  | Direct lv60 GOP-S           | 0.430 | [0.420, 0.440] | 0.334 |
| C8  | Direct XLSR GOP-S           | 0.463 | [0.453, 0.473] | 0.355 |
| E7  | 84-d MLP                    | 0.446 | [0.433, 0.460] | 0.350 |
| E2  | B4 Transformer              | 0.510 | [0.495, 0.523] | 0.368 |
| E8  | 84-d Transformer            | 0.530 | [0.516, 0.543] | 0.379 |
| E12 | C9 78-d Transformer         | 0.624 | [0.611, 0.637] | 0.380 |
| E14 | 84-d Transformer + embed    | 0.625 | [0.612, 0.637] | 0.403 |
| E10 | C8 78-d Transformer         | 0.639 | [0.626, 0.651] | 0.415 |
| E18 | C9 78-d Transformer + embed | 0.653 | [0.639, 0.664] | 0.427 |
| E16 | C8 78-d Transformer + embed | 0.671 | [0.660, 0.683] | 0.412 |


### F1b — Paired ΔPCC (tất cả CI loại trừ 0)


| Contrast    | ΔPCC   | 95% CI         |
| ----------- | ------ | -------------- |
| C8 − C1     | +0.121 | [0.111, 0.131] |
| C9 − C1     | +0.088 | [0.077, 0.098] |
| C8 − C9     | +0.033 | [0.027, 0.040] |
| E2 − B4_OLS | +0.158 | [0.147, 0.168] |
| E8 − E7     | +0.084 | [0.072, 0.096] |
| E14 − E8    | +0.095 | [0.086, 0.104] |
| E16 − E10   | +0.033 | [0.026, 0.039] |
| E18 − E12   | +0.028 | [0.020, 0.038] |
| E16 − C8    | +0.208 | [0.198, 0.217] |


### F1c — Multi-seed (seeds 0–4; val speakers khóa seed 0)


| ID  | Phương pháp                 | PCC mean ± std | min / max     |
| --- | --------------------------- | -------------- | ------------- |
| E2  | B4 Transformer              | 0.509 ± 0.001  | 0.507 / 0.510 |
| E16 | C8 78-d Transformer + embed | 0.665 ± 0.004  | 0.659 / 0.670 |


### F2 — Error taxonomy (primary_type)


| Type         | C8     | E16    |
| ------------ | ------ | ------ |
| other        | 40 025 | 40 205 |
| T3 accent    | 4 516  | 4 545  |
| T2 confusion | 2 149  | 2 149  |
| T4 context   | 402    | 176    |
| T5 speaker   | 202    | 186    |
| T1 alignment | 75     | 108    |


Mean |err| human≈0: C8 1.31 → E16 0.91; human≈2: C8 0.122 → E16 0.089.

---

## Ranking nhanh (test PCC)


| Hạng | ID    | PCC   | Ghi chú                           |
| ---- | ----- | ----- | --------------------------------- |
| 1    | E16   | 0.671 | C8 78-d + Transformer + SSL embed |
| 2    | E18   | 0.653 | C9 78-d + Transformer + SSL embed |
| 3    | E10   | 0.639 | C8 78-d + Transformer             |
| 4    | E15   | 0.637 | C8 78-d + MLP + SSL embed         |
| 5    | E14   | 0.625 | Kaldi 84-d + Transformer + embed  |
| …    | C10   | 0.498 | Best direct SSL (AF-SD)           |
| …    | C8    | 0.463 | Best direct AF-S                  |
| …    | A2/C1 | 0.323 | Traditional Kaldi GOP baseline    |


