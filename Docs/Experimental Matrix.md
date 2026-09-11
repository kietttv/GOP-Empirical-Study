# Experimental Matrix

| Group | Experiment | IDs | Research purpose |
| ----- | ---------- | --- | ---------------- |
| A | Traditional GOP | A1 | Extract baseline scalar GOP = canonical LPP (Kaldi M13) |
| A | GOP vs human score | A2 | Measure how well traditional GOP reflects expert phoneme scores |
| B | LPP vs LPR vs GOP vector | B1–B4 | Study scalar / rank-2 GOP representation on the locked Kaldi extract |
| B | 84-d LPP+LPR (GOPT-style) | B5 | Full 42-phone confusion-profile GOP feature (OLS, not learned) |
| C | Acoustic model (direct GOP-S) | C1, C8, C9 | Study AM dependency: Kaldi LPP vs off-the-shelf XLSR-53 / lv60 Cao GOP-S (AF-S); no fine-tune on SO762 |
| C | GOP-CTC-AF-SD | C10, C11 | Same AMs as C8/C9; change graph AF-S → AF-SD (deletion in the denominator) |
| D | Phone-level analysis | D1 | Phoneme-specific GOP–human behavior (consonant / vowel; min N) |
| D | Speaker-level analysis | D2 | Robustness across 125 test speakers |
| D | Score-strata analysis | D3 | Behavior across speaker-mean sentence-accuracy tertiles (not CEFR) |
| E | GOP + MLP / Transformer | E1, E2 | Learned scoring on locked Kaldi B4 3-d; fair pair = same X, architecture only |
| E | SSL GOP-S + MLP / Transformer | E3–E6 | Learned scoring on frozen C8/C9 Cao GOP-S scalars |
| E | LPP+LPR × AM × scorer | E7–E12 | GOPT-style concat: Kaldi 84-d / C8 78-d / C9 78-d × MLP / Transformer |
| E | 84-d + phone embed | E13, E14 | Canonical phone embedding on locked Kaldi 84-d (Kaldi 42-slot) |
| E | 78-d + SSL phone embed | E15–E18 | Same C8/C9 78-d plus 39-way IPA embed (`ssl_index`); best in-scope = E16 |
| F | Bootstrap CI + paired ΔPCC | F1a, F1b | Reliability of headline PCC and model deltas on locked test phones |
| F | Multi-seed E2 / E16 | F1c | Training-seed stability (val speakers locked to seed 0) |
| F | Error taxonomy | F2 | Failure modes on C8 vs E16 + Speechocean762 expert markup |

---
