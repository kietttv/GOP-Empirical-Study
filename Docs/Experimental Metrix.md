### Experimental Matrix


| Group | Experiment                 | Research purpose                |
| ----- | -------------------------- | ------------------------------- |
| A     | Traditional GOP            | Establish baseline              |
| A     | GOP vs Human Score         | Measure effectiveness           |
| B     | LPP vs LPR vs GOP          | Study feature representation    |
| B     | 84-d LPP+LPR (GOPT-style)  | Full confusion-profile GOP feature |
| C     | Different acoustic models  | Study acoustic-model dependency |
| C     | C10/C11 GOP-CTC-AF-SD      | Deletion graph on locked C8/C9 AMs |
| D     | Phone-level analysis       | Find phoneme-specific behavior  |
| D     | Speaker-level analysis     | Study robustness                |
| D     | Proficiency-level analysis | Study behavior across learners  |
| E     | GOP + MLP                  | Study learned scoring           |
| E     | GOP + GOPT                 | Study Transformer-based scoring |
| E     | C8/C9 GOP-S + MLP/Transformer | Learned scoring on SSL GOP-S (E3–E6) |
| E     | C10/C11 GOP-SD + MLP/Transformer | Learned scoring on SSL GOP-SD (E19–E22) |
| E     | LPP+LPR × 3 AM × MLP/Transformer | GOPT-style concat (E7–E12; 84-d / 78-d) |
| E     | 84-d + phone embed (E13/E14)   | GOPT-style identity on locked 84-d |
| E     | 78-d + SSL phone embed (E15–E18) | Same 78-d C8/C9 plus 39-way embed |
| F     | Bootstrap CI + paired ΔPCC | Test reliability of headline models |
| F     | Multi-seed E2 / E16        | Training-seed stability         |
| F     | Error taxonomy (C8 / E16)  | Failure modes + expert markup    |


