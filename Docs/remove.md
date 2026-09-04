# Thí nghiệm loại bỏ

## C2 – C7 (AM fine-tune in-house / negative control)

| ID | Phương pháp | PCC | SCC | MAE | MSE |
| -- | ----------- | --: | --: | --: | --: |
| C2 | Wav2Vec2 CTC · mean on Kaldi span | −0.038 | −0.007 | 0.217 | 0.134 |
| C3 | HuBERT CTC · mean on Kaldi span | −0.060 | −0.029 | 0.217 | 0.133 |
| C4 | HuBERT CTC · max on Kaldi span | 0.055 | 0.074 | 0.216 | 0.133 |
| C5 | HuBERT CTC · Cao GOP-S (AF-S) | 0.346 | 0.340 | 0.193 | 0.120 |
| C6 | Wav2Vec2 CTC · max on Kaldi span | 0.074 | 0.094 | 0.216 | 0.133 |
| C7 | Wav2Vec2 CTC · Cao GOP-S (AF-S) | 0.378 | 0.353 | 0.189 | 0.117 |

## E19 – E22 (learned scoring trên Cao GOP-SD)

| ID | Phương pháp | PCC | SCC | MAE | MSE |
| -- | ----------- | --: | --: | --: | --: |
| E19 | C10 GOP-SD + MLP | 0.592 | 0.418 | 0.157 | 0.090 |
| E20 | C10 GOP-SD + Transformer | 0.543 | 0.293 | 0.176 | 0.098 |
| E21 | C11 GOP-SD + MLP | 0.555 | 0.402 | 0.163 | 0.096 |
| E22 | C11 GOP-SD + Transformer | 0.497 | 0.249 | 0.180 | 0.104 |
