Trong `GOP-Empirical-Study/data/`, những thứ **có thể tạo lại bằng code trong repo** như sau.

---

## 1. SSL GOP extracts — `scripts/extract_ssl_gop.py`

Mỗi thư mục: `{tr,te}_ssl_gop.csv` (scalar) hoặc `{tr,te}_ssl_lpp_lpr.npz` (78-d).

| Thư mục | Lệnh | Dùng cho |
|---|---|---|
| `data/wav2vec2_gop/` | `--model wav2vec2` (default `mean`) | C2 |
| `data/wav2vec2_gop_max/` | `--model wav2vec2 --gop max` | C6 |
| `data/wav2vec2_gop_cao/` | `--model wav2vec2 --gop cao_s` | C7 |
| `data/hubert_gop/` | `--model hubert` | C3 |
| `data/hubert_gop_max/` | `--model hubert --gop max` | C4 |
| `data/hubert_gop_cao/` | `--model hubert --gop cao_s` | C5 |
| `data/xlsr_espeak_gop_cao/` | `--model xlsr_espeak --gop cao_s` | C8 |
| `data/lv60_espeak_gop_cao/` | `--model lv60_espeak --gop cao_s` | C9 |
| `data/xlsr_espeak_gop_cao_sd/` | `--model xlsr_espeak --gop cao_sd` | C10 |
| `data/lv60_espeak_gop_cao_sd/` | `--model lv60_espeak --gop cao_sd` | C11 |
| `data/xlsr_espeak_lpp_lpr/` | `--model xlsr_espeak --gop lpp_lpr` | E9/E10/E15/E16 |
| `data/lv60_espeak_lpp_lpr/` | `--model lv60_espeak --gop lpp_lpr` | E11/E12/E17/E18 |

Cần input sẵn: `so762_inputs/` + checkpoint AM tương ứng.

---

## 2. LibriSpeech CTM/CSV — `scripts/build_ls_phoneme_ctm.py`

Trong `data/LibriSpeech ASR corpus/` (cần tree flac sẵn):

| File | Nội dung |
|---|---|
| `train.ctm`, `dev.ctm` | phone CTM (lexicon CMUdict, **không** Kaldi ali) |
| `train.csv`, `dev.csv` | metadata wav cho fine-tune |
| `cmudict.dict` | lexicon tải khi chạy script |

Dùng cho AM C2/C3, không phải extract SO762 GOP.

---

## 3. Phoneme-CTC checkpoints — `scripts/finetune_phoneme_ctc.py` / notebook C2–C3

| Đường dẫn (theo `SOURCE.txt` / CLI) | Cách tạo |
|---|---|
| `data/wav2vec2_phoneme_ctc/` (hoặc `models/...` theo yaml) | fine-tune Wav2Vec2 trên LS-100 |
| `data/hubert_phoneme_ctc/` | fine-tune HuBERT trên LS-100 |

Script ghi `vocab.json`, processor, weights vào `--output-dir`.

---

## Không tạo bằng code trong repo (chỉ copy / khóa tay)

| Path | Lý do |
|---|---|
| `data/scores.json` | copy Speechocean762 / GOPT dump |
| `data/kaldi_gop_librispeech/` | Kaldi extract sẵn — **không** re-run |
| `data/phone_inventory.json` | frozen map |
| `data/cmu_to_espeak.json` | frozen CMU→IPA |
| `data/espeak_ctc_vocab.json` | frozen vocab C8/C9 |
| `data/speechocean762/` | dataset gốc (utt2spk, scores-detail, …) |
| `data/so762_inputs/wavs|segments|keys/` | copy từ pipeline GOPT |
| `data/LibriSpeech ASR corpus/{train-clean-100,dev-clean}/` | corpus audio gốc (chỉ CTM/CSV phía trên là code tạo) |
| `data/SOURCE.txt`, `*/README.txt` | tài liệu |

C8/C9 weights nằm ở `models/wav2vec2-*-espeak-cv-ft/` (`download_espeak_ctc.py`), **không** trong `data/`.

---

**Tóm lại:** Trong `data/`, code tạo được chủ yếu là **12 bộ SSL extract**, **CTM/CSV LibriSpeech**, và **checkpoint C2/C3**. Phần còn lại là input khóa / copy ngoài.