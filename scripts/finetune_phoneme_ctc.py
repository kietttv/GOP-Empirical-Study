#!/usr/bin/env python3
"""Fine-tune Wav2Vec2 or HuBERT as a 39-phone CTC recognizer.

Acoustic-model construction for Group C — not a GOP experiment. The phone
inventory is frozen to data/phone_inventory.json (same 39 CMU phones as the
existing Wav2Vec2 checkpoint vocab). Hidden embeddings are never used as GOP.

Example (HuBERT / C3):

    python scripts/finetune_phoneme_ctc.py ^
        --backbone hubert ^
        --pretrained facebook/hubert-base-ls960 ^
        --processor-dir "../notebook implement/CTC-based-GOP/is24/models/processor_config_gop" ^
        --train-ctm "data/LibriSpeech ASR corpus/train.ctm" ^
        --dev-ctm "data/LibriSpeech ASR corpus/dev.ctm" ^
        --train-csv "data/LibriSpeech ASR corpus/train.csv" ^
        --dev-csv "data/LibriSpeech ASR corpus/dev.csv" ^
        --output-dir data/hubert_phoneme_ctc

CSV must have a file_name (or path) column pointing at 16 kHz wavs. CTM is
Kaldi 5-column phone alignment; silence tokens are dropped to match the 39-phone
vocab. After training, point configs/c_acoustic_model.yaml hubert_checkpoint at
--output-dir and run scripts/extract_ssl_gop.py --model hubert.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from gop_empirical.acoustic.alignment import clean_phone  # noqa: E402
from gop_empirical.acoustic.phones import load_phone_inventory  # noqa: E402

_RE_UTT = re.compile(r"(.*/)*(.*)\.(.*$)")


def _require_train_stack():
    try:
        import torch
        from datasets import Audio, Dataset, load_dataset
        from transformers import (
            AutoFeatureExtractor,
            AutoModelForCTC,
            Trainer,
            TrainingArguments,
            Wav2Vec2CTCTokenizer,
            Wav2Vec2Processor,
        )
    except ImportError as exc:
        raise ImportError(
            "finetune_phoneme_ctc.py needs torch, transformers, datasets. "
            "pip install -r requirements-ssl.txt"
        ) from exc
    return {
        "torch": torch,
        "Audio": Audio,
        "Dataset": Dataset,
        "load_dataset": load_dataset,
        "AutoFeatureExtractor": AutoFeatureExtractor,
        "AutoModelForCTC": AutoModelForCTC,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "Wav2Vec2CTCTokenizer": Wav2Vec2CTCTokenizer,
        "Wav2Vec2Processor": Wav2Vec2Processor,
    }


def read_phoneme_ctm(path: Path) -> dict[str, list[str]]:
    """utt_id -> scored CMU phones (silence dropped, stress stripped)."""
    trans: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"{path}: expected Kaldi CTM 5 fields, got {parts!r}")
            utt, phone = parts[0], parts[4]
            cleaned = clean_phone(phone)
            if cleaned is None:
                continue
            trans.setdefault(utt, []).append(cleaned)
    return trans


def write_frozen_vocab(processor_dir: Path, inventory_path: Path, output_dir: Path) -> dict[str, int]:
    """Reuse the 39-phone vocab; never invent a new phone set mid-experiment."""
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_src = processor_dir / "vocab.json"
    inv = load_phone_inventory(inventory_path)
    if vocab_src.is_file():
        vocab = json.loads(vocab_src.read_text(encoding="utf-8"))
    else:
        vocab = {"<pad>": 0}
        for i, sym in enumerate(inv.scored_symbols, start=1):
            vocab[sym] = i
    scored = set(inv.scored_symbols)
    extra = {k for k in vocab if k not in scored and k != "<pad>"}
    if extra:
        raise ValueError(f"processor vocab has extra tokens {sorted(extra)}; Group C locks 39 CMU phones")
    missing = scored - set(vocab)
    if missing:
        raise ValueError(f"processor vocab missing scored phones {sorted(missing)}")
    (output_dir / "vocab.json").write_text(json.dumps(vocab, indent=2) + "\n", encoding="utf-8")
    return vocab


@dataclass
class DataCollatorCTCWithPadding:
    processor: Any
    padding: bool = True

    def __call__(self, features: list[dict]) -> dict:
        hf = _require_train_stack()
        torch = hf["torch"]
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features, padding=self.padding, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def load_split(csv_path: Path, trans: dict[str, list[str]], sampling_rate: int):
    hf = _require_train_stack()
    import pandas as pd

    csv_path = csv_path.resolve()
    csv_dir = csv_path.parent
    df = pd.read_csv(csv_path)
    col = "file_name" if "file_name" in df.columns else "path"

    def resolve_audio(p: str) -> str:
        path = Path(p)
        if not path.is_absolute():
            path = (csv_dir / path).resolve()
        return str(path)

    paths = [resolve_audio(str(p)) for p in df[col].tolist()]
    ds = hf["Dataset"].from_dict({"audio": paths, "path": paths})
    ds = ds.cast_column("audio", hf["Audio"](sampling_rate=sampling_rate))

    def attach(batch):
        uid = _RE_UTT.match(batch["path"])
        utt = uid.group(2) if uid else Path(batch["path"]).stem
        phones = trans.get(utt) or trans.get("lbi-" + utt)
        batch["speech"] = batch["audio"]["array"]
        batch["p_text"] = " ".join(phones) if phones else None
        batch["utt_id"] = utt
        return batch

    ds = ds.map(attach, remove_columns=["audio"])
    return ds.filter(lambda ex: ex["p_text"] is not None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True, choices=("wav2vec2", "hubert"))
    parser.add_argument("--pretrained", required=True, help="HF id or local encoder dir")
    parser.add_argument("--processor-dir", required=True, type=Path)
    parser.add_argument("--train-ctm", required=True, type=Path)
    parser.add_argument("--dev-ctm", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--dev-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--inventory", type=Path, default=PACKAGE_ROOT / "data" / "phone_inventory.json")
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--epochs", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    hf = _require_train_stack()
    vocab = write_frozen_vocab(args.processor_dir, args.inventory, args.output_dir)
    tokenizer = hf["Wav2Vec2CTCTokenizer"](
        str(args.output_dir / "vocab.json"),
        unk_token=None,
        pad_token="<pad>",
        word_delimiter_token=None,
        bos_token=None,
        eos_token=None,
    )
    feature_extractor = hf["AutoFeatureExtractor"].from_pretrained(args.pretrained)
    processor = hf["Wav2Vec2Processor"](feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.save_pretrained(args.output_dir)

    train_trans = read_phoneme_ctm(args.train_ctm)
    dev_trans = read_phoneme_ctm(args.dev_ctm)
    train_ds = load_split(args.train_csv, train_trans, args.sampling_rate)
    dev_ds = load_split(args.dev_csv, dev_trans, args.sampling_rate)

    def prepare(batch):
        batch["input_values"] = processor(audio=batch["speech"], sampling_rate=args.sampling_rate).input_values[0]
        batch["labels"] = processor(text=batch["p_text"].split(), is_split_into_words=True).input_ids
        return batch

    train_ds = train_ds.map(prepare)
    dev_ds = dev_ds.map(prepare)

    model = hf["AutoModelForCTC"].from_pretrained(
        args.pretrained,
        ctc_loss_reduction="mean",
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=len(vocab),
        ignore_mismatched_sizes=True,
    )
    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
    elif hasattr(model, "freeze_feature_extractor"):
        model.freeze_feature_extractor()

    training_args = hf["TrainingArguments"](
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=500,
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
        fp16=True,
        seed=args.seed,
        report_to=[],
    )
    trainer = hf["Trainer"](
        model=model,
        data_collator=DataCollatorCTCWithPadding(processor=processor),
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"saved {args.backbone} phoneme-CTC AM to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
