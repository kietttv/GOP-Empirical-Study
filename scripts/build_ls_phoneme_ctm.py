#!/usr/bin/env python3
"""Build Kaldi-format phone CTM + CSV from a local LibriSpeech tree.

This is AM-construction input for Group C (C2/C3 CTC fine-tune), not GOP.

Phone *sequences* come from official *.trans.txt via CMUdict (first variant),
with g2p_en only for OOV words. Timestamps are equal shares of the flac
duration so the file is valid 5-column CTM. Cao / finetune_phoneme_ctc.py
ignore start/dur and only read the phone string.

This is NOT Kaldi force-alignment (no ali-to-phones). Do not call it the
IS24 alignment dump.

Example:

    python scripts/build_ls_phoneme_ctm.py
    # writes train.ctm / dev.ctm into data/LibriSpeech ASR corpus/
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from gop_empirical.acoustic.alignment import clean_phone  # noqa: E402
from gop_empirical.acoustic.phones import load_phone_inventory  # noqa: E402

CMUDICT_URLS = (
    "https://huggingface.co/datasets/bene-ges/en_cmudict/resolve/main/train.txt",
    "https://huggingface.co/datasets/bene-ges/en_cmudict/resolve/main/test.txt",
)
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_G2P_MAP = {
    "AX": "AH",
    "IX": "IH",
    "DX": "D",
    "EL": "L",
    "EM": "M",
    "EN": "N",
    "UX": "UW",
    "Q": None,
}


def flac_duration_s(path: Path) -> float:
    """Read STREAMINFO; no soundfile dependency."""
    with path.open("rb") as f:
        if f.read(4) != b"fLaC":
            raise ValueError(f"not flac: {path}")
        while True:
            hdr = f.read(4)
            if len(hdr) < 4:
                break
            is_last = hdr[0] & 0x80
            block_type = hdr[0] & 0x7F
            size = int.from_bytes(hdr[1:4], "big")
            data = f.read(size)
            if block_type == 0:
                packed = int.from_bytes(data[10:18], "big")
                rate = packed >> 44
                total = packed & ((1 << 36) - 1)
                if rate <= 0 or total <= 0:
                    raise ValueError(f"bad STREAMINFO: {path}")
                return float(total) / float(rate)
            if is_last:
                break
    raise ValueError(f"no STREAMINFO: {path}")


def _parse_lexicon_line(line: str) -> tuple[str, list[str]] | None:
    line = line.strip()
    if not line or line.startswith(";;;") or line.startswith("#"):
        return None
    if "\t" in line:
        word, rest = line.split("\t", 1)
    else:
        parts = line.split()
        if len(parts) < 2:
            return None
        word, rest = parts[0], " ".join(parts[1:])
    word = "".join(word.split()).lower()
    if word.endswith(")") and "(" in word:
        return None
    word = word.strip("'")
    phones = rest.split("#", 1)[0].split()
    if not word or not phones:
        return None
    return word, phones


def load_cmudict(path: Path) -> dict[str, list[str]]:
    """word (lower) -> first-variant CMU phones with stress (AA1, ...)."""
    lexicon: dict[str, list[str]] = {}
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            parsed = _parse_lexicon_line(line)
            if parsed is None:
                continue
            word, phones = parsed
            lexicon.setdefault(word, phones)
    return lexicon


def ensure_cmudict(path: Path) -> dict[str, list[str]]:
    extra = path.parent / "cmudict_test.txt"
    if not path.is_file() or path.stat().st_size < 100_000:
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading CMUdict -> {path}")
        last_err: Exception | None = None
        for url in CMUDICT_URLS[:1]:
            try:
                urllib.request.urlretrieve(url, path)
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        if last_err is not None and (not path.is_file() or path.stat().st_size < 100_000):
            raise RuntimeError(
                f"Could not download CMUdict to {path}. "
                "Place bene-ges/en_cmudict train.txt there (tab-separated letters + phones)."
            ) from last_err
    lex = load_cmudict(path)
    if extra.is_file():
        n0 = len(lex)
        lex.update({k: v for k, v in load_cmudict(extra).items() if k not in lex})
        print(f"merged test lexicon +{len(lex) - n0}")
    print(f"cmudict entries {len(lex)}")
    return lex


def _try_g2p():
    try:
        import nltk

        nltk.data.find("corpora/cmudict")
        from g2p_en import G2p

        return G2p()
    except Exception:
        return None


def map_phone(symbol: str, scored: set[str]) -> str | None:
    mapped = _G2P_MAP.get(symbol, symbol)
    if mapped is None:
        return None
    cleaned = clean_phone(mapped)
    if cleaned is None or cleaned not in scored:
        return None
    return cleaned


# Longest-first English graphemes -> CMU phones (OOV names / rare words only).
_FALLBACK_RULES: list[tuple[str, list[str]]] = [
    ("tion", ["SH", "AH", "N"]),
    ("sion", ["ZH", "AH", "N"]),
    ("ture", ["CH", "ER"]),
    ("ough", ["AH", "F"]),
    ("augh", ["AO", "F"]),
    ("eigh", ["EY"]),
    ("igh", ["AY"]),
    ("que", ["K"]),
    ("qu", ["K", "W"]),
    ("ph", ["F"]),
    ("th", ["TH"]),
    ("sh", ["SH"]),
    ("ch", ["CH"]),
    ("ng", ["NG"]),
    ("ck", ["K"]),
    ("wh", ["W"]),
    ("kn", ["N"]),
    ("wr", ["R"]),
    ("ee", ["IY"]),
    ("oo", ["UW"]),
    ("ea", ["IY"]),
    ("oa", ["OW"]),
    ("ai", ["EY"]),
    ("ay", ["EY"]),
    ("oy", ["OY"]),
    ("oi", ["OY"]),
    ("ow", ["AW"]),
    ("ou", ["AW"]),
    ("au", ["AO"]),
    ("aw", ["AO"]),
    ("ew", ["UW"]),
    ("ie", ["IY"]),
    ("ei", ["IY"]),
    ("er", ["ER"]),
    ("ar", ["AA", "R"]),
    ("or", ["AO", "R"]),
    ("ir", ["ER"]),
    ("ur", ["ER"]),
    ("a", ["AE"]),
    ("e", ["EH"]),
    ("i", ["IH"]),
    ("o", ["AA"]),
    ("u", ["AH"]),
    ("y", ["IY"]),
    ("b", ["B"]),
    ("c", ["K"]),
    ("d", ["D"]),
    ("f", ["F"]),
    ("g", ["G"]),
    ("h", ["HH"]),
    ("j", ["JH"]),
    ("k", ["K"]),
    ("l", ["L"]),
    ("m", ["M"]),
    ("n", ["N"]),
    ("p", ["P"]),
    ("r", ["R"]),
    ("s", ["S"]),
    ("t", ["T"]),
    ("v", ["V"]),
    ("w", ["W"]),
    ("x", ["K", "S"]),
    ("z", ["Z"]),
]
_FALLBACK_RULES.sort(key=lambda item: -len(item[0]))


def fallback_g2p(word: str, scored: set[str]) -> list[str] | None:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return None
    i = 0
    raw: list[str] = []
    while i < len(w):
        hit = False
        for g, phones in _FALLBACK_RULES:
            if w.startswith(g, i):
                raw.extend(phones)
                i += len(g)
                hit = True
                break
        if not hit:
            i += 1
    out = [p for p in (map_phone(x, scored) for x in raw) if p is not None]
    return out or None


def phones_for_word(word: str, lexicon: dict[str, list[str]], g2p, scored: set[str]) -> list[str] | None:
    key = word.lower()
    candidates = [key]
    if key.endswith("'s"):
        candidates.append(key[:-2])
    if "'" in key:
        candidates.append(key.replace("'", ""))
    if "-" in key:
        parts = [p for p in key.split("-") if p]
        part_phones: list[str] = []
        ok = True
        for p in parts:
            seq = phones_for_word(p, lexicon, g2p, scored)
            if seq is None:
                ok = False
                break
            part_phones.extend(seq)
        if ok and part_phones:
            return part_phones

    raw = None
    for cand in candidates:
        if cand in lexicon:
            raw = lexicon[cand]
            break
    if raw is None and g2p is not None:
        try:
            raw = [p for p in g2p(word) if isinstance(p, str) and p.strip() and p not in {" "}]
        except Exception:
            raw = None
    if not raw:
        return fallback_g2p(word, scored)
    out: list[str] = []
    for p in raw:
        mapped = map_phone(p, scored)
        if mapped is None:
            continue
        out.append(mapped)
    return out or fallback_g2p(word, scored)


def iter_transcripts(split_dir: Path) -> list[tuple[str, Path, str]]:
    """(utt_id, flac_path, text) from official LibriSpeech *.trans.txt."""
    rows: list[tuple[str, Path, str]] = []
    for trans in sorted(split_dir.rglob("*.trans.txt")):
        with trans.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                utt, _, text = line.partition(" ")
                flac = trans.parent / f"{utt}.flac"
                if not flac.is_file():
                    wav = trans.parent / f"{utt}.wav"
                    flac = wav if wav.is_file() else flac
                rows.append((utt, flac, text.strip()))
    return rows


def _rel_posix(path: Path, base: Path) -> str:
    """Path relative to base, forward slashes (portable in CSV/meta)."""
    return path.resolve().relative_to(base.resolve()).as_posix()


def write_split(
    name: str,
    split_dir: Path,
    out_ctm: Path,
    out_csv: Path,
    lexicon: dict[str, list[str]],
    g2p,
    scored: set[str],
    path_base: Path,
) -> dict:
    utts = iter_transcripts(split_dir)
    stats: Counter[str] = Counter()
    csv_rows: list[dict[str, str]] = []
    out_ctm.parent.mkdir(parents=True, exist_ok=True)
    missing_audio = 0
    oov_words: Counter[str] = Counter()

    with out_ctm.open("w", encoding="utf-8", newline="\n") as ctm_f:
        for utt, audio, text in utts:
            stats["transcripts"] += 1
            if not audio.is_file():
                missing_audio += 1
                stats["skip_no_audio"] += 1
                continue
            words = _WORD_RE.findall(text)
            phones: list[str] = []
            bad = False
            used_fallback = False
            for w in words:
                key = w.lower()
                in_lex = key in lexicon or key.rstrip("'s") in lexicon or key.replace("'", "") in lexicon
                seq = phones_for_word(w, lexicon, g2p, scored)
                if seq is None:
                    oov_words[w.lower()] += 1
                    bad = True
                    break
                if not in_lex:
                    stats["fallback_words"] += 1
                    used_fallback = True
                    oov_words[w.lower()] += 1
                phones.extend(seq)
            if bad or not phones:
                stats["skip_oov"] += 1
                continue
            if used_fallback:
                stats["utt_with_fallback"] += 1
            try:
                dur_s = flac_duration_s(audio) if audio.suffix.lower() == ".flac" else None
            except ValueError:
                dur_s = None
            if dur_s is None or dur_s <= 0:
                dur_s = 0.03 * len(phones)
            n = len(phones)
            for i, ph in enumerate(phones):
                start = round(i * dur_s / n, 3)
                end = round(dur_s, 3) if i == n - 1 else round((i + 1) * dur_s / n, 3)
                dur = max(end - start, 0.001)
                ctm_f.write(f"{utt} 1 {start:.3f} {dur:.3f} {ph}\n")
            stats["kept"] += 1
            stats["phones"] += len(phones)
            csv_rows.append({"file_name": _rel_posix(audio, path_base), "utt_id": utt})

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file_name", "utt_id"])
        w.writeheader()
        w.writerows(csv_rows)

    report = {
        "split": name,
        "split_dir": _rel_posix(split_dir, path_base),
        "ctm": out_ctm.name,
        "csv": out_csv.name,
        "n_transcripts": stats["transcripts"],
        "n_kept": stats["kept"],
        "n_phones": stats["phones"],
        "n_skip_oov": stats["skip_oov"],
        "n_utt_with_fallback": stats["utt_with_fallback"],
        "n_fallback_words": stats["fallback_words"],
        "n_skip_no_audio": stats["skip_no_audio"],
        "missing_audio": missing_audio,
        "top_oov": oov_words.most_common(20),
    }
    print(
        f"{name}: kept {stats['kept']}/{stats['transcripts']} "
        f"phones {stats['phones']} skip_oov {stats['skip_oov']} "
        f"no_audio {stats['skip_no_audio']}"
    )
    return report


def find_split_dir(root: Path, name: str) -> Path:
    """Official dump is often root/train-clean-100/train-clean-100/speaker/..."""
    direct = root / name
    nested = direct / name
    if nested.is_dir() and any(nested.rglob("*.trans.txt")):
        return nested
    if direct.is_dir() and any(direct.rglob("*.trans.txt")):
        return direct
    hits = [p for p in root.rglob(name) if p.is_dir()]
    for p in sorted(hits, key=lambda x: len(x.parts)):
        if any(p.rglob("*.trans.txt")):
            return p
    raise FileNotFoundError(f"LibriSpeech split {name!r} not found under {root}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--librispeech-root",
        type=Path,
        default=PACKAGE_ROOT / "data" / "LibriSpeech ASR corpus",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: same as --librispeech-root (LibriSpeech ASR corpus).",
    )
    parser.add_argument("--inventory", type=Path, default=PACKAGE_ROOT / "data" / "phone_inventory.json")
    args = parser.parse_args()

    root = args.librispeech_root
    if not root.is_dir():
        raise FileNotFoundError(f"LibriSpeech root not found: {root}")

    inv = load_phone_inventory(args.inventory)
    scored = set(inv.scored_symbols)
    out = args.output_dir if args.output_dir is not None else root
    out.mkdir(parents=True, exist_ok=True)
    lexicon = ensure_cmudict(out / "cmudict.dict")
    g2p = _try_g2p()
    print("g2p_en", "yes" if g2p is not None else "NO (OOV words use grapheme fallback)")

    reports = []
    for split, ctm_name, csv_name in (
        ("train-clean-100", "train.ctm", "train.csv"),
        ("dev-clean", "dev.ctm", "dev.csv"),
    ):
        split_dir = find_split_dir(root, split)
        reports.append(
            write_split(
                split,
                split_dir,
                out / ctm_name,
                out / csv_name,
                lexicon,
                g2p,
                scored,
                path_base=out,
            )
        )

    meta = {
        "label_kind": "cmudict_lexicon_ctm",
        "kaldi_force_align": False,
        "timestamps": "equal_share_of_flac_duration",
        "phone_set": "39_cmu_scored",
        "note": (
            "CTC fine-tune uses phone sequences only. Timestamps are placeholders. "
            "Not Cao/Kaldi ali-to-phones."
        ),
        "g2p_en": g2p is not None,
        "splits": reports,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
