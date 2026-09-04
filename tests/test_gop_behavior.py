"""Unit tests for Group D stratified GOP behavior analysis."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gop_empirical.data.speakers import (  # noqa: E402
    load_split_utt2spk,
    load_speaker_metadata,
)
from gop_empirical.eval.behavior import (  # noqa: E402
    assign_score_stratum,
    attach_score_strata,
    phone_class,
    speaker_mean_sentence_accuracy,
    subset_metrics,
    tertile_cutpoints,
)
from gop_empirical.eval.metrics import apply_linear_map, error_metrics, fit_linear_score_map  # noqa: E402


class PhoneClassTests(unittest.TestCase):
    def test_vowels_and_consonants(self):
        self.assertEqual(phone_class("AE"), "vowel")
        self.assertEqual(phone_class("IH"), "vowel")
        self.assertEqual(phone_class("TH"), "consonant")
        self.assertEqual(phone_class("R"), "consonant")


class SubsetMetricsTests(unittest.TestCase):
    def setUp(self):
        self.mapping = {"slope": 0.5, "intercept": 0.1}

    def test_min_n_suppresses_correlation(self):
        gop = np.array([1.0, 2.0, 3.0])
        human = np.array([1.0, 1.5, 2.0])
        out = subset_metrics(gop, human, self.mapping, min_n=5)
        self.assertEqual(out["n"], 3)
        self.assertFalse(out["reported"])
        self.assertIsNone(out["pcc"])
        self.assertIsNone(out["scc"])
        self.assertIsNotNone(out["mae"])

    def test_constant_human_nulls_pcc(self):
        gop = np.linspace(0.0, 1.0, 20)
        human = np.full(20, 2.0)
        out = subset_metrics(gop, human, self.mapping, min_n=10)
        self.assertTrue(out["reported"])
        self.assertIsNone(out["pcc"])
        self.assertIsNone(out["scc"])
        self.assertAlmostEqual(out["mean_human"], 2.0)

    def test_does_not_refit_mapping_on_subset(self):
        rng = np.random.default_rng(0)
        gop = rng.normal(size=40)
        human = 0.2 * gop + 0.4 + rng.normal(scale=0.05, size=40)
        frozen = {"slope": 2.0, "intercept": -0.5}
        out = subset_metrics(gop, human, frozen, min_n=10)
        pred = apply_linear_map(gop, frozen, clip=(0.0, 2.0))
        expected = error_metrics(pred, human)
        self.assertAlmostEqual(out["mae"], expected["mae"])
        self.assertAlmostEqual(out["mse"], expected["mse"])
        fitted = fit_linear_score_map(gop, human)
        pred_fit = apply_linear_map(gop, fitted, clip=(0.0, 2.0))
        fitted_mae = error_metrics(pred_fit, human)["mae"]
        self.assertNotAlmostEqual(out["mae"], fitted_mae, places=6)


class TertileTests(unittest.TestCase):
    def test_cutpoints_and_labels_are_stable(self):
        values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        q33, q66 = tertile_cutpoints(values)
        self.assertLess(q33, q66)
        self.assertEqual(assign_score_stratum(q33, q33, q66), "Low")
        self.assertEqual(assign_score_stratum((q33 + q66) / 2.0, q33, q66), "Mid")
        self.assertEqual(assign_score_stratum(q66 + 0.01, q33, q66), "High")

    def test_attach_strata_uses_utterance_means(self):
        df = pd.DataFrame(
            {
                "split": ["test"] * 6,
                "speaker": ["0001", "0001", "0002", "0002", "0003", "0003"],
                "utt_id": ["u1", "u1", "u2", "u2", "u3", "u3"],
                "sentence_accuracy": [3.0, 3.0, 6.0, 6.0, 9.0, 9.0],
                "gop": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
                "human_score": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            }
        )
        means = speaker_mean_sentence_accuracy(df)
        self.assertAlmostEqual(float(means["0001"]), 3.0)
        out, meta = attach_score_strata(df, split="test")
        self.assertEqual(meta["n_speakers"], 3)
        self.assertEqual(out.loc[out["speaker"] == "0001", "score_stratum"].iloc[0], "Low")
        self.assertEqual(out.loc[out["speaker"] == "0003", "score_stratum"].iloc[0], "High")


class Utt2SpkTests(unittest.TestCase):
    def test_parse_utt2spk_zero_pads(self):
        with tempfile.TemporaryDirectory() as tmp:
            split_dir = Path(tmp)
            (split_dir / "utt2spk").write_text("000010011 1\n000030012 0003\n", encoding="utf-8")
            mapping = load_split_utt2spk(split_dir)
            self.assertEqual(mapping["000010011"], "0001")
            self.assertEqual(mapping["000030012"], "0003")

    def test_speaker_overlap_empty_on_disjoint_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split, lines in (
                (
                    "train",
                    {
                        "utt2spk": "000010011 0001\n",
                        "spk2age": "0001\t6\n",
                        "spk2gender": "0001\tm\n",
                    },
                ),
                (
                    "test",
                    {
                        "utt2spk": "000030012 0003\n",
                        "spk2age": "0003\t12\n",
                        "spk2gender": "0003\tf\n",
                    },
                ),
            ):
                split_dir = root / split
                split_dir.mkdir()
                for name, text in lines.items():
                    (split_dir / name).write_text(text, encoding="utf-8")
            meta = load_speaker_metadata(root)
            self.assertEqual(meta["utt2spk"]["000010011"], "0001")
            self.assertEqual(meta["spk2age"]["0001"], 6)
            self.assertEqual(meta["speaker_overlap"], set())


if __name__ == "__main__":
    unittest.main()
