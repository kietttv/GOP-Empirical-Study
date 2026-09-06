"""Unit tests for Group F statistical validation and error taxonomy."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gop_empirical.data.scores_detail import (  # noqa: E402
    aggregate_expert_slots,
    parse_expert_phone_string,
)
from gop_empirical.eval.errors import assign_primary_type, classify_errors  # noqa: E402
from gop_empirical.eval.stats import (  # noqa: E402
    bootstrap_metric,
    paired_delta_bootstrap,
    pearson_pcc,
)


class ScoresDetailParseTests(unittest.TestCase):
    def test_readme_examples(self):
        events = parse_expert_phone_string("B (EH0) (R)")
        self.assertEqual([e["kind"] for e in events], ["correct", "incorrect", "incorrect"])
        self.assertEqual([e["phone"] for e in events], ["B", "EH", "R"])

        events = parse_expert_phone_string("B EH0 [L] R")
        self.assertEqual(
            [e["kind"] for e in events],
            ["correct", "correct", "insertion", "correct"],
        )

        events = parse_expert_phone_string("K {AO0} L")
        self.assertEqual([e["kind"] for e in events], ["correct", "accent", "correct"])
        self.assertEqual(events[1]["phone"], "AO")

    def test_aggregate_or_across_experts(self):
        slots = aggregate_expert_slots(
            ["B EH0 R", "B {EH0} R", "B (EH0) R"],
            n_canonical=3,
        )
        self.assertFalse(slots[0]["any_accent"])
        self.assertTrue(slots[1]["any_accent"])
        self.assertTrue(slots[1]["any_incorrect"])
        self.assertEqual(slots[1]["n_experts_accent"], 1)
        self.assertEqual(slots[1]["n_experts_incorrect"], 1)


class BootstrapTests(unittest.TestCase):
    def test_ci_contains_point(self):
        rng = np.random.default_rng(0)
        human = rng.normal(size=200)
        score = 0.6 * human + rng.normal(scale=0.3, size=200)
        out = bootstrap_metric(score, human, "pcc", n_boot=200, seed=0)
        self.assertTrue(out["ci_contains_point"])
        self.assertLessEqual(out["ci_low"], out["point"])
        self.assertGreaterEqual(out["ci_high"], out["point"])

    def test_paired_delta_zero_when_identical(self):
        rng = np.random.default_rng(1)
        human = rng.normal(size=150)
        score = 0.5 * human + rng.normal(scale=0.2, size=150)
        out = paired_delta_bootstrap(score, score, human, n_boot=100, seed=0)
        self.assertAlmostEqual(out["pcc"]["delta"], 0.0, places=12)
        self.assertFalse(out["pcc"]["ci_excludes_zero"])


class TaxonomyTests(unittest.TestCase):
    def test_accent_vs_incorrect_priority(self):
        self.assertEqual(
            assign_primary_type({"T2_confusion": True, "T3_accent": True}),
            "T2_confusion",
        )
        self.assertEqual(assign_primary_type({"T3_accent": True}), "T3_accent")

    def test_classify_confusion_and_accent(self):
        df = pd.DataFrame(
            {
                "human_score": [0.0, 1.0, 2.0],
                "score": [1.8, 1.8, 1.9],
                "any_incorrect": [True, False, False],
                "any_accent": [False, True, False],
                "any_insertion": [False, False, False],
                "competitor_wins": [False, False, False],
                "n_frames_c8": [10, 10, 10],
                "extreme_speaker": [False, False, False],
                "abs_err_c8": [1.0, 0.2, 0.1],
                "abs_err_e16": [0.2, 0.2, 0.1],
            }
        )
        # Classifying C8: T4 when C8 much worse than E16.
        out_c8 = classify_errors(
            df,
            pred_col="score",
            abs_err_self_col="abs_err_c8",
            abs_err_other_col="abs_err_e16",
            context_err_gap=0.5,
        )
        self.assertEqual(out_c8.loc[0, "primary_type"], "T2_confusion")
        self.assertEqual(out_c8.loc[1, "primary_type"], "T3_accent")
        self.assertTrue(bool(out_c8.loc[0, "T4_context"]))

        # Classifying E16: T4 only when E16 worse than C8 (none here).
        out_e16 = classify_errors(
            df,
            pred_col="score",
            abs_err_self_col="abs_err_e16",
            abs_err_other_col="abs_err_c8",
            context_err_gap=0.5,
        )
        self.assertFalse(bool(out_e16.loc[0, "T4_context"]))

    def test_leading_insertion_does_not_shift_canonical(self):
        slots = aggregate_expert_slots(["[S] (AH0)"], n_canonical=1)
        self.assertEqual(len(slots), 1)
        self.assertTrue(slots[0]["any_incorrect"])
        self.assertTrue(slots[0]["any_insertion"])
        self.assertEqual(slots[0]["phone"], "AH")


class SanityPccTests(unittest.TestCase):
    def test_positive_correlation(self):
        x = np.arange(10, dtype=np.float64)
        y = x + 0.1
        self.assertGreater(pearson_pcc(x, y), 0.99)


class GroupFLogTests(unittest.TestCase):
    def test_bootstrap_and_delta_lines(self):
        import io
        from contextlib import redirect_stdout

        from gop_empirical.experiment import (  # noqa: E402
            _fmt_num,
            _print_bootstrap,
            _print_delta,
            _print_type_counts,
        )

        self.assertEqual(_fmt_num(None, 4), "nan")
        self.assertEqual(_fmt_num(0.4981, 3), "0.498")

        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_bootstrap(
                "F1a C8",
                {
                    "n": 47369,
                    "pcc": {"point": 0.463, "ci_low": 0.453, "ci_high": 0.473},
                    "scc": {"point": 0.355},
                },
            )
            _print_delta(
                "F1b C8-C1",
                {
                    "pcc": {
                        "delta": 0.121,
                        "ci_low": 0.111,
                        "ci_high": 0.131,
                        "ci_excludes_zero": True,
                    }
                },
            )
            _print_type_counts(
                "F2 C8",
                {
                    "n": 10,
                    "mean_abs_err": 0.2,
                    "primary": {"T3_accent": 4, "T2_confusion": 2, "other": 4},
                },
            )
        text = buf.getvalue()
        self.assertIn("F1a C8  PCC=0.4630  95% CI [0.453, 0.473]", text)
        self.assertIn("F1b C8-C1  ΔPCC=+0.1210", text)
        self.assertIn("excludes_zero=True", text)
        self.assertIn("T3_accent=4", text)
        self.assertIn("mean|err|=0.200", text)


if __name__ == "__main__":
    unittest.main()
