"""Unit tests for Traditional GOP indexing and train-only score mapping."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gop_empirical.eval.metrics import (  # noqa: E402
    apply_linear_map,
    error_metrics,
    fit_linear_score_map,
)
from gop_empirical.gop.traditional import (  # noqa: E402
    split_lpp_lpr,
    traditional_gop,
    traditional_gop_batch,
)


def _make_feat(phone_id: int, canonical_lpp: float, n_phones: int = 42) -> np.ndarray:
    lpp = np.full(n_phones, -10.0, dtype=np.float64)
    lpp[phone_id] = canonical_lpp
    lpr = canonical_lpp - lpp
    return np.concatenate([[float(phone_id)], lpp, lpr])


class TraditionalGopIndexTests(unittest.TestCase):
    def test_split_layout(self):
        feat = _make_feat(5, -0.2)
        phone_id, lpp, lpr = split_lpp_lpr(feat)
        self.assertEqual(phone_id, 5)
        self.assertEqual(lpp.size, 42)
        self.assertEqual(lpr.size, 42)
        self.assertAlmostEqual(float(lpr[5]), 0.0)
        self.assertAlmostEqual(float(lpp[5]), -0.2)

    def test_canonical_lpp_is_gop(self):
        feat = _make_feat(5, -0.2)
        self.assertAlmostEqual(traditional_gop(feat, phone_index_base=0), -0.2)
        # A competitor slot must not be returned.
        self.assertNotAlmostEqual(traditional_gop(feat), -10.0)

    def test_one_based_index(self):
        n = 42
        phone_id = 6  # 1-based id for slot 5
        lpp = np.full(n, -10.0)
        lpp[5] = -0.4
        lpr = np.zeros(n)
        feat = np.concatenate([[float(phone_id)], lpp, lpr])
        self.assertAlmostEqual(traditional_gop(feat, phone_index_base=1), -0.4)

    def test_out_of_range_raises(self):
        feat = _make_feat(5, -0.2)
        feat[0] = 99
        with self.assertRaises(IndexError):
            traditional_gop(feat)

    def test_batch_matches_scalar(self):
        feats = np.vstack([_make_feat(3, -1.0), _make_feat(10, -0.05)])
        got = traditional_gop_batch(feats, expected_n_phones=42)
        np.testing.assert_allclose(got, [-1.0, -0.05])


class NoLeakMappingTests(unittest.TestCase):
    def test_mapping_uses_train_only(self):
        rng = np.random.default_rng(0)
        gop_train = rng.normal(size=200)
        y_train = 0.4 * gop_train + 1.2 + rng.normal(scale=0.05, size=200)
        gop_test = rng.normal(size=80)
        y_test = np.full(80, 99.0)  # must not affect the fit

        mapping = fit_linear_score_map(gop_train, y_train)
        self.assertAlmostEqual(mapping["slope"], 0.4, places=1)
        self.assertAlmostEqual(mapping["intercept"], 1.2, places=1)

        pred_test = apply_linear_map(gop_test, mapping, clip=None)
        # Predictions are a function of test GOP + train mapping, never y_test.
        np.testing.assert_allclose(
            pred_test, mapping["slope"] * gop_test + mapping["intercept"]
        )
        self.assertFalse(np.allclose(pred_test, y_test))

        # Error metrics see y_test only at evaluation time.
        err = error_metrics(pred_test, y_test)
        self.assertGreater(err["mae"], 50.0)

    def test_clip_to_score_range(self):
        mapping = {"slope": 10.0, "intercept": 0.0}
        pred = apply_linear_map(np.array([-1.0, 0.1, 5.0]), mapping, clip=(0.0, 2.0))
        np.testing.assert_allclose(pred, [0.0, 1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
