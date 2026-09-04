"""Unit tests for Group B GOP representations and train-only multivariate mapping."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gop_empirical.eval.metrics import (  # noqa: E402
    apply_linear_map_multi,
    error_metrics,
    fit_linear_score_map_multi,
)
from gop_empirical.gop.representation import (  # noqa: E402
    B5_N_FEATURES,
    SSL_LPP_LPR_N_FEATURES,
    canonical_lpp,
    gop_feature_vector,
    gop_feature_vector_ols,
    gopt_gop_feature_84,
    lpp_lpr_concat,
    lpp_max_competitor,
    lpr_vs_best_competitor,
    mean_lpp_on_frames,
    mean_lpp_on_span,
)
from gop_empirical.gop.traditional import split_lpp_lpr, traditional_gop_batch  # noqa: E402


def _make_feat(
    phone_id: int,
    canonical_lpp: float,
    competitor_id: int | None = None,
    competitor_lpp: float | None = None,
    n_phones: int = 42,
) -> np.ndarray:
    lpp = np.full(n_phones, -10.0, dtype=np.float64)
    lpp[phone_id] = canonical_lpp
    if competitor_id is not None and competitor_lpp is not None:
        lpp[competitor_id] = competitor_lpp
    lpr = canonical_lpp - lpp
    return np.concatenate([[float(phone_id)], lpp, lpr])


class RepresentationFormulaTests(unittest.TestCase):
    def test_b1_equals_b2_equals_canonical_lpp(self):
        feat = _make_feat(5, -0.2, competitor_id=8, competitor_lpp=-1.5)
        feats = feat.reshape(1, -1)
        b1 = canonical_lpp(feats)
        b2 = canonical_lpp(feats)
        np.testing.assert_allclose(b1, [-0.2])
        np.testing.assert_allclose(b1, b2)
        np.testing.assert_allclose(b1, traditional_gop_batch(feats))

    def test_b3_is_canonical_minus_max_other_lpp(self):
        feat = _make_feat(5, -0.2, competitor_id=8, competitor_lpp=-1.5)
        feats = feat.reshape(1, -1)
        b3 = lpr_vs_best_competitor(feats)
        self.assertAlmostEqual(float(b3[0]), -0.2 - (-1.5))

    def test_b3_matches_min_precomputed_lpr_excluding_canonical(self):
        feat = _make_feat(5, -0.2, competitor_id=8, competitor_lpp=-1.5)
        _phone_id, _lpp, lpr = split_lpp_lpr(feat)
        others = np.delete(lpr, 5)
        feats = feat.reshape(1, -1)
        b3 = lpr_vs_best_competitor(feats)
        self.assertAlmostEqual(float(b3[0]), float(others.min()))

    def test_competitor_wins_makes_b3_negative(self):
        feat = _make_feat(5, -0.8, competitor_id=9, competitor_lpp=-0.1)
        b3 = float(lpr_vs_best_competitor(feat.reshape(1, -1))[0])
        self.assertLess(b3, 0.0)
        self.assertAlmostEqual(b3, -0.8 - (-0.1))

    def test_vector_columns_and_rank_2_identity(self):
        feats = np.vstack(
            [
                _make_feat(3, -1.0, competitor_id=4, competitor_lpp=-0.4),
                _make_feat(10, -0.05, competitor_id=11, competitor_lpp=-2.0),
            ]
        )
        vec = gop_feature_vector(feats, expected_n_phones=42)
        self.assertEqual(vec.shape, (2, 3))
        np.testing.assert_allclose(vec[:, 0], canonical_lpp(feats))
        np.testing.assert_allclose(vec[:, 1], lpp_max_competitor(feats))
        np.testing.assert_allclose(vec[:, 2], vec[:, 0] - vec[:, 1])
        np.testing.assert_allclose(gop_feature_vector_ols(feats), vec[:, :2])

    def test_three_d_ols_matches_two_d_predictions(self):
        rng = np.random.default_rng(0)
        feats = np.vstack(
            [
                _make_feat(
                    5,
                    float(rng.normal()),
                    competitor_id=7,
                    competitor_lpp=float(rng.normal()),
                )
                for _ in range(80)
            ]
        )
        y = rng.normal(size=80)
        vec3 = gop_feature_vector(feats)
        vec2 = gop_feature_vector_ols(feats)
        map3 = fit_linear_score_map_multi(vec3, y)
        map2 = fit_linear_score_map_multi(vec2, y)
        pred3 = apply_linear_map_multi(vec3, map3, clip=None)
        pred2 = apply_linear_map_multi(vec2, map2, clip=None)
        np.testing.assert_allclose(pred3, pred2, rtol=1e-6, atol=1e-6)


class Gopt84FeatureTests(unittest.TestCase):
    def test_shape_and_concat_matches_split(self):
        feats = np.vstack(
            [
                _make_feat(5, -0.2, competitor_id=8, competitor_lpp=-1.5),
                _make_feat(10, -0.05, competitor_id=11, competitor_lpp=-2.0),
            ]
        )
        x84 = gopt_gop_feature_84(feats)
        self.assertEqual(x84.shape, (2, B5_N_FEATURES))
        for i in range(feats.shape[0]):
            _pid, lpp, lpr = split_lpp_lpr(feats[i])
            np.testing.assert_allclose(x84[i], np.concatenate([lpp, lpr]))

    def test_lpr_canonical_is_zero(self):
        feat = _make_feat(5, -0.2, competitor_id=8, competitor_lpp=-1.5)
        x84 = gopt_gop_feature_84(feat.reshape(1, -1))
        self.assertAlmostEqual(float(x84[0, 42 + 5]), 0.0, places=10)

    def test_wrong_n_phones_raises(self):
        feat = _make_feat(3, -1.0, n_phones=40)
        with self.assertRaises(ValueError):
            gopt_gop_feature_84(feat.reshape(1, -1), expected_n_phones=42)

    def test_b5_ols_no_leak_train_only(self):
        rng = np.random.default_rng(0)
        x_train = rng.normal(size=(200, B5_N_FEATURES))
        y_train = 0.01 * x_train[:, 0] - 0.02 * x_train[:, 41] + 1.2 + rng.normal(
            scale=0.05, size=200
        )
        x_test = rng.normal(size=(50, B5_N_FEATURES))
        y_test = np.full(50, 99.0)
        mapping = fit_linear_score_map_multi(x_train, y_train)
        self.assertEqual(mapping["n_features"], B5_N_FEATURES)
        pred_test = apply_linear_map_multi(x_test, mapping, clip=None)
        self.assertFalse(np.allclose(pred_test, y_test))
        self.assertGreater(error_metrics(pred_test, y_test)["mae"], 50.0)


class LppLprConcatTests(unittest.TestCase):
    def test_canonical_lpr_zero_and_dim_78(self):
        rng = np.random.default_rng(0)
        lpp = rng.normal(size=(5, 39))
        idx = np.array([0, 3, 10, 20, 38])
        out = lpp_lpr_concat(lpp, idx)
        self.assertEqual(out.shape, (5, SSL_LPP_LPR_N_FEATURES))
        np.testing.assert_allclose(out[:, :39], lpp)
        for i, can in enumerate(idx):
            self.assertAlmostEqual(float(out[i, 39 + can]), 0.0, places=10)
            np.testing.assert_allclose(out[i, 39:], lpp[i, can] - lpp[i])

    def test_mean_lpp_on_span(self):
        log_p = np.arange(12, dtype=np.float64).reshape(4, 3)
        mean = mean_lpp_on_span(log_p, 1, 3)
        np.testing.assert_allclose(mean, log_p[1:3].mean(axis=0))
        self.assertIsNone(mean_lpp_on_span(log_p, 2, 2))
        picked = mean_lpp_on_frames(log_p, np.array([0, 3]))
        np.testing.assert_allclose(picked, log_p[[0, 3]].mean(axis=0))
        self.assertIsNone(mean_lpp_on_frames(log_p, np.array([], dtype=np.int64)))


class NoLeakMultivariateTests(unittest.TestCase):
    def test_mapping_uses_train_only(self):
        rng = np.random.default_rng(0)
        x_train = rng.normal(size=(200, 2))
        y_train = 0.3 * x_train[:, 0] - 0.5 * x_train[:, 1] + 1.1 + rng.normal(scale=0.05, size=200)
        x_test = rng.normal(size=(80, 2))
        y_test = np.full(80, 99.0)

        mapping = fit_linear_score_map_multi(x_train, y_train)
        self.assertEqual(mapping["n_features"], 2)
        np.testing.assert_allclose(mapping["coef"], [0.3, -0.5], atol=0.05)
        self.assertAlmostEqual(mapping["intercept"], 1.1, places=1)

        pred_test = apply_linear_map_multi(x_test, mapping, clip=None)
        expected = x_test @ np.asarray(mapping["coef"]) + mapping["intercept"]
        np.testing.assert_allclose(pred_test, expected)
        self.assertFalse(np.allclose(pred_test, y_test))
        err = error_metrics(pred_test, y_test)
        self.assertGreater(err["mae"], 50.0)

    def test_clip_to_score_range(self):
        mapping = {"coef": [10.0, 0.0], "intercept": 0.0, "n_features": 2}
        x = np.array([[-1.0, 0.0], [0.1, 9.0], [5.0, -3.0]])
        pred = apply_linear_map_multi(x, mapping, clip=(0.0, 2.0))
        np.testing.assert_allclose(pred, [0.0, 1.0, 2.0])


class ProtocolLockAgainstGroupATests(unittest.TestCase):
    def test_b1_matches_a1_predictions_if_present(self):
        a1_path = ROOT / "outputs" / "A" / "a1_predictions.csv"
        if not a1_path.is_file():
            self.skipTest("Group A artifacts not present")
        a1 = pd.read_csv(a1_path, dtype={"utt_id": str})
        from gop_empirical.experiment import load_config, build_group_b_predictions

        cfg = load_config(ROOT / "configs" / "b_gop_representation.yaml")
        b_df, _stats, x_b5 = build_group_b_predictions(cfg, package_root=ROOT)
        self.assertEqual(x_b5.shape, (len(b_df), B5_N_FEATURES))
        b_join = b_df[["utt_id", "split", "word_id", "phone_id", "b1_gop"]].copy()
        b_join["utt_id"] = b_join["utt_id"].astype(str)
        merged = a1.merge(
            b_join,
            on=["utt_id", "split", "word_id", "phone_id"],
            how="inner",
        )
        self.assertGreater(len(merged), 1000)
        np.testing.assert_allclose(
            merged["gop"].to_numpy(), merged["b1_gop"].to_numpy(), rtol=1e-10, atol=1e-10
        )


if __name__ == "__main__":
    unittest.main()
