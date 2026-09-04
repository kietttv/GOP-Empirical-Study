"""Unit tests for Group E learned scoring: split, scaler, dims, pad mask."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gop_empirical.data.learned import (  # noqa: E402
    CANONICAL_PHONE_COL,
    FeatureScaler,
    assign_roles,
    attach_canonical_phone_index,
    choose_val_speakers,
    feature_stored_columns,
    keep_first_phones,
    load_feature_table,
    load_ssl_lpp_lpr_feature_table,
    matrix_from_table,
    normalize_group_e_features,
    pack_utterances,
    phone_embed_spec,
    scoring_ids,
    scoring_pred_columns,
    uses_phone_embed,
)
from gop_empirical.data.ssl_lpp_lpr import write_ssl_lpp_lpr_split  # noqa: E402
from gop_empirical.eval.metrics import evaluate_predictions  # noqa: E402
from gop_empirical.experiment import merge_group_e_results  # noqa: E402


def _phones(
    *,
    n_train_spk: int = 10,
    n_test_spk: int = 5,
    phones_per_utt: int = 4,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split, n_spk, spk_start in (("train", n_train_spk, 1), ("test", n_test_spk, 101)):
        for s in range(n_spk):
            speaker = f"{spk_start + s:04d}"
            utt = f"{spk_start + s:09d}"
            for p in range(phones_per_utt):
                lpp = 4.0 + 0.1 * p + 0.01 * s
                comp = lpp + 0.2
                rows.append(
                    {
                        "utt_id": utt,
                        "split": split,
                        "word_id": 0,
                        "phone_id": p,
                        "phone": "AH",
                        "speaker": speaker,
                        "human_score": 1.0 + 0.2 * (p % 3),
                        "feat_lpp_canonical": lpp,
                        "feat_lpp_max_competitor": comp,
                        "feat_lpr": lpp - comp,
                        "feat_gop": lpp,
                    }
                )
    return pd.DataFrame(rows)


class FeatureFlagTests(unittest.TestCase):
    def test_normalize_default_and_order(self):
        self.assertEqual(normalize_group_e_features(None), ["b4"])
        self.assertEqual(normalize_group_e_features("a1"), ["a1"])
        self.assertEqual(normalize_group_e_features(["a1", "b4", "a1"]), ["a1", "b4"])
        self.assertEqual(normalize_group_e_features(["c8", "c9", "c8"]), ["c8", "c9"])
        self.assertEqual(normalize_group_e_features(["c10", "c11", "c10"]), ["c10", "c11"])
        self.assertEqual(
            normalize_group_e_features(["b5", "b5_embed", "c8_lpp_lpr", "c8_lpp_lpr_embed"]),
            ["b5", "b5_embed", "c8_lpp_lpr", "c8_lpp_lpr_embed"],
        )

    def test_unknown_feature_raises(self):
        with self.assertRaises(ValueError):
            normalize_group_e_features(["gopt84"])

    def test_dims(self):
        self.assertEqual(len(feature_stored_columns("b4")), 3)
        self.assertEqual(len(feature_stored_columns("a1")), 1)
        self.assertEqual(len(feature_stored_columns("c8")), 1)
        self.assertEqual(len(feature_stored_columns("c9")), 1)
        self.assertEqual(len(feature_stored_columns("b5")), 84)
        self.assertEqual(len(feature_stored_columns("b5_embed")), 84)
        self.assertEqual(len(feature_stored_columns("c8_lpp_lpr")), 78)
        self.assertEqual(len(feature_stored_columns("c9_lpp_lpr")), 78)
        self.assertEqual(len(feature_stored_columns("c8_lpp_lpr_embed")), 78)
        self.assertEqual(len(feature_stored_columns("c9_lpp_lpr_embed")), 78)

    def test_scoring_ids(self):
        self.assertEqual(scoring_ids("b4"), ("E1", "E2"))
        self.assertEqual(scoring_ids("c8"), ("E3", "E4"))
        self.assertEqual(scoring_ids("c9"), ("E5", "E6"))
        self.assertEqual(scoring_ids("c10"), ("E19", "E20"))
        self.assertEqual(scoring_ids("c11"), ("E21", "E22"))
        self.assertEqual(scoring_pred_columns("c8"), ("pred_e3", "pred_e4"))
        self.assertEqual(scoring_pred_columns("c9"), ("pred_e5", "pred_e6"))
        self.assertEqual(scoring_pred_columns("c10"), ("pred_e19", "pred_e20"))
        self.assertEqual(scoring_pred_columns("c11"), ("pred_e21", "pred_e22"))
        self.assertEqual(scoring_ids("b5"), ("E7", "E8"))
        self.assertEqual(scoring_ids("b5_embed"), ("E13", "E14"))
        self.assertEqual(scoring_ids("c8_lpp_lpr"), ("E9", "E10"))
        self.assertEqual(scoring_ids("c9_lpp_lpr"), ("E11", "E12"))
        self.assertEqual(scoring_ids("c8_lpp_lpr_embed"), ("E15", "E16"))
        self.assertEqual(scoring_ids("c9_lpp_lpr_embed"), ("E17", "E18"))
        self.assertEqual(scoring_pred_columns("b5"), ("pred_e7", "pred_e8"))
        self.assertEqual(scoring_pred_columns("b5_embed"), ("pred_e13", "pred_e14"))
        self.assertEqual(scoring_pred_columns("c8_lpp_lpr_embed"), ("pred_e15", "pred_e16"))
        self.assertEqual(scoring_pred_columns("c9_lpp_lpr_embed"), ("pred_e17", "pred_e18"))
        self.assertFalse(uses_phone_embed("c8_lpp_lpr"))
        self.assertTrue(uses_phone_embed("c8_lpp_lpr_embed"))
        self.assertEqual(phone_embed_spec("b5_embed")["space"], "kaldi")
        self.assertEqual(phone_embed_spec("c8_lpp_lpr_embed")["n_phones"], 39)
        self.assertEqual(phone_embed_spec("c8_lpp_lpr_embed")["space"], "ssl")
        self.assertEqual(phone_embed_spec("c9_lpp_lpr_embed")["space"], "ssl")

    def test_canonical_phone_index_uses_kaldi_symbol(self):
        df = pd.DataFrame(
            {
                "utt_id": ["000010011", "000010011"],
                "split": ["train", "train"],
                "word_id": [0, 1],
                "phone_id": [0, 0],
                "phone": ["AH", "T"],
                "human_score": [2.0, 1.0],
            }
        )
        out = attach_canonical_phone_index(df)
        self.assertEqual(out[CANONICAL_PHONE_COL].tolist(), [5, 33])
        # word-level phone_id is not the Kaldi slot
        self.assertEqual(out["phone_id"].tolist(), [0, 0])
        from gop_empirical.acoustic.phones import load_phone_inventory

        inv = load_phone_inventory()
        ssl = attach_canonical_phone_index(df, n_phones=39, space="ssl")
        self.assertEqual(
            ssl[CANONICAL_PHONE_COL].tolist(),
            [inv.ssl_index("AH"), inv.ssl_index("T")],
        )
        self.assertNotEqual(inv.kaldi_symbol_to_id["AH"], inv.ssl_index("AH"))


class SplitLeakTests(unittest.TestCase):
    def test_val_speakers_subset_of_train_disjoint_from_test(self):
        df = _phones()
        train_spk = set(df.loc[df["split"] == "train", "speaker"])
        test_spk = set(df.loc[df["split"] == "test", "speaker"])
        val = choose_val_speakers(sorted(train_spk), frac=0.2, seed=0)
        self.assertTrue(set(val).issubset(train_spk))
        self.assertTrue(set(val).isdisjoint(test_spk))
        self.assertEqual(val, sorted(val))
        assigned = assign_roles(
            df, val_speakers=val, train_speakers=train_spk, test_speakers=test_spk
        )
        self.assertEqual(set(assigned.loc[assigned["role"] == "test", "speaker"]), test_spk)
        self.assertEqual(set(assigned.loc[assigned["role"] == "val", "speaker"]), set(val))
        fit = set(assigned.loc[assigned["role"] == "train", "speaker"])
        self.assertTrue(fit.isdisjoint(set(val)))
        self.assertTrue(fit.isdisjoint(test_spk))
        self.assertEqual(set(assigned.loc[assigned["split"] == "test", "role"]), {"test"})

    def test_val_from_test_speakers_raises(self):
        df = _phones()
        train_spk = set(df.loc[df["split"] == "train", "speaker"])
        test_spk = set(df.loc[df["split"] == "test", "speaker"])
        with self.assertRaises(ValueError):
            assign_roles(
                df,
                val_speakers=sorted(test_spk)[:1],
                train_speakers=train_spk,
                test_speakers=test_spk,
            )


class ScalerLeakTests(unittest.TestCase):
    def test_scaler_fits_train_only(self):
        df = _phones()
        train_spk = set(df.loc[df["split"] == "train", "speaker"])
        test_spk = set(df.loc[df["split"] == "test", "speaker"])
        val = choose_val_speakers(sorted(train_spk), frac=0.2, seed=0)
        df = assign_roles(
            df, val_speakers=val, train_speakers=train_spk, test_speakers=test_spk
        )
        cols = list(feature_stored_columns("b4"))
        train_x = matrix_from_table(df[df["role"] == "train"], cols)
        test_x = matrix_from_table(df[df["role"] == "test"], cols)
        scaler = FeatureScaler.fit(train_x)
        z_train = scaler.transform(train_x)
        np.testing.assert_allclose(z_train.mean(axis=0), 0.0, atol=1e-10)
        leak_scaler = FeatureScaler.fit(test_x)
        self.assertFalse(np.allclose(scaler.mean, leak_scaler.mean))
        z_test = scaler.transform(test_x)
        self.assertFalse(np.allclose(z_test.mean(axis=0), 0.0, atol=1e-3))

    def test_a1_is_one_column(self):
        df = _phones()
        x = matrix_from_table(df, feature_stored_columns("a1"))
        self.assertEqual(x.shape[1], 1)


class CGopLoadTests(unittest.TestCase):
    def test_load_c8_from_csv(self):
        import tempfile

        rows = (
            "utt_id,split,word_id,phone_id,phone,human_score,gop_c8,gop_c9\n"
            "000010011,train,0,0,AH,2.0,-0.5,-0.4\n"
            "000010011,train,0,1,T,1.0,,-0.1\n"
            "000010011,test,0,0,AH,1.8,-1.2,-0.9\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c_predictions.csv"
            path.write_text(rows, encoding="utf-8")
            table = load_feature_table(path, "c8")
        self.assertEqual(list(table.columns), [
            "utt_id",
            "split",
            "word_id",
            "phone_id",
            "phone",
            "human_score",
            "feat_gop_c8",
        ])
        self.assertEqual(len(table), 2)
        self.assertAlmostEqual(float(table.loc[0, "feat_gop_c8"]), -0.5)
        self.assertEqual(table["utt_id"].tolist(), ["000010011", "000010011"])

    def test_load_c10_from_csv(self):
        import tempfile

        rows = (
            "utt_id,split,word_id,phone_id,phone,human_score,gop_c10,gop_c11\n"
            "000010011,train,0,0,AH,2.0,-0.7,-0.6\n"
            "000010011,train,0,1,T,1.0,,-0.2\n"
            "000010011,test,0,0,AH,1.8,-1.4,-1.1\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c_predictions.csv"
            path.write_text(rows, encoding="utf-8")
            table = load_feature_table(path, "c10")
        self.assertEqual(list(table.columns), [
            "utt_id",
            "split",
            "word_id",
            "phone_id",
            "phone",
            "human_score",
            "feat_gop_c10",
        ])
        self.assertEqual(len(table), 2)
        self.assertAlmostEqual(float(table.loc[0, "feat_gop_c10"]), -0.7)


class MergeResultsTests(unittest.TestCase):
    def test_c8_run_keeps_locked_e1_e2(self):
        existing = {
            "group": "E",
            "experiments": ["E1", "E2"],
            "feature_set": "b4",
            "protocol": {
                "acoustic_model": "kaldi_librispeech_m13",
                "gop_type": "b4_gop_vector",
                "features_requested": ["b4"],
                "predictions_path": "outputs/E/e_predictions.csv",
                "prediction_paths": {"b4": "outputs/E/e_predictions.csv"},
            },
            "comparison": {"b4": {"E1": {"pcc": 0.36}, "E2": {"pcc": 0.51}}},
            "b4": {"E1": {"pcc": 0.36}, "E2": {"pcc": 0.51}},
            "E1": {"pcc": 0.36},
            "E2": {"pcc": 0.51},
        }
        new = {
            "group": "E",
            "experiments": ["E3", "E4"],
            "feature_set": "c8",
            "protocol": {
                "acoustic_model": "wav2vec2_xlsr53_espeak_ctc",
                "gop_type": "cao_gop_s",
                "features_requested": ["c8"],
                "predictions_path": None,
                "prediction_paths": {"c8": "outputs/E/e_c8_predictions.csv"},
                "acoustic_model_by_features": {"c8": "wav2vec2_xlsr53_espeak_ctc"},
                "gop_type_by_features": {"c8": "cao_gop_s"},
            },
            "comparison": {"c8": {"E3": {"pcc": 0.48}, "E4": {"pcc": 0.55}}},
            "c8": {"E3": {"pcc": 0.48}, "E4": {"pcc": 0.55}},
            "E3": {"pcc": 0.48},
            "E4": {"pcc": 0.55},
        }
        merged = merge_group_e_results(existing, new)
        self.assertEqual(merged["experiments"], ["E1", "E2", "E3", "E4"])
        self.assertEqual(merged["feature_set"], "b4")
        self.assertEqual(merged["E1"]["pcc"], 0.36)
        self.assertEqual(merged["E3"]["pcc"], 0.48)
        self.assertEqual(merged["protocol"]["acoustic_model"], "kaldi_librispeech_m13")
        self.assertEqual(merged["protocol"]["predictions_path"], "outputs/E/e_predictions.csv")
        self.assertEqual(
            merged["protocol"]["prediction_paths"]["c8"], "outputs/E/e_c8_predictions.csv"
        )
        self.assertEqual(merged["protocol"]["features_requested"], ["c8"])
        self.assertIn("c8", merged["comparison"])
        self.assertIn("b4", merged["comparison"])

    def test_c10_run_keeps_locked_e3(self):
        existing = {
            "group": "E",
            "experiments": ["E1", "E2", "E3", "E4"],
            "feature_set": "b4",
            "protocol": {
                "acoustic_model": "kaldi_librispeech_m13",
                "gop_type": "b4_gop_vector",
                "features_requested": ["c8"],
                "predictions_path": "outputs/E/e_predictions.csv",
                "prediction_paths": {"c8": "outputs/E/e_c8_predictions.csv"},
            },
            "comparison": {"c8": {"E3": {"pcc": 0.48}, "E4": {"pcc": 0.55}}},
            "c8": {"E3": {"pcc": 0.48}, "E4": {"pcc": 0.55}},
            "E3": {"pcc": 0.48},
            "E4": {"pcc": 0.55},
        }
        new = {
            "group": "E",
            "experiments": ["E19", "E20"],
            "feature_set": "c10",
            "protocol": {
                "acoustic_model": "wav2vec2_xlsr53_espeak_ctc",
                "gop_type": "cao_gop_sd",
                "features_requested": ["c10"],
                "prediction_paths": {"c10": "outputs/E/e_c10_predictions.csv"},
                "acoustic_model_by_features": {"c10": "wav2vec2_xlsr53_espeak_ctc"},
                "gop_type_by_features": {"c10": "cao_gop_sd"},
            },
            "comparison": {"c10": {"E19": {"pcc": 0.51}, "E20": {"pcc": 0.58}}},
            "c10": {"E19": {"pcc": 0.51}, "E20": {"pcc": 0.58}},
            "E19": {"pcc": 0.51},
            "E20": {"pcc": 0.58},
        }
        merged = merge_group_e_results(existing, new)
        self.assertEqual(merged["E3"]["pcc"], 0.48)
        self.assertEqual(merged["E19"]["pcc"], 0.51)
        self.assertIn("E3", merged["experiments"])
        self.assertIn("E19", merged["experiments"])
        self.assertEqual(merged["c8"]["E3"]["pcc"], 0.48)
        self.assertEqual(merged["c10"]["E19"]["pcc"], 0.51)


class SslLppLprLoadTests(unittest.TestCase):
    def test_join_78d_from_npz(self):
        import tempfile

        rng = np.random.default_rng(0)
        rows_c = (
            "utt_id,split,word_id,phone_id,phone,human_score,gop_c8\n"
            "000010011,train,0,0,AH,2.0,-0.5\n"
            "000010011,train,0,1,T,1.0,-0.2\n"
            "000010011,train,1,0,IY,1.5,-0.8\n"
            "000010011,test,0,0,AH,1.8,-1.2\n"
        )
        feat0 = rng.normal(size=78)
        feat1 = rng.normal(size=78)
        feat2 = rng.normal(size=78)
        with tempfile.TemporaryDirectory() as tmp:
            c_path = Path(tmp) / "c_predictions.csv"
            c_path.write_text(rows_c, encoding="utf-8")
            npz_dir = Path(tmp) / "lpp_lpr"
            feat3 = rng.normal(size=78)
            write_ssl_lpp_lpr_split(
                [
                    {"key": "000010011.0", "phone": "AH", "n_frames": 4, "features": feat0},
                    {"key": "000010011.1", "phone": "T", "n_frames": 3, "features": feat1},
                    {"key": "000010011.2", "phone": "IY", "n_frames": 2, "features": feat3},
                ],
                npz_dir,
                "train",
            )
            write_ssl_lpp_lpr_split(
                [
                    {"key": "000010011.0", "phone": "AH", "n_frames": 5, "features": feat2},
                ],
                npz_dir,
                "test",
            )
            table = load_ssl_lpp_lpr_feature_table(c_path, npz_dir)
        self.assertEqual(len(table), 4)
        self.assertEqual(len(feature_stored_columns("c8_lpp_lpr")), 78)
        cols = feature_stored_columns("c8_lpp_lpr")
        np.testing.assert_allclose(table.loc[0, cols].to_numpy(dtype=np.float64), feat0)
        # word 1 phone_id=0 must join utt.2, not collide with word 0 phone_id=0
        np.testing.assert_allclose(table.loc[2, cols].to_numpy(dtype=np.float64), feat3)

    def test_b5_run_keeps_locked_e1_e2(self):
        existing = {
            "group": "E",
            "experiments": ["E1", "E2"],
            "feature_set": "b4",
            "protocol": {
                "acoustic_model": "kaldi_librispeech_m13",
                "gop_type": "b4_gop_vector",
                "features_requested": ["b4"],
                "predictions_path": "outputs/E/e_predictions.csv",
                "prediction_paths": {"b4": "outputs/E/e_predictions.csv"},
            },
            "comparison": {"b4": {"E1": {"pcc": 0.36}, "E2": {"pcc": 0.51}}},
            "b4": {"E1": {"pcc": 0.36}, "E2": {"pcc": 0.51}},
            "E1": {"pcc": 0.36},
            "E2": {"pcc": 0.51},
        }
        new = {
            "group": "E",
            "experiments": ["E7", "E8"],
            "feature_set": "b5",
            "protocol": {
                "acoustic_model": "kaldi_librispeech_m13",
                "gop_type": "gopt_gop_feature_84",
                "features_requested": ["b5"],
                "predictions_path": None,
                "prediction_paths": {"b5": "outputs/E/e_b5_predictions.csv"},
                "acoustic_model_by_features": {"b5": "kaldi_librispeech_m13"},
                "gop_type_by_features": {"b5": "gopt_gop_feature_84"},
            },
            "comparison": {"b5": {"E7": {"pcc": 0.40}, "E8": {"pcc": 0.52}}},
            "b5": {"E7": {"pcc": 0.40}, "E8": {"pcc": 0.52}},
            "E7": {"pcc": 0.40},
            "E8": {"pcc": 0.52},
        }
        merged = merge_group_e_results(existing, new)
        self.assertEqual(merged["experiments"], ["E1", "E2", "E7", "E8"])
        self.assertEqual(merged["feature_set"], "b4")
        self.assertEqual(merged["E1"]["pcc"], 0.36)
        self.assertEqual(merged["E7"]["pcc"], 0.40)
        self.assertEqual(merged["protocol"]["predictions_path"], "outputs/E/e_predictions.csv")
        self.assertEqual(merged["protocol"]["features_requested"], ["b5"])

    def test_b5_embed_run_keeps_locked_e7_e8(self):
        existing = {
            "group": "E",
            "experiments": ["E7", "E8"],
            "feature_set": "b4",
            "protocol": {
                "acoustic_model": "kaldi_librispeech_m13",
                "gop_type": "b4_gop_vector",
                "features_requested": ["b5"],
                "predictions_path": "outputs/E/e_predictions.csv",
                "prediction_paths": {"b5": "outputs/E/e_b5_predictions.csv"},
            },
            "comparison": {"b5": {"E7": {"pcc": 0.45}, "E8": {"pcc": 0.53}}},
            "b5": {"E7": {"pcc": 0.45}, "E8": {"pcc": 0.53}},
            "E7": {"pcc": 0.45},
            "E8": {"pcc": 0.53},
        }
        new = {
            "group": "E",
            "experiments": ["E13", "E14"],
            "feature_set": "b5_embed",
            "protocol": {
                "acoustic_model": "kaldi_librispeech_m13",
                "gop_type": "gopt_gop_feature_84_phone_embed",
                "features_requested": ["b5_embed"],
                "predictions_path": None,
                "prediction_paths": {"b5_embed": "outputs/E/e_b5_predictions.csv"},
                "acoustic_model_by_features": {"b5_embed": "kaldi_librispeech_m13"},
                "gop_type_by_features": {"b5_embed": "gopt_gop_feature_84_phone_embed"},
            },
            "comparison": {"b5_embed": {"E13": {"pcc": 0.48}, "E14": {"pcc": 0.55}}},
            "b5_embed": {"E13": {"pcc": 0.48}, "E14": {"pcc": 0.55}},
            "E13": {"pcc": 0.48},
            "E14": {"pcc": 0.55},
        }
        merged = merge_group_e_results(existing, new)
        self.assertEqual(merged["experiments"], ["E7", "E8", "E13", "E14"])
        self.assertEqual(merged["E7"]["pcc"], 0.45)
        self.assertEqual(merged["E8"]["pcc"], 0.53)
        self.assertEqual(merged["E13"]["pcc"], 0.48)
        self.assertEqual(merged["E14"]["pcc"], 0.55)
        self.assertEqual(merged["b5"]["E7"]["pcc"], 0.45)
        self.assertEqual(merged["protocol"]["gop_type"], "b4_gop_vector")
        self.assertEqual(merged["protocol"]["features_requested"], ["b5_embed"])

    def test_ssl_embed_run_keeps_locked_e9_e12(self):
        existing = {
            "group": "E",
            "experiments": ["E9", "E10", "E11", "E12"],
            "feature_set": "b4",
            "protocol": {
                "acoustic_model": "kaldi_librispeech_m13",
                "gop_type": "b4_gop_vector",
                "features_requested": ["c8_lpp_lpr"],
                "predictions_path": "outputs/E/e_predictions.csv",
                "prediction_paths": {"c8_lpp_lpr": "outputs/E/e_c8_lpp_lpr_predictions.csv"},
            },
            "comparison": {
                "c8_lpp_lpr": {"E9": {"pcc": 0.25}, "E10": {"pcc": 0.29}},
                "c9_lpp_lpr": {"E11": {"pcc": 0.20}, "E12": {"pcc": 0.28}},
            },
            "c8_lpp_lpr": {"E9": {"pcc": 0.25}, "E10": {"pcc": 0.29}},
            "c9_lpp_lpr": {"E11": {"pcc": 0.20}, "E12": {"pcc": 0.28}},
            "E9": {"pcc": 0.25},
            "E10": {"pcc": 0.29},
            "E11": {"pcc": 0.20},
            "E12": {"pcc": 0.28},
        }
        new = {
            "group": "E",
            "experiments": ["E15", "E16", "E17", "E18"],
            "feature_set": "c8_lpp_lpr_embed",
            "protocol": {
                "acoustic_model": "wav2vec2_xlsr53_espeak_ctc",
                "gop_type": "gopt_style_lpp_lpr_78_phone_embed",
                "features_requested": ["c8_lpp_lpr_embed", "c9_lpp_lpr_embed"],
                "predictions_path": None,
                "prediction_paths": {
                    "c8_lpp_lpr_embed": "outputs/E/e_c8_lpp_lpr_predictions.csv",
                    "c9_lpp_lpr_embed": "outputs/E/e_c9_lpp_lpr_predictions.csv",
                },
            },
            "comparison": {
                "c8_lpp_lpr_embed": {"E15": {"pcc": 0.31}, "E16": {"pcc": 0.35}},
                "c9_lpp_lpr_embed": {"E17": {"pcc": 0.26}, "E18": {"pcc": 0.33}},
            },
            "c8_lpp_lpr_embed": {"E15": {"pcc": 0.31}, "E16": {"pcc": 0.35}},
            "c9_lpp_lpr_embed": {"E17": {"pcc": 0.26}, "E18": {"pcc": 0.33}},
            "E15": {"pcc": 0.31},
            "E16": {"pcc": 0.35},
            "E17": {"pcc": 0.26},
            "E18": {"pcc": 0.33},
        }
        merged = merge_group_e_results(existing, new)
        self.assertEqual(merged["experiments"], ["E9", "E10", "E11", "E12", "E15", "E16", "E17", "E18"])
        self.assertEqual(merged["E9"]["pcc"], 0.25)
        self.assertEqual(merged["E12"]["pcc"], 0.28)
        self.assertEqual(merged["E16"]["pcc"], 0.35)
        self.assertEqual(merged["E18"]["pcc"], 0.33)
        self.assertEqual(merged["protocol"]["gop_type"], "b4_gop_vector")


class PackMaskTests(unittest.TestCase):
    def test_pad_mask_true_on_pad_false_on_phones(self):
        df = _phones(n_train_spk=2, n_test_spk=1, phones_per_utt=3)
        packed = pack_utterances(df, feature_stored_columns("b4"), max_seq_len=5)
        self.assertEqual(packed["x"].shape, (3, 5, 3))
        for i, n in enumerate(packed["lengths"]):
            self.assertFalse(packed["pad_mask"][i, :n].any())
            self.assertTrue(packed["pad_mask"][i, n:].all())

    def test_keep_first_phones_drops_tail(self):
        df = _phones(n_train_spk=1, n_test_spk=0, phones_per_utt=6)
        kept, n_drop = keep_first_phones(df, max_seq_len=4)
        self.assertEqual(n_drop, 2)
        self.assertEqual(len(kept), 4)

    def test_e1_e2_same_x_rows(self):
        df = _phones(n_train_spk=2, n_test_spk=1, phones_per_utt=3)
        cols = list(feature_stored_columns("b4"))
        x_flat = matrix_from_table(
            df.sort_values(["utt_id", "word_id", "phone_id"]), cols
        )
        packed = pack_utterances(df, cols, max_seq_len=3)
        rebuilt = []
        for i, idx in enumerate(packed["row_indices"]):
            n = int(packed["lengths"][i])
            rebuilt.append(packed["x"][i, :n, :])
        stacked = np.concatenate(rebuilt, axis=0)
        order = np.concatenate(packed["row_indices"])
        np.testing.assert_allclose(stacked, x_flat[order])


class EvaluatePredictionsTests(unittest.TestCase):
    def test_clip_and_metrics(self):
        human = np.array([0.0, 1.0, 2.0, 1.0])
        pred = np.array([-1.0, 1.0, 3.0, 1.5])
        out = evaluate_predictions(pred, human, clip=(0.0, 2.0))
        self.assertEqual(out["n"], 4)
        self.assertGreater(out["pcc"], 0.9)
        self.assertAlmostEqual(out["mae"], 0.125)


class TorchScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("PyTorch is not installed") from None

    def test_masked_mse_ignores_pad(self):
        import torch

        from gop_empirical.scoring.train import masked_mse

        pred = torch.tensor([[1.0, 2.0, 99.0]])
        target = torch.tensor([[1.0, 3.0, 0.0]])
        pad = torch.tensor([[False, False, True]])
        loss = masked_mse(pred, target, pad)
        self.assertAlmostEqual(float(loss), 0.5)

    def test_mlp_and_transformer_input_dim(self):
        import torch

        from gop_empirical.scoring.mlp import PhoneMLP
        from gop_empirical.scoring.transformer import PhoneTransformer

        mlp3 = PhoneMLP(3, hidden_dim=8)
        mlp1 = PhoneMLP(1, hidden_dim=8)
        self.assertEqual(mlp3(torch.zeros(4, 3)).shape, (4,))
        self.assertEqual(mlp1(torch.zeros(4, 1)).shape, (4,))
        with self.assertRaises(RuntimeError):
            mlp3(torch.zeros(4, 1))

        mask = torch.zeros(2, 5, dtype=torch.bool)
        mask[:, 3:] = True
        tf = PhoneTransformer(input_dim=3, d_model=8, nhead=2, nlayers=1, dim_feedforward=16, max_len=5)
        self.assertEqual(tf(torch.zeros(2, 5, 3), mask).shape, (2, 5))
        tf84 = PhoneTransformer(input_dim=84, d_model=8, nhead=2, nlayers=1, dim_feedforward=16, max_len=5)
        self.assertEqual(tf84(torch.zeros(2, 5, 84), mask).shape, (2, 5))
        mlp78 = PhoneMLP(78, hidden_dim=8)
        self.assertEqual(mlp78(torch.zeros(4, 78)).shape, (4,))
        mlp78_emb = PhoneMLP(78, hidden_dim=8, n_phones=39)
        self.assertEqual(mlp78_emb(torch.zeros(4, 78), torch.tensor([2, 5, 10, 30])).shape, (4,))

        ids = torch.tensor([3, 5, 7, 9])
        mlp_emb = PhoneMLP(84, hidden_dim=8, n_phones=42)
        self.assertEqual(mlp_emb(torch.zeros(4, 84), ids).shape, (4,))
        with self.assertRaises(ValueError):
            mlp_emb(torch.zeros(4, 84))
        phn = torch.full((2, 5), 42, dtype=torch.long)
        phn[:, :3] = torch.tensor([[3, 4, 5], [6, 7, 8]])
        tf_emb = PhoneTransformer(
            input_dim=84, d_model=8, nhead=2, nlayers=1, dim_feedforward=16, max_len=5, n_phones=42
        )
        self.assertEqual(tf_emb(torch.zeros(2, 5, 84), mask, phn).shape, (2, 5))
        phn39 = torch.full((2, 5), 39, dtype=torch.long)
        phn39[:, :3] = torch.tensor([[2, 4, 5], [6, 7, 8]])
        tf78_emb = PhoneTransformer(
            input_dim=78, d_model=8, nhead=2, nlayers=1, dim_feedforward=16, max_len=5, n_phones=39
        )
        self.assertEqual(tf78_emb(torch.zeros(2, 5, 78), mask, phn39).shape, (2, 5))

    def test_pack_phone_ids_pad_slot(self):
        from gop_empirical.data.learned import CANONICAL_PHONE_COL

        df = _phones(n_train_spk=2, n_test_spk=1, phones_per_utt=3)
        df[CANONICAL_PHONE_COL] = (df["phone_id"] + 3).astype(np.int64)
        packed = pack_utterances(
            df,
            feature_stored_columns("b4"),
            max_seq_len=5,
            phone_idx_col=CANONICAL_PHONE_COL,
            pad_phone_id=42,
        )
        self.assertEqual(packed["phone_ids"].shape, (3, 5))
        for i, n in enumerate(packed["lengths"]):
            self.assertTrue((packed["phone_ids"][i, n:] == 42).all())
            self.assertTrue((packed["phone_ids"][i, :n] != 42).all())
        packed39 = pack_utterances(
            df,
            feature_stored_columns("b4"),
            max_seq_len=5,
            phone_idx_col=CANONICAL_PHONE_COL,
            pad_phone_id=39,
        )
        for i, n in enumerate(packed39["lengths"]):
            self.assertTrue((packed39["phone_ids"][i, n:] == 39).all())
            self.assertTrue((packed39["phone_ids"][i, :n] != 39).all())

    def test_tiny_train_does_not_use_test_loader(self):
        import torch

        from gop_empirical.scoring.mlp import PhoneMLP
        from gop_empirical.scoring.train import phone_loader, predict_mlp, train_regressor

        rng = np.random.default_rng(0)
        x_train = rng.normal(size=(32, 3))
        y_train = 0.3 * x_train[:, 0] - 0.1 * x_train[:, 1] + 1.2
        x_val = rng.normal(size=(8, 3))
        y_val = 0.3 * x_val[:, 0] - 0.1 * x_val[:, 1] + 1.2
        x_test = rng.normal(size=(8, 3)) + 10.0
        model = PhoneMLP(3, hidden_dim=8)
        train_regressor(
            model,
            phone_loader(x_train, y_train, batch_size=8, shuffle=True, seed=0),
            phone_loader(x_val, y_val, batch_size=8, shuffle=False, seed=0),
            lr=0.05,
            max_epochs=3,
            patience=3,
            seed=0,
            device=torch.device("cpu"),
            sequence=False,
        )
        pred_test = predict_mlp(model, x_test, batch_size=8, device=torch.device("cpu"))
        self.assertEqual(pred_test.shape, (8,))


if __name__ == "__main__":
    unittest.main()
