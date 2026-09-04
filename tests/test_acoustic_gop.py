"""Unit tests for Group C phone inventory, SSL GOP formula, CTM alignment, C1 ≡ A2."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gop_empirical.acoustic.alignment import (  # noqa: E402
    clean_phone,
    drop_silence,
    encoder_hop_samples,
    scale_segments,
    time_to_frame_span,
)
from gop_empirical.acoustic.phones import load_phone_inventory  # noqa: E402
from gop_empirical.eval.metrics import apply_linear_map, fit_linear_score_map  # noqa: E402
from gop_empirical.gop.from_posterior import (  # noqa: E402
    gop_from_log_probs,
    gop_max_from_log_probs,
    log_softmax_over_ids,
)
from gop_empirical.gop.cao import cao_gop_s, cao_gop_sd  # noqa: E402
from gop_empirical.gop.traditional import traditional_gop_batch  # noqa: E402


def _make_feat(phone_id: int, canonical_lpp: float, n_phones: int = 42) -> np.ndarray:
    lpp = np.full(n_phones, -10.0, dtype=np.float64)
    lpp[phone_id] = canonical_lpp
    lpr = canonical_lpp - lpp
    return np.concatenate([[float(phone_id)], lpp, lpr])


class InventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inv = load_phone_inventory(ROOT / "data" / "phone_inventory.json")

    def test_slot_counts(self):
        self.assertEqual(self.inv.n_kaldi_slots, 42)
        self.assertEqual(self.inv.n_scored_phones, 39)
        self.assertEqual(len(self.inv.scored_symbols), 39)
        self.assertEqual(self.inv.kaldi_symbol(0), "SIL")
        self.assertEqual(self.inv.kaldi_symbol(3), "AA")
        self.assertEqual(self.inv.kaldi_symbol(41), "ZH")
        self.assertTrue(self.inv.is_skip("SIL"))
        self.assertFalse(self.inv.is_skip("TH"))

    def test_ssl_scored_index_skips_blank(self):
        self.assertEqual(self.inv.ssl_ctc_id("<pad>"), 0)
        self.assertEqual(self.inv.ssl_ctc_id("AA"), 1)
        self.assertEqual(self.inv.ssl_index("AA"), 0)
        self.assertEqual(self.inv.ssl_index("ZH"), 38)
        self.assertIn(0, self.inv.blank_ctc_ids())


class PosteriorFormulaTests(unittest.TestCase):
    def test_blank_renormalization_sums_to_one(self):
        # vocab: 0=blank, 1=AA, 2=AE ; scored ids = [1, 2]
        logits = np.array([[0.0, 1.0, 2.0], [5.0, 0.0, 0.0]])
        log_p = log_softmax_over_ids(logits, np.array([1, 2]))
        probs = np.exp(log_p)
        np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0], atol=1e-12)
        # blank logit 5.0 on row 1 must not enter the denom
        self.assertAlmostEqual(float(probs[1, 0]), 0.5, places=6)
        self.assertAlmostEqual(float(probs[1, 1]), 0.5, places=6)

    def test_gop_is_mean_log_canonical_ignores_competitor(self):
        # two frames, 3 phones; canonical = 0
        log_p = np.array(
            [
                [-0.2, -2.0, -3.0],
                [-0.4, -0.1, -4.0],
            ]
        )
        gop, n = gop_from_log_probs(log_p, canonical_idx=0, t0=0, t1=2)
        self.assertEqual(n, 2)
        self.assertAlmostEqual(gop, (-0.2 + -0.4) / 2)
        # competitor mass on frame 1 does not change canonical mean
        self.assertNotAlmostEqual(gop, float(log_p[:, 1].mean()))

    def test_empty_span_is_nan_zero_frames(self):
        log_p = np.zeros((4, 39))
        gop, n = gop_from_log_probs(log_p, 0, 2, 2)
        self.assertEqual(n, 0)
        self.assertTrue(np.isnan(gop))

    def test_max_gop_uses_peak_frame_not_mean(self):
        log_p = np.array([[-4.0, -1.0], [-0.1, -3.0], [-5.0, -2.0]])
        mean_gop, n_mean = gop_from_log_probs(log_p, 0, 0, 3)
        max_gop, n_max = gop_max_from_log_probs(log_p, 0, 0, 3)
        self.assertEqual(n_mean, n_max)
        self.assertAlmostEqual(mean_gop, (-4.0 + -0.1 + -5.0) / 3)
        self.assertAlmostEqual(max_gop, -0.1)
        self.assertGreater(max_gop, mean_gop)

    def test_cao_gop_s_higher_when_canonical_is_peaked(self):
        # T=6, V=4 (blank + 3 phones). Mass on phone 1 then phone 2, matching seq.
        logits = np.full((6, 4), -8.0)
        logits[0:3, 1] = 4.0
        logits[3:6, 2] = 4.0
        from gop_empirical.gop.from_posterior import log_softmax_rows

        probs = np.exp(log_softmax_rows(logits))
        gop_ok = cao_gop_s(probs, np.array([1, 2]), blank=0)
        gop_bad = cao_gop_s(probs, np.array([2, 1]), blank=0)
        self.assertEqual(gop_ok.shape, (2,))
        self.assertTrue(np.isfinite(gop_ok).all())
        self.assertTrue(np.isfinite(gop_bad).all())
        self.assertGreater(float(gop_ok.mean()), float(gop_bad.mean()))

    def test_cao_gop_sd_finite_and_differs_when_phone_missing(self):
        from gop_empirical.gop.from_posterior import log_softmax_rows

        logits = np.full((9, 4), -8.0)
        logits[0:3, 1] = 4.0
        logits[3:6, 0] = 4.0
        logits[6:9, 3] = 4.0
        probs = np.exp(log_softmax_rows(logits))
        seq = np.array([1, 2, 3])
        gop_s = cao_gop_s(probs, seq, blank=0)
        gop_sd = cao_gop_sd(probs, seq, blank=0)
        self.assertEqual(gop_sd.shape, (3,))
        self.assertTrue(np.isfinite(gop_s).all())
        self.assertTrue(np.isfinite(gop_sd).all())
        self.assertFalse(np.allclose(gop_s, gop_sd))
        self.assertLess(float(gop_sd[1]), float(gop_s[1]))

    def test_cao_gop_sd_single_phone_does_not_crash(self):
        from gop_empirical.gop.from_posterior import log_softmax_rows

        logits = np.full((4, 3), -8.0)
        logits[:, 1] = 4.0
        probs = np.exp(log_softmax_rows(logits))
        gop = cao_gop_sd(probs, np.array([1]), blank=0)
        self.assertEqual(gop.shape, (1,))
        self.assertTrue(np.isfinite(gop).all())


class AlignmentTests(unittest.TestCase):
    def test_ctm_rescale_when_clocks_disagree(self):
        segs = [(0.0, 1.0, "AA"), (1.0, 1.0, "B")]
        scaled, scale = scale_segments(segs, dur_s=0.4)
        self.assertAlmostEqual(scale, 0.2)
        self.assertAlmostEqual(scaled[0][1], 0.2)
        self.assertAlmostEqual(scaled[1][0] + scaled[1][1], 0.4)

    def test_no_rescale_when_clocks_match(self):
        segs = [(0.0, 0.3, "AA")]
        scaled, scale = scale_segments(segs, dur_s=0.32)
        self.assertEqual(scale, 1.0)
        self.assertEqual(scaled, list(segs))

    def test_time_to_frame_empty_after_clip(self):
        i0, i1 = time_to_frame_span(0.05, 0.05, hop_s=0.02, n_frames=10)
        self.assertGreaterEqual(i1, i0)
        i0, i1 = time_to_frame_span(2.0, 2.1, hop_s=0.02, n_frames=10)
        self.assertEqual((i0, i1), (10, 10))

    def test_clean_phone_strips_stress_and_silence(self):
        self.assertEqual(clean_phone("IY0_E"), "IY")
        self.assertEqual(clean_phone("W_B"), "W")
        self.assertIsNone(clean_phone("SIL"))
        self.assertIsNone(clean_phone("SPN_B"))
        self.assertEqual(drop_silence([(0.0, 0.1, "SIL"), (0.1, 0.2, "AA0")]), [(0.1, 0.2, "AA")])

    def test_wav2vec2_hop_is_320(self):
        self.assertEqual(encoder_hop_samples([5, 2, 2, 2, 2, 2, 2]), 320)


class C1MatchesTraditionalGopTests(unittest.TestCase):
    def test_c1_scalar_matches_traditional_gop_batch(self):
        feats = np.vstack([_make_feat(3, -1.0), _make_feat(24, 5.3)])
        got = traditional_gop_batch(feats, expected_n_phones=42)
        np.testing.assert_allclose(got, [-1.0, 5.3])


class PairedJoinTests(unittest.TestCase):
    def test_unmatched_ssl_keys_are_nan(self):
        from gop_empirical.experiment import _paired_mask

        df = pd.DataFrame(
            {
                "gop_c1": [1.0, 2.0, 3.0],
                "gop_c2": [1.1, np.nan, 3.1],
                "gop_c3": [1.2, 2.2, np.nan],
            }
        )
        mask = _paired_mask(df, ["C1", "C2", "C3"])
        self.assertEqual(int(mask.sum()), 1)
        self.assertTrue(bool(mask.iloc[0]))

    def test_paired_mask_c10(self):
        from gop_empirical.experiment import _paired_mask

        df = pd.DataFrame(
            {
                "gop_c1": [1.0, 2.0],
                "gop_c8": [1.1, 2.1],
                "gop_c10": [1.2, np.nan],
            }
        )
        mask = _paired_mask(df, ["C1", "C8", "C10"])
        self.assertEqual(int(mask.sum()), 1)

    def test_mapping_still_train_only(self):
        rng = np.random.default_rng(0)
        gop_train = rng.normal(size=200)
        y_train = 0.4 * gop_train + 1.2 + rng.normal(scale=0.05, size=200)
        gop_test = rng.normal(size=80)
        y_test = np.full(80, 99.0)
        mapping = fit_linear_score_map(gop_train, y_train)
        pred = apply_linear_map(gop_test, mapping, clip=None)
        self.assertFalse(np.allclose(pred, y_test))


class ProtocolLockAgainstGroupATests(unittest.TestCase):
    def test_c1_matches_a1_predictions_if_present(self):
        a1_path = ROOT / "outputs" / "A" / "a1_predictions.csv"
        if not a1_path.is_file():
            self.skipTest("Group A artifacts not present")
        from gop_empirical.experiment import build_group_c_predictions, load_config

        cfg = load_config(ROOT / "configs" / "c_acoustic_model.yaml")
        df, _stats = build_group_c_predictions(cfg, package_root=ROOT, models=["C1"])
        a1 = pd.read_csv(a1_path, dtype={"utt_id": str})
        merged = a1.merge(
            df[["utt_id", "split", "word_id", "phone_id", "gop_c1"]],
            on=["utt_id", "split", "word_id", "phone_id"],
            how="inner",
        )
        self.assertGreater(len(merged), 1000)
        np.testing.assert_allclose(
            merged["gop"].to_numpy(), merged["gop_c1"].to_numpy(), rtol=1e-10, atol=1e-10
        )

    def test_ssl_gop_csv_keeps_leading_zero_keys(self):
        from gop_empirical.data.ssl_gop import load_ssl_gop_split, write_ssl_gop_split

        with tempfile.TemporaryDirectory() as tmp:
            write_ssl_gop_split(
                [
                    {"key": "000010011.0", "phone": "W", "gop": -1.23, "n_frames": 4},
                    {"key": "000010022.3", "phone": "AH", "gop": -0.5, "n_frames": 3},
                ],
                tmp,
                "train",
            )
            got = load_ssl_gop_split(tmp, "train")
            self.assertIn("000010011.0", got)
            self.assertNotIn("10010011.0", got)
            self.assertAlmostEqual(got["000010011.0"]["gop"], -1.23)

    def test_missing_ssl_extract_raises_clear_error(self):
        from gop_empirical.experiment import build_group_c_predictions, load_config

        cfg = load_config(ROOT / "configs" / "c_acoustic_model.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(cfg)
            cfg["paths"] = dict(cfg["paths"])
            cfg["paths"]["wav2vec2_gop_dir"] = tmp
            with self.assertRaises(FileNotFoundError) as ctx:
                build_group_c_predictions(cfg, package_root=ROOT, models=["C1", "C2"])
            self.assertIn("C2", str(ctx.exception))
            self.assertIn("extract_ssl_gop", str(ctx.exception))

    def test_missing_c4_extract_mentions_max_gop(self):
        from gop_empirical.experiment import build_group_c_predictions, load_config

        cfg = load_config(ROOT / "configs" / "c_acoustic_model.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(cfg)
            cfg["paths"] = dict(cfg["paths"])
            cfg["paths"]["hubert_gop_max_dir"] = tmp
            with self.assertRaises(FileNotFoundError) as ctx:
                build_group_c_predictions(cfg, package_root=ROOT, models=["C1", "C4"])
            self.assertIn("C4", str(ctx.exception))
            self.assertIn("--gop max", str(ctx.exception))

    def test_missing_c6_extract_mentions_wav2vec2_max(self):
        from gop_empirical.experiment import build_group_c_predictions, load_config

        cfg = load_config(ROOT / "configs" / "c_acoustic_model.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(cfg)
            cfg["paths"] = dict(cfg["paths"])
            cfg["paths"]["wav2vec2_gop_max_dir"] = tmp
            with self.assertRaises(FileNotFoundError) as ctx:
                build_group_c_predictions(cfg, package_root=ROOT, models=["C1", "C6"])
            self.assertIn("C6", str(ctx.exception))
            self.assertIn("wav2vec2 --gop max", str(ctx.exception))

    def test_missing_c8_extract_mentions_xlsr_espeak(self):
        from gop_empirical.experiment import build_group_c_predictions, load_config

        cfg = load_config(ROOT / "configs" / "c_acoustic_model.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(cfg)
            cfg["paths"] = dict(cfg["paths"])
            cfg["paths"]["xlsr_espeak_gop_cao_dir"] = tmp
            with self.assertRaises(FileNotFoundError) as ctx:
                build_group_c_predictions(cfg, package_root=ROOT, models=["C1", "C8"])
            self.assertIn("C8", str(ctx.exception))
            self.assertIn("xlsr_espeak --gop cao_s", str(ctx.exception))

    def test_missing_c10_extract_mentions_cao_sd(self):
        from gop_empirical.experiment import build_group_c_predictions, load_config

        cfg = load_config(ROOT / "configs" / "c_acoustic_model.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(cfg)
            cfg["paths"] = dict(cfg["paths"])
            cfg["paths"]["xlsr_espeak_gop_cao_sd_dir"] = tmp
            with self.assertRaises(FileNotFoundError) as ctx:
                build_group_c_predictions(cfg, package_root=ROOT, models=["C1", "C10"])
            self.assertIn("C10", str(ctx.exception))
            self.assertIn("xlsr_espeak --gop cao_sd", str(ctx.exception))

    def test_merge_group_c_results_keeps_c8_when_adding_c10(self):
        from gop_empirical.experiment import merge_group_c_results

        existing = {
            "experiments": ["C1", "C8"],
            "C8": {"pcc": 0.463},
            "comparison": {"test": {"C8": {"pcc": 0.463}}, "test_paired": {"C8": {"pcc": 0.461}}},
            "protocol": {
                "requested_models": ["C1", "C8"],
                "gop_type_by_model": {"C8": "cao_gop_s"},
            },
        }
        new = {
            "experiments": ["C1", "C10"],
            "C10": {"pcc": 0.47},
            "comparison": {"test": {"C10": {"pcc": 0.47}}, "test_paired": {"C10": {"pcc": 0.47}}},
            "protocol": {
                "requested_models": ["C1", "C10"],
                "gop_type_by_model": {"C10": "cao_gop_sd"},
            },
        }
        merged = merge_group_c_results(existing, new)
        self.assertEqual(merged["C8"]["pcc"], 0.463)
        self.assertEqual(merged["C10"]["pcc"], 0.47)
        self.assertIn("C8", merged["experiments"])
        self.assertIn("C10", merged["experiments"])
        self.assertEqual(merged["protocol"]["gop_type_by_model"]["C8"], "cao_gop_s")
        self.assertEqual(merged["protocol"]["gop_type_by_model"]["C10"], "cao_gop_sd")

    def test_c8_c9_checkpoints_are_local_models_dirs(self):
        from gop_empirical.experiment import load_config

        cfg = load_config(ROOT / "configs" / "c_acoustic_model.yaml")
        xlsr = cfg["paths"]["xlsr_espeak_checkpoint"]
        lv60 = cfg["paths"]["lv60_espeak_checkpoint"]
        self.assertTrue(str(xlsr).replace("\\", "/").startswith("models/"))
        self.assertTrue(str(lv60).replace("\\", "/").startswith("models/"))
        self.assertNotIn("facebook/", str(xlsr))
        self.assertNotIn("facebook/", str(lv60))


class EspeakMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gop_empirical.acoustic.espeak_map import (
            ctc_ids_from_vocab,
            load_cmu_to_espeak,
            load_espeak_vocab,
        )

        cls.inv = load_phone_inventory(ROOT / "data" / "phone_inventory.json")
        cls.mapping = load_cmu_to_espeak(
            ROOT / "data" / "cmu_to_espeak.json",
            scored_symbols=cls.inv.scored_symbols,
        )
        cls.vocab = load_espeak_vocab(ROOT / "data" / "espeak_ctc_vocab.json")
        cls.ids = ctc_ids_from_vocab(cls.mapping, cls.vocab)

    def test_map_covers_39_scored_one_to_one(self):
        self.assertEqual(len(self.mapping), 39)
        self.assertEqual(set(self.mapping), set(self.inv.scored_symbols))
        self.assertEqual(len(set(self.mapping.values())), 39)

    def test_tokens_exist_in_frozen_vocab(self):
        self.assertEqual(len(self.ids), 39)
        self.assertEqual(self.vocab["<pad>"], 0)
        self.assertNotIn(0, set(self.ids.values()))

    def test_missing_token_raises(self):
        from gop_empirical.acoustic.espeak_map import ctc_ids_from_vocab

        broken = dict(self.vocab)
        broken.pop(self.mapping["AE"], None)
        with self.assertRaises(KeyError) as ctx:
            ctc_ids_from_vocab(self.mapping, broken)
        self.assertIn("AE", str(ctx.exception))

    def test_tokenizer_missing_token_raises(self):
        from gop_empirical.acoustic.espeak_map import ctc_ids_from_tokenizer

        class _Tok:
            def __init__(self, vocab):
                self._v = dict(vocab)
                self.unk_token_id = 3
                self.unk_token = "<unk>"
                self.pad_token_id = 0

            def get_vocab(self):
                return dict(self._v)

        tok = _Tok(self.vocab)
        tok._v.pop(self.mapping["SH"])
        with self.assertRaises(KeyError):
            ctc_ids_from_tokenizer(self.mapping, tok)

    def test_cao_gop_s_accepts_mapped_seq_ids(self):
        aa = self.ids["AA"]
        t = self.ids["T"]
        logits = np.full((6, max(self.vocab.values()) + 1), -8.0)
        logits[0:3, aa] = 4.0
        logits[3:6, t] = 4.0
        from gop_empirical.gop.from_posterior import log_softmax_rows

        probs = np.exp(log_softmax_rows(logits))
        gop = cao_gop_s(probs, np.array([aa, t]), blank=0)
        self.assertEqual(gop.shape, (2,))
        self.assertTrue(np.isfinite(gop).all())
        gop_sd = cao_gop_sd(probs, np.array([aa, t]), blank=0)
        self.assertEqual(gop_sd.shape, (2,))
        self.assertTrue(np.isfinite(gop_sd).all())


class CtcAlignTests(unittest.TestCase):
    def test_viterbi_assigns_peaked_frames(self):
        from gop_empirical.gop.ctc_align import ctc_label_frames

        # blank=0, phones 1 then 2. Mass on frames 1 and 3.
        t_len, vocab = 5, 3
        logits = np.full((t_len, vocab), -8.0)
        logits[0, 0] = 4.0
        logits[1, 1] = 4.0
        logits[2, 0] = 4.0
        logits[3, 2] = 4.0
        logits[4, 0] = 4.0
        from gop_empirical.gop.from_posterior import log_softmax_rows

        probs = np.exp(log_softmax_rows(logits))
        frames = ctc_label_frames(probs, np.array([1, 2]), blank=0)
        self.assertEqual(len(frames), 2)
        np.testing.assert_array_equal(frames[0], np.array([1]))
        np.testing.assert_array_equal(frames[1], np.array([3]))

    def test_repeat_phones_need_blank(self):
        from gop_empirical.gop.ctc_align import ctc_label_frames
        from gop_empirical.gop.from_posterior import log_softmax_rows

        logits = np.full((5, 2), -8.0)
        logits[0, 0] = 4.0
        logits[1, 1] = 4.0
        logits[2, 0] = 4.0
        logits[3, 1] = 4.0
        logits[4, 0] = 4.0
        probs = np.exp(log_softmax_rows(logits))
        frames = ctc_label_frames(probs, np.array([1, 1]), blank=0)
        np.testing.assert_array_equal(frames[0], np.array([1]))
        np.testing.assert_array_equal(frames[1], np.array([3]))


if __name__ == "__main__":
    unittest.main()
