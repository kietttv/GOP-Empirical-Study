#!/usr/bin/env python3
"""Run a GOP empirical experiment from a YAML config.

Examples:
    python scripts/run_experiment.py --config configs/a_traditional_gop.yaml
    python scripts/run_experiment.py --config configs/b_gop_representation.yaml
    python scripts/run_experiment.py --config configs/c_acoustic_model.yaml --models C1
    python scripts/run_experiment.py --config configs/e_learned_scoring.yaml
    python scripts/run_experiment.py --config configs/e_learned_scoring.yaml --features a1
    python scripts/run_experiment.py --config configs/e_learned_scoring.yaml --features c8 c9
    python scripts/run_experiment.py --config configs/e_learned_scoring.yaml --features c10 c11 --device cuda
    python scripts/run_experiment.py --config configs/e_learned_scoring.yaml --features b5
    python scripts/run_experiment.py --config configs/e_learned_scoring.yaml --features b5_embed
    python scripts/run_experiment.py --config configs/e_learned_scoring.yaml --features c8_lpp_lpr c9_lpp_lpr
    python scripts/run_experiment.py --config configs/e_learned_scoring.yaml --features c8_lpp_lpr_embed c9_lpp_lpr_embed
    python scripts/run_experiment.py --config configs/f_validation.yaml
    python scripts/run_experiment.py --config configs/f_validation.yaml --skip-multiseed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from gop_empirical.experiment import run_from_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run GOP empirical experiments "
            "(Group A: A1+A2; Group B: B1–B5; Group C: C1–C11; "
            "Group D: D1–D3; Group E: E1–E22; Group F: F1–F2)."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="YAML experiment config (e.g. configs/f_validation.yaml)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Group C only: subset of C1–C11 (default: all in the config)",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Group E only: subset of b4 a1 c8 c9 c10 c11 b5 b5_embed c8_lpp_lpr c9_lpp_lpr c8_lpp_lpr_embed c9_lpp_lpr_embed (default: features in the config)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Group E only: override train.device (cpu or cuda)",
    )
    parser.add_argument(
        "--skip-multiseed",
        action="store_true",
        help="Group F only: skip F1c multi-seed retrain (bootstrap + F2 only)",
    )
    args = parser.parse_args()
    cfg_path = args.config
    if not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path).resolve()
        if not cfg_path.is_file():
            cfg_path = (PACKAGE_ROOT / args.config).resolve()
    results = run_from_config(
        cfg_path,
        models=args.models,
        features=args.features,
        skip_multiseed=args.skip_multiseed,
        device=args.device,
    )
    results_path = results.get("results_path")
    predictions_path = results.get("predictions_path")
    if results_path:
        print(f"results: {results_path}", flush=True)
    if predictions_path:
        print(f"predictions: {predictions_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
