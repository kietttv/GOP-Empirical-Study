#!/usr/bin/env python3
"""Train one Group E MLP/Transformer scorer and save a reusable checkpoint.

Examples:
    python scripts/export_group_e_checkpoint.py \\
      --config configs/e_learned_scoring.yaml \\
      --experiment E16 \\
      --features c8_lpp_lpr_embed

    python scripts/export_group_e_checkpoint.py \\
      --config configs/e_learned_scoring.yaml \\
      --experiment E2 \\
      --features b4 \\
      --out outputs/E/e2_phone_transformer.pt

    python scripts/export_group_e_checkpoint.py \\
      --config configs/e_learned_scoring.yaml \\
      --experiment E15 \\
      --features c8_lpp_lpr_embed \\
      --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from gop_empirical.experiment import (  # noqa: E402
    export_group_e_checkpoint,
    load_config,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train and save a Group E MLP/Transformer checkpoint (.pt)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PACKAGE_ROOT / "configs" / "e_learned_scoring.yaml",
        help="Group E YAML config",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment id to save (e.g. E16, E2, E15)",
    )
    parser.add_argument(
        "--features",
        default=None,
        help="Feature set (e.g. c8_lpp_lpr_embed, b4). Default: sole set in config.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .pt path (default: outputs/E/{eid}_phone_{mlp|transformer}.pt)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing checkpoint",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override train.device (cpu|cuda)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    result = export_group_e_checkpoint(
        cfg,
        experiment_id=args.experiment,
        feature_set=args.features,
        out_path=args.out,
        package_root=PACKAGE_ROOT,
        force=bool(args.force),
        device_override=args.device,
    )
    summary = {
        "experiment_id": result["experiment_id"],
        "architecture": result["architecture"],
        "feature_set": result["feature_set"],
        "checkpoint_path": result["checkpoint_path"],
        "fit": result["fit"],
        "test": {
            "pcc": result["metrics"]["test"]["pcc"],
            "scc": result["metrics"]["test"]["scc"],
            "mae": result["metrics"]["test"]["mae"],
            "mse": result["metrics"]["test"]["mse"],
            "n": result["metrics"]["test"]["n"],
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
