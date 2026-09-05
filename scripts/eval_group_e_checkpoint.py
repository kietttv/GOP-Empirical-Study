#!/usr/bin/env python3
"""Evaluate a saved Group E checkpoint on the official test split.

Does not train. Loads ``checkpoints/{eid}/{mlp|transformer}_ckpt.pt``.

Examples:
    python scripts/eval_group_e_checkpoint.py --experiment E15
    python scripts/eval_group_e_checkpoint.py --experiment E1 E2
    python scripts/eval_group_e_checkpoint.py --experiment E16 --device cuda
    python scripts/eval_group_e_checkpoint.py --experiment E2 --checkpoint checkpoints/e2/transformer_ckpt.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from gop_empirical.experiment import eval_group_e_checkpoints, load_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Group E MLP/Transformer checkpoint(s) on official test."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PACKAGE_ROOT / "configs" / "e_learned_scoring.yaml",
        help="Group E YAML config",
    )
    parser.add_argument(
        "--experiment",
        nargs="+",
        required=True,
        help="Experiment id(s) (e.g. E15, E1 E2)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Override .pt path (single --experiment only)",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Official split to score (default: test)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override train.device (cpu|cuda)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    results = eval_group_e_checkpoints(
        cfg,
        args.experiment,
        checkpoint_path=args.checkpoint,
        package_root=PACKAGE_ROOT,
        device_override=args.device,
        split=args.split,
    )
    payload = []
    for result in results:
        m = result["metrics"]
        payload.append(
            {
                "experiment_id": result["experiment_id"],
                "architecture": result["architecture"],
                "feature_set": result["feature_set"],
                "checkpoint_path": result["checkpoint_path"],
                "split": result["split"],
                "metrics": {
                    "pcc": m["pcc"],
                    "scc": m["scc"],
                    "mae": m["mae"],
                    "mse": m["mse"],
                    "n": m["n"],
                },
            }
        )
    print(json.dumps(payload if len(payload) > 1 else payload[0], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
