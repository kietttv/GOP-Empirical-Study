#!/usr/bin/env python3
"""Run Group F multi-seed only and merge into existing outputs/F/f_results.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from gop_empirical.data.learned import choose_val_speakers  # noqa: E402
from gop_empirical.data.speakers import load_speaker_metadata  # noqa: E402
from gop_empirical.experiment import (  # noqa: E402
    PACKAGE_ROOT as ROOT,
    _clip_tuple,
    _fmt_num,
    _group_f_paths,
    _log_f1c_banner,
    _run_multiseed_model,
    load_config,
    rel_path,
    write_results,
)


def main() -> int:
    cfg = load_config(ROOT / "configs" / "f_validation.yaml")
    path_map = _group_f_paths(cfg, ROOT)
    out_dir = path_map["output_dir"]
    results_path = out_dir / "f_results.json"
    if not results_path.is_file():
        raise FileNotFoundError(
            f"missing {results_path}; run Group F first with --skip-multiseed"
        )
    results = json.loads(results_path.read_text(encoding="utf-8"))

    e_cfg = load_config(path_map["e_config"])
    import torch

    multiseed_cfg = cfg.get("multiseed") or {}
    lock_val_seed = int(multiseed_cfg.get("lock_val_seed", 0))
    speaker_meta = load_speaker_metadata(path_map["speechocean_dir"])
    val_speakers = choose_val_speakers(
        speaker_meta["speakers_train"],
        frac=float(e_cfg.get("val_speaker_frac", 0.2)),
        seed=lock_val_seed,
    )
    train_cfg = e_cfg.get("train") or {}
    device_name = str(train_cfg.get("device", "cpu")).lower()
    device = (
        torch.device("cuda")
        if device_name == "cuda" and torch.cuda.is_available()
        else torch.device("cpu")
    )
    clip = _clip_tuple(cfg)

    import numpy as np

    ms_models = [str(m) for m in multiseed_cfg.get("models", ["E2", "E16"])]
    ms_seeds = [int(x) for x in multiseed_cfg.get("seeds", [0, 1, 2, 3, 4])]
    _log_f1c_banner(
        models=ms_models,
        seeds=ms_seeds,
        lock_val_seed=lock_val_seed,
        n_val=len(val_speakers),
        device=device,
    )

    block: dict = {"skipped": False, "models": {}}
    for mid in ms_models:
        runs = []
        for s in ms_seeds:
            runs.append(
                _run_multiseed_model(
                    mid,
                    s,
                    e_cfg=e_cfg,
                    package_root=ROOT,
                    val_speakers=val_speakers,
                    clip=clip,
                    device=device,
                    out_dir=out_dir,
                )
            )
        pccs = np.asarray([r["pcc"] for r in runs], dtype=np.float64)
        pcc_std = float(pccs.std(ddof=1)) if len(pccs) > 1 else 0.0
        block["models"][mid] = {
            "val_speakers_locked_seed": lock_val_seed,
            "n_speakers_val": int(len(val_speakers)),
            "runs": runs,
            "pcc_mean": float(pccs.mean()),
            "pcc_std": pcc_std,
            "pcc_min": float(pccs.min()),
            "pcc_max": float(pccs.max()),
        }
        print(
            f"F1c {mid}  pcc_mean={_fmt_num(float(pccs.mean()), 4)}  "
            f"std={_fmt_num(pcc_std, 4)}  "
            f"min={_fmt_num(float(pccs.min()), 4)}  "
            f"max={_fmt_num(float(pccs.max()), 4)}",
            flush=True,
        )

    results["F1"]["multiseed"] = block
    results["protocol"]["skip_multiseed"] = False
    write_results(results, results_path)
    print(json.dumps(block, indent=2))
    print("wrote", rel_path(results_path, base=ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
