"""Private GPU smoke test of the frozen publication evaluation."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.config import Config
from src.final_data import prepare_locked_final_data
from src.combined_loader import make_combined_loader
from src.evaluation import final_per_seed
from src.vlm import VLMWrapper
from src.medperc import MedPercMetrics

from src.runtime import resolve_device, set_seed, worker_init_fn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication-dir", type=Path, required=True)
    parser.add_argument("--final-data-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()

    device = resolve_device(args.device)
    set_seed()  # deterministic cuDNN, as in the publication notebook
    base = args.publication_dir
    sweep = base / "publication_sweep"

    freeze = json.loads(
        (sweep / "publication_freeze.json").read_text()
    )
    meta = json.loads(
        (base / "final_holdout_eligible_meta.json").read_text()
    )

    print("=== REAL EVALUATION SMOKE TEST ===", flush=True)

    frame = prepare_locked_final_data(
        eligible_manifest=base / "final_holdout_eligible_manifest.csv",
        candidate_manifest=base / "final_holdout_candidate_manifest.csv",
        final_data_dir=args.final_data_dir,
        expected_content_hash=meta["content_hash"],
    )

    assert len(frame) == 204
    frame = frame.head(1).copy()

    config = replace(
        Config(),
        data_dir=str(args.final_data_dir),
        batch_size=1,
        num_workers=0,
    )

    records = [
        record
        for record in freeze["selected_runs"]
        if int(record["seed"]) == 42
    ]
    assert len(records) == 1

    checkpoint_loader = make_combined_loader(
        method="proposed",
        freeze=freeze,
        roots=[base, sweep, base.parent],
        references_dir=sweep / "references",
        device=device,
        evaluation_config=config,
    )

    print("Loading BioViL-T...", flush=True)
    vlm = VLMWrapper(config)
    vlm.to(device)

    print("Loading medical perceptual encoder...", flush=True)
    medperc = MedPercMetrics(device)

    print("Running actual evaluation...", flush=True)

    result = final_per_seed(
        "proposed",
        records,
        frame,
        config=config,
        device=device,
        vlm=vlm,
        medperc=medperc,
        checkpoint_loader=checkpoint_loader,
        worker_init_fn=worker_init_fn,
        noise_types=("gaussian",),
        noise_levels=(0.05,),
    )

    assert len(result) == 1
    assert result["method"].iloc[0] == "proposed"

    metrics = [
        "input_psnr", "input_ssim",
        "psnr", "ssim", "lpips",
        "medperc_l2", "medperc_cos",
        "cos_noisy", "cos_den", "cos_change",
    ]

    print("\n=== SMOKE TEST METRICS ===")
    for name in metrics:
        value = float(result[name].iloc[0])
        assert np.isfinite(value), f"Non-finite metric: {name}"
        print(f"{name}: {value:.6f}")

    print("\nREAL EVALUATION SMOKE TEST: PASS")
    print("Patient identifiers printed: NO")
    print("Patient-level results saved: NO")
    print("Original publication files modified: NONE")


if __name__ == "__main__":
    main()
