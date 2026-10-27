"""Private end-to-end smoke test across four methods."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.config import Config
from src.final_data import prepare_locked_final_data
from src.combined_loader import make_combined_loader
from src.reference_records import load_reference_records
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
    refs = sweep / "references"

    freeze = json.loads(
        (sweep / "publication_freeze.json").read_text()
    )
    meta = json.loads(
        (base / "final_holdout_eligible_meta.json").read_text()
    )

    frame = prepare_locked_final_data(
        base / "final_holdout_eligible_manifest.csv",
        base / "final_holdout_candidate_manifest.csv",
        args.final_data_dir,
        meta["content_hash"],
    )
    assert len(frame) == 204
    frame = frame.head(1).copy()

    config = replace(
        Config(),
        data_dir=str(args.final_data_dir),
        batch_size=1,
        num_workers=0,
    )

    reference_records = load_reference_records(refs)

    methods = {
        "proposed": freeze["selected_runs"],
        "n2v_baseline": freeze["baseline_runs"],
        "noise2noise": [
            r for r in reference_records
            if r["stage"] == "noise2noise"
        ],
        "supervised": [
            r for r in reference_records
            if r["stage"] == "supervised"
        ],
    }

    print("Loading shared evaluation encoders...", flush=True)
    vlm = VLMWrapper(config).to(device)
    medperc = MedPercMetrics(device)

    results = []

    for method, records in methods.items():
        selected = [
            r for r in records if int(r["seed"]) == 42
        ]
        assert len(selected) == 1

        loader = make_combined_loader(
            method=method,
            freeze=freeze,
            roots=[base, sweep, base.parent],
            references_dir=refs,
            device=device,
            evaluation_config=config,
        )

        print(f"Testing {method}...", flush=True)

        result = final_per_seed(
            method,
            selected,
            frame,
            config=config,
            device=device,
            vlm=vlm,
            medperc=medperc,
            checkpoint_loader=loader,
            worker_init_fn=worker_init_fn,
            noise_types=("gaussian",),
            noise_levels=(0.05,),
        )

        assert len(result) == 1
        for metric in (
            "psnr", "ssim", "lpips",
            "medperc_l2", "medperc_cos",
            "cos_noisy", "cos_den", "cos_change",
        ):
            assert np.isfinite(float(result[metric].iloc[0]))

        results.append(result.iloc[0])

    print("\n=== FOUR-METHOD SMOKE TEST ===")
    for r in results:
        print(
            f"{r['method']}: "
            f"PSNR={r['psnr']:.4f}, "
            f"SSIM={r['ssim']:.4f}, "
            f"cos_change={r['cos_change']:.4f}"
        )

    print("\nFOUR-METHOD SMOKE TEST: PASS")
    print("Patient identifiers printed: NO")
    print("Patient-level results saved: NO")


if __name__ == "__main__":
    main()
