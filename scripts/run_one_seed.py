"""Evaluate one frozen model and seed on the locked final holdout."""

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.config import Config
from src.final_data import prepare_locked_final_data
from src.reference_records import load_reference_records
from src.combined_loader import make_combined_loader
from src.evaluation import final_per_seed
from src.runtime import assert_private_output_dir, resolve_device, set_seed, worker_init_fn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        required=True,
        choices=["proposed", "n2v_baseline", "noise2noise", "supervised"],
    )
    parser.add_argument("--seed", type=int, choices=[7, 42, 123], required=True)
    parser.add_argument("--publication-dir", type=Path, required=True)
    parser.add_argument("--final-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="DataLoader workers (default: Config.num_workers)")
    parser.add_argument("--with-raddino", action="store_true",
                        help="also compute the RAD-DINO distance RD-cos (Eq. 10)")
    args = parser.parse_args()

    device = resolve_device(args.device)
    set_seed()  # deterministic cuDNN, as in the publication notebook
    base = args.publication_dir.resolve()
    sweep = base / "publication_sweep"
    references_dir = sweep / "references"

    output_dir = assert_private_output_dir(args.output_dir, base)

    output_file = output_dir / f"{args.method}_seed{args.seed}.csv"
    if output_file.exists():
        raise FileExistsError(f"Output already exists: {output_file}")

    freeze = json.loads((sweep / "publication_freeze.json").read_text())
    meta = json.loads(
        (base / "final_holdout_eligible_meta.json").read_text()
    )

    # Validate the complete locked holdout before evaluation.
    frame = prepare_locked_final_data(
        base / "final_holdout_eligible_manifest.csv",
        base / "final_holdout_candidate_manifest.csv",
        args.final_data_dir,
        meta["content_hash"],
    )

    assert len(frame) == 204
    assert frame["subject_id"].nunique() == 204

    config = replace(
        Config(),
        data_dir=str(args.final_data_dir),
        device=str(device),
    )
    if args.num_workers is not None:
        config = replace(config, num_workers=args.num_workers)

    references = load_reference_records(references_dir)
    groups = {
        "proposed": freeze["selected_runs"],
        "n2v_baseline": freeze["baseline_runs"],
        "noise2noise": [
            r for r in references if r["stage"] == "noise2noise"
        ],
        "supervised": [
            r for r in references if r["stage"] == "supervised"
        ],
    }

    records = [
        r for r in groups[args.method]
        if int(r["seed"]) == args.seed
    ]
    assert len(records) == 1

    # Load heavyweight encoders only after all preflight checks pass.
    from src.vlm import VLMWrapper
    from src.medperc import MedPercMetrics

    print("Loading evaluation encoders...", flush=True)
    vlm = VLMWrapper(config).to(device)
    medperc = MedPercMetrics(device)
    independent_metric = None
    if args.with_raddino:
        from src.independent_metric import IndependentPreservationMetric
        independent_metric = IndependentPreservationMetric(device, config)

    loader = make_combined_loader(
        method=args.method,
        freeze=freeze,
        roots=[base, sweep, base.parent],
        references_dir=references_dir,
        device=device,
        evaluation_config=config,
    )

    print(
        f"Evaluating {args.method}, seed={args.seed}, "
        "204 images, 12 noise configurations",
        flush=True,
    )

    result = final_per_seed(
        args.method,
        records,
        frame,
        config=config,
        device=device,
        vlm=vlm,
        medperc=medperc,
        checkpoint_loader=loader,
        worker_init_fn=worker_init_fn,
        noise_types=("gaussian", "poisson", "speckle"),
        noise_levels=(0.05, 0.10, 0.15, 0.20),
        independent_metric=independent_metric,
    )

    expected_rows = 204 * 3 * 4
    if len(result) != expected_rows:
        raise RuntimeError(
            f"Unexpected result count: {len(result)}; "
            f"expected {expected_rows}"
        )

    numeric = [
        "input_psnr", "input_ssim", "psnr", "ssim",
        "lpips", "medperc_l2", "medperc_cos",
        "cos_noisy", "cos_den", "cos_change",
    ]
    if independent_metric is not None:
        numeric += ["indep_perc_l2", "indep_perc_cos"]

    if not np.isfinite(result[numeric].to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite evaluation metrics detected")

    # Patient-level data must remain private.
    old_umask = os.umask(0o077)
    try:
        output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        with output_file.open("x", encoding="utf-8") as file:
            result.to_csv(file, index=False)
    finally:
        os.umask(old_umask)

    print("\n=== COMPLETE SINGLE-SEED EVALUATION ===")
    print("Method:", args.method)
    print("Seed:", args.seed)
    print("Records evaluated:", len(result))
    print("Unique patients:", result["subject_id"].nunique())
    print("Output: Private CSV saved outside repository")
    print("Original publication files modified: NO")
    print("SINGLE-SEED EVALUATION: PASS")


if __name__ == "__main__":
    main()
