"""Supplementary patient-level inference on the locked-holdout results.

Computes patient-clustered bootstrap confidence intervals, paired sign-flip
p-values and Holm-adjusted per-condition tests. These analyses are NOT
reported in the paper, whose tables use seed-level mean and SD (Eq. 11); use
``scripts/make_paper_tables.py`` to recompute the published tables.

Input: the 12 private per-seed CSV files written by ``scripts/run_one_seed.py``.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.statistics import (
    patient_clustered_paired_bootstrap,
    holm_bonferroni,
)

METHODS = ("proposed", "n2v_baseline", "noise2noise", "supervised")
SEEDS = (7, 42, 123)
METRICS = ("psnr", "ssim", "lpips", "medperc_l2", "medperc_cos", "cos_change")
NOISE_TYPES = ("gaussian", "poisson", "speckle")
NOISE_LEVELS = (0.05, 0.10, 0.15, 0.20)


def load_results(results_dir):
    frames = []

    for method in METHODS:
        for seed in SEEDS:
            path = results_dir / f"{method}_seed{seed}.csv"
            if not path.is_file():
                raise FileNotFoundError(path)

            frame = pd.read_csv(
                path, dtype={"subject_id": str, "dicom_id": str}
            )

            if len(frame) != 2448:
                raise ValueError(f"Unexpected row count: {path.name}")

            if not frame["method"].eq(method).all():
                raise ValueError(f"Method mismatch: {path.name}")

            if not frame["seed"].eq(seed).all():
                raise ValueError(f"Seed mismatch: {path.name}")

            frames.append(frame)

    results = pd.concat(frames, ignore_index=True)

    keys = [
        "method", "seed", "subject_id", "dicom_id",
        "noise_type", "noise_sigma"
    ]

    if results.duplicated(keys).any():
        raise ValueError("Duplicate evaluation records")

    if not np.isfinite(results[list(METRICS)].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite evaluation metrics")

    if len(results) != 29376:
        raise ValueError("Incomplete evaluation")

    return results


def reproduce(results):
    seed_summary = (
        results.groupby(["method", "seed"], as_index=False)[list(METRICS)]
        .mean()
    )

    overall = pd.DataFrame([
        patient_clustered_paired_bootstrap(results, metric=metric)
        for metric in METRICS
    ])

    rows = []

    for metric in METRICS:
        for noise_type in NOISE_TYPES:
            for sigma in NOISE_LEVELS:
                rows.append(
                    patient_clustered_paired_bootstrap(
                        results,
                        metric=metric,
                        noise_type=noise_type,
                        noise_sigma=sigma,
                        alpha=0.05 / 12,
                    )
                )

    conditions = pd.DataFrame(rows)
    conditions["p_holm"] = np.nan

    for metric in METRICS:
        mask = conditions["metric"].eq(metric)
        conditions.loc[mask, "p_holm"] = holm_bonferroni(
            conditions.loc[mask, "p_value"].to_numpy()
        )

    conditions["significant_holm_0p05"] = conditions["p_holm"] < 0.05
    conditions["family_size"] = 12
    conditions["adjustment"] = "bonferroni CI + holm p within metric"

    return seed_summary, overall, conditions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.results_dir.is_dir():
        raise FileNotFoundError(args.results_dir)

    # Never silently overwrite an existing reproduction report.
    filenames = (
        "reproduced_seed_summary.csv",
        "reproduced_overall_statistics.csv",
        "reproduced_condition_statistics.csv",
    )
    if args.output_dir.exists():
        for name in filenames:
            if (args.output_dir / name).exists():
                raise FileExistsError(args.output_dir / name)

    results = load_results(args.results_dir)
    seed_summary, overall, conditions = reproduce(results)

    assert len(seed_summary) == 12
    assert len(overall) == 6
    assert len(conditions) == 72

    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    for filename, frame in zip(
        filenames, (seed_summary, overall, conditions)
    ):
        frame.to_csv(args.output_dir / filename, index=False)

    print("=== STATISTICAL REPRODUCTION COMPLETE ===")
    print("Evaluation records: 29376")
    print("Seed summaries:", len(seed_summary))
    print("Overall comparisons:", len(overall))
    print("Condition comparisons:", len(conditions))
    print("Patient identifiers included in saved tables: NO")
    print("Original publication files modified: NO")


if __name__ == "__main__":
    main()
