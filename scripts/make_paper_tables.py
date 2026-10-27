"""Recompute the locked-holdout tables of the paper from per-image results.

Inputs (private, never committed): either the per-seed CSV files written by
``scripts/run_one_seed.py`` (``<method>_seed<seed>.csv``) or the combined
``final_test_per_image_per_seed.csv`` produced by the publication notebook.

Outputs: aggregate tables only (no patient or image identifiers).
"""

import argparse
from pathlib import Path

import pandas as pd

from src.paper_tables import (
    available_metrics,
    by_condition,
    noisy_input_row,
    table4_paired,
    table5_absolute,
)

METHODS = ("proposed", "n2v_baseline", "noise2noise", "supervised")
SEEDS = (7, 42, 123)


def load(args) -> pd.DataFrame:
    if args.final_csv is not None:
        return pd.read_csv(args.final_csv, dtype={"subject_id": str, "dicom_id": str})
    frames = []
    for method in METHODS:
        for seed in SEEDS:
            path = args.results_dir / f"{method}_seed{seed}.csv"
            if path.is_file():
                frames.append(pd.read_csv(path, dtype={"subject_id": str, "dicom_id": str}))
            else:
                print(f"[missing] {path.name}")
    if not frames:
        raise FileNotFoundError(f"No result files found in {args.results_dir}")
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--results-dir", type=Path)
    source.add_argument("--final-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="optional directory for aggregate CSV tables")
    args = parser.parse_args()

    frame = load(args)
    metrics = available_metrics(frame)["metrics"]

    pd.set_option("display.width", 200)
    fmt = lambda v: f"{v:.6f}"  # noqa: E731

    table5 = table5_absolute(frame, metrics)
    noisy = noisy_input_row(frame)
    print("=" * 78)
    print("TABLE 5  locked final holdout, absolute patient-balanced means")
    print("=" * 78)
    if noisy is not None:
        print(f"Noisy input: PSNR={noisy['psnr']:.3f}  SSIM={noisy['ssim']:.4f}")
    print(table5.to_string(float_format=fmt))

    table4 = None
    if {"proposed", "n2v_baseline"}.issubset(set(frame["method"])):
        table4 = table4_paired(frame, metrics=metrics)
        print("\n" + "=" * 78)
        print("TABLE 4  paired proposed - N2V (mean and SD across seed-level differences)")
        print("=" * 78)
        print(table4.to_string(index=False, float_format=fmt))

    conditions = by_condition(frame, metrics)
    print("\n" + "=" * 78)
    print("BY CONDITION  patient-balanced means per noise family and severity")
    print("=" * 78)
    print(conditions.to_string(index=False, float_format=fmt))

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        table5.to_csv(args.output_dir / "table5_absolute.csv")
        if table4 is not None:
            table4.to_csv(args.output_dir / "table4_paired.csv", index=False)
        conditions.to_csv(args.output_dir / "by_condition.csv", index=False)
        print(f"\nAggregate tables written to {args.output_dir}")


if __name__ == "__main__":
    main()
