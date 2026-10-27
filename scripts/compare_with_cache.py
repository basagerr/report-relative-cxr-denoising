"""Compare a re-run of ``run_one_seed.py`` with the archived per-image cache.

Both inputs are private (they contain patient and image identifiers); only
aggregate differences are printed. Rows are matched on subject, image, noise
family and severity.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["subject_id", "dicom_id", "noise_type", "noise_sigma"]
METRICS = ["psnr", "ssim", "lpips", "medperc_l2", "medperc_cos", "cos_noisy", "cos_den", "cos_change"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-csv", type=Path, required=True,
                        help="output of scripts/run_one_seed.py")
    parser.add_argument("--cache-csv", type=Path, required=True,
                        help="archived final_test_per_image_per_seed.csv")
    parser.add_argument("--aggregate-tolerance", type=float, default=1e-4,
                        help="largest accepted difference of the seed-level means")
    args = parser.parse_args()

    new = pd.read_csv(args.new_csv, dtype={"subject_id": str, "dicom_id": str})
    if new["method"].nunique() != 1 or new["seed"].nunique() != 1:
        raise ValueError("The new CSV must contain exactly one method and one seed")
    method, seed = new["method"].iloc[0], int(new["seed"].iloc[0])

    cache = pd.read_csv(args.cache_csv, dtype={"subject_id": str, "dicom_id": str})
    cache = cache[(cache["method"] == method) & (cache["seed"] == seed)]

    for name, frame in (("new", new), ("cache", cache)):
        if frame.duplicated(KEYS).any():
            raise ValueError(f"Duplicate keys in the {name} file")

    joined = new.merge(cache, on=KEYS, how="outer", suffixes=("_new", "_cache"),
                       indicator=True, validate="one_to_one")
    matched = joined[joined["_merge"] == "both"]
    metrics = [m for m in METRICS if f"{m}_new" in matched and f"{m}_cache" in matched]

    print(f"=== {method}, seed {seed} ===")
    print(f"rows: new={len(new)} cache={len(cache)} matched={len(matched)} "
          f"unmatched={len(joined) - len(matched)}")
    print(f"{'metric':12s} {'mean_diff':>12s} {'mean_abs':>11s} {'max_abs':>11s} "
          f"{'seed_mean_new':>14s} {'seed_mean_cache':>15s}")

    passed = len(matched) == len(new) == len(cache) > 0
    for metric in metrics:
        d = (matched[f"{metric}_new"] - matched[f"{metric}_cache"]).to_numpy(float)
        balanced_new = matched.groupby("subject_id")[f"{metric}_new"].mean().mean()
        balanced_cache = matched.groupby("subject_id")[f"{metric}_cache"].mean().mean()
        print(f"{metric:12s} {d.mean():+12.2e} {np.abs(d).mean():11.2e} {np.abs(d).max():11.2e} "
              f"{balanced_new:14.6f} {balanced_cache:15.6f}")
        if np.isnan(d).any() or abs(balanced_new - balanced_cache) > args.aggregate_tolerance:
            passed = False

    print("\nRESULT:", "PASS" if passed else "REVIEW NEEDED",
          f"(all rows matched; seed-level means within {args.aggregate_tolerance:g})")
    print("Patient identifiers printed: NO")


if __name__ == "__main__":
    main()
