"""Aggregation used for the locked-holdout tables of the paper (Tables 4 and 5).

Ported from the table cell of the publication notebook. Aggregation
follows paper Section 4.5 and Equations 8 and 11:

* within a seed, metrics are averaged over conditions per patient and then
  over patients (patient-balanced);
* Table 5 reports, per method, the mean of these seed-level values;
* Table 4 forms the proposed-minus-N2V difference within each seed and
  reports its mean and standard deviation (ddof=1) across seeds, plus the
  percentage of patients whose seed-averaged difference is favourable.

Only aggregate values are returned; no patient identifier leaves this module.
"""

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

TABLE_METRICS = ("psnr", "ssim", "lpips", "medperc_l2", "medperc_cos", "cos_change")
METHOD_ORDER = ("n2v_baseline", "proposed", "noise2noise", "supervised")
HIGHER_IS_BETTER = {
    "psnr": True,
    "ssim": True,
    "cos_change": True,
    "lpips": False,
    "medperc_l2": False,
    "medperc_cos": False,
    "indep_perc_l2": False,
    "indep_perc_cos": False,
}
REQUIRED_COLUMNS = ("method", "seed", "subject_id", "noise_type", "noise_sigma")


def _check(frame: pd.DataFrame, metrics: Iterable[str]) -> None:
    missing = sorted(set(REQUIRED_COLUMNS).union(metrics) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing result columns: {missing}")


def per_seed_patient_balanced(frame: pd.DataFrame, metrics=TABLE_METRICS) -> pd.DataFrame:
    """One row per (method, seed): patient-balanced means (Eq. 8 convention)."""
    metrics = list(metrics)
    _check(frame, metrics)
    return (
        frame.groupby(["method", "seed", "subject_id"], as_index=False)[metrics].mean()
        .groupby(["method", "seed"], as_index=False)[metrics].mean()
    )


def table5_absolute(frame: pd.DataFrame, metrics=TABLE_METRICS) -> pd.DataFrame:
    """Absolute holdout means per method, averaged across seeds (Table 5)."""
    per_seed = per_seed_patient_balanced(frame, metrics)
    present = [m for m in METHOD_ORDER if m in set(per_seed["method"])]
    table = per_seed.groupby("method")[list(metrics)].mean().reindex(present)
    table.insert(0, "n_seeds", per_seed.groupby("method")["seed"].nunique().reindex(present))
    return table


def noisy_input_row(frame: pd.DataFrame) -> Optional[pd.Series]:
    """Patient-balanced PSNR/SSIM of the undenoised input (first row of Table 5).

    The noisy input is identical for every method and seed, which is checked.
    Returns None if the input columns are absent.
    """
    if not {"input_psnr", "input_ssim", "dicom_id"}.issubset(frame.columns):
        return None
    keys = ["dicom_id", "noise_type", "noise_sigma"]
    spread = frame.groupby(keys)[["input_psnr", "input_ssim"]].agg(lambda v: v.max() - v.min())
    if float(spread.to_numpy().max()) > 1e-6:
        raise ValueError("The noisy input differs between methods or seeds")
    one = frame.drop_duplicates(keys)
    return (
        one.groupby("subject_id")[["input_psnr", "input_ssim"]].mean().mean()
        .rename({"input_psnr": "psnr", "input_ssim": "ssim"})
    )


def table4_paired(
    frame: pd.DataFrame,
    method_a: str = "proposed",
    method_b: str = "n2v_baseline",
    metrics=TABLE_METRICS,
) -> pd.DataFrame:
    """Paired method_a - method_b comparison (Table 4, Eq. 11)."""
    metrics = list(metrics)
    per_seed = per_seed_patient_balanced(frame, metrics)
    a = per_seed[per_seed.method == method_a].set_index("seed")[metrics]
    b = per_seed[per_seed.method == method_b].set_index("seed")[metrics]
    seeds = a.index.intersection(b.index)
    if len(seeds) == 0:
        raise ValueError(f"No common seeds for {method_a} and {method_b}")
    seed_diff = a.loc[seeds] - b.loc[seeds]

    per_patient = (
        frame.groupby(["method", "seed", "subject_id"], as_index=False)[metrics].mean()
        .groupby(["method", "subject_id"], as_index=False)[metrics].mean()
    )
    pa = per_patient[per_patient.method == method_a].set_index("subject_id")[metrics]
    pb = per_patient[per_patient.method == method_b].set_index("subject_id")[metrics]
    patient_diff = (pa - pb).dropna()

    rows = []
    for metric in metrics:
        favourable = (
            (patient_diff[metric] > 0) if HIGHER_IS_BETTER[metric] else (patient_diff[metric] < 0)
        ).mean()
        rows.append({
            "metric": metric,
            "mean_difference": float(seed_diff[metric].mean()),
            "sd_across_seeds": float(seed_diff[metric].std(ddof=1)) if len(seeds) > 1 else np.nan,
            "per_seed_differences": ";".join(f"{v:+.6f}" for v in seed_diff[metric]),
            "patients_favourable_percent": float(100.0 * favourable),
            "n_patients": int(len(patient_diff)),
            "n_seeds": int(len(seeds)),
        })
    return pd.DataFrame(rows)


def by_condition(frame: pd.DataFrame, metrics=TABLE_METRICS) -> pd.DataFrame:
    """Table 5 quantities split by noise family and severity."""
    metrics = list(metrics)
    _check(frame, metrics)
    per_seed = (
        frame.groupby(["method", "noise_type", "noise_sigma", "seed", "subject_id"],
                      as_index=False)[metrics].mean()
        .groupby(["method", "noise_type", "noise_sigma", "seed"], as_index=False)[metrics].mean()
    )
    return (
        per_seed.groupby(["method", "noise_type", "noise_sigma"], as_index=False)[metrics].mean()
        .sort_values(["noise_type", "noise_sigma", "method"])
        .reset_index(drop=True)
    )


def available_metrics(frame: pd.DataFrame) -> Dict[str, list]:
    """Table metrics plus the optional RAD-DINO columns when present."""
    extra = [m for m in ("indep_perc_l2", "indep_perc_cos") if m in frame.columns]
    return {"metrics": list(TABLE_METRICS) + extra}
