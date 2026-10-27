"""
Statistical inference for report-relative CXR denoising.

Functions extracted from the original publication notebook.
Preserves the original patient-level inference protocol.
"""

import numpy as np
import pandas as pd

def patient_clustered_paired_bootstrap(
    df,
    method_a="proposed",
    method_b="n2v_baseline",
    metric="psnr",
    n_boot=10000,
    n_perm=None,
    seed=20260731,
    noise_type=None,
    noise_sigma=None,
    alpha=0.05,
):
    """
    Paired patient-level inference.

    Confidence intervals:
        Patient-level nonparametric bootstrap.

    P-value:
    Two-sided paired patient-level random sign-flip test under the
    paired sign-exchangeability null.

    Training seeds and repeated noise conditions are averaged before
    patient-level inference, so patients—not images—are the inferential units.
    """

    work = df.copy()

    if noise_type is not None:
        work = work[
            work.noise_type.eq(noise_type)
        ]

    if noise_sigma is not None:
        work = work[
            np.isclose(
                work.noise_sigma.astype(float),
                float(noise_sigma),
            )
        ]

    # Average training seeds for the same patient/condition/method first.
    averaged = (
        work.groupby(
            [
                "method",
                "subject_id",
                "noise_type",
                "noise_sigma",
            ],
            as_index=False,
        )[metric]
        .mean()
    )

    wide = averaged.pivot_table(
        index=[
            "subject_id",
            "noise_type",
            "noise_sigma",
        ],
        columns="method",
        values=metric,
    ).dropna(
        subset=[
            method_a,
            method_b,
        ]
    )

    # Paired difference: method A minus method B.
    #
    # For the overall result, average repeated conditions inside each patient
    # so that each patient contributes exactly one inferential observation.
    patient_differences = (
        wide[method_a] - wide[method_b]
    ).groupby(
        level="subject_id"
    ).mean()

    values = patient_differences.to_numpy(
        dtype=float
    )

    if len(values) == 0:
        raise RuntimeError(
            f"No paired patient values for "
            f"metric={metric}, "
            f"noise_type={noise_type}, "
            f"noise_sigma={noise_sigma}"
        )

    observed_mean = float(values.mean())

    # ------------------------------------------------------------
    # Patient-level bootstrap confidence interval
    # ------------------------------------------------------------
    bootstrap_rng = np.random.default_rng(seed)

    bootstrap_means = np.array([
        bootstrap_rng.choice(
            values,
            size=len(values),
            replace=True,
        ).mean()
        for _ in range(n_boot)
    ])

    ci_low = float(
        np.quantile(
            bootstrap_means,
            alpha / 2.0,
        )
    )

    ci_high = float(
        np.quantile(
            bootstrap_means,
            1.0 - alpha / 2.0,
        )
    )

    # ------------------------------------------------------------
    # Proper null randomisation for the paired p-value
    # ------------------------------------------------------------
    #
    # Under the paired sign-exchangeability null, each patient's paired
    # difference may be sign-flipped to generate the null distribution.
    # Randomly flipping signs therefore generates a null distribution centred
    # at zero without changing the magnitudes of the observed patient effects.
    # ------------------------------------------------------------
    if n_perm is None:
        n_perm = n_boot

    permutation_rng = np.random.default_rng(
        seed + 1
    )

    null_means = np.empty(
        n_perm,
        dtype=float,
    )

    for index in range(n_perm):
        signs = permutation_rng.choice(
            (-1.0, 1.0),
            size=len(values),
            replace=True,
        )
        null_means[index] = np.mean(
            values * signs
        )

    # Two-sided randomisation p-value with +1 correction.
    p_value = (
        np.count_nonzero(
            np.abs(null_means)
            >= abs(observed_mean)
        )
        + 1
    ) / (
        n_perm + 1
    )

    higher_is_better = metric in {
        "psnr",
        "ssim",
        "cos_change",
    }

    improved = (
        values > 0
        if higher_is_better
        else values < 0
    )

    return {
        "metric": metric,
        "better_direction": (
            "higher"
            if higher_is_better
            else "lower"
        ),
        "difference_definition": (
            f"{method_a} - {method_b}"
        ),
        "noise_type": (
            "all"
            if noise_type is None
            else str(noise_type)
        ),
        "noise_sigma": (
            "all"
            if noise_sigma is None
            else float(noise_sigma)
        ),
        "n_patients": int(len(values)),
        "mean_difference": observed_mean,
        "median_difference": float(
            np.median(values)
        ),
        "proportion_improved": float(
            np.mean(improved)
        ),
        "alpha": float(alpha),
        "ci_level": float(
            1.0 - alpha
        ),
        "ci_method": (
            "patient_clustered_percentile_bootstrap"
        ),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value_method": (
            "paired_patient_sign_flip_two_sided"
        ),
        "n_bootstrap": int(n_boot),
        "n_permutations": int(n_perm),
        "p_value": float(p_value),
    }



def holm_bonferroni(p_values):
    """Return Holm-Bonferroni adjusted p-values, order preserved."""
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (n - rank) * p_values[index]
        running = max(running, candidate)
        adjusted[index] = min(running, 1.0)
    return adjusted
