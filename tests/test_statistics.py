"""Synthetic-data tests; no patient data required."""

import unittest

import numpy as np
import pandas as pd

from src.statistics import (
    patient_clustered_paired_bootstrap,
    holm_bonferroni,
)


class TestStatistics(unittest.TestCase):

    @staticmethod
    def synthetic_data():
        rows = []
        for patient in range(12):
            for seed in (7, 42, 123):
                for noise in ("gaussian", "poisson"):
                    baseline = 25.0 + patient * 0.1
                    improvement = 0.2 + patient * 0.01

                    for method, value in (
                        ("n2v_baseline", baseline),
                        ("proposed", baseline + improvement),
                    ):
                        rows.append({
                            "method": method,
                            "seed": seed,
                            "subject_id": f"synthetic_{patient:03d}",
                            "noise_type": noise,
                            "noise_sigma": 0.1,
                            "psnr": value,
                        })

        return pd.DataFrame(rows)

    def test_paired_mean_and_patient_count(self):
        result = patient_clustered_paired_bootstrap(
            self.synthetic_data(),
            metric="psnr",
            n_boot=200,
            n_perm=200,
        )
        self.assertEqual(result["n_patients"], 12)
        self.assertAlmostEqual(result["mean_difference"], 0.255)
        self.assertEqual(result["difference_definition"],
                         "proposed - n2v_baseline")
        self.assertEqual(result["proportion_improved"], 1.0)

    def test_determinism(self):
        data = self.synthetic_data()
        first = patient_clustered_paired_bootstrap(
            data, metric="psnr", n_boot=200, n_perm=200
        )
        second = patient_clustered_paired_bootstrap(
            data, metric="psnr", n_boot=200, n_perm=200
        )
        self.assertEqual(first, second)

    def test_condition_filter(self):
        result = patient_clustered_paired_bootstrap(
            self.synthetic_data(),
            metric="psnr",
            noise_type="gaussian",
            noise_sigma=0.1,
            n_boot=200,
            n_perm=200,
        )
        self.assertEqual(result["n_patients"], 12)
        self.assertAlmostEqual(result["mean_difference"], 0.255)

    def test_holm_correction(self):
        values = holm_bonferroni([0.04, 0.01, 0.03])
        np.testing.assert_allclose(values, [0.06, 0.03, 0.06])

    def test_holm_order_preserved(self):
        values = holm_bonferroni([0.5, 0.001, 0.02])
        np.testing.assert_allclose(values, [0.5, 0.003, 0.04])


if __name__ == "__main__":
    unittest.main()
