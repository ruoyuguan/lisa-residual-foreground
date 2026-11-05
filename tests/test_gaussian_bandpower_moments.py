import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "repro" / "repro_scripts"
sys.path.insert(0, str(SCRIPTS))


class ExponentialSumMomentsTests(unittest.TestCase):
    def moments(self, means):
        spec = importlib.util.find_spec("check_gaussian_bandpower_moments")
        self.assertIsNotNone(spec, "The direct-moment verification module is missing.")
        from check_gaussian_bandpower_moments import exponential_sum_moments

        return exponential_sum_moments(means)

    def test_equal_means_recover_exact_chi_square_moments(self):
        for count in (1, 315, 316):
            result = self.moments(np.full(count, 2.0))
            self.assertAlmostEqual(result["mean"], 2.0 * count)
            self.assertAlmostEqual(result["variance"], 4.0 * count)
            self.assertAlmostEqual(result["nu_eff"], 2.0 * count)
            self.assertAlmostEqual(result["skewness"], 2.0 / np.sqrt(count))
            self.assertAlmostEqual(result["excess_kurtosis"], 6.0 / count)

    def test_unequal_means_use_direct_not_matched_higher_moments(self):
        result = self.moments([1.0, 2.0])
        self.assertAlmostEqual(result["mean"], 3.0)
        self.assertAlmostEqual(result["variance"], 5.0)
        self.assertAlmostEqual(result["nu_eff"], 18.0 / 5.0)
        self.assertAlmostEqual(result["skewness"], 18.0 / 5.0**1.5)
        self.assertAlmostEqual(result["excess_kurtosis"], 102.0 / 25.0)
        self.assertGreater(result["skewness"], np.sqrt(8.0 / result["nu_eff"]))
        self.assertGreater(result["excess_kurtosis"], 12.0 / result["nu_eff"])

    def test_standardized_moments_are_invariant_under_power_rescaling(self):
        reference = self.moments([1.0, 2.0, 4.0])
        for scale in (1.0e-100, 1.0e100):
            result = self.moments(scale * np.array([1.0, 2.0, 4.0]))
            for key in ("nu_eff", "skewness", "excess_kurtosis"):
                self.assertAlmostEqual(result[key], reference[key])

    def test_invalid_mode_means_are_rejected(self):
        for values in ([], [0.0, 0.0], [-1.0, 2.0], [np.nan], [np.inf], [[1.0]]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    self.moments(values)


if __name__ == "__main__":
    unittest.main()
