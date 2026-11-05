import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "repro" / "repro_scripts" / "21_make_all_bin_publication_figures.py"


def load_figure_module():
    spec = importlib.util.spec_from_file_location("publication_figures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


figures = load_figure_module()


class PublicationSummaryStatisticsTests(unittest.TestCase):
    def test_fisher_uses_stored_mode_sum_variance(self) -> None:
        rows = pd.DataFrame(
            {
                "T_alpha_p0": [3.0, 2.0, 1.0],
                "LWA_X_res_power_sum": [1.2, 0.5, 0.0],
                "bandpower_variance_gaussian_alpha_p0": [4.0, 2.0, 1.0],
            }
        )

        result = figures.fisher_for_rows(
            rows,
            alpha=0.0,
            omega0=1.0,
            threshold=1.1,
            beta_fiducial=1.0,
        )

        expected_oo = 3.0**2 / 4.0 + 2.0**2 / 2.0 + 1.0
        expected_bb = 1.2**2 / 4.0 + 0.5**2 / 2.0
        expected_ob = 3.0 * 1.2 / 4.0 + 2.0 * 0.5 / 2.0
        self.assertAlmostEqual(result["F_OmegaOmega"], expected_oo)
        self.assertAlmostEqual(result["F_betabeta"], expected_bb)
        self.assertAlmostEqual(result["F_Omegabeta"], expected_ob)

    def test_empirical_percentile_preserves_an_unbounded_upper_tail(self) -> None:
        values = np.concatenate([np.full(160, 0.1), np.full(40, np.inf)])

        self.assertEqual(figures.empirical_percentile(values, 16), 0.1)
        self.assertTrue(np.isinf(figures.empirical_percentile(values, 84)))

    def test_nonfinite_prior_widths_break_plotted_curves(self) -> None:
        result = figures.plotted_values_with_gaps(np.array([0.1, np.inf, 0.2]))

        np.testing.assert_allclose(result[[0, 2]], [0.1, 0.2])
        self.assertTrue(np.isnan(result[1]))

    def test_residual_free_bins_retain_fixed_beta_sgwb_information(self) -> None:
        rows = pd.DataFrame(
            {
                "T_alpha_p0": [2.0, 1.0],
                "LWA_X_res_power_sum": [0.0, 0.0],
                "bandpower_variance_gaussian_alpha_p0": [3.0, 2.0],
            }
        )

        result = figures.fisher_for_rows(
            rows,
            alpha=0.0,
            omega0=1.0,
            threshold=1.1,
            beta_fiducial=1.0,
        )

        self.assertGreater(result["F_OmegaOmega"], 0.0)
        self.assertEqual(result["F_betabeta"], 0.0)
        self.assertEqual(result["F_Omegabeta"], 0.0)
        self.assertTrue(np.isfinite(result["sigma_omega_fixed"]))
        self.assertEqual(result["rho"], 0.0)
        self.assertEqual(result["I0"], 1.0)
        self.assertEqual(result["bias_over_sigma_fixed"], 0.0)
        self.assertTrue(np.isinf(result["sigma_beta"]))

    def test_exactly_singular_one_bin_fisher_matrix_has_infinite_i0(self) -> None:
        rows = pd.DataFrame(
            {
                "T_alpha_p0": [2.0],
                "LWA_X_res_power_sum": [1.0],
                "bandpower_variance_gaussian_alpha_p0": [3.0],
            }
        )

        result = figures.fisher_for_rows(
            rows,
            alpha=0.0,
            omega0=1.0,
            beta_fiducial=1.0,
        )

        self.assertEqual(result["rho"], 1.0)
        self.assertTrue(np.isinf(result["I0"]))


if __name__ == "__main__":
    unittest.main()
