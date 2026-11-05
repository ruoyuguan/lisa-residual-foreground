import hashlib
import os
import re
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "repro" / "repro_scripts"
sys.path.insert(0, str(SCRIPTS))

import lisa_catalog_statistics as statistics


TABLES = Path(os.environ.get("LISA_REPRO_TABLES", ROOT / "repro" / "tables"))
TAG = "LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz"


def finite_relative_error(lhs: np.ndarray, rhs: np.ndarray) -> float:
    scale = np.maximum(np.maximum(np.abs(lhs), np.abs(rhs)), np.finfo(float).tiny)
    return float(np.max(np.abs(lhs - rhs) / scale))


class ReproductionOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bin_summary = pd.read_csv(
            TABLES / "all_bins_LWA_X_summary_C0p9_overlap0p9_f0p4_6mHz.csv"
        )
        cls.fisher = pd.read_csv(
            TABLES / f"all_bins_fisher_baseline_{TAG}.csv"
        )
        cls.prior_table = pd.read_csv(
            TABLES / f"all_bins_publication_prior_requirement_summary_{TAG}.csv"
        )
        cls.validation_summary = pd.read_csv(
            TABLES / f"all_bins_random_phase_moment_check_summary_{TAG}.csv"
        ).iloc[0]
        cls.cumulative = pd.read_csv(
            TABLES / f"all_bins_cumulative_top_gamma2_I0_{TAG}.csv"
        )
        cls.validation = pd.read_csv(
            TABLES / f"all_bins_random_phase_moment_check_{TAG}.csv"
        )
        cls.robustness = pd.read_csv(
            TABLES / f"all_bins_bin_selection_robustness_prior_requirement_{TAG}.csv"
        )
        cls.bin_width_sensitivity = pd.read_csv(
            ROOT / "repro" / "sensitivity" / "bin_width"
            / "bin_width_sensitivity_summary.csv"
        )
        cls.orientation_sensitivity = pd.read_csv(
            ROOT / "repro" / "sensitivity" / "static_michelson_orientation"
            / "physical_snapshot_sensitivity_summary.csv"
        )
        cls.frequency_smearing = pd.read_csv(
            ROOT / "repro" / "sensitivity" / "bin_width"
            / "frequency_smearing_scales.csv"
        )
        cls.residual_profile_sensitivity = pd.read_csv(
            ROOT / "repro" / "sensitivity" / "residual_fourier_profile"
            / "residual_fourier_profile_sensitivity.csv"
        )
        cls.residual_covariance_endpoints = pd.read_csv(
            ROOT / "repro" / "sensitivity" / "residual_covariance_ensemble"
            / "fixed_vs_gaussian_residual_covariance.csv"
        )
        with np.load(
            ROOT / "repro" / "sensitivity" / "residual_fourier_profile"
            / "residual_mode_power_profiles.npz",
            allow_pickle=False,
        ) as mode_profiles:
            cls.residual_mode_profiles = {
                name: mode_profiles[name].copy() for name in mode_profiles.files
            }
        cls.residual_frequency_support = pd.read_csv(
            ROOT / "repro" / "sensitivity" / "residual_fourier_profile"
            / "residual_frequency_support_audit.csv"
        ).iloc[0]
        cls.instrument_noise_sensitivity = pd.read_csv(
            ROOT / "repro" / "sensitivity" / "instrument_noise"
            / "instrument_noise_normalization_sensitivity.csv"
        )
        cls.sgwb_amplitude_dependence = pd.read_csv(
            TABLES
            / "all_bins_sgwb_amplitude_dependence_"
            "LWA_X_C0p9_overlap0p9_f0p4_6mHz.csv"
        )

    def test_recovered_source_matching_closes_without_ambiguity(self) -> None:
        matching = pd.read_csv(
            TABLES
            / "erebor_sangria_recovered_source_matching_summary_"
            "C0p9_overlap0p9_f0p4_6mHz.csv"
        ).iloc[0]

        self.assertEqual(int(matching["accepted_comparison_rows"]), 6525)
        self.assertEqual(int(matching["matched_in_scope"]), 6511)
        self.assertEqual(int(matching["matched_out_of_scope"]), 14)
        self.assertEqual(int(matching["ambiguous"]), 0)
        self.assertEqual(int(matching["unmatched"]), 0)
        self.assertEqual(int(matching["duplicate_catalog_match"]), 0)

    def test_response_and_frequency_statistic_metadata_are_explicit(self) -> None:
        rows = self.bin_summary
        self.assertEqual(
            set(rows["analysis_channel"]),
            {"long_wavelength_michelson_X_equivalent_strain"},
        )
        self.assertEqual(
            set(rows["response_model"]),
            {"orbit_averaged_long_wavelength_michelson_x_power"},
        )
        self.assertEqual(set(rows["orbit_model"]), {"LDC_equal_arm_first_order"})
        self.assertEqual(
            set(rows["orbit_average_role"]),
            {"catalog_antenna_power_only"},
        )
        self.assertEqual(set(rows["observation_metadata_dataset"]), {"obs/tdi"})
        self.assertEqual(set(rows["response_phase_count"]), {12})
        self.assertEqual(
            set(rows["response_phase_quadrature"]),
            {"equally_spaced_over_one_orbit"},
        )
        self.assertEqual(
            set(rows["power_convention"]),
            {"single_michelson_equivalent_incident_strain"},
        )
        self.assertEqual(
            set(rows["bandpower_statistic"]),
            {"integrated_one_sided_periodogram"},
        )
        self.assertEqual(
            set(rows["bandpower_covariance_model"]),
            {"independent_positive_frequency_gaussian_modes"},
        )
        self.assertEqual(
            set(rows["bandpower_cross_bin_covariance"]),
            {"diagonal"},
        )
        self.assertEqual(
            set(rows["fisher_covariance_treatment"]),
            {"fixed_at_fiducial_model"},
        )
        self.assertEqual(
            set(rows["residual_spectral_model"]),
            {"piecewise_constant_within_coarse_bin"},
        )
        self.assertEqual(
            set(rows["random_phase_fourth_moment_used_in_covariance"]),
            {False},
        )
        for column, expected in (
            ("lisa_arm_length_m", 2.5e9),
            ("h0_km_s_Mpc", 67.4),
            ("sgwb_reference_frequency_Hz", 3.0e-3),
            ("omega0_fiducial", 1.0e-11),
            ("beta_fiducial", 1.0),
            ("orbit_guiding_center_phase_kappa_rad", 0.0),
            ("orbit_constellation_phase_lambda_rad", 0.0),
        ):
            np.testing.assert_allclose(
                rows[column].to_numpy(float),
                expected,
                rtol=1.0e-14,
                atol=0.0,
            )
        self.assertEqual(set(rows["observation_sample_count"]), {6_307_200})
        np.testing.assert_allclose(
            rows["observation_cadence_s"].to_numpy(float),
            5.0,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            rows["LWA_X_averaged_response"].to_numpy(float),
            3.0 / 20.0,
            rtol=0.0,
            atol=1.0e-15,
        )
        self.assertEqual(set(rows["fourier_mode_count"].to_numpy(int)), {315, 316})
        self.assertTrue((rows["n_eff_bandpower"] > 0.0).all())
        self.assertTrue(
            np.all(
                rows["n_eff_bandpower"].to_numpy(float)
                <= 2.0 * rows["fourier_mode_count"].to_numpy(float) + 1.0e-10
            )
        )

    def test_source_counts_and_power_sums_close_in_every_bin(self) -> None:
        rows = self.bin_summary
        for quantity in ("n_used_rows", "power_sum", "power_sum2"):
            injected = rows[f"LWA_X_inj_{quantity}"].to_numpy(float)
            recovered = rows[f"LWA_X_rec_{quantity}"].to_numpy(float)
            residual = rows[f"LWA_X_res_{quantity}"].to_numpy(float)
            error = finite_relative_error(injected, recovered + residual)
            self.assertLess(error, 2.0e-12, msg=f"closure failed for {quantity}")
        self.assertEqual(len(rows), 560)
        self.assertEqual(int((rows["LWA_X_res_power_sum"] > 0.0).sum()), 487)

    def test_mode_sum_mean_variance_and_derived_dof_close(self) -> None:
        rows = self.bin_summary
        mean = (
            rows["P_inst_bin"].to_numpy(float)
            + rows["P_sgwb_bin_fid"].to_numpy(float)
            + rows["LWA_X_res_power_sum"].to_numpy(float)
        )
        np.testing.assert_allclose(
            rows["bandpower_mean_fid"].to_numpy(float),
            mean,
            rtol=5.0e-14,
            atol=0.0,
        )
        expected_variance = (
            2.0
            * mean**2
            / rows["n_eff_bandpower"].to_numpy(float)
        )
        np.testing.assert_allclose(
            rows["bandpower_variance_gaussian"].to_numpy(float),
            expected_variance,
            rtol=5.0e-14,
            atol=0.0,
        )

    def test_random_phase_moment_diagnostic_is_not_used_as_covariance(self) -> None:
        rows = self.bin_summary
        forbidden = {
            "D_covariance_mismatch",
            "gaussian_sigma_over_fourth_moment_sigma",
        }
        self.assertTrue(forbidden.isdisjoint(rows.columns))
        self.assertEqual(
            set(rows["gamma2_total_role"]),
            {"response_averaged_random_phase_source_sum_diagnostic"},
        )
        summary = self.validation_summary
        self.assertEqual(int(summary["n_bins_total"]), 487)
        self.assertEqual(int(summary["n_bins_mc"]), 487)
        self.assertEqual(int(summary["n_bins_without_mc"]), 0)
        self.assertFalse(bool(summary["used_in_bandpower_covariance"]))

    def test_manuscript_monte_carlo_output_uses_fifty_thousand_draws(self) -> None:
        summary = self.validation_summary
        validation = self.validation

        self.assertEqual(int(summary["mc_n_draws_per_bin"]), 50_000)
        self.assertEqual(set(validation["mc_n_draws"].astype(int)), {50_000})
        np.testing.assert_allclose(
            validation["pred_excess_kurtosis"].to_numpy(float),
            -1.5 * validation["C2"].to_numpy(float),
            rtol=5.0e-14,
            atol=0.0,
        )
        np.testing.assert_allclose(
            validation["delta_mc_minus_pred_excess_kurtosis"].to_numpy(float),
            validation["mc_excess_kurtosis"].to_numpy(float)
            - validation["pred_excess_kurtosis"].to_numpy(float),
            rtol=1.0e-12,
            atol=5.0e-16,
        )

    def test_all_bin_fisher_elements_recompute_from_bin_columns(self) -> None:
        rows = self.bin_summary
        residual = rows["LWA_X_res_power_sum"].to_numpy(float)

        for alpha, suffix in ((-2.0, "m2"), (0.0, "p0"), (2.0, "p2")):
            template = rows[f"T_alpha_{suffix}"].to_numpy(float)
            variance = rows[
                f"bandpower_variance_gaussian_alpha_{suffix}"
            ].to_numpy(float)
            expected = {
                "F_oo": float(np.sum(template**2 / variance)),
                "F_ob": float(np.sum(template * residual / variance)),
                "F_bb": float(np.sum(residual**2 / variance)),
            }
            result = self.fisher.loc[
                (self.fisher["bin_set"] == "all_analysis_bins")
                & np.isclose(self.fisher["alpha"], alpha)
            ].iloc[0]
            for column, value in expected.items():
                self.assertAlmostEqual(float(result[column]) / value, 1.0, places=13)

            rho = expected["F_ob"] / np.sqrt(
                expected["F_oo"] * expected["F_bb"]
            )
            i0 = 1.0 / np.sqrt(1.0 - rho**2)
            sigma_fixed = 1.0 / np.sqrt(expected["F_oo"])
            bias = expected["F_ob"] / expected["F_oo"]
            target = 1.1
            prior_precision = (
                expected["F_ob"] ** 2
                / (expected["F_oo"] * (1.0 - target**-2))
                - expected["F_bb"]
            )
            sigma_beta = (
                1.0 / np.sqrt(prior_precision)
                if prior_precision > 0.0
                else np.inf
            )

            self.assertAlmostEqual(
                float(result["no_prior_abs_corr"]), abs(rho), places=13
            )
            self.assertAlmostEqual(
                float(result["no_prior_inflation"]), i0, places=13
            )
            self.assertAlmostEqual(
                float(result["snr_omega_fixed"]), 1.0e-11 / sigma_fixed, places=12
            )
            self.assertAlmostEqual(
                float(result["bias_ignore_over_sigma_fixed"]),
                bias / sigma_fixed,
                places=12,
            )
            if np.isinf(sigma_beta):
                self.assertTrue(np.isinf(float(result["sigma_beta_for_threshold"])))
            else:
                self.assertAlmostEqual(
                    float(result["sigma_beta_for_threshold"]) / sigma_beta,
                    1.0,
                    places=13,
                )

    def test_one_bin_two_parameter_fisher_results_are_singular(self) -> None:
        rows = self.cumulative.loc[self.cumulative["N_bin"] == 1]

        self.assertEqual(len(rows), 3)
        self.assertTrue(np.isinf(rows["I0"]).all())
        np.testing.assert_allclose(rows["rho_abs"], 1.0, rtol=0.0, atol=0.0)

    def test_frequency_subset_table_uses_one_point_kurtosis_ranking(self) -> None:
        expected_sizes = {6, 12, 24, 36}
        self.assertEqual(set(self.prior_table["N_bin"].astype(int)), expected_sizes)
        self.assertNotIn("D_covariance_mismatch", self.prior_table.columns)
        self.assertTrue(
            {
                "sigma_beta_10pct",
                "I0",
                "rho_abs",
                "bias_over_sigma_fixed",
            }.issubset(self.prior_table.columns)
        )




    def test_random_subset_intervals_are_present_at_every_displayed_size(self) -> None:
        rows = self.robustness.loc[
            self.robustness["selection"] == "random_subset"
        ].sort_values("N_bin")

        self.assertEqual(set(rows["N_bin"].astype(int)), {6, 12, 24, 36})
        for quantity in (
            "sigma_beta_10pct",
            "I0",
            "bias_over_sigma_fixed",
        ):
            median = rows[quantity].to_numpy(float)
            lower = rows[f"{quantity}_p16"].to_numpy(float)
            upper = rows[f"{quantity}_p84"].to_numpy(float)
            self.assertTrue(np.isfinite(lower).all())
            self.assertTrue(np.isfinite(upper).all())
            self.assertTrue(np.all(lower <= median))
            self.assertTrue(np.all(median <= upper))

    def test_formal_driver_reproduces_manuscript_monte_carlo_and_sensitivities(self) -> None:
        driver = (
            ROOT / "repro" / "repro_scripts" / "00_run_reproduction_pipeline.ps1"
        ).read_text(encoding="utf-8")

        self.assertIn("[int]$MCNDraws = 50000", driver)
        for script_name in (
            "26_compute_uniform_finite_source_covariance.py",
            "27_compute_sgwb_amplitude_dependence.py",
            "23_run_bin_width_sensitivity.py",
            "24_run_static_michelson_orientation_sensitivity.py",
            "25_summarize_frequency_smearing_scales.py",
        ):
            self.assertIn(script_name, driver)

    def test_bin_width_sensitivity_preserves_integrated_component_powers(self) -> None:
        rows = self.bin_width_sensitivity.sort_values("bin_width_uHz")

        self.assertEqual(set(rows["bin_width_uHz"]), {5.0, 10.0, 20.0})
        np.testing.assert_allclose(
            rows["n_analysis_bins"].to_numpy(float)
            * rows["bin_width_Hz"].to_numpy(float),
            5.6e-3,
            rtol=0.0,
            atol=2.0e-18,
        )
        for column in (
            "residual_power_band_sum",
            "instrument_power_band_sum",
            "fiducial_sgwb_power_band_sum",
        ):
            values = rows[column].to_numpy(float)
            np.testing.assert_allclose(
                values,
                values[0],
                rtol=5.0e-14,
                atol=0.0,
            )
        self.assertTrue(
            np.all(
                rows["n_eff_bandpower_max"].to_numpy(float)
                <= 2.0 * rows["fourier_mode_count_max"].to_numpy(float)
                + 1.0e-8
            )
        )

    def test_orbit_average_is_stable_against_six_phase_quadrature(self) -> None:
        formal = self.fisher.loc[
            (self.fisher["bin_set"] == "all_analysis_bins")
            & np.isclose(self.fisher["alpha"], 0.0)
        ].iloc[0]
        six_phase = self.orientation_sensitivity.loc[
            self.orientation_sensitivity["response_average"]
            == "mean_of_6_equally_spaced_snapshots"
        ].iloc[0]

        self.assertLess(
            abs(
                float(six_phase["all_bin_I0"])
                - float(formal["no_prior_inflation"])
            ),
            1.0e-3,
        )
        self.assertLess(
            abs(
                float(six_phase["all_bin_abs_rho"])
                - float(formal["no_prior_abs_corr"])
            ),
            1.0e-3,
        )

    def test_frequency_smearing_is_smaller_than_reference_coarse_bin(self) -> None:
        maximum = self.frequency_smearing.loc[
            (self.frequency_smearing["metric"] == "conservative_sum")
            & np.isclose(self.frequency_smearing["percentile"], 100.0)
        ].iloc[0]

        self.assertEqual(int(maximum["n_catalog_sources"]), 4_179_619)
        self.assertLess(float(maximum["value_Hz"]), 1.0e-5)
        self.assertGreater(
            float(maximum["value_Hz"]),
            float(maximum["fourier_resolution_Hz"]),
        )

    def test_residual_frequency_profiles_conserve_catalog_power(self) -> None:
        rows = self.residual_profile_sensitivity.loc[
            (self.residual_profile_sensitivity["bin_set"] == "all_analysis_bins")
            & np.isclose(self.residual_profile_sensitivity["alpha"], 0.0)
        ]

        self.assertEqual(
            set(rows["residual_frequency_profile"]),
            {
                "piecewise_constant_coarse_bin",
                "catalog_frequency_support",
                "nearest_fourier_mode",
            },
        )
        np.testing.assert_allclose(
            rows["residual_power_band_sum"].to_numpy(float),
            rows["residual_power_band_sum"].iloc[0],
            rtol=5.0e-12,
            atol=0.0,
        )
        self.assertEqual(
            int(self.residual_frequency_support["zero_catalog_bins_receiving_support_power"]),
            5,
        )
        self.assertLess(
            float(
                self.residual_frequency_support[
                    "catalog_support_power_moved_between_bins_fraction"
                ]
            ),
            float(self.residual_frequency_support["boundary_crossing_power_fraction"]),
        )

    def test_profile_sensitivity_is_reported_as_model_dependence(self) -> None:
        rows = self.residual_profile_sensitivity.loc[
            (self.residual_profile_sensitivity["bin_set"] == "all_analysis_bins")
            & np.isclose(self.residual_profile_sensitivity["alpha"], 0.0)
        ].set_index("residual_frequency_profile")

        self.assertGreater(
            float(rows.loc["piecewise_constant_coarse_bin", "no_prior_inflation"]),
            float(rows.loc["catalog_frequency_support", "no_prior_inflation"]),
        )
        self.assertGreater(
            float(rows.loc["catalog_frequency_support", "no_prior_inflation"]),
            float(rows.loc["nearest_fourier_mode", "no_prior_inflation"]),
        )
        self.assertTrue(
            np.isinf(
                float(rows.loc["nearest_fourier_mode", "sigma_beta_for_threshold"])
            )
        )

    def test_gaussian_covariance_endpoint_matches_profile_sensitivity(self) -> None:
        endpoint = self.residual_covariance_endpoints.loc[
            self.residual_covariance_endpoints["covariance_endpoint"]
            == "gaussian_residual_ensemble"
        ].sort_values(["residual_frequency_profile", "alpha", "bin_set"])
        profile = self.residual_profile_sensitivity.sort_values(
            ["residual_frequency_profile", "alpha", "bin_set"]
        )
        self.assertEqual(len(endpoint), len(profile))
        for column in (
            "F_oo",
            "F_ob",
            "F_bb",
            "sigma_omega_fixed",
            "no_prior_inflation",
            "no_prior_abs_corr",
            "delta_omega_omitted_residual",
            "bias_ignore_over_sigma_fixed",
            "sigma_beta_for_threshold",
        ):
            np.testing.assert_allclose(
                endpoint[column].to_numpy(float),
                profile[column].to_numpy(float),
                rtol=2.0e-14,
                atol=0.0,
            )

    def test_fixed_residual_endpoint_closes_from_cached_mode_powers(self) -> None:
        cache = self.residual_mode_profiles
        grid = statistics.FourierBandGrid(
            frequency_hz=cache["frequency_hz"],
            bin_index=cache["bin_index"].astype(int),
            mode_count=cache["mode_count"].astype(int),
            fourier_spacing_hz=float(cache["fourier_spacing_hz"]),
            bin_edges_hz=cache["bin_edges_hz"],
        )
        instrument_psd = statistics.single_michelson_lwa_equivalent_strain_psd(
            grid.frequency_hz,
            2.5e9,
        )
        template_psd = statistics.sgwb_strain_psd_per_omega(
            grid.frequency_hz,
            alpha=0.0,
            reference_frequency_hz=0.003,
            h0_km_s_mpc=67.4,
        )
        stochastic_mode_power = grid.fourier_spacing_hz * (
            instrument_psd + 1.0e-11 * template_psd
        )

        for profile_name in (
            "piecewise_constant_coarse_bin",
            "catalog_frequency_support",
            "nearest_fourier_mode",
        ):
            residual_mode_power = cache[profile_name]
            residual_band_power = np.bincount(
                grid.bin_index,
                weights=residual_mode_power,
                minlength=grid.mode_count.size,
            )
            template_band_power = np.bincount(
                grid.bin_index,
                weights=grid.fourier_spacing_hz * template_psd,
                minlength=grid.mode_count.size,
            )
            fixed_variance = np.bincount(
                grid.bin_index,
                weights=(
                    stochastic_mode_power**2
                    + 2.0 * stochastic_mode_power * residual_mode_power
                ),
                minlength=grid.mode_count.size,
            )
            expected = statistics.two_parameter_fisher_summary(
                template_band_power,
                residual_band_power,
                fixed_variance,
                omega0=1.0e-11,
                beta_fiducial=1.0,
                uncertainty_threshold=1.1,
            )
            reported = self.residual_covariance_endpoints.loc[
                (self.residual_covariance_endpoints["residual_frequency_profile"]
                 == profile_name)
                & (self.residual_covariance_endpoints["covariance_endpoint"]
                   == "fixed_residual_realization")
                & (self.residual_covariance_endpoints["bin_set"]
                   == "all_analysis_bins")
                & np.isclose(self.residual_covariance_endpoints["alpha"], 0.0)
            ].iloc[0]
            for column in (
                "F_oo",
                "F_ob",
                "F_bb",
                "no_prior_inflation",
                "no_prior_abs_corr",
                "delta_omega_omitted_residual",
                "bias_ignore_over_sigma_fixed",
                "sigma_beta_for_threshold",
            ):
                self.assertAlmostEqual(
                    float(reported[column]) / float(expected[column]),
                    1.0,
                    places=13,
                )

    def test_covariance_endpoint_variance_ordering_is_physical(self) -> None:
        fixed = self.residual_covariance_endpoints.loc[
            self.residual_covariance_endpoints["covariance_endpoint"]
            == "fixed_residual_realization"
        ]
        for column in (
            "fixed_to_gaussian_variance_ratio_min",
            "fixed_to_gaussian_variance_ratio_median",
            "fixed_to_gaussian_variance_ratio_max",
        ):
            values = fixed[column].to_numpy(float)
            self.assertTrue(np.all(values > 0.0))
            self.assertTrue(np.all(values <= 1.0 + 1.0e-14))

    def test_uniform_finite_source_random_phase_output_closes_from_cache(self) -> None:
        output_path = (
            ROOT
            / "repro"
            / "sensitivity"
            / "residual_covariance_ensemble"
            / "uniform_finite_source_random_phase_covariance.csv"
        )
        self.assertTrue(output_path.is_file())
        reported_rows = pd.read_csv(output_path)
        self.assertEqual(len(reported_rows), 6)
        self.assertEqual(
            set(reported_rows["covariance_treatment"]),
            {"finite_source_random_phase_diagonal"},
        )
        self.assertTrue(
            np.all(~reported_rows["cross_frequency_covariance_included"].astype(bool))
        )

        cache = self.residual_mode_profiles
        grid = statistics.FourierBandGrid(
            frequency_hz=cache["frequency_hz"],
            bin_index=cache["bin_index"].astype(int),
            mode_count=cache["mode_count"].astype(int),
            fourier_spacing_hz=float(cache["fourier_spacing_hz"]),
            bin_edges_hz=cache["bin_edges_hz"],
        )
        instrument_psd = statistics.single_michelson_lwa_equivalent_strain_psd(
            grid.frequency_hz,
            2.5e9,
        )
        template_psd = statistics.sgwb_strain_psd_per_omega(
            grid.frequency_hz,
            alpha=0.0,
            reference_frequency_hz=0.003,
            h0_km_s_mpc=67.4,
        )
        model = statistics.build_uniform_random_phase_catalog_bandpower_model(
            grid,
            instrument_psd,
            template_psd,
            self.bin_summary["LWA_X_res_power_sum"].to_numpy(float),
            self.bin_summary["LWA_X_res_C2"].to_numpy(float),
            omega0=1.0e-11,
            beta=1.0,
        )
        expected = statistics.two_parameter_fisher_summary(
            model.sgwb_template,
            model.residual_power,
            model.variance,
            omega0=1.0e-11,
            beta_fiducial=1.0,
            uncertainty_threshold=1.1,
        )
        reported = reported_rows.loc[
            (reported_rows["bin_set"] == "all_analysis_bins")
            & np.isclose(reported_rows["alpha"], 0.0)
        ].iloc[0]
        for column in (
            "F_oo",
            "F_ob",
            "F_bb",
            "sigma_omega_fixed",
            "sigma_omega_marginalized",
            "no_prior_inflation",
            "no_prior_abs_corr",
            "delta_omega_omitted_residual",
            "bias_ignore_over_sigma_fixed",
            "sigma_beta_for_threshold",
        ):
            self.assertAlmostEqual(
                float(reported[column]) / float(expected[column]),
                1.0,
                places=13,
            )

    def test_sgwb_amplitude_dependence_contains_recomputed_fiducial_models(self) -> None:
        rows = self.sgwb_amplitude_dependence
        self.assertEqual(
            set(rows["covariance_treatment"]),
            {
                "gaussian_residual_amplitudes",
                "finite_source_random_phase_diagonal",
            },
        )
        for treatment in rows["covariance_treatment"].unique():
            subset = rows.loc[rows["covariance_treatment"] == treatment].sort_values(
                "omega0_fiducial"
            )
            self.assertEqual(len(subset), 61)
            self.assertAlmostEqual(float(subset.iloc[0]["omega0_fiducial"]), 1.0e-13)
            self.assertAlmostEqual(float(subset.iloc[-1]["omega0_fiducial"]), 1.0e-10)
            np.testing.assert_allclose(
                subset["uncertainty_increase_percent"].to_numpy(float),
                100.0 * (subset["no_prior_inflation"].to_numpy(float) - 1.0),
                rtol=2.0e-15,
                atol=0.0,
            )
            np.testing.assert_allclose(
                subset["inverse_fractional_fixed_uncertainty"].to_numpy(float),
                subset["omega0_fiducial"].to_numpy(float)
                / subset["sigma_omega_fixed"].to_numpy(float),
                rtol=2.0e-15,
                atol=0.0,
            )
            np.testing.assert_allclose(
                subset["marginalized_amplitude_to_uncertainty_ratio"].to_numpy(float),
                subset["omega0_fiducial"].to_numpy(float)
                / subset["sigma_omega_marginalized"].to_numpy(float),
                rtol=2.0e-15,
                atol=0.0,
            )
            np.testing.assert_allclose(
                subset[
                    "instrument_noise_normalization_and_beta_marginalized_ratio"
                ].to_numpy(float),
                subset[
                    "instrument_noise_normalization_marginalized_fixed_beta_ratio"
                ].to_numpy(float)
                / subset[
                    "instrument_noise_normalization_marginalized_no_prior_inflation"
                ].to_numpy(float),
                rtol=2.0e-15,
                atol=0.0,
            )

        gaussian = rows.loc[
            (rows["covariance_treatment"] == "gaussian_residual_amplitudes")
            & np.isclose(rows["omega0_fiducial"], 1.0e-11, rtol=1.0e-12, atol=0.0)
        ].iloc[0]
        baseline = self.fisher.loc[
            (self.fisher["bin_set"] == "all_analysis_bins")
            & np.isclose(self.fisher["alpha"], 0.0)
        ].iloc[0]
        for column in (
            "F_oo",
            "F_ob",
            "F_bb",
            "sigma_omega_fixed",
            "sigma_omega_marginalized",
            "no_prior_inflation",
            "sigma_beta_for_threshold",
        ):
            self.assertAlmostEqual(
                float(gaussian[column]) / float(baseline[column]),
                1.0,
                places=13,
            )

        finite = rows.loc[
            (rows["covariance_treatment"]
             == "finite_source_random_phase_diagonal")
            & np.isclose(rows["omega0_fiducial"], 1.0e-11, rtol=1.0e-12, atol=0.0)
        ].iloc[0]
        finite_reference = pd.read_csv(
            ROOT
            / "repro"
            / "sensitivity"
            / "residual_covariance_ensemble"
            / "uniform_finite_source_random_phase_covariance.csv"
        ).loc[lambda frame: (
            (frame["bin_set"] == "all_analysis_bins")
            & np.isclose(frame["alpha"], 0.0)
        )].iloc[0]
        for column in (
            "F_oo",
            "F_ob",
            "F_bb",
            "sigma_omega_fixed",
            "sigma_omega_marginalized",
            "no_prior_inflation",
            "sigma_beta_for_threshold",
        ):
            self.assertAlmostEqual(
                float(finite[column]) / float(finite_reference[column]),
                1.0,
                places=13,
            )


    def test_fixed_noise_row_closes_against_two_parameter_baseline(self) -> None:
        noise_row = self.instrument_noise_sensitivity.loc[
            (self.instrument_noise_sensitivity["residual_frequency_profile"]
             == "piecewise_constant_coarse_bin")
            & np.isclose(
                self.instrument_noise_sensitivity[
                    "noise_normalization_prior_sigma"
                ],
                0.0,
            )
        ].iloc[0]
        baseline = self.fisher.loc[
            (self.fisher["bin_set"] == "all_analysis_bins")
            & np.isclose(self.fisher["alpha"], 0.0)
        ].iloc[0]

        self.assertAlmostEqual(
            float(noise_row["no_prior_inflation"]),
            float(baseline["no_prior_inflation"]),
            places=13,
        )
        self.assertAlmostEqual(
            float(noise_row["sigma_beta_for_threshold"]),
            float(baseline["sigma_beta_for_threshold"]),
            places=13,
        )







if __name__ == "__main__":
    unittest.main()
