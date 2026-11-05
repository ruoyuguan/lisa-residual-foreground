import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "repro" / "repro_scripts"
sys.path.insert(0, str(SCRIPTS))

import lisa_catalog_statistics as statistics


class StaticLisaGeometryTests(unittest.TestCase):
    def test_reference_constellation_is_equilateral_and_inclined(self) -> None:
        positions = statistics.lisa_equal_arm_reference_positions()
        separations = np.array(
            [
                np.linalg.norm(positions[1] - positions[0]),
                np.linalg.norm(positions[2] - positions[1]),
                np.linalg.norm(positions[0] - positions[2]),
            ]
        )

        np.testing.assert_allclose(
            separations,
            np.full(3, separations.mean()),
            rtol=0.0,
            atol=1.0e-14,
        )
        self.assertAlmostEqual(
            statistics.constellation_plane_inclination(positions),
            np.pi / 3.0,
            places=14,
        )

    def test_reference_michelson_has_single_channel_response(self) -> None:
        detector = statistics.static_lisa_michelson_x_tensor()

        response = statistics.isotropic_single_polarization_response(detector)

        self.assertAlmostEqual(response, 3.0 / 20.0, places=15)


class CoarseBandpowerTests(unittest.TestCase):
    def test_actual_sangria_fourier_mode_counts_are_315_or_316(self) -> None:
        edges = np.arange(0.4e-3, 6.0e-3 + 1.0e-5, 1.0e-5)

        grid = statistics.positive_fourier_modes_by_bin(
            n_time_samples=6_307_200,
            cadence_s=5.0,
            bin_edges_hz=edges,
        )

        self.assertEqual(set(np.unique(grid.mode_count)), {315, 316})
        self.assertAlmostEqual(grid.fourier_spacing_hz, 1.0 / 31_536_000.0)

    def test_constant_psd_gives_two_degrees_of_freedom_per_mode(self) -> None:
        edges = np.array([0.10, 0.20, 0.305])
        grid = statistics.positive_fourier_modes_by_bin(
            n_time_samples=100,
            cadence_s=1.0,
            bin_edges_hz=edges,
        )
        psd = np.full(grid.frequency_hz.size, 7.0)

        moments = statistics.integrate_gaussian_periodogram(grid, psd)

        np.testing.assert_allclose(
            moments.effective_degrees_of_freedom,
            2.0 * grid.mode_count,
            rtol=0.0,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(
            moments.variance,
            moments.mean**2 / grid.mode_count,
            rtol=0.0,
            atol=1.0e-14,
        )

    def test_frequency_dependent_psd_uses_the_mode_sum_variance(self) -> None:
        edges = np.array([0.10, 0.20, 0.305])
        grid = statistics.positive_fourier_modes_by_bin(
            n_time_samples=100,
            cadence_s=1.0,
            bin_edges_hz=edges,
        )
        psd = 2.0 + 5.0 * grid.frequency_hz**2

        moments = statistics.integrate_gaussian_periodogram(grid, psd)
        expected_mean = np.bincount(
            grid.bin_index,
            weights=grid.fourier_spacing_hz * psd,
            minlength=grid.mode_count.size,
        )
        expected_variance = np.bincount(
            grid.bin_index,
            weights=(grid.fourier_spacing_hz * psd) ** 2,
            minlength=grid.mode_count.size,
        )

        np.testing.assert_allclose(moments.mean, expected_mean)
        np.testing.assert_allclose(moments.variance, expected_variance)
        self.assertTrue(
            np.all(
                moments.effective_degrees_of_freedom
                <= 2.0 * grid.mode_count + 1.0e-12
            )
        )

    def test_piecewise_constant_residual_psd_closes_band_power(self) -> None:
        edges = np.array([0.10, 0.20, 0.305])
        grid = statistics.positive_fourier_modes_by_bin(
            n_time_samples=100,
            cadence_s=1.0,
            bin_edges_hz=edges,
        )
        residual_power = np.array([3.2, 7.1])

        residual_psd = statistics.piecewise_constant_residual_psd(
            grid,
            residual_power,
        )
        recovered = statistics.integrate_mode_values(grid, residual_psd)

        np.testing.assert_allclose(recovered, residual_power, rtol=0.0, atol=1.0e-14)

    def test_catalog_frequency_support_uses_initial_frequency_and_chirp(self) -> None:
        frequency = np.array([2.0e-3, 3.0e-3])
        frequency_derivative = np.array([2.0e-17, -4.0e-17])
        latitude = np.array([0.0, np.pi / 2.0])
        duration = 31_536_000.0

        lower, upper = statistics.catalog_binary_frequency_support(
            frequency,
            frequency_derivative,
            latitude,
            duration,
        )

        doppler_half_width = (
            statistics.LISA_ORBITAL_SPEED_M_S
            / statistics.C_LIGHT
            * frequency[0]
        )
        self.assertAlmostEqual(lower[0], frequency[0] - doppler_half_width)
        self.assertAlmostEqual(
            upper[0],
            frequency[0]
            + frequency_derivative[0] * duration
            + doppler_half_width,
        )
        self.assertAlmostEqual(
            lower[1], frequency[1] + frequency_derivative[1] * duration
        )
        self.assertAlmostEqual(upper[1], frequency[1])

    def test_catalog_support_profile_conserves_each_source_power(self) -> None:
        edges = np.array([0.10, 0.20, 0.30])
        grid = statistics.positive_fourier_modes_by_bin(
            n_time_samples=100,
            cadence_s=1.0,
            bin_edges_hz=edges,
        )
        lower = np.array([0.135, 0.189, 0.251])
        upper = np.array([0.165, 0.201, 0.2511])
        source_power = np.array([3.0, 5.0, 7.0])

        mode_power = statistics.catalog_support_residual_mode_power(
            grid,
            lower,
            upper,
            source_power,
        )

        self.assertAlmostEqual(float(np.sum(mode_power)), float(np.sum(source_power)))
        np.testing.assert_allclose(
            np.bincount(grid.bin_index, weights=mode_power, minlength=2),
            np.array([5.5, 9.5]),
        )

    def test_catalog_support_profile_has_no_power_outside_source_support(self) -> None:
        edges = np.array([0.01, 0.20, 0.40])
        grid = statistics.positive_fourier_modes_by_bin(
            n_time_samples=10_000,
            cadence_s=1.0,
            bin_edges_hz=edges,
        )
        rng = np.random.default_rng(2)
        lower = rng.uniform(0.02, 0.12, 50_000)
        upper = lower + rng.uniform(0.0, 0.05, lower.size)
        power = 10.0 ** rng.uniform(-48.0, -40.0, lower.size)

        mode_power = statistics.catalog_support_residual_mode_power(
            grid,
            lower_frequency_hz=lower,
            upper_frequency_hz=upper,
            source_power=power,
        )

        self.assertTrue(np.all(mode_power[grid.frequency_hz >= 0.20] == 0.0))

    def test_nearest_mode_profile_conserves_power(self) -> None:
        edges = np.array([0.10, 0.20, 0.30])
        grid = statistics.positive_fourier_modes_by_bin(
            n_time_samples=100,
            cadence_s=1.0,
            bin_edges_hz=edges,
        )
        source_frequency = np.array([0.141, 0.199, 0.251])
        source_power = np.array([3.0, 5.0, 7.0])

        mode_power = statistics.nearest_fourier_mode_residual_power(
            grid,
            source_frequency,
            source_power,
        )

        self.assertAlmostEqual(float(np.sum(mode_power)), float(np.sum(source_power)))
        np.testing.assert_allclose(
            np.bincount(grid.bin_index, weights=mode_power, minlength=2),
            np.array([3.0, 12.0]),
        )

    def test_arbitrary_residual_mode_profile_closes_band_power(self) -> None:
        edges = np.array([0.10, 0.20, 0.30])
        grid = statistics.positive_fourier_modes_by_bin(
            n_time_samples=100,
            cadence_s=1.0,
            bin_edges_hz=edges,
        )
        instrument = np.full(grid.frequency_hz.size, 2.0)
        template = np.full(grid.frequency_hz.size, 3.0)
        residual_mode_power = np.zeros(grid.frequency_hz.size)
        residual_mode_power[[4, 9, 14]] = np.array([1.0, 2.0, 4.0])

        model = statistics.build_gaussian_catalog_bandpower_model_from_mode_power(
            grid,
            instrument,
            template,
            residual_mode_power,
            omega0=0.5,
            beta=1.0,
        )

        expected_residual = np.bincount(
            grid.bin_index,
            weights=residual_mode_power,
            minlength=2,
        )
        np.testing.assert_allclose(model.residual_power, expected_residual)
        np.testing.assert_allclose(
            model.mean,
            model.instrument_power + 0.5 * model.sgwb_template + expected_residual,
        )

    def test_fixed_residual_endpoint_removes_only_residual_self_noise(self) -> None:
        edges = np.array([0.10, 0.20, 0.30])
        grid = statistics.positive_fourier_modes_by_bin(
            n_time_samples=100,
            cadence_s=1.0,
            bin_edges_hz=edges,
        )
        instrument = 1.5 + grid.frequency_hz
        template = 0.8 + 0.5 * grid.frequency_hz
        residual_mode_power = np.zeros(grid.frequency_hz.size)
        residual_mode_power[[4, 9, 14]] = np.array([1.0, 2.0, 4.0])
        omega0 = 0.5
        beta = 1.2

        gaussian = statistics.build_gaussian_catalog_bandpower_model_from_mode_power(
            grid,
            instrument,
            template,
            residual_mode_power,
            omega0=omega0,
            beta=beta,
        )
        fixed = statistics.build_fixed_residual_catalog_bandpower_model_from_mode_power(
            grid,
            instrument,
            template,
            residual_mode_power,
            omega0=omega0,
            beta=beta,
        )

        np.testing.assert_allclose(fixed.mean, gaussian.mean)
        np.testing.assert_allclose(fixed.residual_power, gaussian.residual_power)
        expected_self_noise = np.bincount(
            grid.bin_index,
            weights=(beta * residual_mode_power) ** 2,
            minlength=grid.mode_count.size,
        )
        np.testing.assert_allclose(
            gaussian.variance - fixed.variance,
            expected_self_noise,
            rtol=1.0e-14,
            atol=1.0e-14,
        )
        self.assertTrue(np.all(fixed.variance <= gaussian.variance))

    def test_uniform_random_phase_model_applies_finite_source_correction(self) -> None:
        edges = np.array([0.10, 0.20, 0.30])
        grid = statistics.positive_fourier_modes_by_bin(
            n_time_samples=100,
            cadence_s=1.0,
            bin_edges_hz=edges,
        )
        instrument = 1.5 + grid.frequency_hz
        template = 0.8 + 0.5 * grid.frequency_hz
        residual_band_power = np.array([3.0, 7.0])
        concentration = np.array([0.25, 0.60])
        omega0 = 0.5
        beta = 1.2

        gaussian = statistics.build_gaussian_catalog_bandpower_model(
            grid,
            instrument,
            template,
            residual_band_power,
            omega0=omega0,
            beta=beta,
        )
        random_phase = statistics.build_uniform_random_phase_catalog_bandpower_model(
            grid,
            instrument,
            template,
            residual_band_power,
            concentration,
            omega0=omega0,
            beta=beta,
        )
        residual_mode_power = (
            grid.fourier_spacing_hz
            * statistics.piecewise_constant_residual_psd(grid, residual_band_power)
        )
        expected_correction = np.bincount(
            grid.bin_index,
            weights=(
                concentration[grid.bin_index]
                * (beta * residual_mode_power) ** 2
            ),
            minlength=grid.mode_count.size,
        )

        np.testing.assert_allclose(random_phase.mean, gaussian.mean)
        np.testing.assert_allclose(random_phase.residual_power, gaussian.residual_power)
        np.testing.assert_allclose(
            gaussian.variance - random_phase.variance,
            expected_correction,
            rtol=1.0e-14,
            atol=1.0e-14,
        )

    def test_uniform_random_phase_concentration_recovers_covariance_endpoints(self) -> None:
        edges = np.array([0.10, 0.20, 0.30])
        grid = statistics.positive_fourier_modes_by_bin(
            n_time_samples=100,
            cadence_s=1.0,
            bin_edges_hz=edges,
        )
        instrument = np.full(grid.frequency_hz.size, 2.0)
        template = np.full(grid.frequency_hz.size, 3.0)
        residual_band_power = np.array([3.0, 7.0])
        residual_mode_power = (
            grid.fourier_spacing_hz
            * statistics.piecewise_constant_residual_psd(grid, residual_band_power)
        )

        gaussian = statistics.build_gaussian_catalog_bandpower_model(
            grid,
            instrument,
            template,
            residual_band_power,
            omega0=0.5,
        )
        fixed = statistics.build_fixed_residual_catalog_bandpower_model_from_mode_power(
            grid,
            instrument,
            template,
            residual_mode_power,
            omega0=0.5,
        )
        zero_concentration = (
            statistics.build_uniform_random_phase_catalog_bandpower_model(
                grid,
                instrument,
                template,
                residual_band_power,
                np.zeros(2),
                omega0=0.5,
            )
        )
        unit_concentration = (
            statistics.build_uniform_random_phase_catalog_bandpower_model(
                grid,
                instrument,
                template,
                residual_band_power,
                np.ones(2),
                omega0=0.5,
            )
        )

        np.testing.assert_allclose(zero_concentration.variance, gaussian.variance)
        np.testing.assert_allclose(unit_concentration.variance, fixed.variance)

    def test_uniform_sgwb_amplitude_sweep_recomputes_each_fiducial_model(self) -> None:
        edges = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
        grid = statistics.positive_fourier_modes_by_bin(
            n_time_samples=100,
            cadence_s=1.0,
            bin_edges_hz=edges,
        )
        instrument = 1.5 + grid.frequency_hz
        template = 0.8 + 0.5 * grid.frequency_hz
        residual_band_power = np.array([3.0, 4.5, 2.5, 7.0])
        concentration = np.array([0.25, 0.40, 0.15, 0.60])
        omega0_values = np.array([0.2, 0.5])
        beta = 1.2

        rows = statistics.uniform_sgwb_amplitude_sweep(
            grid,
            instrument,
            template,
            residual_band_power,
            concentration,
            omega0_values,
            beta_fiducial=beta,
            uncertainty_threshold=1.1,
        )

        self.assertEqual(len(rows), 2 * omega0_values.size)
        self.assertEqual(
            {row["covariance_treatment"] for row in rows},
            {
                "gaussian_residual_amplitudes",
                "finite_source_random_phase_diagonal",
            },
        )
        for omega0 in omega0_values:
            models = {
                "gaussian_residual_amplitudes": (
                    statistics.build_gaussian_catalog_bandpower_model(
                        grid,
                        instrument,
                        template,
                        residual_band_power,
                        omega0=omega0,
                        beta=beta,
                    )
                ),
                "finite_source_random_phase_diagonal": (
                    statistics.build_uniform_random_phase_catalog_bandpower_model(
                        grid,
                        instrument,
                        template,
                        residual_band_power,
                        concentration,
                        omega0=omega0,
                        beta=beta,
                    )
                ),
            }
            for treatment, model in models.items():
                expected = statistics.two_parameter_fisher_summary(
                    model.sgwb_template,
                    model.residual_power,
                    model.variance,
                    omega0=omega0,
                    beta_fiducial=beta,
                    uncertainty_threshold=1.1,
                )
                actual = next(
                    row
                    for row in rows
                    if row["covariance_treatment"] == treatment
                    and row["omega0_fiducial"] == omega0
                )
                for key in (
                    "F_oo",
                    "F_ob",
                    "F_bb",
                    "sigma_omega_fixed",
                    "sigma_omega_marginalized",
                    "no_prior_inflation",
                    "sigma_beta_for_threshold",
                ):
                    self.assertAlmostEqual(actual[key], expected[key])
                self.assertAlmostEqual(
                    actual["inverse_fractional_fixed_uncertainty"],
                    omega0 / expected["sigma_omega_fixed"],
                )
                self.assertAlmostEqual(
                    actual["marginalized_amplitude_to_uncertainty_ratio"],
                    omega0 / expected["sigma_omega_marginalized"],
                )
                expected_noise = statistics.instrument_noise_normalization_summary(
                    model.sgwb_template,
                    model.residual_power,
                    model.instrument_power,
                    model.variance,
                    omega0=omega0,
                    noise_normalization_prior_sigma=np.inf,
                    uncertainty_threshold=1.1,
                )
                self.assertAlmostEqual(
                    actual[
                        "instrument_noise_normalization_marginalized_"
                        "no_prior_inflation"
                    ],
                    expected_noise["no_prior_inflation"],
                )
                self.assertAlmostEqual(
                    actual[
                        "instrument_noise_normalization_marginalized_"
                        "fixed_beta_ratio"
                    ],
                    expected_noise["snr_omega_fixed_beta"],
                )
                self.assertAlmostEqual(
                    actual[
                        "instrument_noise_normalization_and_beta_"
                        "marginalized_ratio"
                    ],
                    expected_noise["snr_omega_fixed_beta"]
                    / expected_noise["no_prior_inflation"],
                )
                self.assertAlmostEqual(
                    actual["uncertainty_increase_percent"],
                    100.0 * (expected["no_prior_inflation"] - 1.0),
                )

    def test_fixed_and_gaussian_endpoints_agree_without_residual_power(self) -> None:
        edges = np.array([0.10, 0.20, 0.30])
        grid = statistics.positive_fourier_modes_by_bin(
            n_time_samples=100,
            cadence_s=1.0,
            bin_edges_hz=edges,
        )
        instrument = np.full(grid.frequency_hz.size, 2.0)
        template = np.full(grid.frequency_hz.size, 3.0)
        residual_mode_power = np.zeros(grid.frequency_hz.size)

        gaussian = statistics.build_gaussian_catalog_bandpower_model_from_mode_power(
            grid,
            instrument,
            template,
            residual_mode_power,
            omega0=0.5,
        )
        fixed = statistics.build_fixed_residual_catalog_bandpower_model_from_mode_power(
            grid,
            instrument,
            template,
            residual_mode_power,
            omega0=0.5,
        )

        np.testing.assert_allclose(fixed.mean, gaussian.mean)
        np.testing.assert_allclose(fixed.variance, gaussian.variance)
        np.testing.assert_allclose(
            fixed.effective_degrees_of_freedom,
            gaussian.effective_degrees_of_freedom,
        )

    def test_fisher_uses_gaussian_periodogram_variance_only(self) -> None:
        result = statistics.two_parameter_fisher_summary(
            sgwb_template=np.array([2.0, 1.0, 0.4]),
            residual_power=np.array([0.5, 0.2, 0.0]),
            bandpower_variance=np.array([4.0, 2.0, 1.0]),
            omega0=1.0,
            uncertainty_threshold=1.1,
        )

        self.assertNotIn("fourth_moment_sigma_fixed_over_gaussian", result)
        self.assertNotIn("no_prior_inflation_fourth_moment", result)
        self.assertGreater(result["F_oo"], 0.0)
        self.assertGreater(result["F_bb"], 0.0)

    def test_omitted_residual_shift_scales_with_its_fiducial_amplitude(self) -> None:
        common = {
            "sgwb_template": np.array([2.0, 1.0, 0.4]),
            "residual_power": np.array([0.5, 0.2, 0.0]),
            "bandpower_variance": np.array([4.0, 2.0, 1.0]),
            "omega0": 1.0,
        }

        unit = statistics.two_parameter_fisher_summary(
            **common,
            beta_fiducial=1.0,
        )
        doubled = statistics.two_parameter_fisher_summary(
            **common,
            beta_fiducial=2.0,
        )

        self.assertAlmostEqual(
            doubled["delta_omega_omitted_residual"],
            2.0 * unit["delta_omega_omitted_residual"],
        )
        self.assertAlmostEqual(doubled["F_ob"], unit["F_ob"])

    def test_fixed_noise_limit_matches_two_parameter_fisher_result(self) -> None:
        common = {
            "sgwb_template": np.array([2.0, 1.0, 0.4]),
            "residual_power": np.array([0.5, 0.2, 0.0]),
            "bandpower_variance": np.array([4.0, 2.0, 1.0]),
            "omega0": 1.0,
            "uncertainty_threshold": 1.1,
        }
        two_parameter = statistics.two_parameter_fisher_summary(**common)

        with_noise = statistics.instrument_noise_normalization_summary(
            **common,
            instrument_power=np.array([1.0, 0.8, 0.6]),
            noise_normalization_prior_sigma=0.0,
        )

        self.assertAlmostEqual(
            with_noise["no_prior_inflation"],
            two_parameter["no_prior_inflation"],
        )
        self.assertAlmostEqual(
            with_noise["sigma_beta_for_threshold"],
            two_parameter["sigma_beta_for_threshold"],
        )

    def test_noise_normalization_is_removed_by_schur_complement(self) -> None:
        result = statistics.instrument_noise_normalization_summary(
            sgwb_template=np.array([2.0, 1.0, 0.4]),
            residual_power=np.array([0.5, 0.2, 0.1]),
            instrument_power=np.array([1.0, 0.8, 0.6]),
            bandpower_variance=np.array([4.0, 2.0, 1.0]),
            omega0=1.0,
            noise_normalization_prior_sigma=0.01,
            uncertainty_threshold=1.1,
        )

        self.assertGreater(result["F_oo_after_noise_marginalization"], 0.0)
        self.assertGreater(result["F_bb_after_noise_marginalization"], 0.0)
        self.assertGreaterEqual(result["no_prior_inflation"], 1.0)
        self.assertEqual(result["noise_normalization_prior_sigma"], 0.01)

    def test_fisher_rejects_nonphysical_powers_and_invalid_variances(self) -> None:
        common = {
            "sgwb_template": np.array([2.0, 1.0]),
            "residual_power": np.array([0.5, 0.2]),
            "bandpower_variance": np.array([4.0, 2.0]),
            "omega0": 1.0,
        }

        with self.assertRaises(ValueError):
            statistics.two_parameter_fisher_summary(
                **{**common, "residual_power": np.array([0.5, -0.2])}
            )
        with self.assertRaises(ValueError):
            statistics.two_parameter_fisher_summary(
                **{**common, "bandpower_variance": np.array([4.0, 0.0])}
            )


if __name__ == "__main__":
    unittest.main()
