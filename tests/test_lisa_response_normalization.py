import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "repro" / "repro_scripts"
sys.path.insert(0, str(SCRIPTS))

from galactic_binary_power import phase_averaged_projected_power
import lisa_catalog_statistics as statistics


def load_baseline_module():
    path = SCRIPTS / "20_compute_catalog_bandpower_baseline.py"
    spec = importlib.util.spec_from_file_location("catalog_bandpower_baseline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


baseline = load_baseline_module()


class LisaResponseNormalizationTests(unittest.TestCase):
    def test_static_x_uses_the_sangria_reference_geometry(self) -> None:
        positions = statistics.lisa_equal_arm_reference_positions()
        detector = statistics.static_lisa_michelson_x_tensor()

        self.assertAlmostEqual(
            statistics.constellation_plane_inclination(positions),
            np.pi / 3.0,
            places=14,
        )
        self.assertAlmostEqual(
            statistics.isotropic_single_polarization_response(detector),
            3.0 / 20.0,
            places=15,
        )

    def test_catalog_power_is_response_normalized_once(self) -> None:
        detector = statistics.static_lisa_michelson_x_tensor()
        amplitude = np.array([2.0])
        inclination = np.array([0.7])
        longitude = np.array([1.1])
        latitude = np.array([-0.4])
        polarization = np.array([0.3])
        f_plus, f_cross = baseline.antenna_patterns_vector(
            longitude,
            latitude,
            polarization,
            detector,
        )
        detector_output = phase_averaged_projected_power(
            amplitude,
            inclination,
            f_plus,
            f_cross,
        )

        equivalent_incident = baseline.response_normalized_source_power(
            amplitude,
            inclination,
            longitude,
            latitude,
            polarization,
            detector,
        )

        np.testing.assert_allclose(
            equivalent_incident,
            detector_output / (3.0 / 20.0),
            rtol=1.0e-15,
        )

    def test_instrument_noise_uses_the_same_response_normalization(self) -> None:
        frequency = np.array([1.0e-3, 3.0e-3, 6.0e-3])
        arm_length = 2.5e9
        optical_path = (1.5e-11) ** 2 * (1.0 + (2.0e-3 / frequency) ** 4)
        acceleration = (
            (3.0e-15) ** 2
            * (1.0 + (0.4e-3 / frequency) ** 2)
            * (1.0 + (frequency / 8.0e-3) ** 4)
        )
        detector_output = (
            optical_path
            + 4.0 * acceleration / (2.0 * np.pi * frequency) ** 4
        ) / arm_length**2

        equivalent_incident = statistics.single_michelson_lwa_equivalent_strain_psd(
            frequency,
            arm_length,
        )

        np.testing.assert_allclose(
            equivalent_incident,
            detector_output / (3.0 / 20.0),
            rtol=1.0e-15,
        )

    def test_single_x_noise_is_twice_the_two_channel_lwa_sensitivity(self) -> None:
        frequency = np.array([1.0e-3, 3.0e-3, 6.0e-3])
        arm_length = 2.5e9
        optical_path = (1.5e-11) ** 2 * (1.0 + (2.0e-3 / frequency) ** 4)
        acceleration = (
            (3.0e-15) ** 2
            * (1.0 + (0.4e-3 / frequency) ** 2)
            * (1.0 + (frequency / 8.0e-3) ** 4)
        )
        displacement_noise = (
            optical_path
            + 4.0 * acceleration / (2.0 * np.pi * frequency) ** 4
        )
        two_channel_lwa_sensitivity = 10.0 * displacement_noise / (
            3.0 * arm_length**2
        )

        single_x = statistics.single_michelson_lwa_equivalent_strain_psd(
            frequency,
            arm_length,
        )

        np.testing.assert_allclose(
            single_x,
            2.0 * two_channel_lwa_sensitivity,
            rtol=1.0e-15,
        )

    def test_antenna_patterns_numerically_reproduce_three_over_twenty(self) -> None:
        detector = statistics.static_lisa_michelson_x_tensor()
        cos_colatitude, colatitude_weights = np.polynomial.legendre.leggauss(20)
        longitude = 2.0 * np.pi * np.arange(40, dtype=float) / 40.0
        polarization = np.pi * np.arange(20, dtype=float) / 20.0
        cos_theta, lam, psi = np.meshgrid(
            cos_colatitude,
            longitude,
            polarization,
            indexing="ij",
        )
        latitude = np.arcsin(cos_theta.ravel())
        lam = lam.ravel()
        psi = psi.ravel()
        f_plus, f_cross = baseline.antenna_patterns_vector(
            lam,
            latitude,
            psi,
            detector,
        )
        weights = np.broadcast_to(
            colatitude_weights[:, None, None],
            cos_theta.shape,
        ).ravel()
        weights = weights / np.sum(weights)

        self.assertAlmostEqual(np.sum(weights * f_plus**2), 3.0 / 20.0, places=14)
        self.assertAlmostEqual(np.sum(weights * f_cross**2), 3.0 / 20.0, places=14)

    def test_twelve_phase_antenna_power_quadrature_is_converged(self) -> None:
        rng = np.random.default_rng(20260721)
        n_sources = 200
        amplitude = 10.0 ** rng.uniform(-24.0, -20.0, n_sources)
        inclination = rng.uniform(0.0, np.pi, n_sources)
        longitude = rng.uniform(0.0, 2.0 * np.pi, n_sources)
        latitude = np.arcsin(rng.uniform(-1.0, 1.0, n_sources))
        polarization = rng.uniform(0.0, np.pi, n_sources)

        def averaged_power(n_phase: int, phase_offset: float = 0.0) -> np.ndarray:
            detectors = tuple(
                statistics.static_lisa_michelson_x_tensor(
                    2.0 * np.pi * (index + phase_offset) / n_phase,
                    0.0,
                )
                for index in range(n_phase)
            )
            return baseline.orbit_averaged_response_normalized_source_power(
                amplitude,
                inclination,
                longitude,
                latitude,
                polarization,
                detectors,
            )

        np.testing.assert_allclose(
            averaged_power(12),
            averaged_power(24),
            rtol=3.0e-15,
            atol=0.0,
        )
        np.testing.assert_allclose(
            averaged_power(12),
            averaged_power(12, phase_offset=0.37),
            rtol=3.0e-15,
            atol=0.0,
        )

    def test_active_baseline_has_no_fourth_moment_covariance_path(self) -> None:
        source = (
            SCRIPTS / "20_compute_catalog_bandpower_baseline.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("D_covariance_mismatch", source)
        self.assertNotIn("fourth_moment_sigma", source)
        self.assertIn("used_in_bandpower_covariance\": False", source)

    def test_fourth_moment_check_requires_at_least_four_draws(self) -> None:
        with self.assertRaises(ValueError):
            baseline.monte_carlo_random_phase(
                np.array([1.0, 0.5]),
                n_draws=3,
                batch_size=3,
                max_batch_elements=100,
                rng=np.random.default_rng(1),
            )

    def test_bin_width_sensitivity_uses_the_active_frequency_statistic(self) -> None:
        source = (
            SCRIPTS / "23_run_bin_width_sensitivity.py"
        ).read_text(encoding="utf-8")

        self.assertIn("20_compute_catalog_bandpower_baseline.py", source)
        self.assertNotIn("20_compute_all_bin_lwa_AE_baseline.py", source)
        self.assertNotIn("dof-efficiency", source)
        self.assertNotIn("entropy", source.lower())
        self.assertNotIn("gaussian_sigma_over_fourth_moment_sigma", source)
        self.assertIn("--skip-random-phase-check", source)
        self.assertIn("--output-residual-profile-sensitivity", source)
        self.assertIn("--output-covariance-ensemble-sensitivity", source)
        self.assertIn("--output-residual-mode-profiles", source)
        self.assertIn("--output-frequency-support-audit", source)
        self.assertIn("--output-noise-normalization-sensitivity", source)



if __name__ == "__main__":
    unittest.main()
