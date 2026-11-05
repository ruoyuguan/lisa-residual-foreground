import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "repro" / "repro_scripts"))

from galactic_binary_power import (  # noqa: E402
    ldc_gb_inclination_power_factor,
    ldc_gb_polarization_amplitudes,
)


class GalacticBinaryPowerTests(unittest.TestCase):
    def test_face_on_polarization_amplitudes_follow_ldc_convention(self) -> None:
        amplitude = np.array([3.0])
        inclination = np.array([0.0])

        hplus, hcross = ldc_gb_polarization_amplitudes(amplitude, inclination)

        np.testing.assert_allclose(hplus, [6.0])
        np.testing.assert_allclose(hcross, [6.0])

    def test_edge_on_polarization_amplitudes_follow_ldc_convention(self) -> None:
        amplitude = np.array([3.0])
        inclination = np.array([np.pi / 2.0])

        hplus, hcross = ldc_gb_polarization_amplitudes(amplitude, inclination)

        np.testing.assert_allclose(hplus, [3.0], atol=1.0e-14)
        np.testing.assert_allclose(hcross, [0.0], atol=1.0e-14)

    def test_inclination_power_factor_matches_phase_averaged_polarizations(self) -> None:
        amplitude = np.array([2.5, 7.0])
        inclination = np.array([0.7, 1.3])

        hplus, hcross = ldc_gb_polarization_amplitudes(amplitude, inclination)
        direct = hplus**2 + hcross**2
        factored = amplitude**2 * ldc_gb_inclination_power_factor(inclination)

        np.testing.assert_allclose(direct, factored, rtol=1.0e-15)

    def test_ldc_power_is_four_times_the_previous_half_amplitude_convention(self) -> None:
        amplitude = np.array([2.5, 7.0])
        inclination = np.array([0.7, 1.3])
        cosi = np.cos(inclination)
        old_factor = ((1.0 + cosi**2) ** 2) / 4.0 + cosi**2

        corrected = amplitude**2 * ldc_gb_inclination_power_factor(inclination)
        previous = amplitude**2 * old_factor

        np.testing.assert_allclose(corrected, 4.0 * previous, rtol=1.0e-15)


if __name__ == "__main__":
    unittest.main()
