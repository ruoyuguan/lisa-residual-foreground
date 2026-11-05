from __future__ import annotations

import numpy as np


LISA_LW_SINGLE_MICHELSON_AVERAGED_RESPONSE = 3.0 / 20.0
LISA_LW_AE_AVERAGED_RESPONSE = 2.0 * LISA_LW_SINGLE_MICHELSON_AVERAGED_RESPONSE


def ldc_gb_polarization_amplitudes(
    amplitude: np.ndarray,
    inclination: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the source-frame polarization amplitudes used for Sangria GBs.

    The LDC convention is h_plus = A (1 + cos(iota)^2) cos(Phi) and
    h_cross = 2 A cos(iota) sin(Phi). The returned arrays are the coefficients
    multiplying the cosine and sine phases, respectively.
    """

    amp = np.asarray(amplitude, dtype=float)
    cosi = np.cos(np.asarray(inclination, dtype=float))
    return amp * (1.0 + cosi**2), 2.0 * amp * cosi


def ldc_gb_inclination_power_factor(inclination: np.ndarray) -> np.ndarray:
    """Return (h_plus_amp^2 + h_cross_amp^2) / A^2."""

    cosi = np.cos(np.asarray(inclination, dtype=float))
    return ldc_gb_power_factor_from_cosine(cosi)


def ldc_gb_power_factor_from_cosine(cos_inclination: np.ndarray) -> np.ndarray:
    """Return the Sangria GB polarization-power factor from cos(iota)."""

    cosi = np.asarray(cos_inclination, dtype=float)
    return (1.0 + cosi**2) ** 2 + 4.0 * cosi**2


def phase_averaged_projected_power(
    amplitude: np.ndarray,
    inclination: np.ndarray,
    fplus: np.ndarray,
    fcross: np.ndarray,
) -> np.ndarray:
    """Return phase-averaged strain power for one detector response."""

    hplus, hcross = ldc_gb_polarization_amplitudes(amplitude, inclination)
    return 0.5 * (
        (np.asarray(fplus) * hplus) ** 2
        + (np.asarray(fcross) * hcross) ** 2
    )


def equivalent_incident_strain_power(detector_output_power: np.ndarray) -> np.ndarray:
    """Convert summed static A,E output power to equivalent incident strain power."""

    return np.asarray(detector_output_power, dtype=float) / LISA_LW_AE_AVERAGED_RESPONSE


def equivalent_single_michelson_incident_strain_power(
    detector_output_power: np.ndarray,
) -> np.ndarray:
    """Convert one static Michelson output power to incident strain power."""

    return (
        np.asarray(detector_output_power, dtype=float)
        / LISA_LW_SINGLE_MICHELSON_AVERAGED_RESPONSE
    )
