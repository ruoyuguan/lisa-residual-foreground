"""Physical response and coarse-band statistics for the catalog calculation.

The module defines long-wavelength LISA Michelson tensors at orbital snapshots
and the frequency-domain statistic used by the active catalog pipeline.  The
pipeline averages antenna power over orbital phase; it does not construct a
time-dependent TDI likelihood.  Independent positive-frequency periodogram
ordinates from the one-year Sangria record are integrated into the adopted
coarse frequency bins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


C_LIGHT = 299_792_458.0
LISA_ORBITAL_SPEED_M_S = 29_780.0
MPC_M = 3.085_677_581_491_367_3e22
LISA_LW_SINGLE_MICHELSON_AVERAGED_RESPONSE = 3.0 / 20.0
LISA_LW_TWO_CHANNEL_SUMMED_RESPONSE = 2.0 * LISA_LW_SINGLE_MICHELSON_AVERAGED_RESPONSE


@dataclass(frozen=True)
class FourierBandGrid:
    """Independent positive-frequency modes assigned to coarse bands."""

    frequency_hz: np.ndarray
    bin_index: np.ndarray
    mode_count: np.ndarray
    fourier_spacing_hz: float
    bin_edges_hz: np.ndarray


@dataclass(frozen=True)
class GaussianBandpowerMoments:
    """Mean, variance, and derived degrees of freedom of coarse band powers."""

    mean: np.ndarray
    variance: np.ndarray
    effective_degrees_of_freedom: np.ndarray


@dataclass(frozen=True)
class CatalogBandpowerModel:
    """Mode-integrated components of the fixed-covariance catalog model."""

    instrument_power: np.ndarray
    sgwb_template: np.ndarray
    residual_power: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    effective_degrees_of_freedom: np.ndarray


def alpha_column_suffix(alpha: float) -> str:
    """Return a stable column suffix for a power-law spectral index."""

    value = float(alpha)
    if not np.isfinite(value):
        raise ValueError("alpha must be finite.")
    if value.is_integer():
        integer = int(value)
        return f"m{abs(integer)}" if integer < 0 else f"p{integer}"
    text = f"{abs(value):.8g}".replace(".", "p")
    return f"m{text}" if value < 0.0 else f"p{text}"


def lisa_equal_arm_reference_positions(
    orbital_phase_rad: float = 0.0,
    constellation_rotation_rad: float = 0.0,
) -> np.ndarray:
    """Return dimensionless LISA spacecraft positions at a chosen orbital phase.

    The relative positions follow the first-order equal-arm orbit used for the
    LISA Data Challenge Sangria data set.  A common guiding-center position and
    the overall arm-length scale cancel from the detector tensor, so only the
    relative first-order terms are retained here.  The default phases are the
    ``kappa=lambda=0`` initial conditions specified for this LDC orbit.
    """

    alpha = float(orbital_phase_rad)
    beta = (
        np.arange(3, dtype=float) * (2.0 * np.pi / 3.0)
        + float(constellation_rotation_rad)
    )
    sin_alpha = np.sin(alpha)
    cos_alpha = np.cos(alpha)
    sin_beta = np.sin(beta)
    cos_beta = np.cos(beta)

    x = (
        sin_alpha * cos_alpha * sin_beta
        - (1.0 + sin_alpha**2) * cos_beta
    )
    y = (
        sin_alpha * cos_alpha * cos_beta
        - (1.0 + cos_alpha**2) * sin_beta
    )
    z = -np.sqrt(3.0) * np.cos(alpha - beta)
    positions = np.column_stack((x, y, z))
    return positions - positions.mean(axis=0, keepdims=True)


def constellation_plane_inclination(positions: np.ndarray) -> float:
    """Return the angle between the constellation and ecliptic planes."""

    points = np.asarray(positions, dtype=float)
    if points.shape != (3, 3):
        raise ValueError("positions must have shape (3, 3).")
    arm_12 = points[1] - points[0]
    arm_13 = points[2] - points[0]
    normal = np.cross(arm_12, arm_13)
    norm = np.linalg.norm(normal)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("The spacecraft positions do not define a plane.")
    cosine = abs(float(normal[2])) / norm
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def michelson_tensor_from_positions(
    positions: np.ndarray,
    vertex: int = 0,
) -> np.ndarray:
    """Return the long-wavelength Michelson tensor at one spacecraft."""

    points = np.asarray(positions, dtype=float)
    if points.shape != (3, 3):
        raise ValueError("positions must have shape (3, 3).")
    if vertex not in (0, 1, 2):
        raise ValueError("vertex must identify one of the three spacecraft.")
    others = [index for index in range(3) if index != vertex]
    arms = []
    for other in others:
        arm = points[other] - points[vertex]
        length = np.linalg.norm(arm)
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError("A Michelson arm has zero or invalid length.")
        arms.append(arm / length)
    return 0.5 * (np.outer(arms[0], arms[0]) - np.outer(arms[1], arms[1]))


def static_lisa_michelson_x_tensor(
    orbital_phase_rad: float = 0.0,
    constellation_rotation_rad: float = 0.0,
) -> np.ndarray:
    """Return a long-wavelength Michelson-X tensor at one orbital snapshot.

    Zero orbital and constellation phases give the Sangria reference epoch.
    """

    positions = lisa_equal_arm_reference_positions(
        orbital_phase_rad=orbital_phase_rad,
        constellation_rotation_rad=constellation_rotation_rad,
    )
    return michelson_tensor_from_positions(positions, vertex=0)


def isotropic_single_polarization_response(detector: np.ndarray) -> float:
    """Return the long-wavelength sky average for one tensor polarization."""

    tensor = np.asarray(detector, dtype=float)
    if tensor.shape != (3, 3):
        raise ValueError("detector must have shape (3, 3).")
    return float((2.0 / 5.0) * np.sum(tensor * tensor))


def _first_mode_at_or_above(frequency_hz: float, spacing_hz: float) -> int:
    ratio = float(frequency_hz) / float(spacing_hz)
    return int(np.ceil(np.nextafter(ratio, -np.inf)))


def positive_fourier_modes_by_bin(
    n_time_samples: int,
    cadence_s: float,
    bin_edges_hz: np.ndarray,
) -> FourierBandGrid:
    """Assign independent, non-DC, non-Nyquist Fourier modes to coarse bins."""

    if int(n_time_samples) != n_time_samples or n_time_samples < 3:
        raise ValueError("n_time_samples must be an integer of at least three.")
    if not np.isfinite(cadence_s) or cadence_s <= 0.0:
        raise ValueError("cadence_s must be positive and finite.")
    edges = np.asarray(bin_edges_hz, dtype=float)
    if (
        edges.ndim != 1
        or edges.size < 2
        or not np.all(np.isfinite(edges))
        or not np.all(np.diff(edges) > 0.0)
    ):
        raise ValueError("bin_edges_hz must be a finite, increasing 1D array.")

    n_samples = int(n_time_samples)
    spacing = 1.0 / (n_samples * float(cadence_s))
    positive_stop = (n_samples + 1) // 2
    nyquist_hz = 0.5 / float(cadence_s)
    if edges[0] <= 0.0 or edges[-1] >= nyquist_hz:
        raise ValueError("Coarse bins must exclude the DC and Nyquist frequencies.")

    frequencies = []
    assignments = []
    mode_count = np.empty(edges.size - 1, dtype=np.int64)
    for bin_index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        first = max(1, _first_mode_at_or_above(left, spacing))
        stop = min(positive_stop, _first_mode_at_or_above(right, spacing))
        modes = np.arange(first, stop, dtype=np.int64)
        mode_count[bin_index] = modes.size
        if modes.size:
            frequencies.append(modes.astype(float) * spacing)
            assignments.append(np.full(modes.size, bin_index, dtype=np.int64))

    if np.any(mode_count <= 0):
        raise ValueError("Every coarse frequency bin must contain a Fourier mode.")
    return FourierBandGrid(
        frequency_hz=np.concatenate(frequencies),
        bin_index=np.concatenate(assignments),
        mode_count=mode_count,
        fourier_spacing_hz=spacing,
        bin_edges_hz=edges.copy(),
    )


def _validated_mode_values(grid: FourierBandGrid, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != grid.frequency_hz.shape:
        raise ValueError("Mode values must have one entry per Fourier frequency.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Mode values must be finite.")
    return array


def integrate_mode_values(grid: FourierBandGrid, values: np.ndarray) -> np.ndarray:
    """Integrate a one-sided spectral density over each coarse band."""

    array = _validated_mode_values(grid, values)
    return np.bincount(
        grid.bin_index,
        weights=grid.fourier_spacing_hz * array,
        minlength=grid.mode_count.size,
    )


def integrate_gaussian_periodogram(
    grid: FourierBandGrid,
    one_sided_psd: np.ndarray,
) -> GaussianBandpowerMoments:
    """Return moments of summed independent one-sided periodogram ordinates.

    For a real stationary Gaussian process, each positive-frequency
    periodogram ordinate away from DC and Nyquist is exponential with variance
    equal to the square of its mean.  The coarse-band variance is therefore the
    sum of the squared mode contributions; no common sample count is assumed.
    """

    psd = _validated_mode_values(grid, one_sided_psd)
    if np.any(psd < 0.0):
        raise ValueError("A one-sided power spectral density cannot be negative.")
    contribution = grid.fourier_spacing_hz * psd
    mean = np.bincount(
        grid.bin_index,
        weights=contribution,
        minlength=grid.mode_count.size,
    )
    variance = np.bincount(
        grid.bin_index,
        weights=contribution**2,
        minlength=grid.mode_count.size,
    )
    effective_dof = np.full_like(mean, np.nan)
    np.divide(2.0 * mean**2, variance, out=effective_dof, where=variance > 0.0)
    return GaussianBandpowerMoments(
        mean=mean,
        variance=variance,
        effective_degrees_of_freedom=effective_dof,
    )


def piecewise_constant_residual_psd(
    grid: FourierBandGrid,
    residual_band_power: np.ndarray,
) -> np.ndarray:
    """Distribute each catalog residual band power uniformly over its modes."""

    residual = np.asarray(residual_band_power, dtype=float)
    if residual.shape != grid.mode_count.shape:
        raise ValueError("residual_band_power must have one entry per coarse bin.")
    if not np.all(np.isfinite(residual)) or np.any(residual < 0.0):
        raise ValueError("Residual band powers must be finite and nonnegative.")
    density_by_bin = residual / (
        grid.mode_count.astype(float) * grid.fourier_spacing_hz
    )
    return density_by_bin[grid.bin_index]


def catalog_binary_frequency_support(
    initial_frequency_hz: np.ndarray,
    frequency_derivative_hz_s: np.ndarray,
    ecliptic_latitude_rad: np.ndarray,
    observation_duration_s: float,
    orbital_speed_m_s: float = LISA_ORBITAL_SPEED_M_S,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a conservative one-year frequency interval for each binary.

    The LDC Galactic-binary phase convention defines ``initial_frequency_hz``
    at ``t=0``.  The intrinsic endpoints are therefore ``f0`` and
    ``f0 + fdot*T``.  The annual Doppler half-width is added at both ends.
    This interval records frequency support only; it is not a coherent LISA
    sideband model.
    """

    frequency, derivative, latitude = np.broadcast_arrays(
        np.asarray(initial_frequency_hz, dtype=float),
        np.asarray(frequency_derivative_hz_s, dtype=float),
        np.asarray(ecliptic_latitude_rad, dtype=float),
    )
    if not np.all(np.isfinite(frequency)) or np.any(frequency <= 0.0):
        raise ValueError("Initial frequencies must be positive and finite.")
    if not np.all(np.isfinite(derivative)) or not np.all(np.isfinite(latitude)):
        raise ValueError("Frequency derivatives and latitudes must be finite.")
    if not np.isfinite(observation_duration_s) or observation_duration_s <= 0.0:
        raise ValueError("observation_duration_s must be positive and finite.")
    if not np.isfinite(orbital_speed_m_s) or orbital_speed_m_s < 0.0:
        raise ValueError("orbital_speed_m_s must be finite and nonnegative.")

    final_frequency = frequency + derivative * float(observation_duration_s)
    doppler_half_width = (
        float(orbital_speed_m_s)
        / C_LIGHT
        * frequency
        * np.abs(np.cos(latitude))
    )
    lower = np.minimum(frequency, final_frequency) - doppler_half_width
    upper = np.maximum(frequency, final_frequency) + doppler_half_width
    return lower, upper


def nearest_fourier_mode_residual_power(
    grid: FourierBandGrid,
    source_frequency_hz: np.ndarray,
    source_power: np.ndarray,
) -> np.ndarray:
    """Place each source power at the nearest retained Fourier frequency."""

    frequency, power = np.broadcast_arrays(
        np.asarray(source_frequency_hz, dtype=float),
        np.asarray(source_power, dtype=float),
    )
    if not np.all(np.isfinite(frequency)):
        raise ValueError("Source frequencies must be finite.")
    if not np.all(np.isfinite(power)) or np.any(power < 0.0):
        raise ValueError("Source powers must be finite and nonnegative.")
    flat_frequency = frequency.reshape(-1)
    flat_power = power.reshape(-1)
    insertion = np.searchsorted(grid.frequency_hz, flat_frequency, side="left")
    right = np.clip(insertion, 0, grid.frequency_hz.size - 1)
    left = np.clip(insertion - 1, 0, grid.frequency_hz.size - 1)
    choose_left = np.abs(flat_frequency - grid.frequency_hz[left]) <= np.abs(
        grid.frequency_hz[right] - flat_frequency
    )
    nearest = np.where(choose_left, left, right)
    mode_power = np.zeros(grid.frequency_hz.size, dtype=float)
    np.add.at(mode_power, nearest, flat_power)
    return mode_power


def catalog_support_residual_mode_power(
    grid: FourierBandGrid,
    lower_frequency_hz: np.ndarray,
    upper_frequency_hz: np.ndarray,
    source_power: np.ndarray,
) -> np.ndarray:
    """Spread each source power uniformly over modes in its frequency interval.

    A source whose interval contains no retained Fourier frequency is placed at
    the nearest mode.  The resulting profile preserves each source's total
    power and is used only to test sensitivity to unresolved frequency
    structure; it is not a coherent sideband waveform.
    """

    lower, upper, power = np.broadcast_arrays(
        np.asarray(lower_frequency_hz, dtype=float),
        np.asarray(upper_frequency_hz, dtype=float),
        np.asarray(source_power, dtype=float),
    )
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("Frequency-support bounds must be finite.")
    if np.any(upper < lower):
        raise ValueError("Every upper frequency bound must exceed its lower bound.")
    if not np.all(np.isfinite(power)) or np.any(power < 0.0):
        raise ValueError("Source powers must be finite and nonnegative.")

    flat_lower = lower.reshape(-1)
    flat_upper = upper.reshape(-1)
    flat_power = power.reshape(-1)
    start = np.searchsorted(grid.frequency_hz, flat_lower, side="left")
    stop = np.searchsorted(grid.frequency_hz, flat_upper, side="right")
    start = np.clip(start, 0, grid.frequency_hz.size)
    stop = np.clip(stop, 0, grid.frequency_hz.size)
    count = stop - start

    mode_power = np.zeros(grid.frequency_hz.size, dtype=float)
    supported_mode = np.zeros(grid.frequency_hz.size, dtype=bool)
    covered = count > 0
    if np.any(covered):
        increment = flat_power[covered] / count[covered]
        difference = np.zeros(grid.frequency_hz.size + 1, dtype=float)
        support_difference = np.zeros(grid.frequency_hz.size + 1, dtype=np.int64)
        np.add.at(difference, start[covered], increment)
        np.add.at(difference, stop[covered], -increment)
        np.add.at(support_difference, start[covered], 1)
        np.add.at(support_difference, stop[covered], -1)
        mode_power += np.cumsum(difference[:-1])
        supported_mode |= np.cumsum(support_difference[:-1]) > 0
    if np.any(~covered):
        midpoint = 0.5 * (flat_lower[~covered] + flat_upper[~covered])
        fallback_power = nearest_fourier_mode_residual_power(
            grid,
            midpoint,
            flat_power[~covered],
        )
        mode_power += fallback_power
        supported_mode |= fallback_power > 0.0
    mode_power[~supported_mode] = 0.0
    target_power = float(np.sum(flat_power))
    largest_mode_power = float(np.max(mode_power, initial=0.0))
    negative_tolerance = 1.0e-12 * max(largest_mode_power, np.finfo(float).tiny)
    if np.min(mode_power, initial=0.0) < -negative_tolerance:
        raise FloatingPointError("Range summation produced negative Fourier-mode power.")
    mode_power[mode_power < 0.0] = 0.0
    if target_power > 0.0:
        correction_index = int(np.argmax(mode_power))
        mode_power[correction_index] += target_power - float(np.sum(mode_power))
    np.testing.assert_allclose(
        np.sum(mode_power),
        target_power,
        rtol=5.0e-15,
        atol=0.0,
        err_msg="The catalog frequency profile does not conserve source power.",
    )
    return mode_power


def single_michelson_lwa_equivalent_strain_psd(
    frequency_hz: np.ndarray,
    arm_length_m: float,
) -> np.ndarray:
    """Nominal single-Michelson LISA noise in equivalent incident strain.

    The detector-output noise is divided by the single-channel long-wavelength
    response ``3/20``.  The commonly used ``10/3`` sensitivity prefactor sums
    the responses of two independent low-frequency channels; it is therefore
    smaller by a factor of two and is not the noise PSD of the scalar
    single-channel periodogram used here.
    """

    frequency = np.asarray(frequency_hz, dtype=float)
    if np.any(~np.isfinite(frequency)) or np.any(frequency <= 0.0):
        raise ValueError("frequency_hz must be positive and finite.")
    if not np.isfinite(arm_length_m) or arm_length_m <= 0.0:
        raise ValueError("arm_length_m must be positive and finite.")
    optical_path = (1.5e-11) ** 2 * (1.0 + (2.0e-3 / frequency) ** 4)
    acceleration = (
        (3.0e-15) ** 2
        * (1.0 + (0.4e-3 / frequency) ** 2)
        * (1.0 + (frequency / 8.0e-3) ** 4)
    )
    detector_output = (
        optical_path
        + 4.0 * acceleration / (2.0 * np.pi * frequency) ** 4
    ) / arm_length_m**2
    return detector_output / LISA_LW_SINGLE_MICHELSON_AVERAGED_RESPONSE


def sgwb_strain_psd_per_omega(
    frequency_hz: np.ndarray,
    alpha: float,
    reference_frequency_hz: float,
    h0_km_s_mpc: float,
) -> np.ndarray:
    """Return one-sided SGWB strain PSD per unit reference amplitude."""

    frequency = np.asarray(frequency_hz, dtype=float)
    if np.any(~np.isfinite(frequency)) or np.any(frequency <= 0.0):
        raise ValueError("frequency_hz must be positive and finite.")
    if not np.isfinite(reference_frequency_hz) or reference_frequency_hz <= 0.0:
        raise ValueError("reference_frequency_hz must be positive and finite.")
    if not np.isfinite(h0_km_s_mpc) or h0_km_s_mpc <= 0.0:
        raise ValueError("h0_km_s_mpc must be positive and finite.")
    h0_si = h0_km_s_mpc * 1000.0 / MPC_M
    return (
        3.0
        * h0_si**2
        / (2.0 * np.pi**2)
        * (frequency / reference_frequency_hz) ** float(alpha)
        / frequency**3
    )


def build_gaussian_catalog_bandpower_model(
    grid: FourierBandGrid,
    instrument_psd: np.ndarray,
    sgwb_template_psd: np.ndarray,
    residual_band_power: np.ndarray,
    omega0: float,
    beta: float = 1.0,
) -> CatalogBandpowerModel:
    """Construct the fixed-covariance mean and Gaussian periodogram variance."""

    residual = np.asarray(residual_band_power, dtype=float)
    residual_psd = piecewise_constant_residual_psd(grid, residual)
    residual_mode_power = grid.fourier_spacing_hz * residual_psd
    return build_gaussian_catalog_bandpower_model_from_mode_power(
        grid,
        instrument_psd,
        sgwb_template_psd,
        residual_mode_power,
        omega0=omega0,
        beta=beta,
    )


def build_gaussian_catalog_bandpower_model_from_mode_power(
    grid: FourierBandGrid,
    instrument_psd: np.ndarray,
    sgwb_template_psd: np.ndarray,
    residual_mode_power: np.ndarray,
    omega0: float,
    beta: float = 1.0,
) -> CatalogBandpowerModel:
    """Construct Gaussian band-power moments for a residual mode profile."""

    instrument = _validated_mode_values(grid, instrument_psd)
    template = _validated_mode_values(grid, sgwb_template_psd)
    if np.any(instrument < 0.0) or np.any(template < 0.0):
        raise ValueError("Instrument and SGWB spectral densities must be nonnegative.")
    if not np.isfinite(omega0) or omega0 < 0.0:
        raise ValueError("omega0 must be finite and nonnegative.")
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError("beta must be finite and nonnegative.")

    mode_power = _validated_mode_values(grid, residual_mode_power)
    if np.any(mode_power < 0.0):
        raise ValueError("Residual Fourier-mode powers must be nonnegative.")
    residual = np.bincount(
        grid.bin_index,
        weights=mode_power,
        minlength=grid.mode_count.size,
    )
    residual_psd = mode_power / grid.fourier_spacing_hz
    total_psd = instrument + float(omega0) * template + float(beta) * residual_psd
    moments = integrate_gaussian_periodogram(grid, total_psd)
    instrument_power = integrate_mode_values(grid, instrument)
    sgwb_template = integrate_mode_values(grid, template)
    reconstructed_mean = (
        instrument_power
        + float(omega0) * sgwb_template
        + float(beta) * residual
    )
    np.testing.assert_allclose(
        moments.mean,
        reconstructed_mean,
        rtol=5.0e-14,
        atol=0.0,
        err_msg="Mode-summed band power does not close against its components.",
    )
    return CatalogBandpowerModel(
        instrument_power=instrument_power,
        sgwb_template=sgwb_template,
        residual_power=residual.copy(),
        mean=moments.mean,
        variance=moments.variance,
        effective_degrees_of_freedom=moments.effective_degrees_of_freedom,
    )


def build_uniform_random_phase_catalog_bandpower_model(
    grid: FourierBandGrid,
    instrument_psd: np.ndarray,
    sgwb_template_psd: np.ndarray,
    residual_band_power: np.ndarray,
    residual_source_concentration: np.ndarray,
    omega0: float,
    beta: float = 1.0,
) -> CatalogBandpowerModel:
    """Construct the finite-source diagonal random-phase band-power moments.

    Each residual source is assigned uniformly to every Fourier frequency in
    its analysis bin.  Under independent source phases, the source
    concentration in that bin is therefore also the concentration of source
    power at each of its Fourier frequencies.  Relative to the complex
    Gaussian residual-amplitude variance, the diagonal variance at a Fourier
    frequency is reduced by ``C2 * H_j**2``.  Cross-frequency phase
    correlations are not included.
    """

    residual = np.asarray(residual_band_power, dtype=float)
    concentration = np.asarray(residual_source_concentration, dtype=float)
    if residual.shape != grid.mode_count.shape:
        raise ValueError("residual_band_power must have one entry per coarse bin.")
    if concentration.shape != grid.mode_count.shape:
        raise ValueError(
            "residual_source_concentration must have one entry per coarse bin."
        )
    if not np.all(np.isfinite(residual)) or np.any(residual < 0.0):
        raise ValueError("Residual band powers must be finite and nonnegative.")
    active = residual > 0.0
    if np.any(~np.isfinite(concentration[active])):
        raise ValueError("Concentration must be finite where residual power is nonzero.")
    finite_concentration = concentration[np.isfinite(concentration)]
    if np.any(finite_concentration < 0.0) or np.any(finite_concentration > 1.0):
        raise ValueError("Source power concentration must lie between zero and one.")
    concentration = np.where(active, concentration, 0.0)

    gaussian = build_gaussian_catalog_bandpower_model(
        grid,
        instrument_psd,
        sgwb_template_psd,
        residual,
        omega0=omega0,
        beta=beta,
    )
    residual_mode_power = (
        grid.fourier_spacing_hz * piecewise_constant_residual_psd(grid, residual)
    )
    finite_source_correction = np.bincount(
        grid.bin_index,
        weights=(
            concentration[grid.bin_index]
            * (float(beta) * residual_mode_power) ** 2
        ),
        minlength=grid.mode_count.size,
    )
    variance = gaussian.variance - finite_source_correction
    if np.any(variance <= 0.0) or not np.all(np.isfinite(variance)):
        raise FloatingPointError("Finite-source band-power variance is not positive.")
    effective_dof = 2.0 * gaussian.mean**2 / variance
    return CatalogBandpowerModel(
        instrument_power=gaussian.instrument_power,
        sgwb_template=gaussian.sgwb_template,
        residual_power=gaussian.residual_power,
        mean=gaussian.mean,
        variance=variance,
        effective_degrees_of_freedom=effective_dof,
    )


def build_fixed_residual_catalog_bandpower_model_from_mode_power(
    grid: FourierBandGrid,
    instrument_psd: np.ndarray,
    sgwb_template_psd: np.ndarray,
    residual_mode_power: np.ndarray,
    omega0: float,
    beta: float = 1.0,
) -> CatalogBandpowerModel:
    """Construct moments conditional on fixed residual Fourier amplitudes.

    Instrumental noise and the fiducial SGWB are treated as independent
    complex Gaussian Fourier amplitudes with mode-integrated stochastic power
    ``N_j``.  Conditional on a fixed residual amplitude with mode power
    ``H_j``, the mode-power variance is
    ``N_j**2 + 2 * beta * N_j * H_j``.  The mean remains
    ``N_j + beta * H_j``.  This treatment therefore differs from the
    Gaussian-residual model only by removal of the residual self-noise term
    ``(beta * H_j)**2``.
    """

    instrument = _validated_mode_values(grid, instrument_psd)
    template = _validated_mode_values(grid, sgwb_template_psd)
    if np.any(instrument < 0.0) or np.any(template < 0.0):
        raise ValueError("Instrument and SGWB spectral densities must be nonnegative.")
    if not np.isfinite(omega0) or omega0 < 0.0:
        raise ValueError("omega0 must be finite and nonnegative.")
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError("beta must be finite and nonnegative.")

    mode_power = _validated_mode_values(grid, residual_mode_power)
    if np.any(mode_power < 0.0):
        raise ValueError("Residual Fourier-mode powers must be nonnegative.")

    spacing = grid.fourier_spacing_hz
    stochastic_mode_power = spacing * (instrument + float(omega0) * template)
    residual_mode_power_fiducial = float(beta) * mode_power
    mode_mean = stochastic_mode_power + residual_mode_power_fiducial
    mode_variance = (
        stochastic_mode_power**2
        + 2.0 * stochastic_mode_power * residual_mode_power_fiducial
    )
    mean = np.bincount(
        grid.bin_index,
        weights=mode_mean,
        minlength=grid.mode_count.size,
    )
    variance = np.bincount(
        grid.bin_index,
        weights=mode_variance,
        minlength=grid.mode_count.size,
    )
    effective_dof = np.full_like(mean, np.nan)
    np.divide(2.0 * mean**2, variance, out=effective_dof, where=variance > 0.0)

    instrument_power = integrate_mode_values(grid, instrument)
    sgwb_template = integrate_mode_values(grid, template)
    residual = np.bincount(
        grid.bin_index,
        weights=mode_power,
        minlength=grid.mode_count.size,
    )
    reconstructed_mean = (
        instrument_power
        + float(omega0) * sgwb_template
        + float(beta) * residual
    )
    np.testing.assert_allclose(
        mean,
        reconstructed_mean,
        rtol=5.0e-14,
        atol=0.0,
        err_msg="Fixed-residual band power does not close against its components.",
    )
    return CatalogBandpowerModel(
        instrument_power=instrument_power,
        sgwb_template=sgwb_template,
        residual_power=residual,
        mean=mean,
        variance=variance,
        effective_degrees_of_freedom=effective_dof,
    )


def prior_width_for_uncertainty_threshold(
    f_oo: float,
    f_ob: float,
    f_bb: float,
    threshold: float,
) -> float:
    """Return the largest Gaussian prior width satisfying the threshold."""

    if f_oo <= 0.0 or f_bb < 0.0 or threshold <= 1.0:
        return np.nan
    required_total_bb = f_ob**2 / (f_oo * (1.0 - threshold**-2.0))
    if f_bb >= required_total_bb:
        return np.inf
    additional_information = required_total_bb - f_bb
    return float(1.0 / np.sqrt(additional_information))


def two_parameter_fisher_summary(
    sgwb_template: np.ndarray,
    residual_power: np.ndarray,
    bandpower_variance: np.ndarray,
    omega0: float,
    beta_fiducial: float = 1.0,
    uncertainty_threshold: float = 1.1,
) -> dict[str, float]:
    """Return fixed-covariance Fisher quantities for ``(Omega0, beta)``."""

    template = np.asarray(sgwb_template, dtype=float)
    residual = np.asarray(residual_power, dtype=float)
    variance = np.asarray(bandpower_variance, dtype=float)
    if template.shape != residual.shape or template.shape != variance.shape:
        raise ValueError("Template, residual power, and variance must share a shape.")
    if template.ndim != 1 or template.size == 0:
        raise ValueError("Fisher inputs must be nonempty one-dimensional arrays.")
    if not np.all(np.isfinite(template)) or np.any(template < 0.0):
        raise ValueError("The SGWB bandpower template must be finite and nonnegative.")
    if not np.all(np.isfinite(residual)) or np.any(residual < 0.0):
        raise ValueError("Residual band powers must be finite and nonnegative.")
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0.0):
        raise ValueError("Bandpower variances must be finite and positive.")
    if not np.isfinite(omega0) or omega0 < 0.0:
        raise ValueError("omega0 must be finite and nonnegative.")
    if not np.isfinite(beta_fiducial) or beta_fiducial < 0.0:
        raise ValueError("beta_fiducial must be finite and nonnegative.")
    if not np.isfinite(uncertainty_threshold) or uncertainty_threshold <= 1.0:
        raise ValueError("uncertainty_threshold must exceed one.")
    f_oo = float(np.sum(template**2 / variance))
    f_ob = float(np.sum(template * residual / variance))
    f_bb = float(np.sum(residual**2 / variance))

    sigma_fixed = np.inf if f_oo <= 0.0 else float(1.0 / np.sqrt(f_oo))
    delta_omega = (
        np.nan
        if f_oo <= 0.0
        else float(beta_fiducial * f_ob / f_oo)
    )
    bias_over_sigma = (
        np.nan
        if not np.isfinite(sigma_fixed) or sigma_fixed <= 0.0
        else float(delta_omega / sigma_fixed)
    )
    if f_oo <= 0.0:
        rho = np.nan
        inflation = np.nan
    elif f_bb <= 0.0:
        rho = 0.0 if f_ob == 0.0 else np.nan
        inflation = 1.0 if f_ob == 0.0 else np.nan
    else:
        rho_raw = f_ob / np.sqrt(f_oo * f_bb)
        if abs(rho_raw) > 1.0 + 1.0e-10:
            raise ValueError("The Fisher cross element violates positive semidefiniteness.")
        rho = float(np.clip(rho_raw, -1.0, 1.0))
        inflation = (
            np.inf
            if abs(rho) == 1.0
            else float(1.0 / np.sqrt(1.0 - rho**2))
        )
    sigma_marginalized = float(sigma_fixed * inflation)
    sigma_beta = prior_width_for_uncertainty_threshold(
        f_oo,
        f_ob,
        f_bb,
        uncertainty_threshold,
    )
    return {
        "F_oo": f_oo,
        "F_ob": f_ob,
        "F_bb": f_bb,
        "beta_fiducial": float(beta_fiducial),
        "sigma_omega_fixed": sigma_fixed,
        "sigma_omega_fixed_over_omega0": (
            sigma_fixed / float(omega0)
            if omega0 > 0.0 and np.isfinite(sigma_fixed)
            else np.nan
        ),
        "sigma_omega_marginalized": sigma_marginalized,
        "no_prior_inflation": inflation,
        "no_prior_abs_corr": abs(rho) if np.isfinite(rho) else np.nan,
        "delta_omega_omitted_residual": delta_omega,
        "bias_ignore_over_sigma_fixed": bias_over_sigma,
        "sigma_beta_for_threshold": sigma_beta,
        "snr_omega_fixed": (
            float(omega0) / sigma_fixed
            if np.isfinite(sigma_fixed) and sigma_fixed > 0.0
            else 0.0
        ),
    }


def uniform_sgwb_amplitude_sweep(
    grid: FourierBandGrid,
    instrument_psd: np.ndarray,
    sgwb_template_psd: np.ndarray,
    residual_band_power: np.ndarray,
    residual_source_concentration: np.ndarray,
    omega0_values: np.ndarray,
    beta_fiducial: float = 1.0,
    uncertainty_threshold: float = 1.1,
) -> list[dict[str, float | str]]:
    """Evaluate the all-bin Fisher quantities across fiducial SGWB amplitudes.

    The residual power is uniform among the Fourier frequencies in each
    analysis bin.  At every amplitude, both covariance treatments are rebuilt
    from the corresponding fiducial total spectrum before the covariance is
    held fixed in the local Fisher calculation.
    """

    amplitudes = np.asarray(omega0_values, dtype=float)
    if amplitudes.ndim != 1 or amplitudes.size == 0:
        raise ValueError("omega0_values must be a nonempty one-dimensional array.")
    if not np.all(np.isfinite(amplitudes)) or np.any(amplitudes < 0.0):
        raise ValueError("Fiducial SGWB amplitudes must be finite and nonnegative.")
    if not np.isfinite(beta_fiducial) or beta_fiducial < 0.0:
        raise ValueError("beta_fiducial must be finite and nonnegative.")
    if not np.isfinite(uncertainty_threshold) or uncertainty_threshold <= 1.0:
        raise ValueError("uncertainty_threshold must exceed one.")

    rows: list[dict[str, float | str]] = []
    for omega0 in amplitudes:
        models = (
            (
                "gaussian_residual_amplitudes",
                build_gaussian_catalog_bandpower_model(
                    grid,
                    instrument_psd,
                    sgwb_template_psd,
                    residual_band_power,
                    omega0=float(omega0),
                    beta=beta_fiducial,
                ),
            ),
            (
                "finite_source_random_phase_diagonal",
                build_uniform_random_phase_catalog_bandpower_model(
                    grid,
                    instrument_psd,
                    sgwb_template_psd,
                    residual_band_power,
                    residual_source_concentration,
                    omega0=float(omega0),
                    beta=beta_fiducial,
                ),
            ),
        )
        for covariance_treatment, model in models:
            result = two_parameter_fisher_summary(
                model.sgwb_template,
                model.residual_power,
                model.variance,
                omega0=float(omega0),
                beta_fiducial=beta_fiducial,
                uncertainty_threshold=uncertainty_threshold,
            )
            noise_result = instrument_noise_normalization_summary(
                model.sgwb_template,
                model.residual_power,
                model.instrument_power,
                model.variance,
                omega0=float(omega0),
                noise_normalization_prior_sigma=np.inf,
                uncertainty_threshold=uncertainty_threshold,
            )
            noise_fixed_beta_ratio = float(noise_result["snr_omega_fixed_beta"])
            noise_inflation = float(noise_result["no_prior_inflation"])
            if np.isnan(noise_inflation) or noise_inflation <= 0.0:
                noise_and_beta_marginalized_ratio = np.nan
            elif np.isinf(noise_inflation):
                noise_and_beta_marginalized_ratio = 0.0
            else:
                noise_and_beta_marginalized_ratio = (
                    noise_fixed_beta_ratio / noise_inflation
                )
            result.update(
                {
                    "omega0_fiducial": float(omega0),
                    "covariance_treatment": covariance_treatment,
                    "uncertainty_threshold": float(uncertainty_threshold),
                    "uncertainty_increase_percent": (
                        100.0 * (float(result["no_prior_inflation"]) - 1.0)
                    ),
                    "inverse_fractional_fixed_uncertainty": (
                        float(omega0) / float(result["sigma_omega_fixed"])
                        if float(result["sigma_omega_fixed"]) > 0.0
                        else np.nan
                    ),
                    "marginalized_amplitude_to_uncertainty_ratio": (
                        float(omega0) / float(result["sigma_omega_marginalized"])
                        if float(result["sigma_omega_marginalized"]) > 0.0
                        else np.nan
                    ),
                    "instrument_noise_normalization_marginalized_no_prior_inflation": (
                        noise_inflation
                    ),
                    "instrument_noise_normalization_marginalized_fixed_beta_ratio": (
                        noise_fixed_beta_ratio
                    ),
                    "instrument_noise_normalization_and_beta_marginalized_ratio": (
                        noise_and_beta_marginalized_ratio
                    ),
                }
            )
            rows.append(result)
    return rows


def instrument_noise_normalization_summary(
    sgwb_template: np.ndarray,
    residual_power: np.ndarray,
    instrument_power: np.ndarray,
    bandpower_variance: np.ndarray,
    omega0: float,
    noise_normalization_prior_sigma: float,
    uncertainty_threshold: float = 1.1,
) -> dict[str, float]:
    """Marginalize an overall instrumental-power normalization.

    The mean derivatives are ordered as ``(Omega0, beta, eta)``, with
    ``eta`` multiplying the instrumental band power.  Eliminating ``eta`` by
    a Schur complement gives the effective two-parameter Fisher matrix for
    ``(Omega0, beta)``.  A zero prior standard deviation fixes ``eta``;
    ``np.inf`` leaves it unconstrained.
    """

    template = np.asarray(sgwb_template, dtype=float)
    residual = np.asarray(residual_power, dtype=float)
    instrument = np.asarray(instrument_power, dtype=float)
    variance = np.asarray(bandpower_variance, dtype=float)
    if not (
        template.shape == residual.shape == instrument.shape == variance.shape
    ):
        raise ValueError("All band-power arrays must share a shape.")
    if template.ndim != 1 or template.size == 0:
        raise ValueError("Band-power arrays must be nonempty and one-dimensional.")
    for name, array in (
        ("SGWB template", template),
        ("residual power", residual),
        ("instrument power", instrument),
    ):
        if not np.all(np.isfinite(array)) or np.any(array < 0.0):
            raise ValueError(f"{name} must be finite and nonnegative.")
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0.0):
        raise ValueError("Band-power variances must be finite and positive.")
    if not np.isfinite(omega0) or omega0 < 0.0:
        raise ValueError("omega0 must be finite and nonnegative.")
    prior_sigma = float(noise_normalization_prior_sigma)
    if np.isnan(prior_sigma) or prior_sigma < 0.0:
        raise ValueError("The noise-normalization prior width cannot be negative.")

    inverse_variance = 1.0 / variance
    f_oo = float(np.sum(template * template * inverse_variance))
    f_ob = float(np.sum(template * residual * inverse_variance))
    f_bb = float(np.sum(residual * residual * inverse_variance))
    f_on = float(np.sum(template * instrument * inverse_variance))
    f_bn = float(np.sum(residual * instrument * inverse_variance))
    f_nn = float(np.sum(instrument * instrument * inverse_variance))

    if prior_sigma == 0.0:
        effective_oo, effective_ob, effective_bb = f_oo, f_ob, f_bb
        noise_precision = np.inf
    else:
        noise_precision = 0.0 if np.isinf(prior_sigma) else prior_sigma**-2.0
        denominator = f_nn + noise_precision
        if denominator <= 0.0 or not np.isfinite(denominator):
            raise ValueError("The instrumental normalization is unconstrained.")
        effective_oo = f_oo - f_on**2 / denominator
        effective_ob = f_ob - f_on * f_bn / denominator
        effective_bb = f_bb - f_bn**2 / denominator

    scale = max(f_oo, f_bb, 1.0)
    tolerance = 1.0e-12 * scale
    if effective_oo < -tolerance or effective_bb < -tolerance:
        raise ValueError("Noise marginalization produced a nonphysical Fisher matrix.")
    effective_oo = max(effective_oo, 0.0)
    effective_bb = max(effective_bb, 0.0)
    if effective_oo <= 0.0:
        rho = np.nan
        inflation = np.nan
        sigma_fixed = np.inf
    elif effective_bb <= 0.0:
        rho = 0.0 if effective_ob == 0.0 else np.nan
        inflation = 1.0 if effective_ob == 0.0 else np.nan
        sigma_fixed = float(1.0 / np.sqrt(effective_oo))
    else:
        rho_raw = effective_ob / np.sqrt(effective_oo * effective_bb)
        if abs(rho_raw) > 1.0 + 1.0e-10:
            raise ValueError("The marginalized Fisher matrix is not positive semidefinite.")
        rho = float(np.clip(rho_raw, -1.0, 1.0))
        inflation = (
            np.inf
            if abs(rho) == 1.0
            else float(1.0 / np.sqrt(1.0 - rho**2))
        )
        sigma_fixed = float(1.0 / np.sqrt(effective_oo))

    return {
        "noise_normalization_prior_sigma": prior_sigma,
        "noise_normalization_prior_precision": noise_precision,
        "F_oo_after_noise_marginalization": effective_oo,
        "F_ob_after_noise_marginalization": effective_ob,
        "F_bb_after_noise_marginalization": effective_bb,
        "no_prior_abs_corr": abs(rho) if np.isfinite(rho) else np.nan,
        "no_prior_inflation": inflation,
        "sigma_omega_fixed_beta": sigma_fixed,
        "snr_omega_fixed_beta": (
            float(omega0) / sigma_fixed
            if np.isfinite(sigma_fixed) and sigma_fixed > 0.0
            else 0.0
        ),
        "sigma_beta_for_threshold": prior_width_for_uncertainty_threshold(
            effective_oo,
            effective_ob,
            effective_bb,
            uncertainty_threshold,
        ),
    }
