#!/usr/bin/env python
"""Construct the catalog-level orbit-averaged LISA-X bandpower calculation.

This stage averages the long-wavelength Michelson-X antenna power over equally
spaced LISA orbital phases and uses the independent positive Fourier
frequencies of the Sangria record.  In the reference calculation, each
analysis bin's residual power is distributed uniformly among its Fourier
frequencies.  Two further power-conserving distributions quantify the
dependence on unresolved within-bin frequency structure.  The Fisher
covariance is obtained by summing the Gaussian Fourier-power variances in each
bin.

The random-phase source-sum calculation checks its analytic fourth-moment
relation and is not inserted into the Fourier-power covariance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from catalog_matching import recovered_row_indices_by_group
from galactic_binary_power import phase_averaged_projected_power
from lisa_catalog_statistics import (
    LISA_LW_SINGLE_MICHELSON_AVERAGED_RESPONSE,
    alpha_column_suffix,
    build_fixed_residual_catalog_bandpower_model_from_mode_power,
    build_gaussian_catalog_bandpower_model_from_mode_power,
    catalog_binary_frequency_support,
    catalog_support_residual_mode_power,
    constellation_plane_inclination,
    instrument_noise_normalization_summary,
    isotropic_single_polarization_response,
    lisa_equal_arm_reference_positions,
    nearest_fourier_mode_residual_power,
    piecewise_constant_residual_psd,
    positive_fourier_modes_by_bin,
    sgwb_strain_psd_per_omega,
    single_michelson_lwa_equivalent_strain_psd,
    static_lisa_michelson_x_tensor,
    two_parameter_fisher_summary,
)


NORMAL_TWO_SIDED_TAILS = {
    2.0: 4.55002638963584e-2,
    3.0: 2.69979606326019e-3,
    4.0: 6.33424836662399e-5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sangria-h5", required=True)
    parser.add_argument(
        "--observation-dataset",
        "--tdi-dataset",
        dest="observation_dataset",
        default="obs/tdi",
        help=(
            "HDF5 dataset used only to read the observation sample count and "
            "cadence; its time-series values are not used."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["sky/dgb/cat", "sky/igb/cat"],
    )
    parser.add_argument("--matching-audit", required=True)
    parser.add_argument("--frequency-min-mHz", type=float, default=0.4)
    parser.add_argument("--frequency-max-mHz", type=float, default=6.0)
    parser.add_argument("--bin-width-Hz", type=float, default=1.0e-5)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--lisa-arm-length-m", type=float, default=2.5e9)
    parser.add_argument(
        "--response-phase-count",
        type=int,
        default=12,
        help=(
            "Number of equally spaced orbital phases used to integrate the "
            "long-wavelength Michelson-X antenna power over one year."
        ),
    )
    parser.add_argument("--h0-km-s-Mpc", type=float, default=67.4)
    parser.add_argument("--omega0", type=float, default=1.0e-11)
    parser.add_argument("--beta-fiducial", type=float, default=1.0)
    parser.add_argument("--alpha-list", nargs="+", type=float, default=[-2.0, 0.0, 2.0])
    parser.add_argument("--sgwb-fref-Hz", type=float, default=0.003)
    parser.add_argument("--inflation-threshold", type=float, default=1.1)
    parser.add_argument(
        "--noise-normalization-prior-list",
        nargs="+",
        type=float,
        default=[0.0, 0.001, 0.01, 0.05, 0.1, float("inf")],
        help=(
            "Gaussian prior standard deviations for an auxiliary overall "
            "instrumental-power normalization; zero fixes the noise and inf "
            "leaves that normalization unconstrained."
        ),
    )
    parser.add_argument("--top-n-list", nargs="+", type=int, default=[6, 12, 24, 36])
    parser.add_argument("--mc-n-draws", type=int, default=2_000)
    parser.add_argument("--mc-batch-size", type=int, default=2_000)
    parser.add_argument("--mc-max-batch-elements", type=int, default=2_000_000)
    parser.add_argument(
        "--skip-random-phase-check",
        action="store_true",
        help=(
            "Recompute the deterministic catalog, covariance, and Fisher "
            "outputs without rewriting the independent random-phase Monte "
            "Carlo tables or figure."
        ),
    )
    parser.add_argument(
        "--output-bin-summary",
        default="repro/tables/all_bins_LWA_X_summary_C0p9_overlap0p9_f0p4_6mHz.csv",
    )
    parser.add_argument(
        "--output-noise-table",
        default=(
            "repro/tables/"
            "all_bins_noise_dilution_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.csv"
        ),
    )
    parser.add_argument(
        "--output-fisher-table",
        default=(
            "repro/tables/"
            "all_bins_fisher_baseline_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.csv"
        ),
    )
    parser.add_argument(
        "--output-validation-table",
        default=(
            "repro/tables/"
            "all_bins_random_phase_moment_check_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.csv"
        ),
    )
    parser.add_argument(
        "--output-validation-summary",
        default=(
            "repro/tables/"
            "all_bins_random_phase_moment_check_summary_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.csv"
        ),
    )
    parser.add_argument(
        "--output-fisher-alpha0-summary-csv",
        default=(
            "repro/tables/"
            "all_bins_fisher_baseline_alpha0_summary_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.csv"
        ),
    )
    parser.add_argument(
        "--output-fisher-alpha0-summary-tex",
        default=(
            "repro/tables/"
            "all_bins_fisher_baseline_alpha0_summary_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.tex"
        ),
    )
    parser.add_argument(
        "--output-residual-profile-sensitivity",
        default=(
            "repro/sensitivity/residual_fourier_profile/"
            "residual_fourier_profile_sensitivity.csv"
        ),
    )
    parser.add_argument(
        "--output-covariance-ensemble-sensitivity",
        default=(
            "repro/sensitivity/residual_covariance_ensemble/"
            "fixed_vs_gaussian_residual_covariance.csv"
        ),
    )
    parser.add_argument(
        "--output-residual-mode-profiles",
        default=(
            "repro/sensitivity/residual_fourier_profile/"
            "residual_mode_power_profiles.npz"
        ),
    )
    parser.add_argument(
        "--output-frequency-support-audit",
        default=(
            "repro/sensitivity/residual_fourier_profile/"
            "residual_frequency_support_audit.csv"
        ),
    )
    parser.add_argument(
        "--output-noise-normalization-sensitivity",
        default=(
            "repro/sensitivity/instrument_noise/"
            "instrument_noise_normalization_sensitivity.csv"
        ),
    )
    parser.add_argument(
        "--output-validation-figure-png",
        default=(
            "repro/figures/all_bins_random_phase_moment_check_"
            "LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.png"
        ),
    )
    parser.add_argument(
        "--output-validation-figure-pdf",
        default=(
            "repro/figures/all_bins_random_phase_moment_check_"
            "LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.pdf"
        ),
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def normalized_name(name: str) -> str:
    return "".join(character.lower() for character in str(name) if character.isalnum())


def find_field(dtype_names: tuple[str, ...], field_options: list[str]) -> str:
    lookup = {normalized_name(name): name for name in dtype_names}
    for option in field_options:
        key = normalized_name(option)
        if key in lookup:
            return lookup[key]
    raise KeyError(f"Could not find any of {field_options} in {dtype_names}.")


def read_observation_metadata(
    h5_path: str | Path,
    dataset_path: str,
) -> tuple[int, float, float]:
    with h5py.File(h5_path, "r") as h5:
        if dataset_path not in h5:
            raise KeyError(f"Observation dataset not found: {dataset_path}")
        dataset = h5[dataset_path]
        if dataset.shape[0] < 1 or "dt" not in dataset.attrs:
            raise ValueError(f"Incomplete observation metadata in {dataset_path}.")
        n_samples = int(dataset.shape[0])
        cadence_s = float(dataset.attrs["dt"])
    if not np.isfinite(cadence_s) or cadence_s <= 0.0:
        raise ValueError(f"Invalid observation cadence: {cadence_s}")
    return n_samples, cadence_s, n_samples * cadence_s


def make_frequency_bins(
    frequency_min_hz: float,
    frequency_max_hz: float,
    bin_width_hz: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    if not 0.0 < frequency_min_hz < frequency_max_hz:
        raise ValueError("The frequency range must be positive and increasing.")
    if not np.isfinite(bin_width_hz) or bin_width_hz <= 0.0:
        raise ValueError("bin_width_hz must be positive and finite.")
    nominal_count = (frequency_max_hz - frequency_min_hz) / bin_width_hz
    n_bins = int(round(nominal_count))
    if not np.isclose(nominal_count, n_bins, rtol=0.0, atol=1.0e-10):
        raise ValueError("The frequency range must contain an integer number of bins.")
    edges = frequency_min_hz + np.arange(n_bins + 1, dtype=float) * bin_width_hz
    edges[-1] = frequency_max_hz
    bins = pd.DataFrame(
        {
            "frequency_bin_index": np.arange(n_bins, dtype=np.int64),
            "f_left_Hz": edges[:-1],
            "f_right_Hz": edges[1:],
            "f_center_Hz": 0.5 * (edges[:-1] + edges[1:]),
        }
    )
    return bins, edges


def locate_frequency_bins(
    frequency_hz: np.ndarray,
    bin_edges_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frequency = np.asarray(frequency_hz, dtype=float)
    index = np.searchsorted(bin_edges_hz, frequency, side="right") - 1
    valid = (
        np.isfinite(frequency)
        & (index >= 0)
        & (index < bin_edges_hz.size - 1)
        & (frequency >= bin_edges_hz[0])
        & (frequency < bin_edges_hz[-1])
    )
    return index, valid


def antenna_patterns_vector(
    longitude: np.ndarray,
    latitude: np.ndarray,
    polarization: np.ndarray,
    detector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    e_beta = np.column_stack(
        (
            -np.sin(latitude) * np.cos(longitude),
            -np.sin(latitude) * np.sin(longitude),
            np.cos(latitude),
        )
    )
    e_lambda = np.column_stack(
        (-np.sin(longitude), np.cos(longitude), np.zeros_like(longitude))
    )
    p = e_beta * np.cos(polarization)[:, None] + e_lambda * np.sin(polarization)[:, None]
    q = -e_beta * np.sin(polarization)[:, None] + e_lambda * np.cos(polarization)[:, None]
    p_dp = np.einsum("ni,ij,nj->n", p, detector, p)
    q_dq = np.einsum("ni,ij,nj->n", q, detector, q)
    p_dq = np.einsum("ni,ij,nj->n", p, detector, q)
    return p_dp - q_dq, 2.0 * p_dq


def response_normalized_source_power(
    amplitude: np.ndarray,
    inclination: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    polarization: np.ndarray,
    detector: np.ndarray,
) -> np.ndarray:
    f_plus, f_cross = antenna_patterns_vector(
        longitude,
        latitude,
        polarization,
        detector,
    )
    detector_power = phase_averaged_projected_power(
        amplitude,
        inclination,
        f_plus,
        f_cross,
    )
    return detector_power / LISA_LW_SINGLE_MICHELSON_AVERAGED_RESPONSE


def orbit_averaged_response_normalized_source_power(
    amplitude: np.ndarray,
    inclination: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    polarization: np.ndarray,
    detectors: tuple[np.ndarray, ...],
) -> np.ndarray:
    """Average the long-wavelength Michelson power over orbital snapshots."""

    if not detectors:
        raise ValueError("At least one detector snapshot is required.")
    result = np.zeros_like(np.asarray(amplitude, dtype=float))
    for detector in detectors:
        result += response_normalized_source_power(
            amplitude,
            inclination,
            longitude,
            latitude,
            polarization,
            detector,
        )
    return result / float(len(detectors))


def init_accumulators(n_bins: int) -> dict[str, np.ndarray]:
    accumulators: dict[str, np.ndarray] = {}
    for prefix in ("inj", "rec", "res"):
        accumulators[f"{prefix}_count"] = np.zeros(n_bins, dtype=np.int64)
        accumulators[f"{prefix}_power_sum"] = np.zeros(n_bins, dtype=float)
        accumulators[f"{prefix}_power_sum2"] = np.zeros(n_bins, dtype=float)
        accumulators[f"{prefix}_power_max"] = np.full(n_bins, np.nan, dtype=float)
    return accumulators


def accumulate_subset(
    accumulators: dict[str, np.ndarray],
    prefix: str,
    bin_index: np.ndarray,
    power: np.ndarray,
    mask: np.ndarray,
) -> None:
    if not np.any(mask):
        return
    selected_bins = bin_index[mask]
    selected_power = power[mask]
    n_bins = accumulators[f"{prefix}_count"].size
    accumulators[f"{prefix}_count"] += np.bincount(selected_bins, minlength=n_bins)
    accumulators[f"{prefix}_power_sum"] += np.bincount(
        selected_bins,
        weights=selected_power,
        minlength=n_bins,
    )
    accumulators[f"{prefix}_power_sum2"] += np.bincount(
        selected_bins,
        weights=selected_power**2,
        minlength=n_bins,
    )
    current = np.where(
        np.isfinite(accumulators[f"{prefix}_power_max"]),
        accumulators[f"{prefix}_power_max"],
        -np.inf,
    )
    np.maximum.at(current, selected_bins, selected_power)
    current[current == -np.inf] = np.nan
    accumulators[f"{prefix}_power_max"] = current


def source_power_concentration(sum_power: np.ndarray, sum_power_squared: np.ndarray) -> np.ndarray:
    concentration = np.full_like(sum_power, np.nan, dtype=float)
    np.divide(
        sum_power_squared,
        sum_power**2,
        out=concentration,
        where=sum_power > 0.0,
    )
    return concentration


def monte_carlo_random_phase(
    source_power: np.ndarray,
    n_draws: int,
    batch_size: int,
    max_batch_elements: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    power = np.asarray(source_power, dtype=float)
    if power.ndim != 1 or power.size == 0 or np.any(power <= 0.0):
        raise ValueError("source_power must be a nonempty positive 1D array.")
    if n_draws < 4:
        raise ValueError("At least four Monte Carlo draws are required for a fourth moment.")
    if batch_size < 1 or max_batch_elements < 1:
        raise ValueError("Monte Carlo batch sizes must be positive.")
    amplitude = np.sqrt(2.0 * power)
    true_sigma = np.sqrt(np.sum(power))
    batch_cap = max(1, max_batch_elements // power.size)
    effective_batch = min(batch_size, batch_cap)
    sums = np.zeros(4, dtype=float)
    tails = {threshold: 0 for threshold in NORMAL_TWO_SIDED_TAILS}
    completed = 0
    while completed < n_draws:
        count = min(effective_batch, n_draws - completed)
        values = np.zeros(count, dtype=float)
        source_chunk = max(1, max_batch_elements // count)
        for start in range(0, power.size, source_chunk):
            stop = min(start + source_chunk, power.size)
            phases = rng.uniform(0.0, 2.0 * np.pi, size=(stop - start, count))
            values += np.sum(amplitude[start:stop, None] * np.cos(phases), axis=0)
        sums[0] += np.sum(values)
        sums[1] += np.sum(values**2)
        sums[2] += np.sum(values**3)
        sums[3] += np.sum(values**4)
        standardized = np.abs(values / true_sigma)
        for threshold in tails:
            tails[threshold] += int(np.sum(standardized > threshold))
        completed += count
    raw1, raw2, raw3, raw4 = sums / float(n_draws)
    variance = raw2 - raw1**2
    central3 = raw3 - 3.0 * raw1 * raw2 + 2.0 * raw1**3
    central4 = raw4 - 4.0 * raw1 * raw3 + 6.0 * raw1**2 * raw2 - 3.0 * raw1**4
    return {
        "mc_n_draws": int(n_draws),
        "mc_mean": float(raw1),
        "mc_variance": float(variance),
        "mc_skewness": float(central3 / variance**1.5),
        "mc_excess_kurtosis": float(central4 / variance**2 - 3.0),
        **{
            f"mc_tail_gt_{int(threshold)}sigma": tails[threshold] / float(n_draws)
            for threshold in tails
        },
    }


def random_phase_moment_check(
    summary: pd.DataFrame,
    residual_source_power: list[list[float]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260523)
    rows: list[dict[str, float | int]] = []
    for row in summary.itertuples(index=False):
        bin_index = int(row.frequency_bin_index)
        residual_power = float(row.LWA_X_res_power_sum)
        if residual_power <= 0.0:
            continue
        source_power = np.asarray(residual_source_power[bin_index], dtype=float)
        expected_count = int(row.LWA_X_res_n_used_rows)
        if source_power.size != expected_count:
            raise RuntimeError(
                f"Random-phase source cache has {source_power.size} entries for bin "
                f"{bin_index}, expected {expected_count}."
            )
        concentration = float(row.LWA_X_res_C2)
        predicted = -1.5 * concentration
        result: dict[str, float | int] = {
            "frequency_bin_index": bin_index,
            "f_center_Hz": float(row.f_center_Hz),
            "f_mHz": float(row.f_center_mHz),
            "n_residual_sources": expected_count,
            "C2": concentration,
            "pred_excess_kurtosis": predicted,
        }
        result.update(
            monte_carlo_random_phase(
                source_power,
                args.mc_n_draws,
                args.mc_batch_size,
                args.mc_max_batch_elements,
                rng,
            )
        )
        result["delta_mc_minus_pred_excess_kurtosis"] = (
            float(result["mc_excess_kurtosis"]) - predicted
        )
        rows.append(result)
    table = pd.DataFrame(rows)
    summary_row = pd.DataFrame(
        [
            {
                "n_bins_total": int((summary["LWA_X_res_power_sum"] > 0.0).sum()),
                "n_bins_mc": int(len(table)),
                "n_bins_without_mc": int(
                    (summary["LWA_X_res_power_sum"] > 0.0).sum() - len(table)
                ),
                "mc_n_draws_per_bin": int(args.mc_n_draws),
                "median_abs_mc_minus_pred_excess_kurtosis": float(
                    np.median(np.abs(table["delta_mc_minus_pred_excess_kurtosis"]))
                ),
                "moment_model": "independent_random_phase_source_sum",
                "used_in_bandpower_covariance": False,
            }
        ]
    )
    return table, summary_row


def save_random_phase_figure(
    table: pd.DataFrame,
    png_path: Path,
    pdf_path: Path,
    dpi: int,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.75), constrained_layout=True)
    predicted = table["pred_excess_kurtosis"].to_numpy(float)
    measured = table["mc_excess_kurtosis"].to_numpy(float)
    axes[0].scatter(predicted, measured, s=9, alpha=0.65, color="#2f6f9f")
    finite = np.isfinite(predicted) & np.isfinite(measured)
    lower = min(np.min(predicted[finite]), np.min(measured[finite]))
    upper = max(np.max(predicted[finite]), np.max(measured[finite]))
    axes[0].plot([lower, upper], [lower, upper], color="0.2", linewidth=0.9)
    axes[0].set_xlabel(r"Analytic $\gamma_{2,\rm res}$")
    axes[0].set_ylabel(r"Monte Carlo $\gamma_{2,\rm res}$")
    axes[0].text(0.03, 0.95, "(a)", transform=axes[0].transAxes, va="top")
    axes[1].scatter(
        table["C2"],
        table["delta_mc_minus_pred_excess_kurtosis"],
        s=9,
        alpha=0.65,
        color="#8a4f32",
    )
    axes[1].axhline(0.0, color="0.2", linewidth=0.9)
    axes[1].set_xlabel(r"Source power concentration $C_2$")
    axes[1].set_ylabel(r"$\gamma_{2,\rm MC}-\gamma_{2,\rm res}$")
    axes[1].text(0.03, 0.95, "(b)", transform=axes[1].transAxes, va="top")
    for axis in axes:
        axis.grid(True, alpha=0.2, linewidth=0.5)
        axis.tick_params(direction="in", top=True, right=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)


def fisher_for_rows(
    rows: pd.DataFrame,
    alpha: float,
    omega0: float,
    beta_fiducial: float,
    uncertainty_threshold: float,
) -> dict[str, float]:
    suffix = alpha_column_suffix(alpha)
    return two_parameter_fisher_summary(
        rows[f"T_alpha_{suffix}"].to_numpy(float),
        rows["LWA_X_res_power_sum"].to_numpy(float),
        rows[f"bandpower_variance_gaussian_alpha_{suffix}"].to_numpy(float),
        omega0=omega0,
        beta_fiducial=beta_fiducial,
        uncertainty_threshold=uncertainty_threshold,
    )


def build_bin_sets(summary: pd.DataFrame, top_n_list: list[int]) -> dict[str, pd.DataFrame]:
    nonzero = summary.loc[summary["LWA_X_res_power_sum"] > 0.0].copy()
    ranked = nonzero.sort_values("abs_gamma2_total", ascending=False)
    subsets = {
        "all_analysis_bins": summary.copy(),
        "all_nonzero_residual_bins": nonzero,
    }
    for n_bins in top_n_list:
        if n_bins <= len(ranked):
            subsets[f"top_abs_gamma2_{n_bins}"] = ranked.head(n_bins).copy()
    return subsets


def write_alpha0_summary(fisher: pd.DataFrame, csv_path: Path, tex_path: Path) -> None:
    table = fisher.loc[np.isclose(fisher["alpha"], 0.0)].copy()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    columns = [
        "bin_set",
        "n_bins",
        "no_prior_inflation",
        "no_prior_abs_corr",
        "sigma_beta_for_threshold",
        "bias_ignore_over_sigma_fixed",
    ]
    tex_path.write_text(table[columns].to_latex(index=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    for path in (Path(args.sangria_h5), Path(args.matching_audit)):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.beta_fiducial < 0.0:
        raise ValueError("beta-fiducial must be nonnegative.")
    if args.response_phase_count < 1:
        raise ValueError("response-phase-count must be positive.")
    if not args.skip_random_phase_check and args.mc_n_draws < 4:
        raise ValueError("mc-n-draws must be at least four.")
    if any(np.isnan(width) or width < 0.0 for width in args.noise_normalization_prior_list):
        raise ValueError("Noise-normalization prior widths cannot be negative or NaN.")

    frequency_min_hz = args.frequency_min_mHz * 1.0e-3
    frequency_max_hz = args.frequency_max_mHz * 1.0e-3
    bins, bin_edges = make_frequency_bins(
        frequency_min_hz,
        frequency_max_hz,
        args.bin_width_Hz,
    )
    n_samples, cadence_s, observation_duration_s = read_observation_metadata(
        args.sangria_h5,
        args.observation_dataset,
    )
    fourier_grid = positive_fourier_modes_by_bin(
        n_time_samples=n_samples,
        cadence_s=cadence_s,
        bin_edges_hz=bin_edges,
    )

    matching_audit = pd.read_csv(args.matching_audit)
    recovered_rows = recovered_row_indices_by_group(
        matching_audit,
        expected_groups=set(args.datasets),
    )
    expected_recovered_count = sum(len(rows) for rows in recovered_rows.values())

    orbital_phases = (
        2.0 * np.pi * np.arange(args.response_phase_count, dtype=float)
        / float(args.response_phase_count)
    )
    detectors_x = tuple(
        static_lisa_michelson_x_tensor(phase, 0.0) for phase in orbital_phases
    )
    responses = np.array(
        [isotropic_single_polarization_response(detector) for detector in detectors_x]
    )
    inclinations = np.array(
        [
            constellation_plane_inclination(
                lisa_equal_arm_reference_positions(phase, 0.0)
            )
            for phase in orbital_phases
        ]
    )
    if not np.allclose(
        responses,
        LISA_LW_SINGLE_MICHELSON_AVERAGED_RESPONSE,
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise RuntimeError("A Michelson-X snapshot response is not 3/20.")
    if not np.allclose(inclinations, np.pi / 3.0, rtol=0.0, atol=1.0e-14):
        raise RuntimeError("A LISA snapshot plane is not inclined by 60 degrees.")
    response = float(responses[0])
    inclination = float(inclinations[0])

    accumulators = init_accumulators(len(bins))
    residual_source_power: list[list[float]] | None = (
        None
        if args.skip_random_phase_check
        else [[] for _ in range(len(bins))]
    )
    residual_mode_power_nearest = np.zeros(fourier_grid.frequency_hz.size, dtype=float)
    residual_mode_power_support = np.zeros(fourier_grid.frequency_hz.size, dtype=float)
    support_audit_totals = {
        "source_count": 0,
        "source_power": 0.0,
        "boundary_crossing_source_count": 0,
        "boundary_crossing_source_power": 0.0,
        "support_width_sum_Hz": 0.0,
        "support_width_power_sum_Hz_strain2": 0.0,
        "support_width_max_Hz": 0.0,
        "support_without_fourier_mode_count": 0,
    }
    recovered_seen = 0
    with h5py.File(args.sangria_h5, "r") as h5:
        for dataset_path in args.datasets:
            if dataset_path not in h5:
                raise KeyError(f"Dataset not found: {dataset_path}")
            dataset = h5[dataset_path]
            if dataset.dtype.names is None:
                raise TypeError(f"Dataset {dataset_path} is not structured.")
            names = dataset.dtype.names
            fields = {
                "frequency": find_field(names, ["Frequency", "frequency"]),
                "frequency_derivative": find_field(
                    names,
                    [
                        "FrequencyDerivative",
                        "Frequency Derivative",
                        "frequency_derivative",
                        "fdot",
                    ],
                ),
                "amplitude": find_field(names, ["Amplitude", "amplitude"]),
                "inclination": find_field(names, ["Inclination", "inclination"]),
                "polarization": find_field(names, ["Polarization", "polarization"]),
                "longitude": find_field(
                    names,
                    ["EclipticLongitude", "Ecliptic Longitude", "lambda", "lon"],
                ),
                "latitude": find_field(
                    names,
                    ["EclipticLatitude", "Ecliptic Latitude", "beta", "lat"],
                ),
            }
            expected_rows = recovered_rows[dataset_path]
            for start in range(0, len(dataset), args.chunksize):
                stop = min(start + args.chunksize, len(dataset))
                chunk = dataset[start:stop]
                frequency = np.asarray(chunk[fields["frequency"]], dtype=float).reshape(-1)
                bin_index, in_band = locate_frequency_bins(frequency, bin_edges)
                if not np.any(in_band):
                    continue
                selected_frequency = frequency[in_band]
                selected_frequency_derivative = np.asarray(
                    chunk[fields["frequency_derivative"]], dtype=float
                ).reshape(-1)[in_band]
                selected_latitude = np.asarray(
                    chunk[fields["latitude"]], dtype=float
                ).reshape(-1)[in_band]
                selected_rows = np.flatnonzero(in_band).astype(np.int64)
                catalog_rows = selected_rows + start
                positions_in_audit = np.searchsorted(expected_rows, catalog_rows)
                recovered = np.zeros(catalog_rows.size, dtype=bool)
                within = positions_in_audit < expected_rows.size
                recovered[within] = (
                    expected_rows[positions_in_audit[within]] == catalog_rows[within]
                )
                recovered_seen += int(np.count_nonzero(recovered))
                power = orbit_averaged_response_normalized_source_power(
                    np.asarray(chunk[fields["amplitude"]], dtype=float).reshape(-1)[in_band],
                    np.asarray(chunk[fields["inclination"]], dtype=float).reshape(-1)[in_band],
                    np.asarray(chunk[fields["longitude"]], dtype=float).reshape(-1)[in_band],
                    selected_latitude,
                    np.asarray(chunk[fields["polarization"]], dtype=float).reshape(-1)[in_band],
                    detectors_x,
                )
                selected_bins = bin_index[in_band].astype(np.int64)
                valid_power = np.isfinite(power) & (power > 0.0)
                residual = ~recovered
                accumulate_subset(accumulators, "inj", selected_bins, power, valid_power)
                accumulate_subset(
                    accumulators,
                    "rec",
                    selected_bins,
                    power,
                    valid_power & recovered,
                )
                accumulate_subset(
                    accumulators,
                    "res",
                    selected_bins,
                    power,
                    valid_power & residual,
                )
                cache_mask = valid_power & residual
                if np.any(cache_mask):
                    residual_frequency = selected_frequency[cache_mask]
                    residual_frequency_derivative = selected_frequency_derivative[cache_mask]
                    residual_latitude = selected_latitude[cache_mask]
                    residual_power_chunk = power[cache_mask]
                    residual_bin = selected_bins[cache_mask]
                    support_lower, support_upper = catalog_binary_frequency_support(
                        residual_frequency,
                        residual_frequency_derivative,
                        residual_latitude,
                        observation_duration_s,
                    )
                    residual_mode_power_nearest += nearest_fourier_mode_residual_power(
                        fourier_grid,
                        residual_frequency,
                        residual_power_chunk,
                    )
                    residual_mode_power_support += catalog_support_residual_mode_power(
                        fourier_grid,
                        support_lower,
                        support_upper,
                        residual_power_chunk,
                    )

                    source_bin_left = bin_edges[residual_bin]
                    source_bin_right = bin_edges[residual_bin + 1]
                    crosses_boundary = (
                        (support_lower < source_bin_left)
                        | (support_upper >= source_bin_right)
                    )
                    support_width = support_upper - support_lower
                    mode_start = np.searchsorted(
                        fourier_grid.frequency_hz,
                        support_lower,
                        side="left",
                    )
                    mode_stop = np.searchsorted(
                        fourier_grid.frequency_hz,
                        support_upper,
                        side="right",
                    )
                    mode_start = np.clip(
                        mode_start,
                        0,
                        fourier_grid.frequency_hz.size,
                    )
                    mode_stop = np.clip(
                        mode_stop,
                        0,
                        fourier_grid.frequency_hz.size,
                    )
                    support_audit_totals["source_count"] += int(cache_mask.sum())
                    support_audit_totals["source_power"] += float(
                        np.sum(residual_power_chunk)
                    )
                    support_audit_totals["boundary_crossing_source_count"] += int(
                        np.count_nonzero(crosses_boundary)
                    )
                    support_audit_totals["boundary_crossing_source_power"] += float(
                        np.sum(residual_power_chunk[crosses_boundary])
                    )
                    support_audit_totals["support_width_sum_Hz"] += float(
                        np.sum(support_width)
                    )
                    support_audit_totals[
                        "support_width_power_sum_Hz_strain2"
                    ] += float(np.sum(support_width * residual_power_chunk))
                    support_audit_totals["support_width_max_Hz"] = max(
                        support_audit_totals["support_width_max_Hz"],
                        float(np.max(support_width)),
                    )
                    support_audit_totals["support_without_fourier_mode_count"] += int(
                        np.count_nonzero(mode_stop <= mode_start)
                    )
                if residual_source_power is not None:
                    for bin_id in np.unique(selected_bins[cache_mask]):
                        residual_source_power[int(bin_id)].extend(
                            power[cache_mask & (selected_bins == bin_id)]
                            .astype(float)
                            .tolist()
                        )
            print(f"  {dataset_path}: processed {len(dataset)} sources")

    if recovered_seen != expected_recovered_count:
        raise RuntimeError(
            "Recovered-source audit mismatch: "
            f"encountered {recovered_seen}, expected {expected_recovered_count}."
        )

    summary = bins.copy()
    summary["f_center_mHz"] = summary["f_center_Hz"] * 1.0e3
    summary["analysis_channel"] = "long_wavelength_michelson_X_equivalent_strain"
    summary["response_model"] = "orbit_averaged_long_wavelength_michelson_x_power"
    summary["orbit_model"] = "LDC_equal_arm_first_order"
    summary["orbit_guiding_center_phase_kappa_rad"] = 0.0
    summary["orbit_constellation_phase_lambda_rad"] = 0.0
    summary["orbit_average_role"] = "catalog_antenna_power_only"
    summary["detector_plane"] = "lisa_plane_inclined_60_deg_to_ecliptic"
    summary["response_phase_count"] = args.response_phase_count
    summary["response_phase_quadrature"] = "equally_spaced_over_one_orbit"
    summary["constellation_rotation_rad"] = 0.0
    summary["instrument_noise_model"] = "nominal_single_michelson_long_wavelength"
    summary["power_convention"] = "single_michelson_equivalent_incident_strain"
    summary["bandpower_statistic"] = "integrated_one_sided_periodogram"
    summary["bandpower_covariance_model"] = (
        "independent_positive_frequency_gaussian_modes"
    )
    summary["bandpower_cross_bin_covariance"] = "diagonal"
    summary["fisher_covariance_treatment"] = "fixed_at_fiducial_model"
    summary["residual_spectral_model"] = "piecewise_constant_within_coarse_bin"
    summary["random_phase_fourth_moment_used_in_covariance"] = False
    summary["LWA_X_averaged_response"] = response
    summary["lisa_arm_length_m"] = args.lisa_arm_length_m
    summary["h0_km_s_Mpc"] = args.h0_km_s_Mpc
    summary["sgwb_reference_frequency_Hz"] = args.sgwb_fref_Hz
    summary["omega0_fiducial"] = args.omega0
    summary["beta_fiducial"] = args.beta_fiducial
    summary["observation_metadata_dataset"] = args.observation_dataset
    summary["observation_sample_count"] = n_samples
    summary["observation_cadence_s"] = cadence_s
    summary["observation_duration_s"] = observation_duration_s
    summary["fourier_spacing_Hz"] = fourier_grid.fourier_spacing_hz
    summary["fourier_mode_count"] = fourier_grid.mode_count
    summary["bin_width_Hz"] = args.bin_width_Hz
    for prefix in ("inj", "rec", "res"):
        summary[f"LWA_X_{prefix}_n_used_rows"] = accumulators[f"{prefix}_count"]
        summary[f"LWA_X_{prefix}_power_sum"] = accumulators[f"{prefix}_power_sum"]
        summary[f"LWA_X_{prefix}_power_sum2"] = accumulators[f"{prefix}_power_sum2"]
        summary[f"LWA_X_{prefix}_C2"] = source_power_concentration(
            accumulators[f"{prefix}_power_sum"],
            accumulators[f"{prefix}_power_sum2"],
        )
        summary[f"LWA_X_{prefix}_N_eff_2"] = np.where(
            summary[f"LWA_X_{prefix}_C2"] > 0.0,
            1.0 / summary[f"LWA_X_{prefix}_C2"],
            np.nan,
        )
        summary[f"LWA_X_{prefix}_p_max"] = accumulators[f"{prefix}_power_max"]

    injected_power = summary["LWA_X_inj_power_sum"].to_numpy(float)
    recovered_power = summary["LWA_X_rec_power_sum"].to_numpy(float)
    residual_power = summary["LWA_X_res_power_sum"].to_numpy(float)
    flat_residual_mode_power = (
        fourier_grid.fourier_spacing_hz
        * piecewise_constant_residual_psd(fourier_grid, residual_power)
    )
    residual_mode_profiles = {
        "piecewise_constant_coarse_bin": flat_residual_mode_power,
        "catalog_frequency_support": residual_mode_power_support,
        "nearest_fourier_mode": residual_mode_power_nearest,
    }
    for profile_name, mode_power in residual_mode_profiles.items():
        np.testing.assert_allclose(
            np.sum(mode_power),
            np.sum(residual_power),
            rtol=5.0e-12,
            atol=0.0,
            err_msg=f"Residual power is not conserved for {profile_name}.",
        )
    summary["LWA_X_residual_power_fraction"] = np.divide(
        residual_power,
        injected_power,
        out=np.full_like(residual_power, np.nan),
        where=injected_power > 0.0,
    )
    summary["LWA_X_recovered_power_fraction"] = np.divide(
        recovered_power,
        injected_power,
        out=np.full_like(recovered_power, np.nan),
        where=injected_power > 0.0,
    )

    instrument_psd = single_michelson_lwa_equivalent_strain_psd(
        fourier_grid.frequency_hz,
        args.lisa_arm_length_m,
    )
    summary["P_inst_bin"] = np.nan
    profile_sensitivity_rows: list[dict[str, float | int | str | bool]] = []
    covariance_ensemble_rows: list[dict[str, float | int | str | bool]] = []
    noise_sensitivity_rows: list[dict[str, float | int | str | bool]] = []
    for alpha in args.alpha_list:
        suffix = alpha_column_suffix(alpha)
        template_psd = sgwb_strain_psd_per_omega(
            fourier_grid.frequency_hz,
            alpha,
            args.sgwb_fref_Hz,
            args.h0_km_s_Mpc,
        )
        models = {
            profile_name: build_gaussian_catalog_bandpower_model_from_mode_power(
                fourier_grid,
                instrument_psd,
                template_psd,
                mode_power,
                omega0=args.omega0,
                beta=args.beta_fiducial,
            )
            for profile_name, mode_power in residual_mode_profiles.items()
        }
        fixed_residual_models = {
            profile_name: build_fixed_residual_catalog_bandpower_model_from_mode_power(
                fourier_grid,
                instrument_psd,
                template_psd,
                mode_power,
                omega0=args.omega0,
                beta=args.beta_fiducial,
            )
            for profile_name, mode_power in residual_mode_profiles.items()
        }
        model = models["piecewise_constant_coarse_bin"]
        summary["P_inst_bin"] = model.instrument_power
        summary[f"T_alpha_{suffix}"] = model.sgwb_template
        summary[f"P_sgwb_fid_alpha_{suffix}"] = args.omega0 * model.sgwb_template
        summary[f"bandpower_mean_fid_alpha_{suffix}"] = model.mean
        summary[f"bandpower_variance_gaussian_alpha_{suffix}"] = model.variance
        summary[f"n_eff_gaussian_alpha_{suffix}"] = model.effective_degrees_of_freedom

        for profile_name, profile_model in models.items():
            fixed_residual_model = fixed_residual_models[profile_name]
            np.testing.assert_allclose(
                fixed_residual_model.mean,
                profile_model.mean,
                rtol=5.0e-14,
                atol=0.0,
                err_msg=(
                    "Fixed and Gaussian residual ensembles must have the same "
                    f"mean for {profile_name}."
                ),
            )
            variance_ratio = np.divide(
                fixed_residual_model.variance,
                profile_model.variance,
                out=np.full_like(profile_model.variance, np.nan),
                where=profile_model.variance > 0.0,
            )
            for bin_set, mask in (
                (
                    "all_analysis_bins",
                    np.ones(profile_model.residual_power.size, dtype=bool),
                ),
                (
                    "nonzero_profile_residual_bins",
                    profile_model.residual_power > 0.0,
                ),
            ):
                result = two_parameter_fisher_summary(
                    profile_model.sgwb_template[mask],
                    profile_model.residual_power[mask],
                    profile_model.variance[mask],
                    omega0=args.omega0,
                    beta_fiducial=args.beta_fiducial,
                    uncertainty_threshold=args.inflation_threshold,
                )
                result.update(
                    {
                        "residual_frequency_profile": profile_name,
                        "alpha": float(alpha),
                        "bin_set": bin_set,
                        "n_bins": int(np.count_nonzero(mask)),
                        "n_nonzero_residual_bins": int(
                            np.count_nonzero(profile_model.residual_power > 0.0)
                        ),
                        "residual_power_band_sum": float(
                            np.sum(profile_model.residual_power)
                        ),
                        "n_eff_bandpower_min": float(
                            np.min(profile_model.effective_degrees_of_freedom[mask])
                        ),
                        "n_eff_bandpower_max": float(
                            np.max(profile_model.effective_degrees_of_freedom[mask])
                        ),
                        "independent_gaussian_fourier_modes": True,
                        "coherent_sidebands_included": False,
                    }
                )
                profile_sensitivity_rows.append(result)

                gaussian_endpoint = dict(result)
                gaussian_endpoint.update(
                    {
                        "covariance_endpoint": "gaussian_residual_ensemble",
                        "residual_self_noise_included": True,
                        "conditional_on_fixed_residual_amplitudes": False,
                        "stochastic_fourier_amplitudes_gaussian": True,
                        "fixed_to_gaussian_variance_ratio_min": float(
                            np.min(variance_ratio[mask])
                        ),
                        "fixed_to_gaussian_variance_ratio_median": float(
                            np.median(variance_ratio[mask])
                        ),
                        "fixed_to_gaussian_variance_ratio_max": float(
                            np.max(variance_ratio[mask])
                        ),
                    }
                )
                covariance_ensemble_rows.append(gaussian_endpoint)

                fixed_result = two_parameter_fisher_summary(
                    fixed_residual_model.sgwb_template[mask],
                    fixed_residual_model.residual_power[mask],
                    fixed_residual_model.variance[mask],
                    omega0=args.omega0,
                    beta_fiducial=args.beta_fiducial,
                    uncertainty_threshold=args.inflation_threshold,
                )
                fixed_result.update(
                    {
                        "residual_frequency_profile": profile_name,
                        "alpha": float(alpha),
                        "bin_set": bin_set,
                        "n_bins": int(np.count_nonzero(mask)),
                        "n_nonzero_residual_bins": int(
                            np.count_nonzero(fixed_residual_model.residual_power > 0.0)
                        ),
                        "residual_power_band_sum": float(
                            np.sum(fixed_residual_model.residual_power)
                        ),
                        "n_eff_bandpower_min": float(
                            np.min(
                                fixed_residual_model.effective_degrees_of_freedom[
                                    mask
                                ]
                            )
                        ),
                        "n_eff_bandpower_max": float(
                            np.max(
                                fixed_residual_model.effective_degrees_of_freedom[
                                    mask
                                ]
                            )
                        ),
                        "independent_gaussian_fourier_modes": False,
                        "coherent_sidebands_included": False,
                        "covariance_endpoint": "fixed_residual_realization",
                        "residual_self_noise_included": False,
                        "conditional_on_fixed_residual_amplitudes": True,
                        "stochastic_fourier_amplitudes_gaussian": True,
                        "fixed_to_gaussian_variance_ratio_min": float(
                            np.min(variance_ratio[mask])
                        ),
                        "fixed_to_gaussian_variance_ratio_median": float(
                            np.median(variance_ratio[mask])
                        ),
                        "fixed_to_gaussian_variance_ratio_max": float(
                            np.max(variance_ratio[mask])
                        ),
                    }
                )
                covariance_ensemble_rows.append(fixed_result)

            if np.isclose(alpha, 0.0):
                for noise_prior_sigma in args.noise_normalization_prior_list:
                    result = instrument_noise_normalization_summary(
                        profile_model.sgwb_template,
                        profile_model.residual_power,
                        profile_model.instrument_power,
                        profile_model.variance,
                        omega0=args.omega0,
                        noise_normalization_prior_sigma=noise_prior_sigma,
                        uncertainty_threshold=args.inflation_threshold,
                    )
                    result.update(
                        {
                            "residual_frequency_profile": profile_name,
                            "alpha": float(alpha),
                            "n_bins": int(profile_model.residual_power.size),
                            "instrument_noise_parameter": (
                                "overall_instrumental_power_normalization"
                            ),
                            "frequency_dependent_noise_shape_fixed": True,
                        }
                    )
                    noise_sensitivity_rows.append(result)

    zero_suffix = alpha_column_suffix(0.0)
    if f"T_alpha_{zero_suffix}" not in summary:
        raise ValueError("alpha-list must include 0 for the diagnostic columns.")
    summary["P_sgwb_bin_fid"] = summary[f"P_sgwb_fid_alpha_{zero_suffix}"]
    summary["bandpower_mean_fid"] = summary[f"bandpower_mean_fid_alpha_{zero_suffix}"]
    summary["bandpower_variance_gaussian"] = summary[
        f"bandpower_variance_gaussian_alpha_{zero_suffix}"
    ]
    summary["n_eff_bandpower"] = summary[f"n_eff_gaussian_alpha_{zero_suffix}"]

    gaussian_power = (
        summary["P_inst_bin"].to_numpy(float)
        + summary["P_sgwb_bin_fid"].to_numpy(float)
    )
    total_power = gaussian_power + args.beta_fiducial * residual_power
    residual_fraction_in_total = np.divide(
        args.beta_fiducial * residual_power,
        total_power,
        out=np.zeros_like(residual_power),
        where=total_power > 0.0,
    )
    residual_c2 = summary["LWA_X_res_C2"].to_numpy(float)
    gamma_residual = np.where(np.isfinite(residual_c2), -1.5 * residual_c2, 0.0)
    summary["pred_excess_kurtosis"] = gamma_residual
    summary["residual_fraction_in_total_power"] = residual_fraction_in_total
    summary["gamma2_total"] = gamma_residual * residual_fraction_in_total**2
    summary["abs_gamma2_total"] = np.abs(summary["gamma2_total"])
    summary["gamma2_total_role"] = (
        "response_averaged_random_phase_source_sum_diagnostic"
    )

    fisher_rows: list[dict[str, float | int | str]] = []
    for alpha in args.alpha_list:
        suffix = alpha_column_suffix(alpha)
        for set_name, rows in build_bin_sets(summary, args.top_n_list).items():
            result = fisher_for_rows(
                rows,
                alpha=alpha,
                omega0=args.omega0,
                beta_fiducial=args.beta_fiducial,
                uncertainty_threshold=args.inflation_threshold,
            )
            result.update(
                {
                    "bin_set": set_name,
                    "alpha": float(alpha),
                    "omega0": args.omega0,
                    "n_bins": int(len(rows)),
                    "n_nonzero_residual_bins": int(
                        (rows["LWA_X_res_power_sum"] > 0.0).sum()
                    ),
                    "n_eff_bandpower_min": float(
                        rows[f"n_eff_gaussian_alpha_{suffix}"].min()
                    ),
                    "n_eff_bandpower_max": float(
                        rows[f"n_eff_gaussian_alpha_{suffix}"].max()
                    ),
                }
            )
            fisher_rows.append(result)
    fisher = pd.DataFrame(fisher_rows)
    profile_sensitivity = pd.DataFrame(profile_sensitivity_rows)
    covariance_ensemble_sensitivity = pd.DataFrame(covariance_ensemble_rows)
    noise_sensitivity = pd.DataFrame(noise_sensitivity_rows)
    support_residual_power = np.bincount(
        fourier_grid.bin_index,
        weights=residual_mode_power_support,
        minlength=len(summary),
    )
    nearest_residual_power = np.bincount(
        fourier_grid.bin_index,
        weights=residual_mode_power_nearest,
        minlength=len(summary),
    )
    total_residual_source_power = float(support_audit_totals["source_power"])
    if total_residual_source_power <= 0.0:
        raise RuntimeError("The catalog contains no positive residual source power.")
    original_zero = residual_power == 0.0
    frequency_support_audit = pd.DataFrame(
        [
            {
                "frequency_support_definition": (
                    "initial_to_final_intrinsic_frequency_plus_annual_doppler_half_width"
                ),
                "catalog_frequency_reference": "LDC_initial_frequency_at_t0",
                "observation_duration_s": observation_duration_s,
                "fourier_spacing_Hz": fourier_grid.fourier_spacing_hz,
                "coarse_bin_width_Hz": args.bin_width_Hz,
                "residual_source_count": int(support_audit_totals["source_count"]),
                "residual_source_power": total_residual_source_power,
                "boundary_crossing_source_count": int(
                    support_audit_totals["boundary_crossing_source_count"]
                ),
                "boundary_crossing_source_fraction": (
                    support_audit_totals["boundary_crossing_source_count"]
                    / support_audit_totals["source_count"]
                ),
                "boundary_crossing_source_power": float(
                    support_audit_totals["boundary_crossing_source_power"]
                ),
                "boundary_crossing_power_fraction": (
                    support_audit_totals["boundary_crossing_source_power"]
                    / total_residual_source_power
                ),
                "mean_support_width_Hz": (
                    support_audit_totals["support_width_sum_Hz"]
                    / support_audit_totals["source_count"]
                ),
                "power_weighted_mean_support_width_Hz": (
                    support_audit_totals["support_width_power_sum_Hz_strain2"]
                    / total_residual_source_power
                ),
                "maximum_support_width_Hz": float(
                    support_audit_totals["support_width_max_Hz"]
                ),
                "support_without_fourier_mode_count": int(
                    support_audit_totals["support_without_fourier_mode_count"]
                ),
                "piecewise_constant_nonzero_bins": int(
                    np.count_nonzero(residual_power > 0.0)
                ),
                "catalog_support_nonzero_bins": int(
                    np.count_nonzero(support_residual_power > 0.0)
                ),
                "nearest_mode_nonzero_bins": int(
                    np.count_nonzero(nearest_residual_power > 0.0)
                ),
                "zero_catalog_bins_receiving_support_power": int(
                    np.count_nonzero(original_zero & (support_residual_power > 0.0))
                ),
                "zero_catalog_bins_receiving_nearest_mode_power": int(
                    np.count_nonzero(original_zero & (nearest_residual_power > 0.0))
                ),
                "catalog_support_power_moved_between_bins_fraction": float(
                    0.5
                    * np.sum(np.abs(support_residual_power - residual_power))
                    / total_residual_source_power
                ),
                "nearest_mode_power_moved_between_bins_fraction": float(
                    0.5
                    * np.sum(np.abs(nearest_residual_power - residual_power))
                    / total_residual_source_power
                ),
                "source_power_conserved_within_analysis_band": True,
                "coherent_sidebands_included": False,
            }
        ]
    )
    outputs = [
        Path(args.output_bin_summary),
        Path(args.output_noise_table),
        Path(args.output_fisher_table),
        Path(args.output_fisher_alpha0_summary_csv),
        Path(args.output_fisher_alpha0_summary_tex),
        Path(args.output_residual_profile_sensitivity),
        Path(args.output_covariance_ensemble_sensitivity),
        Path(args.output_residual_mode_profiles),
        Path(args.output_frequency_support_audit),
        Path(args.output_noise_normalization_sensitivity),
    ]
    if not args.skip_random_phase_check:
        outputs.extend(
            [
                Path(args.output_validation_table),
                Path(args.output_validation_summary),
                Path(args.output_validation_figure_png),
                Path(args.output_validation_figure_pdf),
            ]
        )
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(args.output_bin_summary, index=False)
    noise_columns = [
        "frequency_bin_index",
        "f_center_Hz",
        "f_center_mHz",
        "fourier_mode_count",
        "P_inst_bin",
        "P_sgwb_bin_fid",
        "LWA_X_res_power_sum",
        "LWA_X_res_C2",
        "pred_excess_kurtosis",
        "residual_fraction_in_total_power",
        "gamma2_total",
        "abs_gamma2_total",
        "bandpower_variance_gaussian",
        "n_eff_bandpower",
    ]
    summary[noise_columns].to_csv(args.output_noise_table, index=False)
    fisher.to_csv(args.output_fisher_table, index=False)
    profile_sensitivity.to_csv(args.output_residual_profile_sensitivity, index=False)
    covariance_ensemble_sensitivity.to_csv(
        args.output_covariance_ensemble_sensitivity,
        index=False,
    )
    np.savez_compressed(
        args.output_residual_mode_profiles,
        frequency_hz=fourier_grid.frequency_hz,
        bin_index=fourier_grid.bin_index,
        mode_count=fourier_grid.mode_count,
        bin_edges_hz=fourier_grid.bin_edges_hz,
        fourier_spacing_hz=np.asarray(fourier_grid.fourier_spacing_hz),
        **residual_mode_profiles,
    )
    frequency_support_audit.to_csv(args.output_frequency_support_audit, index=False)
    noise_sensitivity.to_csv(args.output_noise_normalization_sensitivity, index=False)
    write_alpha0_summary(
        fisher,
        Path(args.output_fisher_alpha0_summary_csv),
        Path(args.output_fisher_alpha0_summary_tex),
    )
    if not args.skip_random_phase_check:
        if residual_source_power is None:
            raise RuntimeError("Random-phase source powers were not retained.")
        moment_table, moment_summary = random_phase_moment_check(
            summary,
            residual_source_power,
            args,
        )
        moment_table.to_csv(args.output_validation_table, index=False)
        moment_summary.to_csv(args.output_validation_summary, index=False)
        save_random_phase_figure(
            moment_table,
            Path(args.output_validation_figure_png),
            Path(args.output_validation_figure_pdf),
            args.dpi,
        )

    print("\nCatalog bandpower baseline:")
    print(
        "  analysis statistic: orbit-averaged long-wavelength Michelson-X "
        "equivalent incident strain power"
    )
    print(f"  response quadrature phases: {args.response_phase_count}")
    print(f"  LISA-plane inclination: {np.degrees(inclination):.12g} deg")
    print(f"  averaged single-polarization response: {response:.12g}")
    print(f"  observation duration: {observation_duration_s:.12g} s")
    print(f"  Fourier spacing: {fourier_grid.fourier_spacing_hz:.12g} Hz")
    print(f"  Fourier frequencies per analysis bin: {sorted(np.unique(fourier_grid.mode_count))}")
    print(
        "  alpha=0 derived n_eff range: "
        f"[{summary['n_eff_bandpower'].min():.12g}, "
        f"{summary['n_eff_bandpower'].max():.12g}]"
    )
    print(f"  nonzero residual bins: {(residual_power > 0.0).sum()}/{len(summary)}")
    print(
        "  residual sources crossing an analysis-bin boundary: "
        f"{int(support_audit_totals['boundary_crossing_source_count'])}/"
        f"{int(support_audit_totals['source_count'])}"
    )
    print(
        "  residual power carried by boundary-crossing sources: "
        f"{frequency_support_audit.iloc[0]['boundary_crossing_power_fraction']:.6%}"
    )
    print(
        "  residual power moved between analysis bins when each source is "
        "distributed within its drift and Doppler interval: "
        f"{frequency_support_audit.iloc[0]['catalog_support_power_moved_between_bins_fraction']:.6%}"
    )
    print(f"  recovered-source matching closure: {recovered_seen}/{expected_recovered_count}")
    print("  random-phase fourth moment used in Fisher covariance: no")
    print(f"Saved bin summary: {args.output_bin_summary}")
    print(f"Saved Fisher summary: {args.output_fisher_table}")
    print(f"Saved within-bin frequency sensitivity: {args.output_residual_profile_sensitivity}")
    print(
        "Saved fixed-versus-Gaussian residual covariance endpoints: "
        f"{args.output_covariance_ensemble_sensitivity}"
    )
    print(f"Saved residual Fourier-mode powers: {args.output_residual_mode_profiles}")
    print(f"Saved source frequency-interval audit: {args.output_frequency_support_audit}")
    print(
        "Saved instrumental-power-factor sensitivity: "
        f"{args.output_noise_normalization_sensitivity}"
    )
    if args.skip_random_phase_check:
        print("Random-phase Monte Carlo products retained without modification.")


if __name__ == "__main__":
    main()
