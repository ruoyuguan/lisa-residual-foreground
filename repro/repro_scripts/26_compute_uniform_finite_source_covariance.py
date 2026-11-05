"""Finite-source random-phase covariance for the uniform mode distribution."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lisa_catalog_statistics import (
    FourierBandGrid,
    build_gaussian_catalog_bandpower_model,
    build_uniform_random_phase_catalog_bandpower_model,
    sgwb_strain_psd_per_omega,
    single_michelson_lwa_equivalent_strain_psd,
    two_parameter_fisher_summary,
)


DEFAULT_SUMMARY = (
    "repro/tables/all_bins_LWA_X_summary_C0p9_overlap0p9_f0p4_6mHz.csv"
)
DEFAULT_MODE_CACHE = (
    "repro/sensitivity/residual_fourier_profile/residual_mode_power_profiles.npz"
)
DEFAULT_OUTPUT = (
    "repro/sensitivity/residual_covariance_ensemble/"
    "uniform_finite_source_random_phase_covariance.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-bin-summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--mode-power-profiles", default=DEFAULT_MODE_CACHE)
    parser.add_argument("--output-table", default=DEFAULT_OUTPUT)
    parser.add_argument("--alpha-list", nargs="+", type=float, default=[-2.0, 0.0, 2.0])
    parser.add_argument("--uncertainty-threshold", type=float, default=1.1)
    return parser.parse_args()


def constant_metadata(table: pd.DataFrame, column: str) -> float:
    if column not in table:
        raise KeyError(f"Required metadata column is missing: {column}")
    values = table[column].to_numpy(float)
    if not np.all(np.isfinite(values)) or not np.allclose(values, values[0]):
        raise ValueError(f"Metadata column must be finite and constant: {column}")
    return float(values[0])


def load_grid(path: Path) -> FourierBandGrid:
    with np.load(path, allow_pickle=False) as cache:
        return FourierBandGrid(
            frequency_hz=cache["frequency_hz"].copy(),
            bin_index=cache["bin_index"].astype(int),
            mode_count=cache["mode_count"].astype(int),
            fourier_spacing_hz=float(cache["fourier_spacing_hz"]),
            bin_edges_hz=cache["bin_edges_hz"].copy(),
        )


def main() -> None:
    args = parse_args()
    summary_path = Path(args.all_bin_summary)
    mode_cache_path = Path(args.mode_power_profiles)
    output_path = Path(args.output_table)

    summary = pd.read_csv(summary_path)
    grid = load_grid(mode_cache_path)
    if len(summary) != grid.mode_count.size:
        raise ValueError("The bin summary and Fourier-mode cache have different bin counts.")
    expected_index = np.arange(grid.mode_count.size)
    np.testing.assert_array_equal(
        summary["frequency_bin_index"].to_numpy(int),
        expected_index,
        err_msg="The bin summary is not in Fourier-grid order.",
    )
    np.testing.assert_allclose(
        summary["fourier_mode_count"].to_numpy(int),
        grid.mode_count,
        rtol=0.0,
        atol=0.0,
        err_msg="Fourier-mode counts differ between the cached products.",
    )

    residual_power = summary["LWA_X_res_power_sum"].to_numpy(float)
    concentration = summary["LWA_X_res_C2"].to_numpy(float)
    omega0 = constant_metadata(summary, "omega0_fiducial")
    beta = constant_metadata(summary, "beta_fiducial")
    arm_length_m = constant_metadata(summary, "lisa_arm_length_m")
    h0_km_s_mpc = constant_metadata(summary, "h0_km_s_Mpc")
    reference_frequency_hz = constant_metadata(
        summary,
        "sgwb_reference_frequency_Hz",
    )
    instrument_psd = single_michelson_lwa_equivalent_strain_psd(
        grid.frequency_hz,
        arm_length_m,
    )

    rows: list[dict[str, float | int | str | bool]] = []
    for alpha in args.alpha_list:
        template_psd = sgwb_strain_psd_per_omega(
            grid.frequency_hz,
            alpha=float(alpha),
            reference_frequency_hz=reference_frequency_hz,
            h0_km_s_mpc=h0_km_s_mpc,
        )
        finite_source = build_uniform_random_phase_catalog_bandpower_model(
            grid,
            instrument_psd,
            template_psd,
            residual_power,
            concentration,
            omega0=omega0,
            beta=beta,
        )
        gaussian = build_gaussian_catalog_bandpower_model(
            grid,
            instrument_psd,
            template_psd,
            residual_power,
            omega0=omega0,
            beta=beta,
        )
        variance_ratio = finite_source.variance / gaussian.variance

        for bin_set, mask in (
            ("all_analysis_bins", np.ones(residual_power.size, dtype=bool)),
            ("nonzero_residual_bins", residual_power > 0.0),
        ):
            result = two_parameter_fisher_summary(
                finite_source.sgwb_template[mask],
                finite_source.residual_power[mask],
                finite_source.variance[mask],
                omega0=omega0,
                beta_fiducial=beta,
                uncertainty_threshold=args.uncertainty_threshold,
            )
            result.update(
                {
                    "residual_frequency_profile": "piecewise_constant_coarse_bin",
                    "covariance_treatment": "finite_source_random_phase_diagonal",
                    "source_phase_model": "independent_uniform_phases",
                    "mode_source_concentration": "bin_source_power_concentration",
                    "cross_frequency_covariance_included": False,
                    "alpha": float(alpha),
                    "bin_set": bin_set,
                    "n_bins": int(np.count_nonzero(mask)),
                    "n_nonzero_residual_bins": int(np.count_nonzero(residual_power > 0.0)),
                    "finite_source_to_gaussian_variance_ratio_min": float(
                        np.min(variance_ratio[mask])
                    ),
                    "finite_source_to_gaussian_variance_ratio_median": float(
                        np.median(variance_ratio[mask])
                    ),
                    "finite_source_to_gaussian_variance_ratio_max": float(
                        np.max(variance_ratio[mask])
                    ),
                }
            )
            rows.append(result)

    output = pd.DataFrame(rows).sort_values(["alpha", "bin_set"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    alpha_zero = output.loc[
        np.isclose(output["alpha"], 0.0)
        & (output["bin_set"] == "all_analysis_bins")
    ].iloc[0]
    print("Uniform finite-source random-phase diagonal covariance:")
    print(f"  I0 = {alpha_zero['no_prior_inflation']:.10f}")
    print(
        "  uncertainty increase = "
        f"{100.0 * (alpha_zero['no_prior_inflation'] - 1.0):.6f}%"
    )
    print(f"  sigma_beta(10%) = {alpha_zero['sigma_beta_for_threshold']:.10g}")
    print(f"  output = {output_path}")


if __name__ == "__main__":
    main()
