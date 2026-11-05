#!/usr/bin/env python
"""Compare frequency-bin widths with the catalog bandpower calculation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sangria-h5",
        type=Path,
        default=ROOT / "data/raw/sangria/LDC2_sangria_training_v2.h5",
    )
    parser.add_argument(
        "--matching-audit",
        type=Path,
        default=(
            ROOT
            / "repro/tables/erebor_sangria_recovered_source_matching_audit_"
            "C0p9_overlap0p9_f0p4_6mHz.csv"
        ),
    )
    parser.add_argument("--frequency-min-mHz", type=float, default=0.4)
    parser.add_argument("--frequency-max-mHz", type=float, default=6.0)
    parser.add_argument("--widths-uHz", nargs="+", type=float, default=[5.0, 10.0, 20.0])
    parser.add_argument("--response-phase-count", type=int, default=12)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "repro/sensitivity/bin_width",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=ROOT / "repro/sensitivity/bin_width/bin_width_sensitivity_summary.csv",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def width_tag(width_uhz: float) -> str:
    return f"df_{width_uhz:g}uHz".replace(".", "p")


def output_arguments(output_dir: Path) -> tuple[list[str], Path, Path]:
    tables = output_dir / "tables"
    summary = tables / "all_bins_LWA_X_summary.csv"
    fisher = tables / "all_bins_fisher_baseline_LWA_X.csv"
    arguments = [
        "--output-bin-summary", str(summary),
        "--output-noise-table", str(tables / "all_bins_noise_dilution_LWA_X.csv"),
        "--output-fisher-table", str(fisher),
        "--output-fisher-alpha0-summary-csv", str(tables / "all_bins_fisher_alpha0_summary_LWA_X.csv"),
        "--output-fisher-alpha0-summary-tex", str(tables / "all_bins_fisher_alpha0_summary_LWA_X.tex"),
        "--output-residual-profile-sensitivity", str(
            tables / "within_bin_frequency_sensitivity.csv"
        ),
        "--output-covariance-ensemble-sensitivity", str(
            tables / "fixed_vs_gaussian_residual_covariance.csv"
        ),
        "--output-residual-mode-profiles", str(
            tables / "residual_mode_power_profiles.npz"
        ),
        "--output-frequency-support-audit", str(
            tables / "source_frequency_interval_audit.csv"
        ),
        "--output-noise-normalization-sensitivity", str(
            tables / "instrument_power_factor_sensitivity.csv"
        ),
    ]
    return arguments, summary, fisher


def summarize_width(
    width_uhz: float,
    summary_path: Path,
    fisher_path: Path,
) -> dict[str, float | int | str]:
    bins = pd.read_csv(summary_path)
    fisher = pd.read_csv(fisher_path)
    all_bins = fisher[
        (fisher["bin_set"] == "all_analysis_bins")
        & np.isclose(fisher["alpha"], 0.0)
    ]
    nonzero_bins = fisher[
        (fisher["bin_set"] == "all_nonzero_residual_bins")
        & np.isclose(fisher["alpha"], 0.0)
    ]
    if len(all_bins) != 1 or len(nonzero_bins) != 1:
        raise ValueError(f"Unexpected all-bin Fisher rows for {width_uhz:g} uHz")
    all_row = all_bins.iloc[0]
    nonzero_row = nonzero_bins.iloc[0]
    nonzero = bins["LWA_X_res_power_sum"] > 0.0
    mode_count = bins["fourier_mode_count"].to_numpy(float)
    effective_dof = bins["n_eff_bandpower"].to_numpy(float)
    if np.any(effective_dof > 2.0 * mode_count + 1.0e-10):
        raise RuntimeError("Derived bandpower degrees of freedom exceed 2N_mode.")
    return {
        "bin_width_uHz": width_uhz,
        "bin_width_Hz": float(bins["bin_width_Hz"].iloc[0]),
        "analysis_channel": str(bins["analysis_channel"].iloc[0]),
        "response_model": str(bins["response_model"].iloc[0]),
        "response_phase_count": int(bins["response_phase_count"].iloc[0]),
        "observation_duration_s": float(bins["observation_duration_s"].iloc[0]),
        "fourier_mode_count_min": int(mode_count.min()),
        "fourier_mode_count_max": int(mode_count.max()),
        "n_eff_bandpower_min": float(effective_dof.min()),
        "n_eff_bandpower_max": float(effective_dof.max()),
        "n_eff_bandpower_mean": float(effective_dof.mean()),
        "n_analysis_bins": int(len(bins)),
        "n_nonzero_residual_bins": int(nonzero.sum()),
        "n_zero_residual_bins": int((~nonzero).sum()),
        "residual_power_band_sum": float(bins["LWA_X_res_power_sum"].sum()),
        "instrument_power_band_sum": float(bins["P_inst_bin"].sum()),
        "fiducial_sgwb_power_band_sum": float(bins["P_sgwb_bin_fid"].sum()),
        "median_abs_gamma2_total_nonzero": float(
            bins.loc[nonzero, "abs_gamma2_total"].median()
        ),
        "max_abs_gamma2_total_nonzero": float(
            bins.loc[nonzero, "abs_gamma2_total"].max()
        ),
        "all_bin_abs_rho_alpha0": float(all_row["no_prior_abs_corr"]),
        "all_bin_I0_alpha0": float(all_row["no_prior_inflation"]),
        "all_bin_sigma_beta_10pct_alpha0": float(
            all_row["sigma_beta_for_threshold"]
        ),
        "all_bin_sigma_omega_fixed_alpha0": float(all_row["sigma_omega_fixed"]),
        "all_bin_fixed_beta_snr_alpha0": float(all_row["snr_omega_fixed"]),
        "all_bin_bias_over_sigma_fixed_alpha0": float(
            all_row["bias_ignore_over_sigma_fixed"]
        ),
        "nonzero_bin_I0_alpha0": float(nonzero_row["no_prior_inflation"]),
        "nonzero_bin_sigma_beta_10pct_alpha0": float(
            nonzero_row["sigma_beta_for_threshold"]
        ),
    }


def main() -> None:
    args = parse_args()
    if args.response_phase_count < 1:
        raise ValueError("response_phase_count must be positive.")
    if not 0.0 < args.frequency_min_mHz < args.frequency_max_mHz:
        raise ValueError("The frequency range must be positive and increasing.")
    for path in (args.sangria_h5, args.matching_audit):
        if not path.is_file():
            raise FileNotFoundError(path)

    records = []
    for width_uhz in args.widths_uHz:
        if not np.isfinite(width_uhz) or width_uhz <= 0.0:
            raise ValueError(f"Invalid bin width: {width_uhz}")
        width_hz = width_uhz * 1.0e-6
        output_dir = args.output_root / width_tag(width_uhz)
        output_args, summary_path, fisher_path = output_arguments(output_dir)
        run(
            [
                sys.executable,
                str(SCRIPTS / "20_compute_catalog_bandpower_baseline.py"),
                "--sangria-h5", str(args.sangria_h5),
                "--datasets", "sky/dgb/cat", "sky/igb/cat",
                "--matching-audit", str(args.matching_audit),
                "--frequency-min-mHz", f"{args.frequency_min_mHz:.17g}",
                "--frequency-max-mHz", f"{args.frequency_max_mHz:.17g}",
                "--bin-width-Hz", f"{width_hz:.17g}",
                "--response-phase-count", str(args.response_phase_count),
                "--skip-random-phase-check",
                *output_args,
            ]
        )
        records.append(summarize_width(width_uhz, summary_path, fisher_path))

    comparison = pd.DataFrame(records).sort_values("bin_width_uHz")
    residual_totals = comparison["residual_power_band_sum"].to_numpy(float)
    relative_residual_spread = np.ptp(residual_totals) / np.max(
        np.abs(residual_totals)
    )
    if relative_residual_spread > 1.0e-12:
        raise RuntimeError(
            "Residual band power is not conserved across frequency-bin widths: "
            f"relative spread={relative_residual_spread:.3e}."
        )
    covered_bandwidth = (
        comparison["n_analysis_bins"].to_numpy(float)
        * comparison["bin_width_Hz"].to_numpy(float)
    )
    expected_bandwidth = (args.frequency_max_mHz - args.frequency_min_mHz) * 1.0e-3
    if not np.allclose(
        covered_bandwidth,
        expected_bandwidth,
        rtol=0.0,
        atol=1.0e-14,
    ):
        raise RuntimeError("The sensitivity runs do not cover the same frequency band.")
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_summary, index=False)
    print("\nBin-width sensitivity summary:")
    print(comparison.to_string(index=False))
    print(f"\nSaved: {args.output_summary}")


if __name__ == "__main__":
    main()
