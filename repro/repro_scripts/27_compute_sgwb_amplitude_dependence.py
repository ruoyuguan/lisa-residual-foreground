"""All-bin dependence on the fiducial SGWB amplitude for alpha = 0."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from lisa_catalog_statistics import (
    FourierBandGrid,
    sgwb_strain_psd_per_omega,
    single_michelson_lwa_equivalent_strain_psd,
    uniform_sgwb_amplitude_sweep,
)


DEFAULT_SUMMARY = (
    "repro/tables/all_bins_LWA_X_summary_C0p9_overlap0p9_f0p4_6mHz.csv"
)
DEFAULT_MODE_CACHE = (
    "repro/sensitivity/residual_fourier_profile/residual_mode_power_profiles.npz"
)
DEFAULT_OUTPUT_TABLE = (
    "repro/tables/all_bins_sgwb_amplitude_dependence_"
    "LWA_X_C0p9_overlap0p9_f0p4_6mHz.csv"
)
DEFAULT_OUTPUT_PNG = (
    "repro/figures/all_bins/all_bins_sgwb_amplitude_dependence_"
    "LWA_X_C0p9_overlap0p9_f0p4_6mHz.png"
)
DEFAULT_OUTPUT_PDF = (
    "repro/figures/all_bins/all_bins_sgwb_amplitude_dependence_"
    "LWA_X_C0p9_overlap0p9_f0p4_6mHz.pdf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-bin-summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--mode-power-profiles", default=DEFAULT_MODE_CACHE)
    parser.add_argument("--output-table", default=DEFAULT_OUTPUT_TABLE)
    parser.add_argument("--output-png", default=DEFAULT_OUTPUT_PNG)
    parser.add_argument("--output-pdf", default=DEFAULT_OUTPUT_PDF)
    parser.add_argument("--omega0-min", type=float, default=1.0e-13)
    parser.add_argument("--omega0-max", type=float, default=1.0e-10)
    parser.add_argument("--n-amplitudes", type=int, default=61)
    parser.add_argument("--fiducial-omega0", type=float, default=1.0e-11)
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


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.width": 0.55,
            "ytick.minor.width": 0.55,
            "axes.linewidth": 0.7,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, png_path: Path, pdf_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def make_figure(
    table: pd.DataFrame,
    fiducial_omega0: float,
    uncertainty_threshold: float,
    output_png: Path,
    output_pdf: Path,
) -> None:
    setup_matplotlib()
    styles = {
        "gaussian_residual_amplitudes": {
            "label": "Gaussian residual amplitudes",
            "color": "#2b6f9e",
            "linestyle": "-",
        },
        "finite_source_random_phase_diagonal": {
            "label": "Finite-source random phase",
            "color": "#b13c2e",
            "linestyle": "--",
        },
    }

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 2.75),
        sharex=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.25, top=0.78, wspace=0.42)
    for treatment, style in styles.items():
        subset = table.loc[table["covariance_treatment"] == treatment].sort_values(
            "omega0_fiducial"
        )
        omega0 = subset["omega0_fiducial"].to_numpy(float)
        axes[0].plot(
            omega0,
            subset["no_prior_inflation"],
            color=style["color"],
            ls=style["linestyle"],
            lw=1.5,
            label=style["label"],
        )
        prior_width = subset["sigma_beta_for_threshold"].to_numpy(float)
        axes[1].plot(
            omega0,
            np.where(np.isfinite(prior_width), prior_width, np.nan),
            color=style["color"],
            ls=style["linestyle"],
            lw=1.5,
        )
        axes[2].plot(
            omega0,
            subset["sigma_omega_fixed_over_omega0"],
            color=style["color"],
            ls=style["linestyle"],
            lw=1.5,
        )

        fiducial = subset.loc[
            np.isclose(
                subset["omega0_fiducial"],
                fiducial_omega0,
                rtol=1.0e-12,
                atol=0.0,
            )
        ]
        if len(fiducial) != 1:
            raise ValueError("The amplitude grid must contain the fiducial value once.")
        row = fiducial.iloc[0]
        axes[0].plot(
            fiducial_omega0,
            row["no_prior_inflation"],
            marker="o",
            ms=3.6,
            color=style["color"],
            zorder=4,
        )
        if np.isfinite(row["sigma_beta_for_threshold"]):
            axes[1].plot(
                fiducial_omega0,
                row["sigma_beta_for_threshold"],
                marker="o",
                ms=3.6,
                color=style["color"],
                zorder=4,
            )
        axes[2].plot(
            fiducial_omega0,
            row["sigma_omega_fixed_over_omega0"],
            marker="o",
            ms=3.6,
            color=style["color"],
            zorder=4,
        )

    for ax in axes:
        ax.set_xscale("log")
        ax.axvline(fiducial_omega0, color="0.45", lw=0.8, ls=":", zorder=0)
        ax.grid(True, which="major", alpha=0.24, linewidth=0.45)
        ax.grid(True, which="minor", alpha=0.10, linewidth=0.35)

    axes[0].axhline(
        uncertainty_threshold,
        color="0.55",
        lw=0.7,
        ls="-.",
        zorder=0,
    )
    axes[0].set_ylabel(r"$I_0$")
    axes[1].yaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: f"{1.0e3 * value:.1f}")
    )
    axes[1].set_ylabel(r"$\sigma_\beta^{10\%}$ ($10^{-3}$)")
    axes[2].set_yscale("log")
    axes[2].axhline(1.0, color="0.55", lw=0.7, ls="-.", zorder=0)
    axes[2].set_ylabel(r"$\sigma_{\Omega,\mathrm{fixed}}/\Omega_0$")
    for index, ax in enumerate(axes):
        ax.text(
            0.025,
            1.02,
            f"({chr(ord('a') + index)})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.supxlabel(r"SGWB amplitude $\Omega_0$", y=0.055, fontsize=9)
    save_figure(fig, output_png, output_pdf)


def main() -> None:
    args = parse_args()
    if args.omega0_min <= 0.0 or args.omega0_max <= args.omega0_min:
        raise ValueError("The amplitude limits must satisfy 0 < min < max.")
    if args.n_amplitudes < 2:
        raise ValueError("n-amplitudes must be at least two.")
    if not (args.omega0_min <= args.fiducial_omega0 <= args.omega0_max):
        raise ValueError("The fiducial amplitude must lie within the scan range.")

    summary = pd.read_csv(args.all_bin_summary)
    grid = load_grid(Path(args.mode_power_profiles))
    if len(summary) != grid.mode_count.size:
        raise ValueError("The bin summary and Fourier grid have different bin counts.")
    np.testing.assert_array_equal(
        summary["frequency_bin_index"].to_numpy(int),
        np.arange(grid.mode_count.size),
        err_msg="The bin summary is not in Fourier-grid order.",
    )

    residual_power = summary["LWA_X_res_power_sum"].to_numpy(float)
    concentration = summary["LWA_X_res_C2"].to_numpy(float)
    beta_fiducial = constant_metadata(summary, "beta_fiducial")
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
    template_psd = sgwb_strain_psd_per_omega(
        grid.frequency_hz,
        alpha=0.0,
        reference_frequency_hz=reference_frequency_hz,
        h0_km_s_mpc=h0_km_s_mpc,
    )

    amplitudes = np.geomspace(
        args.omega0_min,
        args.omega0_max,
        args.n_amplitudes,
    )
    amplitudes = np.unique(np.sort(np.append(amplitudes, args.fiducial_omega0)))
    rows = uniform_sgwb_amplitude_sweep(
        grid,
        instrument_psd,
        template_psd,
        residual_power,
        concentration,
        amplitudes,
        beta_fiducial=beta_fiducial,
        uncertainty_threshold=args.uncertainty_threshold,
    )
    output = pd.DataFrame(rows)
    output.insert(0, "alpha", 0.0)
    output["n_bins"] = int(grid.mode_count.size)
    output["n_nonzero_residual_bins"] = int(np.count_nonzero(residual_power > 0.0))
    output["residual_frequency_distribution"] = "uniform_within_analysis_bin"
    output["cross_frequency_covariance_included"] = False
    output["instrument_noise_normalization_parameter"] = "eta"
    output["instrument_noise_frequency_dependence_fixed"] = True
    output = output.sort_values(
        ["covariance_treatment", "omega0_fiducial"]
    ).reset_index(drop=True)

    output_table = Path(args.output_table)
    output_table.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_table, index=False)
    make_figure(
        output,
        fiducial_omega0=args.fiducial_omega0,
        uncertainty_threshold=args.uncertainty_threshold,
        output_png=Path(args.output_png),
        output_pdf=Path(args.output_pdf),
    )

    print("All-bin SGWB-amplitude dependence (alpha = 0):")
    for treatment in (
        "gaussian_residual_amplitudes",
        "finite_source_random_phase_diagonal",
    ):
        row = output.loc[
            (output["covariance_treatment"] == treatment)
            & np.isclose(
                output["omega0_fiducial"],
                args.fiducial_omega0,
                rtol=1.0e-12,
                atol=0.0,
            )
        ].iloc[0]
        print(
            f"  {treatment}: I0={row['no_prior_inflation']:.10f}, "
            f"increase={row['uncertainty_increase_percent']:.6f}%, "
            f"sigma_beta(10%)={row['sigma_beta_for_threshold']:.10g}, "
            f"fixed_amplitude_ratio="
            f"{row['inverse_fractional_fixed_uncertainty']:.6f}, "
            f"marginalized_amplitude_ratio="
            f"{row['marginalized_amplitude_to_uncertainty_ratio']:.6f}, "
            f"eta_beta_marginalized_amplitude_ratio="
            f"{row['instrument_noise_normalization_and_beta_marginalized_ratio']:.6f}"
        )
    print(f"  table = {output_table}")
    print(f"  figure = {args.output_pdf}")


if __name__ == "__main__":
    main()
