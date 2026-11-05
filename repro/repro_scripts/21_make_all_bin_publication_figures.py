#!/usr/bin/env python
"""Make publication-style figures from the all-bin Michelson-X calculation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lisa_catalog_statistics import (
    alpha_column_suffix,
    prior_width_for_uncertainty_threshold,
    two_parameter_fisher_summary,
)


OMEGA0_DEFAULT = 1.0e-11


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
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
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.linewidth": 0.7,
            "legend.title_fontsize": 8,
        }
    )


def positive_finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values) & (values > 0)]


def safe_norm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    pos = positive_finite(values)
    if pos.size == 0:
        return np.full_like(values, np.nan, dtype=float)
    return values / np.nanmax(pos)


def empirical_percentile(values: np.ndarray, percentile: float) -> float:
    """Return a linear empirical percentile while preserving infinite bounds."""

    samples = np.asarray(values, dtype=float)
    samples = np.sort(samples[~np.isnan(samples)])
    if samples.size == 0:
        return np.nan
    rank = (float(percentile) / 100.0) * (samples.size - 1)
    lower_index = int(np.floor(rank))
    upper_index = int(np.ceil(rank))
    lower = float(samples[lower_index])
    upper = float(samples[upper_index])
    if lower_index == upper_index or lower == upper:
        return lower
    fraction = rank - lower_index
    if np.isposinf(upper) and fraction > 0.0:
        return np.inf
    if np.isneginf(lower) and fraction < 1.0:
        return -np.inf
    return float(lower + fraction * (upper - lower))


def plotted_values_with_gaps(values: np.ndarray) -> np.ndarray:
    """Replace unbounded or undefined values by NaN so plotted lines break."""

    values = np.asarray(values, dtype=float)
    return np.where(np.isfinite(values), values, np.nan)


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "f_mHz" not in out.columns:
        if "f_center_mHz" in out.columns:
            out["f_mHz"] = out["f_center_mHz"]
        else:
            out["f_mHz"] = 1.0e3 * out["f_center_Hz"]
    if "f_center_Hz" not in out.columns:
        out["f_center_Hz"] = 1.0e-3 * out["f_mHz"]
    out = out.sort_values("f_center_Hz").reset_index(drop=True)
    return out


def unique_numeric_metadata(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        raise KeyError(f"Required metadata column is absent: {column}")
    values = df[column].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f"Metadata column has no finite values: {column}")
    reference = float(values[0])
    if not np.allclose(values, reference, rtol=1.0e-12, atol=0.0):
        raise ValueError(f"Metadata column is not constant across bins: {column}")
    return reference


def fisher_for_rows(
    rows: pd.DataFrame,
    *,
    alpha: float,
    omega0: float,
    beta_fiducial: float,
    threshold: float | None = None,
) -> dict[str, float | bool]:
    suffix = alpha_column_suffix(alpha)
    fisher = two_parameter_fisher_summary(
        rows[f"T_alpha_{suffix}"].to_numpy(float),
        rows["LWA_X_res_power_sum"].to_numpy(float),
        rows[f"bandpower_variance_gaussian_alpha_{suffix}"].to_numpy(float),
        omega0=omega0,
        beta_fiducial=beta_fiducial,
        uncertainty_threshold=threshold if threshold is not None else 1.1,
    )
    result: dict[str, float | bool] = {
        "alpha": float(alpha),
        "n_bin": int(len(rows)),
        "F_OmegaOmega": fisher["F_oo"],
        "F_betabeta": fisher["F_bb"],
        "F_Omegabeta": fisher["F_ob"],
        "rho": fisher["no_prior_abs_corr"],
        "I0": fisher["no_prior_inflation"],
        "sigma_omega_fixed": fisher["sigma_omega_fixed"],
        "sigma_omega_fixed_over_omega0": fisher["sigma_omega_fixed_over_omega0"],
        "bias_over_sigma_fixed": fisher["bias_ignore_over_sigma_fixed"],
    }
    if threshold is not None:
        result["sigma_beta"] = fisher["sigma_beta_for_threshold"]
    return result


def prior_width(
    f_omegaomega: float,
    f_betabeta: float,
    f_omegabeta: float,
    threshold: float,
) -> float:
    return prior_width_for_uncertainty_threshold(
        f_omegaomega,
        f_omegabeta,
        f_betabeta,
        threshold,
    )


def fisher_scores(
    rows: pd.DataFrame,
    *,
    alpha: float,
    omega0: float,
) -> pd.DataFrame:
    out = rows.copy()
    suffix = alpha_column_suffix(alpha)
    r = out["LWA_X_res_power_sum"].to_numpy(float)
    t = out[f"T_alpha_{suffix}"].to_numpy(float)
    var = out[f"bandpower_variance_gaussian_alpha_{suffix}"].to_numpy(float)
    out["score_omega_beta_cross_abs"] = np.abs(t * r / var)
    out["score_omega_amplitude"] = t * t / var
    out["score_abs_gamma2_total"] = out["abs_gamma2_total"].to_numpy(float)
    return out


def save_figure(fig: plt.Figure, png_path: Path, pdf_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax: plt.Axes, tag: str) -> None:
    ax.text(
        -0.12,
        1.04,
        tag,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        clip_on=False,
    )


def scatter_population(
    ax: plt.Axes,
    all_rows: pd.DataFrame,
    y_col: str,
    *,
    highlighted_rows: pd.DataFrame,
    ylabel: str,
    logy: bool = True,
    ylim: tuple[float, float] | None = None,
) -> None:
    finite = all_rows[np.isfinite(all_rows[y_col].to_numpy(float))]
    ax.scatter(
        finite["f_mHz"],
        finite[y_col],
        s=11,
        color="0.72",
        alpha=0.55,
        linewidth=0.0,
        label=r"all bins with $R_k>0$",
        zorder=1,
    )
    top = highlighted_rows[np.isfinite(highlighted_rows[y_col].to_numpy(float))]
    ax.scatter(
        top["f_mHz"],
        top[y_col],
        s=34,
        marker="x",
        color="#b13c2e",
        linewidth=0.9,
        label=r"top $|\gamma_{2,\rm tot,k}|$ bins",
        zorder=3,
    )
    if logy:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(r"$f$ [mHz]")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.22, linewidth=0.4)


def make_residual_spectrum(df: pd.DataFrame, args: argparse.Namespace) -> None:
    nonzero = df[df["LWA_X_res_power_sum"] > 0].copy()
    highlighted = nonzero.sort_values("abs_gamma2_total", ascending=False).head(36)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), constrained_layout=True)

    ax = axes[0]
    ax.scatter(
        df["f_mHz"],
        df["LWA_X_inj_power_sum"],
        s=10,
        color="0.76",
        alpha=0.50,
        label=r"$P_{\rm inj}$",
    )
    ax.scatter(
        nonzero["f_mHz"],
        nonzero["LWA_X_res_power_sum"],
        s=12,
        color="#2f6f9f",
        alpha=0.62,
        label=r"$R_k$",
    )
    ax.plot(
        df["f_mHz"],
        df["P_inst_bin"],
        color="#2a9d55",
        linewidth=1.0,
        label=r"$P_{\rm inst}$",
    )
    ax.plot(
        df["f_mHz"],
        df["P_sgwb_bin_fid"],
        color="#6b4da0",
        linewidth=1.0,
        label=r"$P_{\rm SGWB}$",
    )
    ax.scatter(
        highlighted["f_mHz"],
        highlighted["LWA_X_res_power_sum"],
        s=34,
        marker="x",
        color="#b13c2e",
        linewidth=0.9,
        label=r"top $|\gamma_{2,\rm tot,k}|$",
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"$f$ [mHz]")
    ax.set_ylabel("binned strain power")
    ax.grid(True, which="both", alpha=0.22, linewidth=0.4)
    ax.legend(frameon=False, ncol=2, loc="best")
    panel_label(ax, "(a)")

    ax = axes[1]
    frac = df["LWA_X_residual_power_fraction"].replace([np.inf, -np.inf], np.nan)
    ax.scatter(
        df["f_mHz"],
        frac,
        s=11,
        color="#2f6f9f",
        alpha=0.50,
        linewidth=0.0,
    )
    ax.scatter(
        highlighted["f_mHz"],
        highlighted["LWA_X_residual_power_fraction"],
        s=34,
        marker="x",
        color="#b13c2e",
        linewidth=0.9,
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"$f$ [mHz]")
    ax.set_ylabel(r"$R_k/P_{{\rm inj},k}$")
    ax.grid(True, which="both", alpha=0.22, linewidth=0.4)
    panel_label(ax, "(b)")

    ax = axes[2]
    r_norm = safe_norm(df["LWA_X_res_power_sum"].to_numpy(float))
    t_norm = safe_norm(df["P_sgwb_bin_fid"].to_numpy(float))
    df_tmp = df.copy()
    df_tmp["R_norm"] = r_norm
    df_tmp["T_norm"] = t_norm
    ax.plot(df_tmp["f_mHz"], df_tmp["T_norm"], color="#6b4da0", lw=1.2, label=r"$T_k(\alpha=0)$")
    residual_rows = df_tmp[df_tmp["R_norm"] > 0.0]
    ax.scatter(
        residual_rows["f_mHz"],
        residual_rows["R_norm"],
        s=11,
        color="#2f6f9f",
        alpha=0.50,
        linewidth=0.0,
        label=r"$R_k>0$ bins",
    )
    top_tmp = df_tmp.loc[highlighted.index]
    ax.scatter(
        top_tmp["f_mHz"],
        top_tmp["R_norm"],
        s=34,
        marker="x",
        color="#b13c2e",
        linewidth=0.9,
        label=r"top $|\gamma_{2,\rm tot,k}|$",
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"$f$ [mHz]")
    ax.set_ylabel("normalized spectral shape")
    ax.grid(True, which="both", alpha=0.22, linewidth=0.4)
    ax.legend(frameon=False, loc="best")
    panel_label(ax, "(c)")

    save_figure(fig, args.output_residual_spectrum_png, args.output_residual_spectrum_pdf)


def make_source_power_concentration(df: pd.DataFrame, args: argparse.Namespace) -> None:
    rows = df[df["LWA_X_res_power_sum"] > 0].copy()
    highlighted = rows.sort_values("abs_gamma2_total", ascending=False).head(36)

    fig, ax = plt.subplots(1, 1, figsize=(4.1, 2.65), constrained_layout=True)
    scatter_population(
        ax,
        rows,
        "LWA_X_res_C2",
        highlighted_rows=highlighted,
        ylabel=r"$C_2=\sum_i w_i^2$",
        logy=True,
    )
    ax.legend(frameon=False, loc="best")
    save_figure(fig, args.output_concentration_png, args.output_concentration_pdf)


def make_noise_dilution(df: pd.DataFrame, args: argparse.Namespace) -> None:
    rows = df[df["LWA_X_res_power_sum"] > 0].copy()
    highlighted = rows.sort_values("abs_gamma2_total", ascending=False).head(36)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), constrained_layout=True)
    panels = [
        ("LWA_X_res_C2", r"$C_{2,k}^{\rm res}$", "(a)"),
        (
            "residual_fraction_in_total_power",
            r"$R_k/\mu_{k,\rm fid}$",
            "(b)",
        ),
        ("abs_gamma2_total", r"$|\gamma_{2,\rm tot,k}|$", "(c)"),
    ]
    for ax, (col, ylabel, tag) in zip(axes, panels):
        scatter_population(
            ax,
            rows,
            col,
            highlighted_rows=highlighted,
            ylabel=ylabel,
            logy=True,
        )
        panel_label(ax, tag)
    axes[0].legend(frameon=False, loc="best")
    save_figure(fig, args.output_noise_png, args.output_noise_pdf)


def make_prior_requirement(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    nonzero = df[df["LWA_X_res_power_sum"] > 0].copy()
    ranked = nonzero.sort_values("abs_gamma2_total", ascending=False)
    rows = []
    for alpha in args.alphas:
        for n_bin in args.top_n:
            subset = ranked.head(n_bin)
            base = fisher_for_rows(
                subset,
                alpha=alpha,
                omega0=args.omega0,
                beta_fiducial=args.beta_fiducial,
            )
            out = {
                "alpha": alpha,
                "N_bin": n_bin,
                "I0": base["I0"],
                "rho_abs": base["rho"],
                "bias_over_sigma_fixed": base["bias_over_sigma_fixed"],
                "sigma_omega_fixed_over_omega0": base["sigma_omega_fixed_over_omega0"],
                "snr_omega_fixed": 1.0 / base["sigma_omega_fixed_over_omega0"],
            }
            for label, threshold in args.thresholds:
                sigma_beta = prior_width(
                    float(base["F_OmegaOmega"]),
                    float(base["F_betabeta"]),
                    float(base["F_Omegabeta"]),
                    threshold,
                )
                out[f"sigma_beta_{label}"] = sigma_beta
                out[f"no_prior_sufficient_{label}"] = bool(np.isinf(sigma_beta))
            rows.append(out)
    table = pd.DataFrame(rows)
    args.output_prior_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_prior_csv, index=False)

    colors = {
        "10pct": "#0072B2",
        "20pct": "#E69F00",
        "50pct": "#009E73",
        "100pct": "#D55E00",
    }
    markers = {"10pct": "o", "20pct": "s", "50pct": "^", "100pct": "D"}
    labels = {
        "10pct": r"$\leq 10\%$",
        "20pct": r"$\leq 20\%$",
        "50pct": r"$\leq 50\%$",
        "100pct": r"$\leq 100\%$",
    }

    finite_values = []
    for label, _threshold in args.thresholds:
        vals = table[f"sigma_beta_{label}"].to_numpy(dtype=float)
        finite_values.extend(vals[np.isfinite(vals)].tolist())
    y_min = max(1.0e-3, 0.75 * min(finite_values)) if finite_values else 0.01
    y_max = 1.55 * max(finite_values) if finite_values else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.65), sharey=True, constrained_layout=False)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.20, top=0.82, wspace=0.07)
    for ax, alpha in zip(axes, args.alphas):
        sub = table[table["alpha"] == alpha].sort_values("N_bin")
        x_all = sub["N_bin"].to_numpy(float)
        for label, _threshold in args.thresholds:
            y = sub[f"sigma_beta_{label}"].to_numpy(float)
            finite = np.isfinite(y)
            if np.any(finite):
                finite_idx = np.flatnonzero(finite)
                split_at = np.where(np.diff(finite_idx) > 1)[0] + 1
                segments = np.split(finite_idx, split_at)
                first_segment = True
                for segment in segments:
                    ax.plot(
                        x_all[segment],
                        y[segment],
                        marker=markers[label],
                        ms=4.8,
                        lw=1.35 if len(segment) > 1 else 0.0,
                        color=colors[label],
                        label=rf"$\sigma_\Omega$ increase {labels[label]}" if first_segment else None,
                        zorder=3,
                    )
                    first_segment = False
        ax.set_title(rf"$\alpha={alpha:g}$", pad=3.0)
        ax.set_xticks(args.top_n)
        ax.set_yscale("log")
        ax.set_ylim(y_min, y_max)
        ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
        ax.grid(True, which="minor", alpha=0.12, linewidth=0.4)
    axes[0].set_ylabel(r"largest allowed prior width $\sigma_\beta$")
    fig.supxlabel(r"$N_{\rm bin}$", y=0.045, fontsize=8.5)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        handlelength=1.8,
        columnspacing=1.2,
    )
    save_figure(fig, args.output_prior_png, args.output_prior_pdf)
    return table


def make_cumulative_gamma_i0(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    nonzero = df[df["LWA_X_res_power_sum"] > 0].copy()
    ranked = nonzero.sort_values("abs_gamma2_total", ascending=False)
    rows = []
    for alpha in args.alphas:
        for n_bin in range(1, len(ranked) + 1):
            base = fisher_for_rows(
                ranked.head(n_bin),
                alpha=alpha,
                omega0=args.omega0,
                beta_fiducial=args.beta_fiducial,
            )
            rows.append(
                {
                    "alpha": alpha,
                    "N_bin": n_bin,
                    "I0": base["I0"],
                    "rho_abs": base["rho"],
                }
            )
    table = pd.DataFrame(rows)
    args.output_cumulative_i0_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_cumulative_i0_csv, index=False)

    colors = {-2.0: "#0072B2", 0.0: "#D55E00", 2.0: "#009E73"}
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35), sharey=True, constrained_layout=False)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.22, top=0.82, wspace=0.08)
    for ax, alpha in zip(axes, args.alphas):
        sub = table[(table["alpha"] == alpha) & (table["N_bin"] >= 2)].sort_values("N_bin")
        ax.plot(
            sub["N_bin"],
            sub["I0"],
            color=colors[alpha],
            lw=1.15,
            zorder=3,
        )
        marked = sub[sub["N_bin"].isin(args.top_n)]
        ax.scatter(
            marked["N_bin"],
            marked["I0"],
            color=colors[alpha],
            s=16,
            zorder=4,
        )
        for _label, threshold in args.thresholds:
            ax.axhline(threshold, color="0.6", lw=0.55, ls="--", zorder=1)
        for n_bin in args.top_n:
            ax.axvline(n_bin, color="0.85", lw=0.45, zorder=0)
        ax.set_title(rf"$\alpha={alpha:g}$", pad=3.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(2, len(ranked))
        ax.set_ylim(0.95, 25.0)
        ax.set_xticks([2, 3, 10, 30, 100, 300])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.grid(True, which="major", alpha=0.25, linewidth=0.5)
        ax.grid(True, which="minor", alpha=0.10, linewidth=0.35)
    axes[0].set_ylabel(r"no-prior increase $I_0$")
    fig.supxlabel(
        r"cumulative top-$|\gamma_{2,\rm tot,k}|$ bin count",
        y=0.055,
        fontsize=8.5,
    )
    save_figure(fig, args.output_cumulative_i0_png, args.output_cumulative_i0_pdf)
    return table


def select_frequency_quantiles(pool: pd.DataFrame, n_bin: int) -> pd.DataFrame:
    """Select approximately equally spaced frequency ranks from occupied bins."""
    ordered = pool.sort_values("f_center_Hz").reset_index(drop=True)
    if n_bin >= len(ordered):
        return ordered.copy()
    idx = np.rint(np.linspace(0, len(ordered) - 1, n_bin)).astype(int)
    idx = np.unique(idx)
    while idx.size < n_bin:
        missing = [i for i in range(len(ordered)) if i not in set(idx)]
        if not missing:
            break
        idx = np.sort(np.r_[idx, missing[: n_bin - idx.size]])
    return ordered.iloc[idx[:n_bin]].copy()


def make_bin_selection_robustness(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    pool = df[df["LWA_X_res_power_sum"] > 0].copy()
    scored = fisher_scores(
        pool,
        alpha=0.0,
        omega0=args.omega0,
    )

    strategy_specs = [
        (
            "one_point_kurtosis",
            r"top $|\gamma_{2,\rm tot,k}|$",
            "score_abs_gamma2_total",
        ),
        ("omega_beta_cross", r"$|F_{\Omega\beta,k}|$", "score_omega_beta_cross_abs"),
        ("omega_amplitude_term", r"$T_k^2/{\rm Var}_{\rm G}$", "score_omega_amplitude"),
    ]
    rows = []
    for key, label, score_col in strategy_specs:
        ranked = scored.sort_values(score_col, ascending=False)
        for n_bin in args.top_n:
            subset = ranked.head(n_bin)
            base = fisher_for_rows(
                subset,
                alpha=0.0,
                omega0=args.omega0,
                beta_fiducial=args.beta_fiducial,
            )
            sigma = prior_width(
                float(base["F_OmegaOmega"]),
                float(base["F_betabeta"]),
                float(base["F_Omegabeta"]),
                args.robustness_threshold,
            )
            rows.append(
                {
                    "selection": key,
                    "label": label,
                    "N_bin": n_bin,
                    "sigma_beta_10pct": sigma,
                    "I0": base["I0"],
                    "bias_over_sigma_fixed": base["bias_over_sigma_fixed"],
                }
            )

    for n_bin in args.top_n:
        subset = select_frequency_quantiles(scored, n_bin)
        base = fisher_for_rows(
            subset,
            alpha=0.0,
            omega0=args.omega0,
            beta_fiducial=args.beta_fiducial,
        )
        sigma = prior_width(
            float(base["F_OmegaOmega"]),
            float(base["F_betabeta"]),
            float(base["F_Omegabeta"]),
            args.robustness_threshold,
        )
        rows.append(
            {
                "selection": "frequency_quantiles",
                "label": "frequency quantiles",
                "N_bin": n_bin,
                "sigma_beta_10pct": sigma,
                "I0": base["I0"],
                "bias_over_sigma_fixed": base["bias_over_sigma_fixed"],
            }
        )

    rng = np.random.default_rng(args.random_seed)
    random_summary = []
    for n_bin in args.top_n:
        sigma_values = []
        i0_values = []
        bias_values = []
        for _ in range(args.n_random):
            subset = scored.sample(n=n_bin, replace=False, random_state=int(rng.integers(0, 2**32 - 1)))
            base = fisher_for_rows(
                subset,
                alpha=0.0,
                omega0=args.omega0,
                beta_fiducial=args.beta_fiducial,
            )
            sigma_values.append(
                prior_width(
                    float(base["F_OmegaOmega"]),
                    float(base["F_betabeta"]),
                    float(base["F_Omegabeta"]),
                    args.robustness_threshold,
                )
            )
            i0_values.append(base["I0"])
            bias_values.append(base["bias_over_sigma_fixed"])
        sigma_arr = np.asarray(sigma_values, dtype=float)
        i0_arr = np.asarray(i0_values, dtype=float)
        bias_arr = np.asarray(bias_values, dtype=float)
        random_summary.append(
            {
                "selection": "random_subset",
                "label": "random subsets",
                "N_bin": n_bin,
                "sigma_beta_10pct": empirical_percentile(sigma_arr, 50),
                "sigma_beta_10pct_p16": empirical_percentile(sigma_arr, 16),
                "sigma_beta_10pct_p84": empirical_percentile(sigma_arr, 84),
                "I0": empirical_percentile(i0_arr, 50),
                "I0_p16": empirical_percentile(i0_arr, 16),
                "I0_p84": empirical_percentile(i0_arr, 84),
                "bias_over_sigma_fixed": empirical_percentile(bias_arr, 50),
                "bias_over_sigma_fixed_p16": empirical_percentile(bias_arr, 16),
                "bias_over_sigma_fixed_p84": empirical_percentile(bias_arr, 84),
            }
        )
    rows.extend(random_summary)
    table = pd.DataFrame(rows)
    args.output_robustness_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_robustness_csv, index=False)

    colors = {
        "one_point_kurtosis": "#b13c2e",
        "omega_beta_cross": "#d6852a",
        "omega_amplitude_term": "#6b4da0",
        "frequency_quantiles": "#496b3a",
        "random_subset": "0.35",
    }
    markers = {
        "one_point_kurtosis": "o",
        "omega_beta_cross": "s",
        "omega_amplitude_term": "^",
        "frequency_quantiles": "D",
    }

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.85), constrained_layout=False)
    fig.subplots_adjust(left=0.080, right=0.985, bottom=0.19, top=0.70, wspace=0.36)
    y_cols = [
        ("sigma_beta_10pct", r"largest allowed prior width $\sigma_\beta$"),
        ("I0", r"no-prior increase $I_0$"),
        ("bias_over_sigma_fixed", r"$\Delta\Omega_{\rm bias}/\sigma_{\Omega,\rm fixed}$"),
    ]
    deterministic = table[table["selection"] != "random_subset"]
    for key, label, _score_col in strategy_specs + [("frequency_quantiles", "frequency quantiles", "")]:
        sub = deterministic[deterministic["selection"] == key].sort_values("N_bin")
        for ax, (col, ylabel) in zip(axes, y_cols):
            y = plotted_values_with_gaps(sub[col].to_numpy(float))
            if np.any(np.isfinite(y)):
                lw = 1.35 if key == "one_point_kurtosis" else 1.0
                zorder = 2.5 if key == "one_point_kurtosis" else 2.0
                ax.plot(
                    sub["N_bin"],
                    y,
                    marker=markers[key],
                    ms=4.0,
                    lw=lw,
                    color=colors[key],
                    label=label if ax is axes[0] else None,
                    zorder=zorder,
                )
    rand = table[table["selection"] == "random_subset"].sort_values("N_bin")
    for ax, (col, ylabel) in zip(axes, y_cols):
        ax.plot(
            rand["N_bin"],
            rand[col],
            marker="x",
            ms=4.2,
            lw=1.0,
            color=colors["random_subset"],
            label="random subsets" if ax is axes[0] else None,
            zorder=2.0,
        )
        p16 = f"{col}_p16"
        p84 = f"{col}_p84"
        if p16 in rand and p84 in rand:
            lower = plotted_values_with_gaps(rand[p16].to_numpy(float))
            upper = plotted_values_with_gaps(rand[p84].to_numpy(float))
            bounded = np.isfinite(lower) & np.isfinite(upper)
            ax.fill_between(
                rand["N_bin"],
                lower,
                upper,
                where=bounded,
                color=colors["random_subset"],
                alpha=0.22,
                linewidth=0.0,
                zorder=0.5,
            )
        ax.set_xlabel(r"$N_{\rm bin}$")
        ax.set_xticks(args.top_n)
        ax.set_ylabel(ylabel, labelpad=2)
        ax.grid(True, which="both", alpha=0.22, linewidth=0.4)
    axes[0].set_yscale("log")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        handlelength=1.5,
        columnspacing=1.0,
    )
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    panel_label(axes[2], "(c)")
    save_figure(fig, args.output_robustness_png, args.output_robustness_pdf)
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_tag = "LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz"
    default_fig_dir = Path("repro/figures/all_bins")
    parser.add_argument(
        "--all-bin-summary",
        type=Path,
        default=Path(f"repro/tables/all_bins_LWA_X_summary_C0p9_overlap0p9_f0p4_6mHz.csv"),
    )
    parser.add_argument(
        "--omega0",
        type=float,
        default=OMEGA0_DEFAULT,
    )
    parser.add_argument("--n-random", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=12345)
    parser.add_argument(
        "--output-residual-spectrum-png",
        type=Path,
        default=default_fig_dir / f"all_bins_publication_residual_spectrum_{default_tag}.png",
    )
    parser.add_argument(
        "--output-residual-spectrum-pdf",
        type=Path,
        default=default_fig_dir / f"all_bins_publication_residual_spectrum_{default_tag}.pdf",
    )
    parser.add_argument(
        "--output-concentration-png",
        type=Path,
        default=default_fig_dir / f"all_bins_source_power_concentration_{default_tag}.png",
    )
    parser.add_argument(
        "--output-concentration-pdf",
        type=Path,
        default=default_fig_dir / f"all_bins_source_power_concentration_{default_tag}.pdf",
    )
    parser.add_argument(
        "--output-noise-png",
        type=Path,
        default=default_fig_dir / f"all_bins_publication_noise_dilution_{default_tag}.png",
    )
    parser.add_argument(
        "--output-noise-pdf",
        type=Path,
        default=default_fig_dir / f"all_bins_publication_noise_dilution_{default_tag}.pdf",
    )
    parser.add_argument(
        "--output-prior-png",
        type=Path,
        default=default_fig_dir / f"all_bins_publication_prior_requirement_summary_clean_{default_tag}.png",
    )
    parser.add_argument(
        "--output-prior-pdf",
        type=Path,
        default=default_fig_dir / f"all_bins_publication_prior_requirement_summary_clean_{default_tag}.pdf",
    )
    parser.add_argument(
        "--output-prior-csv",
        type=Path,
        default=Path(f"repro/tables/all_bins_publication_prior_requirement_summary_{default_tag}.csv"),
    )
    parser.add_argument(
        "--output-cumulative-i0-png",
        type=Path,
        default=default_fig_dir / f"all_bins_cumulative_top_gamma2_I0_{default_tag}.png",
    )
    parser.add_argument(
        "--output-cumulative-i0-pdf",
        type=Path,
        default=default_fig_dir / f"all_bins_cumulative_top_gamma2_I0_{default_tag}.pdf",
    )
    parser.add_argument(
        "--output-cumulative-i0-csv",
        type=Path,
        default=Path(f"repro/tables/all_bins_cumulative_top_gamma2_I0_{default_tag}.csv"),
    )
    parser.add_argument(
        "--output-robustness-png",
        type=Path,
        default=default_fig_dir / f"all_bins_bin_selection_robustness_prior_requirement_{default_tag}.png",
    )
    parser.add_argument(
        "--output-robustness-pdf",
        type=Path,
        default=default_fig_dir / f"all_bins_bin_selection_robustness_prior_requirement_{default_tag}.pdf",
    )
    parser.add_argument(
        "--output-robustness-csv",
        type=Path,
        default=Path(f"repro/tables/all_bins_bin_selection_robustness_prior_requirement_{default_tag}.csv"),
    )
    args = parser.parse_args()
    args.alphas = [-2.0, 0.0, 2.0]
    args.top_n = [6, 12, 24, 36]
    args.thresholds = [("10pct", 1.1), ("20pct", 1.2), ("50pct", 1.5), ("100pct", 2.0)]
    args.robustness_threshold = 1.1
    return args


def main() -> None:
    args = parse_args()
    setup_matplotlib()
    df = ensure_columns(pd.read_csv(args.all_bin_summary))
    expected_channel = "long_wavelength_michelson_X_equivalent_strain"
    if "analysis_channel" not in df.columns or set(df["analysis_channel"]) != {
        expected_channel
    }:
        raise ValueError(
            "The publication script requires the long-wavelength Michelson-X "
            "equivalent-strain summary table."
        )
    unique_numeric_metadata(df, "bin_width_Hz")
    omega0_fiducial = unique_numeric_metadata(df, "omega0_fiducial")
    if not np.isclose(args.omega0, omega0_fiducial, rtol=1.0e-12, atol=0.0):
        raise ValueError(
            "The requested omega0 does not match the fiducial value stored in the summary."
        )
    args.beta_fiducial = unique_numeric_metadata(df, "beta_fiducial")
    if "n_eff_bandpower" not in df.columns:
        raise KeyError("The summary table has no per-bin n_eff_bandpower column.")
    n_eff_values = df["n_eff_bandpower"].to_numpy(float)
    if not np.all(np.isfinite(n_eff_values) & (n_eff_values > 0.0)):
        raise ValueError("The per-bin n_eff_bandpower values must be positive and finite.")
    for alpha in args.alphas:
        suffix = alpha_column_suffix(alpha)
        for column in (
            f"T_alpha_{suffix}",
            f"bandpower_variance_gaussian_alpha_{suffix}",
        ):
            if column not in df.columns:
                raise KeyError(f"The summary table is missing {column}.")

    make_residual_spectrum(df, args)
    make_source_power_concentration(df, args)
    make_noise_dilution(df, args)
    make_prior_requirement(df, args)
    make_cumulative_gamma_i0(df, args)
    make_bin_selection_robustness(df, args)

    print("Wrote all-bin publication figures:")
    print(f"  {args.output_residual_spectrum_pdf}")
    print(f"  {args.output_concentration_pdf}")
    print(f"  {args.output_noise_pdf}")
    print(f"  {args.output_prior_pdf}")
    print(f"  {args.output_cumulative_i0_pdf}")
    print(f"  {args.output_robustness_pdf}")


if __name__ == "__main__":
    main()
