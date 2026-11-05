"""Direct moments of the existing uniform Gaussian band-power baseline.

For independent exponential mode powers with means q_j, the r-th cumulant
of their sum is (r-1)! sum(q_j**r). No moment-matched distribution is used
to calculate skewness or excess kurtosis. This check does not construct a
coherent LISA likelihood or evaluate RP/FR higher moments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import lisa_catalog_statistics as statistics


ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "repro/tables/all_bins_LWA_X_summary_C0p9_overlap0p9_f0p4_6mHz.csv"
CACHE = ROOT / "repro/sensitivity/residual_fourier_profile/residual_mode_power_profiles.npz"
OUTPUT = ROOT / "repro/sensitivity/gaussian_bandpower_moments"


def exponential_sum_moments(mode_means):
    """Exact moments of a sum of independent exponential contributions."""
    q = np.asarray(mode_means, dtype=float)
    if (
        q.ndim != 1 or q.size == 0 or not np.all(np.isfinite(q))
        or np.any(q < 0.0) or not np.any(q > 0.0)
    ):
        raise ValueError("Mode means must be finite, nonnegative, and not all zero.")
    # Rescale before taking powers to avoid underflow in strain-power units.
    scale = float(np.max(q))
    u = q / scale
    s1, s2, s3, s4 = (float(np.sum(u**r)) for r in range(1, 5))
    return {
        "mean": scale * s1,
        "variance": scale**2 * s2,
        "nu_eff": 2.0 * s1**2 / s2,
        "skewness": 2.0 * s3 / s2**1.5,
        "excess_kurtosis": 6.0 * s4 / s2**2,
    }


def verify_baseline():
    with TABLE.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    def column(name):
        return np.array([float(row[name]) for row in rows])

    def constant(name):
        values = np.unique(column(name))
        if len(values) != 1:
            raise ValueError(f"Expected one value for {name}.")
        return float(values[0])

    with np.load(CACHE, allow_pickle=False) as cache:
        grid = statistics.FourierBandGrid(
            frequency_hz=cache["frequency_hz"],
            bin_index=cache["bin_index"],
            mode_count=cache["mode_count"],
            fourier_spacing_hz=float(cache["fourier_spacing_hz"]),
            bin_edges_hz=cache["bin_edges_hz"],
        )
        residual_mode_power = cache["piecewise_constant_coarse_bin"]
    regenerated = statistics.positive_fourier_modes_by_bin(
        int(constant("observation_sample_count")),
        constant("observation_cadence_s"),
        grid.bin_edges_hz,
    )
    for field in ("frequency_hz", "bin_index", "mode_count"):
        np.testing.assert_array_equal(getattr(grid, field), getattr(regenerated, field))
    np.testing.assert_array_equal(grid.mode_count, column("fourier_mode_count"))
    np.testing.assert_allclose(grid.bin_edges_hz[:-1], column("f_left_Hz"), rtol=1e-13)
    np.testing.assert_allclose(grid.bin_edges_hz[1:], column("f_right_Hz"), rtol=1e-13)
    np.testing.assert_allclose(grid.fourier_spacing_hz, constant("fourier_spacing_Hz"), rtol=1e-14)
    uniform_psd = statistics.piecewise_constant_residual_psd(
        grid, column("LWA_X_res_power_sum")
    )
    np.testing.assert_allclose(
        residual_mode_power, grid.fourier_spacing_hz * uniform_psd,
        rtol=2e-12, atol=0.0,
    )
    instrument = statistics.single_michelson_lwa_equivalent_strain_psd(
        grid.frequency_hz, constant("lisa_arm_length_m")
    )
    omega, beta = constant("omega0_fiducial"), constant("beta_fiducial")
    all_bins, summary = [], []
    for alpha in (-2.0, 0.0, 2.0):
        template = statistics.sgwb_strain_psd_per_omega(
            grid.frequency_hz, alpha, constant("sgwb_reference_frequency_Hz"),
            constant("h0_km_s_Mpc"),
        )
        model = statistics.build_gaussian_catalog_bandpower_model_from_mode_power(
            grid, instrument, template, residual_mode_power, omega, beta
        )
        q = grid.fourier_spacing_hz * (instrument + omega * template) + beta * residual_mode_power
        direct = [
            exponential_sum_moments(q[grid.bin_index == k])
            for k in range(grid.mode_count.size)
        ]
        suffix = statistics.alpha_column_suffix(alpha)
        checks = {
            "mean": (np.array([m["mean"] for m in direct]), column(f"bandpower_mean_fid_alpha_{suffix}")),
            "variance": (np.array([m["variance"] for m in direct]), column(f"bandpower_variance_gaussian_alpha_{suffix}")),
            "nu_eff": (np.array([m["nu_eff"] for m in direct]), column(f"n_eff_gaussian_alpha_{suffix}")),
            "instrument_power": (model.instrument_power, column("P_inst_bin")),
            "sgwb_power": (omega * model.sgwb_template, column(f"P_sgwb_fid_alpha_{suffix}")),
        }
        errors = {}
        for name, (actual, expected) in checks.items():
            np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=0.0)
            errors[name] = float(np.max(np.abs(actual / expected - 1.0)))
        np.testing.assert_allclose(checks["mean"][0], model.mean, rtol=2e-12, atol=0.0)
        np.testing.assert_allclose(checks["variance"][0], model.variance, rtol=2e-12, atol=0.0)
        for k, moments in enumerate(direct):
            all_bins.append({
                "alpha": alpha, "omega0": omega, "beta": beta,
                "frequency_bin_index": int(rows[k]["frequency_bin_index"]),
                "f_center_Hz": float(rows[k]["f_center_Hz"]),
                "mode_count": int(grid.mode_count[k]), **moments,
                "nu_eff_over_2M": moments["nu_eff"] / (2.0 * grid.mode_count[k]),
            })
        result = {"alpha": alpha, "omega0": omega, "beta": beta, "reconstruction_max_relative_error": errors}
        for name in ("nu_eff", "skewness", "excess_kurtosis"):
            values = [m[name] for m in direct]
            result[name] = {"min": min(values), "max": max(values)}
        ratios = checks["nu_eff"][0] / (2.0 * grid.mode_count)
        result["nu_eff_over_2M"] = {"min": float(ratios.min()), "max": float(ratios.max())}
        summary.append(result)
    sources = (TABLE, CACHE, Path(statistics.__file__), Path(__file__))
    return all_bins, {
        "model": "independent Gaussian Fourier amplitudes; uniform residual assignment",
        "bins": len(rows), "fourier_modes": int(grid.frequency_hz.size),
        "mode_counts": sorted(int(n) for n in np.unique(grid.mode_count)),
        "formula": "kappa_r = (r-1)! sum_j q_j^r; q_j = delta_f*(G_j+H_j)",
        "units": {"mean": "strain^2", "variance": "strain^4", "standardized_moments": "dimensionless"},
        "scope": "Exact moments within the independent exponential-power model, not RP/FR or coherent LISA likelihood validation.",
        "input_and_code_sha256": {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "results": summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    rows, summary = verify_baseline()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "uniform_baseline_moments.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "verification_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
