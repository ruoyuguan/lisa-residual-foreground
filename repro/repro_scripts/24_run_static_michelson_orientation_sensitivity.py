#!/usr/bin/env python
"""Check the catalog result across physical static LISA orbital snapshots."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from catalog_matching import recovered_row_indices_by_group
from lisa_catalog_statistics import (
    build_gaussian_catalog_bandpower_model,
    positive_fourier_modes_by_bin,
    sgwb_strain_psd_per_omega,
    single_michelson_lwa_equivalent_strain_psd,
    static_lisa_michelson_x_tensor,
    two_parameter_fisher_summary,
)


def load_baseline_module():
    path = SCRIPTS / "20_compute_catalog_bandpower_baseline.py"
    spec = importlib.util.spec_from_file_location("catalog_bandpower_baseline", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(path)
    spec.loader.exec_module(module)
    return module


baseline = load_baseline_module()


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
    parser.add_argument("--datasets", nargs="+", default=["sky/dgb/cat", "sky/igb/cat"])
    parser.add_argument(
        "--orbital-phases-deg",
        nargs="+",
        type=float,
        default=[0.0, 60.0, 120.0, 180.0, 240.0, 300.0],
    )
    parser.add_argument("--frequency-min-mHz", type=float, default=0.4)
    parser.add_argument("--frequency-max-mHz", type=float, default=6.0)
    parser.add_argument("--bin-width-Hz", type=float, default=1.0e-5)
    parser.add_argument("--omega0", type=float, default=1.0e-11)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--inflation-threshold", type=float, default=1.1)
    parser.add_argument("--sgwb-fref-Hz", type=float, default=3.0e-3)
    parser.add_argument("--h0-km-s-Mpc", type=float, default=67.4)
    parser.add_argument("--lisa-arm-length-m", type=float, default=2.5e9)
    parser.add_argument(
        "--observation-dataset",
        "--tdi-dataset",
        dest="observation_dataset",
        default="obs/tdi",
    )
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "repro/sensitivity/static_michelson_orientation/"
            "physical_snapshot_sensitivity_summary.csv"
        ),
    )
    return parser.parse_args()


def validate_phases(phases_deg: list[float]) -> np.ndarray:
    phases = np.asarray(phases_deg, dtype=float)
    if phases.ndim != 1 or phases.size == 0 or not np.all(np.isfinite(phases)):
        raise ValueError("At least one finite orbital phase is required.")
    reduced = np.mod(phases, 360.0)
    if np.unique(np.round(reduced, decimals=12)).size != reduced.size:
        raise ValueError("Orbital phases must be distinct modulo 360 degrees.")
    return reduced


def accumulate_power(
    target_sum: np.ndarray,
    target_sum2: np.ndarray,
    bin_index: np.ndarray,
    power: np.ndarray,
    residual: np.ndarray,
) -> None:
    keep = residual & np.isfinite(power) & (power > 0.0)
    if not np.any(keep):
        return
    selected_bins = bin_index[keep]
    selected_power = power[keep]
    target_sum += np.bincount(
        selected_bins,
        weights=selected_power,
        minlength=target_sum.size,
    )
    target_sum2 += np.bincount(
        selected_bins,
        weights=selected_power**2,
        minlength=target_sum.size,
    )


def summarize_result(
    label: str,
    orbital_phase_deg: float,
    residual_power: np.ndarray,
    residual_power2: np.ndarray,
    fourier_grid,
    instrument_psd: np.ndarray,
    template_psd: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, float | str]:
    model = build_gaussian_catalog_bandpower_model(
        fourier_grid,
        instrument_psd,
        template_psd,
        residual_power,
        omega0=args.omega0,
        beta=1.0,
    )
    fisher = two_parameter_fisher_summary(
        model.sgwb_template,
        residual_power,
        model.variance,
        omega0=args.omega0,
        beta_fiducial=1.0,
        uncertainty_threshold=args.inflation_threshold,
    )
    concentration = np.divide(
        residual_power2,
        residual_power**2,
        out=np.full_like(residual_power, np.nan),
        where=residual_power > 0.0,
    )
    residual_fraction = np.divide(
        residual_power,
        model.mean,
        out=np.zeros_like(residual_power),
        where=model.mean > 0.0,
    )
    gamma_total = np.where(
        np.isfinite(concentration),
        -1.5 * concentration * residual_fraction**2,
        0.0,
    )
    nonzero = residual_power > 0.0
    return {
        "response_average": label,
        "orbital_phase_deg": orbital_phase_deg,
        "n_nonzero_residual_bins": int(np.count_nonzero(nonzero)),
        "residual_power_band_sum": float(np.sum(residual_power)),
        "median_abs_gamma2_total_nonzero": float(
            np.median(np.abs(gamma_total[nonzero]))
        ),
        "n_eff_bandpower_min": float(np.min(model.effective_degrees_of_freedom)),
        "n_eff_bandpower_max": float(np.max(model.effective_degrees_of_freedom)),
        "all_bin_abs_rho": fisher["no_prior_abs_corr"],
        "all_bin_I0": fisher["no_prior_inflation"],
        "all_bin_sigma_beta_threshold": fisher["sigma_beta_for_threshold"],
        "all_bin_sigma_omega_fixed_over_omega0": fisher[
            "sigma_omega_fixed_over_omega0"
        ],
        "all_bin_bias_over_sigma_fixed": fisher["bias_ignore_over_sigma_fixed"],
    }


def main() -> None:
    args = parse_args()
    phases_deg = validate_phases(args.orbital_phases_deg)
    for path in (args.sangria_h5, args.matching_audit):
        if not path.is_file():
            raise FileNotFoundError(path)

    frequency_min_hz = args.frequency_min_mHz * 1.0e-3
    frequency_max_hz = args.frequency_max_mHz * 1.0e-3
    bins, edges = baseline.make_frequency_bins(
        frequency_min_hz,
        frequency_max_hz,
        args.bin_width_Hz,
    )
    n_samples, cadence_s, _duration = baseline.read_observation_metadata(
        args.sangria_h5,
        args.observation_dataset,
    )
    fourier_grid = positive_fourier_modes_by_bin(n_samples, cadence_s, edges)
    detectors = {
        float(phase): static_lisa_michelson_x_tensor(np.deg2rad(phase), 0.0)
        for phase in phases_deg
    }
    matching = pd.read_csv(args.matching_audit)
    recovered_rows = recovered_row_indices_by_group(
        matching,
        expected_groups=set(args.datasets),
    )
    expected_recovered = sum(rows.size for rows in recovered_rows.values())
    n_bins = len(bins)
    residual_sum = {phase: np.zeros(n_bins) for phase in detectors}
    residual_sum2 = {phase: np.zeros(n_bins) for phase in detectors}
    average_sum = np.zeros(n_bins)
    average_sum2 = np.zeros(n_bins)
    recovered_seen = 0

    with h5py.File(args.sangria_h5, "r") as h5:
        for dataset_path in args.datasets:
            dataset = h5[dataset_path]
            names = dataset.dtype.names
            if names is None:
                raise TypeError(f"Dataset {dataset_path} is not structured.")
            fields = {
                "frequency": baseline.find_field(names, ["Frequency", "frequency"]),
                "amplitude": baseline.find_field(names, ["Amplitude", "amplitude"]),
                "inclination": baseline.find_field(names, ["Inclination", "inclination"]),
                "polarization": baseline.find_field(names, ["Polarization", "polarization"]),
                "longitude": baseline.find_field(
                    names,
                    ["EclipticLongitude", "Ecliptic Longitude"],
                ),
                "latitude": baseline.find_field(
                    names,
                    ["EclipticLatitude", "Ecliptic Latitude"],
                ),
            }
            expected_rows = recovered_rows[dataset_path]
            for start in range(0, len(dataset), args.chunksize):
                stop = min(start + args.chunksize, len(dataset))
                chunk = dataset[start:stop]
                frequency = np.asarray(chunk[fields["frequency"]], float).reshape(-1)
                bin_index, in_band = baseline.locate_frequency_bins(frequency, edges)
                if not np.any(in_band):
                    continue
                catalog_rows = np.flatnonzero(in_band).astype(np.int64) + start
                positions = np.searchsorted(expected_rows, catalog_rows)
                recovered = np.zeros(catalog_rows.size, dtype=bool)
                within = positions < expected_rows.size
                recovered[within] = expected_rows[positions[within]] == catalog_rows[within]
                recovered_seen += int(np.count_nonzero(recovered))
                residual = ~recovered
                selected_bins = bin_index[in_band].astype(np.int64)
                amplitude = np.asarray(chunk[fields["amplitude"]], float).reshape(-1)[in_band]
                inclination = np.asarray(chunk[fields["inclination"]], float).reshape(-1)[in_band]
                polarization = np.asarray(chunk[fields["polarization"]], float).reshape(-1)[in_band]
                longitude = np.asarray(chunk[fields["longitude"]], float).reshape(-1)[in_band]
                latitude = np.asarray(chunk[fields["latitude"]], float).reshape(-1)[in_band]
                snapshot_power = []
                for phase, detector in detectors.items():
                    power = baseline.response_normalized_source_power(
                        amplitude,
                        inclination,
                        longitude,
                        latitude,
                        polarization,
                        detector,
                    )
                    snapshot_power.append(power)
                    accumulate_power(
                        residual_sum[phase],
                        residual_sum2[phase],
                        selected_bins,
                        power,
                        residual,
                    )
                mean_source_power = np.mean(snapshot_power, axis=0)
                accumulate_power(
                    average_sum,
                    average_sum2,
                    selected_bins,
                    mean_source_power,
                    residual,
                )
            print(f"  {dataset_path}: processed {len(dataset)} sources", flush=True)

    if recovered_seen != expected_recovered:
        raise RuntimeError(
            f"Recovered-source closure failed: {recovered_seen}/{expected_recovered}"
        )

    instrument_psd = single_michelson_lwa_equivalent_strain_psd(
        fourier_grid.frequency_hz,
        args.lisa_arm_length_m,
    )
    template_psd = sgwb_strain_psd_per_omega(
        fourier_grid.frequency_hz,
        args.alpha,
        args.sgwb_fref_Hz,
        args.h0_km_s_Mpc,
    )
    records = [
        summarize_result(
            "static_snapshot",
            phase,
            residual_sum[phase],
            residual_sum2[phase],
            fourier_grid,
            instrument_psd,
            template_psd,
            args,
        )
        for phase in phases_deg
    ]
    records.append(
        summarize_result(
            f"mean_of_{len(phases_deg)}_equally_spaced_snapshots",
            np.nan,
            average_sum,
            average_sum2,
            fourier_grid,
            instrument_psd,
            template_psd,
            args,
        )
    )
    output = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print("\nPhysical static-snapshot sensitivity:")
    print(output.to_string(index=False))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
