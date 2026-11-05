#!/usr/bin/env python
"""Summarize intrinsic chirp and annual Doppler scales for coarse binning."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
C_LIGHT_M_S = 299_792_458.0
ORBITAL_SPEED_M_S = 29_780.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sangria-h5",
        type=Path,
        default=ROOT / "data/raw/sangria/LDC2_sangria_training_v2.h5",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["sky/dgb/cat", "sky/igb/cat"],
    )
    parser.add_argument(
        "--observation-dataset",
        "--tdi-dataset",
        dest="observation_dataset",
        default="obs/tdi",
    )
    parser.add_argument("--frequency-min-mHz", type=float, default=0.4)
    parser.add_argument("--frequency-max-mHz", type=float, default=6.0)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument(
        "--percentiles",
        nargs="+",
        type=float,
        default=[50.0, 90.0, 99.0, 99.9, 99.99, 100.0],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "repro/sensitivity/bin_width/frequency_smearing_scales.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.sangria_h5.is_file():
        raise FileNotFoundError(args.sangria_h5)
    percentiles = np.asarray(args.percentiles, dtype=float)
    if not np.all(np.isfinite(percentiles) & (percentiles >= 0.0) & (percentiles <= 100.0)):
        raise ValueError("Percentiles must lie in [0, 100].")

    f_min_hz = args.frequency_min_mHz * 1.0e-3
    f_max_hz = args.frequency_max_mHz * 1.0e-3
    chirp_blocks: list[np.ndarray] = []
    frequency_blocks: list[np.ndarray] = []
    latitude_blocks: list[np.ndarray] = []
    with h5py.File(args.sangria_h5, "r") as h5:
        observation = h5[args.observation_dataset]
        cadence_s = float(observation.attrs["dt"])
        observation_duration_s = int(observation.shape[0]) * cadence_s
        for dataset_path in args.datasets:
            dset = h5[dataset_path]
            for start in range(0, len(dset), args.chunksize):
                chunk = dset[start : min(start + args.chunksize, len(dset))]
                frequency = np.asarray(chunk["Frequency"], dtype=float).reshape(-1)
                derivative = np.asarray(
                    chunk["FrequencyDerivative"],
                    dtype=float,
                ).reshape(-1)
                latitude = np.asarray(
                    chunk["EclipticLatitude"],
                    dtype=float,
                ).reshape(-1)
                keep = (
                    np.isfinite(frequency)
                    & np.isfinite(derivative)
                    & np.isfinite(latitude)
                    & (frequency >= f_min_hz)
                    & (frequency < f_max_hz)
                )
                frequency_blocks.append(frequency[keep])
                latitude_blocks.append(latitude[keep])
                chirp_blocks.append(np.abs(derivative[keep]) * observation_duration_s)

    frequency_hz = np.concatenate(frequency_blocks)
    latitude_rad = np.concatenate(latitude_blocks)
    chirp_drift_hz = np.concatenate(chirp_blocks)
    full_doppler_span_hz = (
        2.0
        * ORBITAL_SPEED_M_S
        / C_LIGHT_M_S
        * frequency_hz
        * np.abs(np.cos(latitude_rad))
    )
    conservative_sum_hz = chirp_drift_hz + full_doppler_span_hz
    metrics = {
        "intrinsic_chirp_drift": chirp_drift_hz,
        "full_annual_doppler_span": full_doppler_span_hz,
        "conservative_sum": conservative_sum_hz,
    }
    records = []
    for metric, values_hz in metrics.items():
        quantiles_hz = np.percentile(values_hz, percentiles)
        for percentile, value_hz in zip(percentiles, quantiles_hz, strict=True):
            records.append(
                {
                    "metric": metric,
                    "percentile": percentile,
                    "value_Hz": value_hz,
                    "value_uHz": value_hz * 1.0e6,
                    "n_catalog_sources": len(frequency_hz),
                    "observation_duration_s": observation_duration_s,
                    "fourier_resolution_Hz": 1.0 / observation_duration_s,
                }
            )

    output = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(output.to_string(index=False))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
