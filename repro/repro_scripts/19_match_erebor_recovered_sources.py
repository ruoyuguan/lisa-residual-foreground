#!/usr/bin/env python
"""Match accepted Erebor injection rows to unique Sangria catalog sources."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from catalog_matching import (
    CATALOG_PARAMETER_COLUMNS,
    INJECTION_PARAMETER_COLUMNS,
    CatalogIndex,
    MatchingTolerances,
    match_injection_rows,
    validate_matching_audit,
)


HDF5_FIELDS = (
    "Frequency",
    "Amplitude",
    "FrequencyDerivative",
    "InitialPhase",
    "Inclination",
    "Polarization",
    "EclipticLongitude",
    "EclipticLatitude",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match accepted Erebor injection-comparison rows one-to-one to "
            "Sangria catalog sources using all injected source parameters."
        )
    )
    parser.add_argument("--sangria-h5", required=True)
    parser.add_argument("--injection-comp-h5", required=True)
    parser.add_argument("--injection-comp-key", default="/gf_results")
    parser.add_argument(
        "--catalog-datasets",
        nargs="+",
        default=["sky/dgb/cat", "sky/igb/cat", "sky/vgb/cat"],
    )
    parser.add_argument(
        "--in-scope-datasets",
        nargs="+",
        default=["sky/dgb/cat", "sky/igb/cat"],
    )
    parser.add_argument("--confidence-min", type=float, default=0.9)
    parser.add_argument("--best-overlap-min", type=float, default=0.9)
    parser.add_argument("--frequency-min-mHz", type=float, default=0.4)
    parser.add_argument("--frequency-max-mHz", type=float, default=6.0)
    parser.add_argument("--frequency-tolerance-Hz", type=float, default=1.0e-12)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--angular-tolerance-rad", type=float, default=1.0e-12)
    parser.add_argument("--frequency-derivative-atol", type=float, default=1.0e-30)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument(
        "--output-audit",
        default=(
            "repro/tables/"
            "erebor_sangria_recovered_source_matching_audit_"
            "C0p9_overlap0p9_f0p4_6mHz.csv"
        ),
    )
    parser.add_argument(
        "--output-summary",
        default=(
            "repro/tables/"
            "erebor_sangria_recovered_source_matching_summary_"
            "C0p9_overlap0p9_f0p4_6mHz.csv"
        ),
    )
    return parser.parse_args()


def load_catalog_index(
    h5: h5py.File,
    dataset_path: str,
    *,
    frequency_min_hz: float,
    frequency_max_hz: float,
    chunksize: int,
) -> CatalogIndex:
    if dataset_path not in h5:
        raise KeyError(f"Sangria dataset not found: {dataset_path}")
    dataset = h5[dataset_path]
    if dataset.dtype.names is None:
        raise TypeError(f"Sangria dataset is not structured: {dataset_path}")
    missing = [field for field in HDF5_FIELDS if field not in dataset.dtype.names]
    if missing:
        raise KeyError(f"Sangria dataset {dataset_path} is missing fields: {missing}")

    parameter_blocks: list[np.ndarray] = []
    row_blocks: list[np.ndarray] = []
    for start in range(0, len(dataset), chunksize):
        stop = min(start + chunksize, len(dataset))
        chunk = dataset[start:stop].reshape(-1)
        frequency = np.asarray(chunk["Frequency"], dtype=float)
        mask = (frequency >= frequency_min_hz) & (frequency < frequency_max_hz)
        if not np.any(mask):
            continue
        parameter_blocks.append(
            np.column_stack(
                [np.asarray(chunk[field][mask], dtype=float) for field in HDF5_FIELDS]
            )
        )
        row_blocks.append(np.flatnonzero(mask).astype(np.int64) + start)

    if not parameter_blocks:
        parameters = np.empty((0, len(CATALOG_PARAMETER_COLUMNS)), dtype=float)
        rows = np.empty(0, dtype=np.int64)
    else:
        parameters = np.concatenate(parameter_blocks)
        rows = np.concatenate(row_blocks)
    print(f"Catalog index {dataset_path}: {len(rows)} sources in the analysis band")
    return CatalogIndex.from_unsorted(
        group=dataset_path,
        parameters=parameters,
        row_indices=rows,
    )


def snake_case(name: str) -> str:
    """Normalize Erebor HDF column names without changing their contents."""

    return (
        str(name)
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(".", "")
        .replace("(", "")
        .replace(")", "")
        .lower()
    )


def source_name_text(value: object) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    return str(value)


def attach_source_names(audit: pd.DataFrame, h5: h5py.File) -> pd.DataFrame:
    result = audit.copy()
    matched = result["match_status"].isin(["matched_in_scope", "matched_out_of_scope"])
    for group, subset in result.loc[matched].groupby("catalog_group"):
        rows = subset["catalog_row_index"].astype(np.int64).to_numpy()
        order = np.argsort(rows)
        sorted_rows = rows[order]
        records = h5[group][sorted_rows].reshape(-1)
        names = np.array([source_name_text(value) for value in records["Name"]], dtype=object)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        result.loc[subset.index, "catalog_source_name"] = names[inverse]
    return result


def main() -> None:
    args = parse_args()
    if args.chunksize < 1:
        raise ValueError("chunksize must be positive.")
    if args.frequency_tolerance_Hz <= 0 or args.relative_tolerance <= 0:
        raise ValueError("Matching tolerances must be positive.")

    input_paths = [Path(args.sangria_h5), Path(args.injection_comp_h5)]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    comparison = pd.read_hdf(
        args.injection_comp_h5,
        key=args.injection_comp_key,
    ).copy()
    comparison.insert(0, "erebor_comparison_row", np.arange(len(comparison), dtype=int))
    comparison.columns = [snake_case(column) for column in comparison.columns]
    comparison = comparison.rename(
        columns={
            "gf_eclipitc_latitude": "gf_ecliptic_latitude",
            "inj_eclipitc_latitude": "inj_ecliptic_latitude",
        }
    )
    required = [
        "erebor_comparison_row",
        *INJECTION_PARAMETER_COLUMNS,
        "confidence",
        "best_overlap",
    ]
    missing = [column for column in required if column not in comparison.columns]
    if missing:
        raise KeyError(f"Injection-comparison table is missing columns: {missing}")
    for column in required:
        comparison[column] = pd.to_numeric(comparison[column], errors="coerce")

    frequency_min_hz = args.frequency_min_mHz * 1.0e-3
    frequency_max_hz = args.frequency_max_mHz * 1.0e-3
    accepted = comparison[
        (comparison["confidence"] >= args.confidence_min)
        & (comparison["best_overlap"] >= args.best_overlap_min)
        & (comparison["inj_frequency"] >= frequency_min_hz)
        & (comparison["inj_frequency"] < frequency_max_hz)
    ].copy()
    accepted = accepted.rename(columns={"erebor_comparison_row": "comparison_row"})
    if accepted[list(INJECTION_PARAMETER_COLUMNS)].isna().any().any():
        raise ValueError("Accepted comparison rows contain missing injected parameters.")
    if accepted["comparison_row"].duplicated().any():
        raise ValueError("Accepted comparison-row identifiers are not unique.")

    tolerances = MatchingTolerances(
        frequency_hz=args.frequency_tolerance_Hz,
        relative=args.relative_tolerance,
        angular_rad=args.angular_tolerance_rad,
        frequency_derivative_abs=args.frequency_derivative_atol,
    )
    with h5py.File(args.sangria_h5, "r") as h5:
        indices = {
            dataset_path: load_catalog_index(
                h5,
                dataset_path,
                frequency_min_hz=frequency_min_hz,
                frequency_max_hz=frequency_max_hz,
                chunksize=args.chunksize,
            )
            for dataset_path in args.catalog_datasets
        }
        audit = match_injection_rows(
            accepted,
            indices,
            in_scope_groups=set(args.in_scope_datasets),
            tolerances=tolerances,
        )
        validate_matching_audit(audit, accepted_count=len(accepted))
        audit = attach_source_names(audit, h5)

    status = audit["match_status"].value_counts()
    group = audit["catalog_group"].value_counts()
    summary = pd.DataFrame(
        [
            {
                "accepted_comparison_rows": len(accepted),
                "matched_in_scope": int(status.get("matched_in_scope", 0)),
                "matched_out_of_scope": int(status.get("matched_out_of_scope", 0)),
                "ambiguous": int(status.get("ambiguous", 0)),
                "unmatched": int(status.get("unmatched", 0)),
                "duplicate_catalog_match": int(status.get("duplicate_catalog_match", 0)),
                "matched_dgb": int(group.get("sky/dgb/cat", 0)),
                "matched_igb": int(group.get("sky/igb/cat", 0)),
                "matched_vgb": int(group.get("sky/vgb/cat", 0)),
                "max_abs_delta_frequency_Hz": float(audit["delta_frequency_Hz"].abs().max()),
                "max_abs_relative_delta_amplitude": float(
                    audit["relative_delta_amplitude"].abs().max()
                ),
                "frequency_tolerance_Hz": tolerances.frequency_hz,
                "relative_tolerance": tolerances.relative,
                "angular_tolerance_rad": tolerances.angular_rad,
                "frequency_derivative_atol": tolerances.frequency_derivative_abs,
            }
        ]
    )

    output_audit = Path(args.output_audit)
    output_summary = Path(args.output_summary)
    output_audit.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_audit, index=False)
    summary.to_csv(output_summary, index=False)

    print("Matching status:")
    print(status.to_string())
    print("Catalog groups:")
    print(group.to_string())
    print("Matching summary:")
    print(summary.to_string(index=False))
    print(f"Saved audit: {output_audit}")
    print(f"Saved summary: {output_summary}")


if __name__ == "__main__":
    main()
