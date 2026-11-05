"""One-to-one matching of Erebor injection rows to Sangria catalog sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


CATALOG_PARAMETER_COLUMNS = (
    "frequency",
    "amplitude",
    "frequency_derivative",
    "initial_phase",
    "inclination",
    "polarization",
    "ecliptic_longitude",
    "ecliptic_latitude",
)

INJECTION_PARAMETER_COLUMNS = tuple(
    f"inj_{column}" for column in CATALOG_PARAMETER_COLUMNS
)


@dataclass(frozen=True)
class MatchingTolerances:
    """Numerical tolerances for catalog parameters copied between products."""

    frequency_hz: float = 1.0e-12
    relative: float = 1.0e-12
    angular_rad: float = 1.0e-12
    amplitude_abs: float = 0.0
    frequency_derivative_abs: float = 1.0e-30


@dataclass(frozen=True)
class CatalogIndex:
    """Frequency-sorted parameter index for one Sangria catalog group."""

    group: str
    parameters: np.ndarray
    row_indices: np.ndarray
    source_names: np.ndarray | None

    @classmethod
    def from_unsorted(
        cls,
        *,
        group: str,
        parameters: np.ndarray,
        row_indices: np.ndarray,
        source_names: np.ndarray | None = None,
    ) -> "CatalogIndex":
        matrix = np.asarray(parameters, dtype=float)
        rows = np.asarray(row_indices, dtype=np.int64)
        names = None if source_names is None else np.asarray(source_names).astype(str)
        if matrix.ndim != 2 or matrix.shape[1] != len(CATALOG_PARAMETER_COLUMNS):
            raise ValueError(
                "Catalog parameters must have shape "
                f"(n, {len(CATALOG_PARAMETER_COLUMNS)})."
            )
        if len(matrix) != len(rows) or (names is not None and len(matrix) != len(names)):
            raise ValueError("Catalog parameters, row indices, and names must have equal length.")
        order = np.argsort(matrix[:, 0], kind="mergesort")
        return cls(
            group=group,
            parameters=matrix[order],
            row_indices=rows[order],
            source_names=None if names is None else names[order],
        )


def circular_distance(
    values: np.ndarray,
    target: float,
    period: float = 2.0 * np.pi,
) -> np.ndarray:
    delta = np.remainder(values - target + period / 2.0, period) - period / 2.0
    return np.abs(delta)


def _parameter_match_mask(
    candidates: np.ndarray,
    target: np.ndarray,
    tolerances: MatchingTolerances,
) -> np.ndarray:
    return (
        (np.abs(candidates[:, 0] - target[0]) <= tolerances.frequency_hz)
        & np.isclose(
            candidates[:, 1],
            target[1],
            rtol=tolerances.relative,
            atol=tolerances.amplitude_abs,
        )
        & np.isclose(
            candidates[:, 2],
            target[2],
            rtol=tolerances.relative,
            atol=tolerances.frequency_derivative_abs,
        )
        & (circular_distance(candidates[:, 3], target[3]) <= tolerances.angular_rad)
        & (np.abs(candidates[:, 4] - target[4]) <= tolerances.angular_rad)
        & (circular_distance(candidates[:, 5], target[5]) <= tolerances.angular_rad)
        & (circular_distance(candidates[:, 6], target[6]) <= tolerances.angular_rad)
        & (np.abs(candidates[:, 7] - target[7]) <= tolerances.angular_rad)
    )


def match_injection_rows(
    accepted_rows: pd.DataFrame,
    catalog_indices: Mapping[str, CatalogIndex],
    *,
    in_scope_groups: Iterable[str],
    tolerances: MatchingTolerances | None = None,
) -> pd.DataFrame:
    """Match each accepted comparison row to exactly one catalog source."""

    missing = [
        column for column in INJECTION_PARAMETER_COLUMNS if column not in accepted_rows.columns
    ]
    if missing:
        raise KeyError(f"Accepted comparison rows are missing columns: {missing}")
    tolerances = tolerances or MatchingTolerances()
    in_scope = set(in_scope_groups)
    records: list[dict[str, object]] = []

    for _, row in accepted_rows.iterrows():
        target = row.loc[list(INJECTION_PARAMETER_COLUMNS)].to_numpy(dtype=float)
        matches: list[tuple[CatalogIndex, int]] = []
        frequency_candidate_count = 0

        for index in catalog_indices.values():
            frequency = index.parameters[:, 0]
            left = int(
                np.searchsorted(
                    frequency,
                    target[0] - tolerances.frequency_hz,
                    side="left",
                )
            )
            right = int(
                np.searchsorted(
                    frequency,
                    target[0] + tolerances.frequency_hz,
                    side="right",
                )
            )
            frequency_candidate_count += right - left
            if right <= left:
                continue
            local_matches = np.flatnonzero(
                _parameter_match_mask(index.parameters[left:right], target, tolerances)
            )
            matches.extend((index, left + int(offset)) for offset in local_matches)

        record = row.to_dict()
        record["frequency_candidate_count"] = frequency_candidate_count
        record["parameter_match_count"] = len(matches)
        record["catalog_match_multiplicity"] = 0

        if len(matches) == 0:
            record["match_status"] = "unmatched"
        elif len(matches) > 1:
            record["match_status"] = "ambiguous"
            record["candidate_catalog_groups"] = ";".join(
                sorted({index.group for index, _ in matches})
            )
        else:
            index, position = matches[0]
            source = index.parameters[position]
            record.update(
                {
                    "match_status": (
                        "matched_in_scope"
                        if index.group in in_scope
                        else "matched_out_of_scope"
                    ),
                    "catalog_group": index.group,
                    "catalog_row_index": int(index.row_indices[position]),
                    "catalog_source_name": (
                        ""
                        if index.source_names is None
                        else str(index.source_names[position])
                    ),
                    "delta_frequency_Hz": float(source[0] - target[0]),
                    "relative_delta_amplitude": float(
                        (source[1] - target[1]) / target[1]
                    ),
                    "delta_frequency_derivative_Hz_s": float(source[2] - target[2]),
                    "delta_initial_phase_rad": float(
                        circular_distance(np.array([source[3]]), target[3])[0]
                    ),
                    "delta_inclination_rad": float(source[4] - target[4]),
                    "delta_polarization_rad": float(
                        circular_distance(np.array([source[5]]), target[5])[0]
                    ),
                    "delta_ecliptic_longitude_rad": float(
                        circular_distance(np.array([source[6]]), target[6])[0]
                    ),
                    "delta_ecliptic_latitude_rad": float(source[7] - target[7]),
                }
            )
        records.append(record)

    audit = pd.DataFrame(records)
    matched = audit["match_status"].isin(["matched_in_scope", "matched_out_of_scope"])
    if matched.any():
        identities = list(
            zip(
                audit.loc[matched, "catalog_group"],
                audit.loc[matched, "catalog_row_index"],
            )
        )
        multiplicity = pd.Series(identities).value_counts()
        audit.loc[matched, "catalog_match_multiplicity"] = [
            int(multiplicity[identity]) for identity in identities
        ]
        duplicate = matched & (audit["catalog_match_multiplicity"] > 1)
        audit.loc[duplicate, "match_status"] = "duplicate_catalog_match"
    if "comparison_row" in audit.columns:
        audit["comparison_row"] = pd.to_numeric(
            audit["comparison_row"], errors="raise"
        ).astype(np.int64)
    return audit


def validate_matching_audit(audit: pd.DataFrame, *, accepted_count: int) -> None:
    """Fail if accepted rows are not represented by unique catalog matches."""

    if len(audit) != accepted_count:
        raise ValueError(
            f"Matching audit has {len(audit)} rows for {accepted_count} accepted rows."
        )
    counts = audit["match_status"].value_counts()
    ambiguous = int(counts.get("ambiguous", 0))
    unmatched = int(counts.get("unmatched", 0))
    duplicate = int(counts.get("duplicate_catalog_match", 0))
    if ambiguous or unmatched or duplicate:
        raise ValueError(
            "Catalog matching is incomplete or nonunique: "
            f"ambiguous={ambiguous}, unmatched={unmatched}, "
            f"duplicate_catalog_match={duplicate}."
        )


def recovered_row_indices_by_group(
    audit: pd.DataFrame,
    *,
    expected_groups: Iterable[str],
) -> dict[str, np.ndarray]:
    """Return catalog row indices for unique in-scope recovered sources."""

    required = {"match_status", "catalog_group", "catalog_row_index"}
    missing = sorted(required.difference(audit.columns))
    if missing:
        raise KeyError(f"Matching audit is missing columns: {missing}")
    validate_matching_audit(audit, accepted_count=len(audit))
    groups = set(expected_groups)
    selected = audit[audit["match_status"] == "matched_in_scope"].copy()
    unexpected = sorted(set(selected["catalog_group"].dropna()) - groups)
    if unexpected:
        raise ValueError(f"Matching audit contains unexpected in-scope groups: {unexpected}")
    if selected.duplicated(["catalog_group", "catalog_row_index"]).any():
        raise ValueError("Matching audit contains duplicate in-scope catalog identities.")
    return {
        group: np.sort(
            selected.loc[selected["catalog_group"] == group, "catalog_row_index"]
            .astype(np.int64)
            .to_numpy()
        )
        for group in groups
    }
