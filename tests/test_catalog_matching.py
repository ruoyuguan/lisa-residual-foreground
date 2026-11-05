import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "repro" / "repro_scripts"))

from catalog_matching import (  # noqa: E402
    CatalogIndex,
    MatchingTolerances,
    match_injection_rows,
    recovered_row_indices_by_group,
    validate_matching_audit,
)


PARAMETER_COLUMNS = [
    "inj_frequency",
    "inj_amplitude",
    "inj_frequency_derivative",
    "inj_initial_phase",
    "inj_inclination",
    "inj_polarization",
    "inj_ecliptic_longitude",
    "inj_ecliptic_latitude",
]


def accepted_row(frequency: float, **overrides: float) -> dict[str, float]:
    row = {
        "comparison_row": 7,
        "inj_frequency": frequency,
        "inj_amplitude": 2.0e-22,
        "inj_frequency_derivative": 3.0e-18,
        "inj_initial_phase": 0.4,
        "inj_inclination": 1.2,
        "inj_polarization": 2.1,
        "inj_ecliptic_longitude": 5.3,
        "inj_ecliptic_latitude": -0.2,
        "best_overlap": 0.99,
        "confidence": 1.0,
    }
    row.update(overrides)
    return row


def catalog_index(group: str, rows: list[dict[str, float]]) -> CatalogIndex:
    matrix = np.array(
        [
            [
                row["frequency"],
                row["amplitude"],
                row["frequency_derivative"],
                row["initial_phase"],
                row["inclination"],
                row["polarization"],
                row["ecliptic_longitude"],
                row["ecliptic_latitude"],
            ]
            for row in rows
        ],
        dtype=float,
    )
    return CatalogIndex.from_unsorted(
        group=group,
        parameters=matrix,
        row_indices=np.arange(len(rows), dtype=np.int64),
        source_names=np.array([row.get("name", str(i)) for i, row in enumerate(rows)]),
    )


class CatalogMatchingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tolerances = MatchingTolerances()

    def test_matches_across_a_decimal_rounding_boundary(self) -> None:
        catalog_frequency = 0.00123456789012345
        comparison_frequency = 0.00123456789012355
        old_catalog_key = np.rint(catalog_frequency * 1.0e15)
        old_comparison_key = np.rint(comparison_frequency * 1.0e15)
        self.assertNotEqual(old_catalog_key, old_comparison_key)

        accepted = pd.DataFrame([accepted_row(comparison_frequency)])
        source = {
            "frequency": catalog_frequency,
            "amplitude": 2.0e-22,
            "frequency_derivative": 3.0e-18,
            "initial_phase": 0.4,
            "inclination": 1.2,
            "polarization": 2.1,
            "ecliptic_longitude": 5.3,
            "ecliptic_latitude": -0.2,
            "name": "dgb-17",
        }
        audit = match_injection_rows(
            accepted,
            {"sky/dgb/cat": catalog_index("sky/dgb/cat", [source])},
            in_scope_groups={"sky/dgb/cat", "sky/igb/cat"},
            tolerances=self.tolerances,
        )

        self.assertEqual(audit.loc[0, "match_status"], "matched_in_scope")
        self.assertEqual(audit.loc[0, "catalog_source_name"], "dgb-17")
        self.assertIsInstance(audit.loc[0, "comparison_row"], np.integer)
        self.assertAlmostEqual(audit.loc[0, "delta_frequency_Hz"], -1.0e-16, places=19)

    def test_uses_all_parameters_to_reject_a_nearby_frequency(self) -> None:
        accepted = pd.DataFrame([accepted_row(0.002)])
        true_source = {
            "frequency": 0.002 + 2.0e-16,
            "amplitude": 2.0e-22,
            "frequency_derivative": 3.0e-18,
            "initial_phase": 0.4,
            "inclination": 1.2,
            "polarization": 2.1,
            "ecliptic_longitude": 5.3,
            "ecliptic_latitude": -0.2,
            "name": "true",
        }
        nearby_source = dict(true_source)
        nearby_source.update(frequency=0.002, ecliptic_longitude=1.0, name="nearby")
        audit = match_injection_rows(
            accepted,
            {"sky/dgb/cat": catalog_index("sky/dgb/cat", [nearby_source, true_source])},
            in_scope_groups={"sky/dgb/cat"},
            tolerances=self.tolerances,
        )

        self.assertEqual(audit.loc[0, "catalog_source_name"], "true")
        self.assertEqual(audit.loc[0, "frequency_candidate_count"], 2)
        self.assertEqual(audit.loc[0, "parameter_match_count"], 1)

    def test_classifies_a_unique_vgb_match_as_out_of_scope(self) -> None:
        accepted = pd.DataFrame([accepted_row(0.003)])
        source = {
            "frequency": 0.003,
            "amplitude": 2.0e-22,
            "frequency_derivative": 3.0e-18,
            "initial_phase": 0.4,
            "inclination": 1.2,
            "polarization": 2.1,
            "ecliptic_longitude": 5.3,
            "ecliptic_latitude": -0.2,
            "name": "verification-binary",
        }
        audit = match_injection_rows(
            accepted,
            {"sky/vgb/cat": catalog_index("sky/vgb/cat", [source])},
            in_scope_groups={"sky/dgb/cat", "sky/igb/cat"},
            tolerances=self.tolerances,
        )

        self.assertEqual(audit.loc[0, "match_status"], "matched_out_of_scope")

    def test_reports_ambiguous_and_unmatched_rows(self) -> None:
        accepted = pd.DataFrame(
            [accepted_row(0.004), accepted_row(0.005, comparison_row=8)]
        )
        source = {
            "frequency": 0.004,
            "amplitude": 2.0e-22,
            "frequency_derivative": 3.0e-18,
            "initial_phase": 0.4,
            "inclination": 1.2,
            "polarization": 2.1,
            "ecliptic_longitude": 5.3,
            "ecliptic_latitude": -0.2,
        }
        audit = match_injection_rows(
            accepted,
            {
                "sky/dgb/cat": catalog_index("sky/dgb/cat", [source]),
                "sky/igb/cat": catalog_index("sky/igb/cat", [source]),
            },
            in_scope_groups={"sky/dgb/cat", "sky/igb/cat"},
            tolerances=self.tolerances,
        )

        self.assertEqual(audit.loc[0, "match_status"], "ambiguous")
        self.assertEqual(audit.loc[1, "match_status"], "unmatched")
        with self.assertRaisesRegex(ValueError, "ambiguous=1, unmatched=1"):
            validate_matching_audit(audit, accepted_count=2)

    def test_recovered_row_sets_include_only_unique_in_scope_matches(self) -> None:
        audit = pd.DataFrame(
            [
                {
                    "match_status": "matched_in_scope",
                    "catalog_group": "sky/dgb/cat",
                    "catalog_row_index": 17,
                },
                {
                    "match_status": "matched_out_of_scope",
                    "catalog_group": "sky/vgb/cat",
                    "catalog_row_index": 3,
                },
            ]
        )
        recovered = recovered_row_indices_by_group(
            audit,
            expected_groups={"sky/dgb/cat", "sky/igb/cat"},
        )

        np.testing.assert_array_equal(recovered["sky/dgb/cat"], np.array([17]))
        self.assertEqual(len(recovered["sky/igb/cat"]), 0)


if __name__ == "__main__":
    unittest.main()
