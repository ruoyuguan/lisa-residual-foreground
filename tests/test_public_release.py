import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "repro" / "repro_scripts"))


class PublicReleaseTests(unittest.TestCase):
    def test_release_manifest_has_safe_complete_file_list(self):
        manifest = ROOT / "SHA256SUMS.txt"
        self.assertTrue(manifest.is_file(), "The versioned file manifest is missing.")
        for line in manifest.read_text(encoding="utf-8").splitlines():
            expected, relative = line.split("  ", 1)
            path = (ROOT / relative).resolve()
            self.assertTrue(path.is_relative_to(ROOT.resolve()))
            self.assertTrue(path.is_file(), relative)
            self.assertRegex(expected, r"^[0-9a-f]{64}$")

    def test_saved_binned_power_moments_match_direct_calculation(self):
        from check_gaussian_bandpower_moments import verify_baseline

        rows, summary = verify_baseline()
        saved = pd.read_csv(
            ROOT / "repro/sensitivity/gaussian_bandpower_moments/uniform_baseline_moments.csv"
        )
        self.assertEqual(len(saved), 1680)
        calculated = pd.DataFrame(rows)
        for column in calculated.columns:
            np.testing.assert_allclose(
                calculated[column], saved[column], rtol=2e-12, atol=0,
                err_msg=column,
            )
        self.assertEqual(summary["mode_counts"], [315, 316])

    def test_metadata_identifies_software_and_upstream_data(self):
        metadata = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["upload_type"], "software")
        self.assertEqual(metadata["version"], "1.0.0")
        identifiers = {r["identifier"] for r in metadata["related_identifiers"]}
        self.assertTrue({
            "10.1103/4779-jw6l", "10.5281/zenodo.7132178",
            "10.5281/zenodo.11130700",
        }.issubset(identifiers))
        self.assertIn("CC BY 4.0", (ROOT / "DATA_LICENSE.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
