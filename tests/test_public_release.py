import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "repro" / "repro_scripts"))


class PublicReleaseTests(unittest.TestCase):
    def test_release_manifest_excludes_article_figure_pdfs(self):
        paths = [
            line.split("  ", 1)[1]
            for line in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
        ]
        self.assertFalse(any(path.lower().endswith(".pdf") for path in paths))

    def test_data_license_excludes_article_figures(self):
        text = (ROOT / "DATA_LICENSE.md").read_text(encoding="utf-8")
        self.assertIn("numerical tables and Fourier-power profiles", text)
        self.assertIn("does not license article figures", text)

    def test_recorded_dependencies_include_jinja2(self):
        names = {
            line.partition("==")[0].strip().casefold()
            for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertIn("jinja2", names)

    def test_preflight_reports_jinja2_version(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "repro/repro_scripts/00_preflight_environment.py")],
            check=True, capture_output=True, text=True,
        )
        self.assertIn("jinja2=", result.stdout)

    def test_catalog_summary_exports_csv_and_latex(self):
        path = ROOT / "repro/repro_scripts/20_compute_catalog_bandpower_baseline.py"
        spec = importlib.util.spec_from_file_location("baseline_export_test", path)
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        table = pd.DataFrame({
            "alpha": [0.0, 2.0],
            "bin_set": ["all_analysis_bins", "all_analysis_bins"],
            "n_bins": [560, 560],
            "no_prior_inflation": [1.136, 1.2],
            "no_prior_abs_corr": [0.475, 0.5],
            "sigma_beta_for_threshold": [0.00728, 0.008],
            "bias_ignore_over_sigma_fixed": [119.5, 120.0],
        })
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "summary.csv"
            tex_path = Path(directory) / "summary.tex"
            baseline.write_alpha0_summary(table, csv_path, tex_path)
            saved = pd.read_csv(csv_path)
            self.assertEqual(len(saved), 1)
            self.assertEqual(float(saved["alpha"].iloc[0]), 0.0)
            self.assertIn("\\begin{tabular}", tex_path.read_text(encoding="utf-8"))

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
        self.assertEqual(metadata["version"], "1.0.1")
        citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        self.assertIn("version: 1.0.1", citation)
        self.assertNotIn("10.5281/zenodo.23974185", citation)
        identifiers = {r["identifier"] for r in metadata["related_identifiers"]}
        self.assertTrue({
            "10.1103/4779-jw6l", "10.5281/zenodo.7132178",
            "10.5281/zenodo.11130700",
        }.issubset(identifiers))
        self.assertIn("CC BY 4.0", (ROOT / "DATA_LICENSE.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
