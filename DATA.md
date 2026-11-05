# Numerical products and figure inputs

The CSV tables use one header row, decimal floating-point values, and column
names retained from the analysis. Frequencies with suffix `_Hz` are in Hz;
`_mHz` columns are in millihertz. Integrated strain powers have units of strain
squared and their variances have units of strain to the fourth power. Ratios,
source discreteness, skewness, excess kurtosis, and the residual-power factor
are dimensionless. `inf` denotes an unbounded prior width or a singular
uncertainty, not a missing finite observation.

## Figures

| Figure | Numerical input | Plotting program |
| --- | --- | --- |
| 1, residual spectrum | `repro/tables/all_bins_LWA_X_summary_*.csv` (560 bins) | `21_make_all_bin_publication_figures.py` |
| 2, discreteness and dilution | same table, 487 bins with nonzero residual | `21_make_all_bin_publication_figures.py` |
| 3, SGWB amplitude dependence | `all_bins_sgwb_amplitude_dependence_*.csv` (61 amplitudes, two covariances) | `27_compute_sgwb_amplitude_dependence.py` |
| 4, prior-width thresholds | `all_bins_publication_prior_requirement_summary_*.csv` (12 rows) | `21_make_all_bin_publication_figures.py` |
| 5, frequency subsets | `all_bins_bin_selection_robustness_prior_requirement_*.csv` (20 rows) | `21_make_all_bin_publication_figures.py` |

Full CSV filenames are listed in the directory and in `SHA256SUMS.txt`.
The five accepted-manuscript PDFs are under `repro/figures/all_bins/`.

In Figure 1(c), residual and SGWB shapes are normalized by their respective
maximum positive powers, using the `safe_norm` function. Figure 2 uses
`LWA_X_res_C2`, `residual_fraction_in_total_power`, and `abs_gamma2_total`.
The 36 highlighted bins are selected by decreasing `abs_gamma2_total`.
Figure 3(c) uses `sigma_omega_fixed_over_omega0`; legacy reciprocal columns
remain for compatibility but are not the plotted fractional uncertainty.
Infinite bounds are retained in tables and omitted or used to break plotted
curves as specified by the plotting programs.

Figure 5 stores median and 16th/84th percentiles for the displayed random
subset comparison. Individual draws are not stored; the program, seed 12345,
and 200-subset setting specify their regeneration. These percentile bands
describe variation among subsets, not parameter credible intervals.

## Other results

- `all_bins_fisher_baseline_*.csv`: amplitude, residual normalization,
  correlation, uncertainty increase, and omitted-residual shift.
- `all_bins_noise_dilution_*.csv`: the source-sum kurtosis before and after
  adding instrumental and fiducial SGWB power.
- `erebor_sangria_recovered_source_matching_*.csv`: the source associations
  and their summary, obtained from the upstream injection-comparison table.
- `all_bins_random_phase_moment_check_*.csv`: stored 50,000-realization
  source-sum moment checks and their summary. These are not Monte Carlo
  likelihood or posterior validation results.
- `repro/sensitivity/`: bin-width, orientation, frequency-support, covariance,
  instrumental-noise, and direct binned-power moment results.
- `residual_mode_power_profiles.npz`: numeric Fourier-grid and residual-power
  arrays. Load with `numpy.load(..., allow_pickle=False)`.
- `gaussian_bandpower_moments/uniform_baseline_moments.csv`: 560 bins for
  each of alpha = -2, 0, 2 at Omega_0 = 1e-11 and beta = 1. The manuscript
  quotes the alpha = 0 ranges. `verification_summary.json` records formulas,
  reconstruction errors, and input hashes.

CSV metadata names such as `n_eff_bandpower`, `snr_omega_fixed`, and the
random-phase diagnostic tags are retained for compatibility. The effective
degrees of freedom are derived from the saved mean and variance, not fitted
input parameters. An amplitude-to-uncertainty ratio is not a detection
significance. See `METHODS.md` and the article for the statistical assumptions.
