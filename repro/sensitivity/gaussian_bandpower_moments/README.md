# Direct moments of Gaussian binned power

Direct calculation of the binned-power moments used to assess the Gaussian
approximation for the uniform residual-power distribution. These are the
moments of the exponential sum itself, not of a matched reference distribution.

## Reproduce

From the repository root, with NumPy available:

```powershell
python -m unittest discover -s tests -p test_gaussian_bandpower_moments.py -v
python repro/repro_scripts/check_gaussian_bandpower_moments.py
```

The checker reads the existing all-bin CSV and the uniform source-power
profile in `repro/sensitivity/residual_fourier_profile/residual_mode_power_profiles.npz`.
It reuses `lisa_catalog_statistics.py` for the Fourier grid, instrumental
noise, SGWB spectrum, and Gaussian band-power moments. It verifies the cached
profile against the catalog residual powers and checks reconstructed means,
variances, effective degrees of freedom, and spectral components against the
existing CSV before writing results. It does not rerun source matching or
modify any existing scientific outputs.

## Calculation

Within the model of independent Gaussian Fourier amplitudes, let
`q_j = delta_f_F * (G_j + H_j)` be the mean power contributed by mode j.
The independent exponential contributions have cumulants
`kappa_r = (r - 1)! * sum_j q_j^r` in each bin. Thus:

- Mean: `sum_j q_j` (strain squared).
- Variance: `sum_j q_j^2` (strain to the fourth power).
- Skewness: `2 * sum_j q_j^3 / (sum_j q_j^2)^(3/2)`.
- Excess kurtosis: `6 * sum_j q_j^4 / (sum_j q_j^2)^2`.
- Effective degrees of freedom: `2 * (sum_j q_j)^2 / sum_j q_j^2`.

The implementation rescales the mode means before taking powers; this leaves
the standardized moments unchanged and avoids numerical underflow.
The unit tests include equal means, unequal means (whose higher moments do
not equal those of the matched chi-square distribution), extreme rescaling,
and invalid inputs.

## Outputs and interpretation

`uniform_baseline_moments.csv` contains one row per bin and spectral index.
`verification_summary.json` records ranges, reconstruction errors, and SHA-256
hashes of the input files, checker, and shared statistics module.

The calculation covers 560 bins, 176601 Fourier modes, alpha = -2, 0, 2,
Omega_0 = 1e-11, and beta = 1. Only the alpha = 0 ranges are newly quoted
in the manuscript and response: skewness about 0.11251--0.11275 and excess
kurtosis about 0.01899--0.01907. The per-bin `nu_eff_over_2M` verifies the
near-equal-mean limit; the global minimum alone is not used for this check.

These are exact model moments, not Monte Carlo estimates, and not the
source-sum kurtoses gamma_2,res or gamma_2,tot. They support the Gaussian
approximation for this baseline but do not evaluate RP/FR higher moments,
coherent LISA correlations, likelihood tails, or posterior coverage.
