# Calculation and conventions

The reference calculation covers 0.4--6.0 mHz in bins of width 10 microhertz.
The 560 bins include 487 with nonzero catalog residual power.

1. Accepted Erebor injection-comparison rows are associated one-to-one with
   Sangria sources using the source-parameter vector and the adopted match
   cuts. Only recovered DGB/IGB sources enter the subtraction performed here.
2. Source powers use the Sangria amplitude convention and a long-wavelength
   Michelson X response averaged over 12 LISA orbital phases. Source and
   instrumental powers are divided by the same single-channel isotropic
   single-polarization power response, 3/20, to express equivalent incident
   strain. This is not the two-channel 3/10 response convention.
3. The one-year sample count and cadence determine the positive Fourier
   frequencies. Each reference bin contains 315 or 316 modes. With independent
   Gaussian Fourier amplitudes, the mean binned power is `df sum(S_j)` and
   its variance is `df**2 sum(S_j**2)`.
4. The reference residual power is distributed uniformly among the Fourier
   frequencies within each bin. More concentrated assignments test unresolved
   frequency structure while conserving the catalog residual power.
5. Gaussian, finite-source random-phase, and fixed-residual calculations keep
   the same mean spectrum and change the diagonal variance. Cross-frequency
   correlations are not included. Each local Fisher calculation holds its
   covariance fixed at that calculation's fiducial parameter values.
6. The source-sum moment check tests `gamma_2,res = -3 C2 / 2`. This statistic
   is distinct from the skewness and excess kurtosis of binned Fourier power.
   The latter are calculated directly from independent exponential mode
   powers by `check_gaussian_bandpower_moments.py`.

This is an orbit-averaged scalar catalog calculation. It does not propagate
coherent time-domain binary waveforms, finite-window spectral leakage,
cross-frequency correlations, or annual modulation into a multichannel
likelihood. The supplied numerical results do not validate an exact
non-Gaussian posterior or its coverage.

## Programs

`repro/repro_scripts/00_run_reproduction_pipeline.ps1` lists the full stage
commands. Stages 19--27 perform catalog matching, the reference calculation,
figure construction, bin-width and orientation comparisons, frequency-scale
checks, the random-phase covariance calculation, and the amplitude dependence.
The shared modules are `catalog_matching.py`, `galactic_binary_power.py`, and
`lisa_catalog_statistics.py`. The direct binned-power moment calculation is
also supplied and can be run without downloading the original catalogs.
