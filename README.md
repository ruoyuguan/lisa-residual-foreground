# LISA residual Galactic binary foreground

Analysis code and numerical results accompanying Ruo-Yu Guan and Yan Wang,
*Residual Galactic binary foreground in LISA stochastic gravitational-wave
background inference: source power concentration and spectral degeneracy*,
accepted for publication in Physical Review D.

Article: https://doi.org/10.1103/4779-jw6l

Preprint: https://arxiv.org/abs/2607.25349

Code: https://github.com/ruoyuguan/lisa-residual-foreground

## What is included

Version 1.0.0 contains the catalog matching, source-power calculation,
covariance and Fisher calculations, figure-generation scripts, numerical
tables, and tests associated with the accepted manuscript. It includes
the direct calculation of the skewness and excess kurtosis of binned power.
The original input catalogs, manuscript source, private correspondence,
historical exploratory programs, and private Git history are not included.

The calculation uses orbit-averaged long-wavelength Michelson X source powers
in equivalent incident strain. It is a scalar catalog calculation, not a
fit to the time-dependent, multichannel LISA data. See `METHODS.md` for the
assumptions and `DATA.md` for the numerical products and figure inputs.

## Check the supplied numerical results

The following commands do not require the original HDF5 inputs. Run them
from this repository's root with Python 3.11:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python repro/repro_scripts/check_gaussian_bandpower_moments.py
```

The last command independently reconstructs the binned-power moments from
the saved Fourier-power profiles and writes the resulting CSV and JSON.
The test suite checks scientific formulas and supplied numerical outputs;
it does not test the typography of the manuscript.

## Regenerate the figures

The supplied CSV and NPZ files are sufficient for:

```sh
python repro/repro_scripts/21_make_all_bin_publication_figures.py
python repro/repro_scripts/27_compute_sgwb_amplitude_dependence.py
```

These commands rewrite the corresponding numerical summaries and figures
under `repro/`. Figure 3(c) shows the fractional uncertainty
`sigma_omega_fixed / omega0`, not its reciprocal. The default random-subset
comparison uses 200 subsets at each size and seed 12345.

## Repeat the calculation from the public catalogs

Download the two upstream releases described in `UPSTREAM.md` and place the
required HDF5 files at the specified paths. The full calculation requires
substantially more time and memory than checking the supplied outputs.
From PowerShell, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File repro/repro_scripts/00_run_reproduction_pipeline.ps1 -MCMaxBatchElements 500000
python repro/repro_scripts/check_gaussian_bandpower_moments.py
```

The default calculation uses 50,000 independent random-phase realizations
per nonzero-residual bin. `-SkipRandomPhaseCheck` explicitly reuses the
supplied moment-check table; it is not a fresh Monte Carlo calculation.
Results and a new run log are written under `repro/`. For Linux or macOS,
use PowerShell 7 (`pwsh`) or run the Python stages listed in the driver.

`requirements.txt` records the tested package versions. Numerical results
should be compared with appropriate floating-point tolerances. PDF bytes
can differ because of creation dates or installed fonts even when the
plotted numbers are identical.

The checksums identify the original release files. Regenerating results or
figures can change their bytes without changing the numerical conclusions;
compare checksums before regeneration, and use the numerical tests afterward.

## Citation and reuse

Use the versioned citation in `CITATION.cff` when referring to these code
and numerical results, and cite the associated article for the scientific
analysis. Cite the original Sangria and Erebor releases when using their
data. `SHA256SUMS.txt` identifies the files distributed with this release.

Original code is licensed under MIT (`LICENSE`). Supplied numerical
results and figures are licensed under CC BY 4.0 (`DATA_LICENSE.md`).
Upstream data retain their original authorship and licensing; see
`UPSTREAM.md`. No license is granted here for the separately published
article or for excluded third-party materials.
