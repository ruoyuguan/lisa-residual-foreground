# Public input catalogs

The full calculation reads these two input files. They are not redistributed.

## Sangria (LDC2a), version 2

Maude Le Jeune and Stanislav Babak, *LISA Data Challenge Sangria (LDC2a)*,
version v2 (2022), Zenodo.

DOI: https://doi.org/10.5281/zenodo.7132178

Place `LDC2_sangria_training_v2.h5` at:

```text
data/raw/sangria/LDC2_sangria_training_v2.h5
```

The analysis uses the Galactic-binary injection catalogs. `obs/tdi` supplies
the observation sample count and cadence; its time-series values are not
fitted. Catalog matching also identifies verification-binary matches as
outside the DGB/IGB analysis scope. The blind data are not used.

## Erebor LDC2A training catalogs, version 1.0.4

Michael L. Katz and collaborators, *Erebor LDC2A Training Dataset Output
Catalogs*, version v1.0.4 (2024), Zenodo.

DOI: https://doi.org/10.5281/zenodo.11130700

From that release, extract `erebor_training_catalogs_v1.0.4_injection_comp.h5`
and place it at:

```text
data/raw/erebor/training/erebor_training_catalogs_v1.0.4/erebor_training_catalogs_v1.0.4_injection_comp.h5
```

The analysis uses the injection-comparison table, not the full posterior
chains. The source-matching audit records the association with the Sangria
injections and the adopted recovery cuts.

## Attribution

Both upstream records identify their license as CC BY 4.0. Retain their
citations and license notices when reusing upstream or derived materials.
Consult each original record for the complete creator list and source files.
This repository does not contain the Erebor fitting software or claim
authorship of either upstream catalog.
