param(
    [string]$Python = "python",
    [double]$BinWidthHz = 1.0e-5,
    [int]$MCNDraws = 50000,
    [int]$MCMaxBatchElements = 2000000,
    [switch]$SkipRandomPhaseCheck,
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"

if ($BinWidthHz -le 0) {
    throw "BinWidthHz must be positive."
}
if ([math]::Abs($BinWidthHz - 1.0e-5) -gt 1.0e-15) {
    throw "The reference calculation uses BinWidthHz = 1e-5. Use script 23 for bin-width sensitivity calculations."
}
if (-not $SkipRandomPhaseCheck -and $MCNDraws -lt 4) {
    throw "MCNDraws must be at least four."
}
if ($MCMaxBatchElements -lt 1) {
    throw "MCMaxBatchElements must be positive."
}
# Run from the repository root, for example:
#   powershell -ExecutionPolicy Bypass -File .\repro\repro_scripts\00_run_reproduction_pipeline.ps1
#
# The pipeline treats the full 0.4--6.0 mHz frequency grid as the analysis
# sample and writes products under .\repro.

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ReproRoot = Split-Path -Parent $ScriptDir
$ProjectRoot = Split-Path -Parent $ReproRoot

$Tables = Join-Path $ReproRoot "tables"
$Figures = Join-Path $ReproRoot "figures"
$AllBinFigures = Join-Path $Figures "all_bins"
$EreborRoot = Join-Path $ProjectRoot "data\raw\erebor\training\erebor_training_catalogs_v1.0.4"
$SangriaH5 = Join-Path $ProjectRoot "data\raw\sangria\LDC2_sangria_training_v2.h5"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $ReproRoot "reproduction_pipeline_$Timestamp.log"
$Stage02OptionalArguments = @()
if ($SkipRandomPhaseCheck) {
    $Stage02OptionalArguments += "--skip-random-phase-check"
}

New-Item -ItemType Directory -Force -Path $Tables, $Figures, $AllBinFigures | Out-Null

function ScriptPath([string]$Name) {
    return Join-Path $ScriptDir $Name
}

function TablePath([string]$Name) {
    return Join-Path $Tables $Name
}

function FigurePath([string]$Name) {
    return Join-Path $Figures $Name
}

function AllBinFigurePath([string]$Name) {
    return Join-Path $AllBinFigures $Name
}

function Require-File([string]$Path, [string]$Description) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Description not found: $Path"
    }
}

function Write-Log([string]$Message = "") {
    Write-Host $Message
    Add-Content -Path $LogFile -Encoding UTF8 -Value $Message
}

function Run-Step([string]$Name, [scriptblock]$Command) {
    Write-Log ""
    Write-Log "================================================================"
    Write-Log $Name
    Write-Log "================================================================"
    $global:LASTEXITCODE = 0
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Command 2>&1 |
            ForEach-Object {
                $line = $_.ToString()
                Write-Host $line
                Add-Content -Path $LogFile -Encoding UTF8 -Value $line
            }
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($null -ne $global:LASTEXITCODE -and $global:LASTEXITCODE -ne 0) {
        throw "Step failed with exit code $global:LASTEXITCODE`: $Name"
    }
}

function Write-Output-Manifest {
    Write-Log ""
    Write-Log "Principal products expected from this run:"
    $products = @(
        $(TablePath "erebor_sangria_recovered_source_matching_audit_C0p9_overlap0p9_f0p4_6mHz.csv"),
        $(TablePath "erebor_sangria_recovered_source_matching_summary_C0p9_overlap0p9_f0p4_6mHz.csv"),
        $(TablePath "all_bins_LWA_X_summary_C0p9_overlap0p9_f0p4_6mHz.csv"),
        $(TablePath "all_bins_random_phase_moment_check_summary_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.csv"),
        $(TablePath "all_bins_fisher_baseline_alpha0_summary_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.csv"),
        $(TablePath "all_bins_publication_prior_requirement_summary_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.csv"),
        $(TablePath "all_bins_cumulative_top_gamma2_I0_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.csv"),
        $(TablePath "all_bins_bin_selection_robustness_prior_requirement_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.csv"),
        $(Join-Path $ReproRoot "sensitivity\bin_width\bin_width_sensitivity_summary.csv"),
        $(Join-Path $ReproRoot "sensitivity\static_michelson_orientation\physical_snapshot_sensitivity_summary.csv"),
        $(Join-Path $ReproRoot "sensitivity\bin_width\frequency_smearing_scales.csv"),
        $(Join-Path $ReproRoot "sensitivity\residual_fourier_profile\residual_fourier_profile_sensitivity.csv"),
        $(Join-Path $ReproRoot "sensitivity\residual_fourier_profile\residual_mode_power_profiles.npz"),
        $(Join-Path $ReproRoot "sensitivity\residual_fourier_profile\residual_frequency_support_audit.csv"),
        $(Join-Path $ReproRoot "sensitivity\residual_covariance_ensemble\fixed_vs_gaussian_residual_covariance.csv"),
        $(Join-Path $ReproRoot "sensitivity\residual_covariance_ensemble\uniform_finite_source_random_phase_covariance.csv"),
        $(TablePath "all_bins_sgwb_amplitude_dependence_LWA_X_C0p9_overlap0p9_f0p4_6mHz.csv"),
        $(Join-Path $ReproRoot "sensitivity\instrument_noise\instrument_noise_normalization_sensitivity.csv"),
        $(AllBinFigurePath "all_bins_publication_residual_spectrum_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.pdf"),
        $(AllBinFigurePath "all_bins_random_phase_moment_check_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.pdf"),
        $(AllBinFigurePath "all_bins_source_power_concentration_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.pdf"),
        $(AllBinFigurePath "all_bins_publication_noise_dilution_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.pdf"),
        $(AllBinFigurePath "all_bins_publication_prior_requirement_summary_clean_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.pdf"),
        $(AllBinFigurePath "all_bins_cumulative_top_gamma2_I0_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.pdf"),
        $(AllBinFigurePath "all_bins_bin_selection_robustness_prior_requirement_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.pdf"),
        $(AllBinFigurePath "all_bins_sgwb_amplitude_dependence_LWA_X_C0p9_overlap0p9_f0p4_6mHz.pdf")
    )
    foreach ($product in $products) {
        if (Test-Path -LiteralPath $product) {
            Write-Log "  [ok] $product"
        }
        else {
            Write-Log "  [missing until generated] $product"
        }
    }
}





Push-Location $ProjectRoot
Set-Content -Path $LogFile -Value @(
    "All-bin reproduction pipeline log",
    "Start time: $(Get-Date -Format o)",
    "Host: $env:COMPUTERNAME",
    "User: $env:USERNAME"
) -Encoding UTF8

try {
    Write-Log "Project root: $ProjectRoot"
    Write-Log "Reproduction root: $ReproRoot"
    Write-Log "Tables: $Tables"
    Write-Log "Figures: $Figures"
    Write-Log "Log file: $LogFile"
    Write-Log "Python command: $Python"
    Write-Log "BinWidthHz: $BinWidthHz"
    Write-Log "Band-power covariance: sum over independent positive Fourier powers"
    Write-Log "SkipRandomPhaseCheck: $($SkipRandomPhaseCheck.IsPresent)"
    if (-not $SkipRandomPhaseCheck) {
        Write-Log "MCNDraws for the random-phase moment check: $MCNDraws"
    }
    Write-Log "MCMaxBatchElements: $MCMaxBatchElements"

    Run-Step "Preflight: input files, Python packages, and optional LaTeX tools" {
        Require-File $SangriaH5 "Sangria LDC2A training HDF5 file"
        Require-File (Join-Path $EreborRoot "erebor_training_catalogs_v1.0.4_injection_comp.h5") "Erebor injection-comparison file"
        & $Python (ScriptPath "00_preflight_environment.py")
    }

    if ($PreflightOnly) {
        Write-Log ""
        Write-Log "Preflight completed. No numerical pipeline steps were run because -PreflightOnly was supplied."
        Write-Output-Manifest
        Write-Log "End time: $(Get-Date -Format o)"
        return
    }

    Run-Step "01 one-to-one matching of accepted Erebor rows to Sangria sources" {
        & $Python (ScriptPath "19_match_erebor_recovered_sources.py") `
          --sangria-h5 $SangriaH5 `
          --injection-comp-h5 (Join-Path $EreborRoot "erebor_training_catalogs_v1.0.4_injection_comp.h5") `
          --output-audit (TablePath "erebor_sangria_recovered_source_matching_audit_C0p9_overlap0p9_f0p4_6mHz.csv") `
          --output-summary (TablePath "erebor_sangria_recovered_source_matching_summary_C0p9_overlap0p9_f0p4_6mHz.csv")
    }

    Run-Step "02 catalog band powers, covariance endpoints, and within-bin frequency sensitivity" {
        & $Python (ScriptPath "20_compute_catalog_bandpower_baseline.py") `
          --sangria-h5 $SangriaH5 `
          --datasets sky/dgb/cat sky/igb/cat `
          --matching-audit (TablePath "erebor_sangria_recovered_source_matching_audit_C0p9_overlap0p9_f0p4_6mHz.csv") `
          --omega0 1e-11 `
          --bin-width-Hz $BinWidthHz `
          --mc-n-draws $MCNDraws `
          --mc-max-batch-elements $MCMaxBatchElements `
          --output-bin-summary (TablePath "all_bins_LWA_X_summary_C0p9_overlap0p9_f0p4_6mHz.csv") `
          --output-noise-table (TablePath "all_bins_noise_dilution_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.csv") `
          --output-fisher-table (TablePath "all_bins_fisher_baseline_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.csv") `
          --output-validation-table (TablePath "all_bins_random_phase_moment_check_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.csv") `
          --output-validation-summary (TablePath "all_bins_random_phase_moment_check_summary_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.csv") `
          --output-fisher-alpha0-summary-csv (TablePath "all_bins_fisher_baseline_alpha0_summary_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.csv") `
          --output-fisher-alpha0-summary-tex (TablePath "all_bins_fisher_baseline_alpha0_summary_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.tex") `
          --output-validation-figure-png (AllBinFigurePath "all_bins_random_phase_moment_check_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.png") `
          --output-validation-figure-pdf (AllBinFigurePath "all_bins_random_phase_moment_check_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.pdf") `
          @Stage02OptionalArguments
    }

    Run-Step "03 finite-source random-phase covariance for the uniform distribution" {
        & $Python (ScriptPath "26_compute_uniform_finite_source_covariance.py") `
          --all-bin-summary (TablePath "all_bins_LWA_X_summary_C0p9_overlap0p9_f0p4_6mHz.csv") `
          --mode-power-profiles (Join-Path $ReproRoot "sensitivity\residual_fourier_profile\residual_mode_power_profiles.npz") `
          --output-table (Join-Path $ReproRoot "sensitivity\residual_covariance_ensemble\uniform_finite_source_random_phase_covariance.csv")
    }

    Run-Step "04 dependence on the fiducial SGWB amplitude" {
        & $Python (ScriptPath "27_compute_sgwb_amplitude_dependence.py") `
          --all-bin-summary (TablePath "all_bins_LWA_X_summary_C0p9_overlap0p9_f0p4_6mHz.csv") `
          --mode-power-profiles (Join-Path $ReproRoot "sensitivity\residual_fourier_profile\residual_mode_power_profiles.npz") `
          --output-table (TablePath "all_bins_sgwb_amplitude_dependence_LWA_X_C0p9_overlap0p9_f0p4_6mHz.csv") `
          --output-png (AllBinFigurePath "all_bins_sgwb_amplitude_dependence_LWA_X_C0p9_overlap0p9_f0p4_6mHz.png") `
          --output-pdf (AllBinFigurePath "all_bins_sgwb_amplitude_dependence_LWA_X_C0p9_overlap0p9_f0p4_6mHz.pdf")
    }

    Run-Step "05 publication figures and frequency-subset summaries" {
        & $Python (ScriptPath "21_make_all_bin_publication_figures.py") `
          --all-bin-summary (TablePath "all_bins_LWA_X_summary_C0p9_overlap0p9_f0p4_6mHz.csv") `
          --omega0 1e-11 `
          --output-residual-spectrum-png (AllBinFigurePath "all_bins_publication_residual_spectrum_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.png") `
          --output-residual-spectrum-pdf (AllBinFigurePath "all_bins_publication_residual_spectrum_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.pdf") `
          --output-concentration-png (AllBinFigurePath "all_bins_source_power_concentration_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.png") `
          --output-concentration-pdf (AllBinFigurePath "all_bins_source_power_concentration_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.pdf") `
          --output-noise-png (AllBinFigurePath "all_bins_publication_noise_dilution_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.png") `
          --output-noise-pdf (AllBinFigurePath "all_bins_publication_noise_dilution_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.pdf") `
          --output-prior-png (AllBinFigurePath "all_bins_publication_prior_requirement_summary_clean_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.png") `
          --output-prior-pdf (AllBinFigurePath "all_bins_publication_prior_requirement_summary_clean_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.pdf") `
          --output-prior-csv (TablePath "all_bins_publication_prior_requirement_summary_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.csv") `
          --output-cumulative-i0-png (AllBinFigurePath "all_bins_cumulative_top_gamma2_I0_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.png") `
          --output-cumulative-i0-pdf (AllBinFigurePath "all_bins_cumulative_top_gamma2_I0_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.pdf") `
          --output-cumulative-i0-csv (TablePath "all_bins_cumulative_top_gamma2_I0_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.csv") `
          --output-robustness-png (AllBinFigurePath "all_bins_bin_selection_robustness_prior_requirement_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.png") `
          --output-robustness-pdf (AllBinFigurePath "all_bins_bin_selection_robustness_prior_requirement_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.pdf") `
          --output-robustness-csv (TablePath "all_bins_bin_selection_robustness_prior_requirement_LWA_X_Omega1e-11_C0p9_overlap0p9_f0p4_6mHz.csv")
    }

    Run-Step "06 dependence on frequency-bin width" {
        & $Python (ScriptPath "23_run_bin_width_sensitivity.py")
    }

    Run-Step "07 static Michelson orientation sensitivity" {
        & $Python (ScriptPath "24_run_static_michelson_orientation_sensitivity.py")
    }

    Run-Step "08 intrinsic-drift and annual-Doppler frequency scales" {
        & $Python (ScriptPath "25_summarize_frequency_smearing_scales.py")
    }



    Run-Step "09 direct moments of Gaussian binned power" {
        & $Python (ScriptPath "check_gaussian_bandpower_moments.py")
    }

    Run-Step "10 scientific consistency and regression checks" {
        & $Python -m unittest discover -s tests -v
    }



    Write-Log ""
    Write-Log "All-bin reproduction pipeline completed successfully."
    Write-Log "Tables written to: $Tables"
    Write-Log "Figures written to: $Figures"
    Write-Output-Manifest
    Write-Log "End time: $(Get-Date -Format o)"
}
finally {
    Pop-Location
    Write-Host "Log saved to: $LogFile"
}
