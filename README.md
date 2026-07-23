# FAST FRB Post-search Processing

This directory contains the LPT workspace tools that turn confirmed FAST FRB candidates into calibrated H5 products, reviewed burst labels, physical measurements, result tables, and optional HTML dashboards.

It is a processing subdirectory inside the local LPT workspace. The old standalone AFTER landing-page material has been removed from this README so the file describes the scripts that are actually here.

## Workflow

```text
raw FAST FITS + TOA/DM
  -> cut_burst_data.py or batch_processing/batch_cut_burst_data.py
  -> cut burst H5
  -> calibration.py or batch_processing/batch_calibration.py
  -> calibrated *_cal.h5
  -> burst_detect.py
  -> H5 attrs["bursts"] + detections.json
  -> burst_analysis.py
  -> burst_results.csv + diagnostic plots
  -> burst_dashboard.py
```

The pipeline can also start from an intermediate product:

| Starting point | Required inputs | Continue with |
|---|---|---|
| Raw FAST FITS | FITS directory, source, date, beam, DM, TOA seconds, output directory | Cut, calibrate, detect, review, analyze |
| Cut H5 | `.h5` files and matching `_0001.fits`, RA, DEC, calibration npz | Calibrate, detect, review, analyze |
| Calibrated H5 | `*_cal.h5`, detector model, output directory | Detect, review, analyze |
| Labeled calibrated H5 | `*_cal.h5` with H5 attr `bursts` | Analyze, export, dashboard |

## Files

| Path | Role |
|---|---|
| `cut_burst_data.py` | Helper functions and a local configured example for cutting burst-centered H5 files from raw FAST FITS. |
| `calibration.py` | Helper functions and a local configured example for polarization/flux calibration, downsampling, RFI masks, quick-look plots, and `*_cal.h5`. |
| `burst_detect.py` | CLI for automatic, semi-automatic, or manual burst-region labeling. Writes H5 `attrs["bursts"]`, `detections.json`, and review plots. |
| `burst_analysis.py` | CLI for DM/RM/polarization/flux/fluence/width/SNR measurements from accepted burst labels. |
| `burst_dashboard.py` | CLI for a static HTML observation dashboard from `burst_results.csv`. |
| `burst_dm.py` | Fine DM search utilities used by `burst_analysis.py`. |
| `burst_pol.py` | RM, PA, PAV, and polarization utilities. |
| `burst_properties.py` | Flux, fluence, width, bandwidth, and SNR utilities. |
| `rfi_utils.py` | Calibration and analysis RFI masking helpers. |
| `ZeithAngle.py` | FAST zenith-angle and gain helper functions. |
| `gain_para.csv` | FAST beam gain parameters. |
| `highcal_20201014_psr_tny.npz` | Default noise-calibration reference file. |
| `models/` | Burst detector weights. |
| `batch_processing/` | Batch wrappers for cutting, legacy FITS-to-H5 conversion, selected long-period cuts, and calibration. |
| `skills/fast-frb-observation-processing/` | Codex operating protocol for this processing workflow. |
| `test_burst_dashboard.py` | Dashboard unit tests. |
| `requirements.txt` | Python dependencies. |

## Repository Boundary

Git tracks the processing code, tests, workflow documentation, the production
detector, and the small calibration/gain assets needed by the documented
defaults. Raw observations, generated H5 files, review plots, result tables,
local batch catalogs, caches, and retired model checkpoints remain local and
are covered by `.gitignore`.

The tracked production detector is
`models/best_model_yolo11n_ema.pth`. Put experimental or replacement
checkpoints in `models/` for local testing; they will stay untracked unless the
ignore policy is deliberately updated.

## Environment

Install dependencies in the environment used for this LPT checkout:

```bash
python -m pip install -r requirements.txt
```

For GPU detector inference, install `torch` and `torchvision` with the command that matches the machine CUDA driver. The dependency list intentionally does not pin CUDA-specific PyTorch wheels.

Quick validation from this directory:

```bash
python -m py_compile cut_burst_data.py calibration.py burst_detect.py burst_analysis.py burst_dashboard.py burst_dm.py burst_pol.py burst_properties.py rfi_utils.py ZeithAngle.py batch_processing/batch_calibration.py batch_processing/batch_cut_burst_data.py batch_processing/batch_cut_selected_long_period.py batch_processing/fits_to_h5.py test_burst_dashboard.py
python burst_detect.py --help
python burst_analysis.py --help
python burst_dashboard.py --help
```

`cut_burst_data.py` and `calibration.py` have local configured `__main__` examples. New observations should use the batch wrappers or import their helper functions so old hard-coded paths are not reused accidentally.

## Cutting Raw FITS

Batch wrapper:

```bash
python batch_processing/batch_cut_burst_data.py \
  --burst-txt batch_processing/FRBXXXX_Burst.txt \
  --output-root /path/to/after_data/H5_Cut/FRBXXXX \
  --save-frb-name FRBXXXX \
  --segment-length 65536 \
  --workers 8
```

`FRB*_Burst.txt` format:

```text
base project name date beam dm time
```

The wrapper groups rows by raw data path/date/beam/DM, copies the first beam-matched FITS into the output date directory for calibration, cuts each TOA, and writes `obs_info.json`.

For selected long-period candidates with row-level segment lengths:

```bash
python batch_processing/batch_cut_selected_long_period.py \
  --plan-txt batch_processing/Selected_LongPeriod_Burst.txt \
  --output-root /path/to/after_data/LPT_Selected_Cut \
  --workers 8
```

## Converting Legacy Cut FITS

Use `fits_to_h5.py` when older burst FITS cuts need to be converted to the current H5 schema:

```bash
python batch_processing/fits_to_h5.py \
  --asd-root /path/to/after_data \
  --output-root /path/to/after_data/H5_Cut \
  --catalog-dir batch_processing
```

It copies calibration FITS files ending in `_0001.fits` and writes H5 files with the same datasets/attrs as `cut_burst_data.py`.

## Calibration

Batch wrapper:

```bash
python batch_processing/batch_calibration.py \
  --root-dir /path/to/after_data/H5_Cut \
  --cal-root /path/to/after_data/H5_Cut/H5_Cal \
  --dm-file batch_processing/h5_calibration_dm_file.txt \
  --cal-npz highcal_20201014_psr_tny.npz \
  --workers 8
```

`h5_calibration_dm_file.txt` format:

```text
FRB_name DM RA DEC
```

Common saved-resolution choices:

- `--down-time` / `--down-freq` omitted: save at the automatic plot-friendly resolution.
- `--down-time 1`: keep raw time resolution for peak-flux comparison.
- `--down-freq 1`: keep raw frequency channels for detailed RFI or spectral inspection.

The calibrated H5 stores Stokes I/Q/U/V in `data`, plus `freq`, `rfi_mask`, `rfi_channel`, `gain`, and `gain_err`.

## Detection And Label Review

Automatic mode:

```bash
python burst_detect.py \
  --mode auto \
  --cal-dir /path/to/calibrated_h5 \
  --model-path models/best_model_yolo11n_ema.pth \
  --model-name yolo11n \
  --output-dir /path/to/detections_auto
```

Detection outputs:

- H5 `attrs["bursts"]`, the source of truth for analysis.
- `detections.json`, the resume and review ledger.
- `plots/*_det.png`, review plots.

Auto and semi-auto detection run a single inference pass on the calibrated
Stokes I. After burst boxes are confirmed, detection uses the non-burst samples
to recompute the analysis-style Stokes-I/V RFI union, writes the `burst_rfi_*`
products, and draws the final masked residual plot.

After confidence filtering, boxes wider than `--max-horizontal-aspect` times
their height are removed (default `3`). For boxes with positive-area overlap,
only the largest box is retained before NMS.

Use `--mode semi-auto` to relabel only files removed from an existing `detections.json`. Use `--mode manual` when model suggestions are misleading. In the interactive UI, `x` writes an intentionally empty burst list for a file; `q` or `Esc` saves completed progress and exits normally without marking the current file as processed.

## Analysis

```bash
python burst_analysis.py \
  --cal-dir /path/to/calibrated_h5 \
  --output-dir /path/to/analysis_output \
  --dm-range 5 \
  --dm-step 0.1 \
  --rm-min -1000 \
  --rm-max 1000 \
  --n-rm 100000
```

Measured quantities include TOA, peak flux, fluence, width, burst bandwidth, SNR, DM, RM, linear/circular/total polarization, PA, and PAV. Use an isolated output directory for reruns with different DM/RM ranges.

Primary outputs:

```text
burst_results.csv
DM/RM/polarization diagnostic plots
```

## Dashboard

```bash
python burst_dashboard.py \
  --csv /path/to/analysis/burst_results.csv \
  --output /path/to/analysis/burst_dashboard.html \
  --analysis-dir /path/to/analysis \
  --source FRBNAME \
  --date YYYYMMDD \
  --reference-dm 539 \
  --rm-significance-threshold 5 \
  --top-n 10
```

The dashboard is a self-contained HTML report for local review or PDF printing.

## H5 Contracts

Cut H5:

```text
data: (nsamp, npol, nchan)
freq: (nchan,), MHz
attrs: start_sample, file_mjd, toa_sec, time_reso, npol, nchan,
       segment_length, obs_start_mjd, dm
```

Calibrated H5:

```text
data:        (4, nsamp, nchan), Stokes I/Q/U/V, Jy
freq:        (nchan,), MHz
rfi_mask:    (nsamp, nchan), bool
rfi_channel: (nchan,), bool
gain:        (nchan,), K/Jy
gain_err:    (nchan,), K/Jy
attrs: time_reso_raw, time_reso, down_time, down_freq,
       dm, beam, ra, dec
```

Detection labels:

```json
{"time_start": 120, "time_end": 180, "freq_start": 40, "freq_end": 500, "confidence": 0.82}
```

## Output Policy

Generated products are local run artifacts:

- H5/FITS data: `*.h5`, `*.fits`, `*_cal.h5`
- Diagnostic images and dashboard exports: `*.jpg`, `*.png`, `burst_dashboard.html`
- Detection and analysis outputs: `detections/`, `analysis_output/`, `analysis_outputs/`
- Local batch tables: `batch_processing/*.txt`
- Retired local checkpoints: `models/*.old`

Keep new runs in their own output directories. Do not mix new analysis, relabeling, or dashboard outputs into an older run unless the overwrite is intentional.
