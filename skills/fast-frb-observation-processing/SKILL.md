---
name: fast-frb-observation-processing
description: "Use when processing FAST FRB burst observations with AFTER/data_processing scripts: raw FAST FITS, user-provided TOA lists, cut H5, calibrated H5, already detected H5 attrs['bursts'], burst_detect.py label review, detections.json resume behavior, energy/polarization/DM/RM analysis tables, or installing and validating this workflow on another machine."
---

# AFTER FAST FRB Observation Processing

Use this skill as the operating protocol for AFTER, the AI-assisted FAST Transient End-to-end Reduction workflow. The skill may be installed separately from the scripts, so never assume the skill directory is the script root.

Normal chain:

```text
raw FAST FITS
  + user-provided TOA list
  -> cut_burst_data.py
  -> cut H5
  -> calibration.py
  -> *_cal.h5
  -> burst_detect.py
  -> human review/correction of labels
  -> H5 attrs["bursts"] + detections.json
  -> burst_analysis.py
  -> energy/polarization/DM/RM measurements
  -> burst_results.csv + diagnostic plots
```

## Locate the Script Root

Before any processing command, locate and verify the `data_processing` root.

Use the first working option:

1. Current working directory, if it contains the sentinel files.
2. `DATA_PROCESSING_ROOT`, if set.
3. A path the user gives, such as a cloned `data_processing` repo.

Sentinel files:

```text
cut_burst_data.py
calibration.py
burst_detect.py
burst_analysis.py
rfi_utils.py
ZeithAngle.py
gain_para.csv
```

For detection, also require `models/best_model_yolo11n_ema.pth` unless the user supplies another model. For calibration, require `highcal_20201014_psr_tny.npz` unless the user supplies another noise-calibration file.

If no script root is available, tell the user the installed skill only provides the agent protocol and ask for the cloned `data_processing` path. Do not run commands from the skill folder.

## Install or Validate the Workflow

When the user asks about installing this workflow on another machine, keep instructions concise and point to the repo README for full commands.

Minimum install checks:

1. Clone or copy the `data_processing` repo.
2. Install Python dependencies from `requirements.txt`, with CUDA-specific PyTorch chosen for that machine when GPU detection is needed.
3. Copy `skills/fast-frb-observation-processing/` into the Codex skills directory.
4. Optionally set `DATA_PROCESSING_ROOT` to the repo root so future agents can find the scripts.
5. Run syntax, import, CLI, and skill validation checks from the README.

Do not create extra files inside the skill folder during installation. The skill is intentionally just `SKILL.md` plus `agents/openai.yaml`.

## First Response for Processing

Determine where the user wants to start:

- **Raw FITS**: require user-provided TOA seconds; run cut, calibration, detect, review, energy/polarization analysis, table export.
- **Cut H5**: skip cut; run calibration, detect, review, energy/polarization analysis, table export.
- **Calibrated H5**: skip cut/calibration; run detect, review, energy/polarization analysis, table export.
- **Already detected H5**: verify `attrs["bursts"]`; run energy/polarization analysis and table export.

Ask only for missing blocking inputs. Prefer discovering paths and file metadata locally with shell commands before asking.

Minimum inputs:

| Stage | Required inputs |
|---|---|
| Raw FITS | `data_path`, output cut directory, FRB name, date, beam, DM, user-provided burst TOAs in seconds from observation start, segment length if overriding default, worker count |
| Cut H5 | cut H5 directory, calibration output directory, RA, DEC, `highcal_*.npz`, downsample policy, worker count |
| Calibrated H5 | calibrated H5 directory, model path, model name, detection output directory |
| Energy/polarization analysis | output directory for `burst_results.csv` and plots, DM search range/step, RM range and number of trial points, target downsample if any |

Confirm science choices when they matter:

- Use source- and date-appropriate DM for cutting.
- Preserve raw time resolution with `down_time=1` when peak flux must be comparable to raw-time legacy measurements.
- Use `down_freq=None` for standard plot-like frequency resolution unless the user needs raw channels.
- Choose RM search range from prior knowledge or ask before using a narrow range.

## Execution Rules

- Work from the verified script root and report absolute paths.
- Do not trust hard-coded defaults in `cut_burst_data.py`, `calibration.py`, or batch scripts; many are machine-local examples. Collect explicit inputs or generate a temporary runner.
- Treat `batch_processing/*.txt` as local, untracked observation catalogs. If batch wrappers are used, ask for explicit `--burst-txt`, `--dm-file`, `--catalog-dir`, or `--plan-txt` paths instead of assuming the repo contains those files.
- Use the user's Python environment when given. Otherwise inspect available environments before GPU or plotting workloads.
- Prefer importing functions or generating a one-off runner for cut/calibration, because the single-observation parameters still live in `if __name__ == "__main__"` blocks.
- Do not infer TOAs from plots or filenames. The user or upstream search process must provide TOA seconds for raw-FITS cutting.
- Run `burst_detect.py` and `burst_analysis.py` directly through their CLIs.
- Preserve existing outputs unless the user asks to overwrite.
- Keep a run log: inputs, commands, counts, output paths, skipped files, and warnings.
- Start energy/polarization analysis only after detection quality is checked and the user accepts auto labels or confirms corrections are finished.

## Data Contracts

Raw FAST directory:

- Contains many FITS files for one observation date.
- Beam is identified by `Mxx` in filenames.
- A calibration/noise FITS for the beam should be in the same directory and usually ends with `_0001.fits`.
- Burst TOA uses seconds from observation start.

Cut H5 from `cut_burst_data.py`:

```text
data: (nsamp, npol, nchan)
freq: (nchan,), MHz
attrs: start_sample, file_mjd, toa_sec, time_reso, npol, nchan,
       segment_length, obs_start_mjd, dm
```

Calibrated H5 from `calibration.py`:

```text
data: (4, nsamp, nchan), Stokes I/Q/U/V in Jy
freq: (nchan,), MHz
rfi_mask: (nsamp, nchan)
rfi_channel: (nchan,)
gain, gain_err: (nchan,)
attrs: time_reso_raw, time_reso, down_time, down_freq,
       plot_down_time, plot_down_freq, dm, beam, ra, dec
```

After detection:

```text
attrs["bursts"] = JSON list of regions
burst_rfi_mask, burst_rfi_channel = detection-stage RFI from non-burst noise
burst_rfi_method = "entropy" by default, or "fft" with --rfi-fft
```

Each burst region uses calibrated-H5 saved indices:

```json
{"time_start": 120, "time_end": 180, "freq_start": 40, "freq_end": 500, "confidence": 0.82}
```

`attrs["bursts"]` is the analysis source of truth. `detections.json` is the detection skip/resume ledger.

## Preflight Checklist

Before any stage:

1. Verify script root sentinel files.
2. Check the input directory:
   - Raw FITS start: count `*Mxx*.fits`, inspect one FITS header for `TBIN`, `NCHAN`, `NPOL`, `NSBLK`, `NAXIS2`.
   - Cut H5 start: count `.h5` excluding `_cal.h5`; verify matching `_0001.fits` exists.
   - Cal H5 start: count `*_cal.h5`; inspect one H5 schema and attrs.
3. Check output directories and existing products.
4. Check whether previous `detections.json` exists.

## Stage 1: Cut Raw FITS

Require a TOA list in seconds from observation start, supplied by the user or an upstream search product. Validate that all TOAs fall within observation duration. Codex may reformat, sort, deduplicate, or range-check TOAs, but must not invent them.

Recommended implementation:

1. Import `read_obs_info`, `calc_dispersion_shift`, `cut_one_burst`, and `save_obs_json` from `cut_burst_data`.
2. Build sorted `file_list` using beam pattern `M{beam:02d}`.
3. Copy the first matching beam FITS into the cut output directory so calibration can find `_0001.fits`.
4. Compute dispersion shifts with requested DM.
5. Run `cut_one_burst` for each TOA, using a process pool for multiple bursts.
6. Run `save_obs_json`.

Verify:

- Cut H5 count matches valid TOA count unless some TOAs were outside the observation.
- Filenames follow `{frb}-{date}-M{beam:02d}-{fits_number:04d}-{start_sample:09d}.h5`.
- One sample H5 has `data`, `freq`, and required attrs.
- `obs_info.json` exists.

Common blockers: no FITS matching beam, missing `_0001.fits`, TOA outside observation, FITS missing `DATA` or `DAT_FREQ`.

## Stage 2: Calibration

Inputs: cut H5 directory, calibration output directory, RA, DEC, beam inferred from filenames, `highcal_*.npz`, optional `down_time`, `down_freq`, `rfi_fft`.

Recommended implementation:

1. Import `find_cal_fits`, `fold_noise_cal`, `load_t_cal`, and `process_one_burst` from `calibration`.
2. Group cut H5 files by beam from the `Mxx` filename token.
3. For each beam, find matching `_0001.fits`, fold `noise_cal`, load `t_cal`, and process each H5.
4. Use `rfi_fft=True` unless the user requests entropy RFI at calibration time.

Downsample policy:

- `down_time=None`, `down_freq=None`: save at automatic plot-friendly resolution.
- `down_time=1`: preserve raw time resolution for peak flux comparisons.
- `down_freq=1`: preserve raw frequency channels for detailed RFI or spectra.
- In calibrated H5, `time_reso` is already the effective saved resolution.

Verify:

- Each input H5 has a matching `*_cal.h5`.
- Quick-look `.jpg` plots exist.
- One calibrated H5 has `data`, `freq`, `rfi_mask`, `rfi_channel`, `gain`, `gain_err`.
- Check attrs `down_time`, `down_freq`, `time_reso_raw`, `time_reso`, `nchan_raw`, `nchan`.

## Stage 3: Burst Detection and Label Review

Run auto detection first unless the user asks for manual-only marking:

```bash
python burst_detect.py \
  --mode auto \
  --cal-dir /path/to/cal_date \
  --model-path models/best_model_yolo11n_ema.pth \
  --model-name yolo11n \
  --output-dir /path/to/detections_auto
```

Important behavior:

- The script recursively finds `*_cal.h5` below `--cal-dir`.
- It writes `attrs["bursts"]` into each H5.
- It writes `detections.json` and `plots/*_det.png`.
- Existing entries in `detections.json` are skipped on rerun, including zero-burst entries.
- Detection-stage RFI is entropy by default; pass `--rfi-fft` only when FFT RFI is requested.

After auto detection, inspect quality before analysis. Automatic boxes are provisional; the user must check whether labels are acceptable before analysis starts. Build a review manifest in the detection output directory, for example:

```text
detection_review_manifest.txt
file_name | reason | plot_path | h5_path
```

Flag files with no detections, low confidence, overlapping/split boxes, edge-clipped boxes, implausibly wide or narrow boxes, boxes far from the profile peak, or visually missed/merged bursts.

Tell the user which files need review, where the plots are, how many labels look acceptable, and exactly how to relabel the bad files.

## Stage 4: Manual or Semi-Auto Correction

Codex coordinates the review. Prepare the bad-file list, give the command or launch the interactive window when supported, explain controls, then pause until the user says the labels are fixed.

Use semi-auto for bad auto labels:

1. Remove only bad filenames from that output directory's `detections.json`.
2. Rerun `burst_detect.py --mode semi-auto` with the same `--cal-dir` and `--output-dir`.
3. The script skips all still-listed files and opens only the removed files.
4. Figure controls:
   - Enter accepts current boxes.
   - Left-drag draws boxes.
   - Right-click undoes the latest manual box.
   - `x` writes an intentionally empty `bursts` list for this file and moves on.
   - `q` or Esc quits.

Use manual mode when the model is misleading:

```bash
python burst_detect.py \
  --mode manual \
  --cal-dir /path/to/cal_date \
  --output-dir /path/to/detections_manual
```

When handing control to the user, request a concrete completion signal:

```text
Please relabel the files in detection_review_manifest.txt.
When finished, tell me "labels are fixed" and I will verify the H5 attrs, run energy/polarization analysis, and export burst_results.csv.
```

After the user says labels are fixed:

- Verify every intended H5 has `attrs["bursts"]`.
- Count total files, intentionally empty files, and burst regions.
- Check reviewed files now have plausible non-empty or intentionally empty regions.
- Check detection-stage `burst_rfi_mask`, `burst_rfi_channel`, and `burst_rfi_method` exist when labels were written by current `burst_detect.py`.
- Keep or report the final `detections.json` path.
- Run energy/polarization analysis and export `burst_results.csv` unless verification fails.

Preserve user-corrected labels. If changing output directories, remember that H5 attrs drive analysis while `detections.json` controls detection skip/resume.

## Stage 5: Energy/Polarization Analysis and Results Table

Run after detection is accepted:

```bash
python burst_analysis.py \
  --cal-dir /path/to/cal_date \
  --output-dir /path/to/analysis_output \
  --dm-range 5 \
  --dm-step 0.1 \
  --rm-min -1000 \
  --rm-max 1000 \
  --n-rm 100000
```

Treat these values as examples, not defaults. Confirm DM/RM ranges against source knowledge before long runs.

Notes:

- `burst_analysis.py` processes `*_cal.h5` files directly inside one directory. Loop over date directories for multiple dates.
- `--dm-range` is centered on the cut DM stored in each H5.
- `--rm-min`, `--rm-max`, `--n-rm`, and `--n-boot` control RM search and uncertainty work.
- Pass `--target-down-time` and `--target-down-freq` only for coarser analysis than the saved calibrated H5. Target factors must be integer multiples of saved `down_time/down_freq`.
- Pass `--rfi-fft` only if FFT RFI is requested in analysis; otherwise analysis uses entropy-style RFI.

Analysis logic:

- Rebuild `noise_mask` from accepted burst labels.
- Subtract per-channel baseline before analysis-stage RFI.
- Derive RFI from Stokes I and V, take the union, and apply the mask to all Stokes parameters.
- Use burst frequency range excluding RFI for peak flux and fluence.
- Write DM/RM and polarization plots per burst under the analysis output directory.

Verify:

- `burst_results.csv` exists and has one row per analyzed burst.
- Report row count, SNR range, DM range, RM range, and rows with NaN or non-significant RM.
- Spot-check at least one output plot when possible.

## Recovery Patterns

- **Auto labels look wrong**: create a bad-file list, relabel with semi-auto/manual, wait for the user to confirm, verify H5 `attrs["bursts"]`, then continue.
- **Duplicate or low-SNR page should be excluded**: write `{"bursts": [], "has_burst": false}` through the UI by pressing `x`; editing only `detections.json` is not enough.
- **Need raw-time peak flux**: rerun calibration with `down_time=1`, then rerun detect and analysis.
- **Energy/polarization analysis says no bursts**: inspect H5 attrs for `bursts`; run or rerun detection.
- **Calibrated H5 has unexpected resolution**: inspect `time_reso_raw`, `time_reso`, `down_time`, `down_freq`, `nchan_raw`, `nchan`.
- **No calibration FITS**: ask for the correct `_0001.fits` or copy it into the cut H5 directory before calibration.
- **Existing detection output skips files**: delete selected entries from `detections.json` or use a new output directory.

## Final Report Template

End each completed observation with:

```text
Processed observation:
  FRB/date/beam:
  script root:
  Python environment:
  start stage:
  cut outputs:
  calibration outputs:
  detection outputs:
  manual review status:
  analysis outputs:
  result CSV:
  counts: raw TOAs, cut H5, calibrated H5, intentionally empty files, accepted bursts, analysis rows
  key warnings:
```

If the workflow paused for user review, mark it as pending user action. Give the manifest path, the exact relabel command, and the message the user should send when done. When the user reports completion, resume by verifying labels, running energy/polarization analysis, and exporting `burst_results.csv`.
