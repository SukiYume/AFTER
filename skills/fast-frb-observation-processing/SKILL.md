---
name: fast-frb-observation-processing
description: "Use when a user asks an agent to install, validate, or run AFTER for FAST FRB post-search processing, including raw FAST FITS with user-provided TOAs, cut H5, calibrated H5, burst label review, detections.json, H5 attrs['bursts'], energy/polarization/DM/RM analysis, results-table export, or observation dashboard summary."
---

# AFTER FAST FRB Observation Processing

AFTER is the AI-assisted FAST Transient End-to-end Reduction workflow for post-search FAST FRB burst processing. Use this skill as the agent operating protocol. Locate the script root before running commands because the skill can be installed separately from the processing scripts.

Normal AFTER sequence:

```text
raw FAST FITS + user-provided TOA list
  -> cut_burst_data.py
  -> cut H5
  -> calibration.py
  -> *_cal.h5
  -> burst_detect.py
  -> human label review/correction
  -> H5 attrs["bursts"] + detections.json
  -> burst_analysis.py
  -> energy/polarization/DM/RM measurements
  -> burst_results.csv + diagnostic plots
  -> optional burst_dashboard.py summary
```

## Operating Mode: One Observation, Minimum Interaction

For a single observation, run the pipeline end to end from the earliest valid stage unless the user asks for a command, review, or pause only. Avoid exploratory detours after the user has given an authoritative path, host, script, or parameter.

Environment neutrality:

- Keep the skill generic. Do not encode a user's local drive, username, SSH host, observatory scratch path, FRB/date, or project-specific output root in the skill.
- Treat hosts, paths, Python environments, calibration files, detector checkpoints, and output roots as per-run inputs or facts to discover from the active checkout.
- Put machine-specific values only in commands, run logs, or user-facing reports for that observation.

Use this interaction policy:

1. Ask for missing blocking inputs in one compact message before the stage that needs them. Do not ask stage-by-stage if the whole chain is requested.
2. Discover cheap local facts yourself: file counts, representative H5 attrs, existing outputs, model file existence, script help, and CSV columns.
3. Ask the user directly for non-discoverable or expensive-risk inputs: source/date/beam, raw FITS path, TOA source, DM, RA/DEC, segment length, remote host/script root, remote output root, local pull target, analysis DM/RM ranges, and overwrite/delete permission.
4. Treat user corrections about host, path, script choice, segment length, or workflow scope as authoritative for the rest of the observation.
5. If the user asks only for a command, give the command only. Do not create helper files, runners, manifests, or dashboards unless they are explicitly requested or required by the current pipeline stage.
6. Prefer direct stage scripts in the verified script root (`cut_burst_data.py`, `calibration.py`, `burst_detect.py`, `burst_analysis.py`, `burst_dashboard.py`) or a user-specified project wrapper for that stage. Use `batch_processing/*` only when the user explicitly asks for batch mode or provides a batch catalog.
7. On remote hosts, use the script root and data/output paths the user named. Do not sync code, replace scripts, or invent wrapper layers unless the user requests it or the named script cannot accept the required inputs.
8. Before destructive cleanup or overwrite, verify the resolved absolute target path and ask unless the user has explicitly requested that cleanup for the same target.
9. Keep human pauses to the necessary gates: TOA review, burst-label review, destructive cleanup/overwrite, and unknown expensive parameters. Continue automatically after the user says the review or edit is finished.
10. Keep a run log in the conversation: command, inputs, output paths, counts, key attrs, warnings, and next action.

One-shot intake template when starting from search plots or raw FITS. Treat every path below as a per-run value: ask the user to provide it, or discover a candidate and ask the user to confirm it before use. For other start stages, ask for the corresponding input/output paths from the section 3 table.

```text
Need before cutting:
  source/date/beam:
  search-plot folder or TOA txt:
  raw FITS directory:
  remote host and AFTER/script root:
  cut output root:
  DM:
  RA/DEC:
  segment length:
  local calibrated-data target:
Optional before analysis:
  DM search half-range / step:
  RM min/max / n_rm:
```

## 1. Locate AFTER

Before any processing command, verify the complete AFTER script root. Use the first working source:

1. Current working directory, if it contains the sentinel files.
2. `DATA_PROCESSING_ROOT`, if set.
3. A path the user gives, such as a cloned AFTER or `data_processing` checkout.

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

For calibration, use the included `highcal_*.npz` file only after verifying it is the intended noise-calibration file, or use the user's supplied calibration file. For detection, use the included trained detector checkpoint only after verifying the path/model name, or use the user's supplied model path and matching model name.

When the skill folder is available before the script root, explain that the installed skill provides the AFTER protocol and ask for the complete AFTER script root. Run processing commands from the verified script root.

## 2. Install or Validate AFTER

When the user asks to install AFTER on another machine:

1. Clone or copy the AFTER repository or script bundle.
2. Install Python dependencies from `requirements.txt`; choose CUDA-specific PyTorch packages for GPU inference when needed.
3. Copy `skills/fast-frb-observation-processing/` into the Codex skills directory.
4. Set `DATA_PROCESSING_ROOT` to the complete AFTER script root when the scripts are available.
5. Run README post-install validation: syntax compile, dependency imports, CLI help, and skill validation.

One-line user-facing install request:

```text
Please install the Codex skill from this repository: copy skills/fast-frb-observation-processing into the Codex skills directory; if this machine has the full AFTER/data_processing script checkout, set DATA_PROCESSING_ROOT to that script root and run the post-install validation.
```

Keep the skill folder limited to `SKILL.md` plus `agents/openai.yaml` during installation.

## 3. Choose the Starting Point

Determine the earliest stage that already has valid inputs:

| Start stage | Required inputs | Continue with |
|---|---|---|
| Raw FAST FITS | FITS directory, source, date, beam, DM, user-provided TOA seconds, output cut directory | Cut, calibrate, detect, review labels, analyze, export table |
| Cut H5 | Cut H5 directory, matching `_0001.fits`, calibration output directory, RA, DEC, calibration file | Calibrate, detect, review labels, analyze, export table |
| Calibrated H5 | `*_cal.h5`, model path, model name, detection output directory | Detect, review labels, analyze, export table |
| Already detected H5 | `*_cal.h5` with H5 attr `bursts` | Verify labels, analyze, export table |

Ask only for missing blocking inputs. Prefer discovering paths, counts, FITS headers, and H5 attrs locally before asking the user.

Operating rules:

- Use TOA seconds supplied by the user or an upstream search product for raw-FITS cutting. Ask for the TOA list when it is missing.
- When the TOA list is derived from search-result figures, prepare a human-reviewable TOA file before cutting and apply the review rules in section 5.
- Start energy/polarization analysis after detection quality is checked and the user accepts auto labels or confirms corrections are finished.
- Treat `batch_processing/*.txt` as local, untracked observation catalogs. Ask for explicit `--burst-txt`, `--dm-file`, `--catalog-dir`, or `--plan-txt` paths when using batch wrappers.
- Collect explicit inputs before relying on machine-local hard-coded defaults. If the user says a named remote script already knows data/output roots and only needs TOA, pass only the required TOA input and verify the resulting paths/counts.
- Do not move to cut data until segment length is known or the user explicitly accepts the script default. Segment length affects scientific review and can force expensive re-cut/re-calibration.

## 4. Preflight

Before each stage, perform only the checks needed to prevent a wrong run:

1. Work from the verified AFTER script root and report absolute paths.
2. Inspect the user-specified Python environment, especially for GPU or plotting workloads.
3. Check input counts and one representative file:
   - Raw FITS: count beam-matched `*Mxx*.fits`; inspect one FITS header for timing/channel metadata.
   - Cut H5: count `.h5` excluding `_cal.h5`; verify matching `_0001.fits`.
   - Calibrated H5: count `*_cal.h5`; inspect one H5 for `data`, `freq`, `rfi_mask`, `gain`, `gain_err`, and attrs.
4. Check output directories and existing products.
5. Check whether a previous `detections.json` exists.
6. Preserve existing outputs and request overwrite confirmation when needed. If cleanup is requested, resolve the target path first and keep it within the intended observation output tree.

Keep a run log with inputs, commands, counts, output paths, skipped files, warnings, and user-review status.

## 5. Cut Raw FITS

Run this stage only when starting from raw FAST FITS.

Required inputs: raw FITS directory, FRB/source name, observation date, beam, DM, user-provided TOA seconds, output cut directory, optional segment length, optional worker count.

TOA preparation from search-result figures:

1. Extract candidate TOAs from figure metadata or filenames and keep only two decimal places in the cut-ready TOA text file.
2. Also write a review file that maps each rounded TOA back to the source figure.
3. Ask the user to visually check whether each TOA is centered on the true signal peak. Search plots can mark the local max away from the real peak when the burst is weak.
4. Let the user adjust offset TOAs, remove candidates that are too weak or invisible, and confirm the edited list before cutting.
5. After the user edits timing offsets, sort the TOAs and check adjacent spacings. If two TOAs are separated by less than `0.180` seconds, report them as a merge candidate.
6. For confirmed close pairs, replace the pair with the midpoint TOA so one cut H5 contains both signals. Record the original pair and midpoint in the run log.
7. If no close pairs remain, say that explicitly and continue. Do not re-open the figures unless the user reports an error in the TOA list.

Recommended implementation:

1. Import `read_obs_info`, `calc_dispersion_shift`, `cut_one_burst`, and `save_obs_json` from `cut_burst_data`.
2. Build a sorted FITS list using beam pattern `M{beam:02d}`.
3. Copy the first matching beam FITS into the cut output directory so calibration can find `_0001.fits`.
4. Sort, deduplicate, range-check, and apply any user-confirmed close-pair midpoint merges to the supplied TOAs.
5. Compute dispersion shifts with the requested DM.
6. Run `cut_one_burst` for each valid TOA, using a process pool for multiple bursts.
7. Run `save_obs_json`.

Remote/direct-script rule:

- If the user points to a remote cut wrapper or `cut_burst_data.py` whose data path/output root are already configured, inspect its accepted arguments once, then run it directly with the final TOA file and required parameters. Do not generate batch catalogs or wrapper scripts unless the direct script cannot express the requested segment length or output path.

Verify:

- Cut H5 count matches the in-range valid TOA count.
- Filenames follow `{frb}-{date}-M{beam:02d}-{fits_number:04d}-{start_sample:09d}.h5`.
- One sample H5 has `data`, `freq`, `toa_sec`, `time_reso`, `obs_start_mjd`, and `dm`.
- `obs_info.json` exists.

Common blockers: beam-matched FITS needed, calibration/noise FITS needed, TOA outside observation, FITS missing `DATA` or `DAT_FREQ`.

## 6. Calibrate

Run this stage when starting from cut H5 or after successful cutting.

Required inputs: cut H5 directory, calibration output directory, RA, DEC, beam inferred from filenames, `highcal_*.npz` or alternative calibration file, optional `down_time`, optional `down_freq`, optional RFI strategy.

Recommended implementation:

1. Import `find_cal_fits`, `fold_noise_cal`, `load_t_cal`, and `process_one_burst` from `calibration`.
2. Group cut H5 files by beam from the `Mxx` filename token.
3. For each beam, find matching `_0001.fits`, fold `noise_cal`, load `t_cal`, and process each H5.
4. Use `rfi_fft=True` for standard calibration RFI, or use the user's requested calibration-time RFI strategy.

Downsample policy:

- `down_time=None`, `down_freq=None`: save at automatic plot-friendly resolution.
- `down_time=1`: preserve raw time resolution for peak-flux comparisons.
- `down_freq=1`: preserve raw frequency channels for detailed RFI or spectra.
- In calibrated H5, `time_reso` is the effective saved resolution.

Remote/direct-script rule:

- If the user points to a remote calibration wrapper or `calibration.py`, run that direct calibration path after cut verification. Do not switch to `batch_processing/batch_calibration.py` unless the user asked for batch mode, the direct script is absent, or the direct script cannot process the requested directory.
- Pull back calibrated products after calibration finishes. By default pull `*_cal.h5` and quick-look images into the local calibrated-data target; pull raw cut H5 only when the user asks.

Verify:

- Each input H5 has a matching `*_cal.h5`.
- Quick-look `.jpg` plots exist when plotting is enabled.
- One calibrated H5 has `data`, `freq`, `rfi_mask`, `rfi_channel`, `gain`, and `gain_err`.
- Check attrs `down_time`, `down_freq`, `time_reso_raw`, `time_reso`, `nchan_raw`, `nchan`, `dm`, `beam`, `ra`, and `dec`.

## 7. Detect and Review Burst Labels

Run this stage when starting from calibrated H5 or after successful calibration.

Auto detection example:

```bash
python burst_detect.py \
  --mode auto \
  --cal-dir /path/to/cal_date \
  --model-path /path/to/detector_checkpoint.pth \
  --model-name MODEL_NAME \
  --output-dir /path/to/detections_auto
```

Detection behavior:

- Recursively finds `*_cal.h5` below `--cal-dir`.
- Writes H5 `attrs["bursts"]`.
- Writes `detections.json` and `plots/*_det.png`.
- Skips existing entries in `detections.json`, including zero-burst entries.
- Recomputes detection-stage RFI from non-burst noise and writes `burst_rfi_mask` / `burst_rfi_channel`.

After auto detection:

1. Inspect detection plots and metadata before analysis.
2. Build or update `detection_review_manifest.txt` in the detection output directory.
3. Flag files with no detections, low confidence, overlapping or split boxes, edge-clipped boxes, implausibly wide or narrow boxes, boxes far from the profile peak, or visually missed/merged bursts.
4. Tell the user which files need review, where plots are, how many labels look acceptable, and how to relabel bad files.
5. Pause until the user confirms labels are acceptable or corrections are finished.

If the user asks for a semi-auto/manual command, provide the exact `python burst_detect.py ...` command and stop. Do not create a runner file.

Use semi-auto for bad auto labels:

1. Remove only bad filenames from that output directory's `detections.json`.
2. Rerun `burst_detect.py --mode semi-auto` with the same `--cal-dir` and `--output-dir`.
3. The script skips files still listed in `detections.json` and opens only removed files.

Use manual mode when the model is misleading:

```bash
python burst_detect.py \
  --mode manual \
  --cal-dir /path/to/cal_date \
  --output-dir /path/to/detections_manual
```

Interactive controls:

- Enter accepts current boxes.
- Left-drag draws boxes.
- Right-click undoes the latest manual box.
- `x` writes an intentionally empty `bursts` list for this file and moves on.
- `q` or Esc quits.

When the user says labels are fixed:

- Verify each intended H5 has `attrs["bursts"]`.
- Count total files, intentionally empty files, and accepted burst regions.
- Confirm reviewed files now have plausible non-empty or intentionally empty regions.
- Confirm current-label files include `burst_rfi_mask`, `burst_rfi_channel`, and `burst_rfi_method` when written by current `burst_detect.py`.
- Keep or report the final `detections.json` path.

Remember: `attrs["bursts"]` is the source of truth for analysis. Update H5 attrs when a label should be removed or intentionally marked empty.

## 8. Analyze Energy and Polarization

Run this stage after burst labels are accepted.

Example:

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

Treat these values as examples. Confirm DM/RM ranges against source knowledge before long runs.

Analysis rules:

- `burst_analysis.py` processes `*_cal.h5` files directly inside one directory. Loop over date directories for multiple dates.
- `--dm-range` is centered on the cut DM stored in each H5.
- `--rm-min`, `--rm-max`, `--n-rm`, and `--n-boot` control RM search and uncertainty work.
- Pass `--target-down-time` and `--target-down-freq` only for coarser analysis than the saved calibrated H5. Target factors must be integer multiples of saved `down_time/down_freq`.
- Pass `--rfi-fft` when FFT RFI is requested in analysis.
- Rebuild the noise mask from accepted burst labels, subtract per-channel baseline, derive RFI from Stokes I and V, and apply the union mask to all Stokes parameters.
- Use the accepted burst frequency range excluding RFI for peak flux and fluence.
- RM reliability is decided from the RM search significance/error outputs. If no reliable RM is found, report that the RM search result is reliable as a non-detection and do not interpret linear/circular polarization fractions as trustworthy physical measurements.

Verify:

- `burst_results.csv` exists and has one row per analyzed burst.
- Report row count, SNR range, DM range, RM range, and rows with NaN or non-significant RM.
- Spot-check at least one DM/RM/polarization plot when possible.

## 9. Build Observation Dashboard or Summary

Run this stage after `burst_results.csv` exists when the user asks for a summary, dashboard, visualization panel, or observation overview.

Preferred command shape:

```bash
python burst_dashboard.py \
  --csv /path/to/analysis/burst_results.csv \
  --output /path/to/analysis/burst_dashboard.html \
  --source FRBNAME \
  --date YYYYMMDD \
  --reference-dm DM \
  --rm-significance-threshold 5
```

Dashboard rules:

- Use `burst_results.csv` as the source. Do not re-run analysis just to build the dashboard.
- Lead with observation identity, burst count, time span, SNR range, DM range, fluence, width, and reliable-RM count.
- Plot robust quantities: TOA/SNR timeline, DM results, fluence-width-SNR relation, frequency coverage, RM significance diagnostics, and property distributions.
- If no reliable RM exists, show RM as a non-detection diagnostic and keep `linear_frac` / `circular_frac` only in the detail table with a visible caveat.
- Verify the dashboard output exists, embeds charts, has one detail row per analyzed burst, and opens locally without obvious layout overflow.

## 10. Data Contracts

Raw FAST directory:

- Contains many FITS files for one observation date.
- Beam is identified by `Mxx` in filenames.
- A calibration/noise FITS for the beam should be in the same directory and usually ends with `_0001.fits`.
- Burst TOA uses seconds from observation start.

Cut H5:

```text
data: (nsamp, npol, nchan)
freq: (nchan,), MHz
attrs: start_sample, file_mjd, toa_sec, time_reso, npol, nchan,
       segment_length, obs_start_mjd, dm
```

Calibrated H5:

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

## 11. Recovery Patterns

- **Auto labels look wrong**: create a bad-file list, relabel with semi-auto/manual, wait for user confirmation, verify H5 `attrs["bursts"]`, then continue.
- **Duplicate or low-SNR page should be excluded**: write `{"bursts": [], "has_burst": false}` through the UI by pressing `x`; this updates the H5 analysis source of truth.
- **Need raw-time peak flux**: rerun calibration with `down_time=1`, then rerun detection and analysis.
- **Analysis says no bursts**: inspect H5 attrs for `bursts`; run or rerun detection.
- **Calibrated H5 has unexpected resolution**: inspect `time_reso_raw`, `time_reso`, `down_time`, `down_freq`, `nchan_raw`, and `nchan`.
- **Calibration FITS needed**: ask for the correct `_0001.fits` or copy it into the cut H5 directory before calibration.
- **Existing detection output skips files**: delete selected entries from `detections.json` or use a new output directory.
- **Cut length is wrong after calibration**: stop, clean only the affected cut/calibration outputs after resolved-path verification, re-cut with the confirmed segment length, recalibrate, pull back, and re-check one H5 `data.shape` plus `segment_length`/`time_reso` attrs.
- **User narrows scope**: if the user says to stop, only provide a command, or avoid extra work, obey the narrowed scope and do not continue the full chain.
- **RM is not reliable**: treat polarization products as unreliable for interpretation. Keep them for traceability only.

## 12. Final Report Template

End each completed observation with:

```text
Processed observation:
  FRB/date/beam:
  AFTER script root:
  Python environment:
  start stage:
  cut outputs:
  calibration outputs:
  detection outputs:
  manual review status:
  analysis outputs:
  result CSV:
  dashboard:
  counts: raw TOAs, cut H5, calibrated H5, intentionally empty files, accepted bursts, analysis rows
  key warnings:
```

If the workflow paused for user review, mark it as pending user action. Give the manifest path, the exact relabel command, and the message the user should send when done. When the user reports completion, resume by verifying labels, running energy/polarization analysis, and exporting `burst_results.csv`.
