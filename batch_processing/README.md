# AFTER Batch Processing

[简体中文](README.zh-CN.md)

`batch_processing/` contains AFTER's batch entry points for turning observation
directories and confirmed event tables into the standard products used by
detection, review, and physical-property analysis:

```text
confirmed TOAs + raw FAST FITS
  -> batch cut H5
  -> batch flux/polarization calibration
  -> *_cal.h5

legacy burst FITS
  -> current cut-H5 schema
  -> batch calibration
```

All commands below assume that the current directory is the AFTER repository
root. See the [main README](../README.md) for installation and the complete
scientific workflow.

## Choose an entry point

| Task | Entry point | Use when |
|---|---|---|
| Cut a standard `*_Burst.txt` catalog | `batch_cut_burst_data.py` | Every selected event uses the same `segment_length`. |
| Cut long-period or variable-window candidates | `batch_cut_selected_long_period.py` | Each row specifies its own `segment_length`; source/date filtering is useful. |
| Convert legacy burst FITS to the current H5 schema | `fits_to_h5.py` | Old cut FITS already exist and the full observation does not need to be read again. |
| Calibrate cut H5 files in batches | `batch_calibration.py` | Produce Stokes I/Q/U/V, RFI masks, and `*_cal.h5`. |

Inspect the full CLI before running a batch:

```bash
python batch_processing/batch_cut_burst_data.py --help
python batch_processing/batch_cut_selected_long_period.py --help
python batch_processing/fits_to_h5.py --help
python batch_processing/batch_calibration.py --help
```

## 1. Cut a standard event catalog

### Input table

`batch_cut_burst_data.py` reads a whitespace-separated `*_Burst.txt`. A header
is optional; each data row must contain at least seven columns:

```text
base project name date beam dm time
```

| Column | Meaning |
|---|---|
| `base` | First component of the raw-data root, without the leading `/`. |
| `project` | Project directory. |
| `name` | Raw-observation source directory. |
| `date` | Observation date directory, normally `YYYYMMDD`. |
| `beam` | FAST beam number; for example, `1` selects `M01`. |
| `dm` | DM used for cut boundaries and metadata. |
| `time` | Confirmed TOA in seconds from the start of the complete observation. |

The raw directory is resolved as:

```text
/<base>/<project>/<name>/<date>/
```

`time` must therefore come from the observer or an upstream search product. It
is not the local time within one FITS segment.

### Run

```bash
python batch_processing/batch_cut_burst_data.py \
  --burst-txt /path/to/catalogs/FRBXXXX_Burst.txt \
  --output-root /path/to/after_runs/cut/FRBXXXX \
  --save-frb-name FRBXXXX \
  --segment-length 65536 \
  --workers 8
```

`--segment-length` is the number of samples in each cut and defaults to
`65536`. The wrapper groups rows by raw path, date, beam, and DM; copies the
first matching beam FITS needed for calibration; and calls the
`cut_burst_data.py` helpers for every TOA.

Typical output:

```text
<output-root>/
  <date>/
    *_0001.fits
    *.h5
    obs_info.json
```

Pass `--overwrite` explicitly when existing cuts with the same names must be
rebuilt.

## 2. Cut candidates with per-row window lengths

`batch_cut_selected_long_period.py` is intended for candidates whose cut
windows differ. Its minimal table extends the standard seven columns with
`segment_length`:

```text
base project name date beam dm time segment_length [selected_images] [note]
```

`selected_images` and `note` are optional provenance fields and do not change
the cut calculation. The parser also accepts the older extended layout that
contains extra time-range columns.

Start with a dry run:

```bash
python batch_processing/batch_cut_selected_long_period.py \
  --plan-txt /path/to/catalogs/Selected_LongPeriod_Burst.txt \
  --output-root /path/to/after_runs/long_period_cut \
  --workers 8 \
  --dry-run
```

Remove `--dry-run` after checking the paths and groups. Source and date filters
can be repeated:

```bash
python batch_processing/batch_cut_selected_long_period.py \
  --plan-txt /path/to/catalogs/Selected_LongPeriod_Burst.txt \
  --output-root /path/to/after_runs/long_period_cut \
  --only-source FRBXXXX \
  --only-date YYYYMMDD \
  --workers 8
```

Output layout:

```text
<output-root>/<source>/<date>/
  *_0001.fits
  *.h5
  obs_info.json
```

`--overwrite` clears and rebuilds existing cut products for the selected
source/date scope. Verify that scope with `--dry-run` first.

## 3. Convert legacy burst FITS

Use `fits_to_h5.py` when legacy cut FITS already exist but downstream stages
need the current H5 schema. The expected input resembles:

```text
<legacy-root>/
  <FRB>/
    <date>/
      *_0001.fits
      <FRB>-<date>-Mxx-<fits-number>-<start-sample>.fits
```

Place the matching `<FRB>_Burst.txt` catalogs under `--catalog-dir`. They use
the same seven-column format described above. The converter matches catalog
metadata using the source directory, date, beam, FITS number, and start sample.

```bash
python batch_processing/fits_to_h5.py \
  --asd-root /path/to/legacy_burst_data \
  --output-root /path/to/after_runs/cut \
  --catalog-dir /path/to/catalogs \
  --workers 16
```

Limit the conversion to named sources:

```bash
python batch_processing/fits_to_h5.py \
  --asd-root /path/to/legacy_burst_data \
  --output-root /path/to/after_runs/cut \
  --catalog-dir /path/to/catalogs \
  --only FRBXXXX FRBYYYY
```

The converter copies `_0001.fits` calibration files, writes current-schema cut
H5 files, and creates `obs_info.json` in each output date directory. Use
`--overwrite` to replace existing converted products.

## 4. Batch flux and polarization calibration

### Input layout

```text
<root-dir>/
  <FRB>/
    <date>/
      *.h5
      *_0001.fits
```

The source table is whitespace separated:

```text
FRB_name DM RA DEC
```

RA and DEC may use colon notation or another Astropy-readable unit format. DM
is retained as source metadata; each cut H5 should also carry its own DM.

### Run

```bash
python batch_processing/batch_calibration.py \
  --root-dir /path/to/after_runs/cut \
  --cal-root /path/to/after_runs/calibrated \
  --dm-file /path/to/catalogs/h5_calibration_dm_file.txt \
  --cal-npz highcal_20201014_psr_tny.npz \
  --workers 8
```

Limit processing to selected sources:

```bash
python batch_processing/batch_calibration.py \
  --root-dir /path/to/after_runs/cut \
  --cal-root /path/to/after_runs/calibrated \
  --dm-file /path/to/catalogs/h5_calibration_dm_file.txt \
  --cal-npz highcal_20201014_psr_tny.npz \
  --only FRBXXXX FRBYYYY \
  --workers 8
```

Saved-resolution choices:

- omit `--down-time` and `--down-freq` for automatic, plot-friendly values;
- use `--down-time 1` to retain the raw time resolution;
- use `--down-freq 1` to retain the raw frequency channels.

Output layout:

```text
<cal-root>/<FRB>/<date>/
  *_cal.h5
  *.jpg
```

`batch_calibration.py` does not expose an `--overwrite` option. Use a separate
`--cal-root` when comparing calibration or downsampling settings.

## Hand off to detection and analysis

Continue with the repository-root entry points:

```bash
python burst_detect.py \
  --mode auto \
  --cal-dir /path/to/after_runs/calibrated \
  --model-path models/best_model_yolo11n_ema.pth \
  --model-name yolo11n \
  --output-dir /path/to/after_runs/detections

python burst_analysis.py \
  --cal-dir /path/to/after_runs/calibrated \
  --output-dir /path/to/after_runs/analysis
```

Automatic boxes are review proposals, not final scientific measurement
regions. Check or correct the regions written to H5 `attrs["bursts"]` before
running energy and polarization analysis.
