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

rebuild catalog + raw FAST FITS
  -> rebuilt legacy burst FITS
  -> current cut-H5 schema
```

All commands below assume that the current directory is the AFTER repository
root. See the [main README](../README.md) for installation and the complete
scientific workflow.

## Choose an entry point

| Task | Entry point | Use when |
|---|---|---|
| Cut a standard `*_Burst.txt` catalog | `batch_cut_burst_data.py` | Every selected event uses the same `segment_length`. |
| Cut long-period or variable-window candidates | `batch_cut_selected_long_period.py` | Each row specifies its own `segment_length`; source/date filtering is useful. |
| Rebuild legacy burst FITS | `cut_burst_fits.py` | The old FITS were removed but a checked rebuild catalog and the raw observation remain available. |
| Convert legacy burst FITS to the current H5 schema | `fits_to_h5.py` | Old cut FITS already exist and the full observation does not need to be read again. |
| Calibrate cut H5 files in batches | `batch_calibration.py` | Produce Stokes I/Q/U/V, RFI masks, and `*_cal.h5`. |

Inspect the full CLI before running a batch:

```bash
python batch_processing/batch_cut_burst_data.py --help
python batch_processing/batch_cut_selected_long_period.py --help
python batch_processing/cut_burst_fits.py --help
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

Copy [`Burst.example.txt`](Burst.example.txt) as a header-only starting point.

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
`after.cut_burst_data` helpers for every TOA.

Typical output:

```text
<output-root>/
  <date>/
    *_0001.fits
    *.h5
    obs_info.json
```

Pass `--overwrite` explicitly when existing cuts with the same names must be
rebuilt. `obs_info.json` is derived from every H5 in the date directory:
uniform values remain scalars, while mixed beam, DM, and segment-length values
become lists and are also recorded per burst.

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

## 3. Rebuild or convert legacy burst FITS

If old cut FITS were removed after their exact reconstruction metadata was
saved, `cut_burst_fits.py` can recreate them from the raw observation. Start
with `--dry-run`; this validates the catalog, frozen raw-file order, and
availability of uncompressed FITS without writing burst data:

```bash
python batch_processing/cut_burst_fits.py \
  --burst-txt /path/to/catalogs/FRBXXXX_legacy_fits_rebuild.txt \
  --raw-root /path/to/raw-root \
  --output-root /path/to/rebuilt-fits \
  --dry-run
```

Remove `--dry-run` only after checking every reported directory and group.
Rows whose `rebuild_status` is not `ready*` stop the run by default; use
`--skip-blocked` only after reviewing why those rows cannot be reconstructed
exactly. The script writes only dedispersed burst FITS and does no calibration.

Use `fits_to_h5.py` when legacy cut FITS already exist but downstream stages
need the current H5 schema. The expected input resembles:

```text
<legacy-root>/
  <FRB>/
    <date>/
      *_0001.fits
      <FRB>-<date>-Mxx-<fits-number>-<start-sample>.fits
```

Place the matching `<FRB>_Burst.txt` catalogs under `--catalog-dir`. The normal
input uses the same seven-column format described above. The legacy six-column
`name beam project dm date time` format is also accepted. Date directory and
filename tokens may be either `YYYYMMDD` or split forms such as `YYYYMMDD_1`.
The converter matches catalog metadata using the source directory, date, beam,
FITS number, and start sample.

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
`--dry-run` to inspect the scope without creating directories or files, and use
`--overwrite` to replace existing converted products. `--asd-root` is required
so the converter never guesses which legacy tree to scan.

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
Every beam group uses its matching `Mxx..._0001.fits` in the same date
directory. For split same-day directories such as `YYYYMMDD_1` and
`YYYYMMDD_2`, the batch first validates the current segment's calibration
file. If its noise-diode diagnostic fails, the batch may fall back to a valid
same-beam calibration file from another segment of that same date; it never
falls back across dates. Processing stops when no same-day candidate is
available. The selected file is recorded in each H5's `calibration_fits` attr.

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
- use `--time-crop-samples 512` to center-crop 512 raw time samples before
  downsampling; unless explicitly overridden, time downsampling becomes 1 and
  the JPG is redrawn directly at the saved resolution.
- use `--target-time-reso-ms 0.786432 --output-time-samples 512` to calibrate
  the complete input first, choose a separate integer time-downsampling factor
  for each raw resolution, and only then center-crop the downsampled result to
  512 samples. `--target-time-reso-ms` cannot be combined with `--down-time`,
  and `--output-time-samples` cannot be combined with `--time-crop-samples`.

FFT RFI detection is enabled by default. Pass `--no-rfi-fft` to use entropy
mode instead.

Output layout:

```text
<cal-root>/<FRB>/<date>/
  *_cal.h5
  *.jpg
```

Each H5 records `calibration_beam`, `calibration_fits`, and `calibration_npz`
attrs. Use a separate `--cal-root` when comparing calibration or downsampling
settings.

## Hand off to detection and analysis

Continue with **Detect and review burst regions** and **Analyze physical
properties** in the [main README](../README.md#quick-start). Both entry points
recursively discover `*_cal.h5`, so `--cal-dir` can be the batch `<cal-root>`.
Review or correct the proposed regions written to H5 `attrs["bursts"]` before
running energy and polarization analysis.
