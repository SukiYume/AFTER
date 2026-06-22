"""Convert legacy cut FITS burst files to the current H5 cut format.

The expected remote layout is::

    /path/to/after_data/
    |-- FRB20251229A/
    |   |-- 20260106/
    |   |   |-- FRB20251229A_tracking-M01_0001.fits
    |   |   `-- FRB20251229A-20260106-M01-0075-009823391.fits
    |   `-- ...
    `-- H5_Cut/

The Burst.txt catalogs are read from --catalog-dir. The default is the
directory containing this script, so the curated txt files can live beside the
batch scripts.

By default the script scans FRB20* directories under asd_root and writes:

    /path/to/after_data/H5_Cut/<FRB>/<date>/*.h5

Only legacy burst FITS names are converted. Calibration FITS files ending in
``_0001.fits`` are copied unchanged so the H5 directory mirrors the direct H5
cut output layout.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits


DEFAULT_ASD_ROOT = "/path/to/after_data"
DEFAULT_OUTPUT_ROOT = "/path/to/after_data/H5_Cut"
DEFAULT_CATALOG_DIR = Path(__file__).resolve().parent


def parse_old_filename(filename: str):
    """Parse legacy burst FITS metadata from its filename."""
    m = re.match(r"^(.+)-(\d{8})-M(\d{2})-(\d{4})-(\d{9})\.fits$", filename)
    if m:
        return {
            "frb_name": m.group(1),
            "date": m.group(2),
            "beam": int(m.group(3)),
            "fits_number": int(m.group(4)),
            "start_sample": int(m.group(5)),
        }

    m = re.match(r"^(.+)-(\d{8})-(\d{4})-(\d{9})\.fits$", filename)
    if m:
        return {
            "frb_name": m.group(1),
            "date": m.group(2),
            "beam": 1,
            "fits_number": int(m.group(3)),
            "start_sample": int(m.group(4)),
        }

    return None


def is_cal_fits(filename: str) -> bool:
    return filename.endswith("_0001.fits")


def is_date_dir(path: Path) -> bool:
    return path.is_dir() and re.match(r"^\d{8}$", path.name) is not None


def is_source_frb_dir(path: Path, prefix: str) -> bool:
    if not path.is_dir() or not path.name.startswith(prefix):
        return False
    if path.name.endswith("_H5") or path.name == "H5_Cut":
        return False
    return True


def parse_burst_catalog(catalog_dir: Path, prefix: str):
    """Read FRB*_Burst.txt files as whitespace-delimited tables.

    Returns
    -------
    catalog : dict[str, list[dict]]
        Rows keyed by the save FRB name derived from the txt filename.
    """
    catalog = defaultdict(list)
    for txt_path in sorted(catalog_dir.glob(f"{prefix}*_Burst.txt")):
        save_frb = txt_path.name[: -len("_Burst.txt")]
        with txt_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                parts = line.split()
                if not parts or parts[0].lower() == "base":
                    continue
                if len(parts) < 7:
                    print(f"[WARN] skip malformed row {txt_path.name}:{line_no}: {line.rstrip()}")
                    continue
                base, project, raw_name, date, beam, dm, toa = parts[:7]
                try:
                    catalog[save_frb].append(
                        {
                            "base": base,
                            "project": project,
                            "name": raw_name,
                            "date": date,
                            "beam": int(beam),
                            "dm": float(dm),
                            "time": float(toa),
                        }
                    )
                except ValueError:
                    print(f"[WARN] skip unparsable row {txt_path.name}:{line_no}: {line.rstrip()}")
    return dict(catalog)


def read_obs_info(cal_fits_path: Path):
    """Read observation metadata from one calibration FITS file."""
    with fits.open(cal_fits_path) as hdul:
        h0 = hdul[0].header
        h1 = hdul[1].header
        return {
            "time_reso": h1["TBIN"],
            "nsblk": h1["NSBLK"],
            "naxis2": h1["NAXIS2"],
            "file_nsamp": h1["NAXIS2"] * h1["NSBLK"],
            "npol": h1["NPOL"],
            "nchan": h1["NCHAN"],
            "freq": hdul[1].data["DAT_FREQ"][0, :].astype(np.float64),
            "start_mjd": h0["STT_IMJD"]
            + (h0.get("STT_SMJD", 0) + h0.get("STT_OFFS", 0)) / 86400.0,
        }


def load_fits_data(filepath: Path):
    """Read legacy cut burst FITS data as (nsamp, npol, nchan)."""
    with fits.open(filepath) as hdul:
        h1 = hdul[1].header
        raw = hdul[1].data["DATA"]
        return raw.reshape(h1["NAXIS2"] * h1["NSBLK"], h1["NPOL"], h1["NCHAN"])


def copy_cal_files(date_path: Path, output_path: Path, overwrite: bool):
    """Copy calibration FITS files and return the first calibration path."""
    first_cal = None
    output_path.mkdir(parents=True, exist_ok=True)
    for path in sorted(date_path.iterdir()):
        if not path.is_file() or not is_cal_fits(path.name):
            continue
        dst = output_path / path.name
        if overwrite or not dst.exists():
            shutil.copy2(path, dst)
        if first_cal is None:
            first_cal = path
    return first_cal


def match_catalog_row(name_info, nsamp: int, info: dict, rows: list[dict]):
    """Find the Burst.txt row that produced this legacy FITS file."""
    if not rows:
        return None

    time_reso = info["time_reso"]
    file_nsamp = info["file_nsamp"]
    tolerance = max(2, int(0.06 / time_reso))
    candidates = []

    for row in rows:
        if row["beam"] != name_info["beam"]:
            continue
        sample = int(row["time"] / time_reso)
        expected_start = sample - nsamp // 2
        expected_fits_number = sample // file_nsamp + 1
        start_delta = abs(expected_start - name_info["start_sample"])
        fits_delta = abs(expected_fits_number - name_info["fits_number"])
        if start_delta <= tolerance:
            candidates.append((fits_delta, start_delta, row))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def save_to_h5(save_path: Path, filename: str, data: np.ndarray, meta: dict):
    """Write H5 using the same dataset/attribute layout as cut_burst_data.py."""
    save_path.mkdir(parents=True, exist_ok=True)
    filepath = save_path / filename
    with h5py.File(filepath, "w") as f:
        f.create_dataset("data", data=data, compression="gzip", compression_opts=4)
        f.create_dataset("freq", data=meta["freq"])
        f.attrs["start_sample"] = meta["start_sample"]
        f.attrs["file_mjd"] = meta["file_mjd"]
        f.attrs["toa_sec"] = meta["toa_sec"]
        f.attrs["time_reso"] = meta["time_reso"]
        f.attrs["npol"] = meta["npol"]
        f.attrs["nchan"] = meta["nchan"]
        f.attrs["segment_length"] = meta["segment_length"]
        f.attrs["obs_start_mjd"] = meta["obs_start_mjd"]
        f.attrs["dm"] = meta["dm"]
    return filepath


def convert_one_fits(args):
    fits_path, output_dir, rows, info, overwrite = args
    fits_path = Path(fits_path)
    output_dir = Path(output_dir)

    name_info = parse_old_filename(fits_path.name)
    if name_info is None:
        return "skip"

    h5_name = (
        f"{name_info['frb_name']}-{name_info['date']}-"
        f"M{name_info['beam']:02d}-{name_info['fits_number']:04d}-"
        f"{name_info['start_sample']:09d}.h5"
    )
    h5_path = output_dir / h5_name
    if h5_path.exists() and not overwrite:
        return "exists"

    data = load_fits_data(fits_path)
    nsamp = int(data.shape[0])
    row = match_catalog_row(name_info, nsamp, info, rows)
    if row is None:
        toa_sec = (name_info["start_sample"] + nsamp // 2) * info["time_reso"]
        dm = rows[0]["dm"] if rows else np.nan
        matched = False
    else:
        toa_sec = row["time"]
        dm = row["dm"]
        matched = True

    start_sample = name_info["start_sample"]
    meta = {
        "start_sample": start_sample,
        "file_mjd": info["start_mjd"] + start_sample * info["time_reso"] / 86400.0,
        "toa_sec": toa_sec,
        "time_reso": info["time_reso"],
        "npol": info["npol"],
        "nchan": info["nchan"],
        "segment_length": nsamp,
        "obs_start_mjd": info["start_mjd"],
        "dm": dm,
        "freq": info["freq"],
    }

    save_to_h5(output_dir, h5_name, data, meta)
    return "ok" if matched else "ok_no_catalog_match"


def save_obs_json(output_dir: Path):
    h5_files = sorted(
        p for p in output_dir.iterdir()
        if p.suffix == ".h5" and not p.name.endswith("_cal.h5")
    )
    if not h5_files:
        return

    bursts = []
    dm_values = []
    segment_length = None
    info_values = None

    for h5_path in h5_files:
        with h5py.File(h5_path, "r") as f:
            if info_values is None:
                info_values = {
                    "obs_start_mjd": float(f.attrs["obs_start_mjd"]),
                    "nchan": int(f.attrs["nchan"]),
                    "time_reso": float(f.attrs["time_reso"]),
                    "npol": int(f.attrs["npol"]),
                }
                segment_length = int(f.attrs["segment_length"])
            bursts.append({"file": h5_path.name, "toa_sec": round(float(f.attrs["toa_sec"]), 4)})
            dm_values.append(float(f.attrs["dm"]))

    unique_dm = sorted(set(round(dm, 8) for dm in dm_values))
    obs_info = {
        **info_values,
        "dm": unique_dm[0] if len(unique_dm) == 1 else unique_dm,
        "segment_length": segment_length,
        "bursts": bursts,
    }

    json_path = output_dir / "obs_info.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(obs_info, f, indent=2, ensure_ascii=False)


def collect_tasks(asd_root: Path, output_root: Path, catalog_dir: Path,
                  prefix: str, only: set[str], overwrite: bool):
    catalog = parse_burst_catalog(catalog_dir, prefix)
    if not catalog:
        print(f"No {prefix}*_Burst.txt catalog rows found under {catalog_dir}")
        return [], []

    tasks = []
    date_output_dirs = set()

    frb_dirs = [
        path
        for path in sorted(asd_root.iterdir())
        if is_source_frb_dir(path, prefix) and (not only or path.name in only)
    ]
    print(f"Found {len(frb_dirs)} {prefix}* directories")

    for frb_path in frb_dirs:
        rows = catalog.get(frb_path.name, [])
        rows_by_date = defaultdict(list)
        for row in rows:
            rows_by_date[row["date"]].append(row)
        if not rows:
            print(f"[WARN] {frb_path.name}: no catalog rows, metadata will fall back to filename")

        frb_count = 0
        for date_path in sorted(path for path in frb_path.iterdir() if is_date_dir(path)):
            output_path = output_root / frb_path.name / date_path.name
            cal_path = copy_cal_files(date_path, output_path, overwrite)
            if cal_path is None:
                print(f"[WARN] {frb_path.name}/{date_path.name}: no *_0001.fits calibration file")
                continue

            info = read_obs_info(cal_path)
            burst_files = [
                path
                for path in sorted(date_path.iterdir())
                if path.is_file() and path.suffix == ".fits" and parse_old_filename(path.name) is not None
            ]
            for burst_path in burst_files:
                tasks.append((str(burst_path), str(output_path), rows_by_date[date_path.name], info, overwrite))
            frb_count += len(burst_files)
            if burst_files:
                date_output_dirs.add(output_path)

        print(f"{frb_path.name}: {frb_count} burst FITS")

    return tasks, sorted(date_output_dirs)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asd-root", default=DEFAULT_ASD_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--catalog-dir", default=str(DEFAULT_CATALOG_DIR))
    parser.add_argument("--frb-prefix", default="FRB20")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--only", nargs="*", default=None, help="Optional FRB directory names to process")
    return parser.parse_args()


def main():
    args = parse_args()
    asd_root = Path(args.asd_root)
    output_root = Path(args.output_root)
    catalog_dir = Path(args.catalog_dir)
    only = set(args.only or [])

    tasks, date_output_dirs = collect_tasks(
        asd_root, output_root, catalog_dir, args.frb_prefix, only, args.overwrite,
    )
    if not tasks:
        print("No burst FITS files to convert")
        return

    print(f"Converting {len(tasks)} burst FITS with {args.workers} workers")
    if args.workers > 1 and len(tasks) > 1:
        with Pool(args.workers) as pool:
            results = pool.map(convert_one_fits, tasks)
    else:
        results = [convert_one_fits(task) for task in tasks]

    summary = {key: results.count(key) for key in sorted(set(results))}
    print(f"Conversion summary: {summary}")

    print(f"Writing obs_info.json for {len(date_output_dirs)} date directories")
    for output_dir in date_output_dirs:
        save_obs_json(output_dir)

    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
