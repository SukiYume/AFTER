# fmt: off

"""Convert legacy cut FITS burst files to the current H5 cut format.

An input legacy tree can be arranged as::

    /path/to/legacy_fits/
    `-- FRB20251229A/20260106/
        |-- FRB20251229A_tracking-M01_0001.fits
        `-- FRB20251229A-20260106-M01-0075-009823391.fits

The Burst.txt catalogs are read from --catalog-dir. The default is the
``batch_processing`` directory containing this script.

The script scans matching FRB directories below ``--asd-root`` and writes:

    data_processing/H5_Cut/<FRB>/<date>/*.h5

Only legacy burst FITS names are converted. Calibration FITS files ending in
``_0001.fits`` are copied unchanged so the H5 directory mirrors the direct H5
cut output layout.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPT_DIR  = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from after.obs_metadata import write_obs_info_json  # noqa: E402

DEFAULT_OUTPUT_ROOT = str(PROJECT_DIR / "H5_Cut")
DEFAULT_CATALOG_DIR = SCRIPT_DIR


def parse_old_filename(filename: str):
    """Parse legacy burst FITS metadata from its filename."""
    date_pattern  = r"\d{8}(?:_\d+)?"
    # 观测起点附近的切片允许 start_sample < 0；AFTER 的文件名约定用
    # ``n`` 代替负号，例如 n000000123 表示 -123。
    start_pattern = r"n?\d{9}"
    m = re.match(
        rf"^(.+)-({date_pattern})-M(\d{{2}})-(\d{{4}})-({start_pattern})\.fits$",
        filename,
    )
    if m:
        start_token = m.group(5)
        return {
            "frb_name": m.group(1),
            "date": m.group(2),
            "beam": int(m.group(3)),
            "fits_number": int(m.group(4)),
            "start_sample": (
                -int(start_token[1:])
                if start_token.startswith("n")
                else int(start_token)
            ),
        }

    m = re.match(
        rf"^(.+)-({date_pattern})-(\d{{4}})-({start_pattern})\.fits$",
        filename,
    )
    if m:
        start_token = m.group(4)
        return {
            "frb_name": m.group(1),
            "date": m.group(2),
            "beam": 1,
            "fits_number": int(m.group(3)),
            "start_sample": (
                -int(start_token[1:])
                if start_token.startswith("n")
                else int(start_token)
            ),
        }

    return None


def is_cal_fits(filename: str) -> bool:
    return filename.endswith("_0001.fits")


def is_date_dir(path: Path) -> bool:
    """识别八位日期目录以及同一天拆分出的 ``_1``、``_2`` 等目录。"""

    return path.is_dir() and re.fullmatch(r"\d{8}(?:_\d+)?", path.name) is not None


def is_source_frb_dir(path: Path, prefix: str) -> bool:
    return (
        path.is_dir()
        and path.name.startswith(prefix)
        and not path.name.endswith("_H5")
        and path.name != "H5_Cut"
    )


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
                if parts[0].lower() in {"base", "name"}:
                    continue
                if len(parts) >= 7:
                    base, project, raw_name, date, beam, dm, toa = parts[:7]
                elif len(parts) >= 6:
                    # 六列清单按 name beam project dm date time 解析。
                    raw_name, beam, project, dm, date, toa = parts[:6]
                    base = ""
                else:
                    print(
                        f"[WARN] skip malformed row {txt_path.name}:{line_no}: {line.rstrip()}"
                    )
                    continue
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
                    print(
                        f"[WARN] skip unparsable row {txt_path.name}:{line_no}: {line.rstrip()}"
                    )
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
        h1  = hdul[1].header
        raw = hdul[1].data["DATA"]
        return raw.reshape(h1["NAXIS2"] * h1["NSBLK"], h1["NPOL"], h1["NCHAN"])


def copy_cal_files(
    date_path: Path,
    output_path: Path,
    overwrite: bool,
    dry_run: bool = False,
):
    """Copy calibration FITS files and return the first calibration path."""
    first_cal = None
    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)
    for path in sorted(date_path.iterdir()):
        if not path.is_file() or not is_cal_fits(path.name):
            continue
        dst = output_path / path.name
        if not dry_run and (overwrite or not dst.exists()):
            temp_path = dst.with_name(f"{dst.name}.tmp")
            temp_path.unlink(missing_ok=True)
            try:
                shutil.copy2(path, temp_path)
                os.replace(temp_path, dst)
            finally:
                temp_path.unlink(missing_ok=True)
        if first_cal is None:
            first_cal = path
    return first_cal


def match_catalog_row(name_info, nsamp: int, info: dict, rows: list[dict]):
    """Find the Burst.txt row that produced this legacy FITS file."""
    if not rows:
        return None

    time_reso  = info["time_reso"]
    file_nsamp = info["file_nsamp"]
    tolerance  = max(2, int(0.06 / time_reso))
    candidates = []

    for row in rows:
        if row["beam"] != name_info["beam"]:
            continue
        sample               = int(row["time"] / time_reso)
        expected_start       = sample - nsamp // 2
        expected_fits_number = sample // file_nsamp + 1
        start_delta          = abs(expected_start - name_info["start_sample"])
        fits_delta           = abs(expected_fits_number - name_info["fits_number"])
        if start_delta <= tolerance:
            candidates.append((fits_delta, start_delta, row))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def save_to_h5(save_path: Path, filename: str, data: np.ndarray, meta: dict):
    """Write H5 using the same dataset/attribute layout as :mod:`after.cut_burst_data`."""
    save_path.mkdir(parents=True, exist_ok=True)
    filepath  = save_path / filename
    temp_path = filepath.with_name(f"{filepath.name}.tmp")
    temp_path.unlink(missing_ok=True)
    try:
        with h5py.File(temp_path, "w") as f:
            f.create_dataset("data", data=data, compression="gzip", compression_opts=4)
            f.create_dataset("freq", data=meta["freq"])
            f.attrs["start_sample"]   = meta["start_sample"]
            f.attrs["file_mjd"]       = meta["file_mjd"]
            f.attrs["toa_sec"]        = meta["toa_sec"]
            f.attrs["time_reso"]      = meta["time_reso"]
            f.attrs["npol"]           = meta["npol"]
            f.attrs["nchan"]          = meta["nchan"]
            f.attrs["segment_length"] = meta["segment_length"]
            f.attrs["obs_start_mjd"]  = meta["obs_start_mjd"]
            f.attrs["beam"]           = meta["beam"]
            f.attrs["dm"]             = meta["dm"]
        os.replace(temp_path, filepath)
    finally:
        temp_path.unlink(missing_ok=True)
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
        + (
            f"n{abs(name_info['start_sample']):09d}.h5"
            if name_info["start_sample"] < 0
            else f"{name_info['start_sample']:09d}.h5"
        )
    )
    h5_path = output_dir / h5_name
    if h5_path.exists() and not overwrite:
        return "exists"

    data  = load_fits_data(fits_path)
    nsamp = int(data.shape[0])
    row   = match_catalog_row(name_info, nsamp, info, rows)
    if row is None:
        toa_sec = (name_info["start_sample"] + nsamp // 2) * info["time_reso"]
        dm      = rows[0]["dm"] if rows else np.nan
        matched = False
    else:
        toa_sec = row["time"]
        dm      = row["dm"]
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
        "beam": name_info["beam"],
        "dm": dm,
        "freq": info["freq"],
    }

    save_to_h5(output_dir, h5_name, data, meta)
    return "ok" if matched else "ok_no_catalog_match"


def save_obs_json(output_dir: Path):
    write_obs_info_json(output_dir)


def collect_tasks(
    asd_root: Path,
    output_root: Path,
    catalog_dir: Path,
    prefix: str,
    only: set[str],
    overwrite: bool,
    dry_run: bool = False,
):
    catalog = parse_burst_catalog(catalog_dir, prefix)
    if not catalog:
        print(f"No {prefix}*_Burst.txt catalog rows found under {catalog_dir}")
        return [], []

    tasks            = []
    date_output_dirs = set()

    frb_dirs = [
        path
        for path in sorted(asd_root.iterdir())
        if is_source_frb_dir(path, prefix) and (not only or path.name in only)
    ]
    print(f"Found {len(frb_dirs)} {prefix}* directories")

    for frb_path in frb_dirs:
        rows         = catalog.get(frb_path.name, [])
        rows_by_date = defaultdict(list)
        for row in rows:
            rows_by_date[row["date"]].append(row)
        if not rows:
            print(
                f"[WARN] {frb_path.name}: no catalog rows, metadata will fall back to filename"
            )

        frb_count = 0
        for date_path in sorted(
            path for path in frb_path.iterdir() if is_date_dir(path)
        ):
            output_path = output_root / frb_path.name / date_path.name
            cal_path    = copy_cal_files(date_path, output_path, overwrite, dry_run)
            if cal_path is None:
                print(
                    f"[WARN] {frb_path.name}/{date_path.name}: no *_0001.fits calibration file"
                )
                continue

            info = read_obs_info(cal_path)
            burst_files = [
                path
                for path in sorted(date_path.iterdir())
                if path.is_file()
                and path.suffix == ".fits"
                and parse_old_filename(path.name) is not None
            ]
            for burst_path in burst_files:
                tasks.append(
                    (
                        str(burst_path),
                        str(output_path),
                        rows_by_date[date_path.name],
                        info,
                        overwrite,
                    )
                )
            frb_count += len(burst_files)
            if burst_files:
                date_output_dirs.add(output_path)

        print(f"{frb_path.name}: {frb_count} burst FITS")

    return tasks, sorted(date_output_dirs)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asd-root",
        required = True,
        help     = "待迁移的旧 burst FITS 根目录；必须显式指定，避免误扫目录",
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--catalog-dir", default=str(DEFAULT_CATALOG_DIR))
    parser.add_argument("--frb-prefix", default="FRB20")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action = "store_true",
        help   = "只统计待转换文件，不创建或修改输出文件",
    )
    parser.add_argument(
        "--only",
        nargs   = "*",
        default = None,
        help    = "Optional FRB directory names to process",
    )
    return parser.parse_args()


def main():
    args        = parse_args()
    asd_root    = Path(args.asd_root)
    output_root = Path(args.output_root)
    catalog_dir = Path(args.catalog_dir)
    only        = set(args.only or [])

    tasks, date_output_dirs = collect_tasks(
        asd_root,
        output_root,
        catalog_dir,
        args.frb_prefix,
        only,
        args.overwrite,
        args.dry_run,
    )
    if not tasks:
        print("No burst FITS files to convert")
        return

    if args.dry_run:
        print(f"Dry run: {len(tasks)} burst FITS would be considered")
        print(f"Dry run: {len(date_output_dirs)} date output directories")
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

# fmt: on
