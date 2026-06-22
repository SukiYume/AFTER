"""Batch calibration for H5_Cut burst products.

Input layout:
    ROOT_DIR/
      FRB20201124A/
        20210526/
          *.h5
          *_0001.fits

Output layout:
    CAL_ROOT/<FRB>/<date>/*_cal.h5
    CAL_ROOT/<FRB>/<date>/*.jpg

The source table is whitespace separated:
    FRB_name  DM  RA  DEC

DM is kept for bookkeeping; each burst H5 already carries its own DM attrs.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

from astropy.io import fits

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from calibration import (
    find_cal_fits,
    fold_noise_cal,
    load_t_cal,
    process_one_burst,
)


DEFAULT_ROOT_DIR = "/path/to/after_data/H5_Cut"
DEFAULT_DM_FILE = str(SCRIPT_DIR / "h5_calibration_dm_file.txt")
DEFAULT_CAL_NPZ = str(PROJECT_DIR / "highcal_20201014_psr_tny.npz")
DEFAULT_CAL_ROOT = "/path/to/after_data/H5_Cut/H5_Cal"


def extract_beam(filename):
    """Return the Mxx beam number encoded in a cut H5 filename."""
    match = re.search(r"M(\d{2})", filename)
    return int(match.group(1)) if match else 1


def parse_dm_file(path):
    """Parse source table rows into [{name, dm, ra, dec}, ...]."""
    def normalize_ra(value):
        if ":" not in value or "h" in value:
            return value
        hh, mm, ss = value.split(":", 2)
        return f"{hh}h{mm}m{ss}s"

    def normalize_dec(value):
        if ":" not in value or "d" in value:
            return value
        sign = ""
        body = value
        if body[0] in "+-":
            sign, body = body[0], body[1:]
        dd, mm, ss = body.split(":", 2)
        return f"{sign}{dd}d{mm}m{ss}s"

    sources = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                print(f"  [WARN] skip malformed source row {path}:{line_no}: {line}")
                continue
            sources.append(
                {
                    "name": parts[0],
                    "dm": float(parts[1]),
                    "ra": normalize_ra(parts[2]),
                    "dec": normalize_dec(parts[3]),
                }
            )
    return sources


def collect_calibration_groups(root_dir, cal_root, sources, cal_npz, down_time, down_freq,
                               rfi_fft, only):
    """Collect date/beam groups.

    Grouping keeps calibration arrays local to a worker process and avoids
    duplicating noise_cal/t_cal into every single burst task.
    """
    groups = []
    only = set(only or [])

    for src in sources:
        if only and src["name"] not in only:
            continue

        frb_dir = os.path.join(root_dir, src["name"])
        if not os.path.isdir(frb_dir):
            print(f'  [SKIP] {src["name"]} directory missing: {frb_dir}')
            continue

        dates = sorted(
            d
            for d in os.listdir(frb_dir)
            if d.isdigit() and len(d) == 8 and os.path.isdir(os.path.join(frb_dir, d))
        )
        print(f'  {src["name"]}: {len(dates)} date dirs')

        for date in dates:
            date_dir = os.path.join(frb_dir, date)
            burst_list = sorted(
                os.path.join(date_dir, f)
                for f in os.listdir(date_dir)
                if f.endswith(".h5") and not f.endswith("_cal.h5")
            )
            if not burst_list:
                continue

            beam_groups = defaultdict(list)
            for h5_path in burst_list:
                beam_groups[extract_beam(os.path.basename(h5_path))].append(h5_path)

            for beam, h5_list in sorted(beam_groups.items()):
                cal_fits_path = find_cal_fits(date_dir, beam) or find_cal_fits(date_dir, 1)
                if cal_fits_path is None:
                    print(
                        f'    [SKIP] {src["name"]}/{date} M{beam:02d}: '
                        f'no calibration FITS for {len(h5_list)} bursts'
                    )
                    continue

                groups.append(
                    {
                        "source": src["name"],
                        "date": date,
                        "date_dir": date_dir,
                        "output_dir": os.path.join(cal_root, src["name"], date),
                        "h5_list": h5_list,
                        "cal_fits_path": cal_fits_path,
                        "cal_npz": cal_npz,
                        "ra": src["ra"],
                        "dec": src["dec"],
                        "beam": beam,
                        "down_time": down_time,
                        "down_freq": down_freq,
                        "rfi_fft": rfi_fft,
                    }
                )

    return groups


def process_group(group):
    """Calibrate all bursts in one source/date/beam group."""
    with fits.open(group["cal_fits_path"]) as f:
        nchan = f[1].header["NCHAN"]

    noise_cal = fold_noise_cal(group["cal_fits_path"])
    t_cal = load_t_cal(group["cal_npz"], group["beam"], nchan)

    print(
        f'[{group["source"]}/{group["date"]} M{group["beam"]:02d}] '
        f'{len(group["h5_list"])} bursts'
    )
    for h5_path in group["h5_list"]:
        process_one_burst(
            h5_path,
            group["output_dir"],
            noise_cal,
            t_cal,
            group["ra"],
            group["dec"],
            group["beam"],
            group["down_time"],
            group["down_freq"],
            group["rfi_fft"],
        )
    return group["source"], group["date"], group["beam"], len(group["h5_list"])


def batch_calibrate(root_dir, cal_root, dm_file, cal_npz, down_time=None, down_freq=None,
                    rfi_fft=True, num_workers=8, only=None):
    sources = parse_dm_file(dm_file)
    print(f"[source table] {len(sources)} sources")

    groups = collect_calibration_groups(
        root_dir,
        cal_root,
        sources,
        cal_npz,
        down_time,
        down_freq,
        rfi_fft,
        only,
    )
    if not groups:
        print("No calibration groups to process")
        return []

    total_bursts = sum(len(g["h5_list"]) for g in groups)
    print(f"Start calibration: {len(groups)} groups, {total_bursts} bursts, workers={num_workers}")

    if num_workers > 1 and len(groups) > 1:
        with Pool(num_workers) as pool:
            results = pool.map(process_group, groups)
    else:
        results = [process_group(group) for group in groups]

    print("Calibration finished")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default=DEFAULT_ROOT_DIR)
    parser.add_argument("--cal-root", default=DEFAULT_CAL_ROOT)
    parser.add_argument("--dm-file", default=DEFAULT_DM_FILE)
    parser.add_argument("--cal-npz", default=DEFAULT_CAL_NPZ)
    parser.add_argument("--down-time", type=int, default=None)
    parser.add_argument("--down-freq", type=int, default=None)
    parser.add_argument("--rfi-down-freq", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--rfi-fft", action="store_true", default=True,
                        help="Use FFT RFI flagger during calibration (default)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--only", nargs="*", default=None, help="Optional FRB names to process")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batch_calibrate(
        args.root_dir,
        args.cal_root,
        args.dm_file,
        args.cal_npz,
        down_time=args.down_time,
        down_freq=args.down_freq,
        rfi_fft=args.rfi_fft,
        num_workers=args.workers,
        only=args.only,
    )
