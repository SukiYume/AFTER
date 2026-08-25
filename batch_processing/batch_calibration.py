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

from after.calibration import (  # noqa: E402
    find_cal_fits,
    fold_noise_cal,
    load_t_cal,
    process_one_burst,
)
from after import DEFAULT_CAL_NPZ as DEFAULT_CAL_NPZ_PATH  # noqa: E402


DEFAULT_ROOT_DIR = str(PROJECT_DIR / "H5_Cut")
DEFAULT_DM_FILE = str(SCRIPT_DIR / "h5_calibration_dm_file.txt")
DEFAULT_CAL_NPZ = str(DEFAULT_CAL_NPZ_PATH)
DEFAULT_CAL_ROOT = str(PROJECT_DIR / "H5_Cal")


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
                               rfi_fft, only, time_crop_samples=None,
                               target_time_reso=None, output_time_samples=None):
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
            # 同一天拆成多段观测时，目录名会是 YYYYMMDD_1、YYYYMMDD_2。
            if re.fullmatch(r"\d{8}(?:_\d+)?", d)
            and os.path.isdir(os.path.join(frb_dir, d))
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
                cal_fits_path = find_cal_fits(date_dir, beam)
                if cal_fits_path is None:
                    raise FileNotFoundError(
                        f'{src["name"]}/{date} M{beam:02d}: no matching '
                        f'calibration FITS for {len(h5_list)} bursts'
                    )

                # 同一日期的观测有时会被拆成 YYYYMMDD_1/2。后一段
                # 目录中名为 *_0001.fits 的文件不一定真的含有噪声管信号；
                # 因此先用当前目录的文件，诊断不通过时再按顺序尝试同日
                # 其他分段的同波束定标文件。不跨日期借用定标。
                date_base = date.split("_", 1)[0]
                cal_fits_candidates = [cal_fits_path]
                for sibling_date in dates:
                    if sibling_date == date or sibling_date.split("_", 1)[0] != date_base:
                        continue
                    sibling_path = find_cal_fits(
                        os.path.join(frb_dir, sibling_date), beam
                    )
                    if sibling_path is not None and sibling_path not in cal_fits_candidates:
                        cal_fits_candidates.append(sibling_path)

                groups.append(
                    {
                        "source": src["name"],
                        "date": date,
                        "date_dir": date_dir,
                        "output_dir": os.path.join(cal_root, src["name"], date),
                        "h5_list": h5_list,
                        "cal_fits_path": cal_fits_path,
                        "cal_fits_candidates": cal_fits_candidates,
                        "cal_npz": cal_npz,
                        "ra": src["ra"],
                        "dec": src["dec"],
                        "beam": beam,
                        "down_time": down_time,
                        "down_freq": down_freq,
                        "rfi_fft": rfi_fft,
                        "time_crop_samples": time_crop_samples,
                        "target_time_reso": target_time_reso,
                        "output_time_samples": output_time_samples,
                    }
                )

    return groups


def process_group(group):
    """Calibrate all bursts in one source/date/beam group."""
    candidates = group.get("cal_fits_candidates", [group["cal_fits_path"]])
    calibration_errors = []
    for cal_fits_path in candidates:
        try:
            with fits.open(cal_fits_path) as f:
                nchan = f[1].header["NCHAN"]
            noise_cal = fold_noise_cal(
                cal_fits_path, diagnostic_dir=group["output_dir"]
            )
        except ValueError as exc:
            calibration_errors.append((cal_fits_path, exc))
            print(
                f'  [noise-cal 不可用] {group["source"]}/{group["date"]} '
                f'M{group["beam"]:02d}: {os.path.basename(cal_fits_path)}: {exc}'
            )
            continue
        break
    else:
        details = "; ".join(
            f"{os.path.basename(path)}: {error}"
            for path, error in calibration_errors
        )
        raise ValueError(
            f'{group["source"]}/{group["date"]} M{group["beam"]:02d} '
            f"同日候选定标文件全部不可用: {details}"
        )

    if cal_fits_path != group["cal_fits_path"]:
        print(
            f'  [noise-cal 同日回退] {group["source"]}/{group["date"]} '
            f'M{group["beam"]:02d}: 改用 {cal_fits_path}'
        )
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
            cal_fits_path,
            group["cal_npz"],
            down_time=group["down_time"],
            down_freq=group["down_freq"],
            rfi_fft=group["rfi_fft"],
            time_crop_samples=group["time_crop_samples"],
            target_time_reso=group["target_time_reso"],
            output_time_samples=group["output_time_samples"],
        )
    return group["source"], group["date"], group["beam"], len(group["h5_list"])


def batch_calibrate(root_dir, cal_root, dm_file, cal_npz, down_time=None, down_freq=None,
                    rfi_fft=True, num_workers=8, only=None, time_crop_samples=None,
                    target_time_reso=None, output_time_samples=None):
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
        time_crop_samples,
        target_time_reso,
        output_time_samples,
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
    parser.add_argument(
        "--time-crop-samples",
        type=int,
        default=None,
        help=(
            "Center-crop this many raw time samples before downsampling; "
            "when set, time downsampling defaults to 1 and plots use saved resolution"
        ),
    )
    parser.add_argument(
        "--target-time-reso-ms",
        type=float,
        default=None,
        help=(
            "Target saved time resolution in milliseconds; an integer down_time "
            "is calculated separately for every input H5"
        ),
    )
    parser.add_argument(
        "--output-time-samples",
        type=int,
        default=None,
        help=(
            "Center-crop to this many samples after calibration and saved-resolution "
            "downsampling"
        ),
    )
    parser.add_argument("--rfi-down-freq", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--rfi-fft",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use FFT RFI flagger; pass --no-rfi-fft for entropy mode",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--only", nargs="*", default=None, help="Optional FRB names to process")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.target_time_reso_ms is not None and args.down_time is not None:
        raise SystemExit("--target-time-reso-ms and --down-time are mutually exclusive")
    if args.output_time_samples is not None and args.time_crop_samples is not None:
        raise SystemExit("--output-time-samples and --time-crop-samples are mutually exclusive")
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
        time_crop_samples=args.time_crop_samples,
        target_time_reso=(
            None
            if args.target_time_reso_ms is None
            else args.target_time_reso_ms * 1e-3
        ),
        output_time_samples=args.output_time_samples,
    )
