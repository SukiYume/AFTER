"""Run cut_burst_data.py on a Burst.txt table and write date-grouped H5 cuts."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from cut_burst_data import (
    calc_dispersion_shift,
    cut_one_burst,
    read_obs_info,
    save_obs_json,
)


DEFAULT_SEGMENT_LENGTH = 4096 * 16


def read_burst_txt(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.split()
            if not parts or parts[0].lower() == "base":
                continue
            if len(parts) < 7:
                raise ValueError(f"Malformed row {path}:{line_no}: {line.rstrip()}")
            base, project, name, date, beam, dm, toa = parts[:7]
            rows.append(
                {
                    "base": base,
                    "project": project,
                    "name": name,
                    "date": date,
                    "beam": int(beam),
                    "dm": float(dm),
                    "time": float(toa),
                }
            )
    return rows


def find_file_list(data_path: Path, beam: int):
    pattern = f"M{beam:02d}"
    return sorted(
        name
        for name in os.listdir(data_path)
        if pattern in name
        and name.endswith(".fits")
        and all(x not in name for x in ["_F_", "_N_", "_W_"])
    )


def run_group(args):
    key, rows, output_root, save_frb_name, segment_length, workers, overwrite = args
    base, project, raw_name, date, beam, dm = key
    data_path = Path(f"/{base}/{project}/{raw_name}/{date}/")
    save_path = Path(output_root) / date

    if not data_path.exists():
        raise FileNotFoundError(data_path)

    file_list = find_file_list(data_path, beam)
    if not file_list:
        raise FileNotFoundError(f"No M{beam:02d} FITS under {data_path}")

    save_path.mkdir(parents=True, exist_ok=True)
    cal_dst = save_path / file_list[0]
    if overwrite or not cal_dst.exists():
        shutil.copy2(data_path / file_list[0], cal_dst)

    if overwrite:
        for path in save_path.glob(f"{save_frb_name}-{date}-M{beam:02d}-*.h5"):
            path.unlink()
        obs_json = save_path / "obs_info.json"
        if obs_json.exists():
            obs_json.unlink()

    info = read_obs_info(str(data_path), file_list)
    shifts, max_shift = calc_dispersion_shift(dm, info["freq"], info["time_reso"])
    cut_args = [
        (
            str(data_path),
            str(save_path) + "/",
            file_list,
            info,
            dm,
            row["time"],
            shifts,
            max_shift,
            segment_length,
            save_frb_name,
            date,
            beam,
        )
        for row in rows
    ]

    if workers > 1 and len(cut_args) > 1:
        with Pool(workers) as pool:
            pool.starmap(cut_one_burst, cut_args)
    else:
        for cut_arg in cut_args:
            cut_one_burst(*cut_arg)

    save_obs_json(str(save_path), info, dm)
    return date, len(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--burst-txt", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--save-frb-name", default=None)
    parser.add_argument("--segment-length", type=int, default=DEFAULT_SEGMENT_LENGTH)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    burst_txt = Path(args.burst_txt)
    save_frb_name = args.save_frb_name or burst_txt.name[: -len("_Burst.txt")]
    rows = read_burst_txt(burst_txt)
    if not rows:
        raise SystemExit(f"No rows in {burst_txt}")

    grouped = defaultdict(list)
    for row in rows:
        key = (row["base"], row["project"], row["name"], row["date"], row["beam"], row["dm"])
        grouped[key].append(row)

    print(f"Loaded {len(rows)} rows in {len(grouped)} data groups")
    group_args = [
        (key, group_rows, args.output_root, save_frb_name, args.segment_length, args.workers, args.overwrite)
        for key, group_rows in sorted(grouped.items(), key=lambda item: item[0])
    ]
    results = [run_group(group_arg) for group_arg in group_args]
    print("Cut summary:")
    for date, count in results:
        print(f"  {date}: {count}")


if __name__ == "__main__":
    main()
