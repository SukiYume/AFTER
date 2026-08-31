# fmt: off

"""按清单切出选定的 CHIME/ILTJ 长周期暂现源候选体。

``Selected_LongPeriod_Burst.txt`` 在 Burst.txt 字段之外，为每一行提供
``segment_length`` 和图像来源字段。输出写入
``{output_root}/{source}/{date}/``，每组数据使用清单指定的片段长度。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from after.cut_burst_data import (  # noqa: E402  # 先把仓库根目录加入 sys.path
    calc_dispersion_shift,
    cut_one_burst,
    read_obs_info,
    save_obs_json,
)

DEFAULT_PLAN        = SCRIPT_DIR / "Selected_LongPeriod_Burst.txt"
DEFAULT_OUTPUT_ROOT = "/path/to/after_data/LPT_Selected_Cut"


def read_plan(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if parts[0].lower() == "base":
                continue
            if len(parts) < 8:
                raise ValueError(f"Malformed row {path}:{line_no}: {line.rstrip()}")
            if len(parts) >= 12 and ":" in parts[8] and ":" in parts[9]:
                selected_images = parts[10]
                note            = parts[11]
            else:
                selected_images = parts[8] if len(parts) > 8 else ""
                note            = parts[9] if len(parts) > 9 else ""

            row = {
                "line_no": line_no,
                "base": parts[0],
                "project": parts[1],
                "name": parts[2],
                "date": parts[3],
                "beam": int(parts[4]),
                "dm": float(parts[5]),
                "time": float(parts[6]),
                "segment_length": int(parts[7]),
                "selected_images": selected_images,
                "note": note,
            }
            rows.append(row)
    return rows


def find_file_list(data_path: Path, beam: int) -> list[str]:
    pattern = f"M{beam:02d}"
    return sorted(
        name
        for name in os.listdir(data_path)
        if pattern in name
        and name.endswith(".fits")
        and all(x not in name for x in ["_F_", "_N_", "_W_"])
    )


def clear_existing_outputs(output_root: Path, rows: list[dict]) -> None:
    cleared = set()
    for row in rows:
        key = (row["name"], row["date"], row["beam"])
        if key in cleared:
            continue
        cleared.add(key)
        save_path = output_root / row["name"] / row["date"]
        if not save_path.exists():
            continue
        for path in save_path.glob(
            f"{row['name']}-{row['date']}-M{row['beam']:02d}-*.h5"
        ):
            path.unlink()
        obs_json = save_path / "obs_info.json"
        if obs_json.exists():
            obs_json.unlink()


def run_group(group_key, rows, output_root: Path, workers: int, dry_run: bool):
    base, project, source, date, beam, dm, segment_length = group_key
    data_path = Path(f"/{base}/{project}/{source}/{date}/")
    save_path = output_root / source / date

    if dry_run:
        return {
            "source": source,
            "date": date,
            "beam": beam,
            "segment_length": segment_length,
            "rows": len(rows),
            "data_path": str(data_path),
            "save_path": str(save_path),
            "status": "dry-run",
        }

    if not data_path.exists():
        raise FileNotFoundError(data_path)

    file_list = find_file_list(data_path, beam)
    if not file_list:
        raise FileNotFoundError(f"No M{beam:02d} FITS under {data_path}")

    save_path.mkdir(parents=True, exist_ok=True)
    cal_dst = save_path / file_list[0]
    if not cal_dst.exists():
        shutil.copy2(data_path / file_list[0], cal_dst)

    info              = read_obs_info(str(data_path), file_list)
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
            source,
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

    save_obs_json(str(save_path))
    return {
        "source": source,
        "date": date,
        "beam": beam,
        "segment_length": segment_length,
        "rows": len(rows),
        "data_path": str(data_path),
        "save_path": str(save_path),
        "status": "cut",
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-txt", default=str(DEFAULT_PLAN))
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--only-source", action="append", default=[])
    parser.add_argument("--only-date", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args        = parse_args()
    plan_path   = Path(args.plan_txt)
    output_root = Path(args.output_root)

    rows = read_plan(plan_path)
    if args.only_source:
        wanted = set(args.only_source)
        rows   = [row for row in rows if row["name"] in wanted]
    if args.only_date:
        wanted = set(args.only_date)
        rows   = [row for row in rows if row["date"] in wanted]
    if not rows:
        raise SystemExit(f"No rows selected from {plan_path}")

    if args.overwrite and not args.dry_run:
        clear_existing_outputs(output_root, rows)

    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["base"],
            row["project"],
            row["name"],
            row["date"],
            row["beam"],
            row["dm"],
            row["segment_length"],
        )
        grouped[key].append(row)

    print(f"Loaded {len(rows)} rows in {len(grouped)} cut groups")
    results = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        group_rows.sort(key=lambda row: row["time"])
        results.append(
            run_group(key, group_rows, output_root, args.workers, args.dry_run)
        )

    print("Cut summary:")
    for result in results:
        print(
            "  {source} {date} M{beam:02d} seg={segment_length} rows={rows} "
            "{status} -> {save_path}".format(**result)
        )


if __name__ == "__main__":
    main()

# fmt: on
