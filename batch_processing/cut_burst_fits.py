# fmt: off

"""根据 ``Burst.txt`` 清单从原始 PSRFITS 中重新切出消色散后的 FITS。

这个入口只负责“原始观测 -> burst FITS”，不做流量、偏振或噪声管定标。
它复用 :mod:`after.cut_burst_data` 中已经过测试的跨文件读取和消色散逻辑，
并把结果写成短 PSRFITS 格式。

典型用法::

    python cut_burst_fits.py \
        --burst-txt ../burst_txt/FRB20190417A_Burst.txt \
        --output-root /path/to/rebuilt_fits

清单通常包含 ``base project name date beam dm time`` 七列。其中 ``base``
写成 ``data31`` 或 ``data32`` 时，原始目录会解释为
``/<base>/<project>/<name>/<date>``。没有 ``base`` 列的清单必须额外传入
``--raw-root``，避免程序猜测数据位置。
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits


def _find_project_root() -> Path:
    """从脚本位置向上寻找包含 ``after`` 包的项目根目录。"""

    for parent in Path(__file__).resolve().parents:
        if (parent / "after" / "cut_burst_data.py").is_file():
            return parent
    raise RuntimeError("找不到 data_processing 项目根目录")


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from after.cut_burst_data import (  # noqa: E402  # 先把项目根目录加入 sys.path
    calc_dispersion_shift,
    dedisperse,
    extract_segment,
    read_obs_info,
)


@dataclass(frozen=True)
class BurstRow:
    """清单中的一条爆发记录。"""

    raw_dir: Path
    project: str
    raw_name: str
    date: str
    beam: int
    dm: float
    toa_sec: float
    start_sample: int | None       = None
    output_file: str | None        = None
    segment_length: int | None     = None
    output_subdir: str | None      = None
    template_file: str | None      = None
    raw_file_manifest: Path | None = None
    rebuild_status: str            = "ready"


def infer_output_name(path: Path) -> str:
    """从清单名推断输出 FRB 名称，同时兼容 legacy FITS 清单。"""

    name = path.name
    for suffix in (
        "_FourFRBs_legacy_fits_rebuild.txt",
        "_legacy_fits_rebuild.txt",
        "_legacy_fits_Burst.txt",
        "_Burst.txt",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def read_burst_txt(path: Path, raw_root: Path | None = None) -> list[BurstRow]:
    """读取空白分隔的 Burst 表，并把每行解析成明确的原始观测目录。"""

    text_lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not text_lines:
        raise ValueError(f"空清单: {path}")
    # TSV 精确索引需要保留 blocked 行中的空 manifest 列；普通 Burst.txt
    # 使用任意空白分隔。
    if "\t" in text_lines[0]:
        lines = list(csv.reader(text_lines, delimiter="\t"))
    else:
        lines = [line.split() for line in text_lines]

    header   = [item.lower() for item in lines[0]]
    required = {"project", "name", "date", "beam", "dm", "time"}
    missing  = required.difference(header)
    if missing:
        raise ValueError(f"{path} 缺少列: {', '.join(sorted(missing))}")
    if "base" not in header and raw_root is None:
        raise ValueError(f"{path} 没有 base 列，必须传入 --raw-root")

    rows: list[BurstRow] = []
    for line_number, fields in enumerate(lines[1:], start=2):
        if len(fields) != len(header):
            raise ValueError(
                f"{path}:{line_number} 列数为 {len(fields)}，表头为 {len(header)}"
            )
        record   = dict(zip(header, fields, strict=True))
        project  = record["project"]
        raw_name = record["name"]
        date     = record["date"]
        if raw_root is not None:
            data_root = raw_root
        else:
            # ``base`` 可以是 data31，也可以是 home/user/raw 这样的相对根路径。
            data_root = Path("/") / record["base"].lstrip("/")
        raw_file_manifest: Path | None = None
        manifest_value                 = record.get("raw_file_manifest")
        if manifest_value:
            manifest_relative = Path(manifest_value)
            if manifest_relative.is_absolute() or ".." in manifest_relative.parts:
                raise ValueError(
                    f"{path}:{line_number} raw_file_manifest 必须是清单目录内的相对路径"
                )
            manifest_root     = path.parent.resolve()
            raw_file_manifest = (manifest_root / manifest_relative).resolve()
            try:
                raw_file_manifest.relative_to(manifest_root)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number} raw_file_manifest 超出清单目录"
                ) from exc
        rows.append(
            BurstRow(
                raw_dir           = data_root / project / raw_name / date,
                project           = project,
                raw_name          = raw_name,
                date              = date,
                beam              = int(record["beam"]),
                dm                = float(record["dm"]),
                toa_sec           = float(record["time"]),
                start_sample      = (
                    int(record["start_sample"]) if "start_sample" in record else None
                ),
                output_file       = record.get("output_file"),
                segment_length    = (
                    int(record["segment_length"])
                    if "segment_length" in record
                    else None
                ),
                output_subdir     = record.get("output_subdir"),
                template_file     = record.get("template_file"),
                raw_file_manifest = raw_file_manifest,
                rebuild_status    = record.get("rebuild_status", "ready"),
            )
        )
        if (
            rows[-1].output_file
            and Path(rows[-1].output_file).name != rows[-1].output_file
        ):
            raise ValueError(
                f"{path}:{line_number} output_file 必须只是文件名，不能包含目录"
            )
        if (
            rows[-1].output_subdir
            and Path(rows[-1].output_subdir).name != rows[-1].output_subdir
        ):
            raise ValueError(f"{path}:{line_number} output_subdir 必须是单层目录名")
        if (
            rows[-1].template_file
            and Path(rows[-1].template_file).name != rows[-1].template_file
        ):
            raise ValueError(f"{path}:{line_number} template_file 必须只是文件名")
        if not rows[-1].rebuild_status:
            raise ValueError(f"{path}:{line_number} rebuild_status 不能为空")
    if not rows:
        raise ValueError(f"{path} 只有表头，没有爆发记录")
    return rows


def _read_raw_file_manifest(path: Path, beam: int) -> list[str]:
    """读取冻结的原始文件顺序，拒绝目录名、重复项和错误波束。"""

    if not path.is_file():
        raise FileNotFoundError(f"找不到原始文件顺序清单: {path}")
    marker = f"M{beam:02d}"
    files = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not files:
        raise ValueError(f"原始文件顺序清单为空: {path}")
    if len(files) != len(set(files)):
        raise ValueError(f"原始文件顺序清单含重复项: {path}")
    invalid = [
        name
        for name in files
        if Path(name).name != name
        or marker not in name
        or not name.endswith(".fits")
        or "_cal" in name.lower()
        or any(token in name for token in ("_F_", "_N_", "_W_"))
    ]
    if invalid:
        raise ValueError(f"{path} 含非法文件名，例如: {invalid[0]}")
    return files


def find_raw_fits(
    raw_dir: Path,
    beam: int,
    raw_file_manifest: Path | None = None,
) -> list[str]:
    """返回指定波束的原始 FITS；可按冻结清单锁定文件顺序。"""

    marker = f"M{beam:02d}"
    # 原始目录可能位于网络文件系统并包含数千个文件。Path.is_file() 会对每个
    # 条目单独发起一次 stat，代价很高；scandir 通常能直接复用目录项类型信息。
    with os.scandir(raw_dir) as entries:
        names = [entry.name for entry in entries if entry.is_file()]

    if raw_file_manifest is not None:
        files     = _read_raw_file_manifest(raw_file_manifest, beam)
        available = set(names)
        compressed = [
            name
            for name in files
            if name not in available and name + ".xz" in available
        ]
        missing = [
            name
            for name in files
            if name not in available and name + ".xz" not in available
        ]
        if missing:
            raise FileNotFoundError(
                f"{raw_dir} 缺少冻结清单中的 {len(missing)} 个文件，例如 {missing[0]}"
            )
        if compressed:
            raise RuntimeError(
                f"{raw_dir} 中冻结清单有 {len(compressed)} 个 .fits.xz；"
                "请先在临时目录解压，再用 --raw-root 指向解压后的根目录"
            )
        return files

    files = sorted(
        name
        for name in names
        if marker in name
        and name.endswith(".fits")
        and "_cal" not in name.lower()
        and all(token not in name for token in ("_F_", "_N_", "_W_"))
    )
    if files:
        return files

    compressed = sorted(
        name
        for name in names
        if marker in name
        and name.endswith(".fits.xz")
        and "_cal" not in name.lower()
        and all(token not in name for token in ("_F_", "_N_", "_W_"))
    )
    if compressed:
        raise RuntimeError(
            f"{raw_dir} 只有 {len(compressed)} 个 .fits.xz；"
            "请先在临时目录解压，再用 --raw-root 指向解压后的根目录"
        )
    raise FileNotFoundError(f"{raw_dir} 中没有波束 {marker} 的原始 FITS")


def _write_cut_fits(
    template_path: Path,
    output_path: Path,
    data: np.ndarray,
    overwrite: bool,
) -> None:
    """把 ``(time, pol, freq)`` 数据装入缩短后的 PSRFITS 模板。"""

    if output_path.exists() and not overwrite:
        print(f"  [跳过] 已存在: {output_path.name}")
        return

    with fits.open(template_path, memmap=True) as hdul:
        subint = hdul[1]
        if subint.data is None:
            raise ValueError(f"FITS SUBINT 无数据: {template_path}")
        nsblk = int(subint.header["NSBLK"])
        npol  = int(subint.header["NPOL"])
        nchan = int(subint.header["NCHAN"])
        if data.shape[0] % nsblk:
            raise ValueError(f"切片长度 {data.shape[0]} 不能被模板 NSBLK={nsblk} 整除")
        if data.shape[1:] != (npol, nchan):
            raise ValueError(
                f"数据形状 {data.shape} 与模板 (NPOL={npol}, NCHAN={nchan}) 不一致"
            )

        nrows                   = data.shape[0] // nsblk
        subint.data             = fits.FITS_rec.from_columns(subint.data.columns, nrows=nrows)
        subint.data["DATA"]     = data.reshape(nrows, nsblk, npol, nchan, 1)
        subint.header["NAXIS2"] = nrows

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        try:
            hdul.writeto(temp_path, overwrite=True)
            os.replace(temp_path, output_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
    print(f"  [完成] {output_path.name}")


def _group_rows(
    rows: Iterable[BurstRow],
) -> dict[
    tuple[
        Path,
        str,
        int,
        float,
        int | None,
        str | None,
        str | None,
        Path | None,
    ],
    list[BurstRow],
]:
    """同一场观测只读取一次 FITS 元数据和色散位移。"""

    groups: dict[
        tuple[
            Path,
            str,
            int,
            float,
            int | None,
            str | None,
            str | None,
            Path | None,
        ],
        list[BurstRow],
    ] = {}
    for row in rows:
        key = (
            row.raw_dir,
            row.date,
            row.beam,
            row.dm,
            row.segment_length,
            row.output_subdir,
            row.template_file,
            row.raw_file_manifest,
        )
        groups.setdefault(key, []).append(row)
    return groups


def cut_group(
    rows: list[BurstRow],
    output_root: Path,
    output_name: str,
    segment_length: int,
    overwrite: bool,
    copy_first_fits: bool,
) -> None:
    """处理同一原始目录、波束和 DM 下的一组爆发。"""

    first                    = rows[0]
    effective_segment_length = first.segment_length or segment_length
    file_list = find_raw_fits(
        first.raw_dir,
        first.beam,
        first.raw_file_manifest,
    )
    info              = read_obs_info(str(first.raw_dir), file_list)
    shifts, max_shift = calc_dispersion_shift(first.dm, info["freq"], info["time_reso"])
    save_dir          = output_root / output_name / (first.output_subdir or first.date)
    save_dir.mkdir(parents=True, exist_ok=True)

    if copy_first_fits:
        calibration_copy = save_dir / file_list[0]
        if overwrite or not calibration_copy.exists():
            temp_copy = calibration_copy.with_name(f"{calibration_copy.name}.tmp")
            temp_copy.unlink(missing_ok=True)
            try:
                shutil.copy2(first.raw_dir / file_list[0], temp_copy)
                os.replace(temp_copy, calibration_copy)
            finally:
                temp_copy.unlink(missing_ok=True)

    template_name = first.template_file or file_list[min(10, len(file_list) - 1)]
    if template_name not in file_list:
        raise FileNotFoundError(
            f"精确索引指定的模板 {template_name} 不在 {first.raw_dir}"
        )
    template_path = first.raw_dir / template_name
    total_samples = int(info["file_nsamp"]) * len(file_list)
    for row in rows:
        # 两位小数的 time 可能偏离几十个采样点；提供 start_sample 时以精确
        # 采样位置切片，time 只用于科学记录。
        center_sample = (
            row.start_sample + effective_segment_length // 2
            if row.start_sample is not None
            else int(row.toa_sec / info["time_reso"])
        )
        if center_sample < 0 or center_sample >= total_samples:
            raise ValueError(f"{row.date} TOA={row.toa_sec:.6f}s 超出观测范围")
        start_sample = (
            row.start_sample
            if row.start_sample is not None
            else center_sample - effective_segment_length // 2
        )
        segment = extract_segment(
            str(first.raw_dir),
            file_list,
            info,
            start_sample,
            effective_segment_length + max_shift,
        )
        data        = dedisperse(segment, shifts, effective_segment_length)
        fits_number = center_sample // int(info["file_nsamp"]) + 1
        start_token = (
            f"n{abs(start_sample):09d}" if start_sample < 0 else f"{start_sample:09d}"
        )
        output_file = row.output_file or (
            f"{output_name}-{row.date}-M{row.beam:02d}-"
            f"{fits_number:04d}-{start_token}.fits"
        )
        output_path = save_dir / output_file
        _write_cut_fits(template_path, output_path, data, overwrite)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--burst-txt", type=Path, required=True, help="Burst.txt 清单")
    parser.add_argument(
        "--output-root", type=Path, required=True, help="重建 FITS 的根目录"
    )
    parser.add_argument(
        "--raw-root",
        type = Path,
        help = "覆盖清单中的 base；无 base 列的旧清单必须提供",
    )
    parser.add_argument("--output-name", help="输出 FRB 名；默认从清单文件名推断")
    parser.add_argument("--segment-length", type=int, default=4096)
    parser.add_argument(
        "--only-date", action="append", default=[], help="只处理指定日期，可重复"
    )
    parser.add_argument("--limit", type=int, help="仅处理排序后的前 N 条，便于验证")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-copy-first-fits", action="store_true")
    parser.add_argument(
        "--skip-blocked",
        action = "store_true",
        help   = "跳过 rebuild_status 不是 ready* 的记录；默认遇到此类记录即停止",
    )
    parser.add_argument(
        "--dry-run",
        action = "store_true",
        help   = "只检查清单、原始目录与未压缩 FITS，不读写 burst 数据",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_burst_txt(args.burst_txt, args.raw_root)
    if args.only_date:
        selected = set(args.only_date)
        rows     = [row for row in rows if row.date in selected]
    blocked = [row for row in rows if not row.rebuild_status.startswith("ready")]
    if blocked and not args.skip_blocked:
        statuses = sorted({row.rebuild_status for row in blocked})
        raise ValueError(
            f"筛选范围内有 {len(blocked)} 条不可安全重建记录: "
            f"{', '.join(statuses)}；核查原因后可用 --skip-blocked 跳过"
        )
    if blocked:
        print(f"跳过 {len(blocked)} 条不可安全重建记录")
        rows = [row for row in rows if row.rebuild_status.startswith("ready")]
    rows.sort(key=lambda row: (row.date, row.toa_sec, row.beam))
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit 必须大于 0")
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("筛选后没有待处理爆发")

    output_name = args.output_name or infer_output_name(args.burst_txt)
    if Path(output_name).name != output_name or output_name in {".", ".."}:
        raise ValueError("--output-name 必须是单个目录名，不能包含路径")
    groups = _group_rows(rows)
    print(f"读取 {len(rows)} 条爆发，共 {len(groups)} 组原始观测")

    if args.dry_run:
        for (
            raw_dir,
            date,
            beam,
            dm,
            row_segment_length,
            output_subdir,
            template_file,
            raw_file_manifest,
        ), group_rows in sorted(groups.items(), key=lambda item: str(item[0])):
            files  = find_raw_fits(raw_dir, beam, raw_file_manifest)
            length = row_segment_length or args.segment_length
            print(
                f"  [可用] {date} M{beam:02d} DM={dm:g} N={length} "
                f"template={template_file or 'default'} -> {output_subdir or date}: "
                f"{len(group_rows)} bursts, {len(files)} FITS, {raw_dir}"
            )
        return

    for group_rows in groups.values():
        cut_group(
            group_rows,
            output_root     = args.output_root,
            output_name     = output_name,
            segment_length  = args.segment_length,
            overwrite       = args.overwrite,
            copy_first_fits = not args.no_copy_first_fits,
        )


if __name__ == "__main__":
    main()

# fmt: on
