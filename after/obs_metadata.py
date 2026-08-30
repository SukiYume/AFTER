# fmt: off

"""汇总 burst H5 元数据并生成可信的 ``obs_info.json``。

一个输出目录可能包含不同波束、DM 或裁切长度的爆发，因此不能拿“最后处理的一组”
代表整场观测。本模块逐个读取 H5：所有文件取值相同时输出一个标量，不同时输出排序
稳定的列表；同时保留逐 burst 记录，方便追溯每个文件实际使用的参数。
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import h5py
import numpy as np

# 这些字段既写入观测级摘要，也从每个 H5 独立读取，防止批处理中不同分组互相覆盖。
_SUMMARY_FIELDS = (
    "obs_start_mjd",
    "nchan",
    "time_reso",
    "npol",
    "beam",
    "dm",
    "segment_length",
)


def _json_scalar(value):
    """把 HDF5 标量转成严格 JSON 可以编码的 Python 值。

    NumPy 标量先拆成原生类型，字节串按 UTF-8 解码；JSON 不允许的 NaN/inf 统一写成
    ``null``（这里用 ``None`` 表示）。
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _collapse(values):
    """相同取值折叠成一个标量，不同取值返回顺序稳定的去重列表。"""
    unique = []
    for value in values:
        if value not in unique:
            unique.append(value)

    def sort_key(value):
        """让数值、普通文本和缺失值的排序跨文件保持确定性。"""
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return (0, float(value))
        if value is None:
            return (2, "")
        return (1, str(value))

    unique.sort(key=sort_key)
    return unique[0] if len(unique) == 1 else unique


def _beam_from_filename(filename):
    """从文件名的独立 ``Mdd`` 段提取整数波束号；找不到时返回 ``None``。"""
    match = re.search(r"(?:^|[-_])M(\d{2})(?=[-_.]|$)", filename, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _normalize_beam(value, filename):
    """把 H5 中可能是 ``1``、``1.0`` 或 ``M01`` 的波束值统一成整数。

    旧文件没有 ``beam`` 属性时，再退回到文件名解析；无法安全解释的值原样保留，
    避免静默猜错。
    """
    value = _json_scalar(value)
    if value is None:
        return _beam_from_filename(filename)
    if isinstance(value, str):
        match = re.fullmatch(r"M?(\d+)", value, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    if isinstance(value, (int, float)) and float(value).is_integer():
        return int(value)
    return value


def build_obs_info(h5_files):
    """逐个读取 H5，构造观测级摘要和逐 burst 元数据。

    返回值中的顶层字段是全体文件的折叠结果，``bursts`` 则一一记录文件名、TOA、
    波束、DM 和裁切长度。输入为空时返回 ``None``。
    """
    records = []
    for h5_path in sorted(Path(path) for path in h5_files):
        with h5py.File(h5_path, "r") as h5:
            record = {
                field: _json_scalar(h5.attrs.get(field)) for field in _SUMMARY_FIELDS
            }
            record["beam"]    = _normalize_beam(record["beam"], h5_path.name)
            record["file"]    = h5_path.name
            record["toa_sec"] = _json_scalar(h5.attrs.get("toa_sec"))
            if isinstance(record["toa_sec"], float):
                record["toa_sec"] = round(record["toa_sec"], 4)
            records.append(record)

    if not records:
        return None

    obs_info = {
        field: _collapse([record[field] for record in records])
        for field in _SUMMARY_FIELDS
    }
    obs_info["bursts"] = [
        {
            "file": record["file"],
            "toa_sec": record["toa_sec"],
            "beam": record["beam"],
            "dm": record["dm"],
            "segment_length": record["segment_length"],
        }
        for record in records
    ]
    return obs_info


def write_obs_info_json(output_dir):
    """扫描一个观测目录中的未定标 H5，并写出汇总后的 ``obs_info.json``。"""
    output_path = Path(output_dir)
    h5_files = sorted(
        path
        for path in output_path.iterdir()
        if path.suffix == ".h5" and not path.name.endswith("_cal.h5")
    )
    obs_info = build_obs_info(h5_files)
    if obs_info is None:
        return None

    json_path = output_path / "obs_info.json"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(obs_info, stream, indent=2, ensure_ascii=False, allow_nan=False)
    return json_path

# fmt: on
