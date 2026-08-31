# fmt: off

"""叠加标准 AFTER H5 数据，对同步选择的时间采样点搜索旋转量（RM）。

这是通用的 AFTER 分析阶段，不负责导入某次特定观测。预期流程为
``after.calibration`` → ``after.burst_detect`` → ``after.burst_sync_rm``。
输入是一至多个标准 ``*_cal.h5`` 文件：``data`` 的轴顺序为
``(I/Q/U/V, 时间, 频率)``，``freq`` 以 MHz 保存频率，属性 ``bursts`` 则保存
``burst_detect`` 确认的区域；每个区域作为一个独立爆发分量。文件可以来自不同
源、日期、时间/频率分辨率或降采样设置。

程序先在爆发区域和指定频段内仅用 Stokes I 产生候选时间采样点，再把所有分量
的候选点放在一起排序。保留使 ``sum(S/N_I**2) / sqrt(n_time)`` 最大的排序前缀；
这一选择完全不查看 Q、U 或 RM 曲线，避免用待检验的偏振信号反过来调节选样。
最终前缀可以来自多个爆发，也可以只来自其中一个格外明亮的爆发：这里同步和
叠加的基本单元是时间采样点，并不强制每个输入爆发都作贡献。结果状态中的
``both_methods`` 表示两种 RM 统计量一致，而不是“至少两个爆发都独立检出”。
RFI 只按整条频率通道屏蔽，故意不读取或使用像素级掩码。

最终同时报告两种互补的共同 RM 统计量：

``time_pa_power``
    对每个入选时间点的法拉第功率先按噪声归一化，再跨时间点和爆发求和。
    不同时间点可以拥有不同的偏振位置角（PA）；这是主要科学统计量。

``linear_degree_stack``
    为每个爆发构造 RM—线偏振度曲线，分别做稳健标准化，再按预先固定的权重
    合并。它适合缓存和批处理，并作为主要统计量的独立交叉验证。

经验零分布从脉冲外时间点抽样：每个爆发沿用真实信号相同的时间点数量和通道
掩码，并在指定搜索窗口内取最大值，从而把“搜索了多少 RM”带来的试验因子纳入
显著性估计。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import re
import warnings as py_warnings
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypedDict, cast

import h5py
import matplotlib
import numpy as np
import pandas as pd

from .burst_pol import build_automatic_rm_grid
from .rfi import cal_rfi, robust_channel_mask

matplotlib.use("Agg")
import matplotlib.pyplot as plt             # noqa: E402  # 必须先选择无界面绘图后端再导入
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

C_M_S                   = 299_792_458.0
NULL_RM_GRID_OVERSAMPLE = 4.0
PRIMARY_METHODS         = ("time_pa_power", "linear_degree_stack")
ALL_METHODS             = PRIMARY_METHODS


class BurstRegion(TypedDict):
    """Standard burst_detect region fields used by the RM analysis."""

    time_start: int
    time_end: int
    freq_start: int
    freq_end: int


class LoadFileOptions(TypedDict):
    """Keyword arguments passed unchanged to H5-loading workers."""

    freq_min: float | None
    freq_max: float | None
    min_time_snr: float
    min_channels: int
    stored_masks_only: bool
    rfi_fft: bool
    rfi_channel_sigma: float
    rfi_channel_window: int
    rfi_channel_grow: int


class GlobalTimeInfo(TypedDict):
    """Summary returned by the global Stokes-I time-sample selection."""

    strategy: str
    candidate_component_count: int
    selected_component_count: int
    candidate_sample_count: int
    selected_sample_count: int
    objective: str
    objective_max: float
    selected_snr_min: float
    selected_snr_median: float
    selected_snr_max: float
    candidate_i_snr_squared_sum: float
    selected_i_snr_squared_sum: float
    selected_i_snr_squared_fraction: float


class MethodResult(TypedDict):
    """Machine-readable result for one common-RM statistic."""

    peak_rm_rad_m2: float | None
    peak_statistic: float | None
    peak_robust_z: float | None
    empirical_p_rm_contrast: float | None
    null_exceedances: int | None
    detected_p_le_0_01: bool | None
    leave_one_out_stable_fraction: float | None
    leave_one_out_max_shift_rmsf: float | None


@dataclass
class SearchWindow:
    """一个命名 RM 搜索窗口，``low``/``high`` 的单位均为 rad m⁻²。"""

    name: str
    low: float
    high: float


@dataclass
class BurstRMData:
    """完成通道清理和 Stokes-I 初筛后的单个爆发分量。

    ``time_indices`` 和 ``time_snr`` 描述候选时间点；``freq_mhz`` 与
    ``wave2_m2`` 是保留下来的频率和波长平方；``p_on``/``p_noise`` 分别是
    爆发内与脉冲外的复线偏振 ``Q + iU``，轴顺序均为 ``(时间, 频率)``。
    ``i_time_total`` 保存每个候选时间点跨通道求和后的 Stokes I，后续字段记录
    各类通道掩码的数量，便于审计最终有哪些通道被排除。
    """

    component_id: str
    file_name: str
    file_path: Path
    burst_idx: int
    peak_snr: float
    time_indices: np.ndarray
    time_snr: np.ndarray
    freq_mhz: np.ndarray
    wave2_m2: np.ndarray
    p_on: np.ndarray
    p_noise: np.ndarray
    i_time_total: np.ndarray
    i_total: float
    noise_variance_one_time: float
    stored_cal_rfi_count: int
    stored_burst_rfi_count: int
    recalculated_rfi_count: int
    robust_rfi_count: int
    nonfinite_rfi_count: int
    final_rfi_count: int

    @property
    def n_time(self) -> int:
        """返回当前分量最终保留的时间采样点数。"""
        return int(self.time_indices.size)

    @property
    def n_channel(self) -> int:
        """返回当前分量最终可用于 RM 合成的频率通道数。"""
        return int(self.freq_mhz.size)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析输入文件、选样阈值、RFI、RM 网格、零分布和并行设置。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cal-dir",
        type     = Path,
        required = True,
        help     = (
            "Directory containing standard calibration.py *_cal.h5 files "
            "after burst_detect has written attrs['bursts']."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type     = Path,
        required = True,
        help     = "New output directory. Existing directories are not reused.",
    )
    parser.add_argument(
        "--run-label",
        default = None,
        help    = (
            "Optional observation/group label used in plots and summaries. "
            "It does not affect sample selection or the RM statistics."
        ),
    )
    parser.add_argument(
        "--file-list",
        type    = Path,
        default = None,
        help    = (
            "Optional text file with one H5 path or cal-dir-relative filename "
            "per line. Blank lines and lines beginning with # are ignored."
        ),
    )
    parser.add_argument(
        "--recursive",
        action = "store_true",
        help   = "Find *_cal.h5 recursively below --cal-dir.",
    )
    parser.add_argument(
        "--rm-min",
        type    = float,
        default = -50_000.0,
        help    = "RM lower bound; grid spacing is derived automatically.",
    )
    parser.add_argument(
        "--rm-max",
        type    = float,
        default = 50_000.0,
        help    = "RM upper bound; grid spacing is derived automatically.",
    )
    parser.add_argument(
        "--test-window",
        action  = "append",
        default = None,
        metavar = "NAME:LOW:HIGH",
        help    = (
            "Empirical-null search window. Repeat for multiple windows. "
            "Default: full RM grid."
        ),
    )
    parser.add_argument("--n-null", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=12_110_200)
    parser.add_argument(
        "--null-pool-size",
        type    = int,
        default = 0,
        help    = (
            "Maximum off-pulse samples transformed per burst; 0 uses all. "
            "A smaller value is useful for quick validation runs."
        ),
    )
    parser.add_argument(
        "--min-time-snr",
        type    = float,
        default = 5.0,
        help    = (
            "Absolute Stokes-I S/N floor for candidate time samples. "
            "Candidates from all bursts are then selected globally."
        ),
    )
    parser.add_argument(
        "--min-peak-snr",
        type    = float,
        default = 5.0,
        help    = "Reject components whose I-only peak S/N is below this value.",
    )
    parser.add_argument(
        "--max-bursts",
        type    = int,
        default = None,
        help    = "Keep only the highest-I-S/N N components after loading.",
    )
    parser.add_argument("--freq-min", type=float, default=None)
    parser.add_argument("--freq-max", type=float, default=None)
    parser.add_argument(
        "--min-channels",
        type    = int,
        default = 32,
        help    = "Minimum number of unmasked channels required per component.",
    )
    parser.add_argument(
        "--stored-masks-only",
        action = "store_true",
        help   = (
            "Use only stored rfi_channel and burst_rfi_channel. By default "
            "channel RFI is also recomputed from off-pulse data."
        ),
    )
    parser.add_argument(
        "--rfi-fft",
        action = "store_true",
        help   = "Use FFT instead of entropy for the recalculated channel mask.",
    )
    parser.add_argument("--rfi-channel-sigma", type=float, default=6.0)
    parser.add_argument("--rfi-channel-window", type=int, default=31)
    parser.add_argument("--rfi-channel-grow", type=int, default=1)
    parser.add_argument(
        "--curve-weighting",
        choices = ("equal", "peak-snr"),
        default = "equal",
        help    = (
            "Fixed weights for linear_degree_stack. Equal is the default to "
            "avoid double-weighting high-S/N bursts."
        ),
    )
    parser.add_argument(
        "--max-weight-ratio",
        type    = float,
        default = 4.0,
        help    = "Maximum weight ratio around the median for peak-snr weighting.",
    )
    parser.add_argument(
        "--transform-chunk-size",
        type    = int,
        default = 256,
        help    = "Number of trial RMs per matrix-multiplication chunk.",
    )
    parser.add_argument(
        "--load-workers",
        type    = int,
        default = 1,
        help    = (
            "Number of spawned processes used to read H5 files and recompute "
            "channel RFI. The default 1 preserves serial behavior."
        ),
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    """在创建输出目录前检查路径、数值范围和互相依赖的命令行参数。"""
    if not args.cal_dir.is_dir():
        raise NotADirectoryError(f"Calibrated H5 directory not found: {args.cal_dir}")
    if not np.isfinite(args.rm_min) or not np.isfinite(args.rm_max):
        raise ValueError("--rm-min and --rm-max must be finite")
    if args.rm_max <= args.rm_min:
        raise ValueError("--rm-max must be greater than --rm-min")
    if args.n_null < 0:
        raise ValueError("--n-null cannot be negative")
    if args.null_pool_size < 0:
        raise ValueError("--null-pool-size cannot be negative")
    if args.min_time_snr < 0 or args.min_peak_snr < 0:
        raise ValueError("S/N thresholds cannot be negative")
    if args.max_bursts is not None and args.max_bursts < 1:
        raise ValueError("--max-bursts must be positive")
    if args.min_channels < 2:
        raise ValueError("--min-channels must be at least 2")
    if args.freq_min is not None and args.freq_max is not None and args.freq_max <= args.freq_min:
        raise ValueError("--freq-max must be greater than --freq-min")
    if args.max_weight_ratio < 1:
        raise ValueError("--max-weight-ratio must be at least 1")
    if args.transform_chunk_size < 1:
        raise ValueError("--transform-chunk-size must be positive")
    if args.load_workers < 1:
        raise ValueError("--load-workers must be positive")


def parse_search_windows(
    values: list[str] | None,
    rm_min: float,
    rm_max: float,
) -> list[SearchWindow]:
    """把重复传入的 ``名称:下限:上限`` 转成已校验的 RM 搜索窗口。

    未显式指定时返回覆盖整个 RM 网格的 ``full`` 窗口；显式窗口必须名称唯一、
    上下限有限且完全落在全局 RM 范围内。
    """
    if not values:
        return [SearchWindow("full", float(rm_min), float(rm_max))]

    windows: list[SearchWindow] = []
    names: set[str]             = set()
    for value in values:
        parts = value.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid --test-window {value!r}; expected NAME:LOW:HIGH")
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", parts[0].strip())
        if not name:
            raise ValueError(f"Invalid empty test-window name in {value!r}")
        if name in names:
            raise ValueError(f"Duplicate test-window name: {name}")
        low  = float(parts[1])
        high = float(parts[2])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError(f"Invalid test-window limits in {value!r}")
        if low < rm_min or high > rm_max:
            raise ValueError(
                f"Test window {name}={low:g}:{high:g} lies outside "
                f"the RM grid {rm_min:g}:{rm_max:g}"
            )
        names.add(name)
        windows.append(SearchWindow(name, low, high))
    return windows


def select_primary_plot_window(
    windows: list[SearchWindow],
    rm_grid: np.ndarray,
) -> SearchWindow:
    """选择主图使用的最宽盲搜窗口，结果不依赖命令行中的排列顺序。

    用户可能同时传入完整搜索区间和 RM≈0 泄漏检查等窄诊断窗口。主图不应因为
    某个窄窗口恰好排在最前面而丢失全局搜索结果，因此优先选择覆盖整个 RM 网格
    的窗口；若没有，则选择与网格重叠范围最大者。
    """
    if not windows:
        raise ValueError("At least one search window is required for plotting")
    grid = np.asarray(rm_grid, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("RM grid must contain at least two points")

    grid_low  = float(grid[0])
    grid_high = float(grid[-1])
    tolerance = 0.51 * abs(float(np.median(np.diff(grid))))
    covering = [
        window
        for window in windows
        if window.low <= grid_low + tolerance and window.high >= grid_high - tolerance
    ]
    if covering:
        return min(
            covering,
            key=lambda window: (
                abs(window.low - grid_low) + abs(window.high - grid_high),
                window.name,
            ),
        )

    def overlap(window: SearchWindow) -> float:
        """计算窗口与实际 RM 网格的重叠宽度，单位为 rad m⁻²。"""
        return max(
            0.0,
            min(window.high, grid_high) - max(window.low, grid_low),
        )

    return max(
        windows,
        key=lambda window: (
            overlap(window),
            window.high - window.low,
            window.name,
        ),
    )


def discover_h5_files(
    cal_dir: Path,
    file_list: Path | None = None,
    recursive: bool = False,
) -> list[Path]:
    """发现并校验待处理的 ``*_cal.h5`` 文件，返回去重后的绝对路径。

    若给出 ``file_list``，逐行读取绝对路径或相对 ``cal_dir`` 的路径，并跳过
    空行和 ``#`` 注释；否则按是否启用 ``recursive`` 在目录内匹配文件。
    """
    cal_dir = cal_dir.resolve()
    if file_list is not None:
        lines = file_list.read_text(encoding="utf-8").splitlines()
        files = []
        for line in lines:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            path = Path(value)
            if not path.is_absolute():
                path = cal_dir / path
            files.append(path.resolve())
    else:
        iterator: Iterable[Path]
        iterator = cal_dir.rglob("*_cal.h5") if recursive else cal_dir.glob("*_cal.h5")
        files    = [path.resolve() for path in iterator]

    unique  = sorted(set(files), key=str)
    missing = [path for path in unique if not path.is_file()]
    if missing:
        preview = "\n".join(f"  {path}" for path in missing[:10])
        raise FileNotFoundError(f"Listed calibrated H5 files do not exist:\n{preview}")
    if not unique:
        raise FileNotFoundError(f"No *_cal.h5 files found below {cal_dir}")
    return unique


def decode_regions(value: object) -> list[BurstRegion]:
    """解码并校验 H5 的 ``attrs['bursts']`` 爆发区域列表。

    属性既可为 UTF-8 JSON 字符串/字节串，也可为已解码序列。每个区域必须是
    字典，并至少包含时间和频率方向的起止索引。
    """
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        decoded = json.loads(value)
    elif isinstance(value, (list, tuple)):
        decoded = list(value)
    else:
        raise TypeError(f"Unsupported attrs['bursts'] type: {type(value).__name__}")
    if not isinstance(decoded, list):
        raise ValueError("attrs['bursts'] must decode to a list")
    required                     = {"time_start", "time_end", "freq_start", "freq_end"}
    validated: list[BurstRegion] = []
    for index, region in enumerate(decoded):
        if not isinstance(region, dict):
            raise TypeError(
                f"attrs['bursts'][{index}] must be an object, got "
                f"{type(region).__name__}"
            )
        missing = sorted(required - set(region))
        if missing:
            raise ValueError(
                f"attrs['bursts'][{index}] is missing standard burst_detect "
                f"field(s): {', '.join(missing)}"
            )
        validated.append(cast(BurstRegion, region))
    return validated


def channel_mask(handle: h5py.File, name: str, n_channel: int) -> np.ndarray:
    """读取一维通道掩码；数据集缺失时返回全 False，并严格核对长度。"""
    if name not in handle:
        return np.zeros(n_channel, dtype=bool)
    dataset = handle[name]
    if not isinstance(dataset, h5py.Dataset):
        raise TypeError(f"{handle.filename}: {name} is not a dataset")
    mask = np.asarray(dataset[...], dtype=bool)
    if mask.shape != (n_channel,):
        raise ValueError(
            f"{handle.filename}: dataset {name} has shape {mask.shape}, "
            f"expected {(n_channel,)}"
        )
    return mask


def robust_channel_variance(noise: np.ndarray) -> np.ndarray:
    """按通道估计脉冲外噪声方差，并为退化通道提供稳定回退值。

    主估计量为中位数绝对偏差换算的 ``sigma``；零值或非有限结果先回退到标准
    差，仍无效时再用其他正常通道的中位 ``sigma``，最终返回 ``sigma²``。
    """
    values = np.asarray(noise, dtype=np.float64)
    center = np.nanmedian(values, axis=0, keepdims=True)
    sigma = 1.4826 * np.nanmedian(np.abs(values - center), axis=0)
    fallback = np.nanstd(values, axis=0)
    bad = ~np.isfinite(sigma) | (sigma <= 0)
    sigma[bad] = fallback[bad]
    positive = sigma[np.isfinite(sigma) & (sigma > 0)]
    replacement = float(np.median(positive)) if positive.size else 1.0
    sigma[~np.isfinite(sigma) | (sigma <= 0)] = replacement
    return sigma**2


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    """返回有限样本的稳健中心和尺度：中位数及 MAD 换算的标准差。

    MAD 无法给出正尺度时回退到普通标准差；完全没有有限值时返回两个 NaN。
    """
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return math.nan, math.nan
    center = float(np.median(finite))
    scale  = float(1.4826 * np.median(np.abs(finite - center)))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(finite))
    return center, scale


def standardized_curve(values: np.ndarray) -> np.ndarray:
    """用稳健中心和尺度把一条曲线转成无量纲对比度（robust z）。"""
    center, scale = robust_location_scale(values)
    if not np.isfinite(scale) or scale <= 0:
        return np.zeros_like(values, dtype=np.float64)
    return (np.asarray(values, dtype=np.float64) - center) / scale


def i_only_time_candidates(
    stokes_i: np.ndarray,
    frequency_mask: np.ndarray,
    noise_mask: np.ndarray,
    burst_region: BurstRegion,
    *,
    min_snr: float,
) -> tuple[np.ndarray, dict[str, float | int], np.ndarray]:
    """仅用 Stokes I 选出一个爆发区域内达到绝对 SNR 门限的候选时间点。

    先在有效频率通道上平均成时间轮廓，用脉冲外样本的中位数和 MAD 估计基线与
    噪声，再采用与单爆发分析相同的三点三角核 ``[0.25, 0.5, 0.25]`` 轻度平滑。
    候选点只使用绝对 ``min_snr`` 门限，不使用相对本爆发峰值的门限；区域内
    所有达标点都交给后续全观测联合优化。

    返回全时间轴布尔掩码、峰值/噪声/数量摘要，以及完整的平滑 SNR 轮廓。
    """
    values      = np.asarray(stokes_i, dtype=np.float64)
    good        = np.asarray(frequency_mask, dtype=bool)
    noise       = np.asarray(noise_mask, dtype=bool)
    nsamp       = values.shape[0]
    selected    = np.zeros(nsamp, dtype=bool)
    snr_profile = np.full(nsamp, np.nan, dtype=np.float64)
    ts          = max(0, min(nsamp, int(burst_region["time_start"])))
    te          = max(0, min(nsamp, int(burst_region["time_end"])))
    if te <= ts or not np.any(good):
        return (
            selected,
            {
                "peak_sample": ts,
                "peak_snr": 0.0,
                "noise_sigma": math.nan,
                "sample_count": 0,
            },
            snr_profile,
        )

    with py_warnings.catch_warnings():
        py_warnings.filterwarnings(
            "ignore",
            message  = "Mean of empty slice",
            category = RuntimeWarning,
        )
        profile = np.nanmean(values[:, good], axis=1)
    finite_noise = profile[noise & np.isfinite(profile)]
    if finite_noise.size:
        noise_center = float(np.nanmedian(finite_noise))
        noise_sigma  = float(1.4826 * np.nanmedian(np.abs(finite_noise - noise_center)))
        if not np.isfinite(noise_sigma) or noise_sigma <= 0:
            noise_sigma = float(np.nanstd(finite_noise))
    else:
        noise_center = 0.0
        noise_sigma  = 0.0
    if not np.isfinite(noise_sigma) or noise_sigma <= 0:
        noise_sigma = np.finfo(np.float64).eps

    net = np.nan_to_num(
        profile - noise_center,
        nan    = 0.0,
        posinf = 0.0,
        neginf = 0.0,
    )
    smooth = np.convolve(
        net,
        np.array([0.25, 0.5, 0.25], dtype=np.float64),
        mode="same",
    )
    snr_profile     = np.asarray(smooth / noise_sigma, dtype=np.float64)
    region_snr      = snr_profile[ts:te]
    peak_offset     = int(np.argmax(region_snr))
    peak_sample     = ts + peak_offset
    peak_snr        = float(region_snr[peak_offset])
    selected[ts:te] = region_snr >= float(min_snr)
    return (
        selected,
        {
            "peak_sample": peak_sample,
            "peak_snr": peak_snr,
            "noise_sigma": float(noise_sigma),
            "sample_count": int(np.count_nonzero(selected)),
        },
        snr_profile,
    )


def select_global_time_samples(
    bursts: list[BurstRMData],
) -> tuple[list[BurstRMData], GlobalTimeInfo, pd.DataFrame]:
    """仅依据绝对 Stokes-I SNR，在整次观测范围选择一个统一排序前缀。

    所有分量的候选点按 I SNR 从高到低稳定排序。对长度为 ``k`` 的每个前缀计算
    ``sum(S/N_I**2) / sqrt(k)``，保留第一次达到全局最大值的前缀。这样，明亮爆发
    的强肩部可以排在另一爆发的弱峰之前，同时不会引入任何依赖 Q/U 或 RM 曲线
    的事后调参。

    返回裁切到入选时间点后的分量列表、选择摘要，以及含全部候选点和是否入选
    标记的审计表。
    """
    if not bursts:
        raise ValueError("Global time selection requires at least one burst")

    records: list[dict[str, object]] = []
    for burst in bursts:
        if burst.time_snr.shape != burst.time_indices.shape:
            raise ValueError(
                f"{burst.component_id}: time_snr shape "
                f"{burst.time_snr.shape} does not match time_indices "
                f"{burst.time_indices.shape}"
            )
        if burst.i_time_total.shape != burst.time_indices.shape:
            raise ValueError(
                f"{burst.component_id}: i_time_total shape "
                f"{burst.i_time_total.shape} does not match time_indices "
                f"{burst.time_indices.shape}"
            )
        if burst.p_on.shape[0] != burst.time_indices.size:
            raise ValueError(
                f"{burst.component_id}: p_on time axis does not match time_indices"
            )
        for sample_offset, (time_index, sample_snr) in enumerate(
            zip(burst.time_indices, burst.time_snr, strict=True)
        ):
            value = float(sample_snr)
            if not np.isfinite(value) or value <= 0:
                continue
            records.append(
                {
                    "component_id": burst.component_id,
                    "file_name": burst.file_name,
                    "burst_idx": int(burst.burst_idx),
                    "component_sample_offset": int(sample_offset),
                    "time_index": int(time_index),
                    "i_sample_snr": value,
                    "i_snr_squared": value**2,
                }
            )
    if not records:
        raise RuntimeError("No finite positive Stokes-I candidate samples")

    table = pd.DataFrame(records)
    table = table.sort_values(
        ["i_sample_snr", "component_id", "time_index"],
        ascending = [False, True, True],
        kind      = "mergesort",
    ).reset_index(drop=True)
    table.insert(0, "global_rank", np.arange(1, len(table) + 1, dtype=int))
    cumulative = np.cumsum(table["i_snr_squared"].to_numpy(dtype=np.float64))
    ranks = table["global_rank"].to_numpy(dtype=np.float64)
    objective = cumulative / np.sqrt(ranks)
    best_k = int(np.argmax(objective)) + 1
    table["cumulative_i_snr_squared"] = cumulative
    table["selection_objective"] = objective
    table["selected"] = table["global_rank"] <= best_k

    selected_bursts: list[BurstRMData] = []
    for burst in bursts:
        component_rows = table[
            (table["component_id"] == burst.component_id) & table["selected"]
        ]
        if component_rows.empty:
            continue
        offsets = np.sort(
            cast(pd.Series, component_rows["component_sample_offset"]).to_numpy(
                dtype=int
            )
        )
        i_time_total = burst.i_time_total[offsets]
        selected_bursts.append(
            replace(
                burst,
                time_indices = burst.time_indices[offsets],
                time_snr     = burst.time_snr[offsets],
                p_on         = burst.p_on[offsets],
                i_time_total = i_time_total,
                i_total      = float(np.sum(i_time_total)),
            )
        )

    selected_values = table.loc[table["selected"], "i_sample_snr"].to_numpy(
        dtype=np.float64
    )
    total_snr2    = float(cumulative[-1])
    selected_snr2 = float(cumulative[best_k - 1])
    info: GlobalTimeInfo = {
        "strategy": "global_i_snr2_over_sqrt_n",
        "candidate_component_count": int(len(bursts)),
        "selected_component_count": int(len(selected_bursts)),
        "candidate_sample_count": int(len(table)),
        "selected_sample_count": best_k,
        "objective": "sum(i_sample_snr^2) / sqrt(n_selected)",
        "objective_max": float(objective[best_k - 1]),
        "selected_snr_min": float(np.min(selected_values)),
        "selected_snr_median": float(np.median(selected_values)),
        "selected_snr_max": float(np.max(selected_values)),
        "candidate_i_snr_squared_sum": total_snr2,
        "selected_i_snr_squared_sum": selected_snr2,
        "selected_i_snr_squared_fraction": (
            selected_snr2 / total_snr2 if total_snr2 > 0 else math.nan
        ),
    }
    return selected_bursts, info, table


def short_file_id(file_name: str) -> str:
    """从标准文件名提取四位短观测编号，无法匹配时回退到文件主名。"""
    match = re.search(r"-M\d{2}-(\d{4})-", file_name)
    return match.group(1) if match else Path(file_name).stem


def component_id(file_name: str, burst_idx: int) -> str:
    """由短文件编号和文件内爆发序号构造易读的分量标识。"""
    return f"{short_file_id(file_name)}b{int(burst_idx)}"


def unique_file_id(file_name: str) -> str:
    """取得去掉扩展名及末尾 ``_cal`` 的完整文件标识。"""
    stem = Path(file_name).stem
    return stem[:-4] if stem.lower().endswith("_cal") else stem


def disambiguate_component_ids(bursts: list[BurstRMData]) -> None:
    """短分量标识发生冲突时改用完整观测文件名，并再次确认全局唯一。"""
    counts: dict[str, int] = {}
    for burst in bursts:
        counts[burst.component_id] = counts.get(burst.component_id, 0) + 1
    for burst in bursts:
        if counts[burst.component_id] > 1:
            burst.component_id = (
                f"{unique_file_id(burst.file_name)}b{int(burst.burst_idx)}"
            )
    resolved = [burst.component_id for burst in bursts]
    if len(resolved) != len(set(resolved)):
        raise ValueError("Component IDs remain non-unique after disambiguation")


def complex_faraday_transform(
    p_data: np.ndarray,
    wave2_m2: np.ndarray,
    rm_grid: np.ndarray,
    chunk_size: int = 256,
    output_dtype=np.complex128,
) -> np.ndarray:
    """在中心化的波长平方坐标上去旋转 ``Q+iU``，返回 ``F(时间, RM)``。

    对每个时间点计算
    ``F_t(RM) = Σ_chan (Q+iU) exp[-2i RM (λ²-mean(λ²))]``。减去平均 λ² 只改变
    复相位参考点，不改变功率峰位置；RM 网格分块做矩阵乘法以控制峰值内存。
    """
    dtype          = np.dtype(output_dtype)
    p_work         = np.asarray(p_data, dtype=dtype)
    wave2          = np.asarray(wave2_m2, dtype=np.float64)
    rm_values      = np.asarray(rm_grid, dtype=np.float64)
    wave2_centered = wave2 - float(np.mean(wave2))
    output         = np.empty((p_work.shape[0], rm_values.size), dtype=dtype)
    for first in range(0, rm_values.size, chunk_size):
        last = min(first + chunk_size, rm_values.size)
        phase = np.exp(
            -2j * wave2_centered[:, None] * rm_values[None, first:last]
        ).astype(dtype, copy=False)
        output[:, first:last] = p_work @ phase
    return output


def recompute_channel_rfi(
    iquv: np.ndarray,
    noise_mask: np.ndarray,
    *,
    fft: bool,
    sigma: float,
    local_window: int,
    grow: int,
) -> tuple[np.ndarray, np.ndarray]:
    """从脉冲外 I/Q/U/V 数据重新计算两套整通道 RFI 掩码。

    第一套分别对每个 Stokes 平面运行传统熵或 FFT 检测，再取逻辑并集；第二套
    使用跨 Stokes 的稳健局部通道异常检测。调用方随后把它们与 H5 内已有掩码合并。
    """
    recalculated = []
    for plane in iquv:
        channel, _ = cal_rfi(
            plane,
            noise_mask,
            down_time = 1,
            down_freq = 1,
            fft       = fft,
        )
        recalculated.append(channel)
    entropy_or_fft = np.logical_or.reduce(recalculated)
    robust = robust_channel_mask(
        iquv,
        noise_mask,
        sigma        = sigma,
        local_window = local_window,
        grow         = grow,
    )
    return entropy_or_fft, robust


def load_file_components(
    path: Path,
    *,
    freq_min: float | None,
    freq_max: float | None,
    min_time_snr: float,
    min_channels: int,
    stored_masks_only: bool,
    rfi_fft: bool,
    rfi_channel_sigma: float,
    rfi_channel_window: int,
    rfi_channel_grow: int,
) -> tuple[list[BurstRMData], list[str]]:
    """从一个标准定标 H5 中加载、清理并构造所有可用爆发分量。

    处理顺序如下：

    1. 校验 ``data``/``freq`` 的轴和 ``attrs['bursts']`` 的区域字段；
    2. 把所有爆发区域之外的时间点定义为脉冲外噪声，并逐 Stokes、逐通道减去
       脉冲外中位基线；
    3. 合并定标阶段、爆发探测阶段、重新计算、稳健局部检测及非有限数据产生的
       整通道掩码（始终不使用像素级掩码）；
    4. 对每个区域施加区域频段、用户频率范围和最少通道数要求，仅用 Stokes I
       产生时间候选点；
    5. 保存爆发内/脉冲外 ``Q+iU``、I 强度及 Q/U 噪声方差，供后续 RM 合成。

    无法使用的单个区域写入警告并跳过；结构性错误则直接抛出。返回该文件产生的
    分量列表和警告列表。
    """
    warnings: list[str] = []
    with h5py.File(path, "r") as handle:
        if "data" not in handle or "freq" not in handle:
            raise KeyError(f"{path}: calibrated H5 must contain data and freq")
        data_dataset = handle["data"]
        freq_dataset = handle["freq"]
        if not isinstance(data_dataset, h5py.Dataset):
            raise TypeError(f"{path}: data is not a dataset")
        if not isinstance(freq_dataset, h5py.Dataset):
            raise TypeError(f"{path}: freq is not a dataset")
        iquv = np.asarray(data_dataset[...], dtype=np.float64)
        freq = np.asarray(freq_dataset[...], dtype=np.float64)
        if iquv.ndim != 3 or iquv.shape[0] < 3:
            raise ValueError(
                f"{path}: data shape must be (>=3, nsamp, nchan), got {iquv.shape}"
            )
        if freq.shape != (iquv.shape[2],):
            raise ValueError(
                f"{path}: freq shape {freq.shape} does not match data {iquv.shape}"
            )
        npol = int(handle.attrs.get("npol", 4))
        if npol not in (2, 4):
            raise ValueError(f"{path}: npol must be 2 or 4, got {npol}")
        if npol == 2:
            warnings.append(
                f"{path.name}: npol={npol}; synchronized RM requires four products and was skipped"
            )
            return [], warnings
        if "bursts" not in handle.attrs:
            warnings.append(
                f"{path.name}: missing attrs['bursts']; run after.burst_detect "
                "on this calibrated H5 before burst_sync_rm; skipped"
            )
            return [], warnings
        regions      = decode_regions(handle.attrs["bursts"])
        stored_cal   = channel_mask(handle, "rfi_channel", freq.size)
        stored_burst = channel_mask(handle, "burst_rfi_channel", freq.size)

    if not regions:
        warnings.append(f"{path.name}: attrs['bursts'] is empty; skipped")
        return [], warnings

    nsamp = iquv.shape[1]
    valid_regions: list[tuple[int, BurstRegion]] = []
    for burst_idx, region in enumerate(regions):
        ts = max(0, min(nsamp, int(region["time_start"])))
        te = max(0, min(nsamp, int(region["time_end"])))
        fs = max(0, min(freq.size, int(region["freq_start"])))
        fe = max(0, min(freq.size, int(region["freq_end"])))
        if te <= ts or fe <= fs:
            warnings.append(
                f"{path.name} burst {burst_idx}: empty/clipped region; skipped"
            )
            continue
        clean_region = cast(BurstRegion, dict(region))
        clean_region.update(
            {"time_start": ts, "time_end": te, "freq_start": fs, "freq_end": fe}
        )
        valid_regions.append((burst_idx, clean_region))

    if not valid_regions:
        return [], warnings

    noise_mask = np.ones(nsamp, dtype=bool)
    for _, region in valid_regions:
        noise_mask[int(region["time_start"]) : int(region["time_end"])] = False
    if np.count_nonzero(noise_mask) < 3:
        raise ValueError(f"{path}: fewer than three off-pulse time samples")

    # RM 搜索实际使用 I/Q/U。为坚持“只做整通道屏蔽、不做逐像素掩码”的
    # 契约，只要这三路中任一时间采样出现 NaN/inf，就排除整个频率通道。
    # 这样 burst 区域内的 inf 不会在 ``nan_to_num`` 中变成浮点最大值并污染
    # 法拉第变换；同时仍保留至少三个有效 off-pulse 样本的结构性检查。
    required_stokes    = iquv[:3]
    finite_noise_count = np.sum(np.isfinite(required_stokes[:, noise_mask, :]), axis=1)
    nonfinite_channel = np.any(finite_noise_count < 3, axis=0) | np.any(
        ~np.isfinite(required_stokes), axis=(0, 1)
    )
    with py_warnings.catch_warnings():
        py_warnings.filterwarnings(
            "ignore",
            message  = "All-NaN slice encountered",
            category = RuntimeWarning,
        )
        baseline = np.nanmedian(iquv[:, noise_mask, :], axis=1, keepdims=True)
    iquv = iquv - baseline

    if stored_masks_only:
        recalculated = np.zeros(freq.size, dtype=bool)
        robust       = np.zeros(freq.size, dtype=bool)
    else:
        rfi_work = np.nan_to_num(iquv, nan=0.0, posinf=0.0, neginf=0.0)
        recalculated, robust = recompute_channel_rfi(
            rfi_work,
            noise_mask,
            fft          = rfi_fft,
            sigma        = rfi_channel_sigma,
            local_window = rfi_channel_window,
            grow         = rfi_channel_grow,
        )
    final_rfi = stored_cal | stored_burst | recalculated | robust | nonfinite_channel

    components: list[BurstRMData] = []
    for burst_idx, region in valid_regions:
        good = np.zeros(freq.size, dtype=bool)
        good[int(region["freq_start"]) : int(region["freq_end"])] = True
        good &= ~final_rfi
        if freq_min is not None:
            good &= freq >= freq_min
        if freq_max is not None:
            good &= freq <= freq_max
        if np.count_nonzero(good) < min_channels:
            warnings.append(
                f"{path.name} burst {burst_idx}: only "
                f"{np.count_nonzero(good)} good channels; skipped"
            )
            continue

        time_mask, time_info, time_snr_profile = i_only_time_candidates(
            iquv[0],
            good,
            noise_mask,
            region,
            min_snr=min_time_snr,
        )
        times = np.flatnonzero(time_mask)
        if times.size == 0:
            warnings.append(
                f"{path.name} burst {burst_idx}: empty I-only time gate; skipped"
            )
            continue

        # ``good`` 已排除任何含非有限 I/Q/U 的通道；显式指定正负无穷的
        # 回退值仍作为防御性保护，确保传给 RM 合成的数组始终有限。
        i_on = np.nan_to_num(iquv[0, times][:, good], nan=0.0, posinf=0.0, neginf=0.0)
        q_on = np.nan_to_num(iquv[1, times][:, good], nan=0.0, posinf=0.0, neginf=0.0)
        u_on = np.nan_to_num(iquv[2, times][:, good], nan=0.0, posinf=0.0, neginf=0.0)
        q_noise = np.nan_to_num(
            iquv[1, noise_mask][:, good], nan=0.0, posinf=0.0, neginf=0.0
        )
        u_noise = np.nan_to_num(
            iquv[2, noise_mask][:, good], nan=0.0, posinf=0.0, neginf=0.0
        )
        variance_channel = robust_channel_variance(q_noise) + robust_channel_variance(
            u_noise
        )
        variance_one_time = float(np.sum(variance_channel))
        if not np.isfinite(variance_one_time) or variance_one_time <= 0:
            warnings.append(
                f"{path.name} burst {burst_idx}: invalid Q/U noise variance; skipped"
            )
            continue

        freq_good = freq[good]
        wave2     = (C_M_S / (freq_good * 1e6)) ** 2
        components.append(
            BurstRMData(
                component_id            = component_id(path.name, burst_idx),
                file_name               = path.name,
                file_path               = path,
                burst_idx               = burst_idx,
                peak_snr                = float(time_info["peak_snr"]),
                time_indices            = times,
                time_snr                = time_snr_profile[times],
                freq_mhz                = freq_good,
                wave2_m2                = wave2,
                p_on                    = q_on + 1j * u_on,
                p_noise                 = q_noise + 1j * u_noise,
                i_time_total            = np.sum(i_on, axis=1),
                i_total                 = float(np.sum(i_on)),
                noise_variance_one_time = variance_one_time,
                stored_cal_rfi_count    = int(np.count_nonzero(stored_cal)),
                stored_burst_rfi_count  = int(np.count_nonzero(stored_burst)),
                recalculated_rfi_count  = int(np.count_nonzero(recalculated)),
                robust_rfi_count        = int(np.count_nonzero(robust)),
                nonfinite_rfi_count     = int(np.count_nonzero(nonfinite_channel)),
                final_rfi_count         = int(np.count_nonzero(final_rfi)),
            )
        )
    return components, warnings


def load_file_components_worker(
    payload: tuple[Path, LoadFileOptions],
) -> tuple[list[BurstRMData], list[str]]:
    """适配 ``spawn`` 多进程的顶层入口，用于并行读取 H5 和分析通道 RFI。"""
    path, options = payload
    return load_file_components(path, **options)


def curve_weights(
    bursts: list[BurstRMData],
    mode: str,
    max_weight_ratio: float,
) -> np.ndarray:
    """生成 ``linear_degree_stack`` 的预先固定权重。

    ``equal`` 为每个爆发赋权 1；``peak-snr`` 使用分量峰值 I SNR，但将权重限制
    在中位值的 ``1/max_weight_ratio`` 至 ``max_weight_ratio`` 范围内，再除以
    中位值归一化，防止单个极亮爆发完全支配叠加曲线。
    """
    if mode == "equal":
        return np.ones(len(bursts), dtype=np.float64)
    if mode != "peak-snr":
        raise ValueError(f"Unknown curve-weighting mode: {mode}")
    raw    = np.asarray([max(burst.peak_snr, np.finfo(float).tiny) for burst in bursts])
    median = float(np.median(raw))
    low    = median / max_weight_ratio
    high   = median * max_weight_ratio
    return np.clip(raw, low, high) / median


def individual_rm_curves(
    burst: BurstRMData,
    rm_grid: np.ndarray,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    """为单个爆发计算两种 RM 曲线及逐时间点功率。

    ``linear_degree`` 在每个 RM 上求 ``Σ_time |F| / |Σ I|``；
    ``time_pa_power_samples`` 为每个时间点的 ``|F|²`` 除以单时间 Q/U 噪声方差，
    再沿时间求和得到主要统计量 ``time_pa_power``。后者保留不同时间点各自的 PA，
    不要求先对齐复线偏振相位。
    """
    faraday = complex_faraday_transform(
        burst.p_on,
        burst.wave2_m2,
        rm_grid,
        chunk_size   = chunk_size,
        output_dtype = np.complex128,
    )
    variance = max(burst.noise_variance_one_time, np.finfo(float).tiny)
    linear_degree = np.sum(np.abs(faraday), axis=0) / max(
        abs(burst.i_total), np.finfo(float).tiny
    )
    time_pa_power_samples = np.abs(faraday) ** 2 / variance
    time_pa_power         = np.sum(time_pa_power_samples, axis=0)
    return {
        "linear_degree": linear_degree,
        "time_pa_power": time_pa_power,
        "time_pa_power_samples": time_pa_power_samples,
    }


def combine_curves(
    bursts: list[BurstRMData],
    curves: dict[str, dict[str, np.ndarray]],
    weights: np.ndarray,
) -> dict[str, np.ndarray]:
    """跨爆发合并两种 RM 统计曲线。

    主要 ``time_pa_power`` 直接相加各分量已按噪声归一化的功率；验证用
    ``linear_degree_stack`` 则先分别做稳健标准化，再按固定权重求和，并除以
    ``sqrt(sum(weight²))`` 保持不同成员数下尺度近似可比。
    """
    time_pa_power = sum(curves[burst.component_id]["time_pa_power"] for burst in bursts)
    denominator   = math.sqrt(float(np.sum(weights**2)))
    linear_degree_stack = sum(
        weight * standardized_curve(curves[burst.component_id]["linear_degree"])
        for burst, weight in zip(bursts, weights, strict=True)
    ) / max(denominator, np.finfo(float).tiny)
    return {
        "time_pa_power": np.asarray(time_pa_power, dtype=np.float64),
        "linear_degree_stack": np.asarray(linear_degree_stack, dtype=np.float64),
    }


def observed_curves_on_grid(
    bursts: list[BurstRMData],
    fine_grid: np.ndarray,
    fine_individual: dict[str, dict[str, np.ndarray]],
    fine_combined: dict[str, np.ndarray],
    target_grid: np.ndarray,
    weights: np.ndarray,
) -> dict[str, np.ndarray]:
    """把精细网格上的实测曲线映射到零分布使用的较粗 RM 网格。

    功率曲线可直接线性插值；线偏振度验证曲线必须先逐爆发插值、在目标网格上
    重新稳健标准化，再按原固定权重叠加，才能与零试验采用完全相同的统计定义。
    """
    time_pa_power = np.interp(target_grid, fine_grid, fine_combined["time_pa_power"])
    denominator   = math.sqrt(float(np.sum(weights**2)))
    linear_degree_stack = sum(
        weight
        * standardized_curve(
            np.interp(
                target_grid,
                fine_grid,
                fine_individual[burst.component_id]["linear_degree"],
            )
        )
        for burst, weight in zip(bursts, weights, strict=True)
    ) / max(denominator, np.finfo(float).tiny)
    return {
        "time_pa_power": time_pa_power,
        "linear_degree_stack": np.asarray(linear_degree_stack, dtype=np.float64),
    }


def peak_in_window(
    rm_grid: np.ndarray,
    curve: np.ndarray,
    window: SearchWindow,
) -> tuple[float, float, float]:
    """在指定窗口内找到曲线最大值，返回峰值 RM、原始统计量和稳健 z 值。"""
    inside  = (rm_grid >= window.low) & (rm_grid <= window.high)
    indices = np.flatnonzero(inside)
    if indices.size == 0:
        raise ValueError(f"No RM samples inside test window {window.name}")
    local       = np.asarray(curve)[inside]
    local_index = int(np.nanargmax(local))
    index       = int(indices[local_index])
    z_curve     = standardized_curve(curve)
    return (
        float(rm_grid[index]),
        float(curve[index]),
        float(z_curve[index]),
    )


def precompute_noise_transforms(
    bursts: list[BurstRMData],
    rm_grid: np.ndarray,
    pool_size: int,
    rng: np.random.Generator,
    chunk_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """预计算各分量脉冲外样本的法拉第变换，供所有零试验重复抽样。

    ``pool_size=0`` 使用全部脉冲外样本；否则随机取不超过上限的样本，但至少要
    容纳该爆发真实选中的时间点数。返回变换缓存和所用原始脉冲外索引，后者写入
    归档以保证抽样池可追溯。
    """
    transforms: dict[str, np.ndarray]   = {}
    pool_indices: dict[str, np.ndarray] = {}
    for burst in bursts:
        available = burst.p_noise.shape[0]
        requested = available if pool_size == 0 else min(pool_size, available)
        requested = max(requested, burst.n_time)
        requested = min(requested, available)
        if requested < burst.n_time:
            raise ValueError(
                f"{burst.component_id}: only {available} off-pulse samples for "
                f"a {burst.n_time}-sample burst gate"
            )
        if requested == available:
            chosen = np.arange(available, dtype=int)
        else:
            chosen = np.sort(rng.choice(available, size=requested, replace=False))
        pool_indices[burst.component_id] = chosen
        transforms[burst.component_id] = complex_faraday_transform(
            burst.p_noise[chosen],
            burst.wave2_m2,
            rm_grid,
            chunk_size   = chunk_size,
            output_dtype = np.complex64,
        )
        print(
            f"  null pool {burst.component_id}: {requested}/{available} "
            "off-pulse samples transformed",
            flush=True,
        )
    return transforms, pool_indices


def null_trial_curves(
    bursts: list[BurstRMData],
    noise_transforms: dict[str, np.ndarray],
    weights: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """执行一次脉冲外零试验，并按实测数据的定义合并两种 RM 曲线。

    每个分量从自己的缓存池中无放回抽取与真实爆发相同数量的时间点，因此保留
    该分量的频率覆盖、通道掩码和样本数；随后使用相同噪声归一化及固定权重。
    """
    time_pa_power: np.ndarray | None = None
    linear_stack: np.ndarray | None  = None
    for burst, weight in zip(bursts, weights, strict=True):
        pool         = noise_transforms[burst.component_id]
        chosen       = rng.choice(pool.shape[0], size=burst.n_time, replace=False)
        faraday      = pool[chosen]
        variance     = max(burst.noise_variance_one_time, np.finfo(float).tiny)
        time_curve   = np.sum(np.abs(faraday) ** 2, axis=0) / variance
        linear_curve = np.sum(np.abs(faraday), axis=0)
        linear_z     = weight * standardized_curve(linear_curve)
        time_pa_power = (
            time_curve if time_pa_power is None else time_pa_power + time_curve
        )
        linear_stack = linear_z if linear_stack is None else linear_stack + linear_z
    assert time_pa_power is not None
    assert linear_stack is not None
    linear_stack /= max(math.sqrt(float(np.sum(weights**2))), np.finfo(float).tiny)
    return {
        "time_pa_power": np.asarray(time_pa_power, dtype=np.float64),
        "linear_degree_stack": np.asarray(linear_stack, dtype=np.float64),
    }


def run_empirical_null(
    bursts: list[BurstRMData],
    observed_curves: dict[str, np.ndarray],
    rm_grid: np.ndarray,
    windows: list[SearchWindow],
    weights: np.ndarray,
    *,
    n_null: int,
    pool_size: int,
    seed: int,
    chunk_size: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """构造经过 RM 搜索窗口校正的经验零分布并计算经验 p 值。

    先记录实测曲线在每个方法/窗口中的最大值；随后重复从脉冲外样本生成同结构
    曲线，每次都在整个窗口内取最大值，因此自然包含多 RM 网格点搜索的试验因子。
    同时保存原始最大功率和“曲线内稳健标准化后”的最大对比度。主要经验显著性
    使用后者，并采用 ``(超过次数 + 1) / (试验数 + 1)`` 避免有限模拟给出零 p 值。

    返回逐方法/窗口摘要表、零分布数组归档，以及每个分量使用的脉冲外池索引。
    """
    observed: dict[tuple[str, str], tuple[float, float, float]] = {}
    raw_maxima: dict[tuple[str, str], np.ndarray]               = {}
    contrast_maxima: dict[tuple[str, str], np.ndarray]          = {}
    for method in ALL_METHODS:
        for window in windows:
            key                  = (method, window.name)
            observed[key]        = peak_in_window(rm_grid, observed_curves[method], window)
            raw_maxima[key]      = np.empty(n_null, dtype=np.float64)
            contrast_maxima[key] = np.empty(n_null, dtype=np.float64)

    pool_rng = np.random.default_rng(seed)
    noise_transforms, pool_indices = precompute_noise_transforms(
        bursts,
        rm_grid,
        pool_size,
        pool_rng,
        chunk_size,
    )
    trial_rng     = np.random.default_rng(seed + 1)
    progress_step = max(1, n_null // 10)
    for trial in range(n_null):
        trial_curves = null_trial_curves(bursts, noise_transforms, weights, trial_rng)
        for method in ALL_METHODS:
            contrast_curve = standardized_curve(trial_curves[method])
            for window in windows:
                _, raw_statistic, _ = peak_in_window(
                    rm_grid, trial_curves[method], window
                )
                _, contrast_statistic, _ = peak_in_window(
                    rm_grid, contrast_curve, window
                )
                raw_maxima[(method, window.name)][trial]      = raw_statistic
                contrast_maxima[(method, window.name)][trial] = contrast_statistic
        if (trial + 1) % progress_step == 0 or trial + 1 == n_null:
            print(f"  null trials {trial + 1}/{n_null}", flush=True)

    rows: list[dict[str, object]] = []
    for method in ALL_METHODS:
        for window in windows:
            key = (method, window.name)
            observed_rm, observed_statistic, observed_z = observed[key]
            raw_values = raw_maxima[key]
            contrast_values = contrast_maxima[key]
            raw_exceedances = int(np.count_nonzero(raw_values >= observed_statistic))
            contrast_exceedances = int(np.count_nonzero(contrast_values >= observed_z))
            rows.append(
                {
                    "method": method,
                    "method_role": (
                        "requested" if method in PRIMARY_METHODS else "diagnostic"
                    ),
                    "window": window.name,
                    "window_rm_min": window.low,
                    "window_rm_max": window.high,
                    "null_grid_peak_rm": observed_rm,
                    "observed_null_grid_peak_statistic": observed_statistic,
                    "observed_null_grid_peak_robust_z": observed_z,
                    "raw_null_exceedances": raw_exceedances,
                    "contrast_null_exceedances": contrast_exceedances,
                    # 通用显著性字段使用 RM 曲线对比度零分布。
                    "null_exceedances": contrast_exceedances,
                    "n_null": n_null,
                    "empirical_p_raw_power": float(
                        (raw_exceedances + 1) / (n_null + 1)
                    ),
                    "empirical_p_rm_contrast": float(
                        (contrast_exceedances + 1) / (n_null + 1)
                    ),
                    "empirical_p": float((contrast_exceedances + 1) / (n_null + 1)),
                    "raw_null_p95": float(np.percentile(raw_values, 95)),
                    "raw_null_p99": float(np.percentile(raw_values, 99)),
                    "contrast_null_p95": float(np.percentile(contrast_values, 95)),
                    "contrast_null_p99": float(np.percentile(contrast_values, 99)),
                    "null_p95_max_statistic": float(np.percentile(contrast_values, 95)),
                    "null_p99_max_statistic": float(np.percentile(contrast_values, 99)),
                }
            )
    pool_archive = {
        component: indices.astype(np.int32)
        for component, indices in pool_indices.items()
    }
    null_archive: dict[str, np.ndarray] = {}
    for (method, window_name), values in raw_maxima.items():
        null_archive[f"raw__{method}__{window_name}"] = values
    for (method, window_name), values in contrast_maxima.items():
        null_archive[f"contrast__{method}__{window_name}"] = values
        # 标准归档键与 ``empirical_p`` 使用相同的最大稳健 z 分布。
        null_archive[f"{method}__{window_name}"]           = values
    return pd.DataFrame(rows), null_archive, pool_archive


def fine_grid_summary(
    rm_grid: np.ndarray,
    combined: dict[str, np.ndarray],
    windows: list[SearchWindow],
) -> pd.DataFrame:
    """汇总精细 RM 网格上每种方法、每个窗口的实测峰值。"""
    rows: list[dict[str, object]] = []
    for method in ALL_METHODS:
        for window in windows:
            rm, statistic, robust_z = peak_in_window(rm_grid, combined[method], window)
            rows.append(
                {
                    "method": method,
                    "method_role": "requested",
                    "window": window.name,
                    "fine_grid_peak_rm": rm,
                    "fine_grid_peak_statistic": statistic,
                    "fine_grid_peak_robust_z": robust_z,
                }
            )
    return pd.DataFrame(rows)


def leave_one_burst_out_summary(
    bursts: list[BurstRMData],
    individual: dict[str, dict[str, np.ndarray]],
    weights: np.ndarray,
    rm_grid: np.ndarray,
    window: SearchWindow,
    full_combined: dict[str, np.ndarray],
    rm_rmsf_fwhm: float,
) -> pd.DataFrame:
    """逐个剔除入选分量，测量共同 RM 峰对单个爆发的敏感程度。

    每次用剩余成员和对应权重重新合并曲线，记录相对完整样本峰值的偏移；偏移再
    除以 RM spread function（RMSF）的半高全宽。``stable_within_one_rmsf``
    表示剔除该分量后峰位移动不超过一个 RMSF 分辨单元。
    """
    columns = [
        "excluded_component",
        "excluded_n_time_samples",
        "excluded_i_snr_squared_sum",
        "remaining_components",
        "method",
        "full_peak_rm",
        "loo_peak_rm",
        "loo_peak_robust_z",
        "peak_shift_rad_m2",
        "peak_shift_rmsf",
        "stable_within_one_rmsf",
    ]
    if len(bursts) < 2:
        return pd.DataFrame(columns=columns)

    full_peaks = {
        method: peak_in_window(rm_grid, full_combined[method], window)[0]
        for method in PRIMARY_METHODS
    }
    rows: list[dict[str, object]] = []
    for excluded_index, excluded in enumerate(bursts):
        retained = [
            burst for index, burst in enumerate(bursts) if index != excluded_index
        ]
        retained_weights = np.delete(weights, excluded_index)
        retained_curves = {
            burst.component_id: individual[burst.component_id] for burst in retained
        }
        loo_combined = combine_curves(
            retained,
            retained_curves,
            retained_weights,
        )
        for method in PRIMARY_METHODS:
            loo_rm, _, loo_z = peak_in_window(rm_grid, loo_combined[method], window)
            shift            = abs(float(loo_rm) - float(full_peaks[method]))
            shift_rmsf       = shift / max(float(rm_rmsf_fwhm), np.finfo(float).tiny)
            rows.append(
                {
                    "excluded_component": excluded.component_id,
                    "excluded_n_time_samples": excluded.n_time,
                    "excluded_i_snr_squared_sum": float(np.sum(excluded.time_snr**2)),
                    "remaining_components": len(retained),
                    "method": method,
                    "full_peak_rm": float(full_peaks[method]),
                    "loo_peak_rm": float(loo_rm),
                    "loo_peak_robust_z": float(loo_z),
                    "peak_shift_rad_m2": shift,
                    "peak_shift_rmsf": shift_rmsf,
                    "stable_within_one_rmsf": bool(shift_rmsf <= 1.0),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def write_plot(
    output_path: Path,
    rm_grid: np.ndarray,
    bursts: list[BurstRMData],
    individual: dict[str, dict[str, np.ndarray]],
    combined: dict[str, np.ndarray],
    windows: list[SearchWindow],
    null_table: pd.DataFrame | None,
    null_maxima: dict[str, np.ndarray] | None,
    rm_rmsf_fwhm: float,
    global_time_info: GlobalTimeInfo,
    run_label: str | None = None,
) -> None:
    """绘制并保存共同 RM 搜索的综合诊断图。

    图中依次展示逐时间点的标准化 RM 功率热图、两种合并统计量的全范围曲线、
    主要峰附近的 RMSF 尺度放大图，以及两种方法各自的脉冲外最大值零分布。
    观测标签、入选样本数、频段、网格分辨率和经验 p 值一并写入图面，便于图片
    脱离其他产物后仍能独立解释。若 ``n_null=0``，零分布面板明确显示为禁用。
    """
    colors = {
        "time_pa_power": "#11845B",
        "linear_degree_stack": "#E76F2E",
    }
    labels = {
        "time_pa_power": "Time-resolved PA power (primary)",
        "linear_degree_stack": "Linear-degree stack (validation)",
    }
    primary_window = select_primary_plot_window(windows, rm_grid)
    rm_step        = float(np.median(np.diff(rm_grid)))
    full_width     = float(rm_grid[-1] - rm_grid[0])
    method_z = {
        method: standardized_curve(combined[method]) for method in PRIMARY_METHODS
    }
    method_peaks = {
        method: peak_in_window(rm_grid, combined[method], primary_window)
        for method in PRIMARY_METHODS
    }

    def null_row(method: str) -> pd.Series | None:
        """取主要搜索窗口中指定方法的零分布摘要行；不可用时返回 None。"""
        if null_table is None:
            return None
        selected = null_table[
            (null_table["method"] == method)
            & (null_table["window"] == primary_window.name)
        ]
        return None if selected.empty else selected.iloc[0]

    def p_text(method: str) -> str:
        """把指定方法的经验 p 值和超过次数格式化成图例文字。"""
        row = null_row(method)
        if row is None:
            return "null disabled"
        return (
            f"p={float(cast(Any, row['empirical_p'])):.4g} "
            f"({int(cast(Any, row['null_exceedances']))}/"
            f"{int(cast(Any, row['n_null']))} exceed)"
        )

    observation_matches = [
        re.match(r"(?P<source>.+?)-(?P<date>\d{8})-", burst.file_name)
        for burst in bursts
    ]
    matched_dates = sorted(
        {match.group("date") for match in observation_matches if match is not None}
    )
    matched_sources = {
        match.group("source") for match in observation_matches if match is not None
    }
    normalized_sources = {re.sub(r"S\d+$", "", source) for source in matched_sources}
    if run_label:
        observation = run_label
    elif len(matched_dates) > 1 and len(normalized_sources) == 1:
        source = next(iter(normalized_sources))
        observation = (
            f"{source}  {matched_dates[0]}–{matched_dates[-1]} "
            f"({len(matched_dates)} dates)"
        )
    elif len(matched_dates) == 1 and len(matched_sources) == 1:
        observation = f"{next(iter(matched_sources))}  {matched_dates[0]}"
    else:
        observation = Path(bursts[0].file_name).stem
    band_min   = min(float(burst.freq_mhz.min()) for burst in bursts)
    band_max   = max(float(burst.freq_mhz.max()) for burst in bursts)
    total_time = sum(burst.n_time for burst in bursts)
    n_null = (
        int(null_table["n_null"].iloc[0])
        if null_table is not None and not null_table.empty
        else 0
    )

    with plt.rc_context(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.facecolor": "#FAFBFC",
            "figure.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.7,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
        }
    ):
        fig = plt.figure(figsize=(17, 14), constrained_layout=True)
        grid = fig.add_gridspec(
            3,
            2,
            height_ratios=(1.15, 1.0, 0.95),
        )
        heat_axis = fig.add_subplot(grid[0, :])
        full_axis = fig.add_subplot(grid[1, 0])
        zoom_axis = fig.add_subplot(grid[1, 1])
        null_axes = (
            fig.add_subplot(grid[2, 0]),
            fig.add_subplot(grid[2, 1]),
        )

        fig.suptitle(
            (
                f"{observation} — synchronized time-sample RM stack\n"
                f"{len(bursts)} components · {total_time}/"
                f"{global_time_info['candidate_sample_count']} global-I "
                f"time samples · "
                f"I-S/N cutoff={global_time_info['selected_snr_min']:.1f} · "
                f"{band_min:.1f}–{band_max:.1f} MHz · "
                rf"$\Delta$RM={rm_step:.1f} rad m$^{{-2}}$ · "
                rf"RMSF FWHM={rm_rmsf_fwhm:.1f} rad m$^{{-2}}$ · "
                f"{n_null} null trials · channel masks only (pixel mask off)"
            ),
            fontsize    = 17,
            linespacing = 1.45,
        )

        sample_curves: list[np.ndarray]   = []
        sample_labels: list[str]          = []
        component_centers: list[float]    = []
        component_labels: list[str]       = []
        component_boundaries: list[float] = []
        row_offset                        = 0
        for burst, match in zip(bursts, observation_matches, strict=True):
            sample_power   = individual[burst.component_id]["time_pa_power_samples"]
            expected_shape = (burst.n_time, rm_grid.size)
            if sample_power.shape != expected_shape:
                raise ValueError(
                    f"{burst.component_id}: per-time TimePA power shape "
                    f"{sample_power.shape} does not match {expected_shape}"
                )
            date_prefix = (
                f"{match.group('date')}  "
                if len(matched_dates) > 1 and match is not None
                else ""
            )
            for sample_index, sample_snr, curve in zip(
                burst.time_indices,
                burst.time_snr,
                sample_power,
                strict=True,
            ):
                sample_curves.append(standardized_curve(curve))
                sample_labels.append(
                    f"{date_prefix}{burst.component_id}:t{int(sample_index)}   "
                    f"I-S/N={float(sample_snr):.1f}"
                )
            component_centers.append(row_offset + 0.5 * (burst.n_time - 1))
            component_labels.append(
                f"{date_prefix}{burst.component_id}   "
                f"$N_t$={burst.n_time}, "
                f"I-S/N={burst.time_snr.min():.1f}–"
                f"{burst.time_snr.max():.1f}"
            )
            row_offset += burst.n_time
            if row_offset < total_time:
                component_boundaries.append(row_offset - 0.5)

        individual_matrix = np.vstack(sample_curves)
        clipped_low       = -2.5
        clipped_high      = 6.0
        normalization = TwoSlopeNorm(
            vmin    = clipped_low,
            vcenter = 0.0,
            vmax    = clipped_high,
        )
        image = heat_axis.imshow(
            np.clip(individual_matrix, clipped_low, clipped_high),
            aspect        = "auto",
            origin        = "upper",
            extent        = (
                float(rm_grid[0]),
                float(rm_grid[-1]),
                total_time - 0.5,
                -0.5,
            ),
            cmap          = "RdBu_r",
            norm          = normalization,
            interpolation = "nearest",
        )
        heat_axis.grid(False)
        heat_axis.set_title(
            "Per-time-sample TimePA RM-power response "
            "(each row standardized independently; grouped by burst)"
        )
        heat_axis.set_xlabel(r"RM (rad m$^{-2}$)")
        heat_axis.set_ylabel("Selected time sample")
        if total_time <= 30:
            heat_axis.set_yticks(np.arange(total_time))
            heat_axis.set_yticklabels(sample_labels, fontsize=7)
        else:
            heat_axis.set_yticks(component_centers)
            heat_axis.set_yticklabels(component_labels, fontsize=7)
        for boundary in component_boundaries:
            heat_axis.axhline(
                boundary,
                color     = "black",
                linewidth = 0.8,
                alpha     = 0.65,
                zorder    = 3,
            )
        window_mask    = (rm_grid >= primary_window.low) & (rm_grid <= primary_window.high)
        window_indices = np.flatnonzero(window_mask)
        for row_index, curve in enumerate(individual_matrix):
            peak_index = window_indices[np.argmax(curve[window_mask])]
            heat_axis.scatter(
                rm_grid[peak_index],
                row_index,
                s         = 18,
                marker    = "o",
                facecolor = "none",
                edgecolor = "white",
                linewidth = 0.8,
                zorder    = 4,
            )
        primary_peak_rm = method_peaks["time_pa_power"][0]
        heat_axis.axvline(
            primary_peak_rm,
            color     = "#FFE66D",
            linestyle = "--",
            linewidth = 1.6,
            label     = f"TimePA combined peak: {primary_peak_rm:.0f}",
        )
        heat_axis.legend(loc="upper right", fontsize=9)
        colorbar = fig.colorbar(
            image,
            ax     = heat_axis,
            pad    = 0.01,
            aspect = 35,
            shrink = 0.92,
        )
        colorbar.set_label(
            "Per-time-sample TimePA robust z (display clipped at −2.5, 6)"
        )

        for method in PRIMARY_METHODS:
            peak_rm, _, peak_z = method_peaks[method]
            full_axis.plot(
                rm_grid,
                method_z[method],
                color     = colors[method],
                linewidth = 1.6,
                label     = (
                    f"{labels[method]}\n"
                    f"RM={peak_rm:.0f}, z={peak_z:.2f}, {p_text(method)}"
                ),
            )
            full_axis.axvline(
                peak_rm,
                color     = colors[method],
                linestyle = ":",
                linewidth = 1.1,
                alpha     = 0.9,
            )
        if max(float(np.nanmax(curve)) for curve in method_z.values()) > 30:
            full_axis.set_yscale("symlog", linthresh=3.0, linscale=1.0)
            full_axis.text(
                0.99,
                0.02,
                "symlog y-scale",
                transform = full_axis.transAxes,
                ha        = "right",
                va        = "bottom",
                fontsize  = 8,
                color     = "0.4",
            )
        full_axis.axhline(0.0, color="0.45", linewidth=0.8)
        full_axis.axhline(3.0, color="0.45", linewidth=0.8, linestyle="--", alpha=0.7)
        for window in windows:
            if (window.high - window.low) < 0.98 * full_width:
                full_axis.axvspan(
                    window.low,
                    window.high,
                    color = "#5B8FF9",
                    alpha = 0.06,
                )
        full_axis.set_title("Combined search statistics — full RM range")
        full_axis.set_xlabel(r"RM (rad m$^{-2}$)")
        full_axis.set_ylabel("Robust z across searched RM")
        full_axis.legend(loc="best", fontsize=8)
        peak_separation = abs(
            method_peaks["time_pa_power"][0] - method_peaks["linear_degree_stack"][0]
        )
        full_axis.text(
            0.02,
            0.02,
            (
                rf"Method peak separation: {peak_separation:.0f} rad m$^{{-2}}$ "
                f"= {peak_separation / rm_rmsf_fwhm:.2f} RMSF"
            ),
            transform = full_axis.transAxes,
            ha        = "left",
            va        = "bottom",
            fontsize  = 8,
            bbox      = {
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "0.8",
                "alpha": 0.9,
            },
        )

        zoom_half_width = max(4.0 * rm_rmsf_fwhm, 10.0 * rm_step)
        zoom_low        = max(float(rm_grid[0]), primary_peak_rm - zoom_half_width)
        zoom_high       = min(float(rm_grid[-1]), primary_peak_rm + zoom_half_width)
        zoom_mask       = (rm_grid >= zoom_low) & (rm_grid <= zoom_high)
        for method in PRIMARY_METHODS:
            zoom_axis.plot(
                rm_grid[zoom_mask],
                method_z[method][zoom_mask],
                color     = colors[method],
                linewidth = 1.8,
                label     = labels[method],
            )
        zoom_axis.axvspan(
            primary_peak_rm - 0.5 * rm_rmsf_fwhm,
            primary_peak_rm + 0.5 * rm_rmsf_fwhm,
            color = colors["time_pa_power"],
            alpha = 0.08,
            label = "One RMSF FWHM",
        )
        zoom_axis.axvline(
            primary_peak_rm,
            color     = colors["time_pa_power"],
            linestyle = "--",
            linewidth = 1.3,
        )
        zoom_axis.axhline(0.0, color="0.45", linewidth=0.8)
        zoom_axis.axhline(3.0, color="0.45", linewidth=0.8, linestyle="--", alpha=0.7)
        if max(float(np.nanmax(curve[zoom_mask])) for curve in method_z.values()) > 30:
            zoom_axis.set_yscale("symlog", linthresh=3.0, linscale=1.0)
        zoom_axis.set_xlim(zoom_low, zoom_high)
        zoom_axis.set_title(
            f"Blind-search peak detail — centered at RM={primary_peak_rm:.0f}"
        )
        zoom_axis.set_xlabel(r"RM (rad m$^{-2}$)")
        zoom_axis.set_ylabel("Robust z across searched RM")
        zoom_axis.legend(loc="best", fontsize=8)

        for axis, method in zip(null_axes, PRIMARY_METHODS, strict=True):
            row = null_row(method)
            if row is None or null_maxima is None:
                axis.grid(False)
                axis.text(
                    0.5,
                    0.5,
                    f"{labels[method]}\nEmpirical null disabled (--n-null 0)",
                    ha        = "center",
                    va        = "center",
                    transform = axis.transAxes,
                    fontsize  = 12,
                )
                axis.set_xticks([])
                axis.set_yticks([])
                continue

            values    = null_maxima[f"{method}__{primary_window.name}"]
            observed  = float(cast(Any, row["observed_null_grid_peak_robust_z"]))
            p95       = float(cast(Any, row["null_p95_max_statistic"]))
            p99       = float(cast(Any, row["null_p99_max_statistic"]))
            ratio     = observed / max(p99, np.finfo(float).tiny)
            use_log_x = observed > 5.0 * p99 and np.all(values > 0) and observed > 0
            if use_log_x:
                lower = max(float(np.min(values)) * 0.95, np.finfo(float).tiny)
                bins  = np.geomspace(lower, observed * 1.08, 46).tolist()
                axis.set_xscale("log")
            else:
                bins = 42
            axis.hist(
                values,
                bins      = bins,
                color     = colors[method],
                alpha     = 0.7,
                edgecolor = "white",
                linewidth = 0.35,
            )
            axis.axvspan(p95, p99, color="#F3C969", alpha=0.2)
            axis.axvline(
                p95,
                color     = "#B7791F",
                linestyle = "--",
                linewidth = 1.0,
                label     = f"null p95={p95:.3g}",
            )
            axis.axvline(
                p99,
                color     = "#9C4221",
                linestyle = ":",
                linewidth = 1.3,
                label     = f"null p99={p99:.3g}",
            )
            axis.axvline(
                observed,
                color     = "#C53030",
                linewidth = 2.0,
                label     = f"observed={observed:.3g} ({ratio:.2f}× p99)",
            )
            detected = float(cast(Any, row["empirical_p"])) <= 0.01
            axis.set_title(
                f"{labels[method]} — off-pulse RM-contrast null",
                color="#147D64" if detected else "0.2",
            )
            axis.set_xlabel(
                "Maximum robust-z RM contrast over blind-search window "
                f"({primary_window.name!r})"
            )
            axis.set_ylabel("Null trials")
            axis.legend(loc="upper right", fontsize=7)
            axis.text(
                0.02,
                0.96,
                (
                    f"{'DETECTED' if detected else 'NOT DETECTED'}\n"
                    f"RM={float(cast(Any, row['null_grid_peak_rm'])):.0f} "
                    "rad m$^{-2}$\n"
                    f"{p_text(method)}"
                ),
                transform = axis.transAxes,
                ha        = "left",
                va        = "top",
                fontsize  = 9,
                color     = "#147D64" if detected else "0.3",
                bbox      = {
                    "boxstyle": "round,pad=0.35",
                    "facecolor": "white",
                    "edgecolor": "#9AE6B4" if detected else "0.8",
                    "alpha": 0.94,
                },
            )

        fig.savefig(
            output_path,
            dpi         = 220,
            bbox_inches = "tight",
            facecolor   = "white",
        )
    plt.close(fig)


def sha256_file(path: Path) -> str:
    """以分块方式计算文件 SHA-256，用于运行清单中的代码版本溯源。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_float_or_none(value: object) -> float | None:
    """把输入转换成可安全写入 JSON 的有限浮点数，缺失或无效时返回 None。"""
    if value is None:
        return None
    try:
        converted = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return converted if np.isfinite(converted) else None


def build_run_summary(
    *,
    args: argparse.Namespace,
    files: list[Path],
    bursts: list[BurstRMData],
    global_time_info: GlobalTimeInfo,
    summary_table: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    primary_window: SearchWindow,
    rm_grid_info: dict[str, float],
    warnings: list[str],
) -> dict[str, object]:
    """构建供批量观测任务读取的稳定、紧凑 ``run_summary.json`` 契约。

    摘要整合输入规模、全局时间选样、实际频率覆盖、RM 网格、两种方法的峰值与
    经验显著性、逐一剔除稳定性和产物文件名。检测状态以 RM 对比度经验 p 值
    ``<= 0.01`` 为强检测，并同时报告两种方法的峰是否在一个 RMSF 内一致。
    ``both_methods`` 描述的是两个统计量，不要求最终时间采样来自多个爆发；输入
    与入选分量数另有独立字段，供调用方判断实际叠加范围。
    所有非有限浮点数先转为 ``None``，确保使用 ``allow_nan=False`` 仍能写出
    标准 JSON。
    """
    detection_threshold                     = 0.01
    method_results: dict[str, MethodResult] = {}
    for method in PRIMARY_METHODS:
        selected = summary_table[
            (summary_table["method"] == method)
            & (summary_table["window"] == primary_window.name)
        ]
        if selected.empty:
            continue
        row = selected.iloc[0]
        p_value = finite_float_or_none(
            row.get("empirical_p_rm_contrast", row.get("empirical_p"))
        )
        method_loo = leave_one_out[leave_one_out["method"] == method]
        if method_loo.empty:
            loo_stable_fraction = None
            loo_max_shift_rmsf  = None
        else:
            loo_stable_fraction = float(
                cast(pd.Series, method_loo["stable_within_one_rmsf"])
                .astype(bool)
                .mean()
            )
            loo_max_shift_rmsf = finite_float_or_none(
                method_loo["peak_shift_rmsf"].max()
            )
        method_results[method] = {
            "peak_rm_rad_m2": finite_float_or_none(row.get("fine_grid_peak_rm")),
            "peak_statistic": finite_float_or_none(row.get("fine_grid_peak_statistic")),
            "peak_robust_z": finite_float_or_none(row.get("fine_grid_peak_robust_z")),
            "empirical_p_rm_contrast": p_value,
            "null_exceedances": (
                int(cast(Any, row["contrast_null_exceedances"]))
                if "contrast_null_exceedances" in row
                and pd.notna(row["contrast_null_exceedances"])
                else None
            ),
            "detected_p_le_0_01": (
                bool(p_value <= detection_threshold) if p_value is not None else None
            ),
            "leave_one_out_stable_fraction": loo_stable_fraction,
            "leave_one_out_max_shift_rmsf": loo_max_shift_rmsf,
        }

    method_rms: list[float] = []
    for method in PRIMARY_METHODS:
        if method in method_results:
            peak_rm = method_results[method]["peak_rm_rad_m2"]
            if peak_rm is not None:
                method_rms.append(peak_rm)
    if len(method_rms) == len(PRIMARY_METHODS):
        peak_separation      = abs(float(method_rms[0]) - float(method_rms[1]))
        peak_separation_rmsf = peak_separation / float(rm_grid_info["rmsf_fwhm"])
        methods_consistent   = peak_separation_rmsf <= 1.0
    else:
        peak_separation      = None
        peak_separation_rmsf = None
        methods_consistent   = None

    detections: list[bool] = []
    marginal: list[bool]   = []
    for result in method_results.values():
        detected = result["detected_p_le_0_01"]
        if detected is not None:
            detections.append(detected)
        empirical_p = result["empirical_p_rm_contrast"]
        if empirical_p is not None:
            marginal.append(empirical_p <= 0.05)
    if not detections:
        status = "null_not_run"
    elif len(detections) == len(PRIMARY_METHODS) and all(detections):
        status = (
            "robust_both_methods"
            if methods_consistent
            else "both_methods_peak_disagreement"
        )
    elif any(detections):
        status = (
            "one_method_detected_consistent"
            if methods_consistent
            else "one_method_detected_peak_disagreement"
        )
    elif len(marginal) == len(PRIMARY_METHODS) and all(marginal) and methods_consistent:
        status = "marginal_both_methods"
    else:
        status = "no_robust_detection"

    selected_sample_snr = np.concatenate(
        [burst.time_snr.astype(np.float64, copy=False) for burst in bursts]
    )
    selected_peak_snr = np.asarray(
        [burst.peak_snr for burst in bursts], dtype=np.float64
    )
    channel_counts   = np.asarray([burst.n_channel for burst in bursts], dtype=np.int64)
    i_snr_equivalent = math.sqrt(float(global_time_info["selected_i_snr_squared_sum"]))
    return {
        "schema_version": 1,
        "run_label": args.run_label,
        "status": status,
        "input": {
            "cal_dir": str(args.cal_dir.resolve()),
            "h5_file_count": len(files),
            "selected_h5_file_count": len({str(burst.file_path) for burst in bursts}),
            "candidate_component_count": int(
                global_time_info["candidate_component_count"]
            ),
            "selected_component_count": len(bursts),
            "selected_component_ids": [burst.component_id for burst in bursts],
        },
        "time_selection": {
            "strategy": global_time_info["strategy"],
            "candidate_sample_count": int(global_time_info["candidate_sample_count"]),
            "selected_sample_count": int(global_time_info["selected_sample_count"]),
            "selected_i_snr_cutoff": finite_float_or_none(
                global_time_info["selected_snr_min"]
            ),
            "selected_i_snr_median": float(np.median(selected_sample_snr)),
            "selected_i_snr_max": float(np.max(selected_sample_snr)),
            "selected_i_snr_squared_fraction": finite_float_or_none(
                global_time_info["selected_i_snr_squared_fraction"]
            ),
            "i_snr_equivalent": i_snr_equivalent,
            "component_peak_i_snr_min": float(np.min(selected_peak_snr)),
            "component_peak_i_snr_median": float(np.median(selected_peak_snr)),
            "component_peak_i_snr_max": float(np.max(selected_peak_snr)),
        },
        "frequency": {
            "requested_min_mhz": args.freq_min,
            "requested_max_mhz": args.freq_max,
            "actual_min_mhz": min(float(burst.freq_mhz.min()) for burst in bursts),
            "actual_max_mhz": max(float(burst.freq_mhz.max()) for burst in bursts),
            "usable_channel_count_min": int(np.min(channel_counts)),
            "usable_channel_count_median": float(np.median(channel_counts)),
            "usable_channel_count_max": int(np.max(channel_counts)),
        },
        "rm_search": {
            "min_rad_m2": args.rm_min,
            "max_rad_m2": args.rm_max,
            "grid_size": int(rm_grid_info["n_rm"]),
            "grid_step_rad_m2": float(rm_grid_info["rm_step"]),
            "rmsf_fwhm_rad_m2": float(rm_grid_info["rmsf_fwhm"]),
            "primary_window": {
                "name": primary_window.name,
                "min_rad_m2": primary_window.low,
                "max_rad_m2": primary_window.high,
            },
            "n_null": args.n_null,
            "curve_weighting": args.curve_weighting,
        },
        "methods": method_results,
        "method_comparison": {
            "peak_separation_rad_m2": peak_separation,
            "peak_separation_rmsf": peak_separation_rmsf,
            "consistent_within_one_rmsf": methods_consistent,
        },
        "warnings": warnings,
        "artifacts": {
            "plot": "burst_sync_rm.png",
            "method_summary": "burst_sync_rm_summary.csv",
            "selected_bursts": "selected_bursts.csv",
            "time_sample_selection": "time_sample_selection.csv",
            "leave_one_out": "leave_one_burst_out.csv",
            "curves": "burst_sync_rm_curves.npz",
        },
    }


def write_no_eligible_result(
    *,
    args: argparse.Namespace,
    files: list[Path],
    loaded_bursts: list[BurstRMData],
    output_dir: Path,
    warnings: list[str],
) -> None:
    """当 Stokes-I 门限未保留任何分量时，写出完整的非错误结果集。

    “没有符合条件的爆发”是有效科学结果而非程序崩溃。本函数因此仍生成状态为
    ``no_eligible_components`` 的 JSON/CSV/文本摘要、空选择表和说明图，并记录
    候选规模、门限、频段、警告及脚本哈希；只有依赖 RM 曲线的压缩包不生成。
    """
    candidate_samples = int(sum(burst.n_time for burst in loaded_bursts))
    candidate_files   = len({str(burst.file_path) for burst in loaded_bursts})
    candidate_snr = np.asarray(
        [burst.peak_snr for burst in loaded_bursts], dtype=np.float64
    )
    if args.freq_min is None and args.freq_max is None:
        frequency_text = "all available frequencies"
    else:
        low            = "band start" if args.freq_min is None else f"{args.freq_min:g}"
        high           = "band end" if args.freq_max is None else f"{args.freq_max:g}"
        frequency_text = f"{low}–{high} MHz"
    message = (
        "No burst component survived the configured Stokes-I selection "
        f"(min component peak S/N={args.min_peak_snr:g}, "
        f"min time-sample S/N={args.min_time_snr:g})."
    )
    result_warnings = [*warnings, message]
    empty_methods = {
        method: {
            "peak_rm_rad_m2": None,
            "peak_statistic": None,
            "peak_robust_z": None,
            "empirical_p_rm_contrast": None,
            "null_exceedances": None,
            "detected_p_le_0_01": None,
            "leave_one_out_stable_fraction": None,
            "leave_one_out_max_shift_rmsf": None,
        }
        for method in PRIMARY_METHODS
    }
    run_summary = {
        "schema_version": 1,
        "run_label": args.run_label,
        "status": "no_eligible_components",
        "input": {
            "cal_dir": str(args.cal_dir.resolve()),
            "h5_file_count": len(files),
            "candidate_h5_file_count": candidate_files,
            "selected_h5_file_count": 0,
            "candidate_component_count": len(loaded_bursts),
            "selected_component_count": 0,
            "selected_component_ids": [],
        },
        "time_selection": {
            "strategy": "global_i_snr2_over_sqrt_n",
            "candidate_sample_count": candidate_samples,
            "selected_sample_count": 0,
            "selected_i_snr_cutoff": None,
            "selected_i_snr_median": None,
            "selected_i_snr_max": None,
            "selected_i_snr_squared_fraction": 0.0,
            "i_snr_equivalent": 0.0,
            "component_peak_i_snr_min": (
                float(np.min(candidate_snr)) if candidate_snr.size else None
            ),
            "component_peak_i_snr_median": (
                float(np.median(candidate_snr)) if candidate_snr.size else None
            ),
            "component_peak_i_snr_max": (
                float(np.max(candidate_snr)) if candidate_snr.size else None
            ),
        },
        "frequency": {
            "requested_min_mhz": args.freq_min,
            "requested_max_mhz": args.freq_max,
            "actual_min_mhz": None,
            "actual_max_mhz": None,
            "usable_channel_count_min": None,
            "usable_channel_count_median": None,
            "usable_channel_count_max": None,
        },
        "rm_search": {
            "min_rad_m2": args.rm_min,
            "max_rad_m2": args.rm_max,
            "grid_size": None,
            "grid_step_rad_m2": None,
            "rmsf_fwhm_rad_m2": None,
            "primary_window": None,
            "n_null": args.n_null,
            "curve_weighting": args.curve_weighting,
        },
        "methods": empty_methods,
        "method_comparison": {
            "peak_separation_rad_m2": None,
            "peak_separation_rmsf": None,
            "consistent_within_one_rmsf": None,
        },
        "warnings": result_warnings,
        "artifacts": {
            "plot": "burst_sync_rm.png",
            "method_summary": "burst_sync_rm_summary.csv",
            "selected_bursts": "selected_bursts.csv",
            "time_sample_selection": "time_sample_selection.csv",
            "leave_one_out": "leave_one_burst_out.csv",
            "curves": None,
        },
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    summary_rows = []
    for method in PRIMARY_METHODS:
        summary_rows.append(
            {
                "run_label": args.run_label,
                "status": "no_eligible_components",
                "n_input_h5_files": len(files),
                "n_candidate_components": len(loaded_bursts),
                "n_members": 0,
                "n_candidate_time_samples": candidate_samples,
                "n_selected_time_samples": 0,
                "i_snr_equivalent": 0.0,
                "requested_frequency_min_mhz": args.freq_min,
                "requested_frequency_max_mhz": args.freq_max,
                "method": method,
                "method_role": "requested",
                "window": "full",
                "fine_grid_peak_rm": None,
                "fine_grid_peak_robust_z": None,
                "empirical_p_rm_contrast": None,
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "burst_sync_rm_summary.csv", index=False
    )
    pd.DataFrame(
        columns=[
            "rank",
            "component_id",
            "file_name",
            "file_path",
            "burst_idx",
            "i_peak_snr",
            "n_time_samples",
        ]
    ).to_csv(output_dir / "selected_bursts.csv", index=False)
    candidate_rows = []
    for burst in loaded_bursts:
        for time_index, sample_snr in zip(
            burst.time_indices, burst.time_snr, strict=True
        ):
            candidate_rows.append(
                {
                    "component_id": burst.component_id,
                    "file_name": burst.file_name,
                    "burst_idx": burst.burst_idx,
                    "time_index": int(time_index),
                    "i_sample_snr": float(sample_snr),
                    "selected": False,
                    "rejection": "component_peak_below_min_peak_snr",
                }
            )
    pd.DataFrame(
        candidate_rows,
        columns=[
            "component_id",
            "file_name",
            "burst_idx",
            "time_index",
            "i_sample_snr",
            "selected",
            "rejection",
        ],
    ).to_csv(output_dir / "time_sample_selection.csv", index=False)
    pd.DataFrame(
        columns=[
            "excluded_component",
            "method",
            "full_peak_rm",
            "loo_peak_rm",
            "peak_shift_rmsf",
            "stable_within_one_rmsf",
        ]
    ).to_csv(output_dir / "leave_one_burst_out.csv", index=False)

    with plt.rc_context({"figure.facecolor": "white", "font.size": 12}):
        figure, axis = plt.subplots(figsize=(11, 5.5))
        axis.axis("off")
        axis.text(
            0.5,
            0.62,
            args.run_label or "Stacked-RM search",
            ha       = "center",
            va       = "center",
            fontsize = 22,
            weight   = "bold",
        )
        axis.text(
            0.5,
            0.42,
            "No eligible Stokes-I burst component",
            ha       = "center",
            va       = "center",
            fontsize = 17,
            color    = "#9C4221",
        )
        axis.text(
            0.5,
            0.25,
            (
                f"{len(files)} input H5 · {len(loaded_bursts)} candidates · "
                f"peak S/N ≥ {args.min_peak_snr:g} · "
                f"sample S/N ≥ {args.min_time_snr:g} · "
                f"{frequency_text}"
            ),
            ha       = "center",
            va       = "center",
            fontsize = 12,
            color    = "0.3",
        )
        figure.savefig(
            output_dir / "burst_sync_rm.png",
            dpi         = 180,
            bbox_inches = "tight",
            facecolor   = "white",
        )
        plt.close(figure)
    (output_dir / "burst_sync_rm_summary.txt").write_text(
        "\n".join(
            [
                "Synchronized time-sample RM stack",
                f"run_label={args.run_label}",
                "status=no_eligible_components",
                message,
                "",
                *result_warnings,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    script_path = Path(__file__).resolve()
    manifest = {
        "script": str(script_path),
        "script_sha256": sha256_file(script_path),
        "run_label": args.run_label,
        "cal_dir": str(args.cal_dir.resolve()),
        "output_dir": str(output_dir),
        "input_files": [str(path) for path in files],
        "status": "no_eligible_components",
        "pixel_mask": "not read or applied",
        "selection": {
            "min_time_snr": args.min_time_snr,
            "min_peak_snr": args.min_peak_snr,
            "freq_min": args.freq_min,
            "freq_max": args.freq_max,
            "min_channels": args.min_channels,
        },
        "warnings": result_warnings,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    """执行从标准定标 H5 到共同 RM 结果集的完整命令行流程。

    主要阶段为：校验参数并创建全新输出目录；串行或多进程加载 H5、减基线并
    合并通道掩码；执行分量峰值门限和全观测 Stokes-I 时间选样；自动构造精细与
    零分布 RM 网格；计算并合并两种统计曲线；运行脉冲外经验零试验和逐一剔除
    检查；最后写出 CSV、NPZ、PNG、JSON 清单及文本摘要。成功完成（包括没有
    合格分量这一有效结果）返回 0，输入契约或结构错误则抛出异常。
    """
    # ---- 1. 参数、搜索窗口和输入文件 ----
    args = parse_args(argv)
    validate_args(args)
    cal_dir    = args.cal_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {output_dir}")
    windows = parse_search_windows(args.test_window, args.rm_min, args.rm_max)
    files   = discover_h5_files(cal_dir, args.file_list, args.recursive)

    # ---- 2. 读取 H5、减脉冲外基线并构造候选分量 ----
    # 输入通过基础校验后再创建全新的输出目录。多进程只负责互相独立的
    # 逐文件加载，汇总顺序与输入一致。
    output_dir.mkdir(parents=True)
    print(f"Found {len(files)} calibrated H5 files in {cal_dir}", flush=True)
    bursts: list[BurstRMData] = []
    warnings: list[str]       = []
    load_options: LoadFileOptions = {
        "freq_min": args.freq_min,
        "freq_max": args.freq_max,
        "min_time_snr": args.min_time_snr,
        "min_channels": args.min_channels,
        "stored_masks_only": args.stored_masks_only,
        "rfi_fft": args.rfi_fft,
        "rfi_channel_sigma": args.rfi_channel_sigma,
        "rfi_channel_window": args.rfi_channel_window,
        "rfi_channel_grow": args.rfi_channel_grow,
    }
    payloads = [(path, load_options) for path in files]
    loaded_files: Iterable[tuple[list[BurstRMData], list[str]]]
    executor: ProcessPoolExecutor | None
    if args.load_workers == 1:
        loaded_files = map(load_file_components_worker, payloads)
        executor     = None
    else:
        print(
            f"Loading H5 files with {args.load_workers} spawned workers...",
            flush=True,
        )
        executor = ProcessPoolExecutor(
            max_workers = args.load_workers,
            mp_context  = multiprocessing.get_context("spawn"),
        )
        loaded_files = executor.map(
            load_file_components_worker,
            payloads,
            chunksize=1,
        )
    try:
        progress_step = max(1, len(files) // 10)
        for index, (components, file_warnings) in enumerate(
            loaded_files,
            start=1,
        ):
            bursts.extend(components)
            warnings.extend(file_warnings)
            if args.load_workers > 1 and (
                index % progress_step == 0 or index == len(files)
            ):
                print(
                    f"  loaded {index}/{len(files)} H5 files",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    # ---- 3. 仅依据 Stokes I 完成分量和时间点选择 ----
    # 先按每个分量的 I 峰值淘汰过弱候选，再执行跨分量的统一时间点前缀优化。
    # 若没有合格分量，仍写出结构完整、状态明确的结果集后正常退出。
    loaded_bursts = bursts
    bursts        = [burst for burst in loaded_bursts if burst.peak_snr >= args.min_peak_snr]
    bursts.sort(key=lambda burst: burst.peak_snr, reverse=True)
    if args.max_bursts is not None:
        bursts = bursts[: args.max_bursts]
    if not bursts:
        missing_labels = sum(
            "missing attrs['bursts']" in warning for warning in warnings
        )
        if missing_labels == len(files):
            raise RuntimeError(
                "None of the calibrated H5 files contains attrs['bursts']; "
                "run after.burst_detect before burst_sync_rm"
            )
        write_no_eligible_result(
            args          = args,
            files         = files,
            loaded_bursts = loaded_bursts,
            output_dir    = output_dir,
            warnings      = warnings,
        )
        print(
            f"DONE status=no_eligible_components output={output_dir}",
            flush=True,
        )
        return 0
    disambiguate_component_ids(bursts)
    bursts, global_time_info, time_sample_table = select_global_time_samples(bursts)
    if not bursts:
        raise RuntimeError("Global Stokes-I time selection retained no components")

    print(
        "Global Stokes-I time selection: "
        f"{global_time_info['selected_sample_count']}/"
        f"{global_time_info['candidate_sample_count']} samples, "
        f"{global_time_info['selected_component_count']}/"
        f"{global_time_info['candidate_component_count']} components, "
        f"S/N cutoff={global_time_info['selected_snr_min']:.2f}",
        flush=True,
    )
    print(f"Selected {len(bursts)} burst components:", flush=True)
    for burst in bursts:
        print(
            f"  {burst.component_id}: I-peak S/N={burst.peak_snr:.2f}, "
            f"time={burst.n_time}, "
            f"selected-I-S/N={burst.time_snr.min():.2f}-"
            f"{burst.time_snr.max():.2f}, channels={burst.n_channel}, "
            f"band={burst.freq_mhz.min():.1f}-{burst.freq_mhz.max():.1f} MHz, "
            f"channel-RFI={burst.final_rfi_count}",
            flush=True,
        )
    for warning in warnings:
        print(f"WARNING: {warning}", flush=True)

    # ---- 4. 自动 RM 网格、逐分量曲线与共同曲线 ----
    # 精细网格用于最终峰值测量；更粗的零分布网格降低重复模拟成本，其步长仍由
    # 同一组波长平方覆盖和预设过采样因子决定。
    lambda2_sets = [burst.wave2_m2 for burst in bursts]
    rm_grid, rm_grid_info = build_automatic_rm_grid(
        args.rm_min,
        args.rm_max,
        lambda2_sets,
    )
    null_grid, null_grid_info = build_automatic_rm_grid(
        args.rm_min,
        args.rm_max,
        lambda2_sets,
        oversample=NULL_RM_GRID_OVERSAMPLE,
    )
    weights = curve_weights(bursts, args.curve_weighting, args.max_weight_ratio)

    individual: dict[str, dict[str, np.ndarray]] = {}
    print(
        f"Automatic RM grid: {rm_grid.size} points, "
        f"step={rm_grid_info['rm_step']:.3f} rad/m^2, "
        f"RMSF FWHM={rm_grid_info['rmsf_fwhm']:.3f} rad/m^2",
        flush=True,
    )
    print(f"Computing observed curves on {rm_grid.size} RM points...", flush=True)
    for burst in bursts:
        individual[burst.component_id] = individual_rm_curves(
            burst, rm_grid, args.transform_chunk_size
        )
        print(f"  observed {burst.component_id}", flush=True)
    combined = combine_curves(bursts, individual, weights)
    observed_null_grid = observed_curves_on_grid(
        bursts,
        rm_grid,
        individual,
        combined,
        null_grid,
        weights,
    )

    # ---- 5. 峰值汇总与脉冲外经验零分布 ----
    fine_table = fine_grid_summary(rm_grid, combined, windows)
    null_table: pd.DataFrame | None = None
    null_archive: dict[str, np.ndarray] | None = None
    pool_archive: dict[str, np.ndarray] | None = None
    if args.n_null > 0:
        print(
            f"Computing off-pulse null on {null_grid.size} RM points "
            f"for {args.n_null} trials...",
            flush=True,
        )
        null_table, null_archive, pool_archive = run_empirical_null(
            bursts,
            observed_null_grid,
            null_grid,
            windows,
            weights,
            n_null     = args.n_null,
            pool_size  = args.null_pool_size,
            seed       = args.seed,
            chunk_size = args.transform_chunk_size,
        )
        summary_table = fine_table.merge(
            null_table,
            on  = ["method", "method_role", "window"],
            how = "left",
        )
    else:
        summary_table = fine_table

    # ---- 6. 写出表格化测量结果与逐一剔除稳定性检查 ----
    member_string  = ";".join(burst.component_id for burst in bursts)
    channel_counts = np.asarray([burst.n_channel for burst in bursts], dtype=np.int64)
    run_metadata = {
        "run_label": args.run_label,
        "n_input_h5_files": len(files),
        "n_selected_h5_files": len({str(burst.file_path) for burst in bursts}),
        "n_candidate_components": int(global_time_info["candidate_component_count"]),
        "n_members": len(bursts),
        "members": member_string,
        "n_candidate_time_samples": int(global_time_info["candidate_sample_count"]),
        "n_selected_time_samples": int(global_time_info["selected_sample_count"]),
        "i_snr_equivalent": math.sqrt(
            float(global_time_info["selected_i_snr_squared_sum"])
        ),
        "selected_i_snr_cutoff": float(global_time_info["selected_snr_min"]),
        "selected_i_snr_median": float(global_time_info["selected_snr_median"]),
        "selected_i_snr_max": float(global_time_info["selected_snr_max"]),
        "requested_frequency_min_mhz": args.freq_min,
        "requested_frequency_max_mhz": args.freq_max,
        "actual_frequency_min_mhz": min(
            float(burst.freq_mhz.min()) for burst in bursts
        ),
        "actual_frequency_max_mhz": max(
            float(burst.freq_mhz.max()) for burst in bursts
        ),
        "usable_channels_min": int(np.min(channel_counts)),
        "usable_channels_median": float(np.median(channel_counts)),
        "usable_channels_max": int(np.max(channel_counts)),
        "curve_weighting": args.curve_weighting,
        "rm_grid_size": rm_grid_info["n_rm"],
        "rm_grid_step": rm_grid_info["rm_step"],
        "rm_rmsf_fwhm": rm_grid_info["rmsf_fwhm"],
        "rm_grid_samples_per_fwhm": rm_grid_info["samples_per_fwhm"],
        "null_rm_grid_step": null_grid_info["rm_step"],
    }
    for column_name, value in reversed(list(run_metadata.items())):
        summary_table.insert(0, column_name, value)
    summary_table.to_csv(output_dir / "burst_sync_rm_summary.csv", index=False)
    primary_window = select_primary_plot_window(windows, rm_grid)
    leave_one_out = leave_one_burst_out_summary(
        bursts,
        individual,
        weights,
        rm_grid,
        primary_window,
        combined,
        float(rm_grid_info["rmsf_fwhm"]),
    )
    leave_one_out.to_csv(output_dir / "leave_one_burst_out.csv", index=False)

    time_sample_table.to_csv(output_dir / "time_sample_selection.csv", index=False)
    candidate_counts = time_sample_table.groupby("component_id").size().to_dict()
    selection_rows   = []
    for rank, (burst, weight) in enumerate(zip(bursts, weights, strict=True), start=1):
        selection_rows.append(
            {
                "rank": rank,
                "component_id": burst.component_id,
                "file_name": burst.file_name,
                "file_path": str(burst.file_path),
                "burst_idx": burst.burst_idx,
                "i_peak_snr": burst.peak_snr,
                "curve_weight": float(weight),
                "time_indices": ";".join(
                    str(int(value)) for value in burst.time_indices
                ),
                "selected_time_i_snr": ";".join(
                    f"{float(value):.6g}" for value in burst.time_snr
                ),
                "candidate_time_samples": int(
                    candidate_counts.get(burst.component_id, 0)
                ),
                "n_time_samples": burst.n_time,
                "selected_time_i_snr_min": float(np.min(burst.time_snr)),
                "selected_time_i_snr_median": float(np.median(burst.time_snr)),
                "selected_time_i_snr_max": float(np.max(burst.time_snr)),
                "n_frequency_channels": burst.n_channel,
                "frequency_min_mhz": float(burst.freq_mhz.min()),
                "frequency_max_mhz": float(burst.freq_mhz.max()),
                "stored_cal_rfi_count": burst.stored_cal_rfi_count,
                "stored_burst_rfi_count": burst.stored_burst_rfi_count,
                "recalculated_rfi_count": burst.recalculated_rfi_count,
                "robust_rfi_count": burst.robust_rfi_count,
                "nonfinite_rfi_count": burst.nonfinite_rfi_count,
                "final_rfi_count": burst.final_rfi_count,
                "offpulse_sample_count": int(burst.p_noise.shape[0]),
            }
        )
    pd.DataFrame(selection_rows).to_csv(output_dir / "selected_bursts.csv", index=False)

    # ---- 7. 归档可复算曲线、零分布和实际抽样池 ----
    curve_archive: dict[str, np.ndarray] = {"rm_grid": rm_grid}
    for method, curve in combined.items():
        curve_archive[f"combined__{method}"] = curve
    for burst in bursts:
        for name, curve in individual[burst.component_id].items():
            curve_archive[f"{burst.component_id}__{name}"] = curve
    savez_compressed = cast(Any, np.savez_compressed)
    savez_compressed(output_dir / "burst_sync_rm_curves.npz", **curve_archive)
    if null_archive is not None and pool_archive is not None:
        savez_compressed(
            output_dir / "offpulse_null_maxima.npz",
            rm_grid=null_grid,
            **null_archive,
        )
        savez_compressed(output_dir / "offpulse_pool_indices.npz", **pool_archive)

    # ---- 8. 生成诊断图、机器可读摘要和完整运行清单 ----
    write_plot(
        output_dir / "burst_sync_rm.png",
        rm_grid,
        bursts,
        individual,
        combined,
        windows,
        null_table,
        null_archive,
        float(rm_grid_info["rmsf_fwhm"]),
        global_time_info,
        args.run_label,
    )

    run_summary = build_run_summary(
        args             = args,
        files            = files,
        bursts           = bursts,
        global_time_info = global_time_info,
        summary_table    = summary_table,
        leave_one_out    = leave_one_out,
        primary_window   = primary_window,
        rm_grid_info     = rm_grid_info,
        warnings         = warnings,
    )
    (output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    script_path = Path(__file__).resolve()
    manifest = {
        "script": str(script_path),
        "script_sha256": sha256_file(script_path),
        "run_label": args.run_label,
        "cal_dir": str(cal_dir),
        "output_dir": str(output_dir),
        "input_files": [str(path) for path in files],
        "input_contract": {
            "calibration_stage": "after.calibration",
            "burst_region_stage": "after.burst_detect",
            "required_datasets": ["data", "freq"],
            "data_axis_order": ["stokes", "time", "frequency"],
            "required_stokes": ["I", "Q", "U"],
            "required_attribute": "bursts",
            "required_burst_fields": [
                "time_start",
                "time_end",
                "freq_start",
                "freq_end",
            ],
        },
        "selected_components": [row["component_id"] for row in selection_rows],
        "time_sample_selection_table": str(output_dir / "time_sample_selection.csv"),
        "leave_one_burst_out_table": str(output_dir / "leave_one_burst_out.csv"),
        "run_summary": str(output_dir / "run_summary.json"),
        "pixel_mask": "not read or applied",
        "channel_mask_policy": (
            "stored calibration + stored detection"
            if args.stored_masks_only
            else "stored calibration + stored detection + recalculated all-Stokes "
            "+ robust local channel mask"
        ),
        "time_gate_policy": (
            "Stokes-I only; absolute-S/N candidates ranked across all "
            "components; retain the prefix maximizing "
            "sum(I_sample_S/N^2)/sqrt(n_selected)"
        ),
        "methods": {
            "time_pa_power": (
                "sum over bursts and selected time samples of |F_bt(RM)|^2 "
                "divided by per-time Q/U noise variance"
            ),
            "linear_degree_stack": (
                "fixed-weight sum of robustly standardized per-burst "
                "RM-versus-linear-degree curves"
            ),
        },
        "curve_weighting": args.curve_weighting,
        "load_workers": args.load_workers,
        "curve_weights": {
            burst.component_id: float(weight)
            for burst, weight in zip(bursts, weights, strict=True)
        },
        "rm_grid": {
            "min": args.rm_min,
            "max": args.rm_max,
            "n": rm_grid_info["n_rm"],
            "step": rm_grid_info["rm_step"],
            "delta_lambda2": rm_grid_info["delta_lambda2"],
            "rmsf_fwhm": rm_grid_info["rmsf_fwhm"],
            "oversample_target": rm_grid_info["oversample_target"],
            "samples_per_fwhm": rm_grid_info["samples_per_fwhm"],
        },
        "test_windows": [
            {"name": window.name, "low": window.low, "high": window.high}
            for window in windows
        ],
        "null": {
            "rm_n": null_grid_info["n_rm"],
            "rm_step": null_grid_info["rm_step"],
            "oversample_target": null_grid_info["oversample_target"],
            "n_null": args.n_null,
            "seed": args.seed,
            "pool_size_limit": args.null_pool_size,
        },
        "selection": {
            "min_time_snr": args.min_time_snr,
            "min_peak_snr": args.min_peak_snr,
            "max_bursts": args.max_bursts,
            "freq_min": args.freq_min,
            "freq_max": args.freq_max,
            "min_channels": args.min_channels,
            "global_time_selection": global_time_info,
        },
        "warnings": warnings,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    # 人读文本摘要保留与 CSV 相同的核心结果，便于在终端或批处理日志中快速检查。
    summary_lines = [
        "Synchronized time-sample RM stack",
        f"run_label={args.run_label}",
        f"cal_dir={cal_dir}",
        f"components={member_string}",
        (
            "global_time_selection="
            f"{global_time_info['selected_sample_count']}/"
            f"{global_time_info['candidate_sample_count']} samples; "
            f"cutoff={global_time_info['selected_snr_min']:.3f}"
        ),
        "pixel_mask=not read or applied",
        f"curve_weighting={args.curve_weighting}",
        "",
        summary_table.to_string(index=False),
    ]
    if warnings:
        summary_lines.extend(["", "Warnings:", *warnings])
    (output_dir / "burst_sync_rm_summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    print(summary_table.to_string(index=False), flush=True)
    print(f"DONE output={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# fmt: on
