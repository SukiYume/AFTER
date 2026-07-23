"""FAST 噪声管折叠与定标诊断图。

SEARCH 模式的四路偏振数据按 ``AABBCRCI`` 排列，即 AA、BB、Re(AB*)、
Im(AB*)。本模块采用与 PSRCHIVE 一致的 Stokes 转换：

    I = AA + BB, Q = AA - BB, U = 2 Re(AB*), V = 2 Im(AB*)

噪声管交叉项相位由下式直接计算：

    cos(phi) = U / sqrt(U**2 + V**2)
    sin(phi) = V / sqrt(U**2 + V**2)
    phi      = atan2(V, U)

这里只检查噪声管折叠和 SingleAxis 定标量，不拟合完整的仪器偏振泄漏。
"""

from __future__ import annotations

import argparse
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from astropy.io import fits

matplotlib.use("Agg")
# 必须先选择无界面绘图后端，再导入 pyplot。
import matplotlib.pyplot as plt  # noqa: E402


NOISE_CLOCK_PRODUCT = 4096 * 4096 * 12
DEFAULT_SCIENCE_BAND_MHZ = (1050.0, 1450.0)
DEFAULT_DIAGNOSTIC_BLOCKS = 31
DEFAULT_CHANNEL_CHUNK = 256
SMOOTHING_CHANNELS = 31
PHASE_REFERENCE_MHZ = 1250.0

BLACK = "#000000"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
MAGENTA = "#CC79A7"


# ============================================================
# 数据结构
# ============================================================


@dataclass
class NoiseCalFold:
    """一次噪声管折叠产生的定标量和必要诊断数据。"""

    source_path: Path
    frequency_mhz: np.ndarray
    noise_cal: np.ndarray
    folded_native: np.ndarray
    block_native: np.ndarray
    period_samples: int
    n_periods: int


@dataclass
class _FoldDiagnostic:
    """A/B/C 三幅折叠诊断图共用的数据。"""

    phase: np.ndarray
    normalized_native: np.ndarray
    normalized_stokes: np.ndarray
    normalized_block_i: np.ndarray
    block_start_min: int
    block_start_max: int
    minimum_correlation: float
    on_start_bin: int
    on_stop_bin: int
    step_cv_percent: float


@dataclass
class _BandDiagnostic:
    """D/E/F 三幅频率诊断图共用的数据。"""

    frequency: np.ndarray
    valid: np.ndarray
    phase_valid: np.ndarray
    plotted_band: tuple[float, float]
    auto_raw: np.ndarray
    auto_smooth: np.ndarray
    phase_components_raw: np.ndarray
    phase_components_smooth: np.ndarray
    phase_raw: np.ndarray
    phase_smooth: np.ndarray
    gain_raw: np.ndarray
    gain_smooth: np.ndarray
    polarized_fraction_raw: np.ndarray
    polarized_fraction_smooth: np.ndarray
    phase_at_reference_deg: float
    phase_slope_deg_per_mhz: float
    differential_delay_ns: float


# ============================================================
# 噪声管折叠
# ============================================================


def native_to_stokes(native: np.ndarray) -> np.ndarray:
    """把 ``[..., AA, BB, Re(AB*), Im(AB*)]`` 转为 Stokes I/Q/U/V。"""
    aa, bb, re_ab, im_ab = np.moveaxis(np.asarray(native), -1, 0)
    return np.stack((aa + bb, aa - bb, 2.0 * re_ab, 2.0 * im_ab), axis=-1)


def _frequency_axis(hdul: fits.HDUList, nchan: int) -> np.ndarray:
    """优先读取 DAT_FREQ；缺失时再由中心频率和带宽重建频率轴。"""
    subint = hdul[1]
    names = set(subint.columns.names or [])
    if "DAT_FREQ" in names:
        frequency = np.asarray(subint.data["DAT_FREQ"][0], dtype=float).reshape(-1)
        if frequency.size == nchan:
            return frequency

    primary = hdul[0].header
    sub_header = subint.header
    centre = primary.get("OBSFREQ", sub_header.get("OBSFREQ"))
    bandwidth = primary.get("OBSBW", sub_header.get("OBSBW"))
    if centre is None or bandwidth is None:
        return np.arange(nchan, dtype=float)
    channel_width = float(bandwidth) / nchan
    return (
        float(centre)
        - 0.5 * float(bandwidth)
        + (np.arange(nchan, dtype=float) + 0.5) * channel_width
    )


def compute_noise_cal_fold(
    cal_fits_path: str | os.PathLike[str],
    diagnostic_blocks: int = DEFAULT_DIAGNOSTIC_BLOCKS,
    channel_chunk: int = DEFAULT_CHANNEL_CHUNK,
) -> NoiseCalFold:
    """折叠一个 FAST 噪声管 FITS，并保持旧定标算法的数值结果。

    ``noise_cal`` 仍是 :mod:`calibration` 原实现使用的 on−off。绘图所需的
    时间轮廓单独累计，不能反过来改变定标量；频率方向分块处理，避免读取
    4096 通道、数 GB FITS 时出现过高的瞬时内存占用。
    """
    path = Path(cal_fits_path).expanduser().resolve()
    if diagnostic_blocks < 1:
        raise ValueError("diagnostic_blocks 必须大于等于 1")
    if channel_chunk < 1:
        raise ValueError("channel_chunk 必须大于等于 1")

    with fits.open(path, memmap=True) as hdul:
        hdu = hdul[1]
        header = hdu.header
        nsub = int(header["NAXIS2"])
        nsblk = int(header["NSBLK"])
        npol = int(header["NPOL"])
        nchan = int(header["NCHAN"])
        time_reso = float(header["TBIN"])
        if npol < 2:
            raise ValueError(f"噪声管 FITS 至少需要 2 路偏振，实际为 {npol}")

        raw = np.asarray(hdu.data["DATA"]).reshape(nsub * nsblk, npol, nchan)
        frequency_mhz = _frequency_axis(hdul, nchan)

        period_samples = int(NOISE_CLOCK_PRODUCT / (time_reso * 1e9))
        if period_samples < 2:
            raise ValueError(
                f"由 TBIN={time_reso} 得到的噪声管周期无效：{period_samples} 个采样点"
            )
        n_periods = raw.shape[0] // period_samples
        if n_periods < 1:
            raise ValueError(
                f"{path.name} 只有 {raw.shape[0]} 个采样点，短于一个噪声管周期 "
                f"({period_samples} 个采样点)"
            )

        n_samples_used = n_periods * period_samples
        periodic = raw[:n_samples_used].reshape(n_periods, period_samples, npol, nchan)

        # 先沿频率平均，得到方波轮廓和分时段稳定性图需要的小数组。
        # 显式使用 float64，保持旧版 np.mean 的数值行为。
        per_period_native = np.mean(periodic, axis=3, dtype=np.float64)
        folded_native = np.mean(per_period_native, axis=0, dtype=np.float64)

        power = np.mean(folded_native[:, :2], axis=1)
        threshold_on_mask = power > np.mean(power)
        if threshold_on_mask.all() or (~threshold_on_mask).all():
            raise ValueError("无法分离噪声管 on/off 状态")

        nblocks = min(diagnostic_blocks, n_periods)
        edges = np.linspace(0, n_periods, nblocks + 1, dtype=int)
        block_native = np.empty((nblocks, period_samples, npol), dtype=np.float64)
        for iblock in range(nblocks):
            start, stop = edges[iblock : iblock + 2]
            block_native[iblock] = np.mean(
                per_period_native[start:stop], axis=0, dtype=np.float64
            )

        # 按频率分块折叠，计算与旧版完全相同的 noise_on - noise_off，
        # 但不创建约 0.5 GB 的完整折叠数据立方。
        noise_cal = np.empty((npol, nchan), dtype=np.float64)
        for first in range(0, nchan, channel_chunk):
            last = min(first + channel_chunk, nchan)
            folded_chunk = np.mean(
                periodic[:, :, :, first:last], axis=0, dtype=np.float64
            )
            noise_on = np.mean(
                folded_chunk[threshold_on_mask], axis=0, dtype=np.float64
            )
            noise_off = np.mean(
                folded_chunk[~threshold_on_mask], axis=0, dtype=np.float64
            )
            noise_cal[:, first:last] = noise_on - noise_off

    return NoiseCalFold(
        source_path=path,
        frequency_mhz=frequency_mhz,
        noise_cal=noise_cal,
        folded_native=folded_native,
        block_native=block_native,
        period_samples=period_samples,
        n_periods=n_periods,
    )


# ============================================================
# 诊断量计算
# ============================================================


def _best_half_cycle_start(profile: np.ndarray) -> int:
    """在圆周轮廓上寻找积分功率最大的半周期起点。"""
    profile = np.asarray(profile, dtype=float)
    nbin = profile.size
    half = nbin // 2
    doubled = np.concatenate((profile, profile))
    cumulative = np.concatenate(([0.0], np.cumsum(doubled)))
    starts = np.arange(nbin)
    sums = cumulative[starts + half] - cumulative[starts]
    return int(np.nanargmax(sums))


def _circular_half_mask(nbin: int, start: int) -> np.ndarray:
    """返回从 start 开始、覆盖半个周期的圆周布尔掩码。"""
    return ((np.arange(nbin) - start) % nbin) < (nbin // 2)


def _moving_nanmedian(
    values: np.ndarray,
    width: int = SMOOTHING_CHANNELS,
) -> np.ndarray:
    """忽略 NaN 的滑动中值；偶数窗口自动扩为相邻奇数。"""
    width = max(1, int(width))
    if width % 2 == 0:
        width += 1
    half = width // 2
    padded = np.pad(
        np.asarray(values, dtype=float),
        (half, half),
        constant_values=np.nan,
    )
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(windows, axis=-1)


def _robust_limits(
    values: np.ndarray,
    lower: float = 1.0,
    upper: float = 99.0,
    padding: float = 0.10,
) -> tuple[float, float]:
    """按稳健分位数计算绘图范围，并在两端增加少量留白。"""
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    low, high = np.nanpercentile(finite, (lower, upper))
    if np.isclose(low, high):
        pad = max(abs(float(low)) * 0.1, 0.1)
    else:
        pad = float(high - low) * padding
    return float(low - pad), float(high + pad)


def _shade_on_phase(ax: plt.Axes, start_phase: float, stop_phase: float) -> None:
    """标出噪声管 on 半周期及两个跳变位置。"""
    if start_phase < stop_phase:
        ax.axvspan(start_phase, stop_phase, color="#F0E442", alpha=0.10, lw=0)
    else:
        ax.axvspan(0.0, stop_phase, color="#F0E442", alpha=0.10, lw=0)
        ax.axvspan(start_phase, 1.0, color="#F0E442", alpha=0.10, lw=0)
    for transition in (start_phase, stop_phase):
        ax.axvline(transition, color="#666666", lw=0.9, ls="--", alpha=0.8)


def _science_mask(
    frequency_mhz: np.ndarray,
    science_band_mhz: tuple[float, float],
) -> tuple[np.ndarray, tuple[float, float]]:
    """选择科学频段；非标准接收机数据退回到中间 80% 频段。"""
    low, high = map(float, science_band_mhz)
    mask = (frequency_mhz >= low) & (frequency_mhz <= high)
    if np.count_nonzero(mask) >= 4:
        return mask, (low, high)

    # 非标准接收机频段不能直接返回空图，因此保留有限频率的中间 80%。
    finite = frequency_mhz[np.isfinite(frequency_mhz)]
    if finite.size < 4:
        return np.isfinite(frequency_mhz), (
            float(np.nanmin(frequency_mhz)),
            float(np.nanmax(frequency_mhz)),
        )
    low, high = np.nanpercentile(finite, (10.0, 90.0))
    return (frequency_mhz >= low) & (frequency_mhz <= high), (float(low), float(high))


def _diagnostic_label(path: Path) -> str:
    """优先用文件名末尾的四位序号作为图标题标签。"""
    match = re.search(r"[_-](\d{4})$", path.stem)
    return match.group(1) if match else path.stem


def _prepare_fold_diagnostic(folded: NoiseCalFold) -> _FoldDiagnostic:
    """计算折叠相位轮廓和分时段稳定性，供 A/B/C 图共用。"""
    native = folded.folded_native[:, :4]
    block_native = folded.block_native[:, :, :4]
    stokes = native_to_stokes(native)
    block_stokes = native_to_stokes(block_native)
    nblock, nbin, _ = block_stokes.shape
    phase = np.arange(nbin, dtype=float) / nbin

    # 噪声管占空比为 50%；在圆周相位上寻找 Stokes I 总功率最大的半周期。
    start_bin = _best_half_cycle_start(stokes[:, 0])
    stop_bin = (start_bin + nbin // 2) % nbin
    on_mask = _circular_half_mask(nbin, start_bin)

    stokes_on = np.nanmedian(stokes[on_mask], axis=0)
    stokes_off = np.nanmedian(stokes[~on_mask], axis=0)
    delta_i = float(stokes_on[0] - stokes_off[0])
    if not np.isfinite(delta_i) or delta_i <= 0:
        raise ValueError(f"折叠后的 Stokes I 跳变量不是正数：{delta_i}")
    normalized_stokes = (stokes - stokes_off) / delta_i
    native_off = np.nanmedian(native[~on_mask], axis=0)
    normalized_native = (
        (native - native_off) * np.asarray((1.0, 1.0, 2.0, 2.0)) / delta_i
    )

    # 每个时间块使用相同的 on/off 相位，比较方波幅度和边沿是否随时间漂移。
    block_profiles = block_stokes[:, :, 0]
    block_off = np.nanmedian(block_profiles[:, ~on_mask], axis=1)
    block_steps = np.nanmedian(block_profiles[:, on_mask], axis=1) - block_off
    if np.any(~np.isfinite(block_steps)) or np.any(block_steps <= 0):
        raise ValueError("至少一个诊断时间块的 Stokes I 跳变量不是正数")
    normalized_block_i = (block_profiles - block_off[:, None]) / block_steps[:, None]
    block_starts = np.asarray(
        [_best_half_cycle_start(profile) for profile in block_profiles], dtype=int
    )
    template = np.nanmedian(normalized_block_i, axis=0)
    correlations = np.asarray(
        [np.corrcoef(profile, template)[0, 1] for profile in normalized_block_i]
    )
    if nblock > 1:
        step_cv = float(
            100.0 * np.nanstd(block_steps, ddof=1) / np.nanmean(block_steps)
        )
    else:
        step_cv = 0.0

    return _FoldDiagnostic(
        phase=phase,
        normalized_native=normalized_native,
        normalized_stokes=normalized_stokes,
        normalized_block_i=normalized_block_i,
        block_start_min=int(np.min(block_starts)),
        block_start_max=int(np.max(block_starts)),
        minimum_correlation=float(np.nanmin(correlations)),
        on_start_bin=start_bin,
        on_stop_bin=stop_bin,
        step_cv_percent=step_cv,
    )


def _prepare_band_diagnostic(
    folded: NoiseCalFold,
    science_band_mhz: tuple[float, float],
) -> _BandDiagnostic:
    """计算频率方向的相位、带通和 SingleAxis 定标量。"""
    frequency = folded.frequency_mhz
    noise_cal = folded.noise_cal[:4]
    aa, bb = noise_cal[:2]
    delta_stokes = native_to_stokes(noise_cal.T)
    delta_i_channel, delta_q, delta_u, delta_v = np.moveaxis(delta_stokes, -1, 0)
    band_mask, plotted_band = _science_mask(frequency, science_band_mhz)

    # 先排除无效或非正响应通道，再剔除低于中值 5% 的坏通道。
    finite = np.isfinite(noise_cal).all(axis=0) & np.isfinite(frequency)
    positive = (aa > 0) & (bb > 0) & (delta_i_channel > 0)
    initial = band_mask & finite & positive
    if np.count_nonzero(initial) < 4:
        raise ValueError("科学频段内的正响应通道不足 4 个")
    median_i = float(np.nanmedian(delta_i_channel[initial]))
    valid = initial & (delta_i_channel > 0.05 * median_i)
    if np.count_nonzero(valid) < 4:
        raise ValueError("科学频段内可用于噪声管诊断的通道不足 4 个")

    nchan = frequency.size
    auto_raw = np.full((2, nchan), np.nan)
    auto_raw[0, valid] = aa[valid] / np.nanmedian(aa[valid])
    auto_raw[1, valid] = bb[valid] / np.nanmedian(bb[valid])

    # 只有交叉项振幅非零的通道才有定义良好的 cos、sin 和相位。
    cross_amplitude = np.hypot(delta_u, delta_v)
    phase_valid = valid & np.isfinite(cross_amplitude) & (cross_amplitude > 0)
    if np.count_nonzero(phase_valid) < 4:
        raise ValueError("具有有效交叉项相位的通道不足 4 个")
    phase_components_raw = np.full((2, nchan), np.nan)
    phase_components_raw[0, phase_valid] = (
        delta_u[phase_valid] / cross_amplitude[phase_valid]
    )
    phase_components_raw[1, phase_valid] = (
        delta_v[phase_valid] / cross_amplitude[phase_valid]
    )
    phase_raw = np.full(nchan, np.nan)
    phase_raw[phase_valid] = np.rad2deg(
        np.arctan2(delta_v[phase_valid], delta_u[phase_valid])
    )

    gain_raw = np.full(nchan, np.nan)
    gain_raw[valid] = 10.0 * np.log10(aa[valid] / bb[valid])
    polarized_fraction_raw = np.full(nchan, np.nan)
    polarized_fraction_raw[valid] = (
        np.sqrt(delta_q[valid] ** 2 + delta_u[valid] ** 2 + delta_v[valid] ** 2)
        / delta_i_channel[valid]
    )

    auto_smooth = np.stack([_moving_nanmedian(values) for values in auto_raw])
    phase_components_smooth = np.stack(
        [_moving_nanmedian(values) for values in phase_components_raw]
    )
    phase_smooth = np.rad2deg(
        np.arctan2(phase_components_smooth[1], phase_components_smooth[0])
    )
    gain_smooth = _moving_nanmedian(gain_raw)
    polarized_fraction_smooth = _moving_nanmedian(polarized_fraction_raw)

    phase_fit_mask = band_mask & np.isfinite(phase_smooth)
    if np.count_nonzero(phase_fit_mask) < 2:
        raise ValueError("可用于拟合交叉项相位的通道不足 2 个")
    unwrapped_phase = np.rad2deg(np.unwrap(np.deg2rad(phase_smooth[phase_fit_mask])))
    slope, intercept = np.polyfit(frequency[phase_fit_mask], unwrapped_phase, 1)
    phase_at_reference = float(
        (slope * PHASE_REFERENCE_MHZ + intercept + 180.0) % 360.0 - 180.0
    )
    delay_ns = float(slope * 1000.0 / 360.0)

    return _BandDiagnostic(
        frequency=frequency,
        valid=valid,
        phase_valid=phase_valid,
        plotted_band=plotted_band,
        auto_raw=auto_raw,
        auto_smooth=auto_smooth,
        phase_components_raw=phase_components_raw,
        phase_components_smooth=phase_components_smooth,
        phase_raw=phase_raw,
        phase_smooth=phase_smooth,
        gain_raw=gain_raw,
        gain_smooth=gain_smooth,
        polarized_fraction_raw=polarized_fraction_raw,
        polarized_fraction_smooth=polarized_fraction_smooth,
        phase_at_reference_deg=phase_at_reference,
        phase_slope_deg_per_mhz=float(slope),
        differential_delay_ns=delay_ns,
    )


def _build_metrics(
    folded: NoiseCalFold,
    fold_data: _FoldDiagnostic,
    band_data: _BandDiagnostic,
) -> dict[str, float | int | str]:
    """汇总终端日志和后续质量检查需要的标量。"""
    nbin = fold_data.phase.size
    return {
        "source": str(folded.source_path),
        "period_samples": int(folded.period_samples),
        "n_periods": int(folded.n_periods),
        "diagnostic_blocks": int(fold_data.normalized_block_i.shape[0]),
        "on_start_bin": int(fold_data.on_start_bin),
        "on_stop_bin": int(fold_data.on_stop_bin),
        "on_start_phase": float(fold_data.on_start_bin / nbin),
        "on_stop_phase": float(fold_data.on_stop_bin / nbin),
        "delta_i_cv_percent": fold_data.step_cv_percent,
        "minimum_template_correlation": fold_data.minimum_correlation,
        "valid_science_channels": int(np.count_nonzero(band_data.valid)),
        "crosshand_phase_at_1250_mhz_deg": band_data.phase_at_reference_deg,
        "crosshand_phase_slope_deg_per_mhz": band_data.phase_slope_deg_per_mhz,
        "equivalent_differential_delay_ns": band_data.differential_delay_ns,
        "median_differential_gain_db": float(
            np.nanmedian(band_data.gain_raw[band_data.valid])
        ),
        "median_polarized_fraction": float(
            np.nanmedian(band_data.polarized_fraction_raw[band_data.valid])
        ),
    }


# ============================================================
# 六联诊断图
# ============================================================


def _plot_fold_profile(
    ax: plt.Axes,
    fold_data: _FoldDiagnostic,
    values: np.ndarray,
    series: tuple[tuple[str, str], ...],
    title: str,
    legend_columns: int,
) -> None:
    """绘制一幅折叠方波图，统一 A/B 图的坐标和 on 区域标记。"""
    for index, (label, color) in enumerate(series):
        ax.plot(
            fold_data.phase,
            values[:, index],
            lw=1.25,
            color=color,
            label=label,
        )
    nbin = fold_data.phase.size
    start_phase = fold_data.on_start_bin / nbin
    stop_phase = fold_data.on_stop_bin / nbin
    _shade_on_phase(ax, start_phase, stop_phase)
    ax.set(
        title=title,
        xlabel="Fold phase",
        ylabel="Off-subtracted / ΔI",
        xlim=(0, 1),
    )
    ax.grid(alpha=0.18)
    ax.legend(ncol=legend_columns, frameon=False, loc="best")


def _plot_alignment_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    fold_data: _FoldDiagnostic,
) -> None:
    """绘制 C 图：检查不同时间块的方波相位和幅度稳定性。"""
    nblock, nbin = fold_data.normalized_block_i.shape
    image = ax.imshow(
        fold_data.normalized_block_i,
        origin="lower",
        aspect="auto",
        extent=(0.0, 1.0, -0.5, nblock - 0.5),
        interpolation="nearest",
        cmap="cividis",
        vmin=-0.05,
        vmax=1.05,
    )
    for transition_bin in (fold_data.on_start_bin, fold_data.on_stop_bin):
        ax.axvline(
            transition_bin / nbin,
            color="white",
            lw=1.0,
            ls="--",
            alpha=0.9,
        )
    ax.set(
        title=f"C  Phase alignment over {nblock} time blocks",
        xlabel="Fold phase",
        ylabel="Time block",
    )
    ax.text(
        0.98,
        0.05,
        f"start bins {fold_data.block_start_min}–{fold_data.block_start_max}\n"
        f"ΔI CV {fold_data.step_cv_percent:.3f}%\n"
        f"min corr {fold_data.minimum_correlation:.6f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        color="white",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "black",
            "alpha": 0.55,
            "edgecolor": "none",
        },
    )
    fig.colorbar(image, ax=ax, pad=0.015, label="Normalized Stokes I")


def _plot_phase_panel(ax: plt.Axes, band_data: _BandDiagnostic) -> None:
    """绘制 D 图：交叉项的 cos、sin 和相位。"""
    frequency = band_data.frequency
    valid = band_data.phase_valid
    cos_raw, sin_raw = band_data.phase_components_raw
    cos_smooth, sin_smooth = band_data.phase_components_smooth

    ax.scatter(
        frequency[valid],
        cos_raw[valid],
        s=3,
        alpha=0.06,
        color=ORANGE,
        rasterized=True,
    )
    ax.scatter(
        frequency[valid],
        sin_raw[valid],
        s=3,
        alpha=0.06,
        color=GREEN,
        rasterized=True,
    )
    line_cos = ax.plot(
        frequency,
        cos_smooth,
        color=ORANGE,
        lw=1.55,
        label="U/C = cos φ",
    )[0]
    line_sin = ax.plot(
        frequency,
        sin_smooth,
        color=GREEN,
        lw=1.55,
        label="V/C = sin φ",
    )[0]

    # cos/sin 的物理范围是 [-1, 1]；上方额外留白专门放 legend。
    component_ylim = (-1.08, 1.35)
    ax.set(
        title="D  Noise-cal cross-hand phase from Stokes U/V",
        xlabel="Frequency (MHz)",
        ylabel="Normalized cross-hand components",
        xlim=band_data.plotted_band,
        ylim=component_ylim,
    )
    ax.axhline(0.0, color="#777777", lw=0.8, ls="--")
    ax.grid(alpha=0.18)

    ax2 = ax.twinx()
    ax2.scatter(
        frequency[valid],
        band_data.phase_raw[valid],
        s=3,
        alpha=0.04,
        color=BLACK,
        rasterized=True,
    )
    line_phase = ax2.plot(
        frequency,
        band_data.phase_smooth,
        color=BLACK,
        lw=1.25,
        label="φ = atan2(V, U)",
    )[0]
    # 相位轴也在顶部留出 legend 空间，保持上一版诊断图的显示范围。
    ax2.set_ylim(-185.0, 225.0)
    ax2.set_ylabel("Cross-hand phase φ (deg)")
    ax.legend(
        [line_cos, line_sin, line_phase],
        [line_cos.get_label(), line_sin.get_label(), line_phase.get_label()],
        ncol=3,
        frameon=False,
        loc="upper center",
    )
    ax.text(
        0.02,
        0.05,
        f"φ({PHASE_REFERENCE_MHZ:.0f} MHz) = "
        f"{band_data.phase_at_reference_deg:.1f}°\n"
        f"τdiff = {band_data.differential_delay_ns:.2f} ns",
        transform=ax.transAxes,
        va="bottom",
        fontsize=9.3,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "alpha": 0.86,
            "edgecolor": "#BBBBBB",
        },
    )


def _plot_bandpass_panel(ax: plt.Axes, band_data: _BandDiagnostic) -> None:
    """绘制 E 图：AA、BB 的 on−off 归一化带通。"""
    frequency = band_data.frequency
    valid = band_data.valid
    for index, (label, color) in enumerate((("ΔAA", BLUE), ("ΔBB", ORANGE))):
        raw = band_data.auto_raw[index]
        smooth = band_data.auto_smooth[index]
        ax.scatter(
            frequency[valid],
            raw[valid],
            s=3,
            alpha=0.08,
            color=color,
            rasterized=True,
        )
        ax.plot(
            frequency,
            smooth,
            lw=1.6,
            color=color,
            label=f"{label} ({SMOOTHING_CHANNELS}-ch median)",
        )
    ax.set_ylim(*_robust_limits(band_data.auto_raw[:, valid], 0.5, 99.5, 0.08))
    ax.set(
        title="E  On−off auto-correlation bandpass",
        xlabel="Frequency (MHz)",
        ylabel="Normalized response",
        xlim=band_data.plotted_band,
    )
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="best")


def _plot_amplitude_panel(ax: plt.Axes, band_data: _BandDiagnostic) -> None:
    """绘制 F 图：差分增益和噪声管偏振度。"""
    frequency = band_data.frequency
    valid = band_data.valid
    ax.scatter(
        frequency[valid],
        band_data.gain_raw[valid],
        s=3,
        alpha=0.08,
        color=BLUE,
        rasterized=True,
    )
    line_gain = ax.plot(
        frequency,
        band_data.gain_smooth,
        color=BLUE,
        lw=1.6,
        label="Differential gain",
    )[0]
    ax.set(
        title="F  Single-axis amplitude observables",
        xlabel="Frequency (MHz)",
        ylabel="10 log10(ΔAA/ΔBB) (dB)",
        xlim=band_data.plotted_band,
        ylim=_robust_limits(band_data.gain_raw[valid], 0.5, 99.5, 0.10),
    )
    ax.grid(alpha=0.18)

    ax2 = ax.twinx()
    ax2.scatter(
        frequency[valid],
        band_data.polarized_fraction_raw[valid],
        s=3,
        alpha=0.06,
        color=GREEN,
        rasterized=True,
    )
    line_pol = ax2.plot(
        frequency,
        band_data.polarized_fraction_smooth,
        color=GREEN,
        lw=1.35,
        label="Polarized fraction",
    )[0]
    pol_low, pol_high = _robust_limits(
        band_data.polarized_fraction_raw[valid], 0.5, 99.5, 0.10
    )
    ax2.set_ylim(min(pol_low, 0.98), max(pol_high, 1.02))
    ax2.set_ylabel("sqrt(Q²+U²+V²) / I", color=GREEN)
    ax2.tick_params(axis="y", colors=GREEN)
    ax.legend(
        [line_gain, line_pol],
        [line_gain.get_label(), line_pol.get_label()],
        frameon=False,
        loc="best",
    )


def _draw_diagnostic_figure(
    folded: NoiseCalFold,
    fold_data: _FoldDiagnostic,
    band_data: _BandDiagnostic,
) -> plt.Figure:
    """按 A→F 的诊断逻辑组装完整六联图。"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 14), dpi=150)
    fig.suptitle(
        f"FAST Noise-Cal Fold Diagnostic — {_diagnostic_label(folded.source_path)}",
        fontsize=19,
        fontweight="bold",
        y=0.985,
    )

    _plot_fold_profile(
        axes[0, 0],
        fold_data,
        fold_data.normalized_native,
        (
            ("AA", BLUE),
            ("BB", ORANGE),
            ("2 Re(AB*)", GREEN),
            ("2 Im(AB*)", MAGENTA),
        ),
        "A  Native coherency square waves",
        legend_columns=2,
    )
    _plot_fold_profile(
        axes[0, 1],
        fold_data,
        fold_data.normalized_stokes,
        (("I", BLACK), ("Q", BLUE), ("U", ORANGE), ("V", GREEN)),
        "B  PSRCHIVE-equivalent Stokes fold",
        legend_columns=4,
    )
    _plot_alignment_panel(fig, axes[1, 0], fold_data)
    _plot_phase_panel(axes[1, 1], band_data)
    _plot_bandpass_panel(axes[2, 0], band_data)
    _plot_amplitude_panel(axes[2, 1], band_data)

    fig.text(
        0.5,
        0.012,
        "Input products: AABBCRCI.  D: C = sqrt(U²+V²), cos φ = U/C, "
        "sin φ = V/C, φ = atan2(V,U).  Leakage terms are outside this diagnostic.",
        ha="center",
        va="bottom",
        fontsize=9.3,
        color="#444444",
    )
    fig.tight_layout(rect=(0.025, 0.035, 0.985, 0.965), h_pad=2.0, w_pad=1.6)
    return fig


def _as_png_path(path: str | os.PathLike[str]) -> Path:
    """解析输出路径，并统一使用 PNG 扩展名。"""
    output = Path(path).expanduser().resolve()
    return output if output.suffix.lower() == ".png" else output.with_suffix(".png")


def _save_figure(fig: plt.Figure, output: Path) -> None:
    """先写临时文件再原子替换，避免留下不完整诊断图。"""
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.png")
    try:
        fig.savefig(temporary, dpi=220, bbox_inches="tight", format="png")
        os.replace(temporary, output)
    finally:
        plt.close(fig)
        if temporary.exists():
            temporary.unlink()


def plot_noise_cal_diagnostic(
    folded: NoiseCalFold,
    output_path: str | os.PathLike[str],
    science_band_mhz: tuple[float, float] = DEFAULT_SCIENCE_BAND_MHZ,
) -> dict[str, float | int | str]:
    """生成六联噪声管诊断图，并返回关键质量指标。"""
    if folded.noise_cal.shape[0] < 4 or folded.folded_native.shape[-1] < 4:
        raise ValueError("噪声管诊断图需要完整的四路 AABBCRCI 数据")

    output = _as_png_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # 先完成两组数据推导，再按 A→F 顺序绘图；绘图过程不接触定标量计算。
    fold_data = _prepare_fold_diagnostic(folded)
    band_data = _prepare_band_diagnostic(folded, science_band_mhz)
    metrics = _build_metrics(folded, fold_data, band_data)
    figure = _draw_diagnostic_figure(folded, fold_data, band_data)
    _save_figure(figure, output)
    return metrics


# ============================================================
# 定标流程与命令行入口
# ============================================================


def fold_noise_cal(
    cal_fits_path: str | os.PathLike[str],
    diagnostic_dir: str | os.PathLike[str] | None = None,
    diagnostic_path: str | os.PathLike[str] | None = None,
    make_diagnostic: bool = True,
    science_band_mhz: tuple[float, float] = DEFAULT_SCIENCE_BAND_MHZ,
) -> np.ndarray:
    """折叠噪声管、按需写诊断图，并返回定标使用的 on−off 数组。

    未指定输出位置时，诊断图保存在 FITS 同目录；正常定标和批处理流程通过
    ``diagnostic_dir`` 把图放到定标产品目录。``diagnostic_path`` 仅用于显式
    指定单个 PNG 路径，两者不能同时传入。
    """
    if diagnostic_dir is not None and diagnostic_path is not None:
        raise ValueError("diagnostic_dir 和 diagnostic_path 不能同时指定")

    folded = compute_noise_cal_fold(cal_fits_path)
    if make_diagnostic:
        source = folded.source_path
        if diagnostic_path is None:
            output_dir = (
                Path(diagnostic_dir).expanduser()
                if diagnostic_dir is not None
                else source.parent
            )
            diagnostic_path = output_dir / f"{source.stem}_noise_cal_diagnostic.png"
        output = _as_png_path(diagnostic_path)
        metrics = plot_noise_cal_diagnostic(
            folded,
            output,
            science_band_mhz=science_band_mhz,
        )
        print(
            f"  [noise-cal] diagnostic: {output} "
            f"(periods={folded.n_periods}, phase={metrics['on_start_phase']:.4f}/"
            f"{metrics['on_stop_phase']:.4f}, "
            f"ΔI CV={metrics['delta_i_cv_percent']:.3f}%)"
        )
    return folded.noise_cal


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cal_fits", help="FAST noise-cal *_0001.fits")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--output-dir", default=None, help="诊断图输出目录")
    output_group.add_argument("--output", default=None, help="诊断图 PNG 路径")
    parser.add_argument(
        "--science-min",
        type=float,
        default=DEFAULT_SCIENCE_BAND_MHZ[0],
    )
    parser.add_argument(
        "--science-max",
        type=float,
        default=DEFAULT_SCIENCE_BAND_MHZ[1],
    )
    parser.add_argument("--no-diagnostic", action="store_true", help="只折叠，不画图")
    return parser.parse_args()


def _main() -> None:
    """命令行入口。"""
    args = _parse_args()
    result = fold_noise_cal(
        args.cal_fits,
        diagnostic_dir=args.output_dir,
        diagnostic_path=args.output,
        make_diagnostic=not args.no_diagnostic,
        science_band_mhz=(args.science_min, args.science_max),
    )
    print(f"noise_cal shape={result.shape}, dtype={result.dtype}")


if __name__ == "__main__":
    _main()
