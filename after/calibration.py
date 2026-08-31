# fmt: off

"""cut_burst_data 输出的 h5 文件定标 + 裁剪/下采样 + RFI 检测 + 保存。

流程:
  1. 读 BURST_DIR 中所有 burst h5（非 _cal.h5）并按波束分组。
  2. 每个波束找第一个 _0001.fits 作为定标文件, 折叠噪声管 → noise_cal。
  3. 从 npz 加载 t_cal (K)。
  4. 每个 burst:
     · 用 noise_cal 定标：四路得到 IQUV，两路 AA/BB 只把 I 作为有效科学量,
     · 乘以 t_cal/(2*gain) 直接变成 Jy（保持统一的四平面保存格式）,
     · 可选地围绕输入时间轴中心裁剪固定数量的原始采样点,
     · 时间+频率下采样 (保存倍率, 默认 = 画图倍率),
     · 可选地在下采样后再围绕时间轴中心裁出固定点数,
     · 用 after.rfi.cal_rfi (FFT 方法) 在整段数据上检测 RFI 通道/像素,
     · 画图: 在保存数据上再做 extra 倍下采样到"清晰"分辨率(减基线+抹RFI),
     · 存 _cal.h5 (data/freq/rfi_mask/gain/gain_err)。

保存策略: 写入 h5 的 data 是"相对原始"的定标+下采样后数据, 不减基线、不 NaN 掉
RFI; rfi_mask 与 rfi_channel 作为辅助信息一并保存。下游 burst_analysis 会用
真正的噪声段做二次 RFI 精修和按通道基线减除, 比在这里用全时段中值更干净。

下采样规则:
  · 画图倍率 (plot_dt, plot_df) 自动计算:
        频率目标 ~512 通道, 时间目标 ~ 49.152us×8 ≈ 393us.
    比如 49.152us / 4096ch → (8, 8); 98.304us / 4096ch → (4, 8).
  · 保存倍率 (down_time, down_freq) 默认 = 画图倍率；显式值可以
    比默认值小或大。若保存分辨率更高，画图时再做整数倍下采样；
    若保存分辨率已更低，JPG 直接使用保存分辨率。
  · target_time_reso 可根据每个文件的原始时间分辨率自动选择
    down_time；目标必须是原始时间分辨率的整数倍。
  · 指定 time_crop_samples 时，先在原始时间轴上做中心裁剪；若没有显式指定
    down_time，则默认保留原始时间分辨率。图像直接使用保存分辨率，保证 JPG
    与 _cal.h5 中的数据逐像素对应。
  · 指定 output_time_samples 时，顺序严格为“完整数据定标 → 下采样
    → 中心裁剪”。它与原始数据上先裁的 time_crop_samples 互斥。

文件同时保存 time_reso_raw（原始值）和 time_reso（下采样后的有效值）；下游
直接使用 time_reso 计算流量和能量。
"""

# Stokes I 是领域标准符号，保留其大写单字母写法。
# ruff: noqa: E741

import os
import re
from collections import defaultdict
from typing import Any, cast

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
from multiprocessing import Pool

import matplotlib.pyplot as plt
import seaborn as sns  # noqa: F401 - 'mako' 颜色映射需要 seaborn
from astropy.io import fits
from astropy.utils import iers
from matplotlib import gridspec

from . import DEFAULT_CAL_NPZ
from .calibration_noise import DEFAULT_NOISE_PERIOD_S, fold_noise_cal
from .rfi import cal_rfi
from .zenith_angle import get_gain, get_za

iers.conf.auto_download = False
# 远端计算节点通常不联网；允许现有 IERS 表用于较新的观测日期。
iers.conf.auto_max_age  = None


# 画图目标分辨率: 频率 ~512 通道, 时间 ~ 49.152us × 8 ≈ 393us.
# 用于自动推算 plot_dt / plot_df, 也作为 save_dt / save_df 的默认值.
PLOT_TARGET_TIME_RESO = 49.152e-6 * 8
PLOT_TARGET_NCHAN     = 512


def find_cal_fits(directory, beam):
    """查找 directory 中指定波束的定标 fits (_0001.fits 结尾且含 Mdd)。"""
    for fname in sorted(os.listdir(directory)):
        if fname.endswith("_0001.fits") and f"M{beam:02d}" in fname:
            return os.path.join(directory, fname)
    return None


def load_t_cal(cal_npz_path, beam, nchan):
    """从 npz 加载 t_cal 并匹配到目标 nchan, 返回 (2, nchan)。

    t_cal npz 固定 4096 通道, 需要 nchan 与 4096 之间整数倍关系 (FAST
    数据总是 2 的幂, 通常满足). 不整除会直接 assert 报错避免静默截断.
    """
    t_cal = np.load(cal_npz_path)["tcal"][:, :, beam - 1]  # (4096, 2)
    if nchan <= 4096:
        assert 4096 % nchan == 0, (
            f"nchan={nchan} 不是 4096 的因子, t_cal 无法按整数倍合并"
        )
        t_cal = np.mean(t_cal.reshape(nchan, 4096 // nchan, 2), axis=1).T
    else:
        assert nchan % 4096 == 0, (
            f"nchan={nchan} 不是 4096 的整数倍, t_cal 无法按整数倍展开"
        )
        t_cal = np.repeat(t_cal, nchan // 4096, axis=0).T
    return t_cal


def calibrate_to_iquv(data, noise_cal, t_cal, gain, cal_threshold=0.05):
    """偏振定标 + 流量缩放, 一步得到 Jy 单位的四平面兼容数组。

    偏振定标: 用 noise_cal 归一化两个 feed 的增益差异, 并用 arctan2 校正
    交叉项相位;
    流量缩放: 每个偏振独立乘 t_cal[pol]/(2*gain) 再合成 I/Q, 交叉项 U/V 使用
    sqrt(t_cal[0]*t_cal[1])/(2*gain) 作为等效尺度。

    参数
    ----------
    data      : (nsamp, npol, nchan) uint8 或浮点
    noise_cal : (npol, nchan)
    t_cal     : (2, nchan)
    gain      : 标量或 (nchan,)

    返回
    -------
    iquv : (4, nsamp, nchan) float32, 单位 Jy
        ``npol=4`` 时四个 Stokes 都可用；``npol=2`` 时只有 Stokes I
        可用于科学分析，Q 仅保留现有两路功率差，U/V 为兼容下游格式的零值。
    """
    data      = np.asarray(data)
    noise_cal = np.asarray(noise_cal)
    t_cal     = np.asarray(t_cal)
    if data.ndim != 3:
        raise ValueError(f"data 必须是 (nsamp,npol,nchan)，实际形状为 {data.shape}")

    nsamp, npol, nchan = data.shape
    if npol not in (2, 4):
        raise ValueError(f"data 的 npol 必须为 2 或 4，实际为 {npol}")
    if noise_cal.ndim != 2 or noise_cal.shape[1] != nchan or noise_cal.shape[0] < npol:
        raise ValueError(
            f"noise_cal 至少需要 ({npol},{nchan})，实际形状为 {noise_cal.shape}"
        )
    if t_cal.shape != (2, nchan):
        raise ValueError(f"t_cal 必须是 (2,{nchan})，实际形状为 {t_cal.shape}")

    noise_a12 = np.where(noise_cal[0] > cal_threshold, 1.0 / noise_cal[0], 0.0)
    noise_a22 = np.where(noise_cal[1] > cal_threshold, 1.0 / noise_cal[1], 0.0)
    scale_0   = t_cal[0] / (2.0 * gain)
    scale_1   = t_cal[1] / (2.0 * gain)

    I = scale_0 * noise_a12 * data[:, 0, :] + scale_1 * noise_a22 * data[:, 1, :]
    Q = -(scale_0 * noise_a12 * data[:, 0, :] - scale_1 * noise_a22 * data[:, 1, :])

    if npol == 4:
        scale_cross = np.sqrt(scale_0 * scale_1)  # 交叉项等效 t_cal
        noise_dphi  = np.arctan2(noise_cal[3], noise_cal[2])
        noise_a1a2  = np.sqrt(noise_a12 * noise_a22)
        noise_cos   = np.cos(noise_dphi) * noise_a1a2
        noise_sin   = np.sin(noise_dphi) * noise_a1a2
        U           = 2.0 * scale_cross * (noise_cos * data[:, 2, :] + noise_sin * data[:, 3, :])
        V           = 2.0 * scale_cross * (-noise_sin * data[:, 2, :] + noise_cos * data[:, 3, :])
    else:
        U = np.zeros((nsamp, nchan), dtype=np.float32)
        V = np.zeros((nsamp, nchan), dtype=np.float32)

    return np.array([I, Q, U, V], dtype=np.float32)


def process_one_burst(
    h5_input_path,
    output_dir,
    noise_cal,
    t_cal,
    ra,
    dec,
    beam,
    cal_fits_path,
    cal_npz_path,
    down_time=None,
    down_freq=None,
    rfi_fft=True,
    time_crop_samples=None,
    target_time_reso=None,
    output_time_samples=None,
    noise_period_s=DEFAULT_NOISE_PERIOD_S,
):
    """读一个 burst h5, 定标 + 中心裁剪/下采样 + RFI 检测, 写 _cal.h5。

    down_time, down_freq : int or None
        保存下采样倍率. None = 自动取画图清晰倍率
        (频率目标 ~512 通道, 时间目标 ~49.152us×8).
        可显式传任意正整数；画图会在保存分辨率基础上选择
        可行的整数倍额外下采样。
    time_crop_samples : int or None
        在任何时间下采样之前，从输入数据的时间轴中心裁出多少个原始采样点。
        None 表示不裁剪。指定后，down_time 默认为 1，画图直接使用保存数据。
    target_time_reso : float or None
        保存数据的目标时间分辨率（秒）。函数会针对每个输入文件
        自动计算整数 down_time。与显式 down_time 互斥。
    output_time_samples : int or None
        定标和下采样完成后，从时间轴中心保留的采样点数。
        与 time_crop_samples 互斥。

    保存的 data 未减基线、未 NaN 掉 RFI, 给下游更大的处理空间;
    rfi_mask 作为辅助信息一并保存。
    """
    if target_time_reso is not None and down_time is not None:
        raise ValueError("target_time_reso 与 down_time 不能同时指定")
    if output_time_samples is not None and time_crop_samples is not None:
        raise ValueError("output_time_samples 与 time_crop_samples 不能同时指定")
    if target_time_reso is not None:
        target_time_reso = float(target_time_reso)
        if not np.isfinite(target_time_reso) or target_time_reso <= 0:
            raise ValueError("target_time_reso 必须是正的有限数")
    if output_time_samples is not None and int(output_time_samples) <= 0:
        raise ValueError("output_time_samples 必须是正整数")
    noise_period_s = float(noise_period_s)
    if not np.isfinite(noise_period_s) or noise_period_s <= 0:
        raise ValueError("noise_period_s 必须是正的有限数")

    basename = os.path.splitext(os.path.basename(h5_input_path))[0]
    out_h5   = os.path.join(output_dir, basename + "_cal.h5")
    if os.path.exists(out_h5):
        try:
            with h5py.File(out_h5, "r") as existing:
                required   = {"data", "freq", "rfi_mask", "gain", "gain_err"}
                provenance = {
                    "calibration_beam",
                    "calibration_fits",
                    "calibration_npz",
                    "noise_period_s",
                    "npol",
                }
                requested_crop = (
                    0 if time_crop_samples is None else int(time_crop_samples)
                )
                existing_crop = int(existing.attrs.get("time_crop_samples", 0))
                requested_output_samples = (
                    0 if output_time_samples is None else int(output_time_samples)
                )
                existing_output_samples = int(
                    existing.attrs.get("output_time_samples", 0)
                )
                target_matches = target_time_reso is None or np.isclose(
                    float(existing.attrs.get("time_reso", np.nan)),
                    target_time_reso,
                    rtol = 1e-9,
                    atol = 1e-12,
                )
                down_time_matches = down_time is None or int(
                    existing.attrs.get("down_time", 0)
                ) == int(down_time)
                down_freq_matches = down_freq is None or int(
                    existing.attrs.get("down_freq", 0)
                ) == int(down_freq)
                noise_period_matches = np.isclose(
                    float(existing.attrs.get("noise_period_s", np.nan)),
                    noise_period_s,
                    rtol = 0.0,
                    atol = 1e-12,
                )
                if (
                    required.issubset(existing.keys())
                    and provenance.issubset(existing.attrs.keys())
                    and requested_crop == existing_crop
                    and requested_output_samples == existing_output_samples
                    and target_matches
                    and down_time_matches
                    and down_freq_matches
                    and noise_period_matches
                ):
                    print(f"  [跳过] {out_h5} 已存在")
                    return
        except OSError:
            pass
        print(f"  [重写] {out_h5} 文件不完整")

    with h5py.File(h5_input_path, "r") as f:
        data_dataset = f["data"]
        freq_dataset = f["freq"]
        if not isinstance(data_dataset, h5py.Dataset):
            raise TypeError(f"H5 'data' is not a dataset: {h5_input_path}")
        if not isinstance(freq_dataset, h5py.Dataset):
            raise TypeError(f"H5 'freq' is not a dataset: {h5_input_path}")
        # 轴顺序: (时间, 偏振, 频率)
        raw_data = np.asarray(data_dataset[...])
        freq     = np.asarray(freq_dataset[...])
        attrs    = cast(dict[str, Any], dict(f.attrs))

    source_file_mjd     = float(np.asarray(attrs["file_mjd"]).item())
    source_start_sample = int(np.asarray(attrs["start_sample"]).item())
    time_reso_raw       = float(np.asarray(attrs["time_reso"]).item())
    nchan_raw           = int(np.asarray(attrs["nchan"]).item())
    input_npol          = int(raw_data.shape[1])
    declared_npol       = int(np.asarray(attrs.get("npol", input_npol)).item())
    if declared_npol != input_npol:
        raise ValueError(
            f"H5 npol 属性为 {declared_npol}，但 data 实际有 {input_npol} 路"
        )

    # Cut H5 的信号位于时间轴中心。这里在定标和下采样之前直接裁原始数组，
    # 既避免无谓计算，也确保 time_crop_samples 指的是原始时间采样点而不是
    # 下采样后的点数。奇数差值时，多出来的一个点留在右侧。
    source_nsamp   = int(raw_data.shape[0])
    crop_start_raw = 0
    crop_samples   = source_nsamp
    if time_crop_samples is not None:
        crop_samples = int(time_crop_samples)
        if crop_samples <= 0:
            raise ValueError("time_crop_samples 必须是正整数")
        if crop_samples > source_nsamp:
            raise ValueError(
                f"time_crop_samples={crop_samples} 超过输入时间长度 {source_nsamp}"
            )
        crop_start_raw = (source_nsamp - crop_samples) // 2
        raw_data       = raw_data[crop_start_raw : crop_start_raw + crop_samples]

    # 裁剪后第 0 点对应的绝对时间和观测采样号也必须同步平移，否则后续 TOA
    # 会相对原始观测错开 crop_start_raw 个采样点。
    file_mjd     = source_file_mjd + crop_start_raw * time_reso_raw / 86400.0
    start_sample = source_start_sample + crop_start_raw

    # 画图倍率: 频率 / 时间分别向 PLOT_TARGET_NCHAN / PLOT_TARGET_TIME_RESO 看齐,
    # 并把 plot 对齐到 save 的整数倍 (extra = plot // save 反推).
    auto_plot_dt = max(1, int(round(PLOT_TARGET_TIME_RESO / time_reso_raw)))
    auto_plot_df = max(1, int(round(nchan_raw / PLOT_TARGET_NCHAN)))
    if target_time_reso is not None:
        target_ratio = target_time_reso / time_reso_raw
        save_dt      = int(round(target_ratio))
        if save_dt <= 0 or not np.isclose(
            save_dt * time_reso_raw, target_time_reso, rtol=1e-9, atol=1e-12
        ):
            raise ValueError(
                f"target_time_reso={target_time_reso:.12g}s 不是原始时间"
                f"分辨率 {time_reso_raw:.12g}s 的整数倍"
            )
    else:
        save_dt = (
            (1 if time_crop_samples is not None else auto_plot_dt)
            if down_time is None
            else int(down_time)
        )
    save_df = auto_plot_df if down_freq is None else int(down_freq)
    if save_dt <= 0 or save_df <= 0:
        raise ValueError("down_time 和 down_freq 必须是正整数")
    if save_dt > raw_data.shape[0]:
        raise ValueError(f"down_time={save_dt} 超过当前时间长度 {raw_data.shape[0]}")
    if save_df > raw_data.shape[2]:
        raise ValueError(f"down_freq={save_df} 超过当前频率通道数 {raw_data.shape[2]}")

    if time_crop_samples is not None:
        # 原始时间轴裁剪产品的 JPG 与保存数据使用相同分辨率。
        plot_dt, plot_df = save_dt, save_df
        extra_dt         = extra_df = 1
    else:
        extra_dt         = max(1, auto_plot_dt // save_dt)
        extra_df         = max(1, auto_plot_df // save_df)
        plot_dt, plot_df = extra_dt * save_dt, extra_df * save_df

    za             = get_za(file_mjd, ra, dec)
    gain, gain_err = get_gain(za, beam, nchan_raw)
    iquv           = calibrate_to_iquv(raw_data, noise_cal, t_cal, gain)

    # 保存倍率下采样 (iquv / freq / gain 同步). nsamp / nchan 始终跟踪当前形状.
    _, nsamp, nchan = iquv.shape
    if save_dt > 1:
        nt = nsamp // save_dt
        iquv = np.nanmean(
            iquv[:, : nt * save_dt].reshape(4, nt, save_dt, nchan), axis=2
        )
        nsamp = nt
    if save_df > 1:
        nc = nchan // save_df
        iquv = np.nanmean(
            iquv[:, :, : nc * save_df].reshape(4, nsamp, nc, save_df), axis=3
        )
        freq        = freq[: nc * save_df].reshape(nc, save_df).mean(axis=1)
        gain_ds     = gain[: nc * save_df].reshape(nc, save_df).mean(axis=1)
        gain_err_ds = gain_err[: nc * save_df].reshape(nc, save_df).mean(axis=1)
        nchan       = nc
    else:
        gain_ds, gain_err_ds = gain, gain_err
    time_reso_save = time_reso_raw * save_dt

    # output_time_samples 在定标和下采样后执行中心裁剪。奇数差值时，
    # 多出的一个采样点留在右侧。
    nsamp_before_output_crop = nsamp
    output_crop_start        = 0
    if output_time_samples is not None:
        output_samples = int(output_time_samples)
        if output_samples > nsamp:
            raise ValueError(
                f"output_time_samples={output_samples} 超过下采样后时间长度 {nsamp}"
            )
        output_crop_start = (nsamp - output_samples) // 2
        iquv              = iquv[:, output_crop_start : output_crop_start + output_samples]
        nsamp             = output_samples

        # 当前文件第 0 个采样的绝对时间/原始采样号也随裁剪平移。
        output_crop_start_raw = output_crop_start * save_dt
        file_mjd += output_crop_start_raw * time_reso_raw / 86400.0
        start_sample += output_crop_start_raw
    else:
        output_samples        = nsamp
        output_crop_start_raw = 0

    nsamp_ds, nchan_ds = nsamp, nchan

    # RFI 检测: 整段当噪声, 在画图分辨率上找通道级 RFI (extra_dt × extra_df 倍下采样)
    noise_mask = np.ones(nsamp_ds, dtype=bool)
    rfi_channel, rfi_pixel = cal_rfi(
        iquv[0],
        noise_mask,
        down_time = extra_dt,
        down_freq = extra_df,
        fft       = rfi_fft,
    )
    rfi_mask                 = rfi_pixel.copy()
    rfi_mask[:, rfi_channel] = True

    os.makedirs(output_dir, exist_ok=True)

    # 仅画图: extra 倍下采样, 减基线, 抹 RFI (不回写 iquv); 结构对应上面的保存块.
    plot_I           = iquv[0].copy()
    plot_freq        = freq
    plot_I[rfi_mask] = np.nan
    plot_I -= np.nanmedian(plot_I, axis=0)
    nsamp_plot, nchan_plot = plot_I.shape
    if extra_dt > 1:
        nt = nsamp_plot // extra_dt
        plot_I = np.nanmean(
            plot_I[: nt * extra_dt].reshape(nt, extra_dt, nchan_plot), axis=1
        )
        nsamp_plot = nt
    if extra_df > 1:
        nc = nchan_plot // extra_df
        plot_I = np.nanmean(
            plot_I[:, : nc * extra_df].reshape(nsamp_plot, nc, extra_df), axis=2
        )
        plot_freq  = freq[: nc * extra_df].reshape(nc, extra_df).mean(axis=1)
        nchan_plot = nc
    time_reso_eff = time_reso_save * extra_dt

    fig = plt.figure(figsize=(5, 5))
    gs  = gridspec.GridSpec(4, 1, hspace=0)

    time_ms_plot = np.arange(nsamp_plot) * time_reso_eff * 1e3
    ax0          = fig.add_subplot(gs[0, 0])
    ax0.step(
        time_ms_plot, np.nanmean(plot_I, axis=1), where="mid", color="royalblue", lw=0.8
    )
    ax0.set_xlim(0, nsamp_plot * time_reso_eff * 1e3)
    ax0.set_xticks([])
    ax0.set_ylabel("Flux (Jy)")
    ax1        = fig.add_subplot(gs[1:, 0])
    vmin, vmax = np.nanpercentile(plot_I, [5, 95])
    ax1.imshow(
        plot_I.T,
        aspect = "auto",
        origin = "lower",
        cmap   = "mako",
        vmin   = vmin,
        vmax   = vmax,
        extent = (
            0.0,
            float(nsamp_plot * time_reso_eff * 1e3),
            float(plot_freq[0]),
            float(plot_freq[-1]),
        ),
    )
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Frequency (MHz)")
    fig.align_labels()
    plt.savefig(
        os.path.join(output_dir, basename + ".jpg"),
        dpi         = 200,
        bbox_inches = "tight",
        format      = "jpg",
        pil_kwargs  = {"quality": 95},
    )
    plt.close()

    rfi_frac = np.sum(rfi_mask) / rfi_mask.size
    out_attrs = {
        "file_mjd": file_mjd,
        "obs_start_mjd": attrs["obs_start_mjd"],
        "start_sample": start_sample,
        "toa_sec": attrs["toa_sec"],
        "time_reso_raw": time_reso_raw,
        "time_reso": time_reso_save,
        "down_time": save_dt,
        "down_freq": save_df,
        "plot_down_time": plot_dt,  # 记录画图实际用的倍率 (供下游查阅)
        "plot_down_freq": plot_df,
        "nchan_raw": nchan_raw,
        "nsamp_raw": source_nsamp,
        "nchan": nchan_ds,
        "nsamp": nsamp_ds,
        "dm": attrs["dm"],
        "beam": beam,
        "calibration_beam": beam,
        "calibration_fits": os.path.basename(cal_fits_path),
        "calibration_npz": os.path.basename(cal_npz_path),
        "noise_period_s": noise_period_s,
        "npol": input_npol,
        "ra": ra,
        "dec": dec,
        "rfi_fraction": rfi_frac,
    }
    if time_crop_samples is not None:
        out_attrs.update(
            {
                # 保留裁剪前的坐标，便于追溯；file_mjd/start_sample 则描述当前文件。
                "source_file_mjd": source_file_mjd,
                "source_start_sample": source_start_sample,
                "time_crop_start_raw": crop_start_raw,
                "time_crop_samples": crop_samples,
            }
        )
    if output_time_samples is not None:
        out_attrs.update(
            {
                # 记录“先下采样、后裁剪”的完整溯源信息。
                "source_file_mjd": source_file_mjd,
                "source_start_sample": source_start_sample,
                "target_time_reso": time_reso_save,
                "time_crop_stage": "post_downsample",
                "nsamp_before_output_crop": nsamp_before_output_crop,
                "output_time_samples": output_samples,
                "time_crop_start_downsampled": output_crop_start,
                "time_crop_start_raw": output_crop_start_raw,
            }
        )

    temp_path = out_h5 + ".tmp"
    try:
        with h5py.File(temp_path, "w") as f:
            f.create_dataset(
                "data",
                data             = iquv.astype(np.float32),
                compression      = "gzip",
                compression_opts = 4,
            )
            f.create_dataset("freq", data=freq.astype(np.float64))
            f.create_dataset("rfi_mask", data=rfi_mask)
            f.create_dataset("rfi_channel", data=rfi_channel)
            # 增益及其系统误差(K/Jy), 下游用于计算 flux / fluence 的系统误差
            f.create_dataset("gain", data=gain_ds.astype(np.float32))
            f.create_dataset("gain_err", data=gain_err_ds.astype(np.float32))
            f.attrs.update(out_attrs)
        os.replace(temp_path, out_h5)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    print(f"  [完成] {out_h5}  (RFI {rfi_frac * 100:.1f}%)")


if __name__ == "__main__":
    # ---- 配置参数 ----
    BURST_DIR           = "/path/to/after_data/FRB20201124A/20210526/"
    OUTPUT_DIR          = "/path/to/after_data/FRB20201124A/20210526/cal/"
    CAL_NPZ             = str(DEFAULT_CAL_NPZ)
    RA                  = "05h08m03.51s"
    DEC                 = "26d03m38.5s"
    DOWN_TIME           = None  # 保存时间下采样因子, None = 自动取画图清晰倍率
    DOWN_FREQ           = None  # 保存频率下采样因子, None = 自动取画图清晰倍率
    TIME_CROP_SAMPLES   = None  # 中心裁剪的原始时间采样数；None = 不裁剪
    TARGET_TIME_RESO    = None  # 目标有效时间分辨率（秒）；与 DOWN_TIME 互斥
    OUTPUT_TIME_SAMPLES = None  # 下采样后中心保留的时间点数
    RFI_FFT             = True  # True=FFT 最大幅度; False=熵
    NOISE_PERIOD_S      = DEFAULT_NOISE_PERIOD_S  # 噪声管名义总周期；On/Off 自动识别
    NUM_WORKERS         = 8

    # 1. 按波束分组 burst h5
    burst_h5_list = sorted(
        f
        for f in os.listdir(BURST_DIR)
        if f.endswith(".h5") and not f.endswith("_cal.h5")
    )
    if not burst_h5_list:
        print("未找到 burst h5 文件")
        exit()
    print(f"找到 {len(burst_h5_list)} 个 burst 文件")

    # 文件名中的 Mdd 段就是波束编号。
    beam_groups = defaultdict(list)
    for fname in burst_h5_list:
        m = re.search(r"M(\d{2})", fname)
        if m is None:
            raise ValueError(f"文件名缺少波束编号 Mxx: {fname}")
        beam = int(m.group(1))
        beam_groups[beam].append(os.path.join(BURST_DIR, fname))

    # 2. 匹配定标文件 / t_cal, 组装任务列表
    all_args = []
    for beam, h5_list in sorted(beam_groups.items()):
        cal_fits_path = find_cal_fits(BURST_DIR, beam)
        if cal_fits_path is None:
            raise FileNotFoundError(
                f"波束 M{beam:02d}: 未找到同波束定标文件，"
                f"不能定标 {len(h5_list)} 个 burst"
            )

        with fits.open(cal_fits_path) as f:
            nchan = int(cast(Any, f[1]).header["NCHAN"])

        noise_cal = fold_noise_cal(
            cal_fits_path,
            diagnostic_dir = OUTPUT_DIR,
            noise_period_s  = NOISE_PERIOD_S,
        )
        t_cal     = load_t_cal(CAL_NPZ, beam, nchan)

        print(
            f"  波束 M{beam:02d}: {len(h5_list)} 个 burst, "
            f"定标文件: {os.path.basename(cal_fits_path)}"
        )

        for h5_path in h5_list:
            all_args.append(
                (
                    h5_path,
                    OUTPUT_DIR,
                    noise_cal,
                    t_cal,
                    RA,
                    DEC,
                    beam,
                    cal_fits_path,
                    CAL_NPZ,
                    DOWN_TIME,
                    DOWN_FREQ,
                    RFI_FFT,
                    TIME_CROP_SAMPLES,
                    TARGET_TIME_RESO,
                    OUTPUT_TIME_SAMPLES,
                    NOISE_PERIOD_S,
                )
            )

    if not all_args:
        print("无可处理的 burst 文件")
        exit()

    # 3. 并行处理
    if NUM_WORKERS > 1 and len(all_args) > 1:
        with Pool(NUM_WORKERS) as pool:
            pool.starmap(process_one_burst, all_args)
    else:
        for args in all_args:
            process_one_burst(*args)

    print("全部完成")

# fmt: on
