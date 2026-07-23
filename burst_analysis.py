"""
爆发分析流水线 — 编排器。

输入是 calibration.py 输出的 _cal.h5, 其 attrs['bursts'] 存了 burst_detect.py
写进去的区域列表(JSON 字符串)。流程:

  1. 加载 IQUV + freq + bursts。
  2. 用非 burst 时段作为 noise_mask, 先按通道减噪声区中值基线。
  3. 在减完基线的全部 Stokes 上调用 rfi_utils.cal_rfi, 再叠加局部鲁棒通道
     统计和 H5 已保存的 calibration/detection 通道 mask。默认只应用通道级 RFI,
     不把逐像素 mask 用到信号上；需要复现旧行为时才显式启用 pixel mask。
  4. 在每个检测框内用 Stokes I 自动选取超过主峰分数阈值且超过噪声阈值的
     强时间采样点，RM/偏振只使用这些采样点，基本物理量和 DM 仍使用完整框。
  5. 画一张总览动态谱 + burst 区域高亮。
  6. 对每个 burst: calc_burst_properties + analyze_dm + analyze_pol, 汇总为行。
  7. 批量写 CSV。

time_reso 在 calibration.py 阶段已经是 raw × down_time 的有效分辨率, 下游计算
fluence / TOA / 宽度时直接用 time_reso 即可, 不要再乘 down_time。
"""

import os
import glob
import json
import argparse

import numpy as np
import h5py
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec

from rfi_utils       import cal_rfi, robust_channel_mask
from burst_properties import calc_burst_properties
from burst_dm         import analyze_dm
from burst_pol        import analyze_pol, plot_polarization


# ============================================================
# 数据下采样: 把 _cal.h5 里已经下采样过的数据再做一次额外下采样
# ============================================================

def _downsample_mean(arr, axis, factor):
    """沿 axis 用 nanmean 把 arr 整数倍下采样, factor=1 时原样返回。"""
    if factor <= 1:
        return arr
    n_keep = (arr.shape[axis] // factor) * factor
    sl = [slice(None)] * arr.ndim
    sl[axis] = slice(0, n_keep)
    arr = arr[tuple(sl)]
    new_shape = list(arr.shape)
    new_shape[axis] = arr.shape[axis] // factor
    new_shape.insert(axis + 1, factor)
    return np.nanmean(arr.reshape(new_shape), axis=axis + 1)


def _downsample_channel_mask(mask, factor, target_size):
    """频率下采样时按组取 any，保持已有通道 RFI 标记。"""
    if mask is None:
        return np.zeros(target_size, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if factor <= 1:
        result = mask[:target_size].copy()
    else:
        n_keep = (mask.size // factor) * factor
        result = mask[:n_keep].reshape(-1, factor).any(axis=1)
        result = result[:target_size]
    if result.size < target_size:
        result = np.pad(result, (0, target_size - result.size),
                        constant_values=False)
    return result


def _apply_extra_downsample(iquv, freq, gain, gain_err,
                            burst_regions, time_reso,
                            extra_dt, extra_df):
    """对 _cal.h5 里读出的数据再做 extra_dt × extra_df 下采样。

    同时按比例缩放 burst_regions 的时间/频率索引, 以及 time_reso。
    """
    # 时间轴 (iquv axis=1)
    if extra_dt > 1:
        iquv = _downsample_mean(iquv, axis=1, factor=extra_dt)
        time_reso = time_reso * extra_dt
        new_nsamp = iquv.shape[1]    # _downsample_mean 可能会截尾, 按新长度 clamp
        for r in burst_regions:
            r['time_start'] = min(r['time_start'] // extra_dt, new_nsamp)
            r['time_end']   = min(-(-r['time_end'] // extra_dt), new_nsamp)

    # 频率轴 (iquv axis=2)
    if extra_df > 1:
        iquv = _downsample_mean(iquv, axis=2, factor=extra_df)
        n_keep = (freq.shape[0] // extra_df) * extra_df
        freq = np.nanmean(freq[:n_keep].reshape(-1, extra_df), axis=1)
        if gain is not None:
            gain = np.nanmean(gain[:n_keep].reshape(-1, extra_df), axis=1)
        if gain_err is not None:
            gain_err = np.nanmean(gain_err[:n_keep].reshape(-1, extra_df), axis=1)
        new_nchan = iquv.shape[2]
        for r in burst_regions:
            r['freq_start'] = min(r['freq_start'] // extra_df, new_nchan)
            r['freq_end']   = min(-(-r['freq_end'] // extra_df), new_nchan)

    return iquv, freq, gain, gain_err, burst_regions, time_reso


def plot_dynamic_spectrum(stokes_I, freq, time_reso, burst_regions, save_path):
    """动态谱 + 时间轮廓, 高亮爆发时段。"""
    data        = stokes_I
    nsamp, _    = data.shape
    time_ms     = np.arange(nsamp) * time_reso * 1e3

    fig = plt.figure(figsize=(5, 5))
    gs  = gridspec.GridSpec(4, 1, hspace=0)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.step(time_ms, np.nanmean(data, axis=1), where='mid', color='royalblue', lw=0.8)
    ax0.set_xlim(time_ms[0], time_ms[-1])
    ax0.set_xticks([])
    ax0.set_ylabel('Flux (Jy)')
    for r in burst_regions:
        ax0.axvspan(r['time_start'] * time_reso * 1e3,
                    r['time_end']   * time_reso * 1e3,
                    alpha=0.2, color='steelblue')

    ax1 = fig.add_subplot(gs[1:, 0])
    vmin, vmax = np.nanpercentile(data, [2, 98])
    ax1.imshow(data.T, aspect='auto', origin='lower', cmap='mako',
               vmin=vmin, vmax=vmax,
               extent=[time_ms[0], time_ms[-1], freq[0], freq[-1]])
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Frequency (MHz)')

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def _select_strong_time_samples(stokes_I, freq_index, noise_mask,
                                burst_region, peak_fraction=0.5,
                                min_snr=5.0):
    """在检测框内选择用于 RM 的强 Stokes-I 时间采样点。

    选择规则只依赖 I 轮廓，不查看 Q/U 或 RM 曲线，避免为得到某个预期 RM
    而调时间窗。轮廓先用三点三角核轻微平滑，然后保留同时高于主峰指定
    分数和噪声阈值的所有采样点；多峰信号允许得到不连续布尔门。
    """
    nsamp = stokes_I.shape[0]
    mask = np.zeros(nsamp, dtype=bool)
    ts = max(0, min(nsamp, int(burst_region['time_start'])))
    te = max(0, min(nsamp, int(burst_region['time_end'])))
    if te <= ts or not np.any(freq_index):
        return mask, {
            'peak_sample': ts,
            'peak_snr': 0.0,
            'noise_sigma': np.nan,
            'threshold': np.nan,
            'sample_count': 0,
        }

    profile = np.nanmean(stokes_I[:, freq_index], axis=1)
    finite_noise = profile[noise_mask & np.isfinite(profile)]
    if finite_noise.size:
        noise_center = float(np.nanmedian(finite_noise))
        noise_sigma = float(1.4826 * np.nanmedian(
            np.abs(finite_noise - noise_center)))
        if not np.isfinite(noise_sigma) or noise_sigma <= 0:
            noise_sigma = float(np.nanstd(finite_noise))
    else:
        noise_center = 0.0
        noise_sigma = 0.0
    if not np.isfinite(noise_sigma) or noise_sigma <= 0:
        noise_sigma = np.finfo(np.float64).eps

    net = profile - noise_center
    net_filled = np.nan_to_num(net, nan=0.0)
    smooth = np.convolve(net_filled, np.array([0.25, 0.5, 0.25]),
                         mode='same')
    region_smooth = smooth[ts:te]
    peak_offset = int(np.argmax(region_smooth))
    peak_sample = ts + peak_offset
    peak_value = float(region_smooth[peak_offset])
    fraction = float(np.clip(peak_fraction, 0.0, 1.0))
    threshold = max(fraction * peak_value, float(min_snr) * noise_sigma)
    mask[ts:te] = region_smooth >= threshold
    if not np.any(mask):
        mask[peak_sample] = True

    return mask, {
        'peak_sample': peak_sample,
        'peak_snr': peak_value / noise_sigma,
        'noise_sigma': noise_sigma,
        'threshold': threshold,
        'sample_count': int(np.count_nonzero(mask)),
    }


def _true_runs(mask):
    """把布尔数组转换为左闭右开的连续 True 区间。"""
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts, ends))


def plot_rm_selection(stokes_I, freq, time_reso, burst_region,
                      rm_time_mask, rm_freq_index, save_path):
    """保存 RM 所用时间门和频率通道的可审查诊断图。"""
    nsamp = stokes_I.shape[0]
    time_ms = np.arange(nsamp) * time_reso * 1e3
    if np.any(rm_freq_index):
        profile = np.nanmean(stokes_I[:, rm_freq_index], axis=1)
    else:
        profile = np.nanmean(stokes_I, axis=1)

    fig = plt.figure(figsize=(6, 5))
    gs = gridspec.GridSpec(4, 1, hspace=0)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.step(time_ms, profile, where='mid', color='royalblue', lw=0.8)
    ax0.axvspan(burst_region['time_start'] * time_reso * 1e3,
                burst_region['time_end'] * time_reso * 1e3,
                color='steelblue', alpha=0.15, label='detection box')
    for start, end in _true_runs(rm_time_mask):
        ax0.axvspan(start * time_reso * 1e3,
                    end * time_reso * 1e3,
                    color='limegreen', alpha=0.35)
    ax0.set_xlim(time_ms[0], time_ms[-1])
    ax0.set_xticks([])
    ax0.set_ylabel('Flux (Jy)')

    ax1 = fig.add_subplot(gs[1:, 0])
    finite = stokes_I[np.isfinite(stokes_I)]
    if finite.size:
        vmin, vmax = np.nanpercentile(finite, [2, 98])
    else:
        vmin, vmax = 0.0, 1.0
    ax1.imshow(stokes_I.T, aspect='auto', origin='lower', cmap='mako',
               vmin=vmin, vmax=vmax,
               extent=[time_ms[0], time_ms[-1], freq[0], freq[-1]])
    for start, end in _true_runs(rm_time_mask):
        ax1.axvspan(start * time_reso * 1e3,
                    end * time_reso * 1e3,
                    color='limegreen', alpha=0.22)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Frequency (MHz)')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def analyze_one_file(cal_h5_path, output_dir,
                     rfi_fft=False, rfi_channel_only=True,
                     rfi_channel_sigma=6.0,
                     rfi_channel_window=31,
                     rfi_channel_grow=1,
                     dm_range=10.0, dm_step=0.1, dm_snr_threshold=5.0,
                     rm_min=-50000, rm_max=50000, n_rm=20000,
                     rm_peak_fraction=0.5, rm_min_time_snr=5.0,
                     rm_freq_min=None, rm_freq_max=None,
                     strongest_burst_only=False,
                     n_boot=200,
                     target_down_time=None, target_down_freq=None):
    """分析一个 _cal.h5 文件中的所有爆发, 返回结果行列表。

    rfi_channel_only
        True 时只应用通道级 RFI, 忽略 cal_rfi 返回的像素级 RFI。
    rfi_channel_sigma / window / grow
        全 Stokes 局部鲁棒通道 RFI 的阈值、频率窗口和邻道扩展数。
    rm_peak_fraction / rm_min_time_snr
        RM 时间门只保留检测框内同时超过主峰分数阈值和噪声 S/N 阈值的
        Stokes-I 采样点；设 rm_peak_fraction=0 可退回仅按 S/N 选择。
    rm_freq_min / rm_freq_max
        可选的 RM 专用频率边界，不影响 DM 和能量测量。
    strongest_burst_only
        每个 H5 只对峰值 S/N 最高的检测区域出一行，适合一文件一个主信号的
        RM 复测；噪声基线仍排除该文件中的全部已确认 burst 区域。

    target_down_time / target_down_freq
        分析时希望使用的 "相对原始 raw 数据" 的下采样倍率。整除 _cal.h5
        里 attrs['down_time/freq'] 的余数必须为 0; 不允许小于已存的倍率
        (那种情况要求重新跑 calibration 阶段降低保存倍率)。两个参数都
        默认 None, 即沿用 _cal.h5 已有分辨率不再额外下采样。
    """
    basename  = os.path.splitext(os.path.basename(cal_h5_path))[0]
    burst_dir = os.path.join(output_dir, basename)

    # ---- 1. 加载 ----
    with h5py.File(cal_h5_path, 'r') as f:
        iquv         = f['data'][:]                             # (4, nsamp, nchan)
        freq         = f['freq'][:]                             # (nchan,)
        gain         = f['gain'][:]     if 'gain'     in f else None
        gain_err     = f['gain_err'][:] if 'gain_err' in f else None
        stored_cal_channel = (f['rfi_channel'][:].astype(bool)
                              if 'rfi_channel' in f else None)
        stored_burst_channel = (f['burst_rfi_channel'][:].astype(bool)
                                if 'burst_rfi_channel' in f else None)
        attrs        = dict(f.attrs)
    # 只读取已有的一维通道 mask，不读取 calibration/detection 的逐像素 mask。

    burst_regions = json.loads(attrs['bursts']) if 'bursts' in attrs else []
    if not burst_regions:
        print(f'  [{basename}] 无爆发可分析')
        return []

    os.makedirs(burst_dir, exist_ok=True)

    time_reso  = float(attrs['time_reso'])    # 已经是下采样后的有效分辨率
    file_mjd   = float(attrs['file_mjd'])
    dm_zero    = float(attrs['dm'])

    # ---- 1.5 额外下采样 (相对 _cal.h5 已有的倍率再做一次) ----
    saved_dt = int(attrs.get('down_time', 1))
    saved_df = int(attrs.get('down_freq', 1))
    tgt_dt = saved_dt if target_down_time is None else int(target_down_time)
    tgt_df = saved_df if target_down_freq is None else int(target_down_freq)

    if tgt_dt < saved_dt or tgt_df < saved_df:
        raise ValueError(
            f'target_down_time/freq=({tgt_dt},{tgt_df}) 小于 _cal.h5 已存的'
            f' ({saved_dt},{saved_df}); 请重做 calibration 降低保存倍率')
    if tgt_dt % saved_dt != 0 or tgt_df % saved_df != 0:
        raise ValueError(
            f'target_down_time/freq=({tgt_dt},{tgt_df}) 必须能被 _cal.h5 已存的'
            f' ({saved_dt},{saved_df}) 整除')

    extra_dt = tgt_dt // saved_dt
    extra_df = tgt_df // saved_df
    if extra_dt > 1 or extra_df > 1:
        print(f'  [{basename}] 额外下采样 (extra_dt={extra_dt}, extra_df={extra_df})')
        (iquv, freq, gain, gain_err, burst_regions,
         time_reso) = _apply_extra_downsample(
            iquv, freq, gain, gain_err, burst_regions, time_reso,
            extra_dt, extra_df)

    stored_cal_channel = _downsample_channel_mask(
        stored_cal_channel, extra_df, iquv.shape[2])
    stored_burst_channel = _downsample_channel_mask(
        stored_burst_channel, extra_df, iquv.shape[2])

    # 额外下采样 + clamp 之后, 某些窄 burst 可能塌成空区间 (ts >= te 或 fs >= fe),
    # 进入后续 analyze_pol 会在 `ts_idx[0]` 处 IndexError. 这里直接过滤掉.
    n_before = len(burst_regions)
    burst_regions = [r for r in burst_regions
                     if r['time_end'] > r['time_start']
                     and r['freq_end'] > r['freq_start']]
    if len(burst_regions) < n_before:
        print(f'  [{basename}] 下采样后 {n_before - len(burst_regions)} 个 '
              f'爆发区间退化为空, 跳过')
    if not burst_regions:
        return []

    nsamp = iquv.shape[1]

    # ---- 2. 噪声掩码: 爆发之外的时段 ----
    noise_mask = np.ones(nsamp, dtype=bool)
    for r in burst_regions:
        noise_mask[r['time_start']:r['time_end']] = False

    # ---- 3. 减基线: 按通道减噪声区的中值 ----
    # 先减基线再找 RFI, 避免通道间固定基线结构影响通道级 RFI 判定。
    baseline = np.nanmedian(iquv[:, noise_mask, :], axis=1, keepdims=True)
    iquv     = iquv - baseline
    I, V     = iquv[0], iquv[3]

    # ---- 4. 精细 RFI 标记: 在减完基线、target-down 后的数据上重新计算 mask ----
    # RM 直接使用 Q/U，因此四个 Stokes 都参与通道判定；只要任一 Stokes
    # 显示持续污染，就整频道屏蔽。逐像素结果仅在显式请求旧模式时应用。
    cal_channels = []
    pixel_masks = []
    for plane in iquv:
        channel_i, pixel_i = cal_rfi(
            plane, noise_mask, down_time=1, down_freq=1, fft=rfi_fft)
        cal_channels.append(channel_i)
        pixel_masks.append(pixel_i)
    recalculated_channel = np.logical_or.reduce(cal_channels)
    robust_channel = robust_channel_mask(
        iquv, noise_mask,
        sigma=rfi_channel_sigma,
        local_window=rfi_channel_window,
        grow=rfi_channel_grow)
    rfi_channel = (stored_cal_channel | stored_burst_channel |
                   recalculated_channel | robust_channel)

    rfi_mask = np.zeros_like(pixel_masks[0], dtype=bool)
    if not rfi_channel_only:
        rfi_mask |= np.logical_or.reduce(pixel_masks)
    rfi_mask[:, rfi_channel] = True
    mode = '仅通道级' if rfi_channel_only else '通道级 + 像素级'
    print(f'  [{basename}] RFI 模式: {mode}; '
          f'坏通道 {np.count_nonzero(rfi_channel)}/{rfi_channel.size} '
          f'(cal={np.count_nonzero(stored_cal_channel)}, '
          f'detect={np.count_nonzero(stored_burst_channel)}, '
          f'重算={np.count_nonzero(recalculated_channel)}, '
          f'增强={np.count_nonzero(robust_channel)}), '
          f'总掩码像素 {np.count_nonzero(rfi_mask)}/{rfi_mask.size}')

    pd.DataFrame({
        'channel_index': np.arange(freq.size),
        'frequency_mhz': freq,
        'stored_calibration': stored_cal_channel,
        'stored_detection': stored_burst_channel,
        'recalculated_all_stokes': recalculated_channel,
        'robust_local_statistics': robust_channel,
        'final_channel_mask': rfi_channel,
    }).to_csv(os.path.join(burst_dir, 'rfi_channels.csv'), index=False)

    # 所有 Stokes 对齐 NaN
    for s in iquv:
        s[rfi_mask] = np.nan
    I, Q, U, V = iquv[0], iquv[1], iquv[2], iquv[3]

    # ---- 5. 总览动态谱 ----
    plot_dynamic_spectrum(I, freq, time_reso, burst_regions,
                          os.path.join(burst_dir, 'dynamic_spectrum.png'))

    # ---- 6. 逐 burst 分析 ----
    # 全局 RFI 通道: 所有时间点都 NaN → 频率分析要跳过
    rfi_chan = np.all(np.isnan(I), axis=0)
    nchan    = I.shape[1]

    indexed_regions = list(enumerate(burst_regions))
    if strongest_burst_only and len(indexed_regions) > 1:
        scored = []
        for original_idx, candidate in indexed_regions:
            candidate_freq = np.zeros(nchan, dtype=bool)
            candidate_freq[candidate['freq_start']:candidate['freq_end']] = True
            candidate_freq[rfi_chan] = False
            _, candidate_info = _select_strong_time_samples(
                I, candidate_freq, noise_mask, candidate,
                peak_fraction=rm_peak_fraction,
                min_snr=rm_min_time_snr)
            scored.append((candidate_info['peak_snr'], original_idx,
                           candidate))
        _, keep_idx, keep_region = max(scored, key=lambda item: item[0])
        indexed_regions = [(keep_idx, keep_region)]
        print(f'  [{basename}] strongest-burst-only: 保留爆发 {keep_idx}, '
              f'文件内原有 {len(burst_regions)} 个区域')

    results = []
    # 跨爆发合并 PA: 每个爆发只在自己时间窗口内有 PAV/PAE, 窗口外为 NaN.
    # 复刻老代码 plot_spec(..., comb=True) 的 PDF 输出.
    PAV_ALL, PAE_ALL = [], []
    last_pol_profiles = None     # (prof_I, prof_L, prof_V) 取最后一个成功的爆发
    for bi, region in indexed_regions:
        print(f'  [{basename}] 分析爆发 {bi}...')

        ts, te = region['time_start'], region['time_end']
        fs, fe = region['freq_start'], region['freq_end']

        # 该爆发的有效频率通道 = 指定频率范围 ∩ 非 RFI
        freq_index       = np.zeros(nchan, dtype=bool)
        freq_index[fs:fe] = True
        freq_index[rfi_chan] = False

        # RM/偏振可以进一步限制频率范围，但不改变能量和 DM 的频率口径。
        rm_freq_index = freq_index.copy()
        if rm_freq_min is not None:
            rm_freq_index &= freq >= float(rm_freq_min)
        if rm_freq_max is not None:
            rm_freq_index &= freq <= float(rm_freq_max)

        # 只用 I 轮廓中的强时间采样点测 RM；允许多峰产生不连续布尔门。
        rm_time_mask, rm_time_info = _select_strong_time_samples(
            I, rm_freq_index, noise_mask, region,
            peak_fraction=rm_peak_fraction,
            min_snr=rm_min_time_snr)
        rm_indices = np.flatnonzero(rm_time_mask)
        print(f'    RM 时间采样: {rm_indices.tolist()} '
              f'({rm_indices.size} 点, peak S/N={rm_time_info["peak_snr"]:.1f}); '
              f'有效频率通道 {np.count_nonzero(rm_freq_index)}')
        plot_rm_selection(
            I, freq, time_reso, region, rm_time_mask, rm_freq_index,
            os.path.join(burst_dir, f'burst{bi}_rm_window.png'))

        # Stokes I 物理量(TOA, flux, fluence, width, 带宽)
        # 传入 freq_index 使 TOA 峰值在爆发频段内搜索;
        # 传入 gain/gain_err 以计算流量/能量的系统误差.
        props = calc_burst_properties(I, freq, time_reso, file_mjd,
                                      region, noise_mask, rfi_mask,
                                      freq_index,
                                      gain=gain, gain_err=gain_err,
                                      n_boot=n_boot)

        # DM 精化
        dm_out = analyze_dm(I, freq, time_reso, dm_zero, region,
                            burst_dir, bi,
                            dm_range=dm_range, dm_step=dm_step,
                            snr_threshold=dm_snr_threshold)

        # 偏振 (analyze_pol 返回 (标量 dict, PA 数组 dict))
        try:
            pol_out, pa_arrays = analyze_pol(
                I, Q, U, V, freq, time_reso,
                rm_time_mask, rm_freq_index, noise_mask,
                burst_dir, bi,
                rm_min=rm_min, rm_max=rm_max, n_rm=n_rm)
        except Exception as e:
            print(f'    偏振分析失败: {e}')
            pol_out = {
                'rm': 0.0, 'rm_err': np.nan, 'rm_significance': 0.0,
                'linear_frac': 0.0, 'linear_frac_err': 0.0,
                'circular_frac': 0.0, 'circular_frac_err': 0.0,
                'center_freq': float(np.mean(freq)),
            }
            pa_arrays = None

        # 跨爆发合并 PDF 所需的 PA 数组 + 最后一个 burst 的轮廓
        if pa_arrays is not None:
            PAV_ALL.append(pa_arrays['PAV'])
            PAE_ALL.append(pa_arrays['PAE'])
            last_pol_profiles = (pa_arrays['profile_I'],
                                 pa_arrays['profile_L'],
                                 pa_arrays['profile_V'])

        # 汇总一行
        row = {
            'file_name': os.path.basename(cal_h5_path),
            'burst_idx': bi,
            'rm_time_indices': ';'.join(str(int(i)) for i in rm_indices),
            'rm_time_sample_count': int(rm_indices.size),
            'rm_time_start_ms': (float(rm_indices[0] * time_reso * 1e3)
                                 if rm_indices.size else np.nan),
            'rm_time_end_ms': (float((rm_indices[-1] + 1) * time_reso * 1e3)
                               if rm_indices.size else np.nan),
            'rm_time_peak_snr': float(rm_time_info['peak_snr']),
            'rm_peak_fraction': float(rm_peak_fraction),
            'rm_min_time_snr': float(rm_min_time_snr),
            'rm_freq_channel_count': int(np.count_nonzero(rm_freq_index)),
            'rm_freq_low_mhz': (float(np.min(freq[rm_freq_index]))
                                if np.any(rm_freq_index) else np.nan),
            'rm_freq_high_mhz': (float(np.max(freq[rm_freq_index]))
                                 if np.any(rm_freq_index) else np.nan),
            'rfi_channel_count': int(np.count_nonzero(rfi_channel)),
            'rfi_pixel_mask_applied': bool(not rfi_channel_only),
        }
        row.update(props)
        row.update(dm_out)
        row.update(pol_out)
        results.append(row)

    # ---- 7. 跨爆发合并 PA 的 PDF (复刻 plot_spec(..., comb=True)) ----
    if PAV_ALL and last_pol_profiles is not None:
        PAV_comb = np.full_like(PAV_ALL[0], np.nan)
        PAE_comb = np.full_like(PAE_ALL[0], np.nan)
        for pav_i, pae_i in zip(PAV_ALL, PAE_ALL):
            ok = ~np.isnan(pav_i)
            PAV_comb[ok] = pav_i[ok]
            PAE_comb[ok] = pae_i[ok]
        prof_I, prof_L, prof_V = last_pol_profiles
        PAT = np.arange(nsamp, dtype=np.float64)
        plot_polarization(
            PAT, PAV_comb, PAE_comb, prof_I, prof_L, prof_V,
            I, freq, time_reso,
            os.path.join(burst_dir, 'combined_polarization.pdf'),
            fmt='pdf')

    print(f'  [{basename}] 共分析 {len(results)} 个爆发')
    return results


def analyze_all(cal_dir, output_dir,
                rfi_fft=False, rfi_channel_only=True,
                rfi_channel_sigma=6.0,
                rfi_channel_window=31,
                rfi_channel_grow=1,
                dm_range=10.0, dm_step=0.1, dm_snr_threshold=5.0,
                rm_min=-50000, rm_max=50000, n_rm=20000,
                rm_peak_fraction=0.5, rm_min_time_snr=5.0,
                rm_freq_min=None, rm_freq_max=None,
                strongest_burst_only=False,
                n_boot=200,
                target_down_time=None, target_down_freq=None):
    """批量分析 cal_dir 下所有 _cal.h5, 汇总写 CSV。"""
    h5_files = sorted(glob.glob(os.path.join(cal_dir, '*_cal.h5')))
    if not h5_files:
        print(f'未找到 _cal.h5: {cal_dir}')
        return pd.DataFrame()

    os.makedirs(output_dir, exist_ok=True)
    all_results = []
    for h5_path in h5_files:
        all_results.extend(analyze_one_file(
            h5_path, output_dir,
            rfi_fft=rfi_fft,
            rfi_channel_only=rfi_channel_only,
            rfi_channel_sigma=rfi_channel_sigma,
            rfi_channel_window=rfi_channel_window,
            rfi_channel_grow=rfi_channel_grow,
            dm_range=dm_range, dm_step=dm_step,
            dm_snr_threshold=dm_snr_threshold,
            rm_min=rm_min, rm_max=rm_max, n_rm=n_rm,
            rm_peak_fraction=rm_peak_fraction,
            rm_min_time_snr=rm_min_time_snr,
            rm_freq_min=rm_freq_min, rm_freq_max=rm_freq_max,
            strongest_burst_only=strongest_burst_only,
            n_boot=n_boot,
            target_down_time=target_down_time,
            target_down_freq=target_down_freq))

    if all_results:
        df       = pd.DataFrame(all_results)
        csv_path = os.path.join(output_dir, 'burst_results.csv')
        df.to_csv(csv_path, index=False)
        print(f'\n[OK] 已保存 {len(df)} 个爆发结果: {csv_path}')
    else:
        df = pd.DataFrame()
        print('\n[警告] 无爆发结果可保存')
    return df


if __name__ == '__main__':

    # 默认配置
    CAL_DIR           = './cal/'
    OUTPUT_DIR        = './analysis/'
    RFI_FFT           = False    # True = FFT 法; False = 熵法
    RFI_CHANNEL_SIGMA = 6.0
    RFI_CHANNEL_WINDOW = 31
    RFI_CHANNEL_GROW  = 1
    DM_RANGE          = 10.0
    DM_STEP           = 0.1
    DM_SNR_THRESHOLD  = 5.0
    RM_MIN            = -50000
    RM_MAX            = 50000
    N_RM              = 20000
    RM_PEAK_FRACTION  = 0.5
    RM_MIN_TIME_SNR   = 5.0
    N_BOOT            = 200

    parser = argparse.ArgumentParser(description='爆发分析流水线')
    parser.add_argument('--cal-dir',          default=CAL_DIR,          help='定标 h5 目录')
    parser.add_argument('--output-dir',       default=OUTPUT_DIR,       help='输出目录')
    parser.add_argument('--rfi-fft',          default=RFI_FFT,          action='store_true', help='RFI 改用 FFT 法')
    rfi_mode = parser.add_mutually_exclusive_group()
    rfi_mode.add_argument('--rfi-channel-only', dest='rfi_channel_only',
                          action='store_true', default=True,
                          help='只应用通道级 RFI, 忽略像素级 RFI (默认)')
    rfi_mode.add_argument('--rfi-pixel-mask', dest='rfi_channel_only',
                          action='store_false',
                          help='显式恢复通道级 + 像素级 RFI mask')
    parser.add_argument('--rfi-channel-sigma', default=RFI_CHANNEL_SIGMA,
                        type=float, help='局部鲁棒通道 RFI 阈值 (默认 6)')
    parser.add_argument('--rfi-channel-window', default=RFI_CHANNEL_WINDOW,
                        type=int, help='局部通道统计的频率窗口 (默认 31)')
    parser.add_argument('--rfi-channel-grow', default=RFI_CHANNEL_GROW,
                        type=int, help='坏通道向两侧扩展的频道数 (默认 1)')
    parser.add_argument('--dm-range',         default=DM_RANGE,         type=float, help='DM 搜索范围')
    parser.add_argument('--dm-step',          default=DM_STEP,          type=float, help='DM 搜索步长')
    parser.add_argument('--dm-snr-threshold', default=DM_SNR_THRESHOLD, type=float, help='DM 低SNR压制阈值')
    parser.add_argument('--rm-min',           default=RM_MIN,           type=float, help='RM 下限')
    parser.add_argument('--rm-max',           default=RM_MAX,           type=float, help='RM 上限')
    parser.add_argument('--n-rm',             default=N_RM,             type=int,   help='RM 试验点数')
    parser.add_argument('--rm-peak-fraction', default=RM_PEAK_FRACTION,
                        type=float, help='RM 时间门的主峰分数阈值 (默认 0.5)')
    parser.add_argument('--rm-min-time-snr', default=RM_MIN_TIME_SNR,
                        type=float, help='RM 时间门的最低 I 轮廓 S/N (默认 5)')
    parser.add_argument('--rm-freq-min', default=None, type=float,
                        help='RM 专用最低频率 MHz (默认使用检测框下限)')
    parser.add_argument('--rm-freq-max', default=None, type=float,
                        help='RM 专用最高频率 MHz (默认使用检测框上限)')
    parser.add_argument('--strongest-burst-only', action='store_true',
                        help='每个 H5 只分析峰值 S/N 最高的检测区域')
    parser.add_argument('--n-boot',           default=N_BOOT,           type=int,   help='Bootstrap 次数')
    parser.add_argument('--target-down-time', default=None, type=int,
                        help='相对原始数据的目标时间下采样倍率, 必须是 _cal.h5 已存倍率的整数倍')
    parser.add_argument('--target-down-freq', default=None, type=int,
                        help='相对原始数据的目标频率下采样倍率, 必须是 _cal.h5 已存倍率的整数倍')
    args = parser.parse_args()

    df = analyze_all(
        args.cal_dir, args.output_dir,
        rfi_fft=args.rfi_fft,
        rfi_channel_only=args.rfi_channel_only,
        rfi_channel_sigma=args.rfi_channel_sigma,
        rfi_channel_window=args.rfi_channel_window,
        rfi_channel_grow=args.rfi_channel_grow,
        dm_range=args.dm_range, dm_step=args.dm_step,
        dm_snr_threshold=args.dm_snr_threshold,
        rm_min=args.rm_min, rm_max=args.rm_max, n_rm=args.n_rm,
        rm_peak_fraction=args.rm_peak_fraction,
        rm_min_time_snr=args.rm_min_time_snr,
        rm_freq_min=args.rm_freq_min, rm_freq_max=args.rm_freq_max,
        strongest_burst_only=args.strongest_burst_only,
        n_boot=args.n_boot,
        target_down_time=args.target_down_time,
        target_down_freq=args.target_down_freq)

    if not df.empty:
        print(f'\n汇总:')
        print(f'  爆发总数: {len(df)}')
        print(f'  SNR 范围: {df["snr"].min():.1f} – {df["snr"].max():.1f}')
        print(f'  DM 范围: {df["dm"].min():.3f} – {df["dm"].max():.3f}')
