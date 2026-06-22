"""
爆发分析流水线 — 编排器。

输入是 calibration.py 输出的 _cal.h5, 其 attrs['bursts'] 存了 burst_detect.py
写进去的区域列表(JSON 字符串)。流程:

  1. 加载 IQUV + freq + bursts。
  2. 用非 burst 时段作为 noise_mask, 先按通道减噪声区中值基线。
  3. 在减完基线的 Stokes I 和 V 上分别调用 rfi_utils.cal_rfi 做二次精细 RFI
     标记并取并集(比 calibration 阶段更干净, 因为噪声区不含信号)。如果需要改变
     RFI 检测分辨率, 先用 target_down_time / target_down_freq 对 analysis
     数据整体下采样; RFI 函数直接在当前 analysis 数据分辨率上运行。
  4. 只使用 analysis 阶段由 noise_mask 得到的 RFI mask, 所有 Stokes 对应
     位置置 NaN。
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

from rfi_utils       import cal_rfi
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


def analyze_one_file(cal_h5_path, output_dir,
                     rfi_fft=False,
                     dm_range=10.0, dm_step=0.1, dm_snr_threshold=5.0,
                     rm_min=-50000, rm_max=50000, n_rm=20000,
                     n_boot=200,
                     target_down_time=None, target_down_freq=None):
    """分析一个 _cal.h5 文件中的所有爆发, 返回结果行列表。

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
        attrs        = dict(f.attrs)
    # analysis 阶段的 RFI 只根据非 burst 噪声采样点重新计算, 不读取/合并
    # _cal.h5 中 calibration 阶段保存的 rfi_mask。

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
    # I 和 V 对持续窄带污染的响应不完全相同; 分别判定后取并集
    rfi_channel_i, rfi_pixel_i = cal_rfi(I, noise_mask,
                                         down_time=1, down_freq=1,
                                         fft=rfi_fft)
    rfi_channel_v, rfi_pixel_v = cal_rfi(V, noise_mask,
                                         down_time=1, down_freq=1,
                                         fft=rfi_fft)
    rfi_channel = rfi_channel_i | rfi_channel_v
    rfi_pixel   = rfi_pixel_i | rfi_pixel_v
    rfi_mask = rfi_pixel.copy()
    rfi_mask[:, rfi_channel] = True

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

    results = []
    # 跨爆发合并 PA: 每个爆发只在自己时间窗口内有 PAV/PAE, 窗口外为 NaN.
    # 复刻老代码 plot_spec(..., comb=True) 的 PDF 输出.
    PAV_ALL, PAE_ALL = [], []
    last_pol_profiles = None     # (prof_I, prof_L, prof_V) 取最后一个成功的爆发
    for bi, region in enumerate(burst_regions):
        print(f'  [{basename}] 分析爆发 {bi}...')

        ts, te = region['time_start'], region['time_end']
        fs, fe = region['freq_start'], region['freq_end']

        # 该爆发的有效频率通道 = 指定频率范围 ∩ 非 RFI
        freq_index       = np.zeros(nchan, dtype=bool)
        freq_index[fs:fe] = True
        freq_index[rfi_chan] = False

        # 该爆发的时间掩码
        burst_mask          = np.zeros(nsamp, dtype=bool)
        burst_mask[ts:te]   = True

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
                burst_mask, freq_index, noise_mask,
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
                rfi_fft=False,
                dm_range=10.0, dm_step=0.1, dm_snr_threshold=5.0,
                rm_min=0, rm_max=50000, n_rm=20000,
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
            dm_range=dm_range, dm_step=dm_step,
            dm_snr_threshold=dm_snr_threshold,
            rm_min=rm_min, rm_max=rm_max, n_rm=n_rm,
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
    DM_RANGE          = 10.0
    DM_STEP           = 0.1
    DM_SNR_THRESHOLD  = 5.0
    RM_MIN            = -50000
    RM_MAX            = 50000
    N_RM              = 20000
    N_BOOT            = 200

    parser = argparse.ArgumentParser(description='爆发分析流水线')
    parser.add_argument('--cal-dir',          default=CAL_DIR,          help='定标 h5 目录')
    parser.add_argument('--output-dir',       default=OUTPUT_DIR,       help='输出目录')
    parser.add_argument('--rfi-fft',          default=RFI_FFT,          action='store_true', help='RFI 改用 FFT 法')
    parser.add_argument('--dm-range',         default=DM_RANGE,         type=float, help='DM 搜索范围')
    parser.add_argument('--dm-step',          default=DM_STEP,          type=float, help='DM 搜索步长')
    parser.add_argument('--dm-snr-threshold', default=DM_SNR_THRESHOLD, type=float, help='DM 低SNR压制阈值')
    parser.add_argument('--rm-min',           default=RM_MIN,           type=float, help='RM 下限')
    parser.add_argument('--rm-max',           default=RM_MAX,           type=float, help='RM 上限')
    parser.add_argument('--n-rm',             default=N_RM,             type=int,   help='RM 试验点数')
    parser.add_argument('--n-boot',           default=N_BOOT,           type=int,   help='Bootstrap 次数')
    parser.add_argument('--target-down-time', default=None, type=int,
                        help='相对原始数据的目标时间下采样倍率, 必须是 _cal.h5 已存倍率的整数倍')
    parser.add_argument('--target-down-freq', default=None, type=int,
                        help='相对原始数据的目标频率下采样倍率, 必须是 _cal.h5 已存倍率的整数倍')
    args = parser.parse_args()

    df = analyze_all(
        args.cal_dir, args.output_dir,
        rfi_fft=args.rfi_fft,
        dm_range=args.dm_range, dm_step=args.dm_step,
        dm_snr_threshold=args.dm_snr_threshold,
        rm_min=args.rm_min, rm_max=args.rm_max, n_rm=args.n_rm,
        n_boot=args.n_boot,
        target_down_time=args.target_down_time,
        target_down_freq=args.target_down_freq)

    if not df.empty:
        print(f'\n汇总:')
        print(f'  爆发总数: {len(df)}')
        print(f'  SNR 范围: {df["snr"].min():.1f} – {df["snr"].max():.1f}')
        print(f'  DM 范围: {df["dm"].min():.3f} – {df["dm"].max():.3f}')
