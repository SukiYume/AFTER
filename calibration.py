"""cut_burst_data 输出的 h5 文件定标 + 下采样 + RFI 检测 + 保存。

流程:
  1. 读 BURST_DIR 中所有 burst h5（非 _cal.h5）并按波束分组。
  2. 每个波束找第一个 _0001.fits 作为定标文件, 折叠噪声管 → noise_cal。
  3. 从 npz 加载 t_cal (K)。
  4. 每个 burst:
     · 用 noise_cal 做偏振定标得到归一化的 IQUV,
     · 乘以 t_cal/(2*gain) 直接变成 Jy（偏振+流量一步完成, 不再分两份保存）,
     · 时间+频率下采样 (保存倍率, 默认 = 画图倍率),
     · 用 rfi_utils.cal_rfi (FFT 方法) 在整段数据上检测 RFI 通道/像素,
     · 画图: 在保存数据上再做 extra 倍下采样到"清晰"分辨率(减基线+抹RFI),
     · 存 _cal.h5 (data/freq/rfi_mask/gain/gain_err)。

保存策略: 写入 h5 的 data 是"相对原始"的定标+下采样后数据, 不减基线、不 NaN 掉
RFI; rfi_mask 与 rfi_channel 作为辅助信息一并保存。下游 burst_analysis 会用
真正的噪声段做二次 RFI 精修和按通道基线减除, 比在这里用全时段中值更干净。

下采样规则:
  · 画图倍率 (plot_dt, plot_df) 自动计算:
        频率目标 ~512 通道, 时间目标 ~ 49.152us×8 ≈ 393us.
    比如 49.152us / 4096ch → (8, 8); 98.304us / 4096ch → (4, 8).
  · 保存倍率 (down_time, down_freq) 默认 = 画图倍率, 用户可显式传更小的值
    (= 更高分辨率的保存数据), 但必须满足 plot_* % save_* == 0, 否则会被
    向下取整对齐.
  · 画图时在保存数据上再做 extra = plot // save 倍下采样, 信号更清楚.

保存 time_reso_raw(原始) 和 time_reso(下采样后的有效值), 供后续计算流量/能量时
直接用 time_reso, 不再把 down_time 乘进去。
"""

import os
import re
from collections import defaultdict
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import seaborn as sns # 'mako' 颜色映射需要 seaborn 支持
import matplotlib.pyplot as plt
from matplotlib import gridspec
from astropy.io import fits
from astropy.utils import iers
from multiprocessing import Pool
from ZeithAngle import get_za, get_gain
from rfi_utils import cal_rfi

iers.conf.auto_download = False


# 画图目标分辨率: 频率 ~512 通道, 时间 ~ 49.152us × 8 ≈ 393us.
# 用于自动推算 plot_dt / plot_df, 也作为 save_dt / save_df 的默认值.
PLOT_TARGET_TIME_RESO = 49.152e-6 * 8
PLOT_TARGET_NCHAN     = 512


def find_cal_fits(directory, beam):
    """查找 directory 中指定波束的定标 fits (_0001.fits 结尾且含 Mdd)。"""
    for fname in sorted(os.listdir(directory)):
        if fname.endswith('_0001.fits') and f'M{beam:02d}' in fname:
            return os.path.join(directory, fname)
    return None


def fold_noise_cal(cal_fits_path):
    """折叠噪声管数据, 返回 noise_on - noise_off, 形状 (npol, nchan)。"""
    with fits.open(cal_fits_path) as f:
        h         = f[1].header
        time_reso = h['TBIN']
        data      = f[1].data['DATA'].reshape(
            h['NAXIS2'] * h['NSBLK'], h['NPOL'], h['NCHAN']
        )

    # FAST 噪声管周期(采样点数): 噪声管频率 = 1e9 / (4096 * 4096 * 12) Hz
    noise_period = int(4096 * 4096 * 12 / (time_reso * 1e9))
    n_periods    = data.shape[0] // noise_period
    data = data[:n_periods * noise_period].reshape(
        n_periods, noise_period, h['NPOL'], h['NCHAN']).mean(axis=0)

    power     = np.mean(data[:, :2, :], axis=(1, 2))
    on_mask   = power > np.mean(power)
    noise_on  = np.mean(data[on_mask],  axis=0)
    noise_off = np.mean(data[~on_mask], axis=0)
    return noise_on - noise_off


def load_t_cal(cal_npz_path, beam, nchan):
    """从 npz 加载 t_cal 并匹配到目标 nchan, 返回 (2, nchan)。

    t_cal npz 固定 4096 通道, 需要 nchan 与 4096 之间整数倍关系 (FAST
    数据总是 2 的幂, 通常满足). 不整除会直接 assert 报错避免静默截断.
    """
    t_cal = np.load(cal_npz_path)['tcal'][:, :, beam - 1]  # (4096, 2)
    if nchan <= 4096:
        assert 4096 % nchan == 0, \
            f'nchan={nchan} 不是 4096 的因子, t_cal 无法按整数倍合并'
        t_cal = np.mean(t_cal.reshape(nchan, 4096 // nchan, 2), axis=1).T
    else:
        assert nchan % 4096 == 0, \
            f'nchan={nchan} 不是 4096 的整数倍, t_cal 无法按整数倍展开'
        t_cal = np.repeat(t_cal, nchan // 4096, axis=0).T
    return t_cal


def calibrate_to_iquv(data, noise_cal, t_cal, gain, cal_threshold=0.05):
    """偏振定标 + 流量缩放, 一步得到 Jy 单位的 IQUV。

    偏振定标: 用 noise_cal 归一化两个 feed 的增益差异, 并用 arctan2 校正
    交叉项相位;
    流量缩放: 每个偏振独立乘 t_cal[pol]/(2*gain) 再合成 I/Q, 交叉项 U/V 使用
    sqrt(t_cal[0]*t_cal[1])/(2*gain) 作为等效尺度。与 processing_old 的
    `burst_data * intensity_cal / gain * t_cal` 再按偏振平均一致。

    Parameters
    ----------
    data      : (nsamp, npol, nchan) uint8 或浮点
    noise_cal : (npol, nchan)
    t_cal     : (2, nchan)
    gain      : 标量或 (nchan,)

    Returns
    -------
    iquv : (4, nsamp, nchan) float32, 单位 Jy
    """
    nsamp, npol, nchan = data.shape

    noise_a12       = np.where(noise_cal[0] > cal_threshold, 1.0 / noise_cal[0], 0.0)
    noise_a22       = np.where(noise_cal[1] > cal_threshold, 1.0 / noise_cal[1], 0.0)
    scale_0         = t_cal[0] / (2.0 * gain)
    scale_1         = t_cal[1] / (2.0 * gain)

    I               =    scale_0 * noise_a12 * data[:, 0, :] + scale_1 * noise_a22 * data[:, 1, :]
    Q               = - (scale_0 * noise_a12 * data[:, 0, :] - scale_1 * noise_a22 * data[:, 1, :])

    if npol == 4:
        scale_cross = np.sqrt(scale_0 * scale_1)   # 交叉项等效 t_cal
        noise_dphi  = np.arctan2(noise_cal[3], noise_cal[2])
        noise_a1a2  = np.sqrt(noise_a12 * noise_a22)
        noise_cos   = np.cos(noise_dphi) * noise_a1a2
        noise_sin   = np.sin(noise_dphi) * noise_a1a2
        U           = 2.0 * scale_cross * ( noise_cos * data[:, 2, :] + noise_sin * data[:, 3, :])
        V           = 2.0 * scale_cross * (-noise_sin * data[:, 2, :] + noise_cos * data[:, 3, :])
    else:
        U           = np.zeros((nsamp, nchan), dtype=np.float32)
        V           = np.zeros((nsamp, nchan), dtype=np.float32)

    return np.array([I, Q, U, V], dtype=np.float32)


def process_one_burst(h5_input_path, output_dir, noise_cal, t_cal,
                      ra, dec, beam, down_time=None, down_freq=None,
                      rfi_fft=True):
    """读一个 burst h5, 定标 + 下采样 + RFI 检测, 写 _cal.h5。

    down_time, down_freq : int or None
        保存下采样倍率. None = 自动取画图清晰倍率
        (频率目标 ~512 通道, 时间目标 ~49.152us×8).
        可显式传更小的值得到更高分辨率的保存数据;
        画图仍按清晰倍率额外下采样, 视觉效果不变.

    保存的 data 未减基线、未 NaN 掉 RFI, 给下游更大的处理空间;
    rfi_mask 作为辅助信息一并保存。
    """
    basename = os.path.splitext(os.path.basename(h5_input_path))[0]
    out_h5   = os.path.join(output_dir, basename + '_cal.h5')
    if os.path.exists(out_h5):
        print(f'  [跳过] {out_h5} 已存在')
        return

    with h5py.File(h5_input_path, 'r') as f:
        raw_data  = f['data'][:]                 # (nsamp, npol, nchan)
        freq      = f['freq'][:]
        attrs     = dict(f.attrs)

    file_mjd      = attrs['file_mjd']
    time_reso_raw = attrs['time_reso']
    nchan_raw     = attrs['nchan']

    # 画图倍率: 频率 / 时间分别向 PLOT_TARGET_NCHAN / PLOT_TARGET_TIME_RESO 看齐,
    # 并把 plot 对齐到 save 的整数倍 (extra = plot // save 反推).
    plot_dt = max(1, int(round(PLOT_TARGET_TIME_RESO / time_reso_raw)))
    plot_df = max(1, int(round(nchan_raw / PLOT_TARGET_NCHAN)))
    save_dt = plot_dt if down_time is None else min(int(down_time), plot_dt)
    save_df = plot_df if down_freq is None else min(int(down_freq), plot_df)
    extra_dt, extra_df = plot_dt // save_dt, plot_df // save_df
    plot_dt,  plot_df  = extra_dt * save_dt, extra_df * save_df

    za             = get_za(file_mjd, ra, dec)
    gain, gain_err = get_gain(za, beam, nchan_raw)
    iquv           = calibrate_to_iquv(raw_data, noise_cal, t_cal, gain)

    # 保存倍率下采样 (iquv / freq / gain 同步). nsamp / nchan 始终跟踪当前形状.
    _, nsamp, nchan = iquv.shape
    if save_dt > 1:
        nt    = nsamp // save_dt
        iquv  = np.nanmean(iquv[:, :nt * save_dt].reshape(4, nt, save_dt, nchan), axis=2)
        nsamp = nt
    if save_df > 1:
        nc          = nchan // save_df
        iquv        = np.nanmean(iquv[:, :, :nc * save_df].reshape(4, nsamp, nc, save_df), axis=3)
        freq        = freq[:nc * save_df].reshape(nc, save_df).mean(axis=1)
        gain_ds     = gain[:nc * save_df].reshape(nc, save_df).mean(axis=1)
        gain_err_ds = gain_err[:nc * save_df].reshape(nc, save_df).mean(axis=1)
        nchan       = nc
    else:
        gain_ds, gain_err_ds = gain, gain_err
    time_reso_save     = time_reso_raw * save_dt
    nsamp_ds, nchan_ds = nsamp, nchan

    # RFI 检测: 整段当噪声, 在画图分辨率上找通道级 RFI (extra_dt × extra_df 倍下采样)
    noise_mask               = np.ones(nsamp_ds, dtype=bool)
    rfi_channel, rfi_pixel   = cal_rfi(
        iquv[0], noise_mask, down_time=extra_dt, down_freq=extra_df, fft=rfi_fft,
    )
    rfi_mask                 = rfi_pixel.copy()
    rfi_mask[:, rfi_channel] = True

    os.makedirs(output_dir, exist_ok=True)

    # 仅画图: extra 倍下采样, 减基线, 抹 RFI (不回写 iquv); 结构对应上面的保存块.
    plot_I    = iquv[0].copy()
    plot_freq = freq
    plot_I[rfi_mask]  = np.nan
    plot_I           -= np.nanmedian(plot_I, axis=0)
    nsamp_plot, nchan_plot = plot_I.shape
    if extra_dt > 1:
        nt         = nsamp_plot // extra_dt
        plot_I     = np.nanmean(plot_I[:nt * extra_dt].reshape(nt, extra_dt, nchan_plot), axis=1)
        nsamp_plot = nt
    if extra_df > 1:
        nc         = nchan_plot // extra_df
        plot_I     = np.nanmean(plot_I[:, :nc * extra_df].reshape(nsamp_plot, nc, extra_df), axis=2)
        plot_freq  = freq[:nc * extra_df].reshape(nc, extra_df).mean(axis=1)
        nchan_plot = nc
    time_reso_eff = time_reso_save * extra_dt

    fig = plt.figure(figsize=(5, 5))
    gs  = gridspec.GridSpec(4, 1, hspace=0)

    time_ms_plot = np.arange(nsamp_plot) * time_reso_eff * 1e3
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.step(time_ms_plot, np.nanmean(plot_I, axis=1), where='mid', color='royalblue', lw=0.8)
    ax0.set_xlim(0, nsamp_plot * time_reso_eff * 1e3)
    ax0.set_xticks([])
    ax0.set_ylabel('Flux (Jy)')
    ax1 = fig.add_subplot(gs[1:, 0])
    vmin, vmax = np.nanpercentile(plot_I, [5, 95])
    ax1.imshow(
        plot_I.T, aspect='auto', origin='lower', cmap='mako', vmin=vmin, vmax=vmax,
        extent=[0, nsamp_plot * time_reso_eff * 1e3, plot_freq[0], plot_freq[-1]]
    )
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Frequency (MHz)')
    fig.align_labels()
    plt.savefig(
        os.path.join(output_dir, basename + '.jpg'),
        dpi=200, bbox_inches='tight', format='jpg', pil_kwargs={'quality': 95},
    )
    plt.close()

    rfi_frac  = np.sum(rfi_mask) / rfi_mask.size
    out_attrs = {
        'file_mjd'       : file_mjd,
        'obs_start_mjd'  : attrs['obs_start_mjd'],
        'start_sample'   : int(attrs['start_sample']),
        'toa_sec'        : attrs['toa_sec'],
        'time_reso_raw'  : time_reso_raw,
        'time_reso'      : time_reso_save,
        'down_time'      : save_dt,
        'down_freq'      : save_df,
        'plot_down_time' : plot_dt,   # 记录画图实际用的倍率 (供下游查阅)
        'plot_down_freq' : plot_df,
        'nchan_raw'      : nchan_raw,
        'nchan'          : nchan_ds,
        'nsamp'          : nsamp_ds,
        'dm'             : attrs['dm'],
        'beam'           : beam,
        'ra'             : ra,
        'dec'            : dec,
        'rfi_fraction'   : rfi_frac,
    }

    with h5py.File(out_h5, 'w') as f:
        f.create_dataset('data',        data=iquv.astype(np.float32), compression='gzip', compression_opts=4)
        f.create_dataset('freq',        data=freq.astype(np.float64))
        f.create_dataset('rfi_mask',    data=rfi_mask)
        f.create_dataset('rfi_channel', data=rfi_channel)
        # 增益及其系统误差(K/Jy), 下游用于计算 flux / fluence 的系统误差
        f.create_dataset('gain',        data=gain_ds.astype(np.float32))
        f.create_dataset('gain_err',    data=gain_err_ds.astype(np.float32))
        f.attrs.update(out_attrs)

    print(f'  [完成] {out_h5}  (RFI {rfi_frac*100:.1f}%)')


if __name__ == '__main__':

    # ---- 配置参数 ----
    BURST_DIR   = '/path/to/after_data/FRB20201124A/20210526/'
    OUTPUT_DIR  = '/path/to/after_data/FRB20201124A/20210526/cal/'
    CAL_NPZ     = '/path/to/after_data/highcal_20201014_psr_tny.npz'
    RA          = '05h08m03.51s'
    DEC         = '26d03m38.5s'
    DOWN_TIME   = None     # 保存时间下采样因子, None = 自动取画图清晰倍率
    DOWN_FREQ   = None     # 保存频率下采样因子, None = 自动取画图清晰倍率
    RFI_FFT     = True     # True=FFT 最大幅度; False=熵
    NUM_WORKERS = 8

    # 1. 按波束分组 burst h5
    burst_h5_list = sorted(
        f for f in os.listdir(BURST_DIR)
        if f.endswith('.h5') and not f.endswith('_cal.h5')
    )
    if not burst_h5_list:
        print('未找到 burst h5 文件')
        exit()
    print(f'找到 {len(burst_h5_list)} 个 burst 文件')

    # 文件名中 Mdd 段就是波束编号; 缺省按主波束 M01 处理.
    beam_groups = defaultdict(list)
    for fname in burst_h5_list:
        m    = re.search(r'M(\d{2})', fname)
        beam = int(m.group(1)) if m else 1
        beam_groups[beam].append(os.path.join(BURST_DIR, fname))

    # 2. 匹配定标文件 / t_cal, 组装任务列表
    all_args = []
    for beam, h5_list in sorted(beam_groups.items()):
        cal_fits_path = find_cal_fits(BURST_DIR, beam) or find_cal_fits(BURST_DIR, 1)
        if cal_fits_path is None:
            print(f'  波束 M{beam:02d}: 未找到定标文件, 跳过 {len(h5_list)} 个 burst')
            continue

        with fits.open(cal_fits_path) as f:
            nchan = f[1].header['NCHAN']

        noise_cal = fold_noise_cal(cal_fits_path)
        t_cal     = load_t_cal(CAL_NPZ, beam, nchan)

        print(f'  波束 M{beam:02d}: {len(h5_list)} 个 burst, '
              f'定标文件: {os.path.basename(cal_fits_path)}')

        for h5_path in h5_list:
            all_args.append((
                h5_path, OUTPUT_DIR, noise_cal, t_cal, RA, DEC, beam, DOWN_TIME, DOWN_FREQ, RFI_FFT
            ))

    if not all_args:
        print('无可处理的 burst 文件')
        exit()

    # 3. 并行处理
    if NUM_WORKERS > 1 and len(all_args) > 1:
        with Pool(NUM_WORKERS) as pool:
            pool.starmap(process_one_burst, all_args)
    else:
        for args in all_args:
            process_one_burst(*args)

    print('全部完成')
