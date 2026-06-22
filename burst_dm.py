"""
DM 精化搜索 — 基于相干功率谱方法。

算法来自 DM_phase.py，剥离了 psrchive / GUI 依赖。
使用差分消色散：以 cut_burst_data 阶段的 dm_zero 为零点，
在 ±dm_range 范围内搜索最优 DM。另外提供 analyze_dm 一体化编排。
"""

import os
import numpy as np
from numpy.fft import fft
import scipy.signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# 消色散
# ============================================================

def dedisperse_waterfall(wfall, dm, freq, dt, ref_freq='top'):
    """对 waterfall 做消色散（roll 法）。

    Parameters
    ----------
    wfall : ndarray (nchan, nsamp)
        频率—时间矩阵。
    dm : float
        色散量 (pc/cm³)。
    freq : ndarray (nchan,)
        频率数组 (MHz)。
    dt : float
        时间分辨率 (秒)。
    ref_freq : str
        参考频率位置: 'top'、'center' 或 'bottom'。

    Returns
    -------
    dedisp : ndarray (nchan, nsamp)
    """
    k_dm = 1.0 / 2.41e-4

    if ref_freq == 'top':
        ref = freq[-1]
    elif ref_freq == 'center':
        ref = freq[len(freq) // 2]
    elif ref_freq == 'bottom':
        ref = freq[0]
    else:
        ref = freq[-1]

    # 每个频率通道需要滚动的采样点数
    shift = (k_dm * dm * (ref ** -2 - freq ** -2) / dt).round().astype(int)
    dedisp = np.zeros_like(wfall)
    for i, row in enumerate(wfall):
        dedisp[i] = np.roll(row, shift[i])
    return dedisp


# ============================================================
# 相干功率谱
# ============================================================

def coherent_power_spectrum(waterfall):
    """计算 FFT 相位相干功率谱。

    对每个频率通道做 FFT，除以幅值仅保留相位信息，
    再对频率求和取模平方，得到相干功率谱。

    Parameters
    ----------
    waterfall : ndarray (nchan, nsamp)

    Returns
    -------
    power_spectrum : ndarray (nsamp//2,)
    """
    ft = fft(waterfall, axis=-1)
    amp = np.abs(ft)
    amp[amp == 0] = 1              # 避免除零
    coherence = np.sum(ft / amp, axis=0)
    power = np.abs(coherence) ** 2
    nbin = waterfall.shape[1] // 2
    return power[:nbin]


# ============================================================
# 涨落频率截断
# ============================================================

def _get_f_threshold(power_spectra, mean, std):
    """复刻 processing_old/DM_phase.py 的自动 fluctuation-frequency 截断。"""
    peak_power = np.max(power_spectra, axis=1)
    snr = (peak_power - mean) / std
    kern = np.round(_get_window(snr) / 2).astype(int)
    if kern < 5:
        kern = 5
    return 0, kern


# ============================================================
# DM 曲线计算
# ============================================================

def _get_dm_curve(power_spectra, dpower_spectra, nchan):
    """从相干功率谱矩阵计算 DM-SNR 曲线。

    通过自适应截断频率和方差最小化来确定最优的涨落频率范围，
    然后对加权功率求和得到每个试验 DM 的信噪比。

    Parameters
    ----------
    power_spectra : ndarray (nbin, n_dm)
        每列是一个试验 DM 对应的相干功率谱。
    dpower_spectra : ndarray (nbin, n_dm)
        按涨落频率指数平方加权的功率谱。
    nchan : int
        频率通道数。

    Returns
    -------
    dm_curve : ndarray (n_dm,)
        加权功率曲线。
    dm_c_err : ndarray (n_dm,)
        功率曲线误差。
    snr_curve : ndarray (n_dm,)
        信噪比曲线。
    """
    n = power_spectra.shape[0]    # nbin
    m = power_spectra.shape[1]    # n_dm

    X, Y = np.meshgrid(np.arange(m), np.arange(n))
    num_el = (n - Y).astype(float)

    # 逐行累积统计量，用于自适应截断
    S = np.divide(
        np.sum(power_spectra, axis=0).T - np.cumsum(power_spectra, axis=0),
        num_el)
    S2 = np.divide(
        np.sum(power_spectra ** 2, axis=0).T - np.cumsum(power_spectra ** 2, axis=0),
        num_el)
    var = np.divide((S2 - S ** 2), num_el)
    var_sm = scipy.signal.convolve2d(var, np.ones([9, 3]) / 27,
                                     mode='same', boundary='wrap')

    # 自适应截断频率索引（方差最小处）
    idx_f = np.argmin(var_sm[:-10, :], axis=0)
    idx_c = np.convolve(idx_f, np.ones(3) / 3., mode='same').astype(int)
    idx_c[idx_c == 0] = 1
    idx_c = np.ones(np.shape(idx_c)) * idx_c

    # 矩量计算（解析噪声期望值和方差）
    cutoff = np.ones([n, 1]) * idx_c
    I2_sum = np.multiply(np.multiply(idx_c, idx_c + 1),
                         2 * idx_c + 1) / 6
    # 按旧 DM_phase.py 原式保留。这里不是标准平方和闭式的常见写法,
    # 但为了复刻历史结果, 不做数学化简或修正。
    I4_sum = np.multiply(np.multiply(np.multiply(idx_c, idx_c + 1),
                                     2 * idx_c + 1),
                         3 * idx_c + 3 * idx_c - 1) / 30

    # 截断范围内的加权功率
    Lo  = np.multiply(Y <= cutoff, dpower_spectra)
    Lo1 = np.multiply(Y <= cutoff,
                      np.multiply(power_spectra, dpower_spectra))

    AV_N_pow    = 2.0 * nchan * np.ones(np.shape(idx_c))
    dm_curve    = Lo.sum(axis=0)
    dn_term     = Lo1.sum(axis=0)
    Noise_curve = np.multiply(AV_N_pow, I2_sum)

    Var_dp   = 2.0 * nchan ** 2 * I4_sum + 1.0 * dn_term
    dm_c_err = Var_dp ** 0.5
    Dem      = (2.0 * nchan ** 2 * (I4_sum + 2.0 * I2_sum)) ** 0.5

    snr_curve = np.divide((dm_curve - 1.0 * Noise_curve), Dem)
    SN_Err = (np.divide(Var_dp, Dem ** 2) + 1
              + np.multiply(np.divide(snr_curve ** 2, idx_c),
                            (1 + 8 * nchan ** 2 / Dem ** 2)))
    snr_curve[np.isnan(snr_curve)] = 0.

    return dm_curve, dm_c_err, snr_curve


# ============================================================
# 多项式极值拟合
# ============================================================

def poly_max(x, y, err, w=None):
    """对 DM 曲线做多项式拟合并寻找极值。

    Parameters
    ----------
    x, y : ndarray
        DM 值和对应的 SNR 曲线。
    err : float
        残差误差下限。
    w : ndarray or None
        拟合权重。

    Returns
    -------
    best_x : float
        极值对应的 DM。
    delta_x : float
        DM 误差估计。
    poly_coeffs : ndarray
        拟合多项式系数。
    x_mean : float
        x 的平均值（多项式在 dx = x - x_mean 上拟合）。
    """
    # 多项式阶数: 数据点少于 7 时自动降阶
    if np.shape(x)[0] < 7:
        n = np.linalg.matrix_rank(np.vander(y))
    else:
        n = 6
    dx = x - x.mean()

    if w is None:
        p = np.polyfit(dx, y, n)
        err = max(np.std(y - np.polyval(p, dx)), err)
    else:
        p = np.polyfit(dx, y, n, w=w)
        err = max((np.sum(np.multiply(w, (y - np.polyval(p, dx)) ** 2.0))
                   / np.sum(w)) ** 0.5, err)

    # 求导数根，找极大值
    dp  = np.polyder(p)
    ddp = np.polyder(dp)
    cands   = np.roots(dp)
    r_cands = np.polyval(ddp, cands)

    # 筛选: 实数根、在数据范围内、二阶导 < 0（极大值）
    first_cut = cands[(cands.imag == 0) &
                      (cands.real >= min(dx)) &
                      (cands.real <= max(dx)) &
                      (r_cands < 0)]
    if first_cut.size > 0:
        value = np.polyval(p, first_cut)
        best = np.real(first_cut[value.argmax()])
        delta_x = np.sqrt(np.abs(2.0 * err / np.polyval(ddp, best)))
    else:
        best = 0.0
        delta_x = 0.0

    return float(np.real(best) + x.mean()), delta_x, p, x.mean()


def _get_window(profile):
    """ACF windowing, copied from processing_old/DM_phase.py."""
    smooth_profile = scipy.signal.detrend(profile)
    autocorrelation = np.correlate(smooth_profile, smooth_profile, 'same')
    window = np.max(np.diff(np.where(autocorrelation < 0)))
    return window


def _check_window(profile, window):
    """复刻旧代码的拟合窗口检查。"""
    convolved = np.convolve(1.0 * profile, 1.0 * np.ones(int(window)), 'same')
    peak_value = np.mean(np.where(convolved == max(convolved)))
    peak = np.where(profile == np.max(profile))

    if (peak_value - peak) ** 2 > window ** 2:
        window += np.abs(peak_value - peak) / 2
        peak_value = (peak_value + peak) / 2

    start = int(peak_value - np.round(1.25 * window))
    end = int(peak_value + np.round(1.25 * window))

    if start < 0:
        start = 0
    if end > profile.size - 1:
        end = profile.size - 1
    return start, end


def _dm_calculation(power_spectra, dpower_spectra, low_idx, up_idx,
                    nchan, dm_list, dm_curve=None, weight=None,
                    dstd=None, SN=None):
    """复刻 processing_old/DM_phase.py::_dm_calculation 的非绘图路径。"""
    if dm_curve is None:
        dm_curve = dpower_spectra[low_idx:up_idx].sum(axis=0)

        mean = 2.0 * nchan
        std = mean / np.sqrt(2)

        m_fact = np.sum(np.arange(low_idx, up_idx) ** 2)
        s_fact = np.sum(np.arange(low_idx, up_idx) ** 4) ** 0.5
        dmean = mean * m_fact
        I = np.transpose(
            1.0 * np.ones([dm_list.size, 1])
            * (1.0 * np.arange(low_idx, up_idx)) ** 2.0)
        dstd1 = (1.0 * (std * s_fact) ** 2.0
                 + np.sum(np.multiply(I, power_spectra[low_idx:up_idx] ** 2.0),
                          axis=0)) ** 0.5
        dm_curve = np.divide((dm_curve - dmean), dstd1)
        weight = np.multiply(dm_curve, dstd1 ** -1.0)
        dstd = np.max(dstd1)

    snr = SN

    if weight is None:
        peak = dm_curve.argmax()
        width = _get_window(dm_curve) / 2
        start, stop = _check_window(dm_curve, width)
    else:
        w_dm_curve = np.multiply(weight, dm_curve)
        peak = dm_curve.argmax()
        curve = power_spectra[low_idx + 1:low_idx + 2].sum(axis=0)
        width = int(_get_window(curve) / 4)
        heavy_weights = np.argwhere(dm_curve > .5 * dm_curve[peak])
        if len(heavy_weights) < 5:
            heavy_weights = np.argwhere(dm_curve > .25 * dm_curve[peak])
        if len(heavy_weights) < 5:
            heavy_weights = np.argwhere(dm_curve > .1 * dm_curve[peak])
        peak = np.mean(heavy_weights)
        width = np.max(heavy_weights) - np.min(heavy_weights)
        start = heavy_weights[
            np.argmin(np.absolute((peak - width) - heavy_weights))]
        stop = heavy_weights[
            np.argmin(np.absolute((peak + width) - heavy_weights))]

        start = int(np.asarray(start).squeeze())
        stop = int(np.asarray(stop).squeeze())
        if start < 0:
            start = 0
        if stop > np.size(w_dm_curve):
            stop = np.size(w_dm_curve)

    plot_range = np.arange(start, stop)
    y = dm_curve[plot_range]
    x = dm_list[plot_range]

    if weight is None:
        new_w = 1.0 * np.ones(x.shape) / np.sum(np.ones(x.shape))
    else:
        new_w = weight[plot_range] / np.sum(weight[plot_range])

    returns_poly = poly_max(x, y, dstd, w=new_w)
    dm = returns_poly[0]
    dm_std = returns_poly[1]
    return dm, dm_std, dm_curve, snr


# ============================================================
# DM 相位搜索 (主入口)
# ============================================================

def dm_phase_search(stokes_I_2d, freq, time_reso, dm_zero,
                    dm_range=10.0, dm_step=0.1, snr_threshold=5.0):
    """使用相干相位法搜索最优 DM。

    以 dm_zero 为零点做差分消色散，在 ±dm_range 内扫描，
    通过相干功率谱 + 多项式拟合确定最优 DM。

    Parameters
    ----------
    stokes_I_2d : ndarray (nsamp, nchan)
        已在 dm_zero 处消色散的 Stokes I。
    freq : ndarray (nchan,)
        频率数组 (MHz)。
    time_reso : float
        时间分辨率 (秒)。
    dm_zero : float
        初始 DM (pc/cm³)，来自 cut_burst_data。
    dm_range : float
        搜索范围: dm_zero ± dm_range。
    dm_step : float
        搜索步长 (pc/cm³)。
    snr_threshold : float
        低于此 SNR 的试验 DM 在拟合时被压制。

    Returns
    -------
    dm_best : float
        最优 DM (绝对值)。
    dm_err : float
        DM 误差。
    dm_snr : float
        峰值 SNR。
    dm_list : ndarray
        试验 DM 列表（绝对值）。
    dm_curve : ndarray
        用于绘图的 SNR 曲线。
    """
    # waterfall: (nchan, nsamp)，DM_phase 的约定
    waterfall = stokes_I_2d.T.copy()
    nchan, nbin_full = waterfall.shape
    nbin = nbin_full // 2

    # 试验 DM 增量（相对于 dm_zero）
    # 旧 DM_phase.py 使用 np.arange(dm_s, dm_e, dm_step), 不包含右端点。
    delta_list = np.arange(-dm_range, dm_range, dm_step)
    dm_list = dm_zero + delta_list

    # 对每个试验 DM 计算相干功率谱
    power_spectra = np.zeros([nbin, len(delta_list)])
    for i, ddm in enumerate(delta_list):
        wf_dd = dedisperse_waterfall(waterfall, ddm, freq, time_reso)
        power_spectra[:, i] = coherent_power_spectrum(wf_dd)

    # 按涨落频率指数平方加权
    ff_idx = np.arange(0, nbin)
    dpower_spectra = power_spectra * ff_idx[:, np.newaxis] ** 2

    # 下面完全复刻 processing_old/DM_phase.py::get_dm 的自动 cutoff 分支。
    mean = nchan
    std = nchan / np.sqrt(2)
    low_idx, up_idx = _get_f_threshold(power_spectra, mean, std)

    dm_curve, dm_c_err, snr_curve = _get_dm_curve(
        power_spectra, dpower_spectra, nchan)
    dm_curve[snr_curve < snr_threshold] = dm_curve[snr_curve < snr_threshold] / 1e6

    w = snr_curve
    w[np.isnan(w)] = 0.0
    w[snr_curve < snr_threshold] = 1 / 1e6
    w = w / np.sum(w)
    dstd = np.max(dm_c_err)

    dm_best, dm_err, dm_curve_out, dm_snr = _dm_calculation(
        power_spectra, dpower_spectra, low_idx, up_idx,
        nchan, dm_list, dm_curve=dm_curve, weight=w,
        dstd=dstd, SN=np.max(snr_curve))

    return dm_best, dm_err, float(dm_snr), dm_list, dm_curve_out


# ============================================================
# 绘图
# ============================================================

def plot_dm_search(dm_list, dm_curve, dm_best, dm_err, save_path):
    """DM 搜索曲线。"""
    plt.figure(figsize=(5, 4))
    plt.plot(dm_list, dm_curve, color='royalblue', alpha=0.7)
    plt.axvline(dm_best, color='red', ls='--', alpha=0.7,
                label=f'DM={dm_best:.3f}$\\pm${dm_err:.3f}')
    plt.axvspan(dm_best - dm_err, dm_best + dm_err, alpha=0.1, color='red')
    plt.xlabel('DM (pc cm$^{-3}$)')
    plt.ylabel('Coherent Power (a.u.)')
    plt.legend()
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# 一体化编排: 单个爆发的 DM 搜索
# ============================================================

def analyze_dm(stokes_I, freq, time_reso, dm_zero, burst_region,
               output_dir, burst_idx,
               dm_range=10.0, dm_step=0.1, snr_threshold=5.0):
    """对单个爆发做 DM 精化搜索并画图。

    提取 burst 区间 + padding 的 Stokes I 片段交给 dm_phase_search,
    失败或搜索不稳时回退到 dm_zero。

    Parameters
    ----------
    stokes_I : ndarray (nsamp, nchan)
        已减基线、已屏蔽 RFI(NaN) 的 Stokes I。
    freq : ndarray (nchan,)  MHz
    time_reso : float        有效时间分辨率 (秒)
    dm_zero : float          cut_burst_data 阶段使用的 DM
    burst_region : dict      含 time_start / time_end
    output_dir : str         图保存目录
    burst_idx : int          爆发编号(用于命名)

    Returns
    -------
    dm : dict
        dm, dm_err  (失败时 dm=dm_zero, dm_err=0.0)
    """
    ts, te = burst_region['time_start'], burst_region['time_end']
    nsamp  = stokes_I.shape[0]

    # 区间外扩半个宽度, 至少 50 采样点
    pad     = max((te - ts) // 2, 50)
    dm_ts   = max(0, ts - pad)
    dm_te   = min(nsamp, te + pad)
    dm_data = np.nan_to_num(stokes_I[dm_ts:dm_te].copy(), nan=0.0)

    try:
        dm_best, dm_err, _, dm_list_out, dm_curve_out = dm_phase_search(
            dm_data, freq, time_reso, dm_zero,
            dm_range=dm_range, dm_step=dm_step,
            snr_threshold=snr_threshold)
        plot_dm_search(dm_list_out, dm_curve_out, dm_best, dm_err,
                       os.path.join(output_dir, f'burst{burst_idx}_dm.png'))
    except Exception as e:
        print(f'    DM 搜索失败: {e}')
        dm_best, dm_err = dm_zero, 0.0

    return {'dm': dm_best, 'dm_err': dm_err}
