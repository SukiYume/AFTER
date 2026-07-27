"""
基于 Stokes I 的爆发物理量计算。

包括: 到达时间 (TOA)、峰值流量、积分流量 (fluence)、脉冲宽度、
信噪比 (SNR)、频率范围、带宽。误差通过噪声区域 bootstrap 估计。
高斯拟合用于独立估计脉冲宽度和频率带宽。
"""

import numpy as np
from scipy.optimize import curve_fit


def _gaussian(x, amp, mu, sigma, offset):
    """一维高斯模型: amp * exp(-(x-mu)^2 / (2*sigma^2)) + offset。"""
    return amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + offset


def _fit_gaussian(x, y, snr_min=3.0):
    """对一维轮廓做鲁棒高斯拟合。

    Parameters
    ----------
    x : ndarray
        自变量（时间采样索引或频率 MHz）。
    y : ndarray
        因变量（强度轮廓，应已减基线）。
    snr_min : float
        峰值低于 snr_min × MAD 噪声时跳过拟合。

    Returns
    -------
    fwhm : float
        高斯 FWHM（与 x 同单位）。拟合失败时返回 NaN。
    fwhm_err : float
        FWHM 的 1σ 误差。拟合失败时返回 NaN。
    mu : float
        高斯中心位置。拟合失败时返回 NaN。
    mu_err : float
        中心位置的 1σ 误差。拟合失败时返回 NaN。
    """
    nan_result = (np.nan, np.nan, np.nan, np.nan)

    # 基本检查
    finite = np.isfinite(y) & np.isfinite(x)
    if np.sum(finite) < 4:
        return nan_result
    xf, yf = x[finite], y[finite]

    # 粗略 SNR 检查: 峰值 / MAD 噪声
    med = np.median(yf)
    mad = np.median(np.abs(yf - med)) * 1.4826
    peak_val = np.max(yf)
    if mad > 0 and (peak_val - med) / mad < snr_min:
        return nan_result

    # 初始参数估计
    peak_idx = np.argmax(yf)
    amp0 = peak_val - med
    mu0 = xf[peak_idx]
    offset0 = med

    # sigma 初始值: 半高宽估计
    half_max = (peak_val + med) / 2
    above = yf > half_max
    if np.sum(above) >= 2:
        indices = np.where(above)[0]
        sigma0 = (xf[indices[-1]] - xf[indices[0]]) / 2.355
    else:
        sigma0 = (xf[-1] - xf[0]) / 4

    sigma0 = max(sigma0, np.abs(xf[1] - xf[0]))  # 至少一个步长

    # 参数边界
    dx = xf[-1] - xf[0]
    bounds_lo = [0, xf[0] - 0.5 * dx, np.abs(xf[1] - xf[0]) * 0.1,
                 med - 3 * max(mad, abs(amp0))]
    bounds_hi = [amp0 * 5, xf[-1] + 0.5 * dx, dx,
                 med + 3 * max(mad, abs(amp0))]

    try:
        popt, pcov = curve_fit(
            _gaussian, xf, yf,
            p0=[amp0, mu0, sigma0, offset0],
            bounds=(bounds_lo, bounds_hi),
            maxfev=5000)
    except (RuntimeError, ValueError):
        return nan_result

    amp_fit, mu_fit, sigma_fit, offset_fit = popt

    # 拟合质量检验: 拟合的振幅应该是正数，且 sigma 合理
    if amp_fit <= 0 or sigma_fit <= 0:
        return nan_result

    # 拟合残差检验: 残差 rms 不应超过峰值的 50%
    residual = yf - _gaussian(xf, *popt)
    res_rms = np.std(residual)
    if res_rms > 0.5 * amp_fit:
        return nan_result

    # 从协方差矩阵提取误差
    perr = np.sqrt(np.diag(pcov))
    if not np.all(np.isfinite(perr)):
        return nan_result

    fwhm = 2.3548200 * sigma_fit  # 2√(2ln2) × σ
    fwhm_err = 2.3548200 * perr[2]
    mu_err = perr[1]

    return float(fwhm), float(fwhm_err), float(mu_fit), float(mu_err)


def _empty_props(file_mjd, freq, fs, fe, nchan):
    """爆发区域无效时返回的默认空结果。"""
    freq_low  = float(freq[fs]) if fs < nchan else float(freq[0])
    freq_high = float(freq[min(fe, nchan) - 1]) if fe > 0 else float(freq[-1])
    return {
        'toa_mjd': file_mjd, 'flux_peak': 0.0, 'flux_err': 0.0,
        'flux_err_sys': 0.0,
        'fluence': 0.0, 'fluence_err': 0.0, 'fluence_err_sys': 0.0,
        'width': 0.0, 'width_err': 0.0,
        'width_gauss': np.nan, 'width_gauss_err': np.nan,
        'snr': 0.0, 'freq_low': freq_low, 'freq_high': freq_high,
        'bandwidth': abs(freq_high - freq_low),
        'bandwidth_gauss': np.nan, 'bandwidth_gauss_err': np.nan,
    }


def calc_burst_properties(stokes_I, freq, time_reso, file_mjd, burst_region,
                          noise_mask, rfi_mask, freq_index,
                          gain=None, gain_err=None, n_boot=200):
    """计算单个爆发的基本物理量。

    Parameters
    ----------
    stokes_I : ndarray (nsamp, nchan)
        Stokes I 动态谱（可含 NaN）。
    freq : ndarray (nchan,)
        频率数组 (MHz)。
    time_reso : float
        时间分辨率 (秒)。
    file_mjd : float
        文件起始 MJD。
    burst_region : dict
        爆发区域，需含 time_start, time_end, freq_start, freq_end。
    noise_mask : ndarray (nsamp,) bool
        True = 噪声采样点。
    rfi_mask : ndarray (nsamp, nchan) bool
        True = RFI 像素。
    freq_index : ndarray (nchan,) bool
        True = 参与该爆发频率积分的通道 (已排除 RFI 与越界)。
    gain, gain_err : ndarray (nchan,) or None
        增益及其系统误差 (K/Jy), 用于计算 flux/fluence 的系统误差; 为 None
        时返回 0。
    n_boot : int
        Bootstrap 重采样次数，用于误差估计。

    Returns
    -------
    props : dict
        包含 toa_mjd, flux_peak, flux_err, flux_err_sys, fluence,
        fluence_err, fluence_err_sys, width, width_err, snr, freq_low,
        freq_high, bandwidth 等。
    """
    nsamp, nchan = stokes_I.shape
    ts, te = burst_region['time_start'], burst_region['time_end']
    fs, fe = burst_region['freq_start'], burst_region['freq_end']

    # 空爆发区域保护
    if ts >= te or ts >= nsamp:
        return _empty_props(file_mjd, freq, fs, fe, nchan)

    # 对 RFI 像素置 NaN
    data = stokes_I.copy()
    data[rfi_mask] = np.nan

    # 时间轮廓: 只在爆发频率范围内平均 (与老代码 cal_energy_burstbw 对齐,
    # 避免全带宽里 RFI 抹平窄带 burst 导致峰值位置偏移)
    if freq_index is not None and np.any(freq_index):
        mean_profile = np.nanmean(data[:, freq_index], axis=1)
    else:
        mean_profile = np.nanmean(data, axis=1)
    noise_profile = mean_profile[noise_mask]

    # 噪声统计量（保护空噪声区域和全 NaN）
    if noise_profile.size > 0 and np.any(np.isfinite(noise_profile)):
        noise_mean = np.nanmean(noise_profile)
        noise_std  = np.nanstd(noise_profile)
    else:
        noise_mean = 0.0
        noise_std  = 1.0   # 避免后续除零，snr 将为 ~0

    burst_profile = mean_profile[ts:te]

    # 爆发轮廓全 NaN 保护
    if burst_profile.size == 0 or not np.any(np.isfinite(burst_profile)):
        return _empty_props(file_mjd, freq, fs, fe, nchan)

    # ---- 峰值流量 & 到达时间 ----
    peak_idx  = np.nanargmax(burst_profile)
    flux_peak = burst_profile[peak_idx] - noise_mean   # 净峰值流量（减基线）
    toa_sample = ts + peak_idx
    toa_mjd    = file_mjd + toa_sample * time_reso / 86400.0

    # ---- 信噪比 ----
    snr = flux_peak / noise_std if noise_std > 0 else 0.0

    # ---- 积分流量 (Jy·ms) ----
    fluence = np.nansum(burst_profile - noise_mean) * (time_reso * 1e3)

    # ---- Bootstrap 误差估计 ----
    n_burst = te - ts
    n_noise = int(np.sum(noise_mask))
    finite_noise = noise_profile[np.isfinite(noise_profile)]
    if finite_noise.size >= n_burst and n_burst > 0:
        boot_peaks    = np.zeros(n_boot)
        boot_fluences = np.zeros(n_boot)
        for b in range(n_boot):
            idx = np.random.randint(0, finite_noise.size, n_burst)
            boot_prof = finite_noise[idx]
            boot_peaks[b]    = np.max(boot_prof)
            boot_fluences[b] = np.sum(boot_prof - noise_mean) * (time_reso * 1e3)
        flux_err    = np.std(boot_peaks)
        fluence_err = np.std(boot_fluences)
    else:
        flux_err    = noise_std
        fluence_err = noise_std * np.sqrt(max(n_burst, 1)) * (time_reso * 1e3)

    # ---- 增益系统误差 -> flux / fluence 的系统误差 ----
    # 老代码: burst_flux * mean_gain * (1/mean_gain - 1/(mean_gain+mean_gain_err))
    # 取爆发频率范围内的增益均值, 排除 RFI 通道 (全时段都是 NaN 的频道).
    # 必须在 width_err 之前计算, width_err 公式同时叠加测量与系统误差.
    if gain is not None and gain_err is not None and np.any(freq_index):
        rfi_chan_1d = np.all(rfi_mask, axis=0) if rfi_mask.ndim == 2 \
                      else rfi_mask.astype(bool)
        valid_chan = freq_index & ~rfi_chan_1d
        if not np.any(valid_chan):
            valid_chan = freq_index
        mean_gain     = float(np.mean(gain[valid_chan]))
        mean_gain_err = float(np.mean(gain_err[valid_chan]))
        if mean_gain > 0 and (mean_gain + mean_gain_err) > 0:
            sys_factor = mean_gain * (1.0 / mean_gain
                                      - 1.0 / (mean_gain + mean_gain_err))
        else:
            sys_factor = 0.0
        flux_err_sys    = abs(flux_peak * sys_factor)
        fluence_err_sys = abs(fluence   * sys_factor)
    else:
        flux_err_sys    = 0.0
        fluence_err_sys = 0.0

    # ---- 等效宽度 (ms) ----
    # width_err 公式与老代码 cal_energy_burstbw 对齐, 系统误差与测量误差线性叠加:
    #   width_err = width * sqrt(((flu_sys+flu_mea)/flu)^2 + ((flux_sys+flux_mea)/flux)^2)
    width = fluence / flux_peak if flux_peak > 0 else 0.0
    if flux_peak > 0 and fluence > 0:
        total_flu_err  = fluence_err + fluence_err_sys
        total_flux_err = flux_err    + flux_err_sys
        width_err = width * np.sqrt((total_flu_err  / fluence)   ** 2 +
                                    (total_flux_err / flux_peak) ** 2)
    else:
        width_err = 0.0

    # ---- 高斯拟合: 时间轮廓宽度 (ms) ----
    # 对减去噪声均值后的爆发轮廓做高斯拟合
    burst_net = burst_profile - noise_mean
    t_samples = np.arange(len(burst_net), dtype=float)
    w_gauss, w_gauss_err, _, _ = _fit_gaussian(t_samples, burst_net)
    # 转换为毫秒
    if np.isfinite(w_gauss):
        width_gauss = w_gauss * time_reso * 1e3
        width_gauss_err = w_gauss_err * time_reso * 1e3
    else:
        width_gauss = np.nan
        width_gauss_err = np.nan

    # ---- 频率范围 ----
    freq_low  = freq[fs] if fs < nchan else freq[0]
    freq_high = freq[min(fe, nchan) - 1] if fe > 0 else freq[-1]
    bandwidth = abs(freq_high - freq_low)

    # ---- 高斯拟合: 频率带宽 (MHz) ----
    # 对爆发频率谱(时间平均)减去"噪声区逐通道平均谱"做高斯拟合. 逐通道基线
    # 比全局 scalar noise_mean 更准 — 不同频道的仪器基线不同.
    noise_spectrum = np.nanmean(data[noise_mask, :], axis=0)
    burst_spectrum = np.nanmean(data[ts:te, :], axis=0) - noise_spectrum
    freq_arr = freq.astype(float)
    bw_gauss, bw_gauss_err, _, _ = _fit_gaussian(freq_arr, burst_spectrum)
    if not np.isfinite(bw_gauss):
        bw_gauss = np.nan
        bw_gauss_err = np.nan

    return {
        'toa_mjd':          toa_mjd,
        'flux_peak':        float(flux_peak),
        'flux_err':         float(flux_err),
        'flux_err_sys':     float(flux_err_sys),
        'fluence':          float(fluence),
        'fluence_err':      float(fluence_err),
        'fluence_err_sys':  float(fluence_err_sys),
        'width':            float(width),
        'width_err':        float(width_err),
        'width_gauss':      float(width_gauss),
        'width_gauss_err':  float(width_gauss_err),
        'snr':              float(snr),
        'freq_low':         float(freq_low),
        'freq_high':        float(freq_high),
        'bandwidth':        float(bandwidth),
        'bandwidth_gauss':      float(bw_gauss),
        'bandwidth_gauss_err':  float(bw_gauss_err),
    }
