"""
偏振分析模块。

包括: RM 合成（numba 并行）、法拉第反旋转、偏振位置角 (PA) 轮廓、
线偏振 / 圆偏振分数计算，以及 analyze_pol 一体化编排。
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns   # Registers seaborn colormaps such as "mako".
from numba import njit, prange
from astropy import constants as const


# ============================================================
# RM 合成
# ============================================================

@njit(parallel=True)
def rm_synthesis(I, Q, U, wave, rm_min=-10000, rm_max=10000, n_rm=20000):
    """Numba 并行化的 RM 合成。

    对每个试验 RM 值，将 Q/U 反旋转后频率求和，
    得到线偏振度随 RM 的变化曲线。

    Parameters
    ----------
    I, Q, U : ndarray (nsamp, nchan)
        Stokes 参量。
    wave : ndarray (nchan,)
        波长 (米)。
    rm_min, rm_max : float
        RM 搜索范围 (rad/m²)。
    n_rm : int
        RM 试验点数。

    Returns
    -------
    rm_list : ndarray (n_rm,)
        试验 RM 值。
    linear_pol : ndarray (n_rm,)
        归一化线偏振度（0–1）。
    """
    rm_list = np.linspace(rm_min, rm_max, n_rm)
    linear = np.zeros(n_rm)

    I_total = np.sum(I)
    if I_total == 0:
        I_total = 1.0    # 避免除零，全零数据 → 线偏振度 = 0

    for i in prange(n_rm):
        PA = 2 * rm_list[i] * wave ** 2
        cos_PA = np.cos(PA)
        sin_PA = np.sin(PA)
        # 反旋转 Q, U
        Q_C =  cos_PA * Q + sin_PA * U
        U_C = -sin_PA * Q + cos_PA * U
        # 频率求和后的线偏振幅
        Lsum = np.sum(np.sqrt(np.sum(Q_C, axis=1) ** 2 +
                               np.sum(U_C, axis=1) ** 2))
        linear[i] = Lsum / I_total

    return rm_list, linear


# ============================================================
# RM 峰值定位
# ============================================================

def find_rm(rm_list, linear_pol, snr, wave=None, significance_threshold=5.0):
    """定位 RM 曲线峰值，评估显著性，计算误差。

    误差估计使用 RMSF 理论半高宽:
        σ_RM = FWHM_RMSF / (2 × SNR)     (Brentjens & de Bruijn 2005)
    其中 FWHM_RMSF ≈ 2√3 / Δ(λ²)。

    当峰值显著性低于阈值时，返回的 rm_err 为 NaN，提示结果不可靠。

    Parameters
    ----------
    rm_list : ndarray
        试验 RM 值。
    linear_pol : ndarray
        线偏振度曲线。
    snr : float
        偏振 SNR。
    wave : ndarray or None
        波长数组 (m)，用于计算理论 RMSF 宽度。
        若为 None，则从 RM 合成曲线实测半高宽。
    significance_threshold : float
        峰值显著性阈值（倍噪声），低于此值标记为不可靠。

    Returns
    -------
    rm_best : float
        最优 RM (rad/m²)。
    rm_err : float
        RM 对称误差 (rad/m²)。峰值不显著时为 NaN。
    rm_significance : float
        RM 峰值显著性 (peak - baseline) / σ_noise。
    """
    peak_idx = np.argmax(linear_pol)
    rm_best = float(rm_list[peak_idx])
    peak_val = linear_pol[peak_idx]

    # ---- 1. 评估峰值显著性 ----
    # 用 MAD 估计 RM 合成曲线的噪声水平（鲁棒，不受峰值影响）
    baseline = np.median(linear_pol)
    noise_sigma = np.median(np.abs(linear_pol - baseline)) * 1.4826
    if noise_sigma > 0:
        rm_significance = float((peak_val - baseline) / noise_sigma)
    else:
        # 曲线完全平坦（如全零数据）
        rm_significance = 0.0

    # ---- 2. 计算 RMSF 半高宽 ----
    if wave is not None and len(wave) > 1:
        # 理论公式: FWHM_RMSF ≈ 2√3 / (λ_max² − λ_min²)
        lambda2 = wave ** 2
        delta_lambda2 = np.max(lambda2) - np.min(lambda2)
        if delta_lambda2 > 0:
            fwhm_rmsf = 2.0 * np.sqrt(3.0) / delta_lambda2
        else:
            fwhm_rmsf = _measure_fwhm(rm_list, linear_pol)
    else:
        fwhm_rmsf = _measure_fwhm(rm_list, linear_pol)

    # ---- 3. RM 误差 = FWHM / (2 × SNR) ----
    snr_safe = max(snr, 1.0)
    rm_err = fwhm_rmsf / (2.0 * snr_safe)

    # 峰值不显著时标记为不可靠
    if rm_significance < significance_threshold:
        rm_err = np.nan

    return rm_best, float(rm_err), float(rm_significance)


def _measure_fwhm(rm_list, linear_pol):
    """从 RM 合成曲线实测半高宽（无波长信息时的 fallback）。"""
    peak_idx = np.argmax(linear_pol)
    peak_val = linear_pol[peak_idx]
    baseline = np.median(linear_pol)
    half_max = (peak_val + baseline) / 2.0

    # 从峰值向左右搜索到半高点
    drm = np.abs(rm_list[1] - rm_list[0]) if len(rm_list) > 1 else 1.0

    # 左侧
    left = peak_idx
    while left > 0 and linear_pol[left] > half_max:
        left -= 1

    # 右侧
    right = peak_idx
    while right < len(linear_pol) - 1 and linear_pol[right] > half_max:
        right += 1

    fwhm = (right - left) * drm
    return max(fwhm, drm)  # 至少一个 RM 步长


# ============================================================
# 法拉第反旋转
# ============================================================

def correct_rm(Q, U, freq, rm):
    """按给定 RM 对 Q, U 做法拉第反旋转。

    Parameters
    ----------
    Q, U : ndarray (nsamp, nchan)
    freq : ndarray (nchan,) MHz
    rm : float (rad/m²)

    Returns
    -------
    Q_corr, U_corr : ndarray (nsamp, nchan)
    """
    wave = const.c.value / (freq * 1e6)   # Hz → 波长 (m)
    PA = 2 * rm * wave ** 2
    Q_C =  np.cos(PA) * Q + np.sin(PA) * U
    U_C = -np.sin(PA) * Q + np.cos(PA) * U
    return Q_C, U_C


# ============================================================
# 偏振信噪比 & 中心频率
# ============================================================

def calc_pol_snr(I_burst, freq, noise_I):
    """计算偏振信噪比、噪声 rms、强度加权中心频率。

    Parameters
    ----------
    I_burst : ndarray (nsamp_burst, nchan)
        爆发区域的 Stokes I。
    freq : ndarray (nchan,)
        频率数组 (MHz)。
    noise_I : ndarray (nsamp_noise, nchan)
        噪声区域的 Stokes I。

    Returns
    -------
    rms : float
        噪声区域频率平均轮廓的标准差。
    snr : float
        偏振信噪比。
    center_freq : float
        强度加权中心频率 (MHz)。
    """
    # 噪声 rms（保护空数组）
    if noise_I.size > 0 and noise_I.shape[0] > 1:
        noise_profile = np.nanmean(noise_I, axis=1)
        rms = np.nanstd(noise_profile)
    else:
        rms = 0.0

    # 偏振 SNR（保护空爆发 / 零 rms）
    if I_burst.size > 0 and I_burst.shape[0] > 0 and rms > 0:
        burst_profile = np.nanmean(I_burst, axis=1)
        snr = np.nansum(burst_profile) / np.sqrt(I_burst.shape[0]) / rms
    else:
        snr = 0.0

    # 强度加权中心频率
    if I_burst.size > 0:
        weights = np.nanmean(I_burst, axis=0)
        weights = np.clip(weights, 0, None)
    else:
        weights = np.zeros_like(freq)

    if np.sum(weights) > 0:
        center_freq = np.average(freq, weights=weights)
    else:
        center_freq = np.mean(freq)

    return rms, snr, center_freq


# ============================================================
# 偏振位置角 (PA) 轮廓
# ============================================================

def calc_pa_profile(I, Q, U, V, burst_mask, freq_mask, noise_mask):
    """计算 PA 轮廓和归一化 Stokes 轮廓。

    PA = 0.5 * arctan2(U, Q)，误差通过 Q/U 噪声传播计算。
    线偏振 L = sqrt(Q² + U²)，做 Wardle & Kronberg 去偏修正（阈值 1.57σ）。

    Parameters
    ----------
    I, Q, U, V : ndarray (nsamp, nchan)
        RM 校正后的 Stokes 参量。
    burst_mask : ndarray (nsamp,) bool
        True = 爆发采样点。
    freq_mask : ndarray (nchan,) bool
        True = 有效频率通道（未被 RFI 全部标记）。
    noise_mask : ndarray (nsamp,) bool
        True = 噪声采样点。

    Returns
    -------
    PAT : ndarray (nsamp,)
        时间索引。
    PAV : ndarray (nsamp,)
        PA 值 (度)，爆发区域外为 NaN。
    PAE : ndarray (nsamp,)
        PA 误差 (度)。
    profile_I, profile_L, profile_V : ndarray (nsamp,)
        归一化（除以峰值）的 I / L / V 轮廓。
    rms_norm : float
        归一化后的噪声 rms。
    """
    nsamp = I.shape[0]
    n_freq_valid = np.sum(freq_mask)

    # 无有效频率通道时返回全零/全 NaN
    if n_freq_valid == 0:
        nan_arr = np.full(nsamp, np.nan)
        zero_arr = np.zeros(nsamp)
        return (np.arange(nsamp, dtype=np.float64),
                nan_arr.copy(), nan_arr.copy(),
                zero_arr.copy(), zero_arr.copy(), zero_arr.copy(), 1.0)

    # 在有效通道上取频率平均
    profile_I = np.nanmean(I[:, freq_mask], axis=1)
    profile_Q = np.nanmean(Q[:, freq_mask], axis=1)
    profile_U = np.nanmean(U[:, freq_mask], axis=1)
    profile_V = np.nanmean(V[:, freq_mask], axis=1)

    # 归一化因子（保护全 NaN / 全负值）
    peak_I = np.nanmax(profile_I) if np.any(np.isfinite(profile_I)) else 0.0
    normfactor = peak_I if peak_I > 0 else 1.0

    # 噪声统计量（用 nanstd 处理残余 NaN）
    n_noise = np.sum(noise_mask)
    if n_noise > 1:
        rms     = np.nanstd(profile_I[noise_mask])
        sigma_Q = np.nanstd(profile_Q[noise_mask])
        sigma_U = np.nanstd(profile_U[noise_mask])
    else:
        rms = sigma_Q = sigma_U = 1.0

    # 如果噪声估计为 0（数据完全恒定），设为安全值
    if rms == 0:
        rms = 1.0
    if sigma_Q == 0:
        sigma_Q = rms
    if sigma_U == 0:
        sigma_U = rms

    # PA 及其误差
    PAV = np.rad2deg(np.arctan2(profile_U, profile_Q) / 2)
    L2 = profile_Q ** 2 + profile_U ** 2
    denom = 4 * L2 ** 2
    denom[denom == 0] = 1.0
    PAE = np.rad2deg(np.sqrt(
        (profile_Q ** 2 * sigma_U ** 2 + profile_U ** 2 * sigma_Q ** 2)
        / denom))
    PAT = np.arange(nsamp, dtype=np.float64)

    # 爆发区域外置 NaN，高误差点也置 NaN
    burst_idx = burst_mask.astype(int)
    PAV[burst_idx != 1] = np.nan
    PAE[burst_idx != 1] = np.nan

    # PA 误差阈值: 用爆发区域内有限值的 nanstd
    burst_PAE = PAE[burst_idx == 1]
    if np.any(np.isfinite(burst_PAE)):
        thres = np.nanstd(burst_PAE)
        bad = PAE > thres * 2
        PAV[bad] = np.nan
        PAE[bad] = np.nan

    # 线偏振: Wardle & Kronberg (1974) 去偏修正
    # 使用 Q/U 噪声的平均值作为偏振噪声 σ_p
    sigma_p = np.sqrt((sigma_Q ** 2 + sigma_U ** 2) / 2)
    profile_L = np.sqrt(profile_Q ** 2 + profile_U ** 2)
    if sigma_p > 0:
        low_snr  = profile_L / sigma_p <= 1.57
        high_snr = ~low_snr
        profile_L[low_snr]  = 0
        profile_L[high_snr] = np.sqrt(
            np.clip(profile_L[high_snr] ** 2 - sigma_p ** 2, 0, None))
    # sigma_p == 0 时跳过去偏修正，保留原始 L（无法估计偏差）

    # 归一化
    profile_I = profile_I / normfactor
    profile_L = profile_L / normfactor
    profile_V = profile_V / normfactor

    return PAT, PAV, PAE, profile_I, profile_L, profile_V, rms / normfactor


# ============================================================
# 偏振分数
# ============================================================

def calc_pol_fractions(profile_I, profile_L, profile_V, burst_mask, rms):
    """计算爆发区域的线偏振和圆偏振百分比。

    Parameters
    ----------
    profile_I, profile_L, profile_V : ndarray (nsamp,)
        归一化后的 Stokes 轮廓。
    burst_mask : ndarray (nsamp,) bool
        True = 爆发采样点。
    rms : float
        归一化后的噪声 rms。

    Returns
    -------
    linear_frac, linear_err : float
        线偏振分数及误差 (%)。
    circular_frac, circular_err : float
        圆偏振分数及误差 (%)。
    """
    burst_idx = burst_mask.astype(int)
    I_sum = profile_I[burst_idx == 1].sum()
    L_sum = profile_L[burst_idx == 1].sum()
    V_sum = profile_V[burst_idx == 1].sum()

    n_burst = np.sum(burst_idx == 1)
    snr = I_sum / np.sqrt(n_burst) / rms if (rms > 0 and n_burst > 0) else 1.0
    snr = max(snr, 1.0)

    if I_sum > 0:
        linear_frac  = L_sum / I_sum * 100
        linear_err   = np.sqrt(1 + L_sum ** 2 / I_sum ** 2) / snr * 100
        circular_frac = V_sum / I_sum * 100
        circular_err  = np.sqrt(1 + V_sum ** 2 / I_sum ** 2) / snr * 100
    else:
        linear_frac = linear_err = circular_frac = circular_err = 0.0

    return linear_frac, linear_err, circular_frac, circular_err


# ============================================================
# 绘图
# ============================================================

def plot_polarization(PAT, PAV, PAE, profile_I, profile_L, profile_V,
                      stokes_I_2d, freq, time_reso, save_path, fmt='png'):
    """三面板偏振图: PA、Stokes 轮廓、动态谱。

    fmt='png' 用于单 burst 输出; fmt='pdf' 用于跨爆发合并 (复刻老代码
    plot_spec(..., comb=True)), PAV / PAE 传入合并后的全时段数组即可 —
    布局完全一样, 只是输出格式不同.
    """
    nsamp, nchan = stokes_I_2d.shape

    fig = plt.figure(figsize=(4, 5))
    gs = gridspec.GridSpec(11, 1)
    plt.subplots_adjust(hspace=0)

    ax0 = plt.subplot(gs[0:3, 0])
    ax0.errorbar(PAT, PAV, PAE, color='r', fmt='.', capsize=3, lw=1, ms=1)
    ax0.axhline( 90, ls='--', color='gray', alpha=0.5, lw=0.5)
    ax0.axhline(-90, ls='--', color='gray', alpha=0.5, lw=0.5)
    ax0.set_xticks([])
    ax0.set_xlim(0, nsamp)
    ax0.set_ylim(-120, 120)
    ax0.set_ylabel('PA (deg)')

    ax1 = plt.subplot(gs[3:6, 0])
    ax1.step(PAT, profile_I, where='mid', color='gray',      alpha=0.8, label='I')
    ax1.step(PAT, profile_L, where='mid', color='r',         alpha=0.8, label='L')
    ax1.step(PAT, profile_V, where='mid', color='royalblue', alpha=0.8, label='V')
    ax1.set_xlim(0, nsamp)
    ax1.set_xticks([])
    ax1.set_ylabel('Intensity (abbr.)')

    ax2 = plt.subplot(gs[6:, 0])
    vmin, vmax = np.nanpercentile(stokes_I_2d, [5, 95])
    ax2.imshow(stokes_I_2d.T, cmap='mako', vmin=vmin, vmax=vmax,
               aspect='auto', origin='lower')
    xticks = np.linspace(0, nsamp, 6)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels((xticks * time_reso * 1e3).astype(int))
    yticks = np.linspace(0, nchan, 6)
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(np.linspace(freq[0], freq[-1], 6).astype(int))
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Frequency (MHz)')

    fig.align_labels()
    if fmt == 'pdf':
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    else:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_rm_synthesis(rm_list, linear_pol, rm_best, save_path):
    """RM 合成曲线。"""
    plt.figure(figsize=(5, 4))
    plt.plot(rm_list, linear_pol * 100, color='royalblue', alpha=0.7)
    plt.axvline(rm_best, color='red', ls='--', alpha=0.7,
                label=f'RM={rm_best:.1f}')
    plt.xlabel('RM (rad/m$^2$)')
    plt.ylabel('Degree of Linear Polarization (%)')
    plt.legend()
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# 一体化编排: 单个爆发的偏振分析
# ============================================================

def analyze_pol(I, Q, U, V, freq, time_reso, burst_mask, freq_index, noise_mask,
                output_dir, burst_idx,
                rm_min=-50000, rm_max=50000, n_rm=20000):
    """对单个爆发做完整的偏振分析并画图。

    上游 (burst_analysis.py) 已经把 RFI 和基线处理好, 这里只关心偏振本身:
    RM 合成 → find_rm → 反旋转 → PA 轮廓 → 偏振分数 → 两张图。

    RM 搜索范围默认对称 ±50000 rad/m² — 部分源 RM 为负, 非对称默认会漏掉.

    Parameters
    ----------
    I, Q, U, V : ndarray (nsamp, nchan)
        已减基线、已屏蔽 RFI(置 NaN) 的 Stokes 参量。
    freq : ndarray (nchan,)  MHz
    time_reso : float        有效时间分辨率 (秒)
    burst_mask : ndarray (nsamp,) bool   True = 该爆发的时间范围
    freq_index : ndarray (nchan,) bool   True = 该爆发的有效频率通道(爆发频率范围 ∩ 非RFI)
    noise_mask : ndarray (nsamp,) bool   True = 噪声时段
    output_dir : str         图片保存目录
    burst_idx : int          爆发编号, 用于命名
    rm_min, rm_max, n_rm     RM 搜索参数

    Returns
    -------
    scalars : dict
        CSV 一行用的标量: rm, rm_err, rm_significance, linear_frac,
        linear_frac_err, circular_frac, circular_frac_err, center_freq。
    pa_arrays : dict
        供跨爆发合并 PDF 的轮廓数组: PAT, PAV, PAE, profile_I,
        profile_L, profile_V (都是 nsamp 长度, 爆发窗口外为 NaN)。
    """
    wave = const.c.value / (freq * 1e6)   # 波长 (m)

    # 爆发时间 × 有效通道: 干净数据, NaN → 0 送入 numba
    ts_idx      = np.where(burst_mask)[0]
    ts, te      = ts_idx[0], ts_idx[-1] + 1
    burst_I     = np.nan_to_num(I[ts:te][:, freq_index], nan=0.0)
    burst_Q     = np.nan_to_num(Q[ts:te][:, freq_index], nan=0.0)
    burst_U     = np.nan_to_num(U[ts:te][:, freq_index], nan=0.0)
    burst_wave  = wave[freq_index]

    # RM 合成
    rm_list_out, linear_pol = rm_synthesis(
        burst_I, burst_Q, burst_U, burst_wave,
        rm_min=rm_min, rm_max=rm_max, n_rm=n_rm)

    # 偏振 SNR / 中心频率
    noise_I = np.nan_to_num(I[noise_mask][:, freq_index], nan=0.0)
    _, snr_pol, center_freq = calc_pol_snr(
        burst_I, freq[freq_index], noise_I)

    rm_best, rm_err, rm_significance = find_rm(
        rm_list_out, linear_pol, snr_pol, wave=burst_wave)

    if np.isnan(rm_err):
        print(f'    RM 峰值不显著 (significance={rm_significance:.1f}), '
              f'结果可能不可靠')

    plot_rm_synthesis(rm_list_out, linear_pol, rm_best,
                      os.path.join(output_dir, f'burst{burst_idx}_rm.png'))

    # RM 反旋转: 全频率做(NaN 通道不影响, 后面用 freq_index 过滤)
    Q_corr, U_corr = correct_rm(Q, U, freq, rm_best)

    # PA 轮廓 + 偏振分数
    PAT, PAV, PAE, prof_I, prof_L, prof_V, rms_norm = calc_pa_profile(
        I, Q_corr, U_corr, V, burst_mask, freq_index, noise_mask)

    lin_frac, lin_err, circ_frac, circ_err = calc_pol_fractions(
        prof_I, prof_L, prof_V, burst_mask, rms_norm)

    plot_polarization(PAT, PAV, PAE, prof_I, prof_L, prof_V,
                      I, freq, time_reso,
                      os.path.join(output_dir, f'burst{burst_idx}_pol.png'))

    scalars = {
        'rm':                rm_best,
        'rm_err':            rm_err,
        'rm_significance':   rm_significance,
        'linear_frac':       lin_frac,
        'linear_frac_err':   lin_err,
        'circular_frac':     circ_frac,
        'circular_frac_err': circ_err,
        'center_freq':       center_freq,
    }
    pa_arrays = {
        'PAT':       PAT,
        'PAV':       PAV,
        'PAE':       PAE,
        'profile_I': prof_I,
        'profile_L': prof_L,
        'profile_V': prof_V,
    }
    return scalars, pa_arrays
