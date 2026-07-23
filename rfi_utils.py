"""通用 RFI 标记工具。

流程中有两个地方需要做 RFI 标记：
  1. calibration 阶段：信号位置未知，把整段数据当作噪声区域来统计 RFI。
  2. analysis 阶段：已通过 burst_detect 得到信号位置，噪声区域是非 burst 时段，
     噪声统计不受信号污染，结果更干净。

两处本质上是同一套算法（熵或 FFT + 像素异常），差异只在传入的 noise_mask。
因此统一抽到本模块。核心逻辑沿用 processing_old/data_process_rfi.cal_rfi 的
熵方法（在旧流程中经过大量观测验证）。
"""

import numpy as np
from scipy.stats import entropy, median_abs_deviation
from scipy.ndimage import median_filter


def z_score_flagger(data, sigma):
    """基于中值 + MAD 的 z-score 异常点标记。

    MAD 对少量离群点更鲁棒，比标准差合适。

    Parameters
    ----------
    data : ndarray
        任意形状的数值数组。
    sigma : float
        偏离中值的 MAD 倍数阈值。

    Returns
    -------
    mask : ndarray (与 data 同形) bool
        True = 异常点。
    """
    med = np.nanmedian(data)
    mad = median_abs_deviation(data, axis=None, nan_policy='omit')
    return np.abs(data - med) > sigma * mad


def cal_rfi(data, noise_mask, down_time=1, down_freq=1, fft=False):
    """通用 RFI 标记：返回 RFI 通道布尔数组 + 异常像素布尔图。

    步骤：
      1. 像素级：对去通道均值后的数据用 50σ MAD 阈值标记孤立异常像素；
      2. 下采样：对 (时间, 频率) 做均值下采样以加速并平滑噪声；
      3. 通道级：在噪声区段上用熵或 FFT 方法标记持续污染的通道。

    Parameters
    ----------
    data : ndarray (nsamp, nchan)
        Stokes I 或定标后功率谱。函数内部会先复制再处理，不改原数组。
    noise_mask : ndarray (nsamp,) bool
        True = 噪声采样点。
        - calibration 阶段：没有信号标注时，传全 True（整段当噪声）。
        - analysis 阶段：由 burst 区间反推得到（非 burst 时段）。
    down_time, down_freq : int
        通道级检测前的下采样因子，既能加速又能平滑随机涨落。
    fft : bool
        False（默认）= 熵方法，对 FAST 数据经验最稳。
        True = FFT 最大幅度法，遇到明显周期性 RFI 时可选。

    Returns
    -------
    rfi_channel : ndarray (nchan,) bool
        True = 该频率通道被判为 RFI。
    rfi_pixel : ndarray (nsamp, nchan) bool
        True = 该像素为孤立异常点。
    """
    work = data.copy().astype(np.float32, copy=False)

    # 1) 像素级：50σ 孤立异常点（保留原始分辨率）
    centered  = work - np.nanmean(work, axis=0, keepdims=True)
    rfi_pixel = z_score_flagger(centered, sigma=50)
    work[rfi_pixel] = np.nanmedian(work)

    # 2) 下采样：截断到 down_time / down_freq 的整数倍
    nsamp, nchan = work.shape
    nt, nf = nsamp // down_time, nchan // down_freq
    if nt == 0 or nf == 0:
        return np.zeros(nchan, dtype=bool), rfi_pixel

    work_ds  = work[:nt * down_time, :nf * down_freq].reshape(nt, down_time, nf, down_freq).mean(axis=(1, 3))
    noise_ds = noise_mask[:nt * down_time].reshape(nt, down_time).mean(axis=1) > 0.5
    if noise_ds.sum() < 3:
        return np.zeros(nchan, dtype=bool), rfi_pixel

    # 3) 通道级：在噪声段上用熵 or FFT 找持续污染的通道
    ds_noise = work_ds[noise_ds]
    if fft:
        mag      = np.max(np.abs(np.fft.fft(ds_noise, axis=0)[1:]), axis=0)
        chan_bad = mag > np.nanpercentile(mag, 99)
    else:
        ent            = entropy(np.exp(-ds_noise), axis=0)
        chan_bad       = z_score_flagger(ent, sigma=20)
        base           = np.mean(ds_noise, axis=0)
        base[chan_bad] = np.nanmedian(base)
        chan_bad       = chan_bad | z_score_flagger(base, sigma=20)

    rfi_channel = np.zeros(nchan, dtype=bool)
    rfi_channel[:nf * down_freq] = np.repeat(chan_bad, down_freq)
    return rfi_channel, rfi_pixel


def robust_channel_mask(data, noise_mask, sigma=6.0,
                        local_window=31, grow=1):
    """标记持续窄带 RFI，只返回通道级 mask。

    ``cal_rfi`` 的熵方法擅长抓很强的坏通道，但偏振分析里仍可能残留一些
    只在 Q/U 中明显、或相对邻近频道噪声略高的窄带污染。这里在非 burst
    时段上对每个 Stokes 分别计算三种稳健时间统计量：MAD、相邻时间差 MAD、
    以及绝对残差的 99 百分位。每个统计量先除去频率方向的局部中值趋势，再
    用全带宽 MAD 找正向离群频道，最后在频率轴上扩展少量邻道以覆盖泄漏。

    该函数不会生成或应用逐像素 mask，适合 RM/偏振测量前的通道级清理。

    Parameters
    ----------
    data : ndarray (nsamp, nchan) or (npol, nsamp, nchan)
        已按频道减去非 burst 基线的 Stokes 数据。
    noise_mask : ndarray (nsamp,) bool
        True 表示用于估计 RFI 的非 burst 采样点。
    sigma : float
        局部趋势去除后的稳健离群阈值；数值越小，屏蔽越积极。
    local_window : int
        频率方向局部中值窗口，自动修正为不小于 3 的奇数。
    grow : int
        每个坏频道向两侧扩展的频道数。

    Returns
    -------
    mask : ndarray (nchan,) bool
        True 表示应在后续分析中整频道屏蔽。
    """
    planes = np.asarray(data, dtype=np.float64)
    if planes.ndim == 2:
        planes = planes[np.newaxis, ...]
    if planes.ndim != 3:
        raise ValueError('data 必须是 (nsamp,nchan) 或 (npol,nsamp,nchan)')

    noise_mask = np.asarray(noise_mask, dtype=bool)
    if noise_mask.ndim != 1 or noise_mask.size != planes.shape[1]:
        raise ValueError('noise_mask 长度必须等于时间采样数')
    if np.count_nonzero(noise_mask) < 3:
        return np.zeros(planes.shape[2], dtype=bool)

    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        return np.zeros(planes.shape[2], dtype=bool)
    local_window = max(3, int(local_window))
    if local_window % 2 == 0:
        local_window += 1
    grow = max(0, int(grow))

    combined = np.zeros(planes.shape[2], dtype=bool)
    for plane in planes:
        noise = plane[noise_mask]
        center = np.nanmedian(noise, axis=0, keepdims=True)
        residual = noise - center
        diff = np.diff(residual, axis=0)
        metrics = (
            1.4826 * np.nanmedian(np.abs(residual), axis=0),
            1.4826 * np.nanmedian(
                np.abs(diff - np.nanmedian(diff, axis=0, keepdims=True)),
                axis=0),
            np.nanpercentile(np.abs(residual), 99, axis=0),
        )

        for metric in metrics:
            positive = metric[np.isfinite(metric) & (metric > 0)]
            if positive.size == 0:
                continue
            floor = max(float(np.nanmedian(positive)) * 1e-6,
                        np.finfo(np.float64).tiny)
            log_metric = np.log10(np.maximum(metric, floor))
            local = median_filter(log_metric, size=local_window,
                                  mode='nearest')
            excess = log_metric - local
            center_excess = np.nanmedian(excess)
            scale = 1.4826 * np.nanmedian(
                np.abs(excess - center_excess))
            if not np.isfinite(scale) or scale <= 0:
                continue
            combined |= excess > center_excess + sigma * scale

    if grow > 0 and np.any(combined):
        kernel = np.ones(2 * grow + 1, dtype=np.int16)
        combined = np.convolve(combined.astype(np.int16), kernel,
                               mode='same') > 0
    return combined
