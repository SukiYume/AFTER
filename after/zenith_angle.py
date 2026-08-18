"""根据观测时刻和波束计算 FAST 天顶角及频率相关增益。

``get_za`` 用 FAST 台址和源坐标把 MJD 转成天顶角；``get_gain`` 再读取实测增益参数，
先按天顶角求九个参考频点的增益，随后插值到数据的全部频率通道。返回的增益及误差
单位都是 K/Jy，供 :mod:`after.calibration` 把噪声管定标温度换算为 Jy。
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

from . import DEFAULT_GAIN_CSV


C_M_S = 299_792_458.0

def get_za(mjd, source_ra='05h08m03.5077', source_dec='+26d03m38.504s'):
    """计算指定 UTC MJD 时刻源在 FAST 台址的天顶角，返回角度值（度）。

    这里显式给出 FAST 的经纬度、高度和典型气象参数，并以 1250 MHz 对应波长进行
    大气折射修正。RA/Dec 接受 Astropy 能识别的时角/角度字符串。
    """
    target            = SkyCoord(ra=source_ra, dec=source_dec, frame='icrs')
    lat               = '25d39m10.626537s'
    lon               = '106d51m24.000740s'
    height            = 1110.028801 * u.m
    pressure          = 925 * u.mBa
    relative_humidity = 0.7
    temperature       = 25 * u.deg_C
    obswl             = (C_M_S * u.m / u.s) / (1250 * u.MHz)
    bear_mountain     = EarthLocation(lon=lon, lat=lat, height=height)

    ob_time      = Time(mjd, format='mjd', scale='utc')
    frame_time   = AltAz(
        obstime           = ob_time,
        location          = bear_mountain,
        pressure          = pressure,
        relative_humidity = relative_humidity,
        temperature       = temperature,
        obswl             = obswl
    )
    target_altaz = target.transform_to(frame_time)
    zenith = target_altaz.zen
    if zenith is None:
        raise ValueError("Astropy did not provide a zenith angle")
    return zenith.value


def get_gain(ZA, beam, freq_reso):
    """计算给定天顶角、波束和通道数下的增益曲线及其误差。

    ``gain_para.csv`` 给出每个波束在九个参考频点上的分段线性系数。天顶角 26.4°
    前后使用不同斜率且保持连续，再乘基准增益 25.6 K/Jy。中心 1050--1450 MHz
    频段做线性插值，两侧频段用最近端点外推，最终返回两个 ``(freq_reso,)`` 数组。
    """

    data           = pd.read_csv(DEFAULT_GAIN_CSV, header=[0, 1])
    gain_zero      = 25.6
    beam           = 'M{:0>2d}'.format(beam)

    a, b, c        = (data.loc[(data.beam.beam==beam), 'freq'] * data.loc[(data.beam.beam==beam), 'coef'].values).values
    a_err, b_err, c_err = (data.loc[(data.beam.beam==beam), 'freq_err'] * data.loc[(data.beam.beam==beam), 'coef'].values).values

    # FAST 在较大天顶角处增益下降规律改变；第二段加连续性修正，避免 26.4° 跳变。
    if ZA > 26.4:
        gain       = c * ZA + b + 26.4 * (a - c)
        gain_err   = c_err * ZA + b_err + 26.4 * (a_err - c_err)
    else:
        gain       = a * ZA + b
        gain_err   = a_err * ZA + b_err
    gain, gain_err = gain * gain_zero, gain_err * gain_zero

    # 增益表覆盖 500 MHz 接收带宽中的中心 400 MHz。先确定中心通道数，再把九点
    # 曲线插值到这些通道；带外两端使用边界增益，避免无约束线性外推。
    center_channel = int(400 / 500 * freq_reso)
    center_channel = center_channel if center_channel % 2 == 0 else center_channel + 1
    edge_channel   = (freq_reso - center_channel) // 2

    a        = interp1d(np.linspace(1050, 1450, 9), gain)
    b        = np.linspace(1050, 1450, center_channel)
    gain     = np.hstack([[gain[0]] * edge_channel, a(b), [gain[-1]] * edge_channel])

    a        = interp1d(np.linspace(1050, 1450, 9), gain_err)
    b        = np.linspace(1050, 1450, center_channel)
    gain_err = np.hstack([[gain_err[0]] * edge_channel, a(b), [gain_err[-1]] * edge_channel])

    return gain, gain_err
