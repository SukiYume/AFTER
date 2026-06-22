import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from pathlib import Path

import astropy.units as u
from astropy.time import Time
from astropy.constants import c
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

### 根据时间计算某个源的天顶角
def get_za(mjd, source_ra='05h08m03.5077', source_dec='+26d03m38.504s'):
    target            = SkyCoord(ra=source_ra, dec=source_dec, frame='icrs')
    lat               = '25d39m10.626537s'
    lon               = '106d51m24.000740s'
    height            = 1110.028801 * u.m
    pressure          = 925 * u.mBa
    relative_humidity = 0.7
    temperature       = 25 * u.deg_C
    obswl             = c / (1250 * u.MHz)
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
    return target_altaz.zen.value


### 计算不同天顶角的增益
def get_gain(ZA, beam, freq_reso):

    gain_path      = Path(__file__).with_name('gain_para.csv')
    data           = pd.read_csv(gain_path, header=[0, 1])
    gain_zero      = 25.6
    beam           = 'M{:0>2d}'.format(beam)

    a, b, c        = (data.loc[(data.beam.beam==beam), 'freq'] * data.loc[(data.beam.beam==beam), 'coef'].values).values
    a_err, b_err, c_err = (data.loc[(data.beam.beam==beam), 'freq_err'] * data.loc[(data.beam.beam==beam), 'coef'].values).values

    if ZA > 26.4:
        gain       = c * ZA + b + 26.4 * (a - c)
        gain_err   = c_err * ZA + b_err + 26.4 * (a_err - c_err)
    else:
        gain       = a * ZA + b
        gain_err   = a_err * ZA + b_err
    gain, gain_err = gain * gain_zero, gain_err * gain_zero

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
