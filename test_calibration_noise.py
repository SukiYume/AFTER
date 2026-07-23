from pathlib import Path

import numpy as np
from astropy.io import fits

from calibration_noise import NOISE_CLOCK_PRODUCT, fold_noise_cal


def _write_synthetic_noise_cal(path: Path):
    nsub, nsblk, npol, nchan = 4, 64, 4, 64
    period = 16
    tbin = NOISE_CLOCK_PRODUCT / (period * 1e9)
    frequency = np.linspace(1000.0, 1500.0, nchan, endpoint=False)
    phase = np.linspace(-0.8 * np.pi, 0.8 * np.pi, nchan)

    baseline = np.asarray((20.0, 18.0, 0.5, -0.25))[:, None]
    cal_step = np.stack(
        (
            np.full(nchan, 4.0),
            np.full(nchan, 4.0),
            3.0 * np.cos(phase),
            3.0 * np.sin(phase),
        )
    )
    data = np.empty((nsub * nsblk, npol, nchan), dtype=np.float32)
    for sample in range(data.shape[0]):
        is_on = (sample % period) < (period // 2)
        data[sample] = baseline + (cal_step if is_on else 0.0)
    data = data.reshape(nsub, nsblk, npol, nchan)

    data_column = fits.Column(
        name="DATA",
        format=f"{nsblk * npol * nchan}E",
        dim=f"({nchan},{npol},{nsblk})",
        array=data.reshape(nsub, -1),
    )
    frequency_column = fits.Column(
        name="DAT_FREQ",
        format=f"{nchan}D",
        array=np.tile(frequency, (nsub, 1)),
    )
    subint = fits.BinTableHDU.from_columns([frequency_column, data_column])
    subint.header["NSBLK"] = nsblk
    subint.header["NPOL"] = npol
    subint.header["NCHAN"] = nchan
    subint.header["TBIN"] = tbin
    subint.header["POL_TYPE"] = "AABBCRCI"
    fits.HDUList([fits.PrimaryHDU(), subint]).writeto(path)
    return cal_step


def test_fold_noise_cal_writes_phase_diagnostic(tmp_path):
    fits_path = tmp_path / "synthetic-M01_0001.fits"
    expected = _write_synthetic_noise_cal(fits_path)

    noise_cal = fold_noise_cal(fits_path, diagnostic_dir=tmp_path)

    np.testing.assert_allclose(noise_cal, expected, atol=1e-6)
    diagnostic = tmp_path / "synthetic-M01_0001_noise_cal_diagnostic.png"
    assert diagnostic.is_file()
    assert diagnostic.stat().st_size > 100_000
