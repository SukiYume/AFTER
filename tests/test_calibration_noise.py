# fmt: off

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from after.calibration_noise import (
    compute_noise_cal_fold,
    fold_noise_cal,
    plot_noise_cal_diagnostic,
)

# 独立于生产常量的合成采样间隔：名义 0.2 s 档对应 16 个采样点。
SYNTHETIC_TBIN_S = 0.012582912


def _write_synthetic_noise_cal(
    path: Path,
    npol=4,
    period=16,
    on_samples=None,
    on_start=0,
):
    nsub, nsblk, nchan = 4, 64, 64
    tbin                = SYNTHETIC_TBIN_S
    frequency           = np.linspace(1000.0, 1500.0, nchan, endpoint=False)
    phase               = np.linspace(-0.8 * np.pi, 0.8 * np.pi, nchan)

    baseline = np.asarray((20.0, 18.0, 0.5, -0.25))[:npol, None]
    cal_step = np.stack(
        (
            np.full(nchan, 4.0),
            np.full(nchan, 4.0),
            3.0 * np.cos(phase),
            3.0 * np.sin(phase),
        )
    )[:npol]
    on_samples = period // 2 if on_samples is None else int(on_samples)
    data       = np.empty((nsub * nsblk, npol, nchan), dtype=np.float32)
    for sample in range(data.shape[0]):
        is_on        = ((sample - on_start) % period) < on_samples
        data[sample] = baseline + (cal_step if is_on else 0.0)
    data = data.reshape(nsub, nsblk, npol, nchan)

    data_column = fits.Column(
        name   = "DATA",
        format = f"{nsblk * npol * nchan}E",
        dim    = f"({nchan},{npol},{nsblk})",
        array  = data.reshape(nsub, -1),
    )
    frequency_column = fits.Column(
        name   = "DAT_FREQ",
        format = f"{nchan}D",
        array  = np.tile(frequency, (nsub, 1)),
    )
    subint                    = fits.BinTableHDU.from_columns([frequency_column, data_column])
    subint.header["NSBLK"]    = nsblk
    subint.header["NPOL"]     = npol
    subint.header["NCHAN"]    = nchan
    subint.header["TBIN"]     = tbin
    subint.header["POL_TYPE"] = "AABBCRCI" if npol == 4 else "AABB"
    fits.HDUList([fits.PrimaryHDU(), subint]).writeto(path)
    return cal_step


def test_fold_noise_cal_writes_phase_diagnostic(tmp_path):
    fits_path = tmp_path / "synthetic-M01_0001.fits"
    expected  = _write_synthetic_noise_cal(fits_path)

    noise_cal = fold_noise_cal(fits_path, diagnostic_dir=tmp_path)

    np.testing.assert_allclose(noise_cal, expected, atol=1e-6)
    diagnostic = tmp_path / "synthetic-M01_0001_noise_cal_diagnostic.png"
    assert diagnostic.is_file()
    assert diagnostic.stat().st_size > 0


def test_fold_noise_cal_accepts_two_auto_correlations(tmp_path):
    fits_path = tmp_path / "synthetic-two-pol-M01_0001.fits"
    expected  = _write_synthetic_noise_cal(fits_path, npol=2)

    noise_cal = fold_noise_cal(fits_path, diagnostic_dir=tmp_path)

    np.testing.assert_allclose(noise_cal, expected, atol=1e-6)
    diagnostic = tmp_path / "synthetic-two-pol-M01_0001_noise_cal_diagnostic.png"
    assert not diagnostic.exists()


@pytest.mark.parametrize(
    ("period", "noise_period_s"),
    (
        pytest.param(8, 0.1, id="0.1s"),
        pytest.param(32, 0.4, id="0.4s"),
        pytest.param(80, 1.0, id="1s"),
        pytest.param(160, 2.0, id="2s"),
    ),
)
def test_fold_noise_cal_accepts_configured_period(
    tmp_path,
    period,
    noise_period_s,
):
    fits_path = tmp_path / f"synthetic-{noise_period_s:g}s-M01_0001.fits"
    expected  = _write_synthetic_noise_cal(fits_path, period=period)

    noise_cal = fold_noise_cal(
        fits_path,
        make_diagnostic = False,
        noise_period_s  = noise_period_s,
    )

    np.testing.assert_allclose(noise_cal, expected, atol=1e-6)


@pytest.mark.parametrize(
    ("npol", "make_diagnostic"),
    (
        pytest.param(2, True, id="two-pol"),
        pytest.param(4, False, id="no-diagnostic"),
    ),
)
def test_fold_noise_cal_rejects_fragmented_on_mask(
    tmp_path,
    npol,
    make_diagnostic,
):
    fits_path = tmp_path / f"wrong-period-{npol}pol-M01_0001.fits"
    _write_synthetic_noise_cal(fits_path, npol=npol, period=16)

    with pytest.raises(ValueError, match="noise_period_s"):
        fold_noise_cal(
            fits_path,
            make_diagnostic = make_diagnostic,
            noise_period_s  = 0.4,
        )


@pytest.mark.parametrize(
    ("period", "on_samples", "on_start", "noise_period_s"),
    (
        pytest.param(64, 8, 60, 0.8, id="one-to-seven-wrapped"),
        pytest.param(128, 8, 124, 1.6, id="one-to-fifteen-wrapped"),
    ),
)
def test_fold_and_diagnostic_infer_low_duty_cycle(
    tmp_path,
    period,
    on_samples,
    on_start,
    noise_period_s,
):
    fits_path = tmp_path / f"synthetic-low-duty-{period}-M01_0001.fits"
    expected  = _write_synthetic_noise_cal(
        fits_path,
        period     = period,
        on_samples = on_samples,
        on_start   = on_start,
    )

    folded = compute_noise_cal_fold(
        fits_path,
        noise_period_s = noise_period_s,
    )

    expected_on_mask = ((np.arange(period) - on_start) % period) < on_samples
    np.testing.assert_array_equal(folded.on_mask, expected_on_mask)
    np.testing.assert_allclose(folded.noise_cal, expected, atol=1e-6)

    diagnostic = tmp_path / f"synthetic-low-duty-{period}.png"
    metrics    = plot_noise_cal_diagnostic(folded, diagnostic)
    assert diagnostic.is_file()
    assert metrics["on_samples"] == on_samples
    assert metrics["off_samples"] == period - on_samples
    assert metrics["on_fraction"] == pytest.approx(on_samples / period)
    assert metrics["on_start_bin"] == on_start
    assert metrics["on_stop_bin"] == (on_start + on_samples) % period

# fmt: on
