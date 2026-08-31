# fmt: off

import os

import numpy as np

from after import burst_dm


def test_dedisperse_waterfall_aligns_integer_channel_delays():
    waterfall       = np.zeros((2, 8), dtype=float)
    waterfall[0, 3] = 1.0
    waterfall[1, 2] = 1.0

    dedispersed = burst_dm.dedisperse_waterfall(
        waterfall,
        dm       = 2.41e-4,
        freq     = np.array([1.0, 2.0]),
        dt       = 1.0,
        ref_freq = "top",
    )

    expected       = np.zeros((2, 8), dtype=float)
    expected[:, 2] = 1.0
    np.testing.assert_array_equal(dedispersed, expected)


def test_coherent_power_spectrum_is_fully_coherent_for_identical_channels():
    channel     = np.zeros(8, dtype=float)
    channel[2]  = 1.0
    waterfall   = np.tile(channel, (4, 1))
    power       = burst_dm.coherent_power_spectrum(waterfall)
    full_power  = np.full(4, 16.0)

    np.testing.assert_allclose(power, full_power)


def test_analyze_dm_reports_success_and_writes_the_expected_plot(monkeypatch, tmp_path):
    stokes_i        = np.arange(120 * 4, dtype=float).reshape(120, 4)
    stokes_i[3, 0]  = np.nan
    captured_search = {}
    captured_plot   = {}

    def fake_dm_phase_search(data, freq, time_reso, dm_zero, **kwargs):
        captured_search.update(
            data      = data.copy(),
            freq      = freq.copy(),
            time_reso = time_reso,
            dm_zero   = dm_zero,
            kwargs    = kwargs,
        )
        return (
            565.2,
            0.3,
            12.0,
            np.array([564.0, 565.0, 566.0]),
            np.array([1.0, 3.0, 2.0]),
        )

    def fake_plot(dm_list, dm_curve, dm_best, dm_err, save_path):
        captured_plot.update(
            dm_list   = dm_list.copy(),
            dm_curve  = dm_curve.copy(),
            dm_best   = dm_best,
            dm_err    = dm_err,
            save_path = save_path,
        )

    monkeypatch.setattr(burst_dm, "dm_phase_search", fake_dm_phase_search)
    monkeypatch.setattr(burst_dm, "plot_dm_search", fake_plot)

    result = burst_dm.analyze_dm(
        stokes_i,
        np.array([1000.0, 1100.0, 1200.0, 1300.0]),
        0.001,
        560.0,
        {"time_start": 10, "time_end": 20},
        str(tmp_path),
        3,
        dm_range      = 8.0,
        dm_step       = 0.2,
        snr_threshold = 6.0,
    )

    assert result == {
        "dm": 565.2,
        "dm_err": 0.3,
        "dm_status": "ok",
        "dm_error_reason": "",
    }
    np.testing.assert_array_equal(
        captured_search["data"],
        np.nan_to_num(stokes_i[:70], nan=0.0),
    )
    assert captured_search["time_reso"] == 0.001
    assert captured_search["dm_zero"] == 560.0
    assert captured_search["kwargs"] == {
        "dm_range": 8.0,
        "dm_step": 0.2,
        "snr_threshold": 6.0,
    }
    assert captured_plot["dm_best"] == 565.2
    assert captured_plot["dm_err"] == 0.3
    assert captured_plot["save_path"] == os.path.join(tmp_path, "burst3_dm.png")


def test_analyze_dm_exposes_search_failure_without_plotting(monkeypatch, tmp_path):
    def fail_search(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("synthetic search failure")

    def fail_if_plotted(*args, **kwargs):
        del args, kwargs
        raise AssertionError("failed DM searches must not produce a nominal plot")

    monkeypatch.setattr(burst_dm, "dm_phase_search", fail_search)
    monkeypatch.setattr(burst_dm, "plot_dm_search", fail_if_plotted)

    result = burst_dm.analyze_dm(
        np.ones((128, 4)),
        np.linspace(1000.0, 1300.0, 4),
        0.001,
        560.0,
        {"time_start": 50, "time_end": 70},
        str(tmp_path),
        4,
    )

    assert np.isnan(result["dm"])
    assert np.isnan(result["dm_err"])
    assert result["dm_status"] == "failed"
    assert result["dm_error_reason"] == "RuntimeError: synthetic search failure"

# fmt: on
