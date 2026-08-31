# fmt: off

import inspect
import json

import h5py
import numpy as np
import pytest

from after import burst_analysis, burst_pol


def _write_analysis_h5(path, npol):
    data          = np.zeros((4, 16, 8), dtype=np.float32)
    data[0, 6:10] = 5.0
    with h5py.File(path, "w") as h5:
        h5.create_dataset("data", data=data)
        h5.create_dataset("freq", data=np.linspace(1000.0, 1500.0, 8))
        attrs = {
            "bursts": json.dumps(
                [{"time_start": 6, "time_end": 10, "freq_start": 0, "freq_end": 8}]
            ),
            "time_reso": 0.001,
            "file_mjd": 60000.0,
            "dm": 100.0,
            "down_time": 1,
            "down_freq": 1,
        }
        if npol is not None:
            attrs["npol"] = npol
        h5.attrs.update(attrs)
    return path


def test_analysis_rm_interfaces_have_no_manual_grid_size():
    assert "n_rm" not in inspect.signature(burst_pol.rm_synthesis).parameters
    assert "n_rm" not in inspect.signature(burst_pol.analyze_pol).parameters
    assert "n_rm" not in inspect.signature(burst_analysis.analyze_one_file).parameters
    assert "n_rm" not in inspect.signature(burst_analysis.analyze_all).parameters


def test_select_strong_time_samples_keeps_only_main_peak_samples():
    rng      = np.random.default_rng(121102)
    stokes_i = rng.normal(0.0, 0.01, size=(64, 32))
    stokes_i[20] += 1.0
    stokes_i[21] += 2.0
    stokes_i[22] += 1.0
    stokes_i[30] += 0.35
    noise_mask        = np.ones(64, dtype=bool)
    noise_mask[10:40] = False

    selected, info = burst_analysis._select_strong_time_samples(
        stokes_i,
        np.ones(32, dtype=bool),
        noise_mask,
        {"time_start": 10, "time_end": 40},
        peak_fraction = 0.5,
        min_snr       = 5.0,
    )

    assert np.flatnonzero(selected).tolist() == [20, 21, 22]
    assert info["sample_count"] == 3
    assert info["peak_sample"] == 21
    assert info["peak_snr"] > 100


@pytest.mark.parametrize(
    (
        "npol",
        "expected_status",
        "expected_rm",
        "expected_rfi_calls",
        "expected_pol_calls",
    ),
    (
        pytest.param(2, "unavailable", np.nan, 1, 0, id="two-product"),
        pytest.param(4, "ok", 25.0, 4, 1, id="four-product"),
        pytest.param(None, "ok", 25.0, 4, 1, id="legacy-without-npol"),
    ),
)
def test_analysis_respects_available_products(
    monkeypatch,
    tmp_path,
    npol,
    expected_status,
    expected_rm,
    expected_rfi_calls,
    expected_pol_calls,
):
    h5_path   = _write_analysis_h5(tmp_path / "burst_cal.h5", npol)
    rfi_calls = []
    pol_calls = []

    def fake_cal_rfi(plane, *args, **kwargs):
        del args, kwargs
        rfi_calls.append(plane.copy())
        return np.zeros(plane.shape[1], dtype=bool), np.zeros_like(plane, dtype=bool)

    def fake_select(stokes_i, *args, **kwargs):
        del args, kwargs
        selected    = np.zeros(stokes_i.shape[0], dtype=bool)
        selected[7] = True
        return selected, {"peak_snr": 10.0}

    def fake_analyze_pol(*args, **kwargs):
        del args, kwargs
        pol_calls.append(True)
        return {"rm": 25.0}, None

    monkeypatch.setattr(burst_analysis, "cal_rfi", fake_cal_rfi)
    monkeypatch.setattr(
        burst_analysis,
        "robust_channel_mask",
        lambda planes, *args, **kwargs: np.zeros(planes.shape[2], dtype=bool),
    )
    monkeypatch.setattr(burst_analysis, "plot_dynamic_spectrum", lambda *args: None)
    monkeypatch.setattr(
        burst_analysis, "calc_burst_properties", lambda *args, **kwargs: {"snr": 10.0}
    )
    monkeypatch.setattr(
        burst_analysis, "analyze_dm", lambda *args, **kwargs: {"dm": 100.0}
    )
    monkeypatch.setattr(burst_analysis, "_select_strong_time_samples", fake_select)
    monkeypatch.setattr(burst_analysis, "plot_rm_selection", lambda *args: None)
    monkeypatch.setattr(burst_analysis, "analyze_pol", fake_analyze_pol)

    rows = burst_analysis.analyze_one_file(h5_path, tmp_path / "analysis")

    assert len(rows) == 1
    assert rows[0]["snr"] == 10.0
    assert rows[0]["dm"] == 100.0
    assert rows[0]["pol_status"] == expected_status
    if np.isnan(expected_rm):
        assert np.isnan(rows[0]["rm"])
        assert rows[0]["pol_error_reason"] == (
            "输入观测只有 AA/BB 两路，RM 和偏振不可用"
        )
    else:
        assert rows[0]["rm"] == expected_rm
        assert rows[0]["pol_error_reason"] == ""
    assert len(rfi_calls) == expected_rfi_calls
    assert len(pol_calls) == expected_pol_calls

# fmt: on
