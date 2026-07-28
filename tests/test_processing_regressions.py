from pathlib import Path

import h5py
import numpy as np

from after import burst_properties, calibration, cut_burst_data


def test_gaussian_fit_accepts_descending_axis():
    x = np.linspace(1000.0, 1500.0, 501)
    y = 0.2 + 5.0 * np.exp(-0.5 * ((x - 1250.0) / 42.466) ** 2)

    ascending = burst_properties._fit_gaussian(x, y)
    descending = burst_properties._fit_gaussian(x[::-1], y[::-1])

    np.testing.assert_allclose(descending, ascending, rtol=1e-6, atol=1e-6)


def test_bootstrap_is_reproducible_with_seed():
    source_rng = np.random.default_rng(7)
    data = source_rng.normal(0.0, 1.0, (128, 32))
    data[60:68] += 8.0
    kwargs = {
        "stokes_I": data,
        "freq": np.linspace(1000.0, 1500.0, 32),
        "time_reso": 0.001,
        "file_mjd": 60000.0,
        "burst_region": {
            "time_start": 60,
            "time_end": 68,
            "freq_start": 0,
            "freq_end": 32,
        },
        "noise_mask": np.r_[np.ones(56, dtype=bool), np.zeros(16, dtype=bool),
                            np.ones(56, dtype=bool)],
        "rfi_mask": np.zeros_like(data, dtype=bool),
        "freq_index": np.ones(32, dtype=bool),
        "n_boot": 50,
    }

    first = burst_properties.calc_burst_properties(
        **kwargs, rng=np.random.default_rng(42)
    )
    second = burst_properties.calc_burst_properties(
        **kwargs, rng=np.random.default_rng(42)
    )

    assert first["flux_err"] == second["flux_err"]
    assert first["fluence_err"] == second["fluence_err"]


def test_cut_before_observation_start_keeps_logical_time(monkeypatch, tmp_path):
    info = {
        "time_reso": 1.0,
        "file_nsamp": 16,
        "npol": 2,
        "nchan": 4,
        "freq": np.linspace(1000.0, 1500.0, 4),
        "start_mjd": 60000.0,
    }
    monkeypatch.setattr(
        cut_burst_data,
        "extract_segment",
        lambda *args: np.zeros((8, 2, 4), dtype=np.uint8),
    )
    monkeypatch.setattr(
        cut_burst_data,
        "dedisperse",
        lambda segment, shifts, length: segment[:length],
    )

    cut_burst_data.cut_one_burst(
        "unused",
        str(tmp_path),
        ["M01_0001.fits"],
        info,
        100.0,
        2.0,
        np.zeros(4, dtype=np.int64),
        0,
        8,
        "FRBTEST",
        "20260728",
        1,
    )

    output = next(tmp_path.glob("*-n000000002.h5"))
    assert not list(tmp_path.glob("*.tmp"))
    with h5py.File(output, "r") as h5:
        assert h5.attrs["start_sample"] == -2
        assert np.isclose(
            h5.attrs["file_mjd"],
            60000.0 - 2.0 / 86400.0,
        )
        reconstructed_toa = h5.attrs["file_mjd"] + 4.0 / 86400.0
        assert np.isclose(
            reconstructed_toa,
            60000.0 + 2.0 / 86400.0,
        )


def test_calibration_rewrites_incomplete_output_with_provenance(
        monkeypatch, tmp_path):
    input_path = tmp_path / "FRBTEST-20260728-M02-0001-000000000.h5"
    output_dir = tmp_path / "cal"
    output_dir.mkdir()
    output_path = output_dir / f"{input_path.stem}_cal.h5"
    with h5py.File(input_path, "w") as h5:
        h5.create_dataset("data", data=np.zeros((16, 2, 4), dtype=np.float32))
        h5.create_dataset("freq", data=np.linspace(1000.0, 1500.0, 4))
        h5.attrs.update({
            "file_mjd": 60000.0,
            "obs_start_mjd": 60000.0,
            "start_sample": 0,
            "toa_sec": 0.008,
            "time_reso": 0.001,
            "nchan": 4,
            "dm": 100.0,
        })
    with h5py.File(output_path, "w") as h5:
        h5.create_dataset("data", data=np.zeros((1,), dtype=np.float32))

    monkeypatch.setattr(calibration, "get_za", lambda *args: 0.0)
    monkeypatch.setattr(
        calibration,
        "get_gain",
        lambda *args: (np.ones(4), np.full(4, 0.1)),
    )
    monkeypatch.setattr(
        calibration,
        "calibrate_to_iquv",
        lambda *args: np.zeros((4, 16, 4), dtype=np.float32),
    )
    monkeypatch.setattr(
        calibration,
        "cal_rfi",
        lambda *args, **kwargs: (
            np.zeros(4, dtype=bool),
            np.zeros((16, 4), dtype=bool),
        ),
    )
    monkeypatch.setattr(calibration.plt, "savefig", lambda *args, **kwargs: None)

    calibration.process_one_burst(
        str(input_path),
        str(output_dir),
        np.zeros((2, 4)),
        np.ones((2, 4)),
        "00h00m00s",
        "00d00m00s",
        2,
        "/cal/M02_0001.fits",
        "/cal/tcal.npz",
        down_time=1,
        down_freq=1,
    )

    assert not Path(f"{output_path}.tmp").exists()
    with h5py.File(output_path, "r") as h5:
        assert {"data", "freq", "rfi_mask", "gain", "gain_err"} <= set(h5)
        assert h5.attrs["calibration_beam"] == 2
        assert h5.attrs["calibration_fits"] == "M02_0001.fits"
        assert h5.attrs["calibration_npz"] == "tcal.npz"
