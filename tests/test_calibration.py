# fmt: off

from pathlib import Path

import h5py
import numpy as np
import pytest

from after import calibration


def _write_cut_h5(
    path,
    nsamp=16,
    npol=2,
    nchan=4,
    time_reso=0.001,
    include_npol=True,
):
    with h5py.File(path, "w") as h5:
        h5.create_dataset("data", data=np.zeros((nsamp, npol, nchan), dtype=np.float32))
        h5.create_dataset("freq", data=np.linspace(1000.0, 1500.0, nchan))
        attrs = {
            "file_mjd": 60000.0,
            "obs_start_mjd": 60000.0,
            "start_sample": 0,
            "toa_sec": nsamp * time_reso / 2,
            "time_reso": time_reso,
            "nchan": nchan,
            "dm": 100.0,
        }
        if include_npol:
            attrs["npol"] = npol
        h5.attrs.update(attrs)
    return path


def _stub_calibration_dependencies(monkeypatch, nchan, calibrator=None):
    if calibrator is None:
        def calibrator(data, *args):
            del args
            return np.zeros((4, data.shape[0], data.shape[2]), dtype=np.float32)

    monkeypatch.setattr(calibration, "get_za", lambda *args: 0.0)
    monkeypatch.setattr(
        calibration,
        "get_gain",
        lambda *args: (np.ones(nchan), np.full(nchan, 0.1)),
    )
    monkeypatch.setattr(calibration, "calibrate_to_iquv", calibrator)
    monkeypatch.setattr(
        calibration,
        "cal_rfi",
        lambda data, *args, **kwargs: (
            np.zeros(data.shape[1], dtype=bool),
            np.zeros(data.shape, dtype=bool),
        ),
    )
    monkeypatch.setattr(calibration.plt, "savefig", lambda *args, **kwargs: None)


def _process_burst(input_path, output_dir, nchan, **kwargs):
    calibration.process_one_burst(
        str(input_path),
        str(output_dir),
        np.zeros((2, nchan)),
        np.ones((2, nchan)),
        "00h00m00s",
        "00d00m00s",
        2,
        "/cal/M02_0001.fits",
        "/cal/tcal.npz",
        **kwargs,
    )


def test_calibrate_to_iquv_preserves_four_product_transform():
    data      = np.asarray(
        [[[20.0, 24.0], [40.0, 48.0], [1.0, 2.0], [0.5, 1.0]]]
    )
    noise_cal = np.asarray(
        [[2.0, 2.0], [4.0, 4.0], [3.0, 3.0], [0.0, 0.0]]
    )
    t_cal     = np.asarray([[10.0, 10.0], [20.0, 20.0]])
    gain      = np.asarray([5.0, 5.0])

    result = calibration.calibrate_to_iquv(data, noise_cal, t_cal, gain)

    expected = np.asarray(
        [
            [[30.0, 36.0]],
            [[10.0, 12.0]],
            [[1.0, 2.0]],
            [[0.5, 1.0]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(result, expected)


def test_calibrate_to_iquv_two_product_keeps_i_and_compatibility_planes():
    data      = np.asarray([[[20.0, 24.0], [40.0, 48.0]]])
    noise_cal = np.asarray([[2.0, 2.0], [4.0, 4.0]])
    t_cal     = np.asarray([[10.0, 10.0], [20.0, 20.0]])
    gain      = np.asarray([5.0, 5.0])

    result = calibration.calibrate_to_iquv(data, noise_cal, t_cal, gain)

    expected = np.asarray(
        [
            [[30.0, 36.0]],
            [[10.0, 12.0]],
            [[0.0, 0.0]],
            [[0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(result, expected)


def test_calibrate_to_iquv_rejects_unsupported_product_count():
    with pytest.raises(ValueError, match="npol 必须为 2 或 4"):
        calibration.calibrate_to_iquv(
            np.zeros((1, 3, 2)),
            np.zeros((3, 2)),
            np.ones((2, 2)),
            np.ones(2),
        )


def test_calibration_rewrites_incomplete_output_with_provenance(monkeypatch, tmp_path):
    input_path = _write_cut_h5(
        tmp_path / "FRBTEST-20260728-M02-0001-000000000.h5",
        include_npol = False,
    )
    output_dir = tmp_path / "cal"
    output_dir.mkdir()
    output_path = output_dir / f"{input_path.stem}_cal.h5"
    with h5py.File(output_path, "w") as h5:
        h5.create_dataset("data", data=np.zeros((1,), dtype=np.float32))

    _stub_calibration_dependencies(monkeypatch, nchan=4)
    _process_burst(
        input_path,
        output_dir,
        nchan = 4,
        down_time         = 1,
        down_freq         = 1,
        time_crop_samples = 8,
    )

    assert not Path(f"{output_path}.tmp").exists()
    with h5py.File(output_path, "r") as h5:
        assert {"data", "freq", "rfi_mask", "gain", "gain_err"} <= set(h5)
        assert h5.attrs["calibration_beam"] == 2
        assert h5.attrs["calibration_fits"] == "M02_0001.fits"
        assert h5.attrs["calibration_npz"] == "tcal.npz"
        assert h5.attrs["noise_period_s"] == 0.2
        assert h5.attrs["npol"] == 2
        assert h5["data"].shape == (4, 8, 4)
        assert h5.attrs["nsamp"] == 8
        assert h5.attrs["nsamp_raw"] == 16
        assert h5.attrs["time_crop_start_raw"] == 4
        assert h5.attrs["time_crop_samples"] == 8
        assert h5.attrs["source_start_sample"] == 0
        assert h5.attrs["start_sample"] == 4
        assert h5.attrs["source_file_mjd"] == pytest.approx(60000.0)
        assert h5.attrs["file_mjd"] == pytest.approx(60000.0 + 0.004 / 86400.0)
        assert h5.attrs["down_time"] == 1
        assert h5.attrs["plot_down_time"] == 1
        assert h5.attrs["plot_down_freq"] == 1


def test_calibration_targets_effective_resolution_then_crops(monkeypatch, tmp_path):
    input_path = _write_cut_h5(
        tmp_path / "FRBTEST-20260728-M02-0001-000000000.h5",
        nsamp = 32,
        nchan = 8,
    )
    output_dir              = tmp_path / "cal"
    calibrated_input_shapes = []

    def fake_calibrate(data, *args):
        calibrated_input_shapes.append(data.shape)
        time_index = np.arange(data.shape[0], dtype=np.float32)
        return np.broadcast_to(
            time_index[np.newaxis, :, np.newaxis],
            (4, data.shape[0], data.shape[2]),
        ).copy()

    _stub_calibration_dependencies(monkeypatch, nchan=8, calibrator=fake_calibrate)
    _process_burst(
        input_path,
        output_dir,
        nchan = 8,
        down_freq           = 2,
        target_time_reso    = 0.004,
        output_time_samples = 4,
    )

    assert calibrated_input_shapes == [(32, 2, 8)]

    output_path = output_dir / f"{input_path.stem}_cal.h5"
    with h5py.File(output_path, "r") as h5:
        assert h5["data"].shape == (4, 4, 4)
        np.testing.assert_allclose(
            h5["data"][0, :, 0],
            np.array([9.5, 13.5, 17.5, 21.5], dtype=np.float32),
        )
        assert h5.attrs["down_time"] == 4
        assert h5.attrs["down_freq"] == 2
        assert h5.attrs["time_reso"] == pytest.approx(0.004)
        assert h5.attrs["target_time_reso"] == pytest.approx(0.004)
        assert h5.attrs["nsamp_raw"] == 32
        assert h5.attrs["nsamp_before_output_crop"] == 8
        assert h5.attrs["output_time_samples"] == 4
        assert h5.attrs["time_crop_stage"] == "post_downsample"
        assert h5.attrs["time_crop_start_downsampled"] == 2
        assert h5.attrs["time_crop_start_raw"] == 8
        assert h5.attrs["source_start_sample"] == 0
        assert h5.attrs["start_sample"] == 8
        assert h5.attrs["source_file_mjd"] == pytest.approx(60000.0)
        assert h5.attrs["file_mjd"] == pytest.approx(60000.0 + 0.008 / 86400.0)


def test_calibration_reuses_matching_output_and_rewrites_changed_period(
    monkeypatch,
    tmp_path,
):
    input_path        = _write_cut_h5(tmp_path / "FRBTEST.h5")
    output_dir        = tmp_path / "cal"
    calibrated_shapes = []

    def fake_calibrate(data, *args):
        calibrated_shapes.append(data.shape)
        return np.zeros((4, data.shape[0], data.shape[2]), dtype=np.float32)

    _stub_calibration_dependencies(monkeypatch, nchan=4, calibrator=fake_calibrate)

    _process_burst(input_path, output_dir, nchan=4, noise_period_s=0.2)
    _process_burst(input_path, output_dir, nchan=4, noise_period_s=0.2)
    assert calibrated_shapes == [(16, 2, 4)]

    _process_burst(input_path, output_dir, nchan=4, noise_period_s=0.4)
    assert calibrated_shapes == [(16, 2, 4), (16, 2, 4)]
    with h5py.File(output_dir / "FRBTEST_cal.h5", "r") as h5:
        assert h5.attrs["noise_period_s"] == 0.4


def test_calibration_rejects_non_integer_target_time_ratio(tmp_path):
    input_path = _write_cut_h5(tmp_path / "FRBTEST.h5")

    with pytest.raises(ValueError, match="整数倍"):
        _process_burst(
            input_path,
            tmp_path / "cal",
            nchan = 4,
            target_time_reso = 0.0025,
        )


@pytest.mark.parametrize(
    ("downsampling", "error"),
    (
        pytest.param(
            {"down_time": 17},
            "down_time=17 超过当前时间长度 16",
            id="time",
        ),
        pytest.param(
            {"down_freq": 5},
            "down_freq=5 超过当前频率通道数 4",
            id="frequency",
        ),
    ),
)
def test_calibration_rejects_downsampling_larger_than_data(
    tmp_path,
    downsampling,
    error,
):
    input_path = _write_cut_h5(tmp_path / "FRBTEST.h5")

    with pytest.raises(ValueError, match=error):
        _process_burst(
            input_path,
            tmp_path / "cal",
            nchan = 4,
            **downsampling,
        )

# fmt: on
