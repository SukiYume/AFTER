import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from after import burst_properties, calibration, cut_burst_data
from batch_processing import batch_calibration


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


def test_extract_segment_zero_pads_a_truncated_final_fits(
        monkeypatch, tmp_path, capsys):
    first = np.arange(10, dtype=np.uint8).reshape(10, 1, 1)
    truncated = np.arange(10, 14, dtype=np.uint8).reshape(4, 1, 1)

    def fake_read(path, header=True):
        values = first if str(path).endswith("0001.fits") else truncated
        return {"DATA": values}, {
            "NAXIS2": values.shape[0],
            "NSBLK": 1,
            "NPOL": 1,
            "NCHAN": 1,
        }

    monkeypatch.setitem(
        sys.modules,
        "fitsio",
        SimpleNamespace(read=fake_read),
    )
    segment = cut_burst_data.extract_segment(
        str(tmp_path),
        ["M01_0001.fits", "M01_0002.fits"],
        {"file_nsamp": 10, "npol": 1, "nchan": 1},
        start_sample=8,
        total_length=8,
    )

    np.testing.assert_array_equal(
        segment[:, 0, 0],
        np.array([8, 9, 10, 11, 12, 13, 0, 0], dtype=np.uint8),
    )
    assert "末尾补零 2 个采样" in capsys.readouterr().out


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
        lambda data, *args: np.zeros(
            (4, data.shape[0], data.shape[2]), dtype=np.float32
        ),
    )
    monkeypatch.setattr(
        calibration,
        "cal_rfi",
        lambda data, *args, **kwargs: (
            np.zeros(data.shape[1], dtype=bool),
            np.zeros(data.shape, dtype=bool),
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
        time_crop_samples=8,
    )

    assert not Path(f"{output_path}.tmp").exists()
    with h5py.File(output_path, "r") as h5:
        assert {"data", "freq", "rfi_mask", "gain", "gain_err"} <= set(h5)
        assert h5.attrs["calibration_beam"] == 2
        assert h5.attrs["calibration_fits"] == "M02_0001.fits"
        assert h5.attrs["calibration_npz"] == "tcal.npz"
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


def test_calibration_targets_effective_resolution_then_crops(
        monkeypatch, tmp_path):
    """验证新路径的顺序是“完整数据定标 → 下采样 → 中心裁剪”。"""
    input_path = tmp_path / "FRBTEST-20260728-M02-0001-000000000.h5"
    output_dir = tmp_path / "cal"
    with h5py.File(input_path, "w") as h5:
        h5.create_dataset("data", data=np.zeros((32, 2, 8), dtype=np.float32))
        h5.create_dataset("freq", data=np.linspace(1000.0, 1500.0, 8))
        h5.attrs.update({
            "file_mjd": 60000.0,
            "obs_start_mjd": 60000.0,
            "start_sample": 0,
            "toa_sec": 0.016,
            "time_reso": 0.001,
            "nchan": 8,
            "dm": 100.0,
        })

    calibrated_input_shapes = []

    def fake_calibrate(data, *args):
        calibrated_input_shapes.append(data.shape)
        time_index = np.arange(data.shape[0], dtype=np.float32)
        return np.broadcast_to(
            time_index[np.newaxis, :, np.newaxis],
            (4, data.shape[0], data.shape[2]),
        ).copy()

    monkeypatch.setattr(calibration, "get_za", lambda *args: 0.0)
    monkeypatch.setattr(
        calibration,
        "get_gain",
        lambda *args: (np.ones(8), np.full(8, 0.1)),
    )
    monkeypatch.setattr(calibration, "calibrate_to_iquv", fake_calibrate)
    monkeypatch.setattr(
        calibration,
        "cal_rfi",
        lambda data, *args, **kwargs: (
            np.zeros(data.shape[1], dtype=bool),
            np.zeros(data.shape, dtype=bool),
        ),
    )
    monkeypatch.setattr(calibration.plt, "savefig", lambda *args, **kwargs: None)

    calibration.process_one_burst(
        str(input_path),
        str(output_dir),
        np.zeros((2, 8)),
        np.ones((2, 8)),
        "00h00m00s",
        "00d00m00s",
        2,
        "/cal/M02_0001.fits",
        "/cal/tcal.npz",
        down_freq=2,
        target_time_reso=0.004,
        output_time_samples=4,
    )

    # 定标函数看到的仍是完整 32 个原始时间点。
    assert calibrated_input_shapes == [(32, 2, 8)]

    output_path = output_dir / f"{input_path.stem}_cal.h5"
    with h5py.File(output_path, "r") as h5:
        assert h5["data"].shape == (4, 4, 4)
        # dt=4 后共 8 点，再中心裁取第 2:6 点。
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


def test_calibration_rejects_non_integer_target_time_ratio(tmp_path):
    input_path = tmp_path / "FRBTEST.h5"
    output_dir = tmp_path / "cal"
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

    with pytest.raises(ValueError, match="整数倍"):
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
            target_time_reso=0.0025,
        )


@pytest.mark.parametrize(
    ("downsampling", "error"),
    [
        ({"down_time": 17}, "down_time=17 超过当前时间长度 16"),
        ({"down_freq": 5}, "down_freq=5 超过当前频率通道数 4"),
    ],
)
def test_calibration_rejects_downsampling_larger_than_data(
        tmp_path, downsampling, error):
    input_path = tmp_path / "FRBTEST.h5"
    output_dir = tmp_path / "cal"
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

    with pytest.raises(ValueError, match=error):
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
            **downsampling,
        )


def test_batch_calibration_falls_back_to_same_date_sibling(monkeypatch):
    """当当前分段的 *_0001.fits 没有有效噪声管时，改用同日候选。"""

    class FakeFits:
        def __enter__(self):
            return [None, SimpleNamespace(header={"NCHAN": 8})]

        def __exit__(self, *args):
            return False

    attempted = []
    processed_with = []

    monkeypatch.setattr(batch_calibration.fits, "open", lambda *args: FakeFits())

    def fake_fold(path, **kwargs):
        attempted.append(path)
        if path == "/cut/20200806_2/bad-M01_0001.fits":
            raise ValueError("噪声管跳变不为正")
        return np.ones((4, 8), dtype=np.float32)

    monkeypatch.setattr(batch_calibration, "fold_noise_cal", fake_fold)
    monkeypatch.setattr(
        batch_calibration,
        "load_t_cal",
        lambda *args: np.ones((2, 8), dtype=np.float32),
    )
    monkeypatch.setattr(
        batch_calibration,
        "process_one_burst",
        lambda *args, **kwargs: processed_with.append(args[7]),
    )

    group = {
        "source": "FRB190520",
        "date": "20200806_2",
        "output_dir": "/cal/20200806_2",
        "h5_list": ["/cut/burst.h5"],
        "cal_fits_path": "/cut/20200806_2/bad-M01_0001.fits",
        "cal_fits_candidates": [
            "/cut/20200806_2/bad-M01_0001.fits",
            "/cut/20200806_1/good-M01_0001.fits",
        ],
        "cal_npz": "/cal/tcal.npz",
        "ra": "00h00m00s",
        "dec": "00d00m00s",
        "beam": 1,
        "down_time": None,
        "down_freq": None,
        "rfi_fft": True,
        "time_crop_samples": None,
        "target_time_reso": 0.000786432,
        "output_time_samples": 512,
    }

    result = batch_calibration.process_group(group)

    assert attempted == group["cal_fits_candidates"]
    assert processed_with == ["/cut/20200806_1/good-M01_0001.fits"]
    assert result == ("FRB190520", "20200806_2", 1, 1)
