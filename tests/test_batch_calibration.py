# fmt: off

from types import SimpleNamespace

import numpy as np

from batch_processing import batch_calibration


class _FakeFits:
    def __enter__(self):
        return [None, SimpleNamespace(header={"NCHAN": 8})]

    def __exit__(self, *args):
        del args
        return False


def _base_group(**updates):
    group = {
        "source": "FRB190520",
        "date": "20200806_2",
        "output_dir": "/cal/20200806_2",
        "h5_list": ["/cut/burst.h5"],
        "cal_fits_path": "/cut/20200806_2/good-M01_0001.fits",
        "cal_fits_candidates": ["/cut/20200806_2/good-M01_0001.fits"],
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
    group.update(updates)
    return group


def _patch_static_inputs(monkeypatch):
    monkeypatch.setattr(batch_calibration.fits, "open", lambda *args: _FakeFits())
    monkeypatch.setattr(
        batch_calibration,
        "load_t_cal",
        lambda *args: np.ones((2, 8), dtype=np.float32),
    )


def _capture_processed_periods(monkeypatch):
    processed = []

    def fake_process(
        h5_input_path,
        output_dir,
        noise_cal,
        t_cal,
        ra,
        dec,
        beam,
        cal_fits_path,
        cal_npz,
        **kwargs,
    ):
        del h5_input_path, output_dir, noise_cal, t_cal, ra, dec, beam, cal_npz
        processed.append((cal_fits_path, kwargs["noise_period_s"]))

    monkeypatch.setattr(batch_calibration, "process_one_burst", fake_process)
    return processed


def test_batch_calibration_falls_back_to_same_date_sibling(monkeypatch):
    """当前分段的噪声管无效时，只回退到同一天的同波束候选。"""
    attempted = []
    _patch_static_inputs(monkeypatch)
    processed_with = _capture_processed_periods(monkeypatch)

    def fake_fold(path, diagnostic_dir, noise_period_s):
        del diagnostic_dir
        attempted.append((path, noise_period_s))
        if path == "/cut/20200806_2/bad-M01_0001.fits":
            raise ValueError("噪声管跳变不为正")
        return np.ones((4, 8), dtype=np.float32)

    monkeypatch.setattr(batch_calibration, "fold_noise_cal", fake_fold)
    group = _base_group(
        cal_fits_path       = "/cut/20200806_2/bad-M01_0001.fits",
        cal_fits_candidates = [
            "/cut/20200806_2/bad-M01_0001.fits",
            "/cut/20200806_1/good-M01_0001.fits",
        ],
        noise_period_s      = 0.4,
    )

    result = batch_calibration.process_group(group)

    assert attempted == [
        ("/cut/20200806_2/bad-M01_0001.fits", 0.4),
        ("/cut/20200806_1/good-M01_0001.fits", 0.4),
    ]
    assert processed_with == [("/cut/20200806_1/good-M01_0001.fits", 0.4)]
    assert result == ("FRB190520", "20200806_2", 1, 1)


def test_batch_calibration_legacy_group_uses_default_noise_period(monkeypatch):
    folded_periods = []
    _patch_static_inputs(monkeypatch)
    processed_with = _capture_processed_periods(monkeypatch)

    def fake_fold(path, diagnostic_dir, noise_period_s):
        del path, diagnostic_dir
        folded_periods.append(noise_period_s)
        return np.ones((4, 8), dtype=np.float32)

    monkeypatch.setattr(batch_calibration, "fold_noise_cal", fake_fold)

    result = batch_calibration.process_group(_base_group())

    assert folded_periods == [0.2]
    assert processed_with == [("/cut/20200806_2/good-M01_0001.fits", 0.2)]
    assert result == ("FRB190520", "20200806_2", 1, 1)

# fmt: on
