# fmt: off

import sys
from types import SimpleNamespace

import h5py
import numpy as np

from after import cut_burst_data


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
    monkeypatch, tmp_path, capsys
):
    first     = np.arange(10, dtype=np.uint8).reshape(10, 1, 1)
    truncated = np.arange(10, 14, dtype=np.uint8).reshape(4, 1, 1)

    def fake_read(path, header=True):
        del header
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
        start_sample = 8,
        total_length = 8,
    )

    np.testing.assert_array_equal(
        segment[:, 0, 0],
        np.array([8, 9, 10, 11, 12, 13, 0, 0], dtype=np.uint8),
    )
    assert "末尾补零 2 个采样" in capsys.readouterr().out

# fmt: on
