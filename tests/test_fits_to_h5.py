from pathlib import Path

import h5py
import numpy as np

from batch_processing import fits_to_h5


def test_parse_old_filename_accepts_split_date_directory():
    parsed = fits_to_h5.parse_old_filename(
        "FRB190520-20200730_2-M01-0042-005438001.fits"
    )

    assert parsed == {
        "frb_name": "FRB190520",
        "date": "20200730_2",
        "beam": 1,
        "fits_number": 42,
        "start_sample": 5438001,
    }


def test_parse_old_filename_accepts_negative_start_token():
    parsed = fits_to_h5.parse_old_filename(
        "FRBTEST-20260728-M01-0001-n000000123.fits"
    )

    assert parsed is not None
    assert parsed["start_sample"] == -123


def test_convert_preserves_negative_start_token(monkeypatch, tmp_path):
    fits_path = tmp_path / "FRBTEST-20260728-M01-0001-n000000123.fits"
    captured = {}
    monkeypatch.setattr(
        fits_to_h5,
        "load_fits_data",
        lambda _path: np.zeros((8, 2, 4), dtype=np.uint8),
    )

    def fake_save(_output_dir, filename, _data, _meta):
        captured["filename"] = filename

    monkeypatch.setattr(fits_to_h5, "save_to_h5", fake_save)

    result = fits_to_h5.convert_one_fits((
        fits_path,
        tmp_path / "output",
        [],
        {
            "time_reso": 0.001,
            "file_nsamp": 16,
            "start_mjd": 60000.0,
            "npol": 2,
            "nchan": 4,
            "freq": np.linspace(1000.0, 1500.0, 4),
        },
        False,
    ))

    assert result == "ok_no_catalog_match"
    assert captured["filename"] == (
        "FRBTEST-20260728-M01-0001-n000000123.h5"
    )


def test_is_date_dir_accepts_split_suffix(tmp_path):
    date_dir = tmp_path / "20200730_2"
    date_dir.mkdir()

    assert fits_to_h5.is_date_dir(date_dir)
    assert not fits_to_h5.is_date_dir(tmp_path / "20200730_extra")


def test_parse_burst_catalog_accepts_legacy_six_column_format(tmp_path):
    catalog_path = tmp_path / "FRB190520_Burst.txt"
    catalog_path.write_text(
        "name beam project dm date time\n"
        "FRB190520 1 RawFRB190520 1200 20200730_2 267.49\n",
        encoding="utf-8",
    )

    catalog = fits_to_h5.parse_burst_catalog(tmp_path, "FRB190")

    assert catalog == {
        "FRB190520": [
            {
                "base": "",
                "project": "RawFRB190520",
                "name": "FRB190520",
                "date": "20200730_2",
                "beam": 1,
                "dm": 1200.0,
                "time": 267.49,
            }
        ]
    }


def test_save_to_h5_uses_atomic_temporary_file(tmp_path):
    data = np.arange(24, dtype=np.uint8).reshape(3, 2, 4)
    meta = {
        "freq": np.linspace(1000.0, 1500.0, 4),
        "start_sample": 10,
        "file_mjd": 60000.0,
        "toa_sec": 1.0,
        "time_reso": 0.001,
        "npol": 2,
        "nchan": 4,
        "segment_length": 3,
        "obs_start_mjd": 60000.0,
        "beam": 1,
        "dm": 1200.0,
    }

    output = fits_to_h5.save_to_h5(tmp_path, "burst.h5", data, meta)

    assert output == Path(tmp_path, "burst.h5")
    assert not Path(tmp_path, "burst.h5.tmp").exists()
    with h5py.File(output, "r") as handle:
        np.testing.assert_array_equal(handle["data"][:], data)


def test_copy_cal_files_dry_run_does_not_create_output(tmp_path):
    date_path = tmp_path / "source"
    date_path.mkdir()
    calibration_path = date_path / "FRB190520_tracking-M01_0001.fits"
    calibration_path.touch()
    output_path = tmp_path / "output"

    selected = fits_to_h5.copy_cal_files(
        date_path,
        output_path,
        overwrite=False,
        dry_run=True,
    )

    assert selected == calibration_path
    assert not output_path.exists()
