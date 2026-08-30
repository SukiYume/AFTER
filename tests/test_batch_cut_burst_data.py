# fmt: off

from batch_processing.batch_cut_burst_data import find_file_list


def test_find_file_list_excludes_separate_calibration_scan(tmp_path):
    names = [
        "FRB121102_cal-M01_0001.fits",
        "FRB121102_tracking-M01_0001.fits",
        "FRB121102_tracking-M01_0002.fits",
        "FRB121102_tracking-M02_0001.fits",
        "FRB121102_tracking-M01_0003_F_test.fits",
    ]
    for name in names:
        (tmp_path / name).touch()

    assert find_file_list(tmp_path, 1) == [
        "FRB121102_tracking-M01_0001.fits",
        "FRB121102_tracking-M01_0002.fits",
    ]

# fmt: on
