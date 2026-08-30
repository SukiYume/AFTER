# fmt: off

import numpy as np
import pytest

from batch_processing import cut_burst_fits


def test_read_burst_txt_resolves_manifest_inside_catalog_directory(tmp_path):
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    manifest_path = manifest_dir / "raw_files.txt"
    manifest_path.write_text("raw-M01_0001.fits\n", encoding="utf-8")
    catalog_path = tmp_path / "FRBTEST_legacy_fits_rebuild.txt"
    catalog_path.write_text(
        "project\tname\tdate\tbeam\tdm\ttime\tstart_sample\t"
        "output_file\tsegment_length\toutput_subdir\ttemplate_file\t"
        "raw_file_manifest\trebuild_status\n"
        "P001\tRAWTEST\t20260728\t1\t100\t1.5\t-4\tburst.fits\t8\t"
        "20260728_1\traw-M01_0001.fits\tmanifests/raw_files.txt\tready_exact\n",
        encoding="utf-8",
    )
    raw_root = tmp_path / "raw"

    rows = cut_burst_fits.read_burst_txt(catalog_path, raw_root)

    assert len(rows) == 1
    assert rows[0].raw_dir == raw_root / "P001" / "RAWTEST" / "20260728"
    assert rows[0].raw_file_manifest == manifest_path.resolve()
    assert rows[0].start_sample == -4
    assert rows[0].rebuild_status == "ready_exact"


def test_read_burst_txt_rejects_manifest_outside_catalog_directory(tmp_path):
    catalog_path = tmp_path / "FRBTEST_legacy_fits_rebuild.txt"
    catalog_path.write_text(
        "project\tname\tdate\tbeam\tdm\ttime\traw_file_manifest\n"
        "P001\tRAWTEST\t20260728\t1\t100\t1.5\t../outside.txt\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="raw_file_manifest"):
        cut_burst_fits.read_burst_txt(catalog_path, tmp_path / "raw")


def test_find_raw_fits_honors_frozen_manifest_order(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for name in ("raw-M01_0001.fits", "raw-M01_0002.fits"):
        (raw_dir / name).touch()
    manifest_path = tmp_path / "raw_files.txt"
    manifest_path.write_text(
        "raw-M01_0002.fits\nraw-M01_0001.fits\n",
        encoding="utf-8",
    )

    files = cut_burst_fits.find_raw_fits(raw_dir, 1, manifest_path)

    assert files == ["raw-M01_0002.fits", "raw-M01_0001.fits"]


def test_cut_group_writes_negative_start_with_n_token(monkeypatch, tmp_path):
    row = cut_burst_fits.BurstRow(
        raw_dir        = tmp_path / "raw",
        project        = "P001",
        raw_name       = "RAWTEST",
        date           = "20260728",
        beam           = 1,
        dm             = 100.0,
        toa_sec        = 0.0,
        start_sample   = -4,
        segment_length = 8,
    )
    written = []
    monkeypatch.setattr(
        cut_burst_fits,
        "find_raw_fits",
        lambda *_args: ["raw-M01_0001.fits"],
    )
    monkeypatch.setattr(
        cut_burst_fits,
        "read_obs_info",
        lambda *_args: {
            "file_nsamp": 100,
            "time_reso": 0.001,
            "freq": np.array([1200.0]),
        },
    )
    monkeypatch.setattr(
        cut_burst_fits,
        "calc_dispersion_shift",
        lambda *_args: (np.array([0]), 0),
    )
    monkeypatch.setattr(
        cut_burst_fits,
        "extract_segment",
        lambda *_args: np.zeros((8, 2, 1), dtype=np.uint8),
    )
    monkeypatch.setattr(
        cut_burst_fits,
        "dedisperse",
        lambda segment, *_args: segment,
    )
    monkeypatch.setattr(
        cut_burst_fits,
        "_write_cut_fits",
        lambda _template, output, _data, _overwrite: written.append(output),
    )

    cut_burst_fits.cut_group(
        [row],
        output_root     = tmp_path / "output",
        output_name     = "FRBTEST",
        segment_length  = 4096,
        overwrite       = False,
        copy_first_fits = False,
    )

    assert written[0].name == ("FRBTEST-20260728-M01-0001-n000000004.fits")

# fmt: on
