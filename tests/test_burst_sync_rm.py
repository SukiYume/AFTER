import json

import h5py
import numpy as np
import pandas as pd

from after import burst_sync_rm
from after.burst_pol import RM_GRID_OVERSAMPLE, build_automatic_rm_grid


def _write_synthetic_cal_h5(path, rm, pa_offset_deg, seed):
    rng = np.random.default_rng(seed)
    nsamp = 80
    nchan = 128
    freq = np.linspace(1200.0, 1500.0, nchan)
    wave2 = (burst_sync_rm.C_M_S / (freq * 1e6)) ** 2
    wave2 -= np.mean(wave2)

    data = rng.normal(0.0, 0.15, size=(4, nsamp, nchan))
    signal_times = np.array([36, 37, 38])
    data[0, signal_times, :] += 12.0
    pa_deg = np.array([0.0, 60.0, 120.0]) + pa_offset_deg
    for time_index, pa in zip(signal_times, pa_deg, strict=True):
        phase = 2.0 * (np.deg2rad(pa) + rm * wave2)
        polarization = 3.0 * np.exp(1j * phase)
        data[1, time_index, :] += polarization.real
        data[2, time_index, :] += polarization.imag

    region = {
        "time_start": 33,
        "time_end": 42,
        "freq_start": 0,
        "freq_end": nchan,
        "confidence": 1.0,
    }
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=data)
        handle.create_dataset("freq", data=freq)
        handle.create_dataset("rfi_channel", data=np.zeros(nchan, dtype=bool))
        handle.create_dataset(
            "burst_rfi_channel", data=np.zeros(nchan, dtype=bool)
        )
        # Deliberately mark every pixel. The joint-RM script must never read it.
        handle.create_dataset(
            "rfi_mask", data=np.ones((nsamp, nchan), dtype=bool)
        )
        handle.attrs["bursts"] = json.dumps([region])
        handle.attrs["time_reso"] = 0.001
        handle.attrs["down_time"] = 1
        handle.attrs["down_freq"] = 1


def _load_synthetic_components(cal_dir):
    components = []
    for path in sorted(cal_dir.glob("*_cal.h5")):
        loaded, warnings = burst_sync_rm.load_file_components(
            path,
            freq_min=None,
            freq_max=None,
            time_peak_fraction=0.5,
            min_time_snr=3.0,
            min_channels=32,
            stored_masks_only=True,
            rfi_fft=False,
            rfi_channel_sigma=6.0,
            rfi_channel_window=31,
            rfi_channel_grow=1,
        )
        assert not warnings
        components.extend(loaded)
    return components


def test_two_pa_independent_methods_recover_common_rm(tmp_path):
    cal_dir = tmp_path / "cal"
    cal_dir.mkdir()
    expected_rm = 1500.0
    _write_synthetic_cal_h5(
        cal_dir / "FRBTEST-20260726-M01-0001-000000001_cal.h5",
        expected_rm,
        0.0,
        1,
    )
    _write_synthetic_cal_h5(
        cal_dir / "FRBTEST-20260726-M01-0002-000000002_cal.h5",
        expected_rm,
        37.0,
        2,
    )

    bursts = _load_synthetic_components(cal_dir)
    assert len(bursts) == 2
    assert all(burst.n_time == 3 for burst in bursts)

    rm_grid = np.linspace(-3000.0, 3000.0, 1201)
    individual = {
        burst.component_id: burst_sync_rm.individual_rm_curves(
            burst, rm_grid, chunk_size=128
        )
        for burst in bursts
    }
    weights = burst_sync_rm.curve_weights(bursts, "equal", 4.0)
    combined = burst_sync_rm.combine_curves(bursts, individual, weights)

    assert set(combined) == {
        "time_pa_power",
        "linear_degree_stack",
    }
    assert all(
        set(curves) == {"linear_degree", "time_pa_power"}
        for curves in individual.values()
    )
    for method in burst_sync_rm.PRIMARY_METHODS:
        recovered = rm_grid[int(np.argmax(combined[method]))]
        assert abs(recovered - expected_rm) <= 10.0


def test_automatic_rm_grid_uses_rmsf_resolution():
    lambda2_wide = np.linspace(0.04, 0.09, 128)
    lambda2_narrow = np.linspace(0.05, 0.07, 64)
    rm_grid, info = build_automatic_rm_grid(
        -50_000.0,
        50_000.0,
        [lambda2_narrow, lambda2_wide],
    )

    expected_fwhm = 2.0 * np.sqrt(3.0) / 0.05
    assert rm_grid[0] == -50_000.0
    assert rm_grid[-1] == 50_000.0
    assert np.isclose(info["rmsf_fwhm"], expected_fwhm)
    assert info["rm_step"] <= expected_fwhm / RM_GRID_OVERSAMPLE
    assert info["samples_per_fwhm"] >= RM_GRID_OVERSAMPLE
    assert info["n_rm"] == rm_grid.size


def test_sync_cli_has_no_manual_rm_sampling_parameter():
    parser_args = burst_sync_rm.parse_args(
        [
            "--cal-dir",
            ".",
            "--output-dir",
            "output",
        ]
    )
    assert not hasattr(parser_args, "n_rm")
    assert not hasattr(parser_args, "rm_step")
    assert not hasattr(parser_args, "null_rm_step")


def test_component_ids_expand_collisions_and_distinguish_beams(tmp_path):
    first_path = tmp_path / "FRBTEST-20260718-M01-0211-027549018_cal.h5"
    second_path = tmp_path / "FRBTEST-20260718-M01-0211-027600287_cal.h5"
    third_path = tmp_path / "FRBTEST-20260718-M02-0211-027549018_cal.h5"
    common = {
        "burst_idx": 0,
        "peak_snr": 10.0,
        "time_indices": np.array([1]),
        "freq_mhz": np.array([1400.0]),
        "wave2_m2": np.array([0.04]),
        "p_on": np.array([[1.0j]]),
        "p_noise": np.array([[0.0j]]),
        "i_total": 1.0,
        "noise_variance_one_time": 1.0,
        "stored_cal_rfi_count": 0,
        "stored_burst_rfi_count": 0,
        "recalculated_rfi_count": 0,
        "robust_rfi_count": 0,
        "nonfinite_rfi_count": 0,
        "final_rfi_count": 0,
    }
    bursts = [
        burst_sync_rm.BurstRMData(
            component_id="0211b0",
            file_name=first_path.name,
            file_path=first_path,
            **common,
        ),
        burst_sync_rm.BurstRMData(
            component_id="0211b0",
            file_name=second_path.name,
            file_path=second_path,
            **common,
        ),
        burst_sync_rm.BurstRMData(
            component_id="0211b0",
            file_name=third_path.name,
            file_path=third_path,
            **common,
        ),
    ]

    burst_sync_rm.disambiguate_component_ids(bursts)

    assert [burst.component_id for burst in bursts] == [
        "FRBTEST-20260718-M01-0211-027549018b0",
        "FRBTEST-20260718-M01-0211-027600287b0",
        "FRBTEST-20260718-M02-0211-027549018b0",
    ]


def test_main_writes_reproducible_direct_h5_products(tmp_path):
    cal_dir = tmp_path / "cal"
    cal_dir.mkdir()
    _write_synthetic_cal_h5(
        cal_dir / "FRBTEST-20260726-M01-0001-000000001_cal.h5",
        1500.0,
        0.0,
        3,
    )
    _write_synthetic_cal_h5(
        cal_dir / "FRBTEST-20260726-M01-0002-000000002_cal.h5",
        1500.0,
        25.0,
        4,
    )
    output_dir = tmp_path / "output"

    result = burst_sync_rm.main(
        [
            "--cal-dir",
            str(cal_dir),
            "--output-dir",
            str(output_dir),
            "--rm-min",
            "-3000",
            "--rm-max",
            "3000",
            "--test-window",
            "full:-3000:3000",
            "--n-null",
            "20",
            "--null-pool-size",
            "32",
            "--min-time-snr",
            "3",
            "--min-peak-snr",
            "3",
            "--stored-masks-only",
        ]
    )
    assert result == 0

    summary = pd.read_csv(output_dir / "burst_sync_rm_summary.csv")
    assert set(summary["method"]) == set(burst_sync_rm.ALL_METHODS)
    for method in burst_sync_rm.PRIMARY_METHODS:
        row = summary[summary["method"] == method].iloc[0]
        assert abs(float(row["fine_grid_peak_rm"]) - 1500.0) <= 20.0
        assert int(row["n_null"]) == 20

    selected = pd.read_csv(output_dir / "selected_bursts.csv")
    assert len(selected) == 2
    assert (selected["n_time_samples"] == 3).all()

    manifest = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["pixel_mask"] == "not read or applied"
    assert set(manifest["methods"]) == set(burst_sync_rm.ALL_METHODS)
    assert set(manifest["methods"]) == {
        "time_pa_power",
        "linear_degree_stack",
    }
    with np.load(output_dir / "burst_sync_rm_curves.npz") as curves:
        assert all("burst_pa" not in name for name in curves.files)
    with np.load(output_dir / "offpulse_null_maxima.npz") as null_maxima:
        assert all("burst_pa" not in name for name in null_maxima.files)
    assert (output_dir / "burst_sync_rm.png").is_file()
    assert (output_dir / "burst_sync_rm_curves.npz").is_file()
