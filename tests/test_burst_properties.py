# fmt: off

import numpy as np

from after import burst_properties


def test_gaussian_fit_accepts_descending_axis():
    x = np.linspace(1000.0, 1500.0, 501)
    y = 0.2 + 5.0 * np.exp(-0.5 * ((x - 1250.0) / 42.466) ** 2)

    ascending  = burst_properties._fit_gaussian(x, y)
    descending = burst_properties._fit_gaussian(x[::-1], y[::-1])

    np.testing.assert_allclose(descending, ascending, rtol=1e-6, atol=1e-6)


def test_bootstrap_is_reproducible_with_seed():
    source_rng = np.random.default_rng(7)
    data       = source_rng.normal(0.0, 1.0, (128, 32))
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
        "noise_mask": np.r_[
            np.ones(56, dtype=bool), np.zeros(16, dtype=bool), np.ones(56, dtype=bool)
        ],
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

# fmt: on
