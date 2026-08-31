# fmt: off

import numpy as np

from after import burst_pol


def test_analyze_pol_uses_noncontiguous_boolean_time_gate(monkeypatch, tmp_path):
    captured = {}

    def fake_rm_synthesis(stokes_i, q, u, wave, **kwargs):
        del q, u, wave, kwargs
        captured["samples"] = stokes_i[:, 0].copy()
        return (
            np.array([-1.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0]),
            {
                "n_rm": 3,
                "rm_step": 1.0,
                "rmsf_fwhm": 8.0,
                "samples_per_fwhm": 8.0,
            },
        )

    monkeypatch.setattr(burst_pol, "rm_synthesis", fake_rm_synthesis)
    monkeypatch.setattr(
        burst_pol, "calc_pol_snr", lambda *args, **kwargs: (1.0, 10.0, 1250.0)
    )
    monkeypatch.setattr(burst_pol, "find_rm", lambda *args, **kwargs: (0.0, 1.0, 10.0))
    monkeypatch.setattr(burst_pol, "plot_rm_synthesis", lambda *args, **kwargs: None)
    monkeypatch.setattr(burst_pol, "correct_rm", lambda q, u, freq, rm: (q, u))

    def fake_pa(stokes_i, q, u, v, burst_mask, freq_mask, noise_mask):
        del q, u, v, burst_mask, freq_mask, noise_mask
        n = stokes_i.shape[0]
        return (
            np.arange(n, dtype=float),
            np.full(n, np.nan),
            np.full(n, np.nan),
            np.ones(n),
            np.ones(n),
            np.zeros(n),
            1.0,
        )

    monkeypatch.setattr(burst_pol, "calc_pa_profile", fake_pa)
    monkeypatch.setattr(
        burst_pol, "calc_pol_fractions", lambda *args, **kwargs: (0.0, 0.0, 0.0, 0.0)
    )
    monkeypatch.setattr(burst_pol, "plot_polarization", lambda *args, **kwargs: None)

    base               = np.arange(8, dtype=float)[:, None] * np.ones((1, 4))
    burst_mask         = np.zeros(8, dtype=bool)
    burst_mask[[2, 5]] = True
    burst_pol.analyze_pol(
        base,
        base,
        base,
        base,
        np.linspace(1000.0, 1500.0, 4),
        0.001,
        burst_mask,
        np.ones(4, dtype=bool),
        ~burst_mask,
        str(tmp_path),
        0,
    )

    np.testing.assert_array_equal(captured["samples"], np.array([2.0, 5.0]))

# fmt: on
