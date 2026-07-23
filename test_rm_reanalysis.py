import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import burst_analysis
import burst_pol
from rfi_utils import robust_channel_mask


def test_robust_channel_mask_finds_persistent_qu_rfi_and_grows_neighbors():
    rng = np.random.default_rng(20260718)
    data = rng.normal(0.0, 1.0, size=(4, 128, 64))
    phase = np.linspace(0.0, 16.0 * np.pi, 128)
    data[1, :, 20] += 20.0 * np.sin(phase)
    data[2, :, 45] += 18.0 * np.cos(phase * 0.7)

    mask = robust_channel_mask(
        data, np.ones(128, dtype=bool), sigma=6.0,
        local_window=15, grow=1)

    assert np.all(mask[19:22])
    assert np.all(mask[44:47])
    assert np.count_nonzero(mask) < 20


def test_select_strong_time_samples_keeps_only_main_peak_samples():
    rng = np.random.default_rng(121102)
    stokes_i = rng.normal(0.0, 0.01, size=(64, 32))
    stokes_i[20] += 1.0
    stokes_i[21] += 2.0
    stokes_i[22] += 1.0
    stokes_i[30] += 0.35
    noise_mask = np.ones(64, dtype=bool)
    noise_mask[10:40] = False

    selected, info = burst_analysis._select_strong_time_samples(
        stokes_i, np.ones(32, dtype=bool), noise_mask,
        {'time_start': 10, 'time_end': 40},
        peak_fraction=0.5, min_snr=5.0)

    assert np.flatnonzero(selected).tolist() == [20, 21, 22]
    assert info['sample_count'] == 3
    assert info['peak_sample'] == 21
    assert info['peak_snr'] > 100


def test_analyze_pol_uses_noncontiguous_boolean_time_gate(monkeypatch, tmp_path):
    captured = {}

    def fake_rm_synthesis(I, Q, U, wave, **kwargs):
        del Q, U, wave, kwargs
        captured['samples'] = I[:, 0].copy()
        return np.array([-1.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])

    monkeypatch.setattr(burst_pol, 'rm_synthesis', fake_rm_synthesis)
    monkeypatch.setattr(
        burst_pol, 'calc_pol_snr',
        lambda *args, **kwargs: (1.0, 10.0, 1250.0))
    monkeypatch.setattr(
        burst_pol, 'find_rm',
        lambda *args, **kwargs: (0.0, 1.0, 10.0))
    monkeypatch.setattr(burst_pol, 'plot_rm_synthesis', lambda *args, **kwargs: None)
    monkeypatch.setattr(burst_pol, 'correct_rm', lambda Q, U, freq, rm: (Q, U))

    def fake_pa(I, Q, U, V, burst_mask, freq_mask, noise_mask):
        del Q, U, V, freq_mask, noise_mask
        n = I.shape[0]
        return (np.arange(n, dtype=float), np.full(n, np.nan),
                np.full(n, np.nan), np.ones(n), np.ones(n),
                np.zeros(n), 1.0)

    monkeypatch.setattr(burst_pol, 'calc_pa_profile', fake_pa)
    monkeypatch.setattr(
        burst_pol, 'calc_pol_fractions',
        lambda *args, **kwargs: (0.0, 0.0, 0.0, 0.0))
    monkeypatch.setattr(burst_pol, 'plot_polarization', lambda *args, **kwargs: None)

    base = np.arange(8, dtype=float)[:, None] * np.ones((1, 4))
    burst_mask = np.zeros(8, dtype=bool)
    burst_mask[[2, 5]] = True
    burst_pol.analyze_pol(
        base, base, base, base, np.linspace(1000.0, 1500.0, 4),
        0.001, burst_mask, np.ones(4, dtype=bool),
        ~burst_mask, str(tmp_path), 0, n_rm=3)

    np.testing.assert_array_equal(captured['samples'], np.array([2.0, 5.0]))
