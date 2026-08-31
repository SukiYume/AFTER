# fmt: off

import numpy as np

from after.rfi import robust_channel_mask


def test_robust_channel_mask_finds_persistent_qu_rfi_and_grows_neighbors():
    rng   = np.random.default_rng(20260718)
    data  = rng.normal(0.0, 1.0, size=(4, 128, 64))
    phase = np.linspace(0.0, 16.0 * np.pi, 128)
    data[1, :, 20] += 20.0 * np.sin(phase)
    data[2, :, 45] += 18.0 * np.cos(phase * 0.7)

    mask = robust_channel_mask(
        data,
        np.ones(128, dtype=bool),
        sigma        = 6.0,
        local_window = 15,
        grow         = 1,
    )

    assert np.all(mask[19:22])
    assert np.all(mask[44:47])
    assert np.count_nonzero(mask) < 20

# fmt: on
