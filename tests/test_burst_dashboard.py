# -*- coding: utf-8 -*-
# fmt: off

"""用合成 catalog 验证 dashboard 的用户可见行为。"""

from argparse import Namespace

import numpy as np
import pandas as pd

from after import burst_dashboard as bd


def _make_csv(path, n=40, reliable_rm=True, seed=1):
    """造一份列齐全的合成 burst CSV，reliable_rm 控制 RM 显著性高/低。"""
    rng     = np.random.default_rng(seed)
    toa     = np.sort(59800.0 + np.cumsum(rng.exponential(0.0008, n)))
    fluence = rng.lognormal(0.4, 0.8, n)
    sig     = rng.uniform(6, 9, n) if reliable_rm else rng.uniform(0, 3, n)
    pd.DataFrame(
        {
            "file_name": [f"FRB121102-20260626-M01-{i // 6:02d}.h5" for i in range(n)],
            "burst_idx": [i % 6 for i in range(n)],
            "toa_mjd": toa,
            "snr": rng.uniform(4, 38, n),
            "flux_peak": rng.uniform(0.2, 5, n),
            "fluence": fluence,
            "width": rng.uniform(0.5, 12, n),
            "freq_low": rng.uniform(1000, 1100, n),
            "freq_high": rng.uniform(1300, 1450, n),
            "bandwidth": rng.uniform(150, 400, n),
            "dm": rng.normal(565, 4, n),
            "dm_err": rng.uniform(1, 9, n),
            "rm": rng.normal(1e5, 500, n),
            "rm_err": rng.uniform(20, 80, n),
            "rm_significance": sig,
            "linear_frac": rng.uniform(20, 95, n),
            "linear_frac_err": rng.uniform(2, 8, n),
            "circular_frac": rng.uniform(-20, 30, n),
            "circular_frac_err": rng.uniform(2, 8, n),
            "center_freq": rng.uniform(1150, 1350, n),
        }
    ).to_csv(path, index=False)


def _args():
    return Namespace(
        snr_threshold             = 5.0,
        dm_err_threshold          = 5.0,
        reference_dm              = 565.0,
        rm_significance_threshold = 5.0,
        top_n                     = 10,
    )


def _build(tmp, n, reliable_rm):
    csv = tmp / "burst_results.csv"
    out = tmp / "burst_dashboard.html"
    _make_csv(csv, n=n, reliable_rm=reliable_rm)
    df   = bd.load_results(csv, 5.0)
    meta = bd.infer_metadata(df, csv)
    return df, bd.build_html(df, csv, out, tmp, meta, _args())


def test_dashboard_renders_reliable_rm_catalog(tmp_path):
    df, html = _build(tmp_path, n=40, reliable_rm=True)

    assert bool(df["rm_reliable"].any())
    for marker in (
        "BURST CATALOG",
        "SIGNAL PROPERTIES",
        "累积通量分布",
        "累积计数",
        "偏振与 RM",
        "<!doctype html>",
    ):
        assert marker in html, f"缺少标记: {marker}"
    expected_fluence_bw = bd.fmt_value(
        bd.fluence_bandwidth_jy_ms_ghz(df), 2, " Jy ms GHz"
    )
    assert "FLUENCE × BW" in html
    assert expected_fluence_bw in html
    assert str(tmp_path.resolve()) not in html
    assert "fonts.googleapis.com" in html
    assert "ENERGY FLUENCE" not in html
    assert 'class="print-hide"' not in html
    assert "打印仅显示" not in html


def test_dashboard_omits_polarization_without_reliable_rm(tmp_path):
    _, html = _build(tmp_path, n=40, reliable_rm=False)

    assert "偏振与 RM" not in html
    assert "累积通量分布" in html


def test_dashboard_marks_rows_above_print_limit(tmp_path):
    _, html = _build(tmp_path, n=260, reliable_rm=True)

    assert 'class="print-hide"' in html
    assert "打印仅显示" in html

# fmt: on
