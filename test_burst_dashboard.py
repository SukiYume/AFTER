# -*- coding: utf-8 -*-
"""burst_dashboard 的最小冒烟测试：用合成数据跑通整条流水线并校验关键产物。

直接运行即可（无需 pytest）：
    python test_burst_dashboard.py
也兼容 pytest：
    pytest test_burst_dashboard.py
"""

from argparse import Namespace
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

import burst_dashboard as bd


def _make_csv(path, n=40, reliable_rm=True, seed=1):
    """造一份列齐全的合成 burst CSV，reliable_rm 控制 RM 显著性高/低。"""
    rng = np.random.default_rng(seed)
    toa = np.sort(59800.0 + np.cumsum(rng.exponential(0.0008, n)))
    fluence = rng.lognormal(0.4, 0.8, n)
    sig = rng.uniform(6, 9, n) if reliable_rm else rng.uniform(0, 3, n)
    pd.DataFrame({
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
    }).to_csv(path, index=False)


def _args():
    return Namespace(
        snr_threshold=5.0, dm_err_threshold=5.0, reference_dm=565.0,
        rm_significance_threshold=5.0, top_n=10,
    )


def _build(tmp, n, reliable_rm):
    csv = tmp / "burst_results.csv"
    out = tmp / "burst_dashboard.html"
    _make_csv(csv, n=n, reliable_rm=reliable_rm)
    df = bd.load_results(csv, 5.0)
    meta = bd.infer_metadata(df, csv)
    return df, bd.build_html(df, csv, out, tmp, meta, _args())


def test_pipeline():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)

        # 1) 有可靠 RM：应包含偏振面板、两张累积图、目录。
        df, html = _build(tmp, n=40, reliable_rm=True)
        assert bool(df["rm_reliable"].any())
        for marker in ("BURST CATALOG", "SIGNAL PROPERTIES", "累积通量分布", "累积计数",
                       "偏振与 RM", "<!doctype html>"):
            assert marker in html, f"缺少标记: {marker}"
        expected_fluence_bw = bd.fmt_value(
            bd.fluence_bandwidth_jy_ms_ghz(df), 2, " Jy ms GHz"
        )
        assert "FLUENCE × BW" in html
        assert expected_fluence_bw in html
        assert "ENERGY FLUENCE" not in html
        # 40 行不触发打印截断（检查真实的行属性/提示文本，而非 CSS 选择器文本）。
        assert 'class="print-hide"' not in html and "打印仅显示" not in html

        # 2) 无可靠 RM：不应出现偏振面板，但累积图仍在。
        _, html_nopol = _build(tmp, n=40, reliable_rm=False)
        assert "偏振与 RM" not in html_nopol
        assert "累积通量分布" in html_nopol

        # 3) 大目录（>200 行）：打印截断的行属性与提示都应出现。
        _, html_big = _build(tmp, n=260, reliable_rm=True)
        assert 'class="print-hide"' in html_big
        assert "打印仅显示" in html_big

    print("[OK] all smoke tests passed")


if __name__ == "__main__":
    test_pipeline()
