# -*- coding: utf-8 -*-
"""
Build a static observation dashboard from burst_results.csv.

The dashboard is self-contained: charts are rendered with matplotlib/seaborn and
embedded as base64 PNGs, so the output HTML opens directly without a web server
or JavaScript runtime.

Design notes
------------
* RM is only treated as a measurement when its search significance clears the
  reliability threshold. When no burst has a reliable RM, the polarization panel
  (RM / linear / circular fraction) is dropped entirely instead of shown with a
  "unreliable" caveat.
* The frequency-coverage panel uses a sweep-line occupancy curve so it stays
  readable whether there are 40 or 4000 bursts.
* The top-SNR gallery embeds each burst's dynamic spectrum (or the combined
  polarization figure when that burst's RM is reliable).
"""

import argparse
import base64
import io
import math
import re
from datetime import datetime
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # noqa: F401  (registers the "mako"/"rocket" colormaps)


# Light "observatory paper" palette. The accent values drive both the page
# chrome (build_css) and the matplotlib charts so the two stay in step.
PALETTE = {
    "ink": "#1A2233",
    "muted": "#6B7686",
    "line": "#E2E7F0",
    "surface": "#FFFFFF",
    "page": "#F4F6FB",
    "blue": "#2E6F95",
    "teal": "#1F9E8F",
    "gold": "#E0982B",
    "rose": "#C44E6B",
    "violet": "#6A4C93",
    "green": "#5A8F3C",
    "red": "#C5443B",
}

# Ordered categorical colors used across the matplotlib charts.
CYCLE = [PALETTE["blue"], PALETTE["teal"], PALETTE["gold"], PALETTE["rose"], PALETTE["violet"], PALETTE["green"]]


# --- received-energy estimate (shown in the summary rail) --------------------
# Fluence in the CSV is a spectral fluence (Jy·ms), measured over each burst's
# own bandwidth, so per-burst values cannot simply be summed. Multiplying by
# bandwidth integrates over frequency to an energy fluence (J/m^2); folding in
# FAST's illuminated aperture turns that into the radio energy intercepted (J).
FAST_EFF_DIAMETER_M = 300.0       # illuminated ("effective") aperture diameter
JY_MS_MHZ_TO_J_PER_M2 = 1.0e-23   # 1 Jy·ms·MHz == 1e-23 J/m^2
PAGE_FLIP_ENERGY_J = 6.6e-3       # ~lifting a 4.5 g A4 sheet by 15 cm (m·g·h)


NUMERIC_COLUMNS = [
    "burst_idx", "toa_mjd", "flux_peak", "flux_err", "flux_err_sys",
    "fluence", "fluence_err", "fluence_err_sys", "width", "width_err",
    "width_gauss", "width_gauss_err", "snr", "freq_low", "freq_high",
    "bandwidth", "bandwidth_gauss", "bandwidth_gauss_err", "dm", "dm_err",
    "rm", "rm_err", "rm_significance", "linear_frac", "linear_frac_err",
    "circular_frac", "circular_frac_err", "center_freq",
]


DETAIL_COLUMNS = [
    "burst_no", "file_name", "burst_idx", "time_s", "toa_mjd", "snr",
    "flux_peak", "fluence", "width", "freq_low", "freq_high", "bandwidth",
    "dm", "dm_err", "rm", "rm_significance", "linear_frac", "circular_frac",
]


COLUMN_LABELS = {
    "burst_no": "#",
    "file_name": "文件",
    "burst_idx": "文件内序号",
    "time_s": "相对首个 burst 时间 (s)",
    "toa_mjd": "TOA (MJD)",
    "snr": "SNR",
    "flux_peak": "峰值流量 (Jy)",
    "fluence": "流量积分 (Jy ms)",
    "width": "宽度 (ms)",
    "freq_low": "低频边界 (MHz)",
    "freq_high": "高频边界 (MHz)",
    "bandwidth": "带宽 (MHz)",
    "dm": "DM (pc cm^-3)",
    "dm_err": "DM 误差",
    "rm": "RM (rad m^-2)",
    "rm_significance": "RM 显著性",
    "linear_frac": "线偏振 (%)",
    "circular_frac": "圆偏振 (%)",
}


def parse_args():
    parser = argparse.ArgumentParser(description="根据 burst_results.csv 生成静态观测面板")
    parser.add_argument("--csv", required=True, help="burst_results.csv 路径")
    parser.add_argument(
        "--output",
        default=None,
        help="输出 HTML 路径；默认写到 CSV 同目录的 burst_dashboard.html",
    )
    parser.add_argument(
        "--analysis-dir",
        default=None,
        help="逐 burst analysis 目录；默认与 CSV 同目录",
    )
    parser.add_argument("--source", default=None, help="源名，例如 FRB121102")
    parser.add_argument("--date", default=None, help="观测日期，例如 20260626")
    parser.add_argument("--title", default=None, help="面板标题")
    parser.add_argument(
        "--reference-dm",
        type=float,
        default=None,
        help="参考 DM；提供后会在 DM 图中画水平线",
    )
    parser.add_argument("--snr-threshold", type=float, default=5.0, help="低 SNR 提示阈值")
    parser.add_argument("--dm-err-threshold", type=float, default=5.0, help="DM 误差提示阈值")
    parser.add_argument(
        "--rm-significance-threshold",
        type=float,
        default=5.0,
        help="RM 可靠性显著性阈值",
    )
    parser.add_argument("--top-n", type=int, default=10, help="动态谱画廊展示的最高 SNR 数量")
    return parser.parse_args()


def load_results(csv_path, rm_threshold):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "file_name" not in df.columns:
        df["file_name"] = ""
    if "burst_idx" not in df.columns:
        df["burst_idx"] = 0

    if "toa_mjd" in df.columns and df["toa_mjd"].notna().any():
        first_toa = df["toa_mjd"].min()
        df["time_s"] = (df["toa_mjd"] - first_toa) * 86400.0
        sort_cols = ["toa_mjd", "file_name", "burst_idx"]
    else:
        df["time_s"] = np.arange(len(df), dtype=float)
        sort_cols = ["file_name", "burst_idx"]

    df = df.sort_values(sort_cols).reset_index(drop=True)
    df["burst_no"] = np.arange(1, len(df) + 1)
    df["burst_label"] = df["burst_no"].map(lambda value: f"B{value:03d}")

    if "rm_significance" in df.columns:
        rm_reliable = df["rm_significance"] >= rm_threshold
    else:
        rm_reliable = pd.Series(False, index=df.index)
    df["rm_reliable"] = rm_reliable.fillna(False)
    return df


def infer_metadata(df, csv_path, source=None, date=None, title=None):
    inferred_source = source
    inferred_date = date
    beam = None

    if "file_name" in df.columns and df["file_name"].notna().any():
        sample = str(df["file_name"].dropna().iloc[0])
        match = re.search(r"^(?P<source>.+?)-(?P<date>\d{8})-M(?P<beam>\d{2})-", sample)
        if match:
            inferred_source = inferred_source or match.group("source")
            inferred_date = inferred_date or match.group("date")
            beam = f"M{match.group('beam')}"

    csv_path = Path(csv_path)
    if inferred_date is None:
        for part in csv_path.parts:
            if re.fullmatch(r"\d{8}", part):
                inferred_date = part
                break

    inferred_source = inferred_source or "Unknown source"
    inferred_date = inferred_date or "Unknown date"
    title = title or f"{inferred_source} {inferred_date} 观测分析面板"
    return {
        "source": inferred_source,
        "date": inferred_date,
        "beam": beam or "未知",
        "title": title,
    }


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def fmt_value(value, digits=2, suffix=""):
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except TypeError:
        pass
    if isinstance(value, (int, np.integer)):
        return f"{value}{suffix}"
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return "—"
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def stat_range(df, col, digits=2, suffix=""):
    if col not in df.columns or not df[col].notna().any():
        return "—"
    return f"{fmt_value(df[col].min(), digits, suffix)} – {fmt_value(df[col].max(), digits, suffix)}"


_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def fmt_sci(value, sig=2, suffix=""):
    """Format a number as 'm×10ⁿ' with unicode superscripts (no HTML needed)."""
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(v) or v == 0:
        return "—"
    exp = int(math.floor(math.log10(abs(v))))
    mant = v / 10.0 ** exp
    return f"{mant:.{sig - 1}f}×10{str(exp).translate(_SUPERSCRIPT)}{suffix}"


def received_energy_joule(df):
    """Radio energy intercepted by the dish over the session, in joules.

    Sum of (fluence × bandwidth) gives an energy fluence (J/m^2); multiplying by
    FAST's effective collecting area yields the energy actually received.
    """
    if not {"fluence", "bandwidth"}.issubset(df.columns):
        return None
    pair = df[["fluence", "bandwidth"]].dropna()
    if pair.empty:
        return None
    energy_fluence = float((pair["fluence"] * pair["bandwidth"]).sum()) * JY_MS_MHZ_TO_J_PER_M2
    area = math.pi * (FAST_EFF_DIAMETER_M / 2.0) ** 2
    return energy_fluence * area


def figure_to_data_uri(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def png_file_to_data_uri(path):
    data = Path(path).read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def setup_plot_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE["surface"],
            "axes.facecolor": PALETTE["surface"],
            "savefig.facecolor": PALETTE["surface"],
            "axes.edgecolor": PALETTE["line"],
            "axes.labelcolor": PALETTE["ink"],
            "axes.titlecolor": PALETTE["ink"],
            "text.color": PALETTE["ink"],
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "grid.color": "#EAEEF5",
            "grid.linewidth": 0.9,
            "axes.prop_cycle": plt.cycler(color=CYCLE),
            "font.sans-serif": [
                "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                "Arial Unicode MS", "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
        }
    )


def scaled_sizes(values, low=42, high=240):
    vals = pd.to_numeric(values, errors="coerce")
    if vals.notna().sum() == 0:
        return np.full(len(vals), (low + high) / 2)
    filled = vals.fillna(vals.median()).to_numpy(dtype=float)
    vmin, vmax = np.nanmin(filled), np.nanmax(filled)
    if np.isclose(vmin, vmax):
        return np.full(len(vals), (low + high) / 2)
    return low + (filled - vmin) / (vmax - vmin) * (high - low)


def add_no_data(ax, message="无可用数据"):
    ax.text(0.5, 0.5, message, ha="center", va="center", color=PALETTE["muted"], fontsize=12)
    ax.set_axis_off()


def style_colorbar(cbar, label):
    cbar.set_label(label, color=PALETTE["ink"])
    cbar.outline.set_edgecolor(PALETTE["line"])
    cbar.ax.tick_params(color=PALETTE["line"], labelcolor=PALETTE["muted"])


# --------------------------------------------------------------------------- #
# charts
# --------------------------------------------------------------------------- #
def plot_timeline(df, snr_threshold):
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    x = df["time_s"] / 60.0
    y = df["snr"] if "snr" in df.columns else pd.Series(np.nan, index=df.index)
    sizes = scaled_sizes(df["fluence"] if "fluence" in df.columns else y)

    if y.notna().any():
        has_fluence = "fluence" in df.columns and df["fluence"].notna().any()
        colors = df["fluence"] if has_fluence else y
        scatter = ax.scatter(
            x, y, s=sizes, c=colors, cmap="mako_r",
            edgecolor="white", linewidth=0.8, alpha=0.92, zorder=3,
        )
        style_colorbar(fig.colorbar(scatter, ax=ax, pad=0.01),
                       "Fluence (Jy ms)" if has_fluence else "SNR")
        ax.axhline(snr_threshold, color=PALETTE["gold"], linestyle="--",
                   linewidth=1.4, label=f"SNR = {snr_threshold:g}", zorder=2)
        ax.legend(loc="upper left", frameon=False)
        ax.set_title("Burst SNR 时间分布")
        ax.set_xlabel("距离首个 burst 的时间 (min)")
        ax.set_ylabel("SNR")
    else:
        add_no_data(ax)
    return figure_to_data_uri(fig)


def plot_dm(df, reference_dm, dm_err_threshold):
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    if "dm" not in df.columns or not df["dm"].notna().any():
        add_no_data(ax)
        return figure_to_data_uri(fig)

    x = df["time_s"] / 60.0
    dm_err = df["dm_err"] if "dm_err" in df.columns else pd.Series(np.nan, index=df.index)
    high_err = (dm_err > dm_err_threshold).fillna(False)
    normal = ~high_err

    ax.scatter(x[normal], df.loc[normal, "dm"], s=66, color=PALETTE["blue"],
               edgecolor="white", linewidth=0.7, label=f"DM err ≤ {dm_err_threshold:g}", zorder=3)
    if high_err.any():
        ax.scatter(x[high_err], df.loc[high_err, "dm"], s=78, facecolor="none",
                   edgecolor=PALETTE["gold"], linewidth=1.7,
                   label=f"DM err > {dm_err_threshold:g}", zorder=3)
    if reference_dm is not None:
        ax.axhline(reference_dm, color=PALETTE["rose"], linestyle="--",
                   linewidth=1.3, label=f"参考 DM {reference_dm:g}")

    ax.set_title("DM 搜索结果")
    ax.set_xlabel("距离首个 burst 的时间 (min)")
    ax.set_ylabel("DM (pc cm$^{-3}$)")
    ax.legend(loc="best", frameon=False)
    return figure_to_data_uri(fig)


def plot_flux_fluence_width(df):
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    required = {"width", "fluence", "snr"}
    if not required.issubset(df.columns) or df[list(required)].dropna().empty:
        add_no_data(ax)
        return figure_to_data_uri(fig)

    data = df.dropna(subset=["width", "fluence", "snr"]).copy()
    sizes = scaled_sizes(data["flux_peak"] if "flux_peak" in data.columns else data["snr"], 48, 220)
    scatter = ax.scatter(data["width"], data["fluence"], s=sizes, c=data["snr"],
                         cmap="rocket_r", edgecolor="white", linewidth=0.8, alpha=0.9, zorder=3)
    if data["fluence"].min() > 0 and data["fluence"].max() / data["fluence"].min() > 12:
        ax.set_yscale("log")
    style_colorbar(fig.colorbar(scatter, ax=ax, pad=0.01), "SNR")
    ax.set_title("能量 · 宽度 · SNR")
    ax.set_xlabel("Width (ms)")
    ax.set_ylabel("Fluence (Jy ms)")
    return figure_to_data_uri(fig)


def plot_frequency_coverage(df):
    """Sweep-line occupancy + center-frequency histogram. Scales to many bursts."""
    required = {"freq_low", "freq_high"}
    if not required.issubset(df.columns) or df[list(required)].dropna().empty:
        fig, ax = plt.subplots(figsize=(9.2, 5.2))
        add_no_data(ax)
        return figure_to_data_uri(fig)

    data = df.dropna(subset=["freq_low", "freq_high"])
    lows = data["freq_low"].to_numpy(dtype=float)
    highs = data["freq_high"].to_numpy(dtype=float)

    # Sweep line: +1 at every freq_low, -1 at every freq_high.
    points = np.concatenate([lows, highs])
    deltas = np.concatenate([np.ones_like(lows), -np.ones_like(highs)])
    order = np.argsort(points, kind="mergesort")
    xs = points[order]
    occ = np.cumsum(deltas[order])

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9.2, 5.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.1], "hspace": 0.08},
    )

    ax_top.fill_between(xs, occ, step="post", color=PALETTE["blue"], alpha=0.22, zorder=2)
    ax_top.step(xs, occ, where="post", color=PALETTE["blue"], linewidth=2.0, zorder=3)
    peak = int(occ.max()) if len(occ) else 0
    ax_top.axhline(peak, color=PALETTE["gold"], linestyle="--", linewidth=1.2,
                   label=f"峰值覆盖 {peak} 个 burst")
    ax_top.set_ylabel("覆盖该频率的 burst 数")
    ax_top.set_ylim(bottom=0)
    ax_top.set_title(f"频率覆盖占用（共 {len(data)} 个 burst）")
    ax_top.legend(loc="upper left", frameon=False)

    center = data["center_freq"] if "center_freq" in data.columns and data["center_freq"].notna().any() \
        else pd.Series((lows + highs) / 2.0, index=data.index)
    ax_bot.hist(center.dropna(), bins=min(60, max(12, int(np.sqrt(len(data)) * 2))),
                color=PALETTE["teal"], alpha=0.85)
    ax_bot.set_ylabel("中心频率\n直方")
    ax_bot.set_xlabel("Frequency (MHz)")
    return figure_to_data_uri(fig)


def plot_distributions(df):
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2))
    ax_snr, ax_flu, ax_wid, ax_bw = axes.ravel()

    def basic_hist(ax, col, label, color):
        if col in df.columns and df[col].notna().any():
            vals = df[col].dropna()
            bins = min(20, max(6, int(np.sqrt(len(vals)))))
            sns.histplot(vals, bins=bins, ax=ax, color=color, edgecolor="white", linewidth=0.6)
            ax.set_xlabel(label)
            ax.set_ylabel("Count")
            ax.set_title(label)
        else:
            add_no_data(ax)

    basic_hist(ax_snr, "snr", "SNR", PALETTE["blue"])

    # Fluence: log-spaced bins + log x-axis (spans orders of magnitude in big samples).
    if "fluence" in df.columns and (df["fluence"] > 0).any():
        vals = df.loc[df["fluence"] > 0, "fluence"].dropna()
        n_bins = min(24, max(6, int(np.sqrt(len(vals)))))
        bins = np.logspace(np.log10(vals.min()), np.log10(vals.max()), n_bins + 1)
        ax_flu.hist(vals, bins=bins, color=PALETTE["gold"], edgecolor="white", linewidth=0.6)
        ax_flu.set_xscale("log")
        ax_flu.set_xlabel("Fluence (Jy ms)")
        ax_flu.set_ylabel("Count")
        ax_flu.set_title("Fluence (log bins)")
    else:
        add_no_data(ax_flu)

    basic_hist(ax_wid, "width", "Width (ms)", PALETTE["green"])
    basic_hist(ax_bw, "bandwidth", "Bandwidth (MHz)", PALETTE["rose"])

    fig.suptitle("关键属性分布", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    return figure_to_data_uri(fig)


def plot_waiting_time(df):
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    if "toa_mjd" not in df.columns or df["toa_mjd"].notna().sum() < 2:
        add_no_data(ax, "burst 数不足，无法计算 waiting time")
        return figure_to_data_uri(fig)

    toa = df["toa_mjd"].dropna().sort_values().to_numpy(dtype=float)
    dt = np.diff(toa) * 86400.0
    dt = dt[dt > 0]
    if dt.size == 0:
        add_no_data(ax, "无有效 waiting time")
        return figure_to_data_uri(fig)

    n_bins = min(28, max(6, int(np.sqrt(dt.size) * 1.5)))
    bins = np.logspace(np.log10(dt.min()), np.log10(dt.max()), n_bins + 1)
    ax.hist(dt, bins=bins, color=PALETTE["violet"], edgecolor="white", linewidth=0.6, alpha=0.9)
    ax.set_xscale("log")
    median = float(np.median(dt))
    ax.axvline(median, color=PALETTE["gold"], linestyle="--", linewidth=1.4,
               label=f"中位数 {median:.1f} s")
    ax.set_title("Waiting time 分布")
    ax.set_xlabel("相邻 burst 间隔 (s)")
    ax.set_ylabel("Count")
    ax.legend(loc="best", frameon=False)
    return figure_to_data_uri(fig)


def plot_polarization(df):
    """RM, linear and circular polarization for bursts with reliable RM."""
    data = df[df["rm_reliable"]].copy()
    fig, (ax_rm, ax_pol) = plt.subplots(
        2, 1, figsize=(9.2, 6.0), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12},
    )
    x = data["time_s"] / 60.0

    rm_err = data["rm_err"] if "rm_err" in data.columns else None
    ax_rm.errorbar(x, data["rm"], yerr=rm_err, fmt="o", color=PALETTE["blue"],
                   ecolor=PALETTE["line"], elinewidth=1.2, capsize=3,
                   markeredgecolor="white", markersize=8, zorder=3)
    ax_rm.set_ylabel("RM (rad m$^{-2}$)")
    ax_rm.set_title("可靠 RM 与偏振")

    lin_err = data["linear_frac_err"] if "linear_frac_err" in data.columns else None
    cir_err = data["circular_frac_err"] if "circular_frac_err" in data.columns else None
    ax_pol.errorbar(x, data["linear_frac"], yerr=lin_err, fmt="s", color=PALETTE["teal"],
                    ecolor=PALETTE["line"], elinewidth=1.1, capsize=3,
                    markeredgecolor="white", markersize=8, label="线偏振 L/I", zorder=3)
    ax_pol.errorbar(x, data["circular_frac"], yerr=cir_err, fmt="^", color=PALETTE["rose"],
                    ecolor=PALETTE["line"], elinewidth=1.1, capsize=3,
                    markeredgecolor="white", markersize=8, label="圆偏振 V/I", zorder=3)
    ax_pol.axhline(0, color=PALETTE["muted"], linewidth=0.8)
    ax_pol.set_ylabel("偏振分数 (%)")
    ax_pol.set_xlabel("距离首个 burst 的时间 (min)")
    ax_pol.legend(loc="best", frameon=False)
    return figure_to_data_uri(fig)


# --------------------------------------------------------------------------- #
# hero detection strip (the signature element)
# --------------------------------------------------------------------------- #
def build_hero_strip(df):
    """Inline SVG: one glowing tick per burst, x = arrival time, height ∝ S/N.

    This is the page's signature — a session's bursts are discrete events in
    time, so the hero shows exactly that rather than a generic statistic.
    """
    if "snr" not in df.columns or not df["snr"].notna().any():
        return ""

    d = df.dropna(subset=["snr"])
    t = d["time_s"].to_numpy(dtype=float)
    s = d["snr"].to_numpy(dtype=float)

    W, H = 1000.0, 168.0
    pad_l, pad_r, base_y, top_y = 5.0, 5.0, 146.0, 14.0
    tmin, tmax = float(t.min()), float(t.max())
    span = (tmax - tmin) or 1.0
    smax = float(s.max()) or 1.0
    xs = pad_l + (t - tmin) / span * (W - pad_l - pad_r)
    hs = np.sqrt(np.clip(s, 0.0, None) / smax) * (base_y - top_y)

    parts = [
        f'<line class="strip-axis" x1="{pad_l:.1f}" y1="{base_y:.1f}" '
        f'x2="{W - pad_r:.1f}" y2="{base_y:.1f}"/>'
    ]
    for x, h, sv in zip(xs, hs, s):
        opacity = 0.32 + 0.68 * (sv / smax)
        parts.append(
            f'<line class="strip-mark" x1="{x:.1f}" y1="{base_y:.1f}" '
            f'x2="{x:.1f}" y2="{base_y - h:.1f}" style="opacity:{opacity:.2f}"/>'
        )
    peak = int(np.argmax(s))
    parts.append(f'<circle class="strip-peak" cx="{xs[peak]:.1f}" cy="{base_y - hs[peak]:.1f}" r="3.4"/>')

    return (
        f'<svg class="strip" viewBox="0 0 {W:.0f} {H:.0f}" '
        f'role="img" aria-label="burst detection strip">{"".join(parts)}</svg>'
    )


# --------------------------------------------------------------------------- #
# top-SNR gallery
# --------------------------------------------------------------------------- #
def build_gallery(df, analysis_dir, top_n):
    if "snr" not in df.columns or not df["snr"].notna().any():
        return ""

    top = df.dropna(subset=["snr"]).sort_values("snr", ascending=False).head(top_n)
    analysis_dir = Path(analysis_dir)
    items = []

    for _, row in top.iterrows():
        stem = str(row.get("file_name", "")).removesuffix(".h5")
        folder = analysis_dir / stem
        burst_idx = int(row["burst_idx"]) if pd.notna(row.get("burst_idx")) else 0
        reliable = bool(row.get("rm_reliable", False))

        uri = None
        kind = ""
        if reliable:
            pol_png = folder / "combined_polarization.png"
            if pol_png.exists():
                uri = png_file_to_data_uri(pol_png)
                kind = "combined_polarization"
        if uri is None:
            ds_path = folder / "dynamic_spectrum.png"
            if ds_path.exists():
                uri = png_file_to_data_uri(ds_path)
                kind = "dynamic_spectrum"

        snr_text = fmt_value(row.get("snr"), 1)
        flu_text = fmt_value(row.get("fluence"), 3)
        tag = "POL" if kind == "combined_polarization" else "WATERFALL"
        if uri is None:
            body = f'<div class="plate-missing">image not found<br><span>{escape(stem)}</span></div>'
        else:
            body = f'<img src="{uri}" alt="{escape(stem)}" loading="lazy">'
        items.append(
            f"""
            <figure class="plate">
              <div class="plate-img">{body}</div>
              <figcaption class="plate-cap">
                <div class="plate-row">
                  <span class="plate-id">{escape(row['burst_label'])}</span>
                  <span class="plate-tag">{tag}</span>
                </div>
                <div class="plate-metrics">
                  <span><i>S/N</i>{escape(snr_text)}</span>
                  <span><i>Fluence</i>{escape(flu_text)}</span>
                </div>
                <span class="plate-file">{escape(stem)} · idx {burst_idx}</span>
              </figcaption>
            </figure>
            """
        )
    return "\n".join(items)


# --------------------------------------------------------------------------- #
# cards + table + html
# --------------------------------------------------------------------------- #
def build_cards(df, snr_threshold, dm_err_threshold):
    burst_count = len(df)
    file_count = df["file_name"].nunique() if "file_name" in df.columns else 0
    span_s = df["time_s"].max() - df["time_s"].min() if "time_s" in df.columns else np.nan
    low_snr = int((df["snr"] < snr_threshold).sum()) if "snr" in df.columns else 0
    high_dm_err = int((df["dm_err"] > dm_err_threshold).sum()) if "dm_err" in df.columns else 0
    rm_reliable = int(df["rm_reliable"].sum())
    peak_snr = df["snr"].max() if "snr" in df.columns and df["snr"].notna().any() else np.nan

    if "toa_mjd" in df.columns and df["toa_mjd"].notna().sum() >= 2 and span_s and span_s > 0:
        rate_text = fmt_value(burst_count / (span_s / 3600.0), 1)
    else:
        rate_text = "—"

    energy_j = received_energy_joule(df)
    if energy_j:
        energy_val = fmt_sci(energy_j, 2, " J")
        energy_note = f"≈ 翻动 {fmt_sci(energy_j / PAGE_FLIP_ENERGY_J, 2)} 页 A4 纸"
    else:
        energy_val, energy_note = "—", "缺少 bandwidth"

    cards = [
        ("EVENTS", f"{burst_count}", f"{file_count} 个文件"),
        ("SPAN", fmt_value(span_s / 60.0, 1, " min"), "按 TOA 计算"),
        ("RATE", rate_text, "events · h⁻¹"),
        ("PEAK S/N", fmt_value(peak_snr, 1), f"{low_snr} 条 < {snr_threshold:g}"),
        ("DM RANGE", stat_range(df, "dm", 1), f"{high_dm_err} 条高误差"),
        ("RECEIVED ENERGY", energy_val, energy_note),
        ("WIDTH", stat_range(df, "width", 1, " ms"), "fluence / peak"),
        ("RELIABLE RM", f"{rm_reliable} / {burst_count}", "达到显著性阈值"),
    ]
    return "\n".join(
        f"""
        <div class="stat">
          <span class="stat-k">{escape(label)}</span>
          <span class="stat-v">{escape(value)}</span>
          <span class="stat-n">{escape(note)}</span>
        </div>
        """
        for label, value, note in cards
    )


def build_detail_table(df, show_pol):
    columns = [c for c in DETAIL_COLUMNS if c in df.columns]
    if not show_pol:
        columns = [c for c in columns if c not in {"rm", "rm_significance", "linear_frac", "circular_frac"}]
    header = "".join(f"<th>{escape(COLUMN_LABELS.get(col, col))}</th>" for col in columns)

    float_cols = {"time_s", "snr", "flux_peak", "fluence", "width", "freq_low",
                  "freq_high", "bandwidth", "dm", "dm_err", "rm", "rm_significance",
                  "linear_frac", "circular_frac"}
    rows = []
    for _, row in df[columns].iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if col == "toa_mjd":
                text = fmt_value(value, 9)
            elif col in float_cols:
                digits = 3 if col in {"flux_peak", "fluence", "dm", "rm_significance"} else 2
                text = fmt_value(value, digits)
            else:
                text = fmt_value(value)
            css = " muted-cell" if col in {"linear_frac", "circular_frac"} else ""
            cells.append(f'<td class="{css.strip()}">{escape(text)}</td>')
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"""
    <div class="table-wrap">
      <table>
        <thead><tr>{header}</tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """


def build_html(df, csv_path, output_path, analysis_dir, metadata, args):
    setup_plot_style()
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    show_pol = bool(df["rm_reliable"].any())

    charts = [
        ("Burst 时间分布", "SNR 随观测内 TOA 变化，点大小/颜色对应 fluence。", plot_timeline(df, args.snr_threshold)),
        ("DM 搜索", "空心点表示 DM 误差超过阈值；参考 DM 仅作配置标尺。", plot_dm(df, args.reference_dm, args.dm_err_threshold)),
        ("能量与宽度", "fluence、width、SNR 与 peak flux 的联合分布。", plot_flux_fluence_width(df)),
        ("频率覆盖", "占用曲线表示每个频率被多少 burst 覆盖，下方为中心频率直方图。", plot_frequency_coverage(df)),
        ("属性分布", "SNR、fluence（对数 bins）、width、bandwidth 的总体分布。", plot_distributions(df)),
        ("Waiting time", "相邻 burst 时间间隔的对数分布。", plot_waiting_time(df)),
    ]
    if show_pol:
        charts.append(
            ("偏振与 RM", "仅展示 RM 达到显著性阈值的 burst：RM、线偏振、圆偏振。", plot_polarization(df))
        )

    chart_html = "\n".join(
        f"""
        <article class="chart-card">
          <div class="chart-head">
            <h3>{escape(title)}</h3>
            <p>{escape(subtitle)}</p>
          </div>
          <img src="{uri}" alt="{escape(title)}">
        </article>
        """
        for title, subtitle, uri in charts
    )

    gallery_html = build_gallery(df, analysis_dir, args.top_n)
    detail_table = build_detail_table(df, show_pol)
    cards = build_cards(df, args.snr_threshold, args.dm_err_threshold)
    strip = build_hero_strip(df)

    # Hero readout values.
    burst_count = len(df)
    file_count = df["file_name"].nunique() if "file_name" in df.columns else 0
    span_min = (df["time_s"].max() - df["time_s"].min()) / 60.0 if "time_s" in df.columns else float("nan")
    if "snr" in df.columns and df["snr"].notna().any():
        pk = df["snr"].idxmax()
        peak_snr_txt = fmt_value(df.loc[pk, "snr"], 1)
        peak_t_txt = fmt_value(df.loc[pk, "time_s"] / 60.0, 1)
    else:
        peak_snr_txt = peak_t_txt = "—"

    strip_block = ""
    if strip:
        strip_block = f"""
      <div class="strip-wrap">
        {strip}
        <div class="strip-legend">
          <span>+0 min</span>
          <span class="strip-note">每条竖线 = 一个 burst · 高度 ∝ S/N · 峰值 S/N {escape(peak_snr_txt)} @ +{escape(peak_t_txt)} min</span>
          <span>+{escape(fmt_value(span_min, 1))} min</span>
        </div>
      </div>"""

    gallery_section = ""
    if gallery_html:
        gal_note = (
            "按 S/N 从高到低；RM 可靠的 burst 展示 combined_polarization，其余展示 dynamic_spectrum。"
            if show_pol else
            "按 S/N 从高到低，展示每个 burst 所在文件的 dynamic_spectrum。"
        )
        gallery_section = f"""
    <section class="block">
      {section_head("STRONGEST DETECTIONS", f"top {args.top_n}")}
      <p class="block-note">{escape(gal_note)}</p>
      <div class="gallery">
        {gallery_html}
      </div>
    </section>"""

    if show_pol:
        seam_class = "seam"
        pol_note = (
            f"发现 {int(df['rm_reliable'].sum())} 个达到显著性阈值（≥ {args.rm_significance_threshold:g}）的 RM；"
            "偏振结果应结合对应 RM 图与误差单独复核。"
        )
    else:
        seam_class = "seam caution"
        pol_note = (
            f"本次观测没有 RM 显著性达到阈值（≥ {args.rm_significance_threshold:g}）的 burst，"
            "因此不展示偏振 / RM 图。linear_frac、circular_frac 仅作为流水线字段保留。"
        )

    css = build_css()

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(metadata['title'])}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>{css}</style>
</head>
<body>
  <main class="shell">
    <div class="topbar">
      <span class="brand">◇ BURST DOSSIER</span>
      <span class="topbar-meta">{escape(str(Path(csv_path).resolve()))}</span>
    </div>

    <header class="hero">
      <div class="hero-eyebrow">FAST · L-BAND DETECTION LOG</div>
      <h1 class="hero-title">{escape(metadata['source'])}</h1>
      <div class="hero-sub">
        <span><b>{escape(metadata['date'])}</b></span><i>/</i>
        <span>BEAM <b>{escape(metadata['beam'])}</b></span><i>/</i>
        <span><b>{burst_count}</b> EVENTS</span><i>/</i>
        <span><b>{file_count}</b> FILES</span>
      </div>{strip_block}
    </header>

    <section class="rail">
      {cards}
    </section>

    <div class="{seam_class}">
      <span class="seam-mark">◆</span>
      <span>{escape(pol_note)}</span>
    </div>

    <section class="block">
      {section_head("SIGNAL PROPERTIES", f"{len(charts)} views")}
      <div class="charts">
        {chart_html}
      </div>
    </section>
    {gallery_section}
    <section class="block">
      {section_head("BURST CATALOG", f"{len(df)} rows")}
      {detail_table}
    </section>

    <footer class="foot">
      <span>Generated by burst_dashboard.py · {escape(generated_at)}</span>
      <span>{escape(str(Path(output_path).resolve()))}</span>
    </footer>
  </main>
</body>
</html>
"""
    return html


def section_head(label, count_text):
    return (
        f'<div class="block-head"><span class="mark">◇</span>'
        f'<h2>{escape(label)}</h2><span class="rule"></span>'
        f'<span class="count">{escape(count_text)}</span></div>'
    )


def build_css():
    root = (
        ":root{"
        "--page:#F4F6FB;--surface:#FFFFFF;--panel-2:#F7F9FC;"
        "--ink:#18222F;--muted:#5C6B82;--faint:#94A1B5;"
        "--line:#E6EBF3;--line-strong:#D6DEEA;"
        "--teal:#138A7C;--teal-soft:#1F9E8F;--blue:#2E6F95;--gold:#BD831E;"
        "--shadow-sm:0 1px 2px rgba(16,24,40,.04);"
        "--shadow:0 14px 34px rgba(16,24,40,.07);"
        '--mono:"IBM Plex Mono","Cascadia Code",Consolas,"Microsoft YaHei",monospace;'
        '--disp:"Space Grotesk","Microsoft YaHei","Segoe UI",system-ui,sans-serif;'
        "}"
    )
    body = r"""
    *{ box-sizing:border-box; }
    html{ scroll-behavior:smooth; }
    body{
      margin:0; color:var(--ink); font-family:var(--disp); line-height:1.55;
      -webkit-font-smoothing:antialiased; overflow-x:hidden;
      background:
        radial-gradient(1100px 560px at 8% -6%, rgba(19,138,124,.06), transparent 60%),
        radial-gradient(960px 640px at 102% 0%, rgba(46,111,149,.07), transparent 55%),
        var(--page);
      background-attachment:fixed;
    }
    .shell{ width:min(1240px, calc(100vw - 40px)); margin:0 auto; padding:0 0 64px; }

    .topbar{
      display:flex; justify-content:space-between; align-items:center; gap:16px;
      padding:18px 2px; border-bottom:1px solid var(--line);
      font-family:var(--mono); font-size:11.5px; letter-spacing:.16em;
      text-transform:uppercase; color:var(--muted);
    }
    .topbar .brand{ color:var(--teal); white-space:nowrap; }
    .topbar-meta{
      overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
      max-width:62%; letter-spacing:.04em; text-transform:none; direction:rtl;
    }

    .hero{ padding:46px 2px 6px; }
    .hero-eyebrow{
      font-family:var(--mono); font-size:12px; letter-spacing:.34em;
      text-transform:uppercase; color:var(--teal); margin-bottom:18px;
    }
    .hero-title{
      margin:0; font-family:var(--disp); font-weight:600;
      font-size:clamp(46px, 8.4vw, 108px); line-height:.9; letter-spacing:-.025em;
      color:var(--blue);
      background:linear-gradient(96deg,#138A7C 0%, #1A7F8C 46%, var(--blue) 100%);
      -webkit-background-clip:text; background-clip:text;
      -webkit-text-fill-color:transparent;
    }
    .hero-sub{
      margin-top:20px; font-family:var(--mono); font-size:13px; letter-spacing:.12em;
      color:var(--muted); display:flex; flex-wrap:wrap; gap:10px; align-items:center;
    }
    .hero-sub i{ color:var(--faint); font-style:normal; }
    .hero-sub b{ color:var(--ink); font-weight:600; }

    .strip-wrap{ margin-top:28px; }
    .strip{ width:100%; height:auto; display:block;
      filter:drop-shadow(0 2px 4px rgba(19,138,124,.22)); }
    .strip-mark{ stroke:var(--teal-soft); stroke-width:2.1; stroke-linecap:round; }
    .strip-axis{ stroke:var(--line-strong); stroke-width:1.2; }
    .strip-peak{ fill:#0E7E72; }
    .strip-legend{
      display:flex; justify-content:space-between; align-items:baseline; gap:14px;
      margin-top:10px; font-family:var(--mono); font-size:11px; letter-spacing:.08em;
      color:var(--faint);
    }
    .strip-legend .strip-note{ color:var(--muted); text-align:center; }

    .rail{
      margin-top:36px; display:grid; grid-template-columns:repeat(4,minmax(0,1fr));
      background:var(--surface); border:1px solid var(--line); border-radius:16px;
      overflow:hidden; box-shadow:var(--shadow-sm), var(--shadow);
    }
    .stat{
      padding:18px 20px; min-width:0;
      border-right:1px solid var(--line); border-bottom:1px solid var(--line);
    }
    .stat:nth-child(4n){ border-right:none; }
    .stat:nth-last-child(-n+4){ border-bottom:none; }
    .stat-k{
      display:block; font-family:var(--mono); font-size:10.5px; letter-spacing:.18em;
      text-transform:uppercase; color:var(--muted);
    }
    .stat-v{
      display:block; margin-top:10px; font-family:var(--mono); font-weight:500;
      font-size:23px; letter-spacing:-.01em; color:var(--ink); overflow-wrap:anywhere;
    }
    .stat-n{ display:block; margin-top:8px; font-size:11.5px; color:var(--faint); }

    .seam{
      --accent:var(--teal);
      margin-top:18px; padding:14px 18px; display:flex; gap:13px; align-items:flex-start;
      background:var(--surface); border:1px solid var(--line);
      border-left:3px solid var(--accent); border-radius:12px;
      font-size:13.5px; color:var(--ink); line-height:1.6; box-shadow:var(--shadow-sm);
    }
    .seam.caution{ --accent:var(--gold); }
    .seam-mark{ font-family:var(--mono); color:var(--accent); flex:0 0 auto; font-size:12px; padding-top:2px; }

    .block{ margin-top:48px; }
    .block-head{ display:flex; align-items:center; gap:14px; margin-bottom:20px; }
    .block-head .mark{ color:var(--teal); font-size:12px; }
    .block-head h2{
      margin:0; font-family:var(--mono); font-weight:600; font-size:13px;
      letter-spacing:.26em; text-transform:uppercase; color:var(--ink); white-space:nowrap;
    }
    .block-head .rule{ flex:1; height:1px;
      background:linear-gradient(90deg, var(--line-strong), rgba(214,222,234,0)); }
    .block-head .count{ font-family:var(--mono); font-size:11px; letter-spacing:.12em;
      color:var(--faint); white-space:nowrap; }
    .block-note{ margin:-6px 0 18px; font-size:12.5px; color:var(--muted); }

    .charts{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }
    .chart-card{
      background:var(--surface); border:1px solid var(--line); border-radius:16px;
      padding:18px 18px 14px; min-width:0; box-shadow:var(--shadow-sm);
      transition:border-color .18s ease, transform .18s ease, box-shadow .18s ease;
    }
    .chart-card:hover{ border-color:rgba(19,138,124,.45); transform:translateY(-2px);
      box-shadow:var(--shadow); }
    .chart-head h3{ margin:0 0 4px; font-family:var(--disp); font-weight:600;
      font-size:17px; color:var(--ink); }
    .chart-head p{ margin:0 0 13px; font-size:12.5px; color:var(--muted); line-height:1.5; }
    .chart-card img{ display:block; width:100%; height:auto; border-radius:9px; }

    .gallery{ display:grid; grid-template-columns:repeat(auto-fill,minmax(252px,1fr)); gap:16px; }
    .plate{
      margin:0; background:var(--surface); border:1px solid var(--line); border-radius:14px;
      overflow:hidden; box-shadow:var(--shadow-sm);
      transition:transform .18s ease, border-color .18s ease, box-shadow .18s ease;
    }
    .plate:hover{ transform:translateY(-3px); border-color:rgba(19,138,124,.45);
      box-shadow:var(--shadow); }
    .plate-img{ background:#FBFCFE; display:block; line-height:0;
      border-bottom:1px solid var(--line); }
    .plate-img img{ display:block; width:100%; height:auto; }
    .plate-missing{ color:var(--muted); text-align:center; font-size:12.5px;
      padding:46px 12px; font-family:var(--mono); line-height:1.6; }
    .plate-missing span{ font-size:10.5px; color:var(--faint); }
    .plate-cap{ padding:13px 15px 15px; display:flex; flex-direction:column; gap:10px; }
    .plate-row{ display:flex; align-items:center; justify-content:space-between; gap:8px; }
    .plate-id{ font-family:var(--mono); font-weight:600; font-size:14px;
      color:var(--teal); letter-spacing:.06em; }
    .plate-tag{
      font-family:var(--mono); font-size:9.5px; letter-spacing:.16em; color:var(--gold);
      border:1px solid rgba(189,131,30,.38); border-radius:999px; padding:2px 9px;
    }
    .plate-metrics{ display:flex; gap:18px; font-family:var(--mono); font-size:12.5px;
      color:var(--ink); }
    .plate-metrics i{ color:var(--faint); font-style:normal; margin-right:6px; }
    .plate-file{ font-family:var(--mono); font-size:10px; color:var(--faint);
      overflow-wrap:anywhere; line-height:1.5; }

    .table-wrap{ background:var(--surface); border:1px solid var(--line); border-radius:16px;
      overflow:auto; max-height:660px; box-shadow:var(--shadow-sm); }
    table{ width:100%; border-collapse:collapse; font-family:var(--mono); font-size:12px;
      white-space:nowrap; }
    thead th{
      position:sticky; top:0; z-index:1; background:var(--panel-2); color:var(--muted);
      font-weight:600; font-size:10.5px; letter-spacing:.1em; text-transform:uppercase;
      padding:12px; text-align:right; border-bottom:1px solid var(--line-strong);
    }
    tbody td{ padding:9px 12px; text-align:right; color:var(--ink);
      border-bottom:1px solid #EEF1F6; }
    tbody tr:last-child td{ border-bottom:none; }
    tbody tr:hover td{ background:rgba(19,138,124,.05); }
    th:nth-child(2), td:nth-child(2){ text-align:left; max-width:340px;
      overflow:hidden; text-overflow:ellipsis; color:var(--muted); }
    .muted-cell{ color:var(--faint); }

    .foot{ margin-top:42px; padding-top:18px; border-top:1px solid var(--line);
      font-family:var(--mono); font-size:11px; letter-spacing:.05em; color:var(--faint);
      display:flex; justify-content:space-between; gap:18px; flex-wrap:wrap; }
    .foot span:last-child{ overflow-wrap:anywhere; }

    :focus-visible{ outline:2px solid var(--teal); outline-offset:2px; }

    @media (max-width:980px){
      .charts{ grid-template-columns:1fr; }
      .rail{ grid-template-columns:repeat(2,1fr); }
      .stat:nth-child(4n){ border-right:1px solid var(--line); }
      .stat:nth-child(2n){ border-right:none; }
      .stat:nth-last-child(-n+4){ border-bottom:1px solid var(--line); }
      .stat:nth-last-child(-n+2){ border-bottom:none; }
    }
    @media (max-width:560px){
      .shell{ width:calc(100vw - 24px); }
      .rail{ grid-template-columns:1fr; }
      .stat{ border-right:none; }
      .stat:not(:last-child){ border-bottom:1px solid var(--line); }
      .stat:last-child{ border-bottom:none; }
    }
    @media (prefers-reduced-motion:reduce){
      *{ transition:none !important; scroll-behavior:auto !important; }
    }
    """
    return root + body


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    output_path = Path(args.output) if args.output else csv_path.with_name("burst_dashboard.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else csv_path.parent

    df = load_results(csv_path, args.rm_significance_threshold)
    metadata = infer_metadata(df, csv_path, source=args.source, date=args.date, title=args.title)
    html = build_html(df, csv_path, output_path, analysis_dir, metadata, args)
    output_path.write_text(html, encoding="utf-8")

    rm_reliable = int(df["rm_reliable"].sum())
    print(f"[OK] Dashboard saved: {output_path}")
    print(f"  bursts: {len(df)}")
    print(f"  files: {df['file_name'].nunique() if 'file_name' in df.columns else 0}")
    print(f"  reliable RM: {rm_reliable}/{len(df)}")


if __name__ == "__main__":
    main()
