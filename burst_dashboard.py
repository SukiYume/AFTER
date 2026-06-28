# -*- coding: utf-8 -*-
"""
根据 burst_results.csv 生成自包含的静态观测面板（A3 横向海报版）。

特点
----
* 自包含：所有图表用 matplotlib 渲染后以 base64 PNG 内嵌，输出的 HTML 直接双击
  打开即可，无需任何 Web 服务或 JavaScript 运行时。
* 面向打印：CSS 里声明了 `@page A3 landscape`，浏览器「打印 → 另存为 PDF」即可
  得到带页边距的 A3 横向海报。

设计约定
--------
* 只有当 RM 的搜索显著性达到阈值时才把它当作一次「测量」。若本次观测没有任何
  burst 拥有可靠 RM，则整个偏振面板（RM / 线偏振 / 圆偏振）直接不展示，而不是
  带着「不可靠」的注脚硬塞进来。
* 频率覆盖面板用扫描线（sweep-line）占用曲线，无论 40 个还是 4000 个 burst 都清晰。
* 最高 SNR 画廊嵌入每个 burst 的动态谱；若该 burst 的 RM 可靠，则改嵌合成偏振图。
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

matplotlib.use("Agg")  # 无 GUI 后端，保证在服务器/批处理环境也能出图
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # 仅用于 set_theme(whitegrid) 提供基础网格主题
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap


# =========================================================================== #
# 一、全局配置：配色、字号、物理常数、列定义
# =========================================================================== #

# 浅色「天文台论文」配色。这些强调色同时驱动页面 chrome（build_css）和 matplotlib
# 图表，两边共用同一套色值才能保持一致。整体略微降饱和，让打印出的海报显得克制、
# 有编辑感，而不是仪表盘那种高饱和的吵闹。
PALETTE = {
    "ink": "#1B2433",
    "muted": "#5E6B82",
    "line": "#E4E9F1",
    "surface": "#FFFFFF",
    "page": "#F2F5FA",
    "blue": "#2E6F95",
    "teal": "#1A917F",
    "gold": "#C28A2C",
    "rose": "#B5566E",
    "violet": "#5A5C9C",
    "green": "#4F8A52",
}

# 唯一的连续色斜坡（青 → 蓝 → 墨，浅=小值、深=大值），页面上所有「按数值上色」的
# 量都走它：fluence、SNR、频率占用。把所有连续编码收敛到同一条斜坡，是让七个面板
# 看起来像一整套印刷图版、而非七张互不相干的图的关键。
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "obs_seq",
    ["#9BDBCF", "#46A8A0", "#2E84A0", "#2A5E84", "#1C2C46"],
)

# 离散类别色（直方图、分立标记用），都取自与 chrome 同一族系，互不冲突；同时作为
# matplotlib 的默认 prop_cycle 兜底色序。
CYCLE = [
    PALETTE["blue"], PALETTE["teal"], PALETTE["gold"],
    PALETTE["rose"], PALETTE["violet"], PALETTE["green"],
]

# 所有图（含偏振这类双子图）都用同一个约 4:3 的长宽比渲染，配合 CSS 里同比例的卡片框，
# 各图便排成整齐的每行 4 个的统一网格。
FIG_SIZE = (5.6, 4.2)
FIGURE_DPI = 160              # 内嵌图 PNG 的渲染分辨率

# --- fluence × bandwidth 汇总（显示在顶部指标栏）-------------------------- #
# CSV 里的 fluence 是谱通量积分（Jy·ms），在每个 burst 各自的带宽上测得，所以逐条
# 数值不能直接相加。按常用 E_iso 公式的观测项，先乘以带宽（GHz）后求和。
MHZ_TO_GHZ = 1.0e-3

# 读入 CSV 后强制转成数值的列（缺失或脏值转为 NaN）。
NUMERIC_COLUMNS = [
    "burst_idx", "toa_mjd", "flux_peak", "flux_err", "flux_err_sys",
    "fluence", "fluence_err", "fluence_err_sys", "width", "width_err",
    "width_gauss", "width_gauss_err", "snr", "freq_low", "freq_high",
    "bandwidth", "bandwidth_gauss", "bandwidth_gauss_err", "dm", "dm_err",
    "rm", "rm_err", "rm_significance", "linear_frac", "linear_frac_err",
    "circular_frac", "circular_frac_err", "center_freq",
]

# 明细表展示的列及顺序（实际只渲染数据里真正存在的那些）。
DETAIL_COLUMNS = [
    "burst_no", "file_name", "burst_idx", "time_s", "toa_mjd", "snr",
    "flux_peak", "fluence", "width", "freq_low", "freq_high", "bandwidth",
    "dm", "dm_err", "rm", "rm_significance", "linear_frac", "circular_frac",
]

# 明细表列名 → 中文表头。
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


# =========================================================================== #
# 二、命令行参数
# =========================================================================== #
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


# =========================================================================== #
# 三、数据载入、元信息推断、会话级汇总
# =========================================================================== #
def load_results(csv_path, rm_threshold):
    """读入 CSV，做类型清洗、按时间排序，并派生几列后续要用的辅助字段。"""
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

    # 有 TOA 就用 TOA 算相对时间（秒）并按时间排序；否则退化为按文件顺序。
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

    # rm_reliable：只有显著性达到阈值才算「可靠 RM」，下游据此决定是否展示偏振。
    if "rm_significance" in df.columns:
        rm_reliable = df["rm_significance"] >= rm_threshold
    else:
        rm_reliable = pd.Series(False, index=df.index)
    df["rm_reliable"] = rm_reliable.fillna(False)
    return df


def infer_metadata(df, csv_path, source=None, date=None, title=None):
    """从文件名 / 路径里推断源名、观测日期、波束号；命令行显式传入的优先。"""
    inferred_source = source
    inferred_date = date
    beam = None

    # 典型文件名形如 FRB121102-20260626-M01-...，从中抽源名、日期、波束。
    if "file_name" in df.columns and df["file_name"].notna().any():
        sample = str(df["file_name"].dropna().iloc[0])
        match = re.search(r"^(?P<source>.+?)-(?P<date>\d{8})-M(?P<beam>\d{2})-", sample)
        if match:
            inferred_source = inferred_source or match.group("source")
            inferred_date = inferred_date or match.group("date")
            beam = f"M{match.group('beam')}"

    # 文件名里没日期，再尝试从路径中找一段 8 位数字当日期。
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


def compute_overview(df):
    """会话级汇总统计，单一数据源同时供顶部抬头与指标栏使用，避免在两处重复计算。"""
    has_snr = "snr" in df.columns and df["snr"].notna().any()
    span_s = (df["time_s"].max() - df["time_s"].min()) if "time_s" in df.columns else float("nan")
    peak_idx = df["snr"].idxmax() if has_snr else None
    return {
        "burst_count": len(df),
        "file_count": df["file_name"].nunique() if "file_name" in df.columns else 0,
        "span_s": span_s,
        "span_min": span_s / 60.0 if isinstance(span_s, float) and math.isfinite(span_s) else float("nan"),
        "peak_snr": df["snr"].max() if has_snr else float("nan"),
        "peak_time_min": (df.loc[peak_idx, "time_s"] / 60.0) if peak_idx is not None else float("nan"),
        "reliable_rm": int(df["rm_reliable"].sum()),
    }


# =========================================================================== #
# 四、通用格式化与数值小工具
# =========================================================================== #
def fmt_value(value, digits=2, suffix=""):
    """统一的数值格式化：None / NaN / 非有限值都显示为破折号「—」。"""
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
    """某列的「最小 – 最大」范围文本，列缺失或全空时返回「—」。"""
    if col not in df.columns or not df[col].notna().any():
        return "—"
    return f"{fmt_value(df[col].min(), digits, suffix)} – {fmt_value(df[col].max(), digits, suffix)}"


# 用 unicode 上标拼科学计数法，避免在 HTML 里写 <sup>。
_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def fmt_sci(value, sig=2, suffix=""):
    """把数字格式化成 'm×10ⁿ'（用 unicode 上标，无需 HTML 标签）。"""
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


def fluence_bandwidth_jy_ms_ghz(df):
    """整场观测的 Σ(fluence × bandwidth_GHz)，单位 Jy ms GHz。

    缺少 fluence/bandwidth 列时返回 None。
    """
    if not {"fluence", "bandwidth"}.issubset(df.columns):
        return None
    pair = df[["fluence", "bandwidth"]].dropna()
    if pair.empty:
        return None
    return float((pair["fluence"] * pair["bandwidth"] * MHZ_TO_GHZ).sum())


# =========================================================================== #
# 五、绘图基础设施（样式、PNG 编码、散点/直方图辅助）
# =========================================================================== #
def figure_to_data_uri(fig):
    """把一个 matplotlib figure 存成 PNG 并编码成 base64 data URI，随后关闭释放。"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=FIGURE_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def png_file_to_data_uri(path):
    """把磁盘上已有的 PNG 文件读进来编码成 data URI（画廊嵌图用）。"""
    data = Path(path).read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def setup_plot_style():
    """统一所有图的 matplotlib 样式，使其与页面 chrome 协调一致。"""
    sns.set_theme(style="whitegrid")
    # 注册共享色斜坡，使其既能作为对象、也能按名字 "obs_seq" 取用。
    try:
        if "obs_seq" not in matplotlib.colormaps:
            matplotlib.colormaps.register(SEQUENTIAL)
    except (AttributeError, ValueError):
        pass

    # 若系统装了和页面同款的 Inter / Noto Sans SC，就让图内文字也用它们，与页面排版
    # 统一；没装则优雅回退到下面的 CJK 兜底链。
    installed = {f.name for f in font_manager.fontManager.ttflist}
    preferred = [name for name in ("Inter", "Noto Sans SC") if name in installed]
    font_stack = preferred + [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
        "Arial Unicode MS", "DejaVu Sans",
    ]

    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE["surface"],
            "axes.facecolor": PALETTE["surface"],
            "savefig.facecolor": PALETTE["surface"],
            "axes.edgecolor": PALETTE["line"],
            "axes.linewidth": 0.9,
            "axes.labelcolor": PALETTE["muted"],
            "axes.titlecolor": PALETTE["ink"],
            "text.color": PALETTE["ink"],
            "axes.titleweight": "semibold",
            "axes.titlesize": 12,
            "axes.titlepad": 7,
            "axes.labelsize": 10,
            "axes.labelpad": 4,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "grid.color": "#EBEFF6",
            "grid.linewidth": 0.8,
            "axes.prop_cycle": plt.cycler(color=CYCLE),
            "font.sans-serif": font_stack,
            "axes.unicode_minus": False,
        }
    )


def scaled_sizes(values, low=42, high=240):
    """把一列数值线性映射到散点的面积区间 [low, high]；全空或同值时取中间大小。"""
    vals = pd.to_numeric(values, errors="coerce")
    if vals.notna().sum() == 0:
        return np.full(len(vals), (low + high) / 2)
    filled = vals.fillna(vals.median()).to_numpy(dtype=float)
    vmin, vmax = np.nanmin(filled), np.nanmax(filled)
    if np.isclose(vmin, vmax):
        return np.full(len(vals), (low + high) / 2)
    return low + (filled - vmin) / (vmax - vmin) * (high - low)


def add_no_data(ax, message="无可用数据"):
    """在某个轴上画一句居中的占位提示，并关掉坐标轴（缺数据时用）。"""
    ax.text(0.5, 0.5, message, ha="center", va="center", color=PALETTE["muted"], fontsize=12)
    ax.set_axis_off()


def draw_hist(ax, values, color, *, bins, logx=False, median=False,
              median_color=None, median_label=None):
    """所有分布面板共用的一套柔和直方图画法。

    柱子用半透明填充 + 同色描边 + 柱间留细缝（rwidth），比实心高饱和色块克制得多；
    可选的虚线中位数参考线在不增加杂乱的前提下给出一个量化锚点。
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        add_no_data(ax)
        return
    ax.hist(values, bins=bins, color=color, alpha=0.5, edgecolor=color,
            linewidth=1.15, rwidth=0.9, zorder=2)
    if logx:
        ax.set_xscale("log")
    if median:
        med = float(np.median(values))
        label = median_label.format(med=med) if median_label else None
        ax.axvline(med, color=median_color or PALETTE["ink"], linestyle=(0, (4, 2)),
                   linewidth=1.2, alpha=0.9 if median_color else 0.5, zorder=3, label=label)
    ax.margins(x=0.02)


def style_colorbar(cbar, label):
    """统一 colorbar 的描边/刻度配色，使其与图表 chrome 一致。"""
    cbar.set_label(label, color=PALETTE["ink"])
    cbar.outline.set_edgecolor(PALETTE["line"])
    cbar.ax.tick_params(color=PALETTE["line"], labelcolor=PALETTE["muted"])


# =========================================================================== #
# 六、各个图表（每个函数返回一张图的 data URI）
# =========================================================================== #
def plot_timeline(df, snr_threshold):
    """01 时间分布：SNR 随观测内 TOA 变化，点大小/颜色编码 fluence。"""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    x = df["time_s"] / 60.0
    y = df["snr"] if "snr" in df.columns else pd.Series(np.nan, index=df.index)
    sizes = scaled_sizes(df["fluence"] if "fluence" in df.columns else y)

    if y.notna().any():
        has_fluence = "fluence" in df.columns and df["fluence"].notna().any()
        colors = df["fluence"] if has_fluence else y
        scatter = ax.scatter(
            x, y, s=sizes, c=colors, cmap=SEQUENTIAL,
            edgecolor="white", linewidth=0.7, alpha=0.92, zorder=3,
        )
        style_colorbar(fig.colorbar(scatter, ax=ax, pad=0.015),
                       "Fluence (Jy ms)" if has_fluence else "SNR")
        ax.axhline(snr_threshold, color=PALETTE["gold"], linestyle="--",
                   linewidth=1.3, label=f"SNR = {snr_threshold:g}", zorder=2)
        ax.legend(loc="upper left", frameon=False)
        ax.set_title("Burst SNR 时间分布")
        ax.set_xlabel("距离首个 burst 的时间 (min)")
        ax.set_ylabel("SNR")
    else:
        add_no_data(ax)
    return figure_to_data_uri(fig)


def plot_dm(df, reference_dm, dm_err_threshold):
    """02 DM 搜索：DM 随时间分布；误差超阈值的点用空心圈区分，可叠加参考 DM。"""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    if "dm" not in df.columns or not df["dm"].notna().any():
        add_no_data(ax)
        return figure_to_data_uri(fig)

    x = df["time_s"] / 60.0
    dm_err = df["dm_err"] if "dm_err" in df.columns else pd.Series(np.nan, index=df.index)
    high_err = (dm_err > dm_err_threshold).fillna(False)
    normal = ~high_err

    ax.scatter(x[normal], df.loc[normal, "dm"], s=60, color=PALETTE["blue"],
               edgecolor="white", linewidth=0.7, label=f"DM err ≤ {dm_err_threshold:g}", zorder=3)
    if high_err.any():
        ax.scatter(x[high_err], df.loc[high_err, "dm"], s=72, facecolor="none",
                   edgecolor=PALETTE["gold"], linewidth=1.6,
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
    """03 能量与宽度：width–fluence 散点，颜色编码 SNR、点大小编码峰值流量。"""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    required = {"width", "fluence", "snr"}
    if not required.issubset(df.columns) or df[list(required)].dropna().empty:
        add_no_data(ax)
        return figure_to_data_uri(fig)

    data = df.dropna(subset=["width", "fluence", "snr"]).copy()
    sizes = scaled_sizes(data["flux_peak"] if "flux_peak" in data.columns else data["snr"], 44, 210)
    scatter = ax.scatter(data["width"], data["fluence"], s=sizes, c=data["snr"],
                         cmap=SEQUENTIAL, edgecolor="white", linewidth=0.7, alpha=0.9, zorder=3)
    # fluence 跨越一个数量级以上时切到对数纵轴，避免小值被压扁。
    if data["fluence"].min() > 0 and data["fluence"].max() / data["fluence"].min() > 12:
        ax.set_yscale("log")
    style_colorbar(fig.colorbar(scatter, ax=ax, pad=0.015), "SNR")
    ax.set_title("能量 · 宽度 · SNR")
    ax.set_xlabel("Width (ms)")
    ax.set_ylabel("Fluence (Jy ms)")
    return figure_to_data_uri(fig)


def plot_frequency_coverage(df):
    """04 频率覆盖：扫描线占用曲线 + 中心频率直方图，对海量 burst 也保持可读。"""
    required = {"freq_low", "freq_high"}
    if not required.issubset(df.columns) or df[list(required)].dropna().empty:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        add_no_data(ax)
        return figure_to_data_uri(fig)

    data = df.dropna(subset=["freq_low", "freq_high"])
    lows = data["freq_low"].to_numpy(dtype=float)
    highs = data["freq_high"].to_numpy(dtype=float)

    # 扫描线：每个 freq_low 处 +1、每个 freq_high 处 -1，前缀和即「该频率被多少
    # burst 覆盖」。这样无论 burst 数多少，复杂度都只是一次排序。
    points = np.concatenate([lows, highs])
    deltas = np.concatenate([np.ones_like(lows), -np.ones_like(highs)])
    order = np.argsort(points, kind="mergesort")
    xs = points[order]
    occ = np.cumsum(deltas[order])

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=FIG_SIZE, sharex=True,
        gridspec_kw={"height_ratios": [3, 1.1], "hspace": 0.1},
    )

    ax_top.fill_between(xs, occ, step="post", color=PALETTE["blue"], alpha=0.18, zorder=2)
    ax_top.step(xs, occ, where="post", color=PALETTE["blue"], linewidth=1.9, zorder=3)
    peak = int(occ.max()) if len(occ) else 0
    ax_top.axhline(peak, color=PALETTE["gold"], linestyle="--", linewidth=1.2,
                   label=f"峰值覆盖 {peak} 个 burst")
    ax_top.set_ylabel("覆盖该频率的 burst 数")
    ax_top.set_ylim(bottom=0)
    ax_top.set_title(f"频率覆盖占用（共 {len(data)} 个 burst）")
    ax_top.legend(loc="upper left", frameon=False)

    # 下方直方图：优先用 center_freq 列，缺失则取 (low+high)/2。
    center = data["center_freq"] if "center_freq" in data.columns and data["center_freq"].notna().any() \
        else pd.Series((lows + highs) / 2.0, index=data.index)
    draw_hist(ax_bot, center.dropna(), PALETTE["teal"],
              bins=min(60, max(12, int(np.sqrt(len(data)) * 2))))
    ax_bot.set_ylabel("中心频率\n直方")
    ax_bot.set_xlabel("Frequency (MHz)")
    return figure_to_data_uri(fig)


def plot_distributions(df):
    """05 属性分布：SNR / fluence(对数) / width / bandwidth 的 2×2 直方图。"""
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE)
    ax_snr, ax_flu, ax_wid, ax_bw = axes.ravel()

    def panel(ax, col, title, color, *, logx=False):
        if col not in df.columns or not df[col].notna().any():
            add_no_data(ax)
            return
        vals = df[col].dropna()
        if logx:
            # 对数轴需要正值，且用对数分箱跨越数量级（大样本里 fluence 常如此）。
            vals = vals[vals > 0]
            if vals.empty:
                add_no_data(ax)
                return
            n_bins = min(24, max(6, int(np.sqrt(len(vals)))))
            bins = np.logspace(np.log10(vals.min()), np.log10(vals.max()), n_bins + 1)
        else:
            bins = min(20, max(6, int(np.sqrt(len(vals)))))
        draw_hist(ax, vals, color, bins=bins, logx=logx)
        ax.set_xlabel(title)
        ax.set_ylabel("Count")
        ax.set_title(title)

    # 四个协调强调色，统一柔和填充，使 2×2 读起来像一家人而非四块吵闹色块。
    panel(ax_snr, "snr", "SNR", PALETTE["blue"])
    panel(ax_flu, "fluence", "Fluence (log)", PALETTE["teal"], logx=True)
    panel(ax_wid, "width", "Width (ms)", PALETTE["violet"])
    panel(ax_bw, "bandwidth", "Bandwidth (MHz)", PALETTE["rose"])

    # 不加 suptitle：卡片本身已有「属性分布」标题，再加一层只会白占纵向空间。
    fig.tight_layout(pad=0.6)
    return figure_to_data_uri(fig)


def plot_waiting_time(df):
    """06 Waiting time：相邻 burst 时间间隔的对数直方图，标注中位数。"""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    if "toa_mjd" not in df.columns or df["toa_mjd"].notna().sum() < 2:
        add_no_data(ax, "burst 数不足，无法计算 waiting time")
        return figure_to_data_uri(fig)

    toa = df["toa_mjd"].dropna().sort_values().to_numpy(dtype=float)
    dt = np.diff(toa) * 86400.0   # MJD 差转秒
    dt = dt[dt > 0]
    if dt.size == 0:
        add_no_data(ax, "无有效 waiting time")
        return figure_to_data_uri(fig)

    n_bins = min(28, max(6, int(np.sqrt(dt.size) * 1.5)))
    bins = np.logspace(np.log10(dt.min()), np.log10(dt.max()), n_bins + 1)
    draw_hist(ax, dt, PALETTE["blue"], bins=bins, logx=True, median=True,
              median_color=PALETTE["gold"], median_label="中位数 {med:.1f} s")
    ax.set_title("Waiting time 分布")
    ax.set_xlabel("相邻 burst 间隔 (s)")
    ax.set_ylabel("Count")
    ax.legend(loc="best", frameon=False)
    return figure_to_data_uri(fig)


def plot_polarization(df):
    """偏振与 RM：仅含可靠 RM 的 burst，左 RM、右线/圆偏振分数（左右并排两连图）。"""
    data = df[df["rm_reliable"]].copy()
    fig, (ax_rm, ax_pol) = plt.subplots(1, 2, figsize=FIG_SIZE)
    fig.subplots_adjust(wspace=0.32)
    x = data["time_s"] / 60.0

    rm_err = data["rm_err"] if "rm_err" in data.columns else None
    ax_rm.errorbar(x, data["rm"], yerr=rm_err, fmt="o", color=PALETTE["blue"],
                   ecolor=PALETTE["line"], elinewidth=1.2, capsize=3,
                   markeredgecolor="white", markersize=7, zorder=3)
    ax_rm.set_ylabel("RM (rad m$^{-2}$)")
    ax_rm.set_xlabel("距离首个 burst 的时间 (min)")
    ax_rm.set_title("可靠 RM")

    lin_err = data["linear_frac_err"] if "linear_frac_err" in data.columns else None
    cir_err = data["circular_frac_err"] if "circular_frac_err" in data.columns else None
    ax_pol.errorbar(x, data["linear_frac"], yerr=lin_err, fmt="s", color=PALETTE["teal"],
                    ecolor=PALETTE["line"], elinewidth=1.1, capsize=3,
                    markeredgecolor="white", markersize=7, label="线偏振 L/I", zorder=3)
    ax_pol.errorbar(x, data["circular_frac"], yerr=cir_err, fmt="^", color=PALETTE["rose"],
                    ecolor=PALETTE["line"], elinewidth=1.1, capsize=3,
                    markeredgecolor="white", markersize=7, label="圆偏振 V/I", zorder=3)
    ax_pol.axhline(0, color=PALETTE["muted"], linewidth=0.8)
    ax_pol.set_ylabel("偏振分数 (%)")
    ax_pol.set_xlabel("距离首个 burst 的时间 (min)")
    ax_pol.set_title("线 / 圆偏振分数")
    ax_pol.legend(loc="best", frameon=False)
    return figure_to_data_uri(fig)


def plot_cumulative_fluence(df):
    """累积通量分布 N(>F)：双对数下其斜率即通量/能量分布的幂律指数（FRB 重复暴常用）。"""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    if "fluence" not in df.columns or not (df["fluence"] > 0).any():
        add_no_data(ax)
        return figure_to_data_uri(fig)

    # 把 fluence 降序排列，秩即「不小于该值的 burst 数」N(>F)。
    f = np.sort(df.loc[df["fluence"] > 0, "fluence"].dropna().to_numpy(dtype=float))[::-1]
    n = np.arange(1, f.size + 1)
    ax.fill_between(f, n, step="post", color=PALETTE["blue"], alpha=0.12, zorder=2)
    ax.step(f, n, where="post", color=PALETTE["blue"], linewidth=1.8, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    # 双对数线性拟合给出幂律斜率 α（N(>F) ∝ F^α）。
    if f.size >= 4:
        slope = float(np.polyfit(np.log10(f), np.log10(n), 1)[0])
        ax.plot([], [], " ", label=f"幂律斜率 α ≈ {slope:.2f}")
        ax.legend(loc="upper right", frameon=False)
    ax.set_title("累积通量分布 N(> F)")
    ax.set_xlabel("Fluence F (Jy ms)")
    ax.set_ylabel("N(> F)")
    return figure_to_data_uri(fig)


def plot_cumulative_count(df):
    """累积 burst 计数随时间：直观反映活动度起伏，成簇爆发会呈台阶状。"""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    if "time_s" not in df.columns or not df["time_s"].notna().any():
        add_no_data(ax)
        return figure_to_data_uri(fig)

    t = np.sort(df["time_s"].dropna().to_numpy(dtype=float)) / 60.0
    n = np.arange(1, t.size + 1)
    ax.fill_between(t, n, step="post", color=PALETTE["teal"], alpha=0.12, zorder=2)
    ax.step(t, n, where="post", color=PALETTE["teal"], linewidth=1.8, zorder=3)
    ax.set_title("累积 burst 计数")
    ax.set_xlabel("距离首个 burst 的时间 (min)")
    ax.set_ylabel("累积 burst 数")
    return figure_to_data_uri(fig)


# =========================================================================== #
# 七、HTML 片段构建（抬头探测条、画廊、指标卡、明细表、区块标题）
# =========================================================================== #
def build_hero_strip(df):
    """内联 SVG：每个 burst 一根发光竖线，x=到达时间、高度 ∝ √(S/N)。

    这是页面的标志性元素——一场观测的 burst 本就是时间上的离散事件，所以抬头直接
    把这点画出来，而不是给个干巴巴的统计数字。
    """
    if "snr" not in df.columns or not df["snr"].notna().any():
        return ""

    d = df.dropna(subset=["snr"])
    t = d["time_s"].to_numpy(dtype=float)
    s = d["snr"].to_numpy(dtype=float)

    # SVG 画布与内边距（坐标系单位，与最终像素无关，靠 viewBox 自适应缩放）。
    W, H = 1000.0, 168.0
    pad_l, pad_r, base_y, top_y = 5.0, 5.0, 146.0, 14.0
    tmin, tmax = float(t.min()), float(t.max())
    span = (tmax - tmin) or 1.0
    smax = float(s.max()) or 1.0
    xs = pad_l + (t - tmin) / span * (W - pad_l - pad_r)
    # 高度用 √(S/N) 压一下，弱事件也不至于完全看不见。
    hs = np.sqrt(np.clip(s, 0.0, None) / smax) * (base_y - top_y)

    parts = [
        f'<line class="strip-axis" x1="{pad_l:.1f}" y1="{base_y:.1f}" '
        f'x2="{W - pad_r:.1f}" y2="{base_y:.1f}"/>'
    ]
    for x, h, sv in zip(xs, hs, s):
        opacity = 0.32 + 0.68 * (sv / smax)   # S/N 越高越不透明
        parts.append(
            f'<line class="strip-mark" x1="{x:.1f}" y1="{base_y:.1f}" '
            f'x2="{x:.1f}" y2="{base_y - h:.1f}" style="opacity:{opacity:.2f}"/>'
        )
    peak = int(np.argmax(s))   # 最强事件用一个小圆点高亮
    parts.append(f'<circle class="strip-peak" cx="{xs[peak]:.1f}" cy="{base_y - hs[peak]:.1f}" r="3.4"/>')

    return (
        f'<svg class="strip" viewBox="0 0 {W:.0f} {H:.0f}" '
        f'role="img" aria-label="burst detection strip">{"".join(parts)}</svg>'
    )


def build_gallery(df, analysis_dir, top_n):
    """最高 SNR 画廊：按 S/N 取前 top_n，嵌入各自的动态谱或合成偏振图。"""
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

        # RM 可靠优先用合成偏振图，否则退回动态谱；两者都没有则显示占位。
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

        # 有图就用 :target 做纯 CSS 灯箱：点击缩略图放大到全屏遮罩、点右上角 × 关闭，
        # 复用同一张已内嵌的图（不重复体积）。缺图则只显示占位、不可点。
        label = str(row["burst_label"])
        if uri is None:
            fig_open = '<figure class="plate">'
            media = (f'<div class="plate-img"><div class="plate-missing">image not found'
                     f'<br><span>{escape(stem)}</span></div></div>')
            close = ""
        else:
            lb = f"lb-{escape(label)}"
            fig_open = f'<figure class="plate" id="{lb}">'
            media = (f'<a class="plate-zoom" href="#{lb}">'
                     f'<div class="plate-img"><img src="{uri}" alt="{escape(stem)}" loading="lazy"></div></a>')
            close = '<a class="plate-close" href="#" aria-label="关闭放大图">×</a>'

        items.append(
            f"""
            {fig_open}
              {media}
              {close}
              <figcaption class="plate-cap">
                <div class="plate-row">
                  <span class="plate-id">{escape(label)}</span>
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


def build_cards(df, overview, snr_threshold, dm_err_threshold):
    """顶部 8 格指标栏。共享统计取自 overview，卡片专属的统计在此就地计算。"""
    burst_count = overview["burst_count"]
    span_s = overview["span_s"]
    low_snr = int((df["snr"] < snr_threshold).sum()) if "snr" in df.columns else 0
    high_dm_err = int((df["dm_err"] > dm_err_threshold).sum()) if "dm_err" in df.columns else 0

    # 事件率：需要有效 TOA 且时间跨度为正才计算，否则给「—」。
    if "toa_mjd" in df.columns and df["toa_mjd"].notna().sum() >= 2 and span_s and span_s > 0:
        rate_text = fmt_value(burst_count / (span_s / 3600.0), 1)
    else:
        rate_text = "—"

    fluence_bw = fluence_bandwidth_jy_ms_ghz(df)
    if fluence_bw:
        energy_val = fmt_value(fluence_bw, 2, " Jy ms GHz")
        energy_note = "sum(fluence x bandwidth_GHz)"
    else:
        energy_val, energy_note = "-", "missing fluence / bandwidth"

    cards = [
        ("EVENTS", f"{burst_count}", f"{overview['file_count']} 个文件"),
        ("SPAN", fmt_value(overview["span_min"], 1, " min"), "按 TOA 计算"),
        ("RATE", rate_text, "events · h⁻¹"),
        ("PEAK S/N", fmt_value(overview["peak_snr"], 1), f"{low_snr} 条 < {snr_threshold:g}"),
        ("DM RANGE", stat_range(df, "dm", 1), f"{high_dm_err} 条高误差"),
        ("FLUENCE × BW", energy_val, energy_note),
        ("WIDTH", stat_range(df, "width", 1, " ms"), "fluence / peak"),
        ("RELIABLE RM", f"{overview['reliable_rm']} / {burst_count}", "达到显著性阈值"),
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


def build_detail_table(df, show_pol, print_cap=200):
    """burst 明细表。无可靠 RM 时隐去偏振相关列。

    大目录处理：网页版始终渲染全部行（可滚动浏览）；但当行数超过 print_cap 时，打印
    版只保留 S/N 最高的 print_cap 行——靠给其余行打上 print-hide 类、在 @media print
    里隐藏实现，所以网页体验完全不受影响。
    """
    columns = [c for c in DETAIL_COLUMNS if c in df.columns]
    if not show_pol:
        columns = [c for c in columns if c not in {"rm", "rm_significance", "linear_frac", "circular_frac"}]
    header = "".join(f"<th>{escape(COLUMN_LABELS.get(col, col))}</th>" for col in columns)

    # 决定打印时要保留哪些 burst_no（按 S/N 取前 print_cap）。
    truncate = len(df) > print_cap and "snr" in df.columns and df["snr"].notna().any()
    keep = set()
    if truncate:
        keep = set(df.dropna(subset=["snr"]).sort_values("snr", ascending=False)
                   .head(print_cap)["burst_no"])

    # 这些列按浮点格式化；其中部分（流量/DM 等）保留 3 位小数。
    float_cols = {"time_s", "snr", "flux_peak", "fluence", "width", "freq_low",
                  "freq_high", "bandwidth", "dm", "dm_err", "rm", "rm_significance",
                  "linear_frac", "circular_frac"}
    rows = []
    for _, row in df[columns].iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if col == "toa_mjd":
                text = fmt_value(value, 9)   # MJD 需要高精度
            elif col in float_cols:
                digits = 3 if col in {"flux_peak", "fluence", "dm", "rm_significance"} else 2
                text = fmt_value(value, digits)
            else:
                text = fmt_value(value)
            css = " muted-cell" if col in {"linear_frac", "circular_frac"} else ""
            cells.append(f'<td class="{css.strip()}">{escape(text)}</td>')
        row_cls = "print-hide" if truncate and row["burst_no"] not in keep else ""
        rows.append(f'<tr class="{row_cls}">{"".join(cells)}</tr>')

    note = ""
    if truncate:
        note = (f'<p class="cat-print-note">打印仅显示 S/N 最高的 {print_cap} 个 burst；'
                f'完整 {len(df)} 行请在网页版滚动查看。</p>')
    return f"""
    {note}
    <div class="table-wrap">
      <table>
        <thead><tr>{header}</tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """


def section_head(label, count_text):
    """各区块的小标题（◇ 标记 + 标题 + 渐隐分割线 + 右侧计数）。"""
    return (
        f'<div class="block-head"><span class="mark">◇</span>'
        f'<h2>{escape(label)}</h2><span class="rule"></span>'
        f'<span class="count">{escape(count_text)}</span></div>'
    )


# =========================================================================== #
# 八、页面样式（CSS）
# =========================================================================== #
def build_css():
    """返回整页 CSS。root 是 :root 自定义属性，body 是其余规则。"""
    root = (
        ":root{"
        "--page:#F1F4F9;--surface:#FFFFFF;--panel-2:#F5F8FC;"
        "--ink:#1B2433;--muted:#56627A;--faint:#94A1B5;"
        "--line:#E4E9F1;--line-strong:#D5DDEA;"
        "--teal:#16877A;--teal-soft:#1A917F;--blue:#2E6F95;"
        "--gold:#B9831F;--rose:#B5566E;"
        "--shadow-sm:0 1px 2px rgba(16,24,40,.04);"
        "--shadow:0 14px 32px rgba(16,24,40,.07);"
        '--font:"Inter","Noto Sans SC","Microsoft YaHei","Segoe UI",system-ui,sans-serif;'
        # A3 横向是一张 √2 比例的纸；这个屏幕宽度既保持海报比例，又留出滚动看目录的余地。
        "--sheet:1460px;"
        "}"
    )
    body = r"""
    *{ box-sizing:border-box; }
    html{ scroll-behavior:smooth; }
    body{
      margin:0; color:var(--ink); font-family:var(--font); line-height:1.4;
      font-size:12.5px; -webkit-font-smoothing:antialiased; overflow-x:hidden;
      -webkit-print-color-adjust:exact; print-color-adjust:exact;
      background:
        radial-gradient(1100px 520px at 5% -8%, rgba(22,135,122,.05), transparent 60%),
        radial-gradient(980px 600px at 104% -2%, rgba(46,111,149,.05), transparent 55%),
        var(--page);
      background-attachment:fixed;
    }
    .shell{ width:min(var(--sheet), calc(100vw - 48px)); margin:0 auto; padding:0 0 32px; }

    .topbar{
      display:flex; justify-content:space-between; align-items:center; gap:16px;
      padding:12px 2px; border-bottom:1px solid var(--line);
      font-size:10.5px; letter-spacing:.16em; font-weight:600;
      text-transform:uppercase; color:var(--muted);
    }
    .topbar .brand{ color:var(--teal); white-space:nowrap; }
    .topbar-meta{
      overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
      max-width:62%; letter-spacing:.02em; text-transform:none; direction:rtl; font-weight:400;
    }

    /* 报头：左为标题块、右为实时探测条，并排在同一带里。 */
    .hero{
      display:grid; grid-template-columns:minmax(0,1.04fr) minmax(0,1fr);
      gap:34px; align-items:end; padding:22px 2px 2px;
    }
    .hero-id{ min-width:0; }
    .hero-eyebrow{
      font-size:10.5px; letter-spacing:.3em; font-weight:700;
      text-transform:uppercase; color:var(--teal); margin-bottom:9px;
    }
    .hero-title{
      margin:0; font-weight:800;
      font-size:clamp(34px, 4.4vw, 66px); line-height:.92; letter-spacing:-.022em;
      color:var(--blue);
      background:linear-gradient(96deg,#16877A 0%, #1A7F8C 46%, var(--blue) 100%);
      -webkit-background-clip:text; background-clip:text;
      -webkit-text-fill-color:transparent;
    }
    .hero-sub{
      margin-top:11px; font-size:11.5px; letter-spacing:.04em; font-weight:500;
      font-variant-numeric:tabular-nums;
      color:var(--muted); display:flex; flex-wrap:wrap; gap:8px; align-items:center;
    }
    .hero-sub i{ color:var(--faint); font-style:normal; }
    .hero-sub b{ color:var(--ink); font-weight:700; }

    .strip-wrap{ min-width:0; }
    .strip{ width:100%; height:auto; display:block;
      filter:drop-shadow(0 2px 4px rgba(22,135,122,.18)); }
    .strip-mark{ stroke:var(--teal-soft); stroke-width:2.1; stroke-linecap:round; }
    .strip-axis{ stroke:var(--line-strong); stroke-width:1.2; }
    .strip-peak{ fill:#0E7E72; }
    .strip-legend{
      display:flex; justify-content:space-between; align-items:baseline; gap:14px;
      margin-top:7px; font-size:10px; letter-spacing:.04em; color:var(--faint);
    }
    .strip-legend .strip-note{ color:var(--muted); text-align:center; }

    /* 指标栏：在宽横版纸上挤成紧凑的一行 8 格。 */
    .rail{
      margin-top:18px; display:grid; grid-template-columns:repeat(8,minmax(0,1fr));
      background:var(--surface); border:1px solid var(--line); border-radius:12px;
      overflow:hidden; box-shadow:var(--shadow-sm), var(--shadow);
    }
    .stat{ padding:11px 13px; min-width:0; border-right:1px solid var(--line); }
    .stat:last-child{ border-right:none; }
    .stat-k{
      display:block; font-size:8.8px; letter-spacing:.13em; font-weight:700;
      text-transform:uppercase; color:var(--muted);
    }
    .stat-v{
      display:block; margin-top:6px; font-weight:700; font-variant-numeric:tabular-nums;
      font-size:18px; letter-spacing:-.01em; color:var(--ink); overflow-wrap:anywhere;
    }
    .stat-n{ display:block; margin-top:4px; font-size:9.8px; color:var(--faint); }

    .seam{
      --accent:var(--teal);
      margin-top:12px; padding:10px 14px; display:flex; gap:11px; align-items:flex-start;
      background:var(--surface); border:1px solid var(--line);
      border-left:3px solid var(--accent); border-radius:9px;
      font-size:12px; color:var(--ink); line-height:1.55; box-shadow:var(--shadow-sm);
    }
    .seam.caution{ --accent:var(--gold); }
    .seam-mark{ color:var(--accent); flex:0 0 auto; font-size:11px; padding-top:2px; }

    .block{ margin-top:24px; }
    .block-head{ display:flex; align-items:center; gap:12px; margin-bottom:12px; }
    .block-head .mark{ color:var(--teal); font-size:11px; }
    .block-head h2{
      margin:0; font-weight:700; font-size:12px;
      letter-spacing:.22em; text-transform:uppercase; color:var(--ink); white-space:nowrap;
    }
    .block-head .rule{ flex:1; height:1px;
      background:linear-gradient(90deg, var(--line-strong), rgba(213,221,234,0)); }
    .block-head .count{ font-size:10.5px; letter-spacing:.1em; font-weight:600;
      color:var(--faint); white-space:nowrap; }
    .block-note{ margin:-4px 0 12px; font-size:11.5px; color:var(--muted); }

    /* 图版网格——每行 4 个，是海报的主体表面。用 flex-wrap 而非 grid，这样不满一行的
       末行会居中（justify-content:center 只分配剩余空间，整行时无位移）。 */
    .charts{ display:flex; flex-wrap:wrap; justify-content:center; gap:11px; }
    .panel{
      width:calc((100% - 3 * 11px) / 4);   /* 每行 4 个，扣掉 3 道间距 */
      background:var(--surface); border:1px solid var(--line); border-top:2px solid var(--teal);
      border-radius:10px; padding:11px 12px 10px; min-width:0; box-shadow:var(--shadow-sm);
      display:flex; flex-direction:column;
      transition:border-color .18s ease, transform .18s ease, box-shadow .18s ease;
    }
    .panel:hover{ transform:translateY(-2px); box-shadow:var(--shadow); }
    .panel-head{ display:flex; gap:9px; align-items:baseline; margin-bottom:7px; }
    .panel-idx{ font-weight:800; font-size:11.5px; font-variant-numeric:tabular-nums;
      letter-spacing:.02em; color:var(--teal); flex:0 0 auto; }
    .panel-titles{ min-width:0; }
    .panel-titles h3{ margin:0 0 2px; font-weight:700;
      font-size:13.5px; line-height:1.18; color:var(--ink); }
    .panel-titles p{ margin:0; font-size:10.5px; color:var(--muted); line-height:1.4; }
    /* 固定 4:3 比例框让每块图版大小一致；图的白底与卡片融为一体，object-fit 留边不可见。 */
    .panel-fig{ margin-top:auto; aspect-ratio:4/3; background:var(--surface);
      border-radius:6px; overflow:hidden; }
    .panel-fig img{ display:block; width:100%; height:100%; object-fit:contain; }

    .gallery{ display:grid; grid-template-columns:repeat(auto-fill,minmax(196px,1fr)); gap:11px; }
    .plate{
      margin:0; background:var(--surface); border:1px solid var(--line); border-top:2px solid var(--gold);
      border-radius:9px; overflow:hidden; box-shadow:var(--shadow-sm);
      transition:transform .18s ease, box-shadow .18s ease;
    }
    .plate:hover{ transform:translateY(-2px); box-shadow:var(--shadow); }
    .plate-img{ background:#FBFCFE; display:block; line-height:0;
      border-bottom:1px solid var(--line); }
    .plate-img img{ display:block; width:100%; height:auto; }
    .plate-missing{ color:var(--muted); text-align:center; font-size:11px;
      padding:34px 10px; line-height:1.5; }
    .plate-missing span{ font-size:9.5px; color:var(--faint); }
    .plate-cap{ padding:9px 11px 10px; display:flex; flex-direction:column; gap:6px; }
    .plate-row{ display:flex; align-items:center; justify-content:space-between; gap:8px; }
    .plate-id{ font-weight:800; font-size:12.5px; color:var(--teal); letter-spacing:.02em;
      font-variant-numeric:tabular-nums; }
    .plate-tag{
      font-size:8.5px; letter-spacing:.13em; font-weight:700; color:var(--gold);
      border:1px solid rgba(185,131,31,.38); border-radius:999px; padding:1px 8px;
    }
    .plate-metrics{ display:flex; gap:14px; font-size:11.5px; font-weight:600;
      font-variant-numeric:tabular-nums; color:var(--ink); }
    .plate-metrics i{ color:var(--faint); font-style:normal; font-weight:500; margin-right:5px; }
    .plate-file{ font-size:9px; color:var(--faint); overflow-wrap:anywhere; line-height:1.45; }

    /* 纯 CSS 灯箱：点缩略图（:target 命中所在 figure）放大成全屏遮罩，点 × 关闭。 */
    .plate-zoom{ display:block; cursor:zoom-in; }
    .plate-close{ display:none; }
    .plate:target{
      position:fixed; inset:0; z-index:60; margin:0; padding:24px;
      border:none; border-radius:0; box-shadow:none;
      background:rgba(18,26,42,.92);
      display:flex; align-items:center; justify-content:center;
    }
    .plate:target .plate-zoom{ cursor:zoom-out; }
    .plate:target .plate-img{ background:transparent; border:none; max-width:94vw; max-height:90vh; }
    .plate:target .plate-img img{ width:auto; height:auto; max-width:94vw; max-height:90vh; object-fit:contain; }
    .plate:target .plate-cap{ display:none; }
    .plate:target .plate-close{
      display:flex; align-items:center; justify-content:center;
      position:fixed; top:18px; right:22px; width:34px; height:34px; border-radius:999px;
      background:rgba(255,255,255,.14); color:#fff; font-size:20px; text-decoration:none;
    }

    .table-wrap{ background:var(--surface); border:1px solid var(--line); border-radius:11px;
      overflow:auto; max-height:560px; box-shadow:var(--shadow-sm); }
    table{ width:100%; border-collapse:collapse; font-size:10px;
      font-variant-numeric:tabular-nums; white-space:nowrap; }
    thead th{
      position:sticky; top:0; z-index:1; background:var(--panel-2); color:var(--muted);
      font-weight:700; font-size:8.5px; letter-spacing:.06em; text-transform:uppercase;
      padding:6px 8px; text-align:right; border-bottom:1px solid var(--line-strong);
    }
    tbody td{ padding:2.4px 8px; text-align:right; color:var(--ink); line-height:1.3;
      border-bottom:1px solid #EEF1F6; }
    tbody tr:last-child td{ border-bottom:none; }
    tbody tr:hover td{ background:rgba(22,135,122,.05); }
    th:nth-child(2), td:nth-child(2){ text-align:left; max-width:300px;
      overflow:hidden; text-overflow:ellipsis; color:var(--muted); }
    .muted-cell{ color:var(--faint); }
    .cat-print-note{ display:none; }   /* 仅在打印时显示（见 @media print） */

    .foot{ margin-top:22px; padding-top:12px; border-top:1px solid var(--line);
      font-size:10px; letter-spacing:.03em; color:var(--faint);
      display:flex; justify-content:space-between; gap:18px; flex-wrap:wrap; }
    .foot span:last-child{ overflow-wrap:anywhere; }

    :focus-visible{ outline:2px solid var(--teal); outline-offset:2px; }

    /* 响应式断点都限定为 screen，使打印时始终保持满密度（8 格 / 3 列）；A3 够宽。 */
    @media screen and (max-width:1200px){
      .panel{ width:calc((100% - 11px) / 2); }   /* 每行 2 个 */
      .rail{ grid-template-columns:repeat(4,minmax(0,1fr)); }
      .stat{ border-bottom:1px solid var(--line); }
      .stat:nth-child(4n){ border-right:none; }
      .stat:nth-last-child(-n+4){ border-bottom:none; }
      .stat:last-child{ border-right:none; }
    }
    @media screen and (max-width:900px){
      .hero{ grid-template-columns:1fr; gap:18px; align-items:start; }
      .strip-wrap{ margin-top:2px; }
    }
    @media screen and (max-width:760px){
      .shell{ width:calc(100vw - 28px); }
      .panel{ width:100%; }                       /* 每行 1 个 */
      .rail{ grid-template-columns:repeat(2,minmax(0,1fr)); }
      .stat:nth-child(odd){ border-right:1px solid var(--line); }
      .stat:nth-child(2n){ border-right:none; }
      .stat:nth-last-child(-n+2){ border-bottom:none; }
    }

    /* 打印 / 「另存为 PDF」→ 带真实页边距的 A3 横向海报。 */
    @page{ size:A3 landscape; margin:9mm; }
    @media print{
      html, body{ background:#fff; }
      .shell{ width:auto; padding:0; }
      .topbar{ display:none; }
      .hero{ padding-top:4px; }
      .block{ margin-top:16px; }
      .charts{ gap:8px; }
      .gallery{ gap:8px; }
      .table-wrap{ max-height:none; overflow:visible; border-radius:8px; }
      .panel, .plate, .stat, .rail, .table-wrap, .seam{ box-shadow:none; }
      .panel, .plate, figure, tr, .seam{ break-inside:avoid; }
      .block-head{ break-after:avoid; }
      .block{ break-inside:auto; }
      /* 目录紧接画廊往下排（填满同一张纸）；逐行 break-inside:avoid 保证溢出时整行不被切断。 */
      .block-catalog{ margin-top:8px; }
      .block-catalog .block-head{ margin-bottom:7px; }
      /* 大目录：打印只留 S/N 最高的若干行（print-hide 标在其余行上），并显示截断提示。 */
      tr.print-hide{ display:none; }
      .cat-print-note{ display:block; margin:-4px 0 9px; font-size:11px; color:var(--muted); }
      /* 万一打印时灯箱处于打开状态，复位回普通流，避免遮挡整页。 */
      .plate:target{ position:static; background:none; padding:0; }
      .plate:target .plate-cap{ display:flex; }
      .plate-close{ display:none !important; }
      *{ transition:none !important; }
    }
    @media (prefers-reduced-motion:reduce){
      *{ transition:none !important; scroll-behavior:auto !important; }
    }
    """
    return root + body


# =========================================================================== #
# 九、组装整页 HTML
# =========================================================================== #
def build_html(df, csv_path, output_path, analysis_dir, metadata, args):
    setup_plot_style()
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    overview = compute_overview(df)
    show_pol = bool(df["rm_reliable"].any())   # 有无可靠 RM 决定是否展示偏振面板

    # 每个条目是 (标题, 副标题, data_uri)。六个核心面板 + 两张累积图 + 偏振图共 8~9 张，
    # 排成每行 4 个的统一网格；不满一行的末行由 CSS 居中（见 .charts 的 flex 布局）。
    charts = [
        ("Burst 时间分布", "SNR 随观测内 TOA 变化，点大小/颜色对应 fluence。", plot_timeline(df, args.snr_threshold)),
        ("DM 搜索", "空心点表示 DM 误差超过阈值；参考 DM 仅作配置标尺。", plot_dm(df, args.reference_dm, args.dm_err_threshold)),
        ("能量与宽度", "fluence、width、SNR 与 peak flux 的联合分布。", plot_flux_fluence_width(df)),
        ("频率覆盖", "占用曲线表示每个频率被多少 burst 覆盖，下方为中心频率直方图。", plot_frequency_coverage(df)),
        ("属性分布", "SNR、fluence（对数 bins）、width、bandwidth 的总体分布。", plot_distributions(df)),
        ("Waiting time", "相邻 burst 时间间隔的对数分布。", plot_waiting_time(df)),
    ]
    # 两张累积图：各有数据时才加。
    if "fluence" in df.columns and (df["fluence"] > 0).any():
        charts.append(
            ("累积通量分布", "log N(>F) 双对数；斜率即通量分布的幂律指数。", plot_cumulative_fluence(df))
        )
    if "time_s" in df.columns and df["time_s"].notna().any():
        charts.append(
            ("累积计数", "累积 burst 数随时间，成簇爆发呈台阶状。", plot_cumulative_count(df))
        )
    if show_pol:
        charts.append(
            ("偏振与 RM", "仅展示 RM 达到显著性阈值的 burst：RM、线偏振、圆偏振。", plot_polarization(df))
        )

    chart_html = "\n".join(
        f"""
        <article class="panel">
          <div class="panel-head">
            <span class="panel-idx">{i:02d}</span>
            <div class="panel-titles">
              <h3>{escape(title)}</h3>
              <p>{escape(subtitle)}</p>
            </div>
          </div>
          <div class="panel-fig"><img src="{uri}" alt="{escape(title)}" loading="lazy"></div>
        </article>
        """
        for i, (title, subtitle, uri) in enumerate(charts, 1)
    )

    gallery_html = build_gallery(df, analysis_dir, args.top_n)
    detail_table = build_detail_table(df, show_pol)
    cards = build_cards(df, overview, args.snr_threshold, args.dm_err_threshold)
    strip = build_hero_strip(df)

    # 探测条下方图例里要用到的峰值读数。
    peak_snr_txt = fmt_value(overview["peak_snr"], 1)
    peak_t_txt = fmt_value(overview["peak_time_min"], 1)

    strip_block = ""
    if strip:
        strip_block = f"""
      <div class="strip-wrap">
        {strip}
        <div class="strip-legend">
          <span>+0 min</span>
          <span class="strip-note">每条竖线 = 一个 burst · 高度 ∝ S/N · 峰值 S/N {escape(peak_snr_txt)} @ +{escape(peak_t_txt)} min</span>
          <span>+{escape(fmt_value(overview['span_min'], 1))} min</span>
        </div>
      </div>"""

    # 画廊为空（无 SNR 数据）时整节不渲染。
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

    # 偏振状态提示条：有可靠 RM 用普通样式，没有则用 caution（金色）样式。
    if show_pol:
        seam_class = "seam"
        pol_note = (
            f"发现 {overview['reliable_rm']} 个达到显著性阈值（≥ {args.rm_significance_threshold:g}）的 RM；"
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
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+SC:wght@400;500;700&display=swap" rel="stylesheet">
  <style>{css}</style>
</head>
<body>
  <main class="shell">
    <div class="topbar">
      <span class="brand">◇ BURST DOSSIER</span>
      <span class="topbar-meta">{escape(str(Path(csv_path).resolve()))}</span>
    </div>

    <header class="hero">
      <div class="hero-id">
        <div class="hero-eyebrow">FAST · L-BAND DETECTION LOG</div>
        <h1 class="hero-title">{escape(metadata['source'])}</h1>
        <div class="hero-sub">
          <span><b>{escape(metadata['date'])}</b></span><i>/</i>
          <span>BEAM <b>{escape(metadata['beam'])}</b></span><i>/</i>
          <span><b>{overview['burst_count']}</b> EVENTS</span><i>/</i>
          <span><b>{overview['file_count']}</b> FILES</span>
        </div>
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
    <section class="block block-catalog">
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


# =========================================================================== #
# 十、入口
# =========================================================================== #
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

    print(f"[OK] Dashboard saved: {output_path}")
    print(f"  bursts: {len(df)}")
    print(f"  files: {df['file_name'].nunique() if 'file_name' in df.columns else 0}")
    print(f"  reliable RM: {int(df['rm_reliable'].sum())}/{len(df)}")


if __name__ == "__main__":
    main()
