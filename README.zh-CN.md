<h1 align="center">AFTER</h1>

<div align="center">

**AI-assisted FAST Transient End-to-end Reduction**

从已确认 burst TOA 到可复核的 FAST FRB 定标测量

[![AFTER](https://img.shields.io/badge/FAST%20FRB-AFTER-1f6feb)](https://github.com/SukiYume/AFTER)
[![GitHub Stars](https://img.shields.io/github/stars/SukiYume/AFTER.svg?label=Stars&logo=github)](https://github.com/SukiYume/AFTER/stargazers)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Codex Skill](https://img.shields.io/badge/Codex%20Skill-%E5%B7%B2%E5%8C%85%E5%90%AB-2ea44f)](skills/fast-frb-observation-processing/SKILL.md)
[![Related](https://img.shields.io/badge/Search-DRAFTS-da282a)](https://github.com/SukiYume/DRAFTS)

[项目概览](#项目概览) ·
[处理流程](#after-处理流程) ·
[安装](#安装) ·
[Agent 一句话安装](#agent-一句话安装) ·
[快速开始](#快速开始) ·
[数据契约](#数据契约) ·
[English](README.md)

</div>

---

## 项目概览

**AFTER** 是一套 **AI-assisted FAST Transient End-to-end Reduction**
工作流，负责 FAST 快速射电暴（FRB）的搜索后处理。上游搜索程序或观测者提供已确认的
候选 TOA 和基本观测信息后，AFTER 将其转换为定标 H5、经过复核的 burst 区域、
物理量测量、诊断图、结果表，以及可选的 HTML 观测面板。

AFTER 覆盖候选发现之后的完整科学处理链：

1. 从原始 FAST FITS 中裁切以 burst 为中心的数据；
2. 完成流量和偏振定标，生成 Stokes I/Q/U/V；
3. 使用 AI 模型提出 burst 区域；
4. 由用户检查、接受或修正自动标记；
5. 测量 TOA、DM、RM、flux、fluence、width、bandwidth、SNR 和偏振；
6. 导出可复核的表格、诊断图和观测面板。

AFTER 与 DRAFTS 组成连续流程：DRAFTS 负责找到 transient candidates，AFTER
负责对已确认的 FAST burst 做定标、
测量和出表。

### AFTER 的特点

- **端到端搜索后处理**：从 TOA 列表一直处理到科学结果表；
- **入口灵活**：可以从原始 FITS、未定标 H5、定标 H5 或已标注 H5 开始；
- **FAST 流量与偏振定标**：结合 beam gain 和噪声管定标数据；
- **AI 辅助、人工确认**：自动框必须经过复核后才进入物理量测量；
- **明确的数据契约**：裁切、定标、标注和分析产物都有固定字段约定；
- **批量与交互兼顾**：既能处理大批次观测，也支持逐文件人工修正；
- **Codex skill 支持**：可由 agent 协助安装、自检、分阶段运行和交接复核。

## AFTER 处理流程

```mermaid
flowchart LR
    A["原始 FAST FITS<br/>+ 已确认 TOA / DM"] --> B["裁切<br/>burst-centered H5"]
    B --> C["定标<br/>Stokes I/Q/U/V"]
    C --> D["AI 检测<br/>候选 burst 区域"]
    D --> E{"人工复核"}
    E -->|"接受 / 修正"| F["H5 attrs['bursts']"]
    F --> G["分析<br/>DM / RM / flux / fluence / polarization"]
    G --> H["burst_results.csv<br/>诊断图"]
    H --> I["可选 HTML 面板"]
```

AFTER 可以从已有的最早产物继续，不要求每次都从原始 FITS 开始：

| 起点 | 必需输入 | AFTER 后续动作 |
|---|---|---|
| 原始 FAST FITS | FITS 目录、源名、日期、beam、DM、已确认 TOA 秒数 | 裁切、定标、检测、复核、分析、出表 |
| 未定标 H5 | `.h5`、匹配的 `_0001.fits`、RA/DEC、定标参考 | 定标、检测、复核、分析、出表 |
| 定标 H5 | `*_cal.h5`、检测模型、输出目录 | 检测、复核、分析、出表 |
| 已标注定标 H5 | 带 H5 attr `bursts` 的 `*_cal.h5` | 检查标记、分析、出表、生成面板 |
| 分析结果表 | `burst_results.csv` 和可选诊断图目录 | 生成或刷新观测面板 |

### 两条科学处理底线

1. **使用观测者确认或上游搜索给出的 TOA 秒数。** 每次 burst 裁切以此为准。
2. **测量前复核自动 burst 框。** analysis 使用 H5 中已接受的区域。

## 仓库结构

| 路径 | 在 AFTER 中的职责 |
|---|---|
| [`after/`](after/) | 可导入的单观测处理包；从仓库根目录使用 `python -m after.<模块>` 运行入口。 |
| [`calibration.py`](calibration.py) 与 [`cut_burst_data.py`](cut_burst_data.py) | 提供 `python calibration.py` 和 `python cut_burst_data.py` 直接命令的根入口。 |
| [`after/cut_burst_data.py`](after/cut_burst_data.py) | 根据 TOA、DM 和 beam 从原始 FAST FITS 裁切 burst-centered H5。 |
| [`after/calibration.py`](after/calibration.py) 与 [`after/calibration_noise.py`](after/calibration_noise.py) | 流量/偏振和噪声管定标、下采样、RFI mask 与定标 H5 输出。 |
| [`after/burst_detect.py`](after/burst_detect.py) | 自动、半自动或手工标记 burst 区域，写入 H5 `attrs["bursts"]`。 |
| [`after/burst_analysis.py`](after/burst_analysis.py) | 测量 DM、RM、偏振、flux、fluence、width、bandwidth 和 SNR。 |
| [`after/burst_sync_rm.py`](after/burst_sync_rm.py) | 从多个已标记的定标 H5 burst 成分合并搜索共同 RM。 |
| [`after/burst_dashboard.py`](after/burst_dashboard.py) | 从 `burst_results.csv` 生成单文件 HTML 观测面板；联网时加载 Google Fonts。 |
| [`after/burst_dm.py`](after/burst_dm.py)、[`after/burst_pol.py`](after/burst_pol.py) 与 [`after/burst_properties.py`](after/burst_properties.py) | analysis 使用的科学测量模块。 |
| [`after/rfi.py`](after/rfi.py)、[`after/obs_metadata.py`](after/obs_metadata.py) 与 [`after/zenith_angle.py`](after/zenith_angle.py) | 共用的 RFI、观测元数据和 FAST beam gain 工具。 |
| [`gain_para.csv`](gain_para.csv) | FAST beam gain 参数。 |
| [`highcal_20201014_psr_tny.npz`](highcal_20201014_psr_tny.npz) | 默认噪声管定标参考。 |
| [`models/`](models/) | 当前生产使用的 burst-region detector checkpoint。 |
| [`batch_processing/`](batch_processing/README.zh-CN.md) | 批量裁切、长周期候选裁切、旧 FITS 转换和批量定标。 |
| [`tests/`](tests/) | 覆盖定标、检测、RM 分析和 dashboard 的回归测试。 |
| [`skills/fast-frb-observation-processing/`](skills/fast-frb-observation-processing/) | Codex 使用 AFTER 的操作协议。 |
| [`requirements.txt`](requirements.txt) | Python 依赖清单。 |

运行资源保留在仓库根目录，并由 [`after/__init__.py`](after/__init__.py) 根据代码位置
解析，因此资源查找与执行命令时的工作目录无关：

- gain 定标：`gain_para.csv`；
- 默认噪声管定标表：`highcal_20201014_psr_tny.npz`；
- 默认 detector checkpoint：`models/best_model_yolo11n_ema.pth`。

两个根入口转发到 `after` 包。单观测配置常量放在 `after/calibration.py` 或
`after/cut_burst_data.py` 中；配置后可使用根命令或对应的
`python -m after.<模块>` 命令。

## 安装

Linux/macOS：

```bash
: "${AFTER_REPOSITORY_URL:?请设置代码仓库地址}"
git clone "$AFTER_REPOSITORY_URL" AFTER
cd AFTER
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Windows PowerShell：

```powershell
if (-not $env:AFTER_REPOSITORY_URL) { throw "请先设置 AFTER_REPOSITORY_URL" }
git clone $env:AFTER_REPOSITORY_URL AFTER
cd AFTER
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

需要 GPU detection 时，应根据目标 CUDA 驱动安装匹配的 `torch` 和
`torchvision`。AFTER 不绑定特定主机、GPU 型号或驱动版本；应在目标环境验证 import
和 CUDA tensor 运算，并把实际运行环境随结果保存。`requirements.txt` 有意不锁定
GPU wheel。

核心依赖包括 NumPy、SciPy、h5py、Astropy、Matplotlib、pandas、Seaborn、Numba、
OpenCV、PyTorch、torchvision 和 Ultralytics。

### 安装后自检

```bash
python -m compileall -q after batch_processing tests calibration.py cut_burst_data.py
python -c "from calibration import process_one_burst; from cut_burst_data import cut_one_burst; print('compatibility imports OK')"
python batch_processing/batch_cut_burst_data.py --help
python batch_processing/batch_calibration.py --help
python -m after.burst_detect --help
python -m after.burst_analysis --help
python -m after.burst_dashboard --help
python -m pytest -q
```

## Agent 一句话安装

AFTER 自带一份可供 Codex 和其他能够读取仓库的 coding agent 使用的操作协议。把下面这
一整句直接发给 agent：

```text
请安装并配置 AFTER 仓库：克隆或打开该仓库，把 skills/fast-frb-observation-processing 安装到当前 agent 的 skills 目录（如果不支持自定义 skill，就直接读取其中的 SKILL.md 作为操作协议），将 DATA_PROCESSING_ROOT 设置为仓库根目录，执行 README 中的安装后自检，并在处理真实观测数据前逐项报告自检结果。
```

Codex 使用的 skill 位于：

```text
skills/fast-frb-observation-processing/
```

Bash 手动安装：

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
cp -R skills/fast-frb-observation-processing \
  "${CODEX_HOME:-$HOME/.codex}/skills/"
export DATA_PROCESSING_ROOT="$(pwd)"
```

Windows PowerShell：

```powershell
$codexRoot = if ($env:CODEX_HOME) {
    $env:CODEX_HOME
} else {
    Join-Path $HOME ".codex"
}
New-Item -ItemType Directory -Force (Join-Path $codexRoot "skills") | Out-Null
Copy-Item -Recurse -Force `
  .\skills\fast-frb-observation-processing `
  (Join-Path $codexRoot "skills")
$env:DATA_PROCESSING_ROOT = (Get-Location).Path
```

如果后续任务也要让 agent 自动定位 AFTER，应把 `DATA_PROCESSING_ROOT` 持久化到 shell
profile 或系统环境变量。

## 快速开始

以下命令都应在仓库根目录执行。示例只使用通用路径，请把 `/path/to/...` 替换为自己的
工作站或计算节点路径。

四个批处理入口的输入表格式、输出布局和重跑参数集中记录在
[`batch_processing/README.zh-CN.md`](batch_processing/README.zh-CN.md)。

### 1. 裁切原始 FAST FITS

使用单观测常量配置时，先编辑 `after/cut_burst_data.py` 底部的配置区，然后运行：

```bash
python cut_burst_data.py
```

批量入口：

```bash
python batch_processing/batch_cut_burst_data.py \
  --burst-txt /path/to/catalogs/FRBXXXX_Burst.txt \
  --output-root /path/to/after_runs/cut/FRBXXXX \
  --save-frb-name FRBXXXX \
  --segment-length 65536 \
  --workers 8
```

可复制 [`batch_processing/Burst.example.txt`](batch_processing/Burst.example.txt)，
或创建带下列表头的空白字符分隔文件：

```text
base project name date beam dm time
```

脚本按原始数据路径、日期、beam 和 DM 分组，复制定标所需的第一个匹配 beam FITS，
逐个裁切已提供的 TOA，并写出 `obs_info.json`。若同一日期目录混有不同 beam、DM 或
segment length，顶层字段会写成列表，每个 burst 条目也保留自身取值。

带逐行 segment length 的长周期候选：

```bash
python batch_processing/batch_cut_selected_long_period.py \
  --plan-txt /path/to/catalogs/Selected_LongPeriod_Burst.txt \
  --output-root /path/to/after_runs/long_period_cut \
  --workers 8
```

### 2. 转换旧版 burst FITS

旧 burst cut 需要转为当前 H5 schema 时：

```bash
python batch_processing/fits_to_h5.py \
  --asd-root /path/to/legacy_burst_data \
  --output-root /path/to/after_runs/cut \
  --catalog-dir /path/to/catalogs
```

脚本会复制匹配的 `_0001.fits` 定标文件，并写出与 `after.cut_burst_data` 兼容的 H5。

### 3. 定标

使用单观测常量配置时，先编辑 `after/calibration.py` 底部的 `BURST_DIR`、
`OUTPUT_DIR`、RA/DEC、分辨率和进程数，然后运行：

```bash
python calibration.py
```

默认 `CAL_NPZ` 已经指向仓库根目录的 `highcal_20201014_psr_tny.npz`。

```bash
python batch_processing/batch_calibration.py \
  --root-dir /path/to/after_runs/cut \
  --cal-root /path/to/after_runs/calibrated \
  --dm-file /path/to/catalogs/h5_calibration_dm_file.txt \
  --cal-npz highcal_20201014_psr_tny.npz \
  --workers 8
```

定标目录表格式：

```text
FRB_name DM RA DEC
```

常用保存分辨率：

- 不传 `--down-time` 和 `--down-freq`：自动选择适合画图的分辨率；
- `--down-time 1`：保留原始时间分辨率，用于 peak-flux 对比；
- `--down-freq 1`：保留原始频率通道，用于频谱和 RFI 细查。

每个 burst beam 使用同 beam 的 `Mxx..._0001.fits`，缺少时停止该 beam 组。批处理
默认使用 FFT RFI；传入 `--no-rfi-fft` 可选择熵方法。
输出写到 `<cal-root>/<FRB>/<date>/`。

### 4. 检测并复核 burst 区域

自动模式：

```bash
python -m after.burst_detect \
  --mode auto \
  --cal-dir /path/to/after_runs/calibrated \
  --model-path models/best_model_yolo11n_ema.pth \
  --model-name yolo11n \
  --output-dir /path/to/after_runs/detections
```

检测阶段写出：

- H5 `attrs["bursts"]`：analysis 使用的标记来源；
- `detections.json`：以相对 `--cal-dir` 的路径为键的续跑与复核记录；
- `plots/*_det.png`：带已接受区域的复核图。

自动和半自动模式直接使用定标后的 Stokes I 推理一次。确认 burst 框后，AFTER 使用
非 burst 样本按 analysis 相同的 Stokes-I/V 并集方法重算 RFI，写入
`burst_rfi_*`，并保存最终 masked residual 图。

置信度过滤后，`--max-horizontal-aspect` 会移除过宽的横向框（默认 `3`）；存在正面积
重叠的多个框只保留面积最大的区域。

自动标记需要修正时使用 `--mode semi-auto`；模型建议明显无用时使用
`--mode manual`。交互界面中，`x` 会记录明确的空 burst 列表；`q` 或 `Esc` 会保存
已完成进度并退出，不会把当前文件误标为完成。

### 5. 分析物理量

```bash
python -m after.burst_analysis \
  --cal-dir /path/to/after_runs/calibrated \
  --output-dir /path/to/after_runs/analysis \
  --dm-range 5 \
  --dm-step 0.1 \
  --rm-min -1000 \
  --rm-max 1000 \
  --seed 42
```

测量内容包括 TOA、peak flux、fluence、width、burst bandwidth、SNR、DM、RM、线偏振、
圆偏振、总偏振、PA 和 PAV。每行 CSV 保存 bootstrap seed 和次数。使用不同 DM/RM
范围重跑时，应写入独立输出目录。`--cal-dir` 会递归查找 `*_cal.h5`，因此可以直接
指向批量定标产生的 `<cal-root>`。

主要输出：

```text
burst_results.csv
DM / RM / polarization 诊断图
```

如果需要直接从多个已标记、已定标的 H5 搜索共同 RM：

```bash
python -m after.burst_sync_rm \
  --cal-dir /path/to/after_runs/calibrated_df1 \
  --output-dir /path/to/after_runs/joint_rm \
  --rm-min -50000 \
  --rm-max 50000 \
  --test-window full:-50000:50000 \
  --test-window expected:30000:40000 \
  --n-null 1000
```

`after.burst_sync_rm` 同时输出逐时间采样 PA 无关的主统计量
`time_pa_power`，以及把每个 burst 的 RM–Linear-Degree 曲线稳健标准化后
按固定权重合并的 `linear_degree_stack`。脚本直接读取 `attrs["bursts"]`，
只根据 Stokes I 选最强时间采样，并使用所有通道级 RFI mask 的并集；不会
应用时间–频率 pixel mask。off-pulse 检验保持每个 burst 的时间采样数和
通道 mask 不变，并校正每个预先声明 RM 窗口内的 look-elsewhere effect。
analysis 和联合搜索都根据实际有效 λ² 覆盖对应的最窄 RMSF 自动计算 RM
步长；用户只指定 `--rm-min` 和 `--rm-max`。

### 6. 生成观测面板

```bash
python -m after.burst_dashboard \
  --csv /path/to/after_runs/analysis/burst_results.csv \
  --output /path/to/after_runs/analysis/burst_dashboard.html \
  --analysis-dir /path/to/after_runs/analysis \
  --source FRBNAME \
  --date YYYYMMDD \
  --reference-dm 539 \
  --rm-significance-threshold 5 \
  --top-n 10
```

生成结果是可本地打开、也可打印为 PDF 的单文件 HTML。图表内嵌，Google Fonts 联网
加载，并配置本地字体回退。

## 数据契约

### 未定标 H5

```text
data: (nsamp, npol, nchan)
freq: (nchan,), MHz
attrs: start_sample, file_mjd, toa_sec, time_reso, npol, nchan,
       segment_length, obs_start_mjd, beam, dm
```

切片延伸到观测起点之前时，`start_sample` 保留负值；文件名使用 `n` 加九位绝对值，
例如 `n000000002`。

### 定标 H5

```text
data:        (4, nsamp, nchan), Stokes I/Q/U/V, Jy
freq:        (nchan,), MHz
rfi_mask:    (nsamp, nchan), bool
rfi_channel: (nchan,), bool
gain:        (nchan,), K/Jy
gain_err:    (nchan,), K/Jy
attrs: time_reso_raw, time_reso, down_time, down_freq,
       dm, beam, ra, dec, calibration_beam,
       calibration_fits, calibration_npz
```

### 已接受 burst 区域

```json
{
  "time_start": 120,
  "time_end": 180,
  "freq_start": 40,
  "freq_end": 500,
  "confidence": 0.82
}
```

明确判定为无 burst 的页面使用空列表记录，而不是缺失复核状态。

### 分析 CSV

每个 DM 和偏振测量都带显式状态：

```text
dm, dm_err, dm_status, dm_error_reason
rm, rm_err, ..., pol_status, pol_error_reason
```

成功行使用 `ok` 和空原因；搜索失败则使用 `failed`、保存异常类型/消息，并把科学量
写为 NaN。因此，失败拟合不会再伪装成测得的零值或裁切时的名义 DM。

## 模型与运行结果

AFTER 默认使用 `models/best_model_yolo11n_ema.pth` 进行 burst 检测。比较或更新
detector 时，可以通过 `--model-path` 指定其他兼容 checkpoint。

仓库内二进制资产使用内容哈希标识：

| 资产 | 已核验内容与 provenance | SHA-256 |
|---|---|---|
| `models/best_model_yolo11n_ema.pth` | 与 YOLO11n 兼容、含 499 项的 `OrderedDict` state dict；哈希标识准确 checkpoint。 | `9BEEF810651B7B4B793A0DD85DFBB0E0959406BAE4B8D322313C841791E830FA` |
| `highcal_20201014_psr_tny.npz` | 19 beam 定标表，包含 `freq (4096,) float32` 和 `tcal (4096, 2, 19) float64`。 | `4FC36ACC2E639962B2A10C7F81803FA88C93F4F85B33D07D657ABC40CD410F66` |

一次完整运行可以产生：

- 裁切和定标后的 H5；
- `detections.json` 和 burst 复核图；
- `burst_results.csv` 与 DM/RM/偏振诊断图；
- 单文件 `burst_dashboard.html`。

每个观测或参数重跑都应使用独立输出目录，避免重新标注、DM/RM 扫描或刷新面板时
无意覆盖旧结果。

## 许可

AFTER 采用 [MIT License](LICENSE)。

## DRAFTS 与 AFTER

```text
DRAFTS：暂现源搜索与候选筛选
    -> 已确认 source / date / beam / TOA / DM
AFTER：裁切、定标、复核、测量和出表
```

需要从观测数据中寻找候选时使用 DRAFTS；候选列表已经确定、目标是做 FAST
定标和物理量分析时使用 AFTER。

---

<div align="center">
  <sub>AFTER · From confirmed FAST transients to calibrated measurements</sub>
</div>
