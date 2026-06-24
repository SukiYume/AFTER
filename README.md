# AFTER

**AI-assisted FAST Transient End-to-end Reduction**

AFTER is an AI-assisted post-search workflow for FAST fast radio burst (FRB) observations. It starts from a known source, observation date, beam, DM, and user-provided burst TOA list, then guides the observation through TOA-based cutting, calibration, burst-label review, energy/polarization/DM/RM analysis, and final table export.

```text
AI-assisted post-search FAST FRB burst processing: TOA-guided cutting, calibration,
label review, energy/polarization analysis, and results-table export.
```

## Language

- [中文说明](#中文说明)
- [English Version](#english-version)

## 中文说明

### 项目定位

AFTER 面向“搜索候选已经确认”之后的 FAST FRB burst 数据处理。它接续搜索 pipeline 或上游流程给出的候选源、日期、beam、DM 和 TOA 秒数列表，负责把这些输入推进到可检查、可复现的后处理结果：

```text
原始 FAST FITS + 用户提供 TOA 列表
  -> 裁切 burst H5
  -> 偏振和流量定标
  -> 自动爆发探测
  -> 人工检查或修正标记
  -> 能量/偏振/DM/RM 分析
  -> burst_results.csv + 诊断图
```

当前 GitHub 首发只公开 `README.md`、`.gitignore`、`requirements.txt` 和 `skills/`。完整运行 AFTER 仍需要本地或内部的处理脚本、FAST 数据、定标文件和模型权重。这样可以先让别人安装 Codex skill 和理解流程，代码和资源文件稍后再公开。

### 当前发布内容

| 路径 | 当前是否发布到 GitHub | 作用 |
|---|---:|---|
| `README.md` | 是 | AFTER 的中英文说明、安装、自检、流程和数据约定。 |
| `.gitignore` | 是 | 忽略 H5/FITS、诊断图、检测输出、本地观测清单和旧权重。 |
| `requirements.txt` | 是 | Python 依赖清单；GPU 机器上仍需按 CUDA 单独选择 PyTorch 安装源。 |
| `skills/fast-frb-observation-processing/` | 是 | Codex agent 使用 AFTER 的操作协议。 |
| `cut_burst_data.py`、`calibration.py`、`burst_detect.py`、`burst_analysis.py` | 暂未发布 | AFTER 的核心处理脚本。 |
| `gain_para.csv`、`highcal_*.npz`、`models/*.pth` | 暂未发布 | FAST gain、噪声管定标文件和检测模型权重。 |
| `batch_processing/` | 暂未发布 | 批量裁切、旧 FITS 转 H5 和批量定标包装脚本。 |

### AFTER 的逻辑流程

完整流程是：

```text
切数据 -> 定标 -> 爆发探测 -> 人工检查标记 -> 能量/偏振分析 -> 出表
```

AFTER 不要求每次从第一步开始。可以根据已有数据选择入口：

| 起点 | 必需输入 | AFTER 后续动作 |
|---|---|---|
| 原始 FAST FITS | 原始 FITS 目录、源名、日期、beam、DM、用户提供的 TOA 秒数列表 | 切数据、定标、爆发探测、人工检查标记、能量/偏振分析、出表 |
| 未定标 H5 | `cut_burst_data.py` 输出的 `.h5` 和同日期 `_0001.fits` | 定标、爆发探测、人工检查标记、能量/偏振分析、出表 |
| 定标后 H5 | `*_cal.h5`，以及 `data`、`freq`、`rfi_mask`、`gain`、`gain_err` 等字段 | 爆发探测、人工检查标记、能量/偏振分析、出表 |
| 已有 burst 标记的定标 H5 | `*_cal.h5` 且 H5 attrs 中已有 `bursts` | 验证标记、能量/偏振分析、出表 |

两条规则最重要：

1. 切数据阶段不能凭图、文件名或猜测生成 TOA。TOA 秒数必须由用户或上游搜索流程提供。
2. 自动爆发探测的框只是候选标记。进入能量和偏振分析前，必须由用户检查自动标记，必要时用半自动或手工模式修正。

### 安装 Python 环境

从完整 AFTER 脚本仓库根目录安装：

```bash
git clone <repo-url> AFTER
cd AFTER
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Windows PowerShell：

```powershell
git clone <repo-url> AFTER
cd AFTER
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

如果需要 GPU 推理，请按本机 CUDA 和驱动版本从 PyTorch 官方安装页选择 `torch` / `torchvision` 安装命令。`requirements.txt` 中保留未固定版本，适合先建立可运行环境；正式批处理建议记录实际 Python、CUDA、PyTorch 和 ultralytics 版本。

核心依赖包括 `numpy`、`scipy`、`h5py`、`astropy`、`matplotlib`、`pandas`、`seaborn`、`numba`、`opencv-python`、`torch`、`torchvision` 和 `ultralytics`。

### 安装 Codex Skill

AFTER 自带 Codex skill：

```text
skills/fast-frb-observation-processing/
```

可以直接让 Codex agent 安装：

```text
请帮我安装本仓库的 Codex skill：复制 skills/fast-frb-observation-processing 到 Codex skills 目录；如果本机已有 AFTER/data_processing 脚本仓库，请把 DATA_PROCESSING_ROOT 设置为该脚本根目录，并完成安装后自检。
```

手动安装：

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
cp -R skills/fast-frb-observation-processing "${CODEX_HOME:-$HOME/.codex}/skills/"
export DATA_PROCESSING_ROOT="$(pwd)"
```

Windows PowerShell：

```powershell
$codexHome = if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $HOME ".codex" }
New-Item -ItemType Directory -Force (Join-Path $codexHome "skills") | Out-Null
Copy-Item -Recurse -Force .\skills\fast-frb-observation-processing (Join-Path $codexHome "skills")
$env:DATA_PROCESSING_ROOT = (Get-Location).Path
```

`DATA_PROCESSING_ROOT` 只对当前 shell 生效。长期使用时，把它写入 shell profile 或系统环境变量。skill 可以单独安装；运行 AFTER 处理脚本时仍需要 `DATA_PROCESSING_ROOT` 指向完整脚本根目录。

### 安装后自检

在完整 AFTER 脚本根目录中做语法、依赖、CLI 和 skill 检查：

```bash
python -m py_compile cut_burst_data.py calibration.py burst_detect.py burst_analysis.py burst_dm.py burst_pol.py burst_properties.py rfi_utils.py ZeithAngle.py batch_processing/batch_calibration.py batch_processing/batch_cut_burst_data.py batch_processing/batch_cut_selected_long_period.py batch_processing/fits_to_h5.py
python -c "import numpy, scipy, h5py, astropy, matplotlib, pandas, seaborn, numba, torch, torchvision, ultralytics, cv2; print('basic imports OK')"
python burst_detect.py --help
python burst_analysis.py --help
```

如果本机有 Codex skill authoring 工具，再验证 skill 结构：

```bash
python /path/to/quick_validate.py skills/fast-frb-observation-processing
```

Codex agent 使用 AFTER skill 时，会先确认脚本根目录包含：

```text
cut_burst_data.py
calibration.py
burst_detect.py
burst_analysis.py
rfi_utils.py
ZeithAngle.py
gain_para.csv
```

### 输入文件和资源文件

原始 FITS 入口需要 TOA 秒数列表，单位是相对观测开始的秒。AFTER 可以帮助整理、去重和检查范围，但不会自行生成 TOA。

批量处理时，本地观测清单通常使用 `batch_processing/*.txt`。这些文件可能包含本地路径、项目号或未公开观测信息，所以默认被 `.gitignore` 忽略。需要提交模板时，请使用 `*.example.txt` 文件名。

常见批量裁切表 `FRB*_Burst.txt` 的列顺序：

```text
base project name date beam dm time
```

常见批量定标表 `h5_calibration_dm_file.txt` 的列顺序：

```text
FRB_name DM RA DEC
```

完整运行 AFTER 还需要：

- `gain_para.csv`：FAST beam gain 参数表。
- `highcal_20201014_psr_tny.npz` 或等价噪声管定标文件。
- `models/best_model_yolo11n_ema.pth` 或用户指定的检测模型权重。

### 阶段 1：切数据

脚本：`cut_burst_data.py`

输入：

- 原始 FAST FITS 目录。
- 源名、观测日期、beam、DM。
- 用户或上游搜索流程提供的 TOA 秒数列表。
- 输出目录和可选的 segment length / worker count。

输出：

```text
{frb}-{date}-M{beam:02d}-{fits_number:04d}-{start_sample:09d}.h5
obs_info.json
```

每个未定标 H5 至少包含：

```text
data: (nsamp, npol, nchan)
freq: (nchan,), MHz
attrs: start_sample, file_mjd, toa_sec, time_reso, npol, nchan,
       segment_length, obs_start_mjd, dm
```

### 阶段 2：定标

脚本：`calibration.py`

输入：

- 未定标 H5 目录。
- 同日期目录中的 beam 对应 `_0001.fits`。
- RA、DEC、beam、噪声管定标文件。
- 下采样策略和 RFI 策略。

输出：

```text
*_cal.h5
quick-look .jpg
```

定标后 H5 至少包含：

```text
data:        (4, nsamp, nchan), Stokes I/Q/U/V, Jy
freq:        (nchan,), MHz
rfi_mask:    (nsamp, nchan), bool
rfi_channel: (nchan,), bool
gain:        (nchan,), K/Jy
gain_err:    (nchan,), K/Jy
```

常用下采样规则：

- `down_time=None`、`down_freq=None`：保存为自动选择的画图/检测友好分辨率。
- `down_time=1`：保留原始时间分辨率，适合 peak flux 对比。
- `down_freq=1`：保留原始频率通道，适合细致 RFI 或频谱检查。

### 阶段 3：爆发探测和人工检查

脚本：`burst_detect.py`

自动模式示例：

```bash
python burst_detect.py \
  --mode auto \
  --cal-dir /path/to/calibrated_h5 \
  --model-path models/best_model_yolo11n_ema.pth \
  --model-name yolo11n \
  --output-dir /path/to/detections_auto
```

检测结果写入两个地方：

- H5 attrs：`attrs["bursts"]`，这是后续分析读取的真值来源。
- 输出目录：`detections.json` 和 `plots/*_det.png`，用于跳过已处理文件、复核和追溯。

自动标记后，AFTER 应生成或维护一份 review manifest，列出需要用户检查的图和原因。进入分析前，用户需要确认自动框可接受，或完成半自动/手工修正。

如果某页重复、低 SNR 或不应进入分析，请在交互窗口中用 `x` 写入：

```json
{"bursts": [], "has_burst": false}
```

只改 `detections.json` 不足以剔除已经写入 H5 attrs 的旧标记。

### 阶段 4：能量/偏振分析和出表

脚本：`burst_analysis.py`

示例：

```bash
python burst_analysis.py \
  --cal-dir /path/to/calibrated_h5 \
  --output-dir /path/to/analysis_output \
  --dm-range 5 \
  --dm-step 0.1 \
  --rm-min -1000 \
  --rm-max 1000 \
  --n-rm 100000
```

`burst_analysis.py` 读取已确认的 `attrs["bursts"]`，逐个 burst 计算：

- TOA、peak flux、fluence、width、bandwidth、SNR。
- DM、RM。
- 线偏振、圆偏振、总偏振比例。
- PA 和 PAV。

主要输出：

```text
burst_results.csv
DM/RM/polarization diagnostic plots
```

### 运行产物和忽略规则

`.gitignore` 默认忽略：

- H5/FITS 数据：`*.h5`、`*.fits`、`*_cal.h5`。
- 图像和诊断图：`*.jpg`、`*.png`。
- 检测和分析产物：`detections/`、`analysis_output/`、`analysis_outputs/`。
- 本地批量输入表：`batch_processing/*.txt`，但保留 `*.example.txt`。
- 本地旧权重：`models/*.old`。

## English Version

### What AFTER Does

AFTER is a post-search FAST FRB burst processing workflow. It assumes the upstream search or the user already knows the candidate source, observation date, beam, DM, and burst TOAs. AFTER then organizes the reduction and analysis steps:

```text
raw FAST FITS + user-provided TOA list
  -> cut burst H5 files
  -> polarization and flux calibration
  -> automatic burst detection
  -> human label review or correction
  -> energy/polarization/DM/RM analysis
  -> burst_results.csv + diagnostic plots
```

The current public GitHub upload contains `README.md`, `.gitignore`, `requirements.txt`, and the Codex skill under `skills/`. The processing scripts, model weights, calibration assets, example data, and observation catalogs are not included in this first public upload. Running AFTER still requires a complete local script checkout and the required FAST data products.

### Workflow Entry Points

The full AFTER sequence is:

```text
cut -> calibrate -> detect -> review labels -> analyze energy/polarization -> export table
```

You can start from any available stage:

| Starting point | Required inputs | Next AFTER steps |
|---|---|---|
| Raw FAST FITS | FITS directory, source, date, beam, DM, user-provided TOA seconds | Cut, calibrate, detect, review labels, analyze, export table |
| Cut H5 | H5 files from `cut_burst_data.py` and matching `_0001.fits` | Calibrate, detect, review labels, analyze, export table |
| Calibrated H5 | `*_cal.h5` with `data`, `freq`, `rfi_mask`, `gain`, `gain_err` | Detect, review labels, analyze, export table |
| Detected calibrated H5 | `*_cal.h5` with existing H5 attr `bursts` | Verify labels, analyze, export table |

Two constraints define the workflow:

1. AFTER must not invent TOAs from plots, filenames, or visual guesses. Raw-FITS cutting requires a TOA list from the user or an upstream search product.
2. Automatic detection boxes are provisional. Energy and polarization analysis should start only after the user accepts or corrects the burst labels.

### Install Python Dependencies

From a complete AFTER script checkout:

```bash
git clone <repo-url> AFTER
cd AFTER
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
git clone <repo-url> AFTER
cd AFTER
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

For GPU inference, install `torch` and `torchvision` with the PyTorch command matching the machine's CUDA and driver versions. Record the actual Python, CUDA, PyTorch, and ultralytics versions for production batch runs.

### Install the Codex Skill

AFTER ships a Codex skill:

```text
skills/fast-frb-observation-processing/
```

One-line request for a Codex agent:

```text
Please install the Codex skill from this repository: copy skills/fast-frb-observation-processing into the Codex skills directory; if this machine has the full AFTER/data_processing script checkout, set DATA_PROCESSING_ROOT to that script root and run the post-install validation.
```

Manual install:

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
cp -R skills/fast-frb-observation-processing "${CODEX_HOME:-$HOME/.codex}/skills/"
export DATA_PROCESSING_ROOT="$(pwd)"
```

The skill can be installed by itself, but processing commands require `DATA_PROCESSING_ROOT` to point to a complete AFTER script root.

### Post-install Validation

Run these checks from a complete AFTER script root:

```bash
python -m py_compile cut_burst_data.py calibration.py burst_detect.py burst_analysis.py burst_dm.py burst_pol.py burst_properties.py rfi_utils.py ZeithAngle.py batch_processing/batch_calibration.py batch_processing/batch_cut_burst_data.py batch_processing/batch_cut_selected_long_period.py batch_processing/fits_to_h5.py
python -c "import numpy, scipy, h5py, astropy, matplotlib, pandas, seaborn, numba, torch, torchvision, ultralytics, cv2; print('basic imports OK')"
python burst_detect.py --help
python burst_analysis.py --help
```

If a Codex skill authoring validator is available:

```bash
python /path/to/quick_validate.py skills/fast-frb-observation-processing
```

### Stage Summary

1. **Cut raw FITS** with `cut_burst_data.py`. Use only user-provided or upstream-provided TOA seconds. The output is cut H5 plus `obs_info.json`.
2. **Calibrate** with `calibration.py`. Use the matching `_0001.fits`, RA/DEC, FAST gain parameters, and a noise-calibration file. The output is `*_cal.h5` plus quick-look plots.
3. **Detect and review bursts** with `burst_detect.py`. Automatic labels are written to H5 attrs and `detections.json`, then reviewed by the user. H5 `attrs["bursts"]` is the source of truth for analysis.
4. **Analyze and export** with `burst_analysis.py`. Confirmed burst labels drive DM/RM, energy, and polarization measurements. The main table is `burst_results.csv`.

### Data and Output Policy

Local observation catalogs under `batch_processing/*.txt`, FITS/H5 data, generated plots, detection outputs, analysis outputs, and retired model checkpoints are intentionally ignored by git. Commit only source, skill files, templates, or explicit public examples.
