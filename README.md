# AFTER

**AI-assisted FAST Transient End-to-end Reduction**

AFTER 是一套面向 FAST FRB burst 的搜索后端到端处理流程。它从用户或上游搜索流程提供的 TOA 列表开始，完成裁切、定标、爆发区域标记和人工复核、能量/偏振/DM/RM 分析，并导出结果表和诊断图。

GitHub description:

```text
AI-assisted post-search FAST FRB burst processing: TOA-guided cutting, calibration, label review, energy/polarization analysis, and results-table export.
```

## FAST FRB Burst 数据处理指南

本目录是一套面向 FAST FRB burst 的后处理脚本。它处理“搜索候选已经确认，已知源名、日期、beam、DM、TOA”之后的工作：从原始观测中裁切 burst，做偏振和流量定标，标记 burst 区域，计算 DM、RM、偏振、flux、fluence、width、bandwidth 等参数。

## 总体流程

```text
原始观测 FITS
  + 用户提供 TOA 列表
  -> cut_burst_data.py
  -> 未定标 burst H5
  -> calibration.py
  -> 定标后 *_cal.h5
  -> burst_detect.py
  -> 人工检查或修正自动标记
  -> attrs["bursts"] / detections.json
  -> burst_analysis.py
  -> 能量/偏振/DM/RM 等结果
  -> burst_results.csv / 诊断图
```

根目录脚本适合单源、单日期、手动或半自动处理。`batch_processing/` 适合批量读取表格并处理多个日期。

## 起点选择

完整流程是“切数据 -> 定标 -> 爆发探测 -> 能量/偏振分析 -> 出表”，但不要求每次都从切数据开始。根据已有数据，可以从以下入口开始：

| 起点 | 需要已有内容 | 后续流程 |
|---|---|---|
| 原始 FAST FITS | 原始 FITS 目录、源名、日期、beam、DM、用户提供的 TOA 秒数列表 | 切数据、定标、爆发探测、人工检查标记、能量/偏振分析、出表 |
| 未定标 H5 | `cut_burst_data.py` 输出的 `.h5` 和同日期 `_0001.fits` | 定标、爆发探测、人工检查标记、能量/偏振分析、出表 |
| 定标后 H5 | `*_cal.h5`、`freq`、`gain`、`rfi_mask` 等定标产物 | 爆发探测、人工检查标记、能量/偏振分析、出表 |
| 已有 burst 标记的定标 H5 | `*_cal.h5` 且 H5 attrs 中已有 `bursts` | 验证标记、能量/偏振分析、出表 |

切数据阶段不能凭图或文件名猜 TOA；TOA 秒数列表必须由用户或上游搜索流程提供。爆发探测阶段的自动标记不是最终结果，进入分析前必须人工检查标记质量，必要时用半自动或手工模式修正。

## 文件职责

| 文件或目录 | 作用 |
|---|---|
| `cut_burst_data.py` | 从原始观测 FITS 中按 TOA、DM、beam 裁切 burst，输出未定标 H5。 |
| `calibration.py` | 对未定标 H5 做偏振和流量定标、下采样、RFI 标记和 quick-look 图。 |
| `burst_detect.py` | 用 YOLO 或交互方式标记 burst 区域，把结果写回 H5 attrs。 |
| `burst_analysis.py` | 读取已标记的定标 H5，计算 DM、RM、偏振和能量相关参数。 |
| `burst_dm.py` | DM 精细搜索模块，由 analysis 调用。 |
| `burst_pol.py` | RM、偏振、PA 和 PAV 处理模块，由 analysis 调用。 |
| `burst_properties.py` | flux、fluence、width、bandwidth、SNR 等测量模块。 |
| `rfi_utils.py` | calibration 和 analysis 共用的 RFI 通道/像素标记函数。 |
| `ZeithAngle.py` | 根据 MJD、坐标和 beam 计算 FAST ZA 与 gain。 |
| `gain_para.csv` | FAST beam gain 参数表。 |
| `highcal_20201014_psr_tny.npz` | 定标所需噪声管参考文件。 |
| `models/` | burst 区域检测模型权重。 |
| `batch_processing/` | 批量裁切、旧 FITS 转 H5、批量定标包装脚本；实际输入表由用户通过命令行传入。 |
| `skills/` | 随脚本分发的 Codex 自动处理协议。skill 负责指导 agent 定位脚本、检查数据、运行流程和等待人工确认。 |
| `requirements.txt` | Python 依赖清单。GPU 机器上建议按本机 CUDA 版本单独选择 PyTorch 安装源。 |

## 安装和运行环境

裁切和定标常在远端 Linux/GPU 机器运行，detect 和 analysis 也可以在本地 Python 环境运行。推荐从仓库根目录安装：

```bash
git clone <repo-url> data_processing
cd data_processing
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Windows PowerShell：

```powershell
git clone <repo-url> data_processing
cd data_processing
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

如果需要 GPU 推理，按本机 CUDA 和驱动版本从 PyTorch 官方安装页选择 `torch` / `torchvision` 安装命令。`requirements.txt` 中的未固定版本适合先建立可运行环境；正式批处理建议记录实际 Python、CUDA、PyTorch 和 ultralytics 版本。

关键依赖包括：`numpy`、`scipy`、`h5py`、`astropy`、`matplotlib`、`pandas`、`seaborn`、`numba`、`opencv-python`、`torch`、`torchvision`、`ultralytics`。

## 资源文件

以下文件需要随仓库或数据包一起提供：

- `gain_para.csv`：FAST beam gain 参数表。
- `highcal_20201014_psr_tny.npz`：默认噪声管参考文件。没有这个文件时，定标命令必须显式传入可替代的 `--cal-npz` 或在临时 runner 中传入对应路径。
- `models/best_model_yolo11n_ema.pth`：默认 YOLO 检测权重。没有这个文件时，自动和半自动检测必须显式传入其他 `--model-path` 和匹配的 `--model-name`。

`detections/`、analysis 输出、H5/FITS 和诊断图属于运行产物，不应作为源码默认提交。`.gitignore` 已经忽略这些运行产物和 `models/*.old` 旧权重。

## Codex skill 安装

本仓库自带一个 Codex skill：

```text
skills/fast-frb-observation-processing/
```

交给 Codex agent 安装时，可以直接说：`请帮我安装本仓库的 Codex skill：复制 skills/fast-frb-observation-processing 到 Codex skills 目录，设置 DATA_PROCESSING_ROOT 为当前仓库，并完成 README 里的安装后自检。`

推荐把这个目录复制到 Codex skills 目录，并设置 `DATA_PROCESSING_ROOT` 指向本仓库根目录。这样新 agent 可以自动找到脚本，而不是从 skill 目录误运行。

Bash：

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
cp -R skills/fast-frb-observation-processing "${CODEX_HOME:-$HOME/.codex}/skills/"
export DATA_PROCESSING_ROOT="$(pwd)"
```

PowerShell：

```powershell
$codexHome = if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $HOME ".codex" }
New-Item -ItemType Directory -Force (Join-Path $codexHome "skills") | Out-Null
Copy-Item -Recurse -Force .\skills\fast-frb-observation-processing (Join-Path $codexHome "skills")
$env:DATA_PROCESSING_ROOT = (Get-Location).Path
```

`DATA_PROCESSING_ROOT` 只对当前 shell 生效。需要长期使用时，把它写入 shell profile 或系统环境变量。

## 安装后自检

在仓库根目录中做语法、依赖、CLI 和 skill 检查：

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

Codex agent 使用该 skill 时，会先确认脚本根目录包含 `cut_burst_data.py`、`calibration.py`、`burst_detect.py`、`burst_analysis.py`、`rfi_utils.py`、`ZeithAngle.py` 和 `gain_para.csv`。缺少脚本根目录时，agent 应要求用户提供 `data_processing` repo 路径。

## 输入表格式

`batch_processing/*.txt` 是本地观测清单和坐标清单，默认不作为源码提交。批量处理时请把自己的清单放在任意目录，并通过 `--burst-txt`、`--dm-file`、`--catalog-dir` 或 `--plan-txt` 显式传入；如果想沿用脚本默认值，也可以在本地 `batch_processing/` 下放同名 `.txt` 文件。需要提交模板时，使用 `*.example.txt` 文件名。

### `FRB*_Burst.txt`

批量裁切和旧 FITS 转 H5 使用这个表。表头可有可无，代码会跳过第一列为 `base` 的行。

列顺序固定为：

```text
base project name date beam dm time
```

| 列 | 含义 | 例子 |
|---|---|---|
| `base` | 数据盘名，不带开头 `/` | `data31` |
| `project` | 观测项目目录 | `ZD2025_5` |
| `name` | 源名或原始观测目录名 | `FRB20251229A` |
| `date` | 8 位观测日期 | `20260106` |
| `beam` | FAST beam 编号，整数形式 | `1` |
| `dm` | 裁切阶段使用的 DM | `192.1` |
| `time` | burst 到达时间，单位秒，相对观测开始 | `1608.13` |

代码会拼出原始 FITS 路径：

```text
/{base}/{project}/{name}/{date}/
```

### `h5_calibration_dm_file.txt`

批量定标使用这个表。

列顺序为：

```text
FRB_name DM RA DEC
```

这里的 DM 主要用于源信息记录。裁切阶段使用的 DM 已保存在未定标 H5 attrs 中，analysis 会从 H5 读取。

## 阶段 1：裁切原始 FITS

单日期脚本的配置在 `cut_burst_data.py` 文件底部：

```python
SEGMENT_LENGTH = 4096 * 16
DM = 411.2
BEAM = 1
NUM_WORKERS = 16
FRB_NAME = 'FRB20201124A'
DATE = '20210526'

DATA_PATH = '/path/to/raw/fits/date/'
SAVE_PATH = '/path/to/H5_Cut/FRB/date/'
TOA_FILE = 'toa_list_fast.txt'
```

`TOA_FILE` 中的秒数必须由用户或上游搜索流程提供，单位是相对观测开始的秒。agent 可以帮助整理、去重和检查范围，但不应自行猜测 TOA。

运行：

```bash
python cut_burst_data.py
```

脚本会：

1. 在 `DATA_PATH` 下找对应 beam 的 FITS。
2. 复制第一个匹配 FITS 到输出目录，作为后续定标用的 `_0001.fits`。
3. 读取 `TOA_FILE` 中的秒数。
4. 对每个 TOA 计算采样点。
5. 根据 DM 计算色散延迟，裁出包含 burst 的时间段。
6. 对每个频率通道做 shift 消色散。
7. 写出未定标 H5 和 `obs_info.json`。

输出命名：

```text
{frb}-{date}-M{beam:02d}-{fits_number:04d}-{start_sample:09d}.h5
```

## 阶段 2：批量裁切

批量入口读取一个 `FRB*_Burst.txt`：

```bash
python batch_processing/batch_cut_burst_data.py \
  --burst-txt /path/to/catalogs/FRB20251229A_Burst.txt \
  --output-root /path/to/H5_Cut/FRB20251229A \
  --save-frb-name FRB20251229A \
  --workers 16
```

`--output-root` 是某一个 FRB 的输出根目录。脚本会写到：

```text
{output-root}/{date}/*.h5
```

重新生成已有文件时加：

```bash
--overwrite
```

## 阶段 3：旧 FITS cut 转 H5

如果某些源已经用旧脚本裁成 burst FITS，可以转换为当前 H5 格式：

```bash
python batch_processing/fits_to_h5.py \
  --asd-root /path/to/asd \
  --output-root /path/to/H5_Cut \
  --catalog-dir /path/to/catalogs \
  --workers 16
```

只处理部分源：

```bash
python batch_processing/fits_to_h5.py --only FRB20220912A FRB20251229A
```

如果表格中找不到完全匹配 TOA，脚本会从文件名中的 `start_sample` 和数据长度反推 `toa_sec`，DM 使用同源同日期表中的值或 `NaN`。

## 阶段 4：定标

`calibration.py` 输入未定标 H5，输出 `*_cal.h5`。核心步骤：

1. 找同日期目录中的 `_0001.fits`。
2. 折叠噪声管，得到 `noise_cal`。
3. 从 `highcal_20201014_psr_tny.npz` 读取 `t_cal`。
4. 根据 MJD、RA、DEC、beam 计算 ZA 和 FAST gain。
5. 把原始数据定标到 I/Q/U/V，单位 Jy。
6. 按保存倍率做时间和频率下采样。
7. 运行 RFI 标记。
8. 输出 H5 和 quick-look `.jpg`。

单日期配置在文件底部：

```python
BURST_DIR = '/path/to/H5_Cut/FRB/date/'
OUTPUT_DIR = '/path/to/H5_Cal/FRB/date/'
CAL_NPZ = 'highcal_20201014_psr_tny.npz'
RA = '05h08m03.51s'
DEC = '26d03m38.5s'
DOWN_TIME = None
DOWN_FREQ = None
RFI_FFT = True
NUM_WORKERS = 8
```

运行：

```bash
python calibration.py
```

批量定标：

```bash
python batch_processing/batch_calibration.py \
  --root-dir /path/to/H5_Cut \
  --cal-root /path/to/H5_Cut/H5_Cal \
  --dm-file /path/to/catalogs/h5_calibration_dm_file.txt \
  --cal-npz highcal_20201014_psr_tny.npz \
  --workers 8
```

只处理部分源：

```bash
python batch_processing/batch_calibration.py --only FRB20220912A FRB20251229A
```

## 阶段 5：自动或人工标记 burst

默认检测模型：

```text
models/best_model_yolo11n_ema.pth
--model-name yolo11n
```

*注意：替换模型权重时，请写清模型名称和 EMA 来源，保留上一版权重便于回滚。替换后同步脚本的默认值。*

自动模式：

```powershell
python burst_detect.py `
  --mode auto `
  --cal-dir C:\path\to\calibrated_h5 `
  --model-path models\best_model_yolo11n_ema.pth `
  --model-name yolo11n `
  --output-dir C:\path\to\detections_auto
```

半自动模式：

```powershell
python burst_detect.py `
  --mode semi-auto `
  --cal-dir C:\path\to\calibrated_h5 `
  --model-path models\best_model_yolo11n_ema.pth `
  --model-name yolo11n `
  --output-dir C:\path\to\detections_semiauto
```

手工模式：

```powershell
python burst_detect.py `
  --mode manual `
  --cal-dir C:\path\to\calibrated_h5 `
  --output-dir C:\path\to\detections_manual
```

检测结果保存到：

- H5 attrs：`attrs["bursts"]`，JSON 字符串。
- 输出目录：`detections.json` 和诊断图。

探测和交互审查仍使用除背景归一化后的动态谱。保存到 `plots/*_det.png` 的诊断图会在确认 `bursts` 后重新绘制：先用非 burst 时段减背景并 mask 掉检测后 RFI，再叠加 box。

同一个文件重新确认标记时会覆盖 H5 中的 `bursts`。人工修正后保留对应输出目录，便于追溯。

### 剔除重复或低 SNR 标记

如果人工检查时发现当前图和上一张图重复，或者当前图信号信噪比太低、不希望进入后续分析，最简单的做法是把这一页最终标成“无 burst”：

```json
{
  "bursts": [],
  "has_burst": false
}
```

注意：`burst_analysis.py` 后续读取的是每个 `*_cal.h5` 里的 `attrs["bursts"]`，不是直接读取 `detections.json`。因此只删除或清空 `detections.json` 中的记录，不一定会影响已经写入 H5 的旧标记。

还在半自动或手工标记窗口里时：

- 直接按 `x`，当前页会写入空的 `bursts` 并进入下一张。
- 手工模式下也可以不画框直接按 Enter；半自动模式下按 `x` 更直接，不需要先清掉模型框。

已经跑完检测、后来发现某张图应当剔除时：

1. 从当前输出目录的 `detections.json` 里删掉这个文件名对应的 entry。
2. 用同一个 `--cal-dir` 和同一个 `--output-dir` 重新运行 `burst_detect.py`。
3. 脚本会跳过 `detections.json` 中仍存在的文件，只重新打开被删掉的文件。
4. 在窗口中按 `x` 把它标为空；脚本会覆盖对应 H5 的 `attrs["bursts"]`，并把 `detections.json` 写回 `{"bursts": [], "has_burst": false}`。

如果同一张图里有多个框，只想剔除其中一个重复或低 SNR 信号，不要把整页标为空；应当只删除或重画 `bursts` 列表中对应的那个 region。

## 阶段 6：能量/偏振分析和出表

`burst_analysis.py` 读取已经带 `bursts` 标记的 `*_cal.h5`，逐个 burst 计算：

- TOA
- peak flux
- fluence
- width
- burst bandwidth
- SNR
- DM
- RM
- 线偏振、圆偏振、总偏振比例
- PA 和 PAV

主要表格输出是 `burst_results.csv`。诊断图用于检查 DM、RM、偏振和 burst 区域测量是否可信。

示例：

```powershell
python burst_analysis.py `
  --cal-dir C:\path\to\calibrated_h5 `
  --output-dir C:\path\to\analysis_output `
  --dm-range 5 `
  --dm-step 0.1 `
  --rm-min -1000 `
  --rm-max 1000 `
  --n-rm 100000
```

说明：

- `--dm-range 5 --dm-step 0.1` 表示围绕 H5 中的 cut DM 搜索 `[-5, +5]`。
- `--rm-min`、`--rm-max`、`--n-rm` 控制 RM trial 网格。
- `--target-down-time` 和 `--target-down-freq` 可在 analysis 阶段统一下采样后再计算。
- flux 和 fluence 使用 burst 带宽内的通道计算，不直接对全观测带宽平均。

## 下采样规则

裁切阶段保留原始时间和频率分辨率。未定标 H5 中：

```text
data shape = (nsamp, npol, nchan_raw)
attrs["time_reso"] = 原始时间分辨率
attrs["nchan"] = 原始频率通道数
```

定标阶段区分保存分辨率和画图/检测分辨率。默认自动计算接近下列目标的画图分辨率：

```text
目标时间分辨率约 49.152 us * 8
目标频率通道数约 512
```

如果 `DOWN_TIME=None`、`DOWN_FREQ=None`，保存倍率等于画图倍率。如果需要保留更高时间分辨率用于 peak flux 对比，可设置 `DOWN_TIME=1`。

定标后 H5 attrs 会记录：

```text
time_reso_raw
time_reso
down_time
down_freq
plot_down_time
plot_down_freq
```

## RFI 处理

`rfi_utils.cal_rfi(data, noise_mask, down_time, down_freq, fft=True)` 返回：

```text
rfi_channel: (nchan,) bool
rfi_pixel:   (nsamp, nchan) bool
```

定标阶段尚不知道 burst 区域，因此整段数据都当作噪声区做快速 RFI 标记，并写入 `rfi_mask` / `rfi_channel`。

`burst_detect.py` 在 YOLO 或人工确认 `bursts` 后，会用非 burst 时段重新估计噪声区、重算 RFI，并把结果写入 `burst_rfi_mask` / `burst_rfi_channel`，不覆盖定标阶段字段。

analysis 阶段也会按同一思路在内存中重算 RFI，以 Stokes I 和 V 的并集作为最终 RFI mask。

## H5 schema

### 未定标 H5

数据集：

```text
data: (nsamp, npol, nchan)
freq: (nchan,), MHz
```

关键 attrs：

```text
start_sample
file_mjd
toa_sec
time_reso
npol
nchan
segment_length
obs_start_mjd
dm
```

### 定标后 H5

数据集：

```text
data:        (4, nsamp, nchan), float32, I/Q/U/V, Jy
freq:        (nchan,), MHz
rfi_mask:    (nsamp, nchan), bool
rfi_channel: (nchan,), bool
burst_rfi_mask:    (nsamp, nchan), bool, detection 后由非 burst 噪声段重算
burst_rfi_channel: (nchan,), bool, detection 后由非 burst 噪声段重算
gain:        (nchan,), K/Jy
gain_err:    (nchan,), K/Jy
```

关键 attrs：

```text
file_mjd
obs_start_mjd
start_sample
toa_sec
time_reso_raw
time_reso
down_time
down_freq
plot_down_time
plot_down_freq
nchan_raw
nchan
nsamp
dm
beam
ra
dec
rfi_fraction
burst_rfi_fraction
burst_rfi_channel_count
burst_rfi_noise_sample_count
burst_rfi_method
burst_rfi_source
bursts
```

`bursts` 是 JSON 字符串，示例：

```json
[
  {
    "time_start": 120,
    "time_end": 180,
    "freq_start": 40,
    "freq_end": 500,
    "confidence": 0.82
  }
]
```

time/freq index 都在定标后 H5 保存分辨率上。

## 完整批处理示例

裁切：

```bash
python batch_processing/batch_cut_burst_data.py \
  --burst-txt /path/to/catalogs/FRB20251229A_Burst.txt \
  --output-root /path/to/H5_Cut/FRB20251229A \
  --save-frb-name FRB20251229A \
  --workers 16
```

批量定标：

```bash
python batch_processing/batch_calibration.py \
  --root-dir /path/to/H5_Cut \
  --cal-root /path/to/H5_Cut/H5_Cal \
  --dm-file /path/to/catalogs/h5_calibration_dm_file.txt \
  --cal-npz highcal_20201014_psr_tny.npz \
  --workers 8
```

检测：

```powershell
python burst_detect.py `
  --mode auto `
  --cal-dir C:\path\to\H5_Cal\FRB20220912A `
  --model-path models\best_model_yolo11n_ema.pth `
  --model-name yolo11n `
  --output-dir C:\path\to\detections_yolo11n_auto
```

分析：

```powershell
python burst_analysis.py `
  --cal-dir C:\path\to\H5_Cal\FRB20220912A `
  --output-dir C:\path\to\analysis_pipeline `
  --dm-range 5 `
  --dm-step 0.1 `
  --rm-min -1000 `
  --rm-max 1000 `
  --n-rm 100000
```

## 检查点

### 裁切前

- 表格中的 `base/project/name/date` 能拼出真实路径。
- `beam` 与 FITS 文件名中的 `Mxx` 一致。
- `time` 是相对观测开始的秒数。
- DM 是该源、该日期应该使用的值。

### 定标前

- 每个日期目录有匹配 beam 的 `_0001.fits`。
- 源坐标表包含该 FRB 的 RA/DEC。
- `highcal_20201014_psr_tny.npz` 可读取。

### 检测前

- `*_cal.h5` 中存在 `data`、`freq`、`rfi_mask`、`gain`、`gain_err`。
- 模型权重存在。
- 观测分辨率和训练数据差异较大时，先看 quick-look 图确认 burst 形态。

### 分析前

- H5 attrs 中已有 `bursts`。
- 人工标记覆盖了完整 burst 区域。
- DM/RM 搜索范围符合源的先验。

## 代码层面的注意事项

- `cut_burst_data.py` 和 `calibration.py` 的单日期参数仍在文件底部修改，不是完整 CLI。
- `cut_burst_data.py`、`calibration.py` 和部分批处理脚本里保留的 `/path/to/user/...`、示例 FRB、示例日期等默认值只表示本地历史配置；外部机器运行前必须改成真实路径，或通过临时 runner / CLI 参数显式传入。
- 批量脚本从 `batch_processing/` 子目录运行时会自动把父目录加入 `sys.path`。
- `batch_calibration.py` 的 `--rfi-down-freq` 是隐藏兼容参数，新流程不单独依赖它。
- `fits_to_h5.py` 默认从自身所在目录读取 `FRB20*_Burst.txt`；因为这些清单是本地输入并被 git 忽略，跨机器运行时建议显式传 `--catalog-dir`。
- analysis 阶段重新根据噪声区做 RFI，优先服务最终物理量测量。
