# FAST FRB 搜索后处理目录

本目录是 LPT 工作区里的 FAST FRB burst 搜索后处理工具集，用于把已确认候选的 TOA/DM 变成裁切 H5、定标 H5、已复核的 burst 区域、物理量表格和可选 HTML 观测面板。

当前 README 聚焦这个目录中实际存在的脚本和用法，服务 LPT 工作区内的后处理流程。

## 处理流程

```text
原始 FAST FITS + TOA/DM
  -> cut_burst_data.py 或 batch_processing/batch_cut_burst_data.py
  -> 未定标 burst H5
  -> calibration.py 或 batch_processing/batch_calibration.py
  -> 定标 *_cal.h5
  -> burst_detect.py
  -> H5 attrs["bursts"] + detections.json
  -> burst_analysis.py
  -> burst_results.csv + 诊断图
  -> burst_dashboard.py
```

也可以从中间产物开始：

| 起点 | 必需输入 | 后续动作 |
|---|---|---|
| 原始 FAST FITS | FITS 目录、源名、日期、beam、DM、TOA 秒数、输出目录 | 裁切、定标、检测、复核、分析 |
| 未定标 H5 | `.h5`、匹配的 `_0001.fits`、RA、DEC、定标 npz | 定标、检测、复核、分析 |
| 定标 H5 | `*_cal.h5`、检测模型、输出目录 | 检测、复核、分析 |
| 已标注定标 H5 | 带 H5 attr `bursts` 的 `*_cal.h5` | 分析、出表、生成面板 |

## 文件职责

| 路径 | 作用 |
|---|---|
| `cut_burst_data.py` | 从原始 FAST FITS 裁切 burst-centered H5 的 helper 函数，以及一个本地硬编码示例。 |
| `calibration.py` | 偏振/流量定标、下采样、RFI mask、quick-look 图和 `*_cal.h5` 写出 helper，以及一个本地硬编码示例。 |
| `burst_detect.py` | 自动、半自动、手动 burst-region 标注 CLI，写 H5 `attrs["bursts"]`、`detections.json` 和复核图。 |
| `burst_analysis.py` | 从已确认 burst 区域测量 DM、RM、偏振、flux、fluence、width、bandwidth 和 SNR。 |
| `burst_dashboard.py` | 根据 `burst_results.csv` 生成静态 HTML 观测面板。 |
| `burst_dm.py` | `burst_analysis.py` 使用的精细 DM 搜索模块。 |
| `burst_pol.py` | RM、PA、PAV 和偏振处理模块。 |
| `burst_properties.py` | flux、fluence、width、bandwidth 和 SNR 测量模块。 |
| `rfi_utils.py` | calibration 和 analysis 共用的 RFI 标记工具。 |
| `ZeithAngle.py` | FAST ZA 与 gain 辅助函数。 |
| `gain_para.csv` | FAST beam gain 参数表。 |
| `highcal_20201014_psr_tny.npz` | 默认噪声管定标参考文件。 |
| `models/` | burst detector 权重。 |
| `batch_processing/` | 批量裁切、旧 FITS 转 H5、长周期候选裁切和批量定标包装脚本。 |
| `skills/fast-frb-observation-processing/` | Codex 使用本流程的操作协议。 |
| `test_burst_dashboard.py` | dashboard 单元测试。 |
| `requirements.txt` | Python 依赖清单。 |

## 版本库边界

Git 追踪处理代码、测试、工作流文档、当前生产检测模型，以及默认流程需要的小型校准和增益文件。原始观测、生成的 H5、复核图片、结果表、本地批处理目录、缓存和退役模型都保留在本机，并由 `.gitignore` 排除。

当前明确追踪的生产检测模型是
`models/best_model_yolo11n_ema.pth`。实验或替换模型可以放在 `models/`
中做本地测试；除非明确调整忽略规则，否则不会加入 Git。

## 环境

在当前 LPT checkout 的 Python 环境中安装依赖：

```bash
python -m pip install -r requirements.txt
```

GPU detector 推理需要根据机器 CUDA 驱动安装匹配的 `torch` / `torchvision`。`requirements.txt` 不固定 CUDA 专用 PyTorch wheel。

从本目录做快速自检：

```bash
python -m py_compile cut_burst_data.py calibration.py burst_detect.py burst_analysis.py burst_dashboard.py burst_dm.py burst_pol.py burst_properties.py rfi_utils.py ZeithAngle.py batch_processing/batch_calibration.py batch_processing/batch_cut_burst_data.py batch_processing/batch_cut_selected_long_period.py batch_processing/fits_to_h5.py test_burst_dashboard.py
python burst_detect.py --help
python burst_analysis.py --help
python burst_dashboard.py --help
```

`cut_burst_data.py` 和 `calibration.py` 底部属于本地示例配置。新观测优先使用 `batch_processing/` 包装脚本，或从 Python 中导入 helper 函数调用，避免误用旧硬编码路径。

## 裁切原始 FITS

批量入口：

```bash
python batch_processing/batch_cut_burst_data.py \
  --burst-txt batch_processing/FRBXXXX_Burst.txt \
  --output-root /path/to/after_data/H5_Cut/FRBXXXX \
  --save-frb-name FRBXXXX \
  --segment-length 65536 \
  --workers 8
```

`FRB*_Burst.txt` 格式：

```text
base project name date beam dm time
```

包装脚本会按原始数据路径、日期、beam 和 DM 分组，复制第一个匹配 beam 的 FITS 到输出日期目录用于定标，逐个 TOA 裁切，并写出 `obs_info.json`。

带行级 segment length 的长周期候选使用：

```bash
python batch_processing/batch_cut_selected_long_period.py \
  --plan-txt batch_processing/Selected_LongPeriod_Burst.txt \
  --output-root /path/to/after_data/LPT_Selected_Cut \
  --workers 8
```

## 旧 cut FITS 转 H5

旧版 burst FITS 需要转成当前 H5 schema 时使用：

```bash
python batch_processing/fits_to_h5.py \
  --asd-root /path/to/after_data \
  --output-root /path/to/after_data/H5_Cut \
  --catalog-dir batch_processing
```

脚本会复制 `_0001.fits` 定标文件，并写出与 `cut_burst_data.py` 一致的数据集和 attrs。

## 定标

批量入口：

```bash
python batch_processing/batch_calibration.py \
  --root-dir /path/to/after_data/H5_Cut \
  --cal-root /path/to/after_data/H5_Cut/H5_Cal \
  --dm-file batch_processing/h5_calibration_dm_file.txt \
  --cal-npz highcal_20201014_psr_tny.npz \
  --workers 8
```

`h5_calibration_dm_file.txt` 格式：

```text
FRB_name DM RA DEC
```

常用保存分辨率：

- 不传 `--down-time` / `--down-freq`：保存自动选择的画图友好分辨率。
- `--down-time 1`：保留原始时间分辨率，用于 peak flux 对比。
- `--down-freq 1`：保留原始频率通道，用于细致 RFI 或频谱检查。

定标 H5 的 `data` 为 Stokes I/Q/U/V，并保存 `freq`、`rfi_mask`、`rfi_channel`、`gain`、`gain_err`。

## 爆发检测与标注复核

自动模式：

```bash
python burst_detect.py \
  --mode auto \
  --cal-dir /path/to/calibrated_h5 \
  --model-path models/best_model_yolo11n_ema.pth \
  --model-name yolo11n \
  --output-dir /path/to/detections_auto
```

检测输出：

- H5 `attrs["bursts"]`：analysis 使用的真值来源。
- `detections.json`：续跑和复核记录。
- `plots/*_det.png`：复核图。

自动和半自动模式直接使用定标后的 Stokes I 做一遍模型推理。确认 burst 框后，
再使用非 burst 时间段按 analysis 相同的 Stokes I/V 并集方法重算 RFI，写入
`burst_rfi_*`，并绘制最终的去基线、mask RFI 结果图。

置信度过滤后，会去掉宽度超过高度 `--max-horizontal-aspect` 倍的横向框（默认
为 `3`）；多个框存在正面积重叠时，在 NMS 前只保留面积最大的框。

自动标注有误时，可以从 `detections.json` 中删除坏文件条目后用 `--mode semi-auto` 重标；模型建议明显误导时用 `--mode manual`。交互界面中按 `x` 会给当前文件写入显式空 burst 列表；按 `q` 或 `Esc` 会保存已完成进度并正常退出，当前文件不会被标记为已处理。

## 分析

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

测量内容包括 TOA、peak flux、fluence、width、burst bandwidth、SNR、DM、RM、线偏振、圆偏振、总偏振、PA 和 PAV。不同 DM/RM 范围的重跑建议写入独立输出目录。

主要输出：

```text
burst_results.csv
DM/RM/polarization diagnostic plots
```

## 观测面板

```bash
python burst_dashboard.py \
  --csv /path/to/analysis/burst_results.csv \
  --output /path/to/analysis/burst_dashboard.html \
  --analysis-dir /path/to/analysis \
  --source FRBNAME \
  --date YYYYMMDD \
  --reference-dm 539 \
  --rm-significance-threshold 5 \
  --top-n 10
```

输出是可直接在浏览器打开、也可打印成 PDF 的自包含 HTML。

## H5 约定

未定标 H5：

```text
data: (nsamp, npol, nchan)
freq: (nchan,), MHz
attrs: start_sample, file_mjd, toa_sec, time_reso, npol, nchan,
       segment_length, obs_start_mjd, dm
```

定标 H5：

```text
data:        (4, nsamp, nchan), Stokes I/Q/U/V, Jy
freq:        (nchan,), MHz
rfi_mask:    (nsamp, nchan), bool
rfi_channel: (nchan,), bool
gain:        (nchan,), K/Jy
gain_err:    (nchan,), K/Jy
attrs: time_reso_raw, time_reso, down_time, down_freq,
       dm, beam, ra, dec
```

burst 标注：

```json
{"time_start": 120, "time_end": 180, "freq_start": 40, "freq_end": 500, "confidence": 0.82}
```

## 输出管理

以下都是本地运行产物：

- H5/FITS 数据：`*.h5`、`*.fits`、`*_cal.h5`
- 诊断图和面板：`*.jpg`、`*.png`、`burst_dashboard.html`
- 检测和分析输出：`detections/`、`analysis_output/`、`analysis_outputs/`
- 本地批量输入表：`batch_processing/*.txt`
- 本地旧模型权重：`models/*.old`

新跑任务建议写入独立输出目录。除非明确要覆盖旧结果，不要把新标注、新分析或新面板混进旧运行目录。
