# AFTER 批处理工具

[English](README.md)

`batch_processing/` 提供 AFTER 的批量入口，用于把观测目录和事件表转换为后续检测、复核
与物理量分析所需的标准数据产品。这里的脚本主要负责四类任务：

```text
已确认 TOA + 原始 FAST FITS
  -> 批量裁切 H5
  -> 批量流量/偏振定标
  -> *_cal.h5

旧版 burst FITS
  -> 当前 cut H5 schema
  -> 批量定标

重建清单 + 原始 FAST FITS
  -> 重建旧版 burst FITS
  -> 当前 cut H5 schema
```

本文所有命令都假设当前目录是 AFTER 仓库根目录。安装依赖和完整科学流程见
[`../README.zh-CN.md`](../README.zh-CN.md)。

## 入口选择

| 任务 | 入口 | 适用情况 |
|---|---|---|
| 根据标准 `*_Burst.txt` 批量裁切 | `batch_cut_burst_data.py` | 每个事件使用同一 `segment_length`。 |
| 裁切长周期或需要逐事件窗口长度的候选 | `batch_cut_selected_long_period.py` | 每行单独指定 `segment_length`，并可按 source/date 筛选。 |
| 重建旧版 burst FITS | `cut_burst_fits.py` | 旧 FITS 已删除，但仍保留经核查的重建清单和原始观测。 |
| 把旧版 burst FITS 转成当前 H5 | `fits_to_h5.py` | 已经存在旧 cut FITS，不需要重新读取完整原始观测。 |
| 批量定标 cut H5 | `batch_calibration.py` | 生成 Stokes I/Q/U/V、RFI mask 和 `*_cal.h5`。 |

先查看目标入口的完整参数：

```bash
python batch_processing/batch_cut_burst_data.py --help
python batch_processing/batch_cut_selected_long_period.py --help
python batch_processing/cut_burst_fits.py --help
python batch_processing/fits_to_h5.py --help
python batch_processing/batch_calibration.py --help
```

## 1. 从标准事件表批量裁切

### 输入表

`batch_cut_burst_data.py` 读取空白字符分隔的 `*_Burst.txt`。首行可以是表头；每个数据行
至少包含七列：

```text
base project name date beam dm time
```

可复制只有表头的 [`Burst.example.txt`](Burst.example.txt) 作为起点。

| 列 | 含义 |
|---|---|
| `base` | 原始数据根的第一段路径，不含开头 `/`。 |
| `project` | 项目目录名。 |
| `name` | 原始观测 source 目录名。 |
| `date` | 观测日期目录，通常为 `YYYYMMDD`。 |
| `beam` | FAST beam 编号，例如 `1` 表示 `M01`。 |
| `dm` | 用于裁切边界和 metadata 的 DM。 |
| `time` | 相对整次观测起点的已确认 TOA，单位为秒。 |

脚本按下式定位原始数据：

```text
/<base>/<project>/<name>/<date>/
```

因此 `time` 必须来自观测者或上游搜索结果，不能使用单个 FITS 文件内的局部时间。

### 运行

```bash
python batch_processing/batch_cut_burst_data.py \
  --burst-txt /path/to/catalogs/FRBXXXX_Burst.txt \
  --output-root /path/to/after_runs/cut/FRBXXXX \
  --save-frb-name FRBXXXX \
  --segment-length 65536 \
  --workers 8
```

`--segment-length` 是每个 cut 的样本数，默认值为 `65536`。脚本按原始路径、日期、beam
和 DM 分组，复制该 beam 的第一个匹配 FITS 作为定标参考，并为每个 TOA 调用
`after.cut_burst_data` 的裁切函数。

典型输出：

```text
<output-root>/
  <date>/
    *_0001.fits
    *.h5
    obs_info.json
```

同名输出需要重建时显式增加 `--overwrite`。`obs_info.json` 会逐个读取日期目录中的
H5：一致值保持标量，混合 beam、DM 和 segment length 写成列表，并在每个 burst 条目
中保留各自取值。

## 2. 裁切逐事件长窗口

`batch_cut_selected_long_period.py` 适用于不同候选需要不同窗口长度的情况。最小输入格式
是在标准七列之后增加 `segment_length`：

```text
base project name date beam dm time segment_length [selected_images] [note]
```

`selected_images` 和 `note` 是可选的来源记录，不参与裁切计算。脚本还兼容包含额外时间
范围列的旧版扩展表。

先用 dry-run 检查路径、分组和筛选结果：

```bash
python batch_processing/batch_cut_selected_long_period.py \
  --plan-txt /path/to/catalogs/Selected_LongPeriod_Burst.txt \
  --output-root /path/to/after_runs/long_period_cut \
  --workers 8 \
  --dry-run
```

确认后去掉 `--dry-run`。可以重复传入筛选参数：

```bash
python batch_processing/batch_cut_selected_long_period.py \
  --plan-txt /path/to/catalogs/Selected_LongPeriod_Burst.txt \
  --output-root /path/to/after_runs/long_period_cut \
  --only-source FRBXXXX \
  --only-date YYYYMMDD \
  --workers 8
```

输出布局：

```text
<output-root>/<source>/<date>/
  *_0001.fits
  *.h5
  obs_info.json
```

`--overwrite` 会清理本次所选 source/date 对应的已有裁切产物后重建。应先配合
`--dry-run` 确认筛选范围。

## 3. 重建或转换旧版 burst FITS

如果旧 cut FITS 已删除，但保留了精确重建所需的索引，可以用 `cut_burst_fits.py`
从原始观测重新生成。先运行 `--dry-run`；它只检查清单、冻结的原始文件顺序和
未压缩 FITS 是否齐全，不写 burst 数据：

```bash
python batch_processing/cut_burst_fits.py \
  --burst-txt /path/to/catalogs/FRBXXXX_legacy_fits_rebuild.txt \
  --raw-root /path/to/raw-root \
  --output-root /path/to/rebuilt-fits \
  --dry-run
```

逐项确认输出的目录和分组后再去掉 `--dry-run`。默认情况下，任何
`rebuild_status` 不以 `ready` 开头的记录都会停止任务；只有核查过不能精确重建的
原因后才应使用 `--skip-blocked`。该脚本只写消色散后的 burst FITS，不做定标。

`fits_to_h5.py` 用于已经存在旧版 burst cut FITS、但后续流程需要当前 H5 schema 的情况。
预期输入大致为：

```text
<legacy-root>/
  <FRB>/
    <date>/
      *_0001.fits
      <FRB>-<date>-Mxx-<fits-number>-<start-sample>.fits
```

`--catalog-dir` 下应放置匹配的 `<FRB>_Burst.txt`。通常使用上一节的标准七列格式；同时
兼容旧版六列 `name beam project dm date time` 格式。日期目录和文件名既可以使用
`YYYYMMDD`，也可以使用 `YYYYMMDD_1` 这类同日分段形式。脚本用目录名、日期、beam、
文件编号和起始样本匹配 catalog metadata。

```bash
python batch_processing/fits_to_h5.py \
  --asd-root /path/to/legacy_burst_data \
  --output-root /path/to/after_runs/cut \
  --catalog-dir /path/to/catalogs \
  --workers 16
```

只转换指定 source：

```bash
python batch_processing/fits_to_h5.py \
  --asd-root /path/to/legacy_burst_data \
  --output-root /path/to/after_runs/cut \
  --catalog-dir /path/to/catalogs \
  --only FRBXXXX FRBYYYY
```

脚本会复制 `_0001.fits` 定标文件，将旧 burst FITS 写成当前 cut H5，并在每个日期目录生成
`obs_info.json`。先用 `--dry-run` 可以只检查范围而不创建目录或文件；使用 `--overwrite`
可以替换已存在的转换结果。`--asd-root` 必须显式指定，脚本不会猜测要扫描哪个旧目录。

## 4. 批量流量与偏振定标

### 输入布局

```text
<root-dir>/
  <FRB>/
    <date>/
      *.h5
      *_0001.fits
```

定标 source 表为空白字符分隔的四列：

```text
FRB_name DM RA DEC
```

RA/DEC 可以使用冒号格式或 Astropy 可识别的带单位格式。DM 用于 source 记录；每个 cut
H5 自身也应保存对应 DM。
每个 beam 组优先使用当前日期目录中匹配的 `Mxx..._0001.fits`。对
`YYYYMMDD_1`、`YYYYMMDD_2` 这类同日分段，若当前分段的噪声管诊断不通过，
批处理可改用同一天其他分段中同波束的有效定标文件，但不会跨日期回退。
同日候选全部不可用时停止该组；实际采用的文件会写入每个 H5 的
`calibration_fits` 属性。
`NPOL=2` 的噪声管文件会继续用 AA/BB 做流量定标，并自动跳过需要交叉项的四路
诊断图；`NPOL=4` 的现有诊断和偏振流程不变。

噪声管周期需要显式指定：默认是 `--noise-period-s 0.2`；应传入观测采用的名义
设置，例如 `0.1`、`0.4`、`1` 或 `2`。只要周期至少覆盖两个采样点，并且定标
FITS 至少包含一个完整周期，代码就不限制具体的正周期值。用户只需提供总周期，
折叠轮廓会自动识别 On/Off 相位、持续时间和占空比。所用名义周期会写入 H5 的
`noise_period_s` 属性。

### 运行

```bash
python batch_processing/batch_calibration.py \
  --root-dir /path/to/after_runs/cut \
  --cal-root /path/to/after_runs/calibrated \
  --dm-file /path/to/catalogs/h5_calibration_dm_file.txt \
  --cal-npz highcal_20201014_psr_tny.npz \
  --noise-period-s 0.2 \
  --workers 8
```

只处理指定 source：

```bash
python batch_processing/batch_calibration.py \
  --root-dir /path/to/after_runs/cut \
  --cal-root /path/to/after_runs/calibrated \
  --dm-file /path/to/catalogs/h5_calibration_dm_file.txt \
  --cal-npz highcal_20201014_psr_tny.npz \
  --only FRBXXXX FRBYYYY \
  --workers 8
```

保存分辨率：

- 省略 `--down-time`、`--down-freq`：自动选择适合检查和画图的分辨率；
- `--down-time 1`：保留原始时间分辨率；
- `--down-freq 1`：保留原始频率通道。
- `--time-crop-samples 512`：先围绕输入时间轴中心裁出 512 个原始采样点；
  未显式指定 `--down-time` 时会自动使用 1，且 JPG 直接按保存分辨率重画。
- `--target-time-reso-ms 0.786432 --output-time-samples 512`：先对完整
  输入做定标，再按每个文件的原始分辨率自动计算整数时间下采样倍率，
  最后从下采样结果的中心裁出 512 点。`--target-time-reso-ms` 不能与
  `--down-time` 并用，`--output-time-samples` 不能与 `--time-crop-samples` 并用。

批处理默认启用 FFT RFI；传入 `--no-rfi-fft` 可改用熵方法。

输出布局：

```text
<cal-root>/<FRB>/<date>/
  *_cal.h5
  *.jpg
```

每个 H5 保存 `calibration_beam`、`calibration_fits`、`calibration_npz` 和
`noise_period_s` attrs。
比较不同定标或降采样设置时，使用独立的 `--cal-root`。

## 输出交接

定标完成后，按[主 README 的快速开始](../README.zh-CN.md#快速开始)继续执行“检测并
复核 burst 区域”和“分析物理量”。两个入口都会递归查找 `*_cal.h5`，因此
`--cal-dir` 可以直接使用批处理 `<cal-root>`。运行能量和偏振分析前，先复核或修正
写入 H5 `attrs["bursts"]` 的候选区域。
