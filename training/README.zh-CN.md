# AFTER Burst Detector 训练

[English](README.md)

本目录训练 [`after/burst_detect.py`](../after/burst_detect.py) 使用的 YOLO burst
定位模型。`after/` 包负责在定标观测上推理和人工复核；`training/` 负责训练数据读取、
模型训练、验证和预览。

先完成[主 README 的安装](../README.zh-CN.md#安装)，再从仓库根目录安装训练附加依赖：

```bash
python -m pip install -r requirements-training.txt
```

## 目录职责

| 路径 | 用途 |
|---|---|
| `yolo_data.py` | 读取 H5、按观测日期切分、增强图像并转换坐标。 |
| `yolo_train.py` | 单 GPU 训练、EMA、验证、early stopping 和 checkpoint 导出。 |
| `yolo_eval.py` | 训练期间使用的检测指标。 |
| `yolo_infer.py` | 使用指定 checkpoint 预览验证集。 |
| `train.sh` | 单 GPU 训练的 Bash wrapper。 |
| `data/` | 被忽略的本地工作区，用于 H5、标注、核验图片和数据准备脚本。 |
| `logs/` | 训练运行和指标；checkpoint 在选为生产模型前保持未跟踪。 |

## 生产 checkpoint

当前部署模型为
[`../models/best_model_yolo11n_ema.pth`](../models/best_model_yolo11n_ema.pth)。
已核验的 SHA-256 和运行命令由[主 README](../README.zh-CN.md#模型与运行结果)统一维护。

仓库跟踪的 `logs/logs_yolo11n_multi_hard/logs_yolo11n.json` 记录了当前生产模型的
历史训练。该次训练使用旧版帧级 validation 切分；当前代码按 `original_date` 分组，
比较新 checkpoint 时使用当前日期分组结果。

## 数据契约

把训练 H5 放到 `training/data/`，或通过 `--data-path`、`DATA_PATH` 指定其他目录。
每个 H5 提供：

- `images`：`(N, 512, 512)`，存储顺序为 `(time, frequency)`；
- `annotations`：`(image_idx, x_freq, y_time, w_freq, h_time)`；
- `original_date`：用于隔离 train 和 validation 的观测标识。

加载器会把标注转换为模型坐标 `(x=time, y=frequency)`。模型输入固定为 512 × 512。

## 训练

从 AFTER 仓库根目录运行：

```bash
python training/yolo_train.py yolo11n \
  --data-path /path/to/training_h5 \
  --device 0 \
  --epochs 100 \
  --batch-size 64
```

Bash wrapper 通过环境变量暴露相同的常用设置：

```bash
DATA_PATH=/path/to/training_h5 \
BATCH_SIZE=64 \
EPOCHS=100 \
bash training/train.sh "0" yolo11n
```

每次模型或数据集对比使用独立的日志目录。

## 预览与部署

预览按日期分组的 validation：

```bash
python training/yolo_infer.py yolo11n \
  --model-path /path/to/checkpoint.pth \
  --data-path /path/to/training_h5 \
  --conf 0.3 \
  --start 0 \
  --end 30
```

根据当前 validation、独立观测检查和端到端 detection 表现选择生产候选。把选定的 EMA
checkpoint 复制为 `models/best_model_yolo11n_ema.pth`，同步更新两个主 README 中的
哈希，并重新执行仓库自检。
