# Burst Detector Training

本目录训练 [`after/burst_detect.py`](../after/burst_detect.py) 使用的 YOLO burst
定位模型。`after/` 负责观测数据上的检测和人工复核，本目录保留训练、验证和预览代码。

当前生产模型为 [`models/best_model_yolo11n_ema.pth`](../models/best_model_yolo11n_ema.pth)，
SHA-256：

```text
9BEEF810651B7B4B793A0DD85DFBB0E0959406BAE4B8D322313C841791E830FA
```

`data/` 是完整的本地数据工作区，可保存 H5、标注、核验图片和数据转换 Python
脚本；整个目录由 `.gitignore` 排除。`logs/` 保存训练输出，其中 checkpoint
不进入 Git；选定的生产 checkpoint 统一复制到 `../models/`。

## 数据约定

训练 H5 的核心字段为：

- `images`：`(N, 512, 512)`，按 `(time, freq)` 保存；
- `annotations`：`(image_idx, x_freq, y_time, w_freq, h_time)`；
- `original_date`：按观测日期隔离 train 和 validation。

加载时会把图像与标注转换为 AFTER 模型坐标 `(freq, time)`，即
`x=time, y=freq`。模型输入固定为 512×512。

## 使用

在 AFTER 仓库根目录安装训练依赖：

```bash
python -m pip install -r requirements-training.txt
```

将准备好的训练 H5 放在 `training/data/`。本地数据转换脚本也统一保存在该目录。

训练 YOLO11n：

```bash
python training/yolo_train.py yolo11n \
  --device 0 \
  --epochs 100 \
  --batch-size 64
```

也可以使用单卡脚本：

```bash
BATCH_SIZE=64 EPOCHS=100 bash training/train.sh "0" yolo11n
```

预览验证集：

```bash
python training/yolo_infer.py yolo11n \
  --conf 0.3 \
  --start 0 \
  --end 30
```

`logs/logs_yolo11n_multi_hard/logs_yolo11n.json` 是当前生产模型的历史训练记录，
来自旧的帧级随机验证划分。当前训练代码按 `original_date` 分组切分，新训练应使用
新的日志结果。

| 文件 | 作用 |
|---|---|
| `yolo_data.py` | H5 读取、日期分组切分、数据增强和坐标转换。 |
| `yolo_train.py` | 单 GPU 训练、EMA、验证和 checkpoint 保存。 |
| `yolo_eval.py` | 训练期间使用的检测指标。 |
| `yolo_infer.py` | 使用生产 checkpoint 预览验证集。 |
