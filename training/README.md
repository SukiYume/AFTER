# AFTER Burst Detector Training

[简体中文](README.zh-CN.md)

This directory trains the YOLO burst-localization model used by
[`after/burst_detect.py`](../after/burst_detect.py). The `after/` package owns
inference and human review on calibrated observations; `training/` owns model
data loading, training, validation, and preview.

Complete the [main installation](../README.md#installation), then install the
additional training dependencies from the repository root:

```bash
python -m pip install -r requirements-training.txt
```

## Directory roles

| Path | Purpose |
|---|---|
| `yolo_data.py` | Read H5 datasets, split by observation date, augment images, and convert coordinates. |
| `yolo_train.py` | Single-GPU training, EMA, validation, early stopping, and checkpoint export. |
| `yolo_eval.py` | Detection metrics used during training. |
| `yolo_infer.py` | Preview validation images with a selected checkpoint. |
| `train.sh` | Bash wrapper for one-GPU training. |
| `data/` | Ignored local workspace for H5, annotations, review images, and preparation scripts. |
| `logs/` | Training runs and metrics; checkpoints remain untracked until a model is promoted. |

## Production checkpoint

The deployed checkpoint is
[`../models/best_model_yolo11n_ema.pth`](../models/best_model_yolo11n_ema.pth).
Its verified SHA-256 and runtime command are maintained in the
[main README](../README.md#models-and-outputs).

The tracked
`logs/logs_yolo11n_multi_hard/logs_yolo11n.json` records the historical run
that produced the current model. That run used a legacy frame-level validation
split. Current training groups samples by `original_date`; use the current
date-grouped results when comparing new checkpoints.

## Data contract

Place training H5 files under `training/data/` or pass another directory with
`--data-path` or `DATA_PATH`. Each H5 provides:

- `images`: `(N, 512, 512)` stored as `(time, frequency)`;
- `annotations`: `(image_idx, x_freq, y_time, w_freq, h_time)`;
- `original_date`: observation identifier used to isolate training and
  validation groups.

The loader converts annotations into the model coordinate system
`(x=time, y=frequency)`. Model input is fixed at 512 × 512.

## Train

Run from the AFTER repository root:

```bash
python training/yolo_train.py yolo11n \
  --data-path /path/to/training_h5 \
  --device 0 \
  --epochs 100 \
  --batch-size 64
```

The Bash wrapper exposes the same common settings through environment
variables:

```bash
DATA_PATH=/path/to/training_h5 \
BATCH_SIZE=64 \
EPOCHS=100 \
bash training/train.sh "0" yolo11n
```

Use a separate log directory for each model or dataset comparison.

## Preview and promote

Preview the date-grouped validation split:

```bash
python training/yolo_infer.py yolo11n \
  --model-path /path/to/checkpoint.pth \
  --data-path /path/to/training_h5 \
  --conf 0.3 \
  --start 0 \
  --end 30
```

Select a production candidate using the current validation split, held-out
observation checks, and end-to-end detection behavior. Copy the chosen EMA
checkpoint to `models/best_model_yolo11n_ema.pth`, then update its hash in both
main READMEs and rerun the repository validation commands.
