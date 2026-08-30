#!/usr/bin/env bash
# AFTER burst detector YOLO 单卡训练 wrapper。
#
# Usage:
#   ./train.sh "<gpu_id>" <model>
#
# Examples:
#   ./train.sh "0" yolo11n
#   ./train.sh "1" yolo26s
#   BATCH_SIZE=32 EPOCHS=200 ./train.sh "0" yolo11n
#
# Models:
#   任何 ultralytics 支持的 yolo* (yolo11n/s/m/l/x, yolo26n/s/...)
#   model name 透传给 yolo_train.py，需要对应的 <model>.yaml 在 ultralytics 内即可。
#   训练入口默认加载同名 <model>.pt 预训练权重。

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir"

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 \"<gpu_id>\" <model>" >&2
  echo "  gpu_id : 单个 GPU 编号（本脚本是单卡训练，不支持多卡）" >&2
  echo "  model  : yolo* (yolo11n / yolo11s / yolo26n / yolo26s / ...)" >&2
  exit 1
fi

gpu_id=$1
model=$2

# 通过环境变量覆盖默认配置；不设置就用 yolo_train.py 的默认
data_path=${DATA_PATH:-./data/}
batch_size=${BATCH_SIZE:-64}
epochs=${EPOCHS:-100}
workers=${NUM_WORKERS:-8}
patience=${PATIENCE:-30}
mosaic_prob=${MOSAIC_PROB:-0.5}
ema_decay=${EMA_DECAY:-0.999}
ema_tau=${EMA_TAU:-1000}
eval_conf_thr=${EVAL_CONF_THR:-0.01}
nms_iou_thr=${NMS_IOU_THR:-0.65}
match_iou_thr=${MATCH_IOU_THR:-0.5}
log_dir=${LOG_DIR:-./logs/logs_${model}_multi_hard/}

# 让 CUDA_VISIBLE_DEVICES 的编号和 nvidia-smi 一致（默认 FASTEST_FIRST 会错位）
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$gpu_id"

mkdir -p "$log_dir"

echo "================================================================"
echo "  Model        : $model"
echo "  GPU          : $gpu_id"
echo "  Log dir      : $log_dir"
echo "  Batch/Epochs : $batch_size / $epochs"
echo "  Patience     : $patience"
echo "  Mosaic prob  : $mosaic_prob"
echo "  EMA decay/tau: $ema_decay / $ema_tau"
echo "  Eval conf    : $eval_conf_thr"
echo "  NMS/Match IoU: $nms_iou_thr / $match_iou_thr"
echo "  Started      : $(date)"
echo "================================================================"

# CUDA_VISIBLE_DEVICES 已经把目标 GPU 重映射成 cuda:0
python yolo_train.py "$model" \
  --device         0 \
  --data-path      "$data_path" \
  --log-dir        "$log_dir" \
  --batch-size     "$batch_size" \
  --epochs         "$epochs" \
  --patience       "$patience" \
  --workers        "$workers" \
  --mosaic-prob    "$mosaic_prob" \
  --ema-decay      "$ema_decay" \
  --ema-tau        "$ema_tau" \
  --eval-conf-thr  "$eval_conf_thr" \
  --nms-iou-thr    "$nms_iou_thr" \
  --match-iou-thr  "$match_iou_thr"

echo "[done] $model finished at $(date)"
