# fmt: off

"""AFTER burst detector 验证集可视化推理（GT 绿、预测红）。

使用方式:
    python yolo_infer.py                          # 默认预览 yolo11n
    python yolo_infer.py yolo11n --conf 0.3
    python yolo_infer.py yolo11n --start 10 --end 50
"""

import argparse
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from yolo_data import get_train_val, normalize_image
from yolo_eval import decode_raw
from yolo_train import build_yolo_model, setup_criterion

device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAINING_DIR       = Path(__file__).resolve().parent
DEFAULT_DATA_DIR   = TRAINING_DIR / "data"
DEFAULT_MODEL_PATH = TRAINING_DIR.parent / "models" / "best_model_yolo11n_ema.pth"


# ---------------------------------------------------------------------------
# 1. 模型加载
# ---------------------------------------------------------------------------


def load_model(model_name="yolo11n", model_path=None):
    """加载指定权重；默认使用 AFTER 当前生产 checkpoint。"""
    model_path = Path(model_path or DEFAULT_MODEL_PATH)
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到模型权重: {model_path}")

    model = build_yolo_model(model_name, nc=1, load_pretrained=False)
    model = setup_criterion(model)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device).eval()
    print(f"[OK] loaded {model_path}")
    return model


# ---------------------------------------------------------------------------
# 2. 单图推理：raw → conf 过滤 → NMS → (conf, boxes_cxcywh)
# ---------------------------------------------------------------------------


def predict_single(model, img_float32, conf=0.25, iou_threshold=0.5):
    """对单张已归一化的 (512, 512) 灰度图做推理。

    这里复用 ``yolo_eval.decode_raw``，保证可视化预览和训练期评估走同一套
    YOLO 输出解码 / NMS 逻辑，避免两个入口的阈值或坐标转换悄悄漂移。

    Returns:
        pred_conf: (N,) np.ndarray 或 None
        pred_boxes:  (N, 4) np.ndarray 或 None；YOLO ``[cx, cy, w, h]``（像素）
    """
    img_t = torch.from_numpy(img_float32).float().unsqueeze(0).expand(3, -1, -1)
    img_t = img_t.contiguous().unsqueeze(0).to(device)  # [1, 3, H, W]

    with torch.no_grad():
        decoded = decode_raw(model(img_t), conf_thr=conf, nms_iou_thr=iou_threshold)
    return decoded[0]


# ---------------------------------------------------------------------------
# 3. 可视化：GT 绿框、预测红框，中心用十字标
# ---------------------------------------------------------------------------


def draw_preview(img, bboxes, pred_conf, pred_boxes, sample_idx):
    image_np  = (img * 255).astype(np.uint8)
    image_vis = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

    # GT：绿色
    if isinstance(bboxes, np.ndarray) and bboxes.ndim == 2 and len(bboxes) > 0:
        for x, y, w, h in bboxes.astype(np.int32):
            cv2.rectangle(image_vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.drawMarker(
                image_vis,
                (int(x + w / 2), int(y + h / 2)),
                (0, 255, 0),
                cv2.MARKER_CROSS,
                6,
                1,
            )

    # 预测：红色
    if pred_boxes is not None:
        for bi, (cx, cy, w, h) in enumerate(pred_boxes.astype(np.int32)):
            x1, y1 = int(cx - w / 2), int(cy - h / 2)
            cv2.rectangle(image_vis, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 1)
            cv2.drawMarker(
                image_vis, (int(cx), int(cy)), (0, 0, 255), cv2.MARKER_CROSS, 6, 1
            )
            print(f"  pred {bi}: conf={pred_conf[bi]:.3f}")

    plt.figure(figsize=(5, 5))
    plt.title(f"Validation Sample {sample_idx}")
    plt.imshow(cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


# ---------------------------------------------------------------------------
# 4. 主流程：用同样的 seed/val_fraction 复现训练时的 val 划分
# ---------------------------------------------------------------------------


def main(args):
    model = load_model(args.model, model_path=args.model_path)
    _, val_df = get_train_val(
        args.data_path, val_fraction=args.val_fraction, seed=args.seed
    )

    preview_indices = range(args.start, min(args.end, len(val_df)))
    h5_cache        = {}  # 进程内复用文件句柄
    try:
        for sample_idx in preview_indices:
            row     = val_df.iloc[sample_idx]
            h5_path = row["h5_path"]
            if h5_path not in h5_cache:
                h5_cache[h5_path] = h5py.File(h5_path, "r")

            # H5 保存为 (time, freq)，AFTER 模型输入为 (freq, time)。
            img    = h5_cache[h5_path]["images"][row["img_idx"]].T.copy()
            img    = normalize_image(img)
            bboxes = row["bboxes"][:, [1, 0, 3, 2]].copy()

            pred_conf, pred_boxes = predict_single(
                model,
                img,
                conf          = args.conf,
                iou_threshold = args.iou_threshold,
            )
            n_pred = 0 if pred_conf is None else len(pred_conf)
            print(f"Sample {sample_idx}: {n_pred} predictions")
            draw_preview(img, bboxes, pred_conf, pred_boxes, sample_idx)
    finally:
        for f in h5_cache.values():
            f.close()


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AFTER burst detector 推理 (直接读 H5 验证集)"
    )
    parser.add_argument(
        "model",
        type    = str,
        nargs   = "?",
        default = "yolo11n",
        help    = "模型名 (yolo11n/s/m/l/x、yolo26* 等)；默认 yolo11n",
    )
    parser.add_argument(
        "--model-path",
        type    = str,
        default = str(DEFAULT_MODEL_PATH),
        help    = "checkpoint 路径；默认使用 AFTER 当前生产模型",
    )
    parser.add_argument(
        "--data-path", type=str, default=str(DEFAULT_DATA_DIR), help="H5 数据集目录"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="须与训练时一致以复现同样的 val 划分"
    )
    parser.add_argument(
        "--val-fraction",
        type    = float,
        default = 0.2,
        help    = "须与训练时一致以复现同样的 val 划分",
    )
    parser.add_argument("--conf", type=float, default=0.3, help="置信度阈值")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="NMS IoU 阈值")
    parser.add_argument("--start", type=int, default=0, help="预览起始索引")
    parser.add_argument("--end", type=int, default=30, help="预览结束索引")
    main(parser.parse_args())

# fmt: on
