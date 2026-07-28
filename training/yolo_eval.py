"""AFTER burst detector YOLO 目标检测评估。

评估口径：
  - 模型输出先解码为预测框与置信度；
  - 同图重复框通过 NMS 合并；
  - 预测框按置信度顺序与未匹配 GT 做 IoU 贪心匹配；
  - P/R/F1 使用 ``match_iou_thr``，checkpoint 主指标使用 mAP50-95。
"""

import numpy as np
import torch
from torchvision.ops import box_iou, nms
from tqdm import tqdm


# 训练日志中的检测指标字段；EMA 用原名，RAW 对照项加 ``_raw`` 后缀。
METRIC_KEYS = (
    "precision", "recall", "f1", "f1_conf",
    "ap50", "map50_95",
)

# COCO-style IoU thresholds: 0.50, 0.55, ..., 0.95.
MAP_IOU_THRESHOLDS = tuple(float(x) for x in np.round(np.arange(0.50, 0.96, 0.05), 2))


# ---------------------------------------------------------------------------
# 1. Box conversion and prediction decoding
# ---------------------------------------------------------------------------

def cxcywh_to_xyxy(boxes):
    """Convert ``[cx, cy, w, h]`` boxes to ``[x1, y1, x2, y2]``."""
    cx, cy, w, h = boxes.unbind(1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)


def xyxy_to_cxcywh(boxes):
    """Convert ``[x1, y1, x2, y2]`` boxes to ``[cx, cy, w, h]``."""
    x1, y1, x2, y2 = boxes.unbind(1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], 1)


def _conf_filter_nms(boxes_cxcywh, scores, conf_thr, nms_iou_thr):
    """单图 YOLO11 输出：置信度过滤后做 NMS，返回 ``(conf, boxes_xyxy)``。"""
    keep = scores > conf_thr
    boxes_cxcywh, scores = boxes_cxcywh[keep], scores[keep]
    if scores.numel() == 0:
        return scores, boxes_cxcywh.new_zeros((0, 4))
    xyxy = cxcywh_to_xyxy(boxes_cxcywh)
    nms_keep = nms(xyxy, scores, nms_iou_thr)
    return scores[nms_keep], xyxy[nms_keep]


def _conf_filter_e2e(dets_xyxy_conf_cls, conf_thr):
    """单图 YOLO26 e2e 输出：过滤 ``[xyxy, conf, cls]`` 后按置信度排序。"""
    scores = dets_xyxy_conf_cls[:, 4]
    keep = scores > conf_thr
    scores, xyxy = scores[keep], dets_xyxy_conf_cls[keep, :4]
    if scores.numel() == 0:
        return scores, xyxy.new_zeros((0, 4))
    order = torch.argsort(scores, descending=True)
    return scores[order], xyxy[order]


_DECODE_PATH_LOGGED = False


def _decode_to_xyxy_batches(raw, conf_thr, nms_iou_thr):
    """返回每图 ``(conf, boxes_xyxy)``。

    YOLO11-style 输出是 ``[B, 4 + nc, anchors]``，需要 NMS；YOLO26 e2e
    输出已经是检测框，只需要按置信度过滤和排序。
    """
    global _DECODE_PATH_LOGGED
    if isinstance(raw, (list, tuple)):
        raw = raw[0]
    if raw.ndim != 3:
        raise ValueError(f"Unexpected YOLO output shape: {tuple(raw.shape)}")

    is_e2e = raw.shape[-1] >= 6 and raw.shape[1] > raw.shape[-1]
    if not _DECODE_PATH_LOGGED:
        _DECODE_PATH_LOGGED = True
        path = "YOLO26 e2e (no NMS)" if is_e2e else "YOLO11 traditional (with NMS)"
        print(f"[YOLO decode] raw shape={tuple(raw.shape)} -> {path}")

    if is_e2e:
        return [_conf_filter_e2e(raw[i], conf_thr) for i in range(raw.shape[0])]

    preds = raw.permute(0, 2, 1)
    return [
        _conf_filter_nms(preds[i, :, :4], preds[i, :, 4], conf_thr, nms_iou_thr)
        for i in range(preds.shape[0])
    ]


def decode_raw(raw, conf_thr, nms_iou_thr):
    """解码 ``DetectionModel`` 输出，供可视化推理入口使用。

    Returns:
        list[tuple]: 每图 ``(conf_np, boxes_cxcywh_np)``；无检测时为 ``(None, None)``。
    """
    results = []
    for scores, xyxy in _decode_to_xyxy_batches(raw, conf_thr, nms_iou_thr):
        if scores.numel() == 0:
            results.append((None, None))
        else:
            results.append((scores.cpu().numpy(), xyxy_to_cxcywh(xyxy).cpu().numpy()))
    return results


# ---------------------------------------------------------------------------
# 2. IoU matching and AP calculation
# ---------------------------------------------------------------------------

def _greedy_iou_tp(iou_matrix, threshold):
    """按置信度顺序做一对一 IoU 贪心匹配。

    每个预测最多匹配一个 GT；每个 GT 被匹配后不再参与后续预测。
    """
    n_pred, n_gt = iou_matrix.shape
    tp = np.zeros(n_pred, dtype=np.float32)
    if n_pred == 0 or n_gt == 0:
        return tp

    taken = set()
    for pi in range(n_pred):
        candidates = iou_matrix[pi].clone()
        if taken:
            candidates[list(taken)] = -1
        best_iou, gi = candidates.max(0)
        gi = int(gi)
        if best_iou.item() >= threshold:
            tp[pi] = 1.0
            taken.add(gi)
    return tp


def _pr_curve(all_tp, all_conf, total_gt):
    """Build a dataset-level precision/recall curve from per-prediction TP flags."""
    if not all_tp or total_gt == 0:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty, empty, empty

    tp = np.concatenate(all_tp)
    conf = np.concatenate(all_conf)
    order = np.argsort(-conf)
    tp, conf = tp[order], conf[order]

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(1 - tp)
    recall = cum_tp / total_gt
    precision = cum_tp / (cum_tp + cum_fp + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1, conf


def _average_precision(precision, recall):
    """Compute interpolated AP from a PR curve.

    This is the standard area under the precision envelope. The recall endpoints
    at 0 and 1 ensure missed GT contributes zero area instead of disappearing.
    """
    if precision.size == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    changed = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changed + 1] - mrec[changed]) * mpre[changed + 1]))


def _pr_metrics_at_threshold(all_tp, all_conf, total_gt, iou_thr):
    """返回指定 IoU 阈值下、最优 F1 置信度工作点的 P/R/F1。"""
    precision, recall, f1, conf = _pr_curve(all_tp, all_conf, total_gt)
    if precision.size == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "f1_conf": 0.0,
            "match_iou_thr": float(iou_thr),
        }

    best = int(np.argmax(f1))
    return {
        "precision": float(precision[best]),
        "recall": float(recall[best]),
        "f1": float(f1[best]),
        "f1_conf": float(conf[best]),
        "match_iou_thr": float(iou_thr),
    }


def _ap_from_matches(all_tp, all_conf, total_gt):
    """由整体验证集匹配结果计算插值 AP。"""
    precision, recall, _, _ = _pr_curve(all_tp, all_conf, total_gt)
    return _average_precision(precision, recall)


# ---------------------------------------------------------------------------
# 3. Main evaluation entry
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_metrics(model, loader, device, imgsz,
                     conf_thr=0.01, nms_iou_thr=0.65, match_iou_thr=0.5):
    """用标准 IoU/AP 目标检测指标评估验证集。

    Args:
        model: YOLO DetectionModel on ``device``.
        loader: validation DataLoader yielding normalized YOLO ``cxcywh`` GT.
        imgsz: image side length in pixels, used to restore GT boxes to pixels.
        conf_thr: 低置信度过滤阈值，用于保留 PR 曲线上的候选点。
        nms_iou_thr: 同图 NMS IoU 阈值。
        match_iou_thr: P/R/F1 的 IoU 匹配阈值。

    Returns:
        ``precision``、``recall``、``f1``、``f1_conf``、``ap50``、
        ``map50_95`` 和 ``match_iou_thr``。
    """
    model.eval()
    eval_thresholds = tuple(sorted(set(MAP_IOU_THRESHOLDS + (float(match_iou_thr),))))
    all_tp = {thr: [] for thr in eval_thresholds}
    all_conf = []
    total_gt = 0

    for batch in tqdm(loader, dynamic_ncols=True, ascii=True, desc="yolo eval"):
        imgs = batch["img"].to(device, non_blocking=True)
        gt_bboxes, gt_batch_idx = batch["bboxes"], batch["batch_idx"]

        raw = model(imgs)
        decoded = _decode_to_xyxy_batches(raw, conf_thr, nms_iou_thr)

        for i in range(imgs.shape[0]):
            gt_i = gt_bboxes[gt_batch_idx == i]
            total_gt += gt_i.shape[0]

            scores, pred_xyxy = decoded[i]
            n_pred = scores.shape[0]
            if n_pred == 0:
                continue

            all_conf.append(scores.cpu().numpy())
            if gt_i.shape[0] > 0:
                gt_cxcywh = gt_i.to(device) * imgsz
                gt_xyxy = cxcywh_to_xyxy(gt_cxcywh)
                iou_mat = box_iou(pred_xyxy, gt_xyxy)
                for thr in eval_thresholds:
                    all_tp[thr].append(_greedy_iou_tp(iou_mat, thr))
            else:
                zeros = np.zeros(n_pred, dtype=np.float32)
                for thr in eval_thresholds:
                    all_tp[thr].append(zeros)

    match_thr = float(match_iou_thr)
    metrics = _pr_metrics_at_threshold(all_tp[match_thr], all_conf, total_gt, match_thr)
    ap_by_thr = {
        thr: _ap_from_matches(all_tp[thr], all_conf, total_gt)
        for thr in MAP_IOU_THRESHOLDS
    }
    metrics["ap50"] = float(ap_by_thr[0.50])
    metrics["map50_95"] = float(np.mean(list(ap_by_thr.values()))) if ap_by_thr else 0.0
    return metrics
