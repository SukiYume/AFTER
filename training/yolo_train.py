"""AFTER burst detector YOLO 训练（单 GPU 入口）。

训练流程：
  - timm cosine + warmup 调度
  - 训练步后更新 EMA，并同时记录 EMA / RAW 验证指标
  - 每 epoch 跑一次验证集 loss + 标准 IoU/AP 检测评估
  - best_model.pth 对应 EMA mAP50-95 最高的 checkpoint
  - best_loss_model.pth 对应验证 loss 最低的 checkpoint
  - JSON 日志记录 loss、P/R/F1、AP50、mAP50-95

启动方式：
  * 单卡：``python yolo_train.py yolo11n``
  * 经 shell wrapper：``./train.sh "0" yolo11n``

支持的 model name 由 ultralytics 决定（``yolo11n/s/m/l/x``、``yolo26*`` 等）；
本脚本只是把 model name 透传给 ``DetectionModel(f"{name}.yaml", ...)``。
"""

import argparse
import json
import os
from pathlib import Path

import torch
from timm.scheduler import CosineLRScheduler
from tqdm import tqdm
from ultralytics.utils.torch_utils import ModelEMA

from yolo_data import IMAGE_SIZE, H5YOLODataset, get_train_val, yolo_collate_fn
from yolo_eval import METRIC_KEYS, evaluate_metrics


LOSS_KEYS = ("loss", "box", "cls", "dfl")
TRAINING_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = TRAINING_DIR / 'data'


# ---------------------------------------------------------------------------
# 1. 通用辅助：设备 / 默认 log dir
# ---------------------------------------------------------------------------

def select_device(device_arg):
    """选择 CPU 或单张 CUDA GPU；显式请求不可用 GPU 时直接失败。"""
    if not device_arg:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device_arg.lower() == 'cpu':
        return torch.device('cpu')
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device_arg} was requested, but CUDA is unavailable")
    try:
        device_idx = int(device_arg)
    except ValueError as exc:
        raise ValueError("--device must be a single GPU index or 'cpu'") from exc
    if not 0 <= device_idx < torch.cuda.device_count():
        raise ValueError(
            f"CUDA device index {device_idx} is outside 0..{torch.cuda.device_count() - 1}"
        )
    return torch.device(f'cuda:{device_idx}')


def default_log_dir(model_name):
    """默认输出目录；``multi_hard`` 后缀表示训练集对多信号 hard case 做了重复采样。"""
    return str(TRAINING_DIR / 'logs' / f'logs_{model_name}_multi_hard')


# ---------------------------------------------------------------------------
# 2. 模型 + 损失：ultralytics DetectionModel + COCO 预训练权重
# ---------------------------------------------------------------------------

def build_yolo_model(model_name='yolo11n', nc=1, load_pretrained=True):
    """构建 DetectionModel（3 通道输入，类别数 ``nc``）；可选加载 ultralytics 预训练。

    AFTER detector 输入在 dataset 内从灰度复制为 3 通道；预训练权重按 tensor shape
    匹配迁移，检测头这类形状不同的层保持随机初始化。
    """
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel(f'{model_name}.yaml', ch=3, nc=nc, verbose=False)

    if load_pretrained:
        from ultralytics import YOLO
        print(f'[Model] Loading {model_name}.pt ...')
        pre_dict   = YOLO(f'{model_name}.pt').model.state_dict()
        model_dict = model.state_dict()
        # 形状能对齐的权重才迁移；不能对齐的（如检测头通道数差异）保持随机初始化
        valid_dict = {k: v for k, v in pre_dict.items()
                      if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(valid_dict)
        model.load_state_dict(model_dict)
        print(f'[Model] Loaded {len(valid_dict)}/{len(model_dict)} layers.')

    return model


def setup_criterion(model):
    """初始化 DetectionModel 内置的 box / cls / dfl loss 权重。"""
    from ultralytics.cfg import get_cfg

    if not hasattr(model, 'args') or model.args is None:
        model.args = get_cfg()
    model.args.box = 7.5
    model.args.cls = 0.5
    model.args.dfl = 1.5
    model.init_criterion()
    return model


# ---------------------------------------------------------------------------
# 3. 数据加载
# ---------------------------------------------------------------------------

def make_loaders(args):
    empty_repeat, single_repeat, multi_repeat = args.repeat_policy
    train_df, val_df = get_train_val(
        args.data_path,
        val_fraction=args.val_fraction,
        seed=args.seed,
        empty_repeat=empty_repeat,
        single_burst_repeat=single_repeat,
        multi_burst_repeat=multi_repeat,
    )
    print(f'[Data] train={len(train_df)} rows  |  val={len(val_df)} rows')
    print(f'[Data] train repeat policy: empty={empty_repeat}, single={single_repeat}, multi={multi_repeat}')

    train_data = H5YOLODataset(train_df, val=False, mosaic_prob=args.mosaic_prob)
    val_data   = H5YOLODataset(val_df, val=True)

    loader_kwargs = dict(
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=yolo_collate_fn,
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, **loader_kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, **loader_kwargs,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# 4. 单 epoch 训练 / 验证循环
# ---------------------------------------------------------------------------

def run_epoch(model, loader, device, optimizer=None, ema=None):
    """跑一个 epoch；返回 ``LOSS_KEYS`` 对应的平均指标。``optimizer=None`` 表示验证。"""
    is_train = optimizer is not None
    model.train(is_train)

    totals = {k: 0.0 for k in LOSS_KEYS}
    total_samples = 0
    tag = 'Train' if is_train else 'Valid'
    pbar = tqdm(loader, dynamic_ncols=True, ascii=True)

    with torch.set_grad_enabled(is_train):
        for idx, batch in enumerate(pbar, start=1):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            current_batch_size = batch['img'].shape[0]
            total_samples += current_batch_size

            if is_train:
                optimizer.zero_grad()
            loss, loss_items = model(batch)
            if loss.ndim > 0:
                loss = loss.sum()
            if is_train:
                loss.backward()
                optimizer.step()
                if ema is not None:
                    ema.update(model)

            # ultralytics DetectionModel：
            #   loss       = batch 内所有样本 loss 之和
            #   loss_items 通常 = [box, cls, dfl]；YOLO26 / 其它变体可能更长，多余位忽略
            totals['loss'] += loss.item()
            n_items = len(loss_items) if hasattr(loss_items, '__len__') else 0
            if n_items >= 1: totals['box'] += loss_items[0].item() * current_batch_size
            if n_items >= 2: totals['cls'] += loss_items[1].item() * current_batch_size
            if n_items >= 3: totals['dfl'] += loss_items[2].item() * current_batch_size

            pbar.set_description(
                f"{tag} [loss={totals['loss']/total_samples:.3f}]"
                f"[box={totals['box']/total_samples:.4f}]"
                f"[cls={totals['cls']/total_samples:.4f}]"
                f"[dfl={totals['dfl']/total_samples:.4f}]"
            )

    averages = {k: totals[k] / total_samples for k in ("box", "cls", "dfl")}
    averages["loss"] = totals["loss"] / total_samples
    return averages


def evaluate_model_pair(model, ema, val_loader, device, args):
    """评估 EMA 与 RAW 两套权重。

    EMA 是 checkpoint 选择口径；RAW 指当前未平滑权重，用 ``*_raw`` 字段写入日志。
    """
    eval_kwargs = dict(
        imgsz=IMAGE_SIZE,
        conf_thr=args.eval_conf_thr,
        nms_iou_thr=args.nms_iou_thr,
        match_iou_thr=args.match_iou_thr,
    )
    metrics = evaluate_metrics(ema.ema, val_loader, device, **eval_kwargs)
    metrics_raw = evaluate_metrics(model, val_loader, device, **eval_kwargs)
    return metrics, metrics_raw


def _format_eval_metrics(tag, metrics):
    """格式化一套检测指标，保持终端输出与 JSON 字段一致。"""
    return (
        f'  [{tag}] IoU@{metrics["match_iou_thr"]:.2f}  '
        f'F1={metrics["f1"]:.4f} @ conf={metrics["f1_conf"]:.3f}  '
        f'P={metrics["precision"]:.4f}  R={metrics["recall"]:.4f}\n'
        f'        AP50={metrics["ap50"]:.4f}  mAP50-95={metrics["map50_95"]:.4f}'
    )


def print_eval_metrics(metrics, metrics_raw):
    """打印 EMA 主指标和 RAW 对照指标。"""
    print(_format_eval_metrics('EMA', metrics))
    print(_format_eval_metrics('RAW', metrics_raw))


def _prefixed(values, suffix, keys):
    return {f'{k}_{suffix}': values[k] for k in keys}


def build_log_entry(epoch, lr, train_m, val_m, metrics, metrics_raw):
    """组装单个 epoch 的 JSON 日志行。

    字段分为优化指标、EMA 检测指标和 RAW 检测指标；主循环只负责传入三类结果。
    """
    return {
        'epoch': epoch,
        'lr': lr,
        **_prefixed(train_m, 'train', LOSS_KEYS),
        **_prefixed(val_m, 'val', LOSS_KEYS),
        **{k: metrics[k] for k in METRIC_KEYS},
        **{f'{k}_raw': metrics_raw[k] for k in METRIC_KEYS},
        'match_iou_thr': metrics['match_iou_thr'],
    }


def save_model_pair(log_dir, model, ema, stem):
    """按同一 stem 保存当前权重和 EMA 权重。"""
    torch.save(model.state_dict(), os.path.join(log_dir, f'{stem}.pth'))
    torch.save(ema.ema.state_dict(), os.path.join(log_dir, f'{stem}_ema.pth'))


# ---------------------------------------------------------------------------
# 5. 主流程
# ---------------------------------------------------------------------------

def main(args):
    device  = select_device(args.device)
    log_dir = os.path.abspath(args.log_dir) if args.log_dir else default_log_dir(args.model)
    os.makedirs(log_dir, exist_ok=True)

    print(f'[Setup] device={device}  log_dir={log_dir}')

    train_loader, val_loader = make_loaders(args)

    # ---- 模型 ----
    model = build_yolo_model(args.model, nc=1, load_pretrained=True)
    model = setup_criterion(model).to(device)
    # EMA 的有效 decay 在前 ~tau 步内从 0 爬升到 ema_decay。
    ema = ModelEMA(model, decay=args.ema_decay, tau=args.ema_tau)

    # ---- 优化器 / 调度器 ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.epochs,
        lr_min=args.lr_min,
        warmup_t=args.warmup_epochs,
        warmup_lr_init=args.warmup_lr,
        warmup_prefix=True,
    )

    # ---- 训练循环（mAP50-95 选 best；patience 触发 early stop）----
    logs = []
    best_map = -float('inf')
    best_val = float('inf')
    early_stop_counter = 0
    log_path = os.path.join(log_dir, f'logs_{args.model}.json')

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print(f'\nEpoch {epoch}/{args.epochs}  |  LR {current_lr:.6f}')

        train_m = run_epoch(model, train_loader, device, optimizer=optimizer, ema=ema)
        val_m   = run_epoch(model, val_loader, device)
        metrics, metrics_raw = evaluate_model_pair(model, ema, val_loader, device, args)

        print(f'Epoch {epoch}/{args.epochs}  '
              f'TrainLoss {train_m["loss"]:.5f}  ValLoss {val_m["loss"]:.5f}')
        print_eval_metrics(metrics, metrics_raw)

        logs.append(build_log_entry(epoch, current_lr, train_m, val_m, metrics, metrics_raw))
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2)

        # best_loss_model 按验证 loss，best_model 按 EMA mAP50-95。
        if val_m['loss'] <= best_val:
            best_val = val_m['loss']
            save_model_pair(log_dir, model, ema, 'best_loss_model')
            print(f'  val_loss new best = {best_val:.5f}, saved best_loss_model.pth / best_loss_model_ema.pth')

        if metrics['map50_95'] > best_map:
            best_map = metrics['map50_95']
            early_stop_counter = 0
            save_model_pair(log_dir, model, ema, 'best_model')
            print(f'  ★ EMA mAP50-95 new best = {best_map:.5f}, saved best_model.pth / best_model_ema.pth')
        else:
            early_stop_counter += 1
            print(f'  (no improvement; patience {early_stop_counter}/{args.patience})')
            if early_stop_counter >= args.patience:
                print('[Early Stop] patience exhausted, stopping.')
                break

        scheduler.step(epoch)


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AFTER burst detector YOLO H5 training')

    parser.add_argument('model', type=str, nargs='?', default='yolo11n',
                        help='model name (yolo11n/s/m/l/x、yolo26* 等)；默认 yolo11n')

    # paths
    parser.add_argument('--data-path', type=str, default=str(DEFAULT_DATA_DIR),
                        help=f'H5 训练数据目录（默认 {DEFAULT_DATA_DIR}）')
    parser.add_argument('--log-dir', type=str, default='',
                        help='输出目录；为空则使用 training/logs/logs_<model>_multi_hard/')

    # hardware
    parser.add_argument('--device', type=str, default='0',
                        help='单张 CUDA GPU 编号（如 0）或 cpu')
    parser.add_argument('--workers', type=int, default=8, help='DataLoader workers')

    # data
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--mosaic-prob', type=float, default=0.5)
    parser.add_argument('--repeat-policy', type=int, nargs=3, default=[1, 3, 6],
                        metavar=('EMPTY', 'SINGLE', 'MULTI'),
                        help='train 重采样倍数：无标注 / 单信号 / 多信号')

    # training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=30, help='early stop patience')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr-min', type=float, default=1e-6)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--warmup-lr', type=float, default=1e-5)
    parser.add_argument('--ema-decay', type=float, default=0.999)
    parser.add_argument('--ema-tau', type=int, default=1000,
                        help='EMA decay 的爬升步数；越小影子权重跟得越快')

    # eval
    parser.add_argument('--eval-conf-thr', type=float, default=0.01,
                        help='低置信度过滤阈值；用于展开完整 PR 曲线')
    parser.add_argument('--nms-iou-thr', type=float, default=0.65,
                        help='同图内 NMS IoU 阈值')
    parser.add_argument('--match-iou-thr', type=float, default=0.5,
                        help='P/R/F1 使用的 IoU 匹配阈值；AP50/mAP50-95 固定按标准阈值计算')

    main(parser.parse_args())
