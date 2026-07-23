"""
爆发检测流水线 (YOLO)

从 calibration.py 输出的 *_cal.h5 中定位爆发。
支持 auto（全自动）和 semi-auto（模型给初始框、人工确认/修改）两种模式。

用法:
    python burst_detect.py                          # 自动模式
    python burst_detect.py --mode semi-auto         # 半自动交互
    python burst_detect.py --conf 0.3               # 自定义置信度阈值

输出:
    detections.json  — 每个文件的爆发区域列表
    plots/           — 每个文件的检测诊断图
"""

import os
import json
import glob
import argparse

import torch
import numpy as np
import h5py
import matplotlib
# 导入 seaborn 会向 Matplotlib 注册 ``mako`` 颜色映射。
import seaborn  # noqa: F401
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
from scipy.ndimage import zoom
from torchvision.ops import nms

from ultralytics.nn.tasks import DetectionModel
from ultralytics.cfg import get_cfg

from rfi_utils import cal_rfi

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_MAX_HORIZONTAL_ASPECT = 3.0


def load_yolo_model(model_path, model_name='yolo11n'):
    """构建 YOLOv11 并加载训练好的权重文件 (.pth)。

    Parameters
    ----------
    model_path : str
        权重文件路径，例如 ``best_model_yolo11n_ema.pth``。
    model_name : str
        训练时使用的模型配置名（如 'yolo11n'），需要与权重匹配。

    Returns
    -------
    model : DetectionModel
        推理模式、已放到 *device* 上的模型。
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型权重: {model_path}")

    # 按配置 YAML 构建模型结构，nc=1 为单类别（爆发）
    model = DetectionModel(f'{model_name}.yaml', ch=3, nc=1, verbose=False)

    # init_criterion 是为了让 state_dict 的 key 与训练时保存的一致
    if not hasattr(model, 'args') or model.args is None:
        model.args = get_cfg()
    model.args.box = 7.5
    model.args.cls = 0.5
    model.args.dfl = 1.5
    model.init_criterion()

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f'[OK] 模型已加载: {model_path}')
    return model


def _fill_nonfinite(data):
    """把 NaN/inf 替换为全局中值，避免下采样和模型输入出现非有限值。

    calibration 阶段可能会把 RFI 或坏通道标成 NaN。YOLO 推理和 matplotlib
    显示都不需要保留 NaN 的语义，因此这里统一用有限值中值补齐。这个函数只
    返回副本，不会修改传入的 H5 数据。
    """
    arr = np.asarray(data, dtype=np.float32).copy()
    bad = ~np.isfinite(arr)
    if np.any(bad):
        good = arr[~bad]
        fill = float(np.median(good)) if good.size else 0.0
        arr[bad] = fill
    return arr


def normalize_image(data, pmin=5, pmax=95):
    """把动态谱按频率通道均值归一化，并百分位裁剪到 [0, 1]。

    统一约定：本脚本内部的动态谱数据保持 ``(time, freq)`` 形状。
    归一化步骤固定为：
      1. 沿时间维度求每个频率通道的平均值；
      2. 每个频率通道除以自己的时间平均值；
      3. 对归一化后的二维图做 5%–95% 百分位 clip；
      4. 线性映射到 [0, 1]，作为模型输入或画图图像。

    这样模型检测、自动/半自动诊断图、纯手工标注窗口都使用完全相同的亮度
    标定，避免不同模式之间出现肉眼看到和模型看到不一致的问题。

    调用方通常会加 .T 转成 (freq, time) 图像坐标用于 imshow / YOLO 输入。

    Parameters
    ----------
    data : ndarray (nsamp, nchan)
        Stokes I 动态谱，第一维是时间，第二维是频率通道。
    pmin, pmax : float
        裁剪的下/上百分位，默认 5%–95%。

    Returns
    -------
    ndarray (nsamp, nchan) float32，取值 [0, 1]
    """
    img = _fill_nonfinite(data)

    # 每个频率通道用自己的时间平均值归一化。denom 形状为 (1, nchan)，
    # 会自动 broadcast 到 (nsamp, nchan)。
    denom    = np.mean(img, axis=0, keepdims=True)
    valid    = np.isfinite(denom) & (np.abs(denom) > 1e-8)
    fallback = float(np.median(denom[valid])) if np.any(valid) else 1.0
    if not np.isfinite(fallback) or abs(fallback) <= 1e-8:
        fallback = 1.0
    denom = np.where(valid, denom, fallback)
    img   = img / denom

    vmin, vmax = np.percentile(img, (pmin, pmax))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(img, dtype=np.float32)

    img = np.clip(img, vmin, vmax)
    img = (img - vmin) / (vmax - vmin + 1e-8)
    return img.astype(np.float32)


def write_detection_results(h5_path, iquv, burst_regions, rfi_fft=False):
    """写入已确认的 burst 区域, 并据此计算/保存检测后 RFI。

    这段逻辑刻意跟 burst_analysis.py 保持一致: 非 burst 采样作为
    noise_mask, 先按噪声段中值减基线, 再分别在 Stokes I / V 上找 RFI,
    最后取二者并集。不覆盖 calibration 阶段的 rfi_mask / rfi_channel。
    """
    work = np.asarray(iquv, dtype=np.float32).copy()
    nsamp = work.shape[1]
    nchan = work.shape[2]

    noise_mask = np.ones(nsamp, dtype=bool)
    for region in burst_regions:
        ts = max(0, min(nsamp, int(region['time_start'])))
        te = max(0, min(nsamp, int(region['time_end'])))
        if te > ts:
            noise_mask[ts:te] = False
    if not np.any(noise_mask):
        noise_mask[:] = True

    baseline = np.nanmedian(work[:, noise_mask, :], axis=1, keepdims=True)
    work = work - baseline

    rfi_channel_i, rfi_pixel_i = cal_rfi(work[0], noise_mask,
                                         down_time=1, down_freq=1,
                                         fft=rfi_fft)
    if work.shape[0] > 3:
        rfi_channel_v, rfi_pixel_v = cal_rfi(work[3], noise_mask,
                                             down_time=1, down_freq=1,
                                             fft=rfi_fft)
    else:
        rfi_channel_v = np.zeros(nchan, dtype=bool)
        rfi_pixel_v = np.zeros((nsamp, nchan), dtype=bool)

    rfi_channel = rfi_channel_i | rfi_channel_v
    rfi_pixel = rfi_pixel_i | rfi_pixel_v
    rfi_mask = rfi_pixel.copy()
    rfi_mask[:, rfi_channel] = True
    rfi_fraction = float(np.sum(rfi_mask) / rfi_mask.size) if rfi_mask.size else 0.0
    plot_I = work[0].copy()
    plot_I[rfi_mask] = np.nan

    with h5py.File(h5_path, 'a') as f:
        f.attrs['bursts'] = json.dumps(burst_regions, ensure_ascii=False)
        for name, data in (
                ('burst_rfi_mask', rfi_mask),
                ('burst_rfi_channel', rfi_channel)):
            if name in f:
                del f[name]
            f.create_dataset(name, data=data)
        f.attrs['burst_rfi_fraction'] = rfi_fraction
        f.attrs['burst_rfi_channel_count'] = int(np.sum(rfi_channel))
        f.attrs['burst_rfi_noise_sample_count'] = int(np.sum(noise_mask))
        f.attrs['burst_rfi_method'] = 'fft' if rfi_fft else 'entropy'
        f.attrs['burst_rfi_source'] = 'burst_detect'
    return rfi_channel, rfi_fraction, plot_I


def prepare_image_tiles(stokes_I, target_size=512, time_factor=None, freq_factor=None):
    """把 Stokes I 下采样并按时间轴切成若干 target_size × target_size 的 YOLO 输入。

    长时间数据 (nsamp_save 远大于 target_size) 时, 单纯压缩到 512 会让窄
    的爆发糊成一团, 模型很难探测. 这里先按 (time_factor, freq_factor) 做
    "清晰下采样" — 默认值跟 calibration 阶段画图用的下采样倍率对齐 —
    再沿时间轴每 target_size 列切成一个 tile, 各 tile 独立送 YOLO 推理,
    最后由 boxes_to_regions_tiled 把各 tile 的像素坐标换算回保存数据的
    采样点 / 通道索引.

    两轴算法对称:
      1. 块均值下采样到 *_factor 整数倍;
      2. 若结果不等于 target_size, 用线性插值精确补到 target_size;
      3. 时间方向再按 target_size 切 tile, 末尾不够的用中值填充.
    步骤 2 让"频率太少 / 太多"和"时间太短"三种情况合并到同一条路径上.

    Parameters
    ----------
    stokes_I : ndarray (nsamp, nchan)
    target_size : int
    time_factor, freq_factor : int or None
        像素 → 保存数据的采样点 / 通道转换倍率. None = 自动 (尽量装进
        单张 target_size × target_size 图).

    Returns
    -------
    tiles : list of ndarray (target_size, target_size) float32  [0, 1]
        每个 tile 已经过 normalize_image，并转成图像坐标 (freq, time)。
    offsets : list of int   每个 tile 的起始时间偏移, 单位 = 保存数据的采样点
    time_factor : int or float   实际像素 → 保存采样点的换算倍率
    freq_factor : int or float   实际像素 → 保存通道的换算倍率
    """
    nsamp, nchan = stokes_I.shape
    data = _fill_nonfinite(stokes_I)

    # ---- 频率轴: 块均值 + (必要时) 线性插值, 最终得到 target_size 通道 ----
    if freq_factor is None:
        freq_factor = max(1, nchan // target_size)
    if freq_factor > 1:
        nc_keep = (nchan // freq_factor) * freq_factor
        data    = data[:, :nc_keep].reshape(nsamp, nc_keep // freq_factor, freq_factor).mean(axis=2)
    if data.shape[1] != target_size:
        # 不能整除或还偏多/偏少时, 都用线性插值精确对齐
        data        = zoom(data, (1, target_size / data.shape[1]), order=1)
        freq_factor = nchan / target_size

    # ---- 时间轴: 块均值下采样, 太短再线性插值补到 target_size ----
    if time_factor is None:
        time_factor = max(1, nsamp // target_size)
    nt_keep = (nsamp // time_factor) * time_factor
    nt_ds   = nt_keep // time_factor
    data    = (data[:nt_keep].reshape(nt_ds, time_factor, target_size).mean(axis=1)
               if time_factor > 1 else data[:nt_keep])
    if nt_ds < target_size:
        data        = zoom(data, (target_size / nt_ds, 1), order=1)
        time_factor = nsamp / target_size
        nt_ds       = target_size

    # ---- 按 target_size 切 tile, 末尾不足的用中值填充 (单 tile 是 n_tiles=1 的特例) ----
    n_tiles = int(np.ceil(nt_ds / target_size))
    med     = float(np.median(data))
    tiles, offsets = [], []
    for k in range(n_tiles):
        start = k * target_size
        end   = start + target_size
        if end <= nt_ds:
            tile = data[start:end]
        else:
            pad  = np.full((end - nt_ds, target_size), med, dtype=data.dtype)
            tile = np.concatenate([data[start:nt_ds], pad], axis=0)
        tiles.append(normalize_image(tile).T)
        offsets.append(start * time_factor)   # 起点偏移 (保存采样点)

    return tiles, offsets, time_factor, freq_factor


def filter_inference_boxes(scores, boxes,
                           max_horizontal_aspect=DEFAULT_MAX_HORIZONTAL_ASPECT):
    """去掉横向伪框；重叠框按面积从大到小只保留最大框。"""
    if scores is None or boxes is None:
        return None, None
    if max_horizontal_aspect <= 0:
        raise ValueError('max_horizontal_aspect must be positive')

    scores = np.asarray(scores)
    boxes = np.asarray(boxes)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f'boxes must have shape (N, 4), got {boxes.shape}')
    if scores.ndim != 1 or len(scores) != len(boxes):
        raise ValueError('scores must have shape (N,) matching boxes')

    width, height = boxes[:, 2], boxes[:, 3]
    valid = (
        np.isfinite(scores)
        & np.all(np.isfinite(boxes), axis=1)
        & (width > 0)
        & (height > 0)
        & (width <= max_horizontal_aspect * height)
    )
    scores, boxes = scores[valid], boxes[valid]
    if not len(boxes):
        return None, None

    half_width, half_height = boxes[:, 2] / 2, boxes[:, 3] / 2
    xyxy = np.column_stack([
        boxes[:, 0] - half_width,
        boxes[:, 1] - half_height,
        boxes[:, 0] + half_width,
        boxes[:, 1] + half_height,
    ])
    order = sorted(
        range(len(boxes)),
        key=lambda i: (float(boxes[i, 2] * boxes[i, 3]), float(scores[i])),
        reverse=True,
    )

    keep = []
    for index in order:
        candidate = xyxy[index]
        overlaps = any(
            min(candidate[2], xyxy[kept][2]) > max(candidate[0], xyxy[kept][0])
            and min(candidate[3], xyxy[kept][3]) > max(candidate[1], xyxy[kept][1])
            for kept in keep
        )
        if not overlaps:
            keep.append(index)

    return scores[keep], boxes[keep]


def predict_single(model, img_float32, conf=0.25, iou_threshold=0.5,
                   max_horizontal_aspect=DEFAULT_MAX_HORIZONTAL_ASPECT):
    """对单张 512×512 灰度图做 YOLO 推理 + NMS。

    Parameters
    ----------
    model : DetectionModel
    img_float32 : ndarray (512, 512) float32，取值 [0, 1]
    conf : float
        置信度阈值，低于此值的预测被过滤。
    iou_threshold : float
        NMS 的 IoU 阈值。
    max_horizontal_aspect : float
        允许的最大模型像素宽高比，默认 3；超过后视为横向 RFI 框。

    Returns
    -------
    pred_scores : ndarray (N,) 或 None
        每个检测框的置信度。
    pred_boxes  : ndarray (N, 4) [cx, cy, w, h] 像素坐标，或 None
        无检测时返回 None。
    """
    # 灰度 → 3 通道 tensor
    img_t = torch.from_numpy(img_float32).float()
    img_t = img_t.unsqueeze(0).expand(3, -1, -1).contiguous()
    img_t = img_t.unsqueeze(0).to(device)  # (1, 3, 512, 512)

    with torch.no_grad():
        preds = model(img_t)

    # 模型输出可能是 tuple（推理模式）或单个 tensor
    pred = preds[0] if isinstance(preds, (list, tuple)) else preds
    pred = pred.permute(0, 2, 1)  # (1, N_anchors, 4+nc) → (1, N, 5)

    box   = pred[0, :, :4]   # [cx, cy, w, h]
    score = pred[0, :, 4]    # 置信度（单类别）

    # 置信度过滤
    keep        = score > conf
    box, score  = box[keep], score[keep]

    if len(box) == 0:
        return None, None

    score_np, box_np = filter_inference_boxes(
        score.cpu().numpy(), box.cpu().numpy(), max_horizontal_aspect)
    if box_np is None:
        return None, None
    box = torch.as_tensor(box_np, dtype=box.dtype, device=box.device)
    score = torch.as_tensor(score_np, dtype=score.dtype, device=score.device)

    # cxcywh → xyxy，用于 NMS
    cx, cy, w, h = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
    boxes_xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)

    keep = nms(boxes_xyxy, score, iou_threshold=iou_threshold)
    return score[keep].cpu().numpy(), box[keep].cpu().numpy()


def boxes_to_regions_tiled(tile_results, time_factor, freq_factor, nsamp, nchan):
    """把多 tile 的 YOLO 像素坐标合成保存数据坐标系下的爆发区域。

    Parameters
    ----------
    tile_results : list of (boxes, scores, time_offset)
        每个 tile 的 YOLO 输出 + 该 tile 在保存采样点轴上的起点偏移.
    time_factor, freq_factor : int or float
        像素 → 保存采样点 / 通道的换算倍率.
    nsamp, nchan : int
        保存数据的维度, 用来 clamp 边界.

    Returns
    -------
    regions : list of dict (time_start/end, freq_start/end, confidence)
    """
    regions = []
    for boxes, scores, t_offset in tile_results:
        if boxes is None:
            continue
        for i in range(len(boxes)):
            cx, cy, w, h = boxes[i]
            t_start = max(0, int((cx - w / 2) * time_factor) + int(t_offset))
            t_end   = min(nsamp, int((cx + w / 2) * time_factor) + int(t_offset))
            f_start = max(0, int((cy - h / 2) * freq_factor))
            f_end   = min(nchan, int((cy + h / 2) * freq_factor))
            regions.append({
                'time_start': t_start,
                'time_end':   t_end,
                'freq_start': f_start,
                'freq_end':   f_end,
                'confidence': float(scores[i]),
            })
    return regions


def _draw_burst_box(ax_profile, ax_spec, t0_ms, t1_ms, f0_mhz, f1_mhz,
                    edge='lime', face='steelblue', alpha=0.2, lw=1.0):
    """在两个面板上画一对 box: 谱面板 Rectangle + profile axvspan, 返回 (rect, span).

    单框单元的最低层入口, 上层既有按 region dict 批量画的 add_region_patches,
    也有交互模式按鼠标拖拽的 ms/MHz 坐标直接调用. 输入两角顺序无所谓.
    """
    t_lo, t_hi = (t0_ms, t1_ms) if t0_ms <= t1_ms else (t1_ms, t0_ms)
    f_lo, f_hi = (f0_mhz, f1_mhz) if f0_mhz <= f1_mhz else (f1_mhz, f0_mhz)
    rect = patches.Rectangle(
        (t_lo, f_lo), t_hi - t_lo, f_hi - f_lo,
        linewidth=lw, edgecolor=edge, facecolor='none', linestyle='--',
    )
    ax_spec.add_patch(rect)
    span = ax_profile.axvspan(t_lo, t_hi, alpha=alpha, facecolor=face)
    return rect, span


def add_region_patches(ax_profile, ax_spec, burst_regions, freq, time_reso, nchan,
                       edge='lime', face='steelblue', label_conf=False, linewidth=1.0):
    """对一组 burst region 在 profile + 谱面板上叠加 box.

    返回 [(rect, span), ...] 跟 burst_regions 一一对应, 方便交互模式做 undo.
    label_conf=True 时会在谱面板每个框顶端写出 YOLO 置信度.
    """
    artists = []
    for r in burst_regions:
        t0 = r['time_start'] * time_reso * 1e3
        t1 = r['time_end']   * time_reso * 1e3
        f0 = freq[np.clip(r['freq_start'],     0, nchan - 1)]
        f1 = freq[np.clip(r['freq_end'] - 1,   0, nchan - 1)]
        rect, span = _draw_burst_box(ax_profile, ax_spec, t0, t1, f0, f1,
                                     edge=edge, face=face, lw=linewidth)
        if label_conf:
            ax_spec.text(t0, max(f0, f1), f"{r['confidence']:.2f}",
                         color=edge, fontsize=8, va='bottom')
        artists.append((rect, span))
    return artists


def bbox_to_region(x0, y0, x1, y1, freq, time_reso, nsamp, nchan, confidence=1.0):
    """把鼠标拖拽得到的矩形 (时间 ms, 频率 MHz) 转换为 burst region 字典。

    手动模式和半自动重标记模式共享这段坐标转换，保证两种交互模式完全一致。
    输入的两个角顺序无所谓，函数内部会取 min/max 自动归一。
    """
    t_min = np.clip(min(x0, x1), 0, nsamp * time_reso * 1e3)
    t_max = np.clip(max(x0, x1), 0, nsamp * time_reso * 1e3)
    f_min = np.clip(min(y0, y1), min(freq[0], freq[-1]), max(freq[0], freq[-1]))
    f_max = np.clip(max(y0, y1), min(freq[0], freq[-1]), max(freq[0], freq[-1]))

    t_start = max(0, int(t_min / (time_reso * 1e3)))
    t_end   = min(nsamp, int(np.ceil(t_max / (time_reso * 1e3))))

    if freq[0] <= freq[-1]:
        f_start = int(np.searchsorted(freq, f_min, side='left'))
        f_end   = int(np.searchsorted(freq, f_max, side='right'))
    else:
        freq_asc = freq[::-1]
        asc_start = int(np.searchsorted(freq_asc, f_min, side='left'))
        asc_end   = int(np.searchsorted(freq_asc, f_max, side='right'))
        f_start = nchan - asc_end
        f_end   = nchan - asc_start
    f_start = max(0, min(nchan, f_start))
    f_end   = max(0, min(nchan, f_end))
    return {
        'time_start': t_start,
        'time_end':   t_end,
        'freq_start': f_start,
        'freq_end':   f_end,
        'confidence': confidence,
    }


def _render_two_panel(fig, stokes_I, freq, time_reso, normalize=True):
    """在 fig 上构建标准两面板布局: 上 profile + 下动态谱, 返回 (ax_profile, ax_spec).

    时间刻度、profile 计算这套视觉自动 / 半自动 / 手动模式共享,
    normalize=True 时保留检测/交互用的除背景归一化显示; normalize=False
    时直接显示调用方传入的数据, 用于保存已减背景并 mask RFI 的结果图.
    抽出来避免 plot_detection 和 review_interactive 的画布逐渐漂移.
    本函数不画任何 burst box, 调用方拿到两个 ax 后自己叠加.
    """
    nsamp, _ = stokes_I.shape
    if normalize:
        image = normalize_image(stokes_I).T  # (freq, time)
        profile = np.mean(image, axis=0)
        vmin, vmax = 0, 1
        ylabel = 'Intensity (abbr.)'
    else:
        image = np.asarray(stokes_I, dtype=np.float32).T
        profile = np.nanmean(stokes_I, axis=1)
        finite = image[np.isfinite(image)]
        if finite.size:
            vmin, vmax = np.nanpercentile(finite, [5, 95])
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                center = float(np.nanmedian(finite))
                vmin, vmax = center - 1.0, center + 1.0
        else:
            image = np.zeros_like(image, dtype=np.float32)
            vmin, vmax = 0, 1
        ylabel = 'Flux (Jy)'
    time_ms  = np.arange(nsamp) * time_reso * 1e3

    gs       = fig.add_gridspec(4, 1, hspace=0)
    ax_prof  = fig.add_subplot(gs[0, 0])
    ax_prof.step(time_ms, profile, where='mid', color='royalblue', lw=0.8)
    ax_prof.set_xlim(time_ms[0], time_ms[-1])
    ax_prof.set_xticks([])
    ax_prof.set_ylabel(ylabel)

    ax_spec  = fig.add_subplot(gs[1:, 0])
    ax_spec.imshow(
        image, aspect='auto', origin='lower', cmap='mako', vmin=vmin, vmax=vmax,
        extent=[time_ms[0], time_ms[-1], freq[0], freq[-1]],
    )
    ax_spec.set_xlabel('Time (ms)')
    ax_spec.set_ylabel('Frequency (MHz)')
    return ax_prof, ax_spec


def plot_detection(stokes_I, freq, time_reso, burst_regions, save_path, normalize=True):
    """绘制动态谱 + 检测框叠加图 (自动模式落盘版)."""
    nsamp, nchan = stokes_I.shape
    fig = plt.figure(figsize=(5, 5))
    ax_prof, ax_spec = _render_two_panel(fig, stokes_I, freq, time_reso,
                                         normalize=normalize)
    add_region_patches(ax_prof, ax_spec, burst_regions, freq, time_reso, nchan,
                       label_conf=True, linewidth=1)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def _raise_window(fig):
    """把图窗提升到前台并激活 (Windows/Qt/Tk 后端兼容版).

    review_interactive 在 plt.show(block=False) 之后单独调一次, 防止图窗
    被终端 / IDE 盖住, 导致用户看不到要审查的画面。各后端 manager.window
    暴露的方法不一样 (Qt: showNormal/raise_/activateWindow, Tk: 没有),
    所以全用 hasattr 探测; 任何异常都吞掉, 这一步失败不应影响主流程。
    """
    try:
        manager = fig.canvas.manager
        if hasattr(manager, 'window'):
            window = manager.window
            if hasattr(window, 'showNormal'):
                window.showNormal()
            if hasattr(window, 'raise_'):
                window.raise_()
            if hasattr(window, 'activateWindow'):
                window.activateWindow()
    except Exception:
        pass


def review_interactive(stokes_I, freq, time_reso, burst_regions=None):
    """显示动态谱供人工确认或重新标注爆发位置。

    burst_regions=None → 纯手动标注模式, 直接拖拽鼠标画框；
    burst_regions 有内容 → 展示模型检测结果, 用户可按 Enter 接受。

    交互逻辑 (一图窗内完成, 不来回切终端):
      * Enter / Return    → 接受当前显示的框 (模型框 或 已经画过的用户框)
      * x                  → 当前文件标为无 burst, 跳到下一张
      * 鼠标左键拖拽       → 立即清掉模型框, 进入手动重画状态; 画完的矩形会
                            实时回显在图上, 可以连续拖出多个框
      * 鼠标右键           → 撤销最近一次画的用户框 (多次右键 = 连续 pop,
                            为空则忽略). 模型框不会被右键撤销.
      * q / Esc           → 正常结束整批处理并保存已完成进度
      * 关闭窗口           → 等价于按 Enter, 误关不会中断批处理
    """
    burst_regions = burst_regions or []
    nsamp, nchan  = stokes_I.shape

    title_review  = 'Enter: accept | x: no burst | drag to redraw | q: quit'
    title_draw    = 'Drag: add box | right-click: undo last | x: no burst | Enter: finish | q: quit'
    showing_model = bool(burst_regions)

    fig = plt.figure(figsize=(8, 5))
    ax_prof, ax_spec = _render_two_panel(fig, stokes_I, freq, time_reso)
    # title 挂在上面板, 才会出现在图窗顶端而不是两面板之间.
    ax_prof.set_title(title_review if showing_model else title_draw)

    # 模型框的 (rect, span) artist 列表; 第一次拖拽时整体清掉.
    model_artists = (add_region_patches(ax_prof, ax_spec, burst_regions, freq, time_reso, nchan,
                                        label_conf=False, linewidth=1.5)
                     if showing_model else [])

    # 把可变状态收进 dict, 方便嵌套回调里 mutate (Python 闭包不能直接赋值外层标量).
    # user_regions / user_artists 是平行栈: on_select 同步 push, on_mouse_press 右键同步 pop.
    state = {
        'command':       None,    # 'accept' | 'empty' | 'quit'
        'showing_model': showing_model,
        'user_regions':  [],      # [(x0, y0, x1, y1), ...] 时间ms / 频率MHz
        'user_artists':  [],      # [(rect, span), ...] 跟 user_regions 一一对应, 用于右键 undo
    }

    def on_select(eclick, erelease):
        """RectangleSelector 回调: 一次拖拽完成 → 累计一个用户框并实时回显."""
        xlim = ax_spec.get_xlim()
        ylim = ax_spec.get_ylim()

        def clamp_event_xy(event):
            """拖拽到坐标轴外时贴到最近边缘, 方便直接框到全边界。"""
            if event.xdata is None and event.ydata is None:
                return None
            if event.xdata is None:
                x = xlim[0] if event.x < ax_spec.bbox.x0 else xlim[1]
            else:
                x = event.xdata
            if event.ydata is None:
                y = ylim[0] if event.y < ax_spec.bbox.y0 else ylim[1]
            else:
                y = event.ydata
            return (
                float(np.clip(x, min(xlim), max(xlim))),
                float(np.clip(y, min(ylim), max(ylim))),
            )

        p0 = clamp_event_xy(eclick)
        p1 = clamp_event_xy(erelease)
        if p0 is None or p1 is None:
            return
        # 第一次拖拽: 清掉模型框 (谱面板 Rectangle + profile axvspan), 切到手动重画.
        if state['showing_model']:
            for rect, span in model_artists:
                rect.remove()
                span.remove()
            model_artists.clear()
            state['showing_model'] = False
            ax_prof.set_title(title_draw)
        x0, x1 = sorted([p0[0], p1[0]])
        y0, y1 = sorted([p0[1], p1[1]])
        # 立刻回显: 黄色虚线框 + profile gold 半透明高亮, 跟模型的 lime/steelblue 区分.
        rect, span = _draw_burst_box(ax_prof, ax_spec, x0, x1, y0, y1,
                                     edge='yellow', face='gold', lw=1.5)
        state['user_regions'].append((x0, y0, x1, y1))
        state['user_artists'].append((rect, span))
        fig.canvas.draw_idle()

    def on_mouse_press(event):
        """右键 = 撤销最近一次画的用户框 (LIFO). 用户栈为空 / 模型框状态时静默忽略.

        button==3 是 matplotlib 对鼠标右键的统一编号 (跨平台). 左键的拖拽走
        RectangleSelector 自己的逻辑 (button=[1] 过滤), 不会进这里.
        """
        if event.button != 3:
            return
        if not state['user_regions']:
            return
        state['user_regions'].pop()
        rect, span = state['user_artists'].pop()
        rect.remove()
        span.remove()
        fig.canvas.draw_idle()

    def on_key(event):
        key = (event.key or '').lower()
        if key in ('enter', 'return'):
            state['command'] = 'accept'
            fig.canvas.stop_event_loop()
        elif key == 'x':
            state['command'] = 'empty'
            fig.canvas.stop_event_loop()
        elif key in ('q', 'escape'):
            state['command'] = 'quit'
            fig.canvas.stop_event_loop()

    def on_close(_event):
        # 关掉窗口等价于按 Enter, 这样误关窗口不会中断整批流程.
        if state['command'] is None:
            state['command'] = 'accept'
            fig.canvas.stop_event_loop()

    selector = RectangleSelector(
        ax_spec, on_select, useblit=True, button=[1],
        minspanx=5, minspany=5, spancoords='pixels', interactive=False,
    )
    cid_key   = fig.canvas.mpl_connect('key_press_event',    on_key)
    cid_close = fig.canvas.mpl_connect('close_event',        on_close)
    cid_mouse = fig.canvas.mpl_connect('button_press_event', on_mouse_press)

    # tight_layout 处理外边距防止 xlabel 被裁掉; 但它会把 GridSpec 的 hspace
    # 撑开, 所以紧接着再 subplots_adjust 把上下面板贴回去.
    plt.tight_layout()
    fig.subplots_adjust(hspace=0)
    plt.show(block=False)
    fig.canvas.draw_idle()
    _raise_window(fig)
    plt.pause(0.2)

    if showing_model:
        print("  图窗内按 Enter 接受模型框, 按 x 标为无爆发, 鼠标左键拖拽即重画 (右键撤销), q/Esc 退出。")
    else:
        print("  在图窗中拖拽鼠标左键画爆发框, 按 x 标为无爆发, 右键撤销最近一框, 画完按 Enter, q/Esc 退出。")

    try:
        fig.canvas.start_event_loop(timeout=0)
    finally:
        fig.canvas.mpl_disconnect(cid_key)
        fig.canvas.mpl_disconnect(cid_close)
        fig.canvas.mpl_disconnect(cid_mouse)

    plt.close(fig)
    # selector 持有一个引用避免被 GC; close 之后即可释放
    del selector

    if state['command'] == 'quit':
        return None
    if state['command'] == 'empty':
        return []

    # 一次都没拖拽 → 接受模型推理结果 (手动模式下 burst_regions 为空, 也合法)
    if state['showing_model']:
        return burst_regions
    # 用户重画过 → 用拖拽得到的矩形 (允许零框, 表示该文件没有真实爆发)
    return [bbox_to_region(*box, freq, time_reso, nsamp, nchan)
            for box in state['user_regions']]


def detect_one_file(h5_path, model, conf=0.25, iou_threshold=0.5,
                    mode='auto', plot_dir=None, rfi_fft=False,
                    max_horizontal_aspect=DEFAULT_MAX_HORIZONTAL_ASPECT):
    """对一个定标后 h5 文件做爆发检测。

    Parameters
    ----------
    h5_path : str
        *_cal.h5 文件路径。
    model : DetectionModel or None
        manual 模式下可传 None.
    conf : float
        置信度阈值。
    iou_threshold : float
        NMS IoU 阈值。
    mode : str
        'auto'(YOLO 全自动) / 'semi-auto'(YOLO + 人工审查) / 'manual'(纯手工标记).
    plot_dir : str or None
        若不为 None，保存检测诊断图到该目录。
    rfi_fft : bool
        检测确认后写入 H5 的 RFI 是否使用 FFT 方法; 默认使用与
        burst_analysis.py 相同的熵方法。
    max_horizontal_aspect : float
        模型框允许的最大宽高比；重叠框始终优先保留面积最大的框。

    长数据切 tile: 当 nsamp_save 经过"清晰下采样"后还宽于 512 时, 把图按
    时间轴每 512 列切一片送 YOLO 推理. 检测下采样倍率默认取 calibration
    阶段画图用的 plot_down_time / down_time, 与目视一致.

    Returns
    -------
    result : dict or None
        包含 'bursts'（区域列表）和 'has_burst'（是否检测到爆发）；用户按
        q/Esc 时返回 None。
    """
    with h5py.File(h5_path, 'r') as f:
        iquv      = f['data'][:]                 # (4, nsamp, nchan)
        freq      = f['freq'][:]                 # (nchan,)
        time_reso = float(f.attrs['time_reso'])
        save_dt   = int(f.attrs.get('down_time', 1))
        plot_dt   = int(f.attrs.get('plot_down_time', save_dt))

    stokes_I = iquv[0]  # (nsamp, nchan)
    nsamp, nchan = stokes_I.shape

    if mode == 'manual':
        burst_regions = review_interactive(stokes_I, freq, time_reso)
    else:
        # auto / semi-auto: YOLO 推理.
        # 检测时间下采样跟 calibration 画图倍率一致 (像素 → 保存采样点的换算)
        det_time_factor = max(1, plot_dt // save_dt)
        tiles, offsets, time_factor, freq_factor = prepare_image_tiles(stokes_I, target_size=512, time_factor=det_time_factor)

        if len(tiles) > 1:
            print(f'    长数据: 分 {len(tiles)} 个 tile 推理 '
                  f'(time_factor={time_factor}, freq_factor={freq_factor})')

        tile_results = []
        for tile, offset in zip(tiles, offsets):
            scores, boxes = predict_single(
                model, tile, conf=conf, iou_threshold=iou_threshold,
                max_horizontal_aspect=max_horizontal_aspect)
            tile_results.append((boxes, scores, offset))
        burst_regions = boxes_to_regions_tiled(tile_results, time_factor, freq_factor, nsamp, nchan)

    # semi-auto: 把 YOLO 结果交给用户审查 / 修改
    if mode == 'semi-auto':
        burst_regions = review_interactive(stokes_I, freq, time_reso, burst_regions)
    if burst_regions is None:
        return None

    fname = os.path.basename(h5_path)
    print(f'  [{fname}] 检测到 {len(burst_regions)} 个爆发')

    rfi_channel, rfi_fraction, plot_I = write_detection_results(
        h5_path, iquv, burst_regions, rfi_fft=rfi_fft)
    print(f'  [{fname}] 检测后 RFI: '
          f'{int(np.sum(rfi_channel))} 通道, {rfi_fraction*100:.1f}% mask')

    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        basename  = os.path.splitext(fname)[0]
        plot_path = os.path.join(plot_dir, basename + '_det.png')
        plot_detection(plot_I, freq, time_reso, burst_regions, plot_path,
                       normalize=False)

    # H5 中已写入 attrs['bursts'] 和检测后 RFI; detections.json 只负责断点续跑。
    return {
        'bursts':    burst_regions,
        'has_burst': len(burst_regions) > 0,
    }


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='YOLO 爆发检测')
    parser.add_argument('--cal-dir',       default='./cal/',                  help='定标 h5 文件目录')
    parser.add_argument('--model-path',    default='./models/best_model_yolo11n_ema.pth', help='YOLO 权重文件路径')
    parser.add_argument('--model-name',    default='yolo11n',                          help='YOLO 模型配置名')
    parser.add_argument('--output-dir',    default='./detections/',           help='输出目录')
    parser.add_argument('--conf',          default=0.25, type=float,          help='置信度阈值')
    parser.add_argument('--iou-threshold', default=0.5,  type=float,          help='NMS IoU 阈值')
    parser.add_argument('--max-horizontal-aspect',
                        default=DEFAULT_MAX_HORIZONTAL_ASPECT, type=float,
                        help='过滤 width/height 超过该值的横向模型框')
    parser.add_argument('--mode',          default='auto',
                        choices=['auto', 'semi-auto', 'manual'],
                        help='auto=全自动; semi-auto=YOLO+人工审查; manual=纯手工标记')
    parser.add_argument('--max-files',     default=None, type=int,
                        help='最多处理多少个文件（调试用）')
    parser.add_argument('--rfi-fft',       action='store_true',
                        help='检测确认后写入 H5 的 RFI 改用 FFT 法')
    args = parser.parse_args()
    if args.max_horizontal_aspect <= 0:
        parser.error('--max-horizontal-aspect 必须大于 0')

    if args.mode == 'auto':
        matplotlib.use('Agg')

    # manual 模式不需要 YOLO 模型
    model = None if args.mode == 'manual' else load_yolo_model(
        args.model_path, args.model_name)

    # 查找定标 h5 文件
    h5_files = sorted(glob.glob(os.path.join(args.cal_dir, '**', '*_cal.h5'),
                                recursive=True))
    if args.max_files is not None:
        h5_files = h5_files[:args.max_files]
    if not h5_files:
        print(f'未找到 *_cal.h5 文件: {args.cal_dir}')
        exit()
    print(f'找到 {len(h5_files)} 个定标文件')

    # 逐文件检测 (支持断点续标)
    os.makedirs(args.output_dir, exist_ok=True)
    plot_dir = os.path.join(args.output_dir, 'plots')
    det_path = os.path.join(args.output_dir, 'detections.json')

    # detections.json 是唯一的 "已处理" 真相: 启动时载入, 已存在 fname 的条目
    # 直接跳过 (零样本文件也算已标 — entry 是 {'bursts': [], 'has_burst': false}).
    # 这是有意保留的断点续跑逻辑: 半自动模式中途退出后, 再次使用同一个
    # output-dir 会从下一个未处理文件继续, 不会覆盖已经人工确认过的 H5 标记.
    # 想重新标记某个文件: 手动从 detections.json 删掉对应行, 或换一个新的
    # output-dir 全量重跑。
    detections = {}
    if os.path.exists(det_path):
        with open(det_path) as f:
            detections = json.load(f)
        print(f'  已载入 {len(detections)} 条历史记录, 已处理的文件会跳过')

    quit_requested = False
    for h5_path in h5_files:
        fname = os.path.basename(h5_path)
        if fname in detections:
            print(f'  [{fname}] 跳过 (detections.json 中已存在)')
            continue
        result = detect_one_file(
            h5_path, model, conf=args.conf,
            iou_threshold=args.iou_threshold,
            mode=args.mode, plot_dir=plot_dir,
            rfi_fft=args.rfi_fft,
            max_horizontal_aspect=args.max_horizontal_aspect)
        if result is None:
            quit_requested = True
            break
        detections[fname] = result
        # 每文件落盘一次, 中途中断 / Ctrl+C 不会丢已经标好的进度
        with open(det_path, 'w') as f:
            json.dump(detections, f, indent=2)

    if quit_requested:
        # 当前文件不记为已处理，下次从它继续。
        with open(det_path, 'w') as f:
            json.dump(detections, f, indent=2)
        n_with = sum(1 for v in detections.values() if v['has_burst'])
        print(f'\n[退出] 当前进度已保存: {det_path}')
        print(f'已处理 {len(detections)}/{len(h5_files)} 个文件，'
              f'其中 {n_with} 个检测到爆发')
    else:
        print(f'[OK] 检测结果已保存: {det_path}')
        n_with = sum(1 for v in detections.values() if v['has_burst'])
        print(f'\n完成: {n_with}/{len(detections)} 个文件检测到爆发')
