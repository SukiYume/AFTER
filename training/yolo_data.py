"""H5 dataset + 数据增强 for AFTER burst detector。

数据：H5 文件中每张 ``(time, freq)`` 的 512×512 灰度图 + 同文件
``annotations`` 表 ``(img_idx, x_freq, y_time, w_freq, h_time)``（``x = -1``
表示该帧无标注的占位行）。
模型输入：先转成与 AFTER 一致的 ``(freq, time)`` 图像坐标，再复制为 3 通道；
模型框的 x 轴表示 time，y 轴表示 freq。

训练集帧级重采样（``get_train_val``）按一帧目标数量分桶：
  - 无标注帧（``empty_repeat``）默认 ×1
  - 单信号帧（``single_burst_repeat``）默认 ×3
  - 多信号 hard case 帧（``multi_burst_repeat``）默认 ×6

多信号样本通常是相邻 burst / 重叠候选框 / 一个大框吞并多个信号——提高这类样本的采样
频率让模型在训练时更常见到「一图多目标」的情况；验证集保持真实分布。

数据增强：
  - mosaic 拼图：2~5 张图按 ``rows × cols`` 排列后 mean-pool 压回 512×512
  - albumentations 几何（HFlip/VFlip/Rotate）+ 像素（噪声/模糊/亮度对比）
  - 推理 / 验证时只做 1% / 99.5% 分位归一化
"""

import os
import random

import albumentations as A
import h5py
import numpy as np
import pandas as pd
import torch


IMAGE_SIZE = 512


# ---------------------------------------------------------------------------
# 1. 通用工具：图像归一化
# ---------------------------------------------------------------------------

def normalize_image(img, pmin=1, pmax=99.5):
    """按 ``pmin / pmax`` 分位截断 + min-max 归一化到 [0, 1]，dtype float32。"""
    vmin, vmax = np.percentile(img, (pmin, pmax))
    img = np.clip(img, vmin, vmax)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


# ---------------------------------------------------------------------------
# 2. 数据切分 + 训练集帧级重采样
# ---------------------------------------------------------------------------

def get_train_val(data_folder, val_fraction=0.2, seed=42,
                  empty_repeat=1, single_burst_repeat=3, multi_burst_repeat=6):
    """扫描 ``data_folder`` 下所有 .h5，按观测日期切分 train / val。

    划分策略：
      1. 使用 ``original_date`` 分组，确保同一观测日期不会同时进入 train / val
      2. 只对训练集做帧级重采样：无标注 ×``empty_repeat``，单信号 ×``single_burst_repeat``，
         多信号 ×``multi_burst_repeat``；验证集保持原始分布

    Returns:
        train_df, val_df: pandas DataFrame，每行一帧，含字段
        ``{h5_path, img_idx, original_date, bboxes, has_annotation, n_bboxes}``。
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")

    h5_files = sorted(
        os.path.join(data_folder, f)
        for f in os.listdir(data_folder)
        if f.endswith('.h5')
    )
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {data_folder}")

    records = []
    for h5_path in h5_files:
        with h5py.File(h5_path, 'r') as f:
            total_images = f['images'].shape[0]
            all_annotations = f['annotations'][:]
            original_dates = f['original_date'][:]
        if len(original_dates) != total_images:
            raise ValueError(f"original_date length mismatch in {h5_path}")
        for img_idx in range(total_images):
            # (img_idx, x, y, w, h)：先按 img_idx 选出该帧的所有行，再用 x >= 0 过滤掉占位行
            img_ann = all_annotations[all_annotations[:, 0] == img_idx, 1:]
            bboxes = img_ann[img_ann[:, 0] >= 0]
            original_date = original_dates[img_idx]
            if isinstance(original_date, bytes):
                original_date = original_date.decode('utf-8')
            records.append({
                'h5_path': h5_path,
                'img_idx': img_idx,
                'original_date': str(original_date),
                'bboxes': bboxes,
                'has_annotation': len(bboxes) > 0,
                'n_bboxes': len(bboxes),
            })

    # 先按 original_date 分组切分，再做训练集重采样。
    df = pd.DataFrame(records)
    groups = np.array(sorted(df['original_date'].unique()), dtype=object)
    if len(groups) < 2:
        raise ValueError("At least two original_date groups are required")

    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    group_counts = df['original_date'].value_counts()
    target_val_count = max(1, int(round(len(df) * val_fraction)))
    val_groups = []
    val_count = 0
    for group in groups:
        if len(val_groups) >= len(groups) - 1:
            break
        next_count = val_count + int(group_counts[group])
        if not val_groups or abs(next_count - target_val_count) < abs(val_count - target_val_count):
            val_groups.append(group)
            val_count = next_count

    is_val = df['original_date'].isin(val_groups)
    train_base = df[~is_val].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = df[is_val].sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # 帧级重采样：按 n_bboxes 分三档
    n = train_base['n_bboxes'].to_numpy()
    repeats = np.full(len(train_base), empty_repeat, dtype=int)
    repeats[n == 1] = single_burst_repeat
    repeats[n >= 2] = multi_burst_repeat
    repeats = np.maximum(repeats, 1)

    train_df = train_base.loc[train_base.index.repeat(repeats)].copy()
    train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, val_df


# ---------------------------------------------------------------------------
# 3. Dataset
# ---------------------------------------------------------------------------

class H5YOLODataset(torch.utils.data.Dataset):
    """逐帧产出 ``{img, cls, bboxes, batch_idx}``，格式与 ultralytics ``DetectionModel`` 对齐。

    - ``img``       : ``[3, 512, 512]`` float32，灰度复制到 3 通道
    - ``cls``       : ``[N, 1]``，类别 id（只有 1 类，全 0）
    - ``bboxes``    : ``[N, 4]``，归一化的 ``[cx, cy, w, h]``
    - ``batch_idx`` : ``[N]``，由 collate fn 填入 batch 内序号
    """

    def __init__(self, dataframe, imgsz=IMAGE_SIZE, val=False, mosaic_prob=0.5):
        if int(imgsz) != IMAGE_SIZE:
            raise ValueError(f"AFTER burst detector uses a fixed image size of {IMAGE_SIZE}")
        self.dataframe   = dataframe
        self.imgsz       = IMAGE_SIZE
        self.val         = bool(val)
        self.mosaic_prob = 0.0 if self.val else float(mosaic_prob)
        self.h5_files    = {}                                       # worker-local h5 句柄缓存
        self.train_transform = None if self.val else self._build_train_transform()

    # ---- 主入口 ------------------------------------------------------------

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # 1. mosaic 拼图（仅训练，按 mosaic_prob 触发）
        if not self.val and random.random() < self.mosaic_prob:
            img, bboxes = self._load_mosaic(idx)
        else:
            img, bboxes = self._load_single(idx)

        # 2. albumentations 增强（仅训练）
        if not self.val:
            img, bboxes = self._augment(img, bboxes)

        # 3. 归一化：训练时分位随机抖动一点，验证用固定分位
        if self.val:
            pmin, pmax = 1.0, 99.5
        else:
            pmin = float(np.random.uniform(0, 2))
            pmax = float(np.random.uniform(98, 100))
        img = normalize_image(img, pmin, pmax)

        # 4. (H, W) 灰度 → (3, H, W) 复制 → tensor
        img_tensor = torch.from_numpy(img).float().unsqueeze(0).expand(3, -1, -1).contiguous()
        cls, bboxes = self._format_targets(bboxes)
        return {
            'img': img_tensor,
            'cls': cls,
            'bboxes': bboxes,
            'batch_idx': torch.zeros(cls.shape[0], dtype=torch.long),
        }

    # ---- 单帧 / mosaic 加载 -----------------------------------------------

    def _load_single(self, idx):
        row = self.dataframe.iloc[idx]
        h5_path = row['h5_path']
        if h5_path not in self.h5_files:
            self.h5_files[h5_path] = h5py.File(h5_path, 'r')        # 每个 worker 独立持有句柄
        # H5 保存为 (time, freq)，AFTER 模型输入为 (freq, time)。
        img = self.h5_files[h5_path]['images'][row['img_idx']].T.copy()
        bboxes = row['bboxes'][:, [1, 0, 3, 2]].copy()
        return img, bboxes

    def _load_mosaic(self, idx):
        """2~5 张图按 2×2 / 1×n / n×1 拼接后 mean-pool 回 512²。"""
        comb_num = np.random.randint(1, 6)
        if comb_num == 1:
            return self._load_single(idx)

        other_idx = np.random.choice(len(self.dataframe), comb_num - 1, replace=False)
        comb_idx  = np.append([idx], other_idx)

        imgs, boxes_list = [], []
        for i in comb_idx:
            img, bboxes = self._load_single(i)
            imgs.append(img)
            boxes_list.append(bboxes)

        # 4 张图有一半概率走 2×2 网格；否则水平 / 垂直条带
        if comb_num == 4 and np.random.rand() > 0.5:
            rows, cols = 2, 2
        elif np.random.rand() > 0.5:
            rows, cols = 1, comb_num
        else:
            rows, cols = comb_num, 1
        return self._mosaic(imgs, boxes_list, rows, cols)

    @staticmethod
    def _mosaic(imgs, boxes_list, rows, cols):
        """把 ``rows*cols`` 张 512² 图拼成 (512·rows, 512·cols)，再按格子 mean-pool 回 512²。

        - 拼接后的画布元素坐标 = 原图坐标 + 该格的左上角偏移
        - mean-pool 等价于按 (rows, cols) 块平均下采样：每个目标自动按 (cols, rows) 缩小
        """
        canvas = np.zeros((IMAGE_SIZE * rows, IMAGE_SIZE * cols), dtype=np.float32)
        bbox_data = []
        for i in range(rows * cols):
            r, c = divmod(i, cols)
            canvas[
                IMAGE_SIZE * r: IMAGE_SIZE * (r + 1),
                IMAGE_SIZE * c: IMAGE_SIZE * (c + 1),
            ] = imgs[i]
            if isinstance(boxes_list[i], np.ndarray) and boxes_list[i].size:
                b = boxes_list[i].copy()
                b[:, 0] += IMAGE_SIZE * c                           # x 偏移
                b[:, 1] += IMAGE_SIZE * r                           # y 偏移
                bbox_data.append(b)

        img = canvas.reshape(IMAGE_SIZE, rows, IMAGE_SIZE, cols).mean(axis=(1, 3))
        if bbox_data:
            boxes = np.vstack(bbox_data)
            boxes[:, [0, 2]] /= cols                                # x, w 同步缩小
            boxes[:, [1, 3]] /= rows                                # y, h 同步缩小
        else:
            boxes = np.array([])
        return img, boxes

    # ---- 增强 + 目标格式转换 ----------------------------------------------

    def _augment(self, img, bboxes):
        """albumentations 几何 + 像素增强，bbox 同步变换。"""
        bboxes_list = bboxes.tolist() if isinstance(bboxes, np.ndarray) and bboxes.ndim == 2 else []
        augmented = self.train_transform(image=img, bboxes=bboxes_list)
        bboxes_aug = augmented['bboxes']
        return augmented['image'], np.array(bboxes_aug) if len(bboxes_aug) else np.array([])

    def _build_train_transform(self):
        """训练增强 pipeline；几何变换 bbox 同步更新，pixel 变换不动 bbox。"""
        return A.Compose([
            # 几何
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=30, p=0.5),
            # 像素
            A.OneOf([
                A.GaussNoise(std_range=(0, 0.4), p=1),
                A.GaussianBlur(blur_limit=(3, 7), p=1),
                A.MotionBlur(blur_limit=5, p=1),
            ], p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            # 尺寸修正：mosaic / 旋转后保证统一 imgsz
            A.Resize(height=self.imgsz, width=self.imgsz, p=1.0),
        ], bbox_params=A.BboxParams(format='coco', min_visibility=0.1, label_fields=[]))

    def _format_targets(self, bboxes):
        """COCO ``[x, y, w, h]``（像素）→ YOLO 归一化 ``[cx, cy, w, h]``；过滤过小框。"""
        cls_list, bbox_list = [], []
        if isinstance(bboxes, np.ndarray) and bboxes.ndim == 2 and len(bboxes) > 0:
            for bx, by, bw, bh in bboxes:
                if bw < 1 or bh < 1:                                # 过小框直接丢
                    continue
                cx = np.clip((bx + bw / 2) / self.imgsz, 0, 1)
                cy = np.clip((by + bh / 2) / self.imgsz, 0, 1)
                nw = np.clip(bw / self.imgsz, 0, 1)
                nh = np.clip(bh / self.imgsz, 0, 1)
                if nw > 0.002 and nh > 0.002:                       # 归一化后过小也丢（数值稳定）
                    cls_list.append(0)
                    bbox_list.append([cx, cy, nw, nh])

        n = len(cls_list)
        cls = torch.tensor(cls_list, dtype=torch.float32).reshape(n, 1) if n else torch.zeros((0, 1))
        boxes = torch.tensor(bbox_list, dtype=torch.float32) if n else torch.zeros((0, 4))
        return cls, boxes


# ---------------------------------------------------------------------------
# 4. Collate fn：把样本 dict 列表拼成 ultralytics DetectionModel 期望的 batch dict
# ---------------------------------------------------------------------------

def yolo_collate_fn(batch):
    imgs = torch.stack([b['img'] for b in batch])
    cls_list, bbox_list, bidx_list = [], [], []
    for i, b in enumerate(batch):
        n = b['cls'].shape[0]
        if n > 0:
            cls_list.append(b['cls'])
            bbox_list.append(b['bboxes'])
            bidx_list.append(torch.full((n,), i, dtype=torch.long))

    if cls_list:
        return {
            'img': imgs,
            'cls': torch.cat(cls_list),
            'bboxes': torch.cat(bbox_list),
            'batch_idx': torch.cat(bidx_list),
        }
    return {
        'img': imgs,
        'cls': torch.zeros((0, 1)),
        'bboxes': torch.zeros((0, 4)),
        'batch_idx': torch.zeros(0, dtype=torch.long),
    }
