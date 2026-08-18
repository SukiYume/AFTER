"""从原始 FITS 观测数据中切出爆发片段并消色散，保存为 h5。

流程：
  1. 获取指定波束的 FITS 文件列表（按文件名排序）。
  2. 复制第一个 FITS（含定标噪声管）到 SAVE_PATH，供后续流量定标使用。
  3. 读取 FITS 的观测属性（时间分辨率、文件长度、通道数等）。
  4. 对 toa_list 中的每个到达时间切 [B - A/2, B + A/2 + C) 的原始数据段，
     其中 A = SEGMENT_LENGTH，C = 最大色散延迟（保证消色散后仍有 A 个点）。
  5. 对切出的数据消色散，只保留 A 个样本，连同观测元数据写入 h5。
  6. 汇总所有 burst 文件信息到 obs_info.json。

所有 I/O、切片逻辑都显式写在主流程，避免只调用一次的函数封装。
"""

import os
import shutil
from importlib import import_module
from typing import Any, cast

import numpy as np
import h5py
from astropy.io import fits
from multiprocessing import Pool

from .obs_metadata import write_obs_info_json


def read_obs_info(data_path, file_list):
    """读取第一个 FITS 文件的观测属性。

    返回
    -------
    info : dict
        time_reso, nsblk, naxis2, file_nsamp (单文件采样点数),
        npol, nchan, freq (一维频率数组), start_mjd。
    """
    fpath = os.path.join(data_path, file_list[0])
    with fits.open(fpath) as f:
        primary_hdu = cast(Any, f[0])
        subint_hdu = cast(Any, f[1])
        if subint_hdu.data is None:
            raise ValueError(f"FITS SUBINT table has no data: {fpath}")
        h0 = primary_hdu.header
        h1 = subint_hdu.header
        freq = subint_hdu.data['DAT_FREQ'][0, :].astype(np.float64)

    return {
        'time_reso'  : h1['TBIN'],
        'nsblk'      : h1['NSBLK'],
        'naxis2'     : h1['NAXIS2'],
        'file_nsamp' : h1['NAXIS2'] * h1['NSBLK'],
        'npol'       : h1['NPOL'],
        'nchan'      : h1['NCHAN'],
        'freq'       : freq,
        'start_mjd'  : h0['STT_IMJD'] + (h0['STT_SMJD'] + h0['STT_OFFS']) / 86400.0,
    }


def calc_dispersion_shift(dm, freq, time_reso):
    """计算每个频率通道的色散延迟采样点数。

    返回
    -------
    shifts : ndarray (nchan,)  每通道需向后偏移的采样点数（最高频率处为 0）。
    max_shift : int            最大偏移量，即消色散所需的额外数据长度 C。
    """
    shifts = np.int64(4.15e3 * dm * (freq ** -2 - freq.max() ** -2) / time_reso)
    return shifts, int(shifts.max())


def extract_segment(data_path, file_list, info, start_sample, total_length):
    """从多个 FITS 文件中提取连续 total_length 个采样点。

    start_sample 可以 < 0 或超过观测总长度，超出部分以 0 填充，
    因此信号靠近观测起止时也不会崩。

    返回
    -------
    segment : ndarray (total_length, npol, nchan), uint8
    """
    file_nsamp = info['file_nsamp']
    n_files    = len(file_list)
    total_obs  = file_nsamp * n_files
    npol       = info['npol']
    nchan      = info['nchan']

    segment = np.zeros((total_length, npol, nchan), dtype=np.uint8)

    read_start = max(start_sample, 0)
    read_end   = min(start_sample + total_length, total_obs)
    if read_end <= read_start:
        return segment

    seg_offset     = read_start - start_sample
    first_file_idx = read_start // file_nsamp
    last_file_idx  = (read_end - 1) // file_nsamp

    cursor = seg_offset
    for fi in range(first_file_idx, last_file_idx + 1):
        fpath = os.path.join(data_path, file_list[fi])
        # 优先用 fitsio（更快），不可用时退回 astropy
        try:
            fitsio = cast(Any, import_module("fitsio"))
            fdata_raw, h = fitsio.read(fpath, header=True)
        except Exception:
            with fits.open(fpath) as f:
                subint_hdu = cast(Any, f[1])
                if subint_hdu.data is None:
                    raise ValueError(f"FITS SUBINT table has no data: {fpath}")
                h = subint_hdu.header
                fdata_raw = subint_hdu.data
        fdata = fdata_raw['DATA'].reshape(
            h['NAXIS2'] * h['NSBLK'], h['NPOL'], h['NCHAN'])

        local_start = max(read_start - fi * file_nsamp, 0)
        local_end   = min(read_end   - fi * file_nsamp, file_nsamp)
        n_copy      = local_end - local_start

        # 一场观测的最后一个 PSRFITS 可能比首文件给出的标称 file_nsamp 更短。
        # 这里仍按标称长度推进整场观测的逻辑时间轴；实际不存在的尾部采样保留为
        # ``segment`` 初始化时的零填充，避免后续文件的时间位置被错误前移。
        available_end = min(local_end, fdata.shape[0])
        available = max(available_end - local_start, 0)
        if available:
            segment[cursor:cursor + available] = \
                fdata[local_start:available_end]
        if available < n_copy:
            print(
                f'  [警告] {file_list[fi]} 截短: 请求 {n_copy} 个采样, '
                f'实际 {available}; 末尾补零 {n_copy - available} 个采样'
            )
        cursor += n_copy

    return segment


def dedisperse(segment, shifts, segment_length):
    """按预计算的 shifts 对每个通道做冷消色散。

    参数
    ----------
    segment : ndarray (A + C, npol, nchan)
    shifts  : ndarray (nchan,)
    segment_length : int    消色散后保留的采样点数 A

    返回
    -------
    out : ndarray (segment_length, npol, nchan)
    """
    npol  = segment.shape[1]
    nchan = segment.shape[2]
    out   = np.zeros((segment_length, npol, nchan), dtype=segment.dtype)
    for i in range(nchan):
        end = shifts[i] + segment_length
        if end <= segment.shape[0]:
            out[:, :, i] = segment[shifts[i]:end, :, i]
        else:
            # 数据不足（信号靠近观测末尾）时只填充可用部分
            available = segment.shape[0] - shifts[i]
            if available > 0:
                out[:available, :, i] = segment[shifts[i]:, :, i]
    return out


def cut_one_burst(data_path, save_path, file_list, info, dm, toa_sec,
                  shifts, max_shift, segment_length, frb_name, date, beam):
    """切一个爆发并消色散，保存为 h5 文件。"""
    time_reso = info['time_reso']
    A = segment_length
    C = max_shift

    # 到达时间对应的采样点 B；切取范围 [B - A/2, B + A/2 + C)
    B            = int(toa_sec / time_reso)
    start_sample = B - A // 2
    total_length = A + C

    file_nsamp = info['file_nsamp']
    total_obs  = file_nsamp * len(file_list)

    # TOA 必须落在观测内；切片起点为负时由 extract_segment 在左侧补零。
    if B < 0 or B >= total_obs:
        print(f'  [跳过] TOA={toa_sec:.6f}s (采样点 {B}) 超出观测范围')
        return

    # 文件命名: 起点非负时为 9 位数字，负值为 n + 9 位绝对值。
    fits_number = B // file_nsamp + 1
    start_token = f'n{abs(start_sample):09d}' if start_sample < 0 else f'{start_sample:09d}'
    h5_name     = f'{frb_name}-{date}-M{beam:02d}-{fits_number:04d}-{start_token}.h5'
    h5_path     = os.path.join(save_path, h5_name)
    if os.path.exists(h5_path):
        try:
            with h5py.File(h5_path, 'r') as existing:
                if 'data' in existing and 'freq' in existing:
                    print(f'  [跳过] {h5_name} 已存在')
                    return
        except OSError:
            pass
        print(f'  [重写] {h5_name} 文件不完整')

    segment          = extract_segment(data_path, file_list, info, start_sample, total_length)
    data_dedispersed = dedisperse(segment, shifts, A)

    # file_mjd = 观测起始 MJD + 该 h5 起点到观测起点的时间差
    file_mjd = info['start_mjd'] + start_sample * time_reso / 86400.0

    os.makedirs(save_path, exist_ok=True)
    temp_path = h5_path + '.tmp'
    try:
        with h5py.File(temp_path, 'w') as f:
            f.create_dataset('data', data=data_dedispersed, compression='gzip', compression_opts=4)
            f.create_dataset('freq', data=info['freq'])
            f.attrs['start_sample']   = start_sample        # 可为负，表示左侧补零后的逻辑起点
            f.attrs['file_mjd']       = file_mjd            # 本文件第 0 个采样对应的 MJD
            f.attrs['toa_sec']        = toa_sec
            f.attrs['time_reso']      = time_reso
            f.attrs['npol']           = info['npol']
            f.attrs['nchan']          = info['nchan']
            f.attrs['segment_length'] = A
            f.attrs['obs_start_mjd']  = info['start_mjd']
            f.attrs['beam']           = beam
            f.attrs['dm']             = dm
        os.replace(temp_path, h5_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    print(f'  [完成] {h5_name}')


def save_obs_json(output_dir, info, dm):
    """扫描每个 burst H5 的属性，生成可表达混合配置的 obs_info.json。"""
    # 保留旧调用签名；汇总值必须来自每个 H5，不能来自最后处理的一组。
    del info, dm
    json_path = write_obs_info_json(output_dir)
    if json_path is not None:
        print(f'  [JSON] {json_path}')


if __name__ == '__main__':

    # ============================================================
    # 配置参数
    # ============================================================
    SEGMENT_LENGTH = 4096               # 切出数据片段的采样点数 (A)
    DM             = 411.2              # 色散量 (pc/cm^3)
    BEAM           = 1                  # 波束编号
    NUM_WORKERS    = 16                 # 并行进程数
    FRB_NAME       = 'FRB20201124A'     # FRB 名称
    DATE           = '20210526'         # 观测日期

    DATA_PATH      = '/path/to/raw_fast/source/date/'
    SAVE_PATH      = '/path/to/after_data/FRB20201124A/20210526/'
    TOA_FILE       = 'toa_list_fast.txt'

    # ============================================================
    # 主流程
    # ============================================================

    # 1. 文件列表：仅保留指定 beam 的 FITS，按文件名排序
    pattern   = 'M{:0>2d}'.format(BEAM)
    file_list = sorted(f for f in os.listdir(DATA_PATH)
                       if pattern in f and f.endswith('.fits'))
    if not file_list:
        print('未找到 fits 文件')
        exit()
    print(f'找到 {len(file_list)} 个 fits 文件')

    # 2. 复制第一个 FITS（含定标噪声管）到输出目录
    os.makedirs(SAVE_PATH, exist_ok=True)
    src = os.path.join(DATA_PATH, file_list[0])
    dst = os.path.join(SAVE_PATH, file_list[0])
    if not os.path.exists(dst):
        shutil.copy(src, dst)

    # 3. 读取观测信息
    info = read_obs_info(DATA_PATH, file_list)
    print(f'时间分辨率: {info["time_reso"]*1e6:.3f} μs')
    print(f'文件采样点数: {info["file_nsamp"]}')
    print(f'偏振数: {info["npol"]}，频率通道数: {info["nchan"]}')
    print(f'观测起始 MJD: {info["start_mjd"]:.10f}')

    # 4. 读取 TOA 列表（秒，相对于观测起点）
    toa_list = np.loadtxt(TOA_FILE).reshape(-1)
    print(f'待切信号数: {len(toa_list)}')

    # 5. 计算消色散参数
    shifts, max_shift = calc_dispersion_shift(DM, info['freq'], info['time_reso'])
    print(f'DM={DM}，最大色散延迟: {max_shift} 采样点 '
          f'({max_shift * info["time_reso"]:.6f} s)')
    print(f'切出片段长度 A={SEGMENT_LENGTH}，额外消色散长度 C={max_shift}')

    # 6. 并行切数据
    args_list = [
        (DATA_PATH, SAVE_PATH, file_list, info, DM, toa,
         shifts, max_shift, SEGMENT_LENGTH, FRB_NAME, DATE, BEAM)
        for toa in toa_list
    ]

    if NUM_WORKERS > 1 and len(args_list) > 1:
        with Pool(NUM_WORKERS) as pool:
            pool.starmap(cut_one_burst, args_list)
    else:
        for args in args_list:
            cut_one_burst(*args)

    # 7. 汇总 obs_info.json
    save_obs_json(SAVE_PATH, info, DM)

    print('全部完成')
