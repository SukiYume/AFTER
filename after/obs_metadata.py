"""Shared helpers for building trustworthy ``obs_info.json`` metadata."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import h5py
import numpy as np


_SUMMARY_FIELDS = (
    'obs_start_mjd',
    'nchan',
    'time_reso',
    'npol',
    'beam',
    'dm',
    'segment_length',
)


def _json_scalar(value):
    """Convert an HDF5 scalar to a strict-JSON-compatible Python value."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode('utf-8', errors='replace')
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _collapse(values):
    """Return one scalar for uniform metadata, otherwise a stable value list."""
    unique = []
    for value in values:
        if value not in unique:
            unique.append(value)
    def sort_key(value):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return (0, float(value))
        if value is None:
            return (2, '')
        return (1, str(value))

    unique.sort(key=sort_key)
    return unique[0] if len(unique) == 1 else unique


def _beam_from_filename(filename):
    match = re.search(r'(?:^|[-_])M(\d{2})(?=[-_.]|$)', filename,
                      flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _normalize_beam(value, filename):
    value = _json_scalar(value)
    if value is None:
        return _beam_from_filename(filename)
    if isinstance(value, str):
        match = re.fullmatch(r'M?(\d+)', value, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    if isinstance(value, (int, float)) and float(value).is_integer():
        return int(value)
    return value


def build_obs_info(h5_files):
    """Build a summary from each H5 file instead of the last processed group."""
    records = []
    for h5_path in sorted(Path(path) for path in h5_files):
        with h5py.File(h5_path, 'r') as h5:
            record = {
                field: _json_scalar(h5.attrs.get(field))
                for field in _SUMMARY_FIELDS
            }
            record['beam'] = _normalize_beam(
                record['beam'], h5_path.name)
            record['file'] = h5_path.name
            record['toa_sec'] = _json_scalar(h5.attrs.get('toa_sec'))
            if isinstance(record['toa_sec'], float):
                record['toa_sec'] = round(record['toa_sec'], 4)
            records.append(record)

    if not records:
        return None

    obs_info = {
        field: _collapse([record[field] for record in records])
        for field in _SUMMARY_FIELDS
    }
    obs_info['bursts'] = [
        {
            'file': record['file'],
            'toa_sec': record['toa_sec'],
            'beam': record['beam'],
            'dm': record['dm'],
            'segment_length': record['segment_length'],
        }
        for record in records
    ]
    return obs_info


def write_obs_info_json(output_dir):
    """Scan one observation directory and write its aggregate metadata."""
    output_path = Path(output_dir)
    h5_files = sorted(
        path for path in output_path.iterdir()
        if path.suffix == '.h5' and not path.name.endswith('_cal.h5')
    )
    obs_info = build_obs_info(h5_files)
    if obs_info is None:
        return None

    json_path = output_path / 'obs_info.json'
    with json_path.open('w', encoding='utf-8') as stream:
        json.dump(
            obs_info, stream, indent=2, ensure_ascii=False, allow_nan=False)
    return json_path
