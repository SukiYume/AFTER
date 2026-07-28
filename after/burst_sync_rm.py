"""Search a common rotation measure by stacking calibrated burst H5 files.

The script reads AFTER ``*_cal.h5`` products directly.  Every accepted region
in ``attrs["bursts"]`` is treated as one component.  Time samples are selected
from Stokes I only, and only channel-level RFI masks are applied.

Two complementary common-RM statistics are reported:

``time_pa_power``
    Sum the noise-normalized Faraday power of every selected time sample.
    Each time sample, and therefore every burst, may have an independent PA.
    This is the primary scientific statistic.

``linear_degree_stack``
    Build the standard per-burst RM-versus-linear-degree curve, robustly
    standardize each curve, and combine the curves with fixed pre-declared
    weights.  This is convenient for cached/batch processing and is kept as an
    independent validation of the primary statistic.

Off-pulse time samples, with the same per-burst sample counts and channel
masks, provide a search-window-corrected empirical null distribution.  Pixel
masks are deliberately never read or applied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import warnings as py_warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib
import numpy as np
import pandas as pd

from .burst_analysis import _select_strong_time_samples
from .burst_pol import build_automatic_rm_grid
from .rfi import cal_rfi, robust_channel_mask

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


C_M_S = 299_792_458.0
NULL_RM_GRID_OVERSAMPLE = 4.0
PRIMARY_METHODS = ("time_pa_power", "linear_degree_stack")
ALL_METHODS = PRIMARY_METHODS


@dataclass
class SearchWindow:
    name: str
    low: float
    high: float


@dataclass
class BurstRMData:
    component_id: str
    file_name: str
    file_path: Path
    burst_idx: int
    peak_snr: float
    time_indices: np.ndarray
    freq_mhz: np.ndarray
    wave2_m2: np.ndarray
    p_on: np.ndarray
    p_noise: np.ndarray
    i_total: float
    noise_variance_one_time: float
    stored_cal_rfi_count: int
    stored_burst_rfi_count: int
    recalculated_rfi_count: int
    robust_rfi_count: int
    nonfinite_rfi_count: int
    final_rfi_count: int

    @property
    def n_time(self) -> int:
        return int(self.time_indices.size)

    @property
    def n_channel(self) -> int:
        return int(self.freq_mhz.size)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cal-dir",
        type=Path,
        required=True,
        help="Directory containing labeled calibrated *_cal.h5 files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New output directory. Existing directories are not reused.",
    )
    parser.add_argument(
        "--file-list",
        type=Path,
        default=None,
        help=(
            "Optional text file with one H5 path or cal-dir-relative filename "
            "per line. Blank lines and lines beginning with # are ignored."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Find *_cal.h5 recursively below --cal-dir.",
    )
    parser.add_argument(
        "--rm-min",
        type=float,
        default=-50_000.0,
        help="RM lower bound; grid spacing is derived automatically.",
    )
    parser.add_argument(
        "--rm-max",
        type=float,
        default=50_000.0,
        help="RM upper bound; grid spacing is derived automatically.",
    )
    parser.add_argument(
        "--test-window",
        action="append",
        default=None,
        metavar="NAME:LOW:HIGH",
        help=(
            "Empirical-null search window. Repeat for multiple windows. "
            "Default: full RM grid."
        ),
    )
    parser.add_argument("--n-null", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=12_110_200)
    parser.add_argument(
        "--null-pool-size",
        type=int,
        default=0,
        help=(
            "Maximum off-pulse samples transformed per burst; 0 uses all. "
            "A smaller value is useful for quick validation runs."
        ),
    )
    parser.add_argument(
        "--time-peak-fraction",
        type=float,
        default=0.5,
        help="Keep Stokes-I samples above this fraction of the smoothed peak.",
    )
    parser.add_argument(
        "--min-time-snr",
        type=float,
        default=5.0,
        help="Minimum Stokes-I sample S/N used by the time gate.",
    )
    parser.add_argument(
        "--min-peak-snr",
        type=float,
        default=5.0,
        help="Reject components whose I-only peak S/N is below this value.",
    )
    parser.add_argument(
        "--max-bursts",
        type=int,
        default=None,
        help="Keep only the highest-I-S/N N components after loading.",
    )
    parser.add_argument("--freq-min", type=float, default=None)
    parser.add_argument("--freq-max", type=float, default=None)
    parser.add_argument(
        "--min-channels",
        type=int,
        default=32,
        help="Minimum number of unmasked channels required per component.",
    )
    parser.add_argument(
        "--stored-masks-only",
        action="store_true",
        help=(
            "Use only stored rfi_channel and burst_rfi_channel. By default "
            "channel RFI is also recomputed from off-pulse data."
        ),
    )
    parser.add_argument(
        "--rfi-fft",
        action="store_true",
        help="Use FFT instead of entropy for the recalculated channel mask.",
    )
    parser.add_argument("--rfi-channel-sigma", type=float, default=6.0)
    parser.add_argument("--rfi-channel-window", type=int, default=31)
    parser.add_argument("--rfi-channel-grow", type=int, default=1)
    parser.add_argument(
        "--curve-weighting",
        choices=("equal", "peak-snr"),
        default="equal",
        help=(
            "Fixed weights for linear_degree_stack. Equal is the default to "
            "avoid double-weighting high-S/N bursts."
        ),
    )
    parser.add_argument(
        "--max-weight-ratio",
        type=float,
        default=4.0,
        help="Maximum weight ratio around the median for peak-snr weighting.",
    )
    parser.add_argument(
        "--transform-chunk-size",
        type=int,
        default=256,
        help="Number of trial RMs per matrix-multiplication chunk.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not args.cal_dir.is_dir():
        raise NotADirectoryError(f"Calibrated H5 directory not found: {args.cal_dir}")
    if not np.isfinite(args.rm_min) or not np.isfinite(args.rm_max):
        raise ValueError("--rm-min and --rm-max must be finite")
    if args.rm_max <= args.rm_min:
        raise ValueError("--rm-max must be greater than --rm-min")
    if args.n_null < 0:
        raise ValueError("--n-null cannot be negative")
    if args.null_pool_size < 0:
        raise ValueError("--null-pool-size cannot be negative")
    if not 0 <= args.time_peak_fraction <= 1:
        raise ValueError("--time-peak-fraction must be between 0 and 1")
    if args.min_time_snr < 0 or args.min_peak_snr < 0:
        raise ValueError("S/N thresholds cannot be negative")
    if args.max_bursts is not None and args.max_bursts < 1:
        raise ValueError("--max-bursts must be positive")
    if args.min_channels < 2:
        raise ValueError("--min-channels must be at least 2")
    if args.freq_min is not None and args.freq_max is not None:
        if args.freq_max <= args.freq_min:
            raise ValueError("--freq-max must be greater than --freq-min")
    if args.max_weight_ratio < 1:
        raise ValueError("--max-weight-ratio must be at least 1")
    if args.transform_chunk_size < 1:
        raise ValueError("--transform-chunk-size must be positive")


def parse_search_windows(
    values: list[str] | None,
    rm_min: float,
    rm_max: float,
) -> list[SearchWindow]:
    if not values:
        return [SearchWindow("full", float(rm_min), float(rm_max))]

    windows: list[SearchWindow] = []
    names: set[str] = set()
    for value in values:
        parts = value.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid --test-window {value!r}; expected NAME:LOW:HIGH"
            )
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", parts[0].strip())
        if not name:
            raise ValueError(f"Invalid empty test-window name in {value!r}")
        if name in names:
            raise ValueError(f"Duplicate test-window name: {name}")
        low = float(parts[1])
        high = float(parts[2])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError(f"Invalid test-window limits in {value!r}")
        if low < rm_min or high > rm_max:
            raise ValueError(
                f"Test window {name}={low:g}:{high:g} lies outside "
                f"the RM grid {rm_min:g}:{rm_max:g}"
            )
        names.add(name)
        windows.append(SearchWindow(name, low, high))
    return windows


def discover_h5_files(
    cal_dir: Path,
    file_list: Path | None = None,
    recursive: bool = False,
) -> list[Path]:
    cal_dir = cal_dir.resolve()
    if file_list is not None:
        lines = file_list.read_text(encoding="utf-8").splitlines()
        files = []
        for line in lines:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            path = Path(value)
            if not path.is_absolute():
                path = cal_dir / path
            files.append(path.resolve())
    else:
        iterator: Iterable[Path]
        iterator = cal_dir.rglob("*_cal.h5") if recursive else cal_dir.glob("*_cal.h5")
        files = [path.resolve() for path in iterator]

    unique = sorted(set(files), key=lambda path: str(path))
    missing = [path for path in unique if not path.is_file()]
    if missing:
        preview = "\n".join(f"  {path}" for path in missing[:10])
        raise FileNotFoundError(f"Listed calibrated H5 files do not exist:\n{preview}")
    if not unique:
        raise FileNotFoundError(f"No *_cal.h5 files found below {cal_dir}")
    return unique


def decode_regions(value: object) -> list[dict[str, object]]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        decoded = json.loads(value)
    elif isinstance(value, (list, tuple)):
        decoded = list(value)
    else:
        raise TypeError(f"Unsupported attrs['bursts'] type: {type(value).__name__}")
    if not isinstance(decoded, list):
        raise ValueError("attrs['bursts'] must decode to a list")
    return decoded


def channel_mask(handle: h5py.File, name: str, n_channel: int) -> np.ndarray:
    if name not in handle:
        return np.zeros(n_channel, dtype=bool)
    mask = np.asarray(handle[name][:], dtype=bool)
    if mask.shape != (n_channel,):
        raise ValueError(
            f"{handle.filename}: dataset {name} has shape {mask.shape}, "
            f"expected {(n_channel,)}"
        )
    return mask


def robust_channel_variance(noise: np.ndarray) -> np.ndarray:
    values = np.asarray(noise, dtype=np.float64)
    center = np.nanmedian(values, axis=0, keepdims=True)
    sigma = 1.4826 * np.nanmedian(np.abs(values - center), axis=0)
    fallback = np.nanstd(values, axis=0)
    bad = ~np.isfinite(sigma) | (sigma <= 0)
    sigma[bad] = fallback[bad]
    positive = sigma[np.isfinite(sigma) & (sigma > 0)]
    replacement = float(np.median(positive)) if positive.size else 1.0
    sigma[~np.isfinite(sigma) | (sigma <= 0)] = replacement
    return sigma**2


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return math.nan, math.nan
    center = float(np.median(finite))
    scale = float(1.4826 * np.median(np.abs(finite - center)))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(finite))
    return center, scale


def standardized_curve(values: np.ndarray) -> np.ndarray:
    center, scale = robust_location_scale(values)
    if not np.isfinite(scale) or scale <= 0:
        return np.zeros_like(values, dtype=np.float64)
    return (np.asarray(values, dtype=np.float64) - center) / scale


def short_file_id(file_name: str) -> str:
    match = re.search(r"-M\d{2}-(\d{4})-", file_name)
    return match.group(1) if match else Path(file_name).stem


def component_id(file_name: str, burst_idx: int) -> str:
    return f"{short_file_id(file_name)}b{int(burst_idx)}"


def unique_file_id(file_name: str) -> str:
    stem = Path(file_name).stem
    return stem[:-4] if stem.lower().endswith("_cal") else stem


def disambiguate_component_ids(bursts: list[BurstRMData]) -> None:
    """Expand colliding short IDs with the full observation filename."""
    counts: dict[str, int] = {}
    for burst in bursts:
        counts[burst.component_id] = counts.get(burst.component_id, 0) + 1
    for burst in bursts:
        if counts[burst.component_id] > 1:
            burst.component_id = (
                f"{unique_file_id(burst.file_name)}b{int(burst.burst_idx)}"
            )
    resolved = [burst.component_id for burst in bursts]
    if len(resolved) != len(set(resolved)):
        raise ValueError("Component IDs remain non-unique after disambiguation")


def complex_faraday_transform(
    p_data: np.ndarray,
    wave2_m2: np.ndarray,
    rm_grid: np.ndarray,
    chunk_size: int = 256,
    output_dtype: np.dtype = np.dtype(np.complex128),
) -> np.ndarray:
    """Return F(time, RM) after derotating Q+iU on a centered lambda^2 grid."""
    dtype = np.dtype(output_dtype)
    p_work = np.asarray(p_data, dtype=dtype)
    wave2 = np.asarray(wave2_m2, dtype=np.float64)
    rm_values = np.asarray(rm_grid, dtype=np.float64)
    wave2_centered = wave2 - float(np.mean(wave2))
    output = np.empty((p_work.shape[0], rm_values.size), dtype=dtype)
    for first in range(0, rm_values.size, chunk_size):
        last = min(first + chunk_size, rm_values.size)
        phase = np.exp(
            -2j * wave2_centered[:, None] * rm_values[None, first:last]
        ).astype(dtype, copy=False)
        output[:, first:last] = p_work @ phase
    return output


def recompute_channel_rfi(
    iquv: np.ndarray,
    noise_mask: np.ndarray,
    *,
    fft: bool,
    sigma: float,
    local_window: int,
    grow: int,
) -> tuple[np.ndarray, np.ndarray]:
    recalculated = []
    for plane in iquv:
        channel, _ = cal_rfi(
            plane,
            noise_mask,
            down_time=1,
            down_freq=1,
            fft=fft,
        )
        recalculated.append(channel)
    entropy_or_fft = np.logical_or.reduce(recalculated)
    robust = robust_channel_mask(
        iquv,
        noise_mask,
        sigma=sigma,
        local_window=local_window,
        grow=grow,
    )
    return entropy_or_fft, robust


def load_file_components(
    path: Path,
    *,
    freq_min: float | None,
    freq_max: float | None,
    time_peak_fraction: float,
    min_time_snr: float,
    min_channels: int,
    stored_masks_only: bool,
    rfi_fft: bool,
    rfi_channel_sigma: float,
    rfi_channel_window: int,
    rfi_channel_grow: int,
) -> tuple[list[BurstRMData], list[str]]:
    warnings: list[str] = []
    with h5py.File(path, "r") as handle:
        if "data" not in handle or "freq" not in handle:
            raise KeyError(f"{path}: calibrated H5 must contain data and freq")
        iquv = np.asarray(handle["data"][:], dtype=np.float64)
        freq = np.asarray(handle["freq"][:], dtype=np.float64)
        if iquv.ndim != 3 or iquv.shape[0] < 3:
            raise ValueError(
                f"{path}: data shape must be (>=3, nsamp, nchan), got {iquv.shape}"
            )
        if freq.shape != (iquv.shape[2],):
            raise ValueError(
                f"{path}: freq shape {freq.shape} does not match data {iquv.shape}"
            )
        if "bursts" not in handle.attrs:
            warnings.append(f"{path.name}: missing attrs['bursts']; skipped")
            return [], warnings
        regions = decode_regions(handle.attrs["bursts"])
        stored_cal = channel_mask(handle, "rfi_channel", freq.size)
        stored_burst = channel_mask(handle, "burst_rfi_channel", freq.size)

    if not regions:
        warnings.append(f"{path.name}: attrs['bursts'] is empty; skipped")
        return [], warnings

    nsamp = iquv.shape[1]
    valid_regions: list[tuple[int, dict[str, object]]] = []
    for burst_idx, region in enumerate(regions):
        ts = max(0, min(nsamp, int(region["time_start"])))
        te = max(0, min(nsamp, int(region["time_end"])))
        fs = max(0, min(freq.size, int(region["freq_start"])))
        fe = max(0, min(freq.size, int(region["freq_end"])))
        if te <= ts or fe <= fs:
            warnings.append(
                f"{path.name} burst {burst_idx}: empty/clipped region; skipped"
            )
            continue
        clean_region = dict(region)
        clean_region.update(
            {"time_start": ts, "time_end": te, "freq_start": fs, "freq_end": fe}
        )
        valid_regions.append((burst_idx, clean_region))

    if not valid_regions:
        return [], warnings

    noise_mask = np.ones(nsamp, dtype=bool)
    for _, region in valid_regions:
        noise_mask[int(region["time_start"]) : int(region["time_end"])] = False
    if np.count_nonzero(noise_mask) < 3:
        raise ValueError(f"{path}: fewer than three off-pulse time samples")

    finite_noise_count = np.sum(
        np.isfinite(iquv[:, noise_mask, :]), axis=1
    )
    nonfinite_channel = np.any(finite_noise_count < 3, axis=0)
    with py_warnings.catch_warnings():
        py_warnings.filterwarnings(
            "ignore",
            message="All-NaN slice encountered",
            category=RuntimeWarning,
        )
        baseline = np.nanmedian(iquv[:, noise_mask, :], axis=1, keepdims=True)
    iquv = iquv - baseline

    if stored_masks_only:
        recalculated = np.zeros(freq.size, dtype=bool)
        robust = np.zeros(freq.size, dtype=bool)
    else:
        rfi_work = np.nan_to_num(
            iquv, nan=0.0, posinf=0.0, neginf=0.0
        )
        recalculated, robust = recompute_channel_rfi(
            rfi_work,
            noise_mask,
            fft=rfi_fft,
            sigma=rfi_channel_sigma,
            local_window=rfi_channel_window,
            grow=rfi_channel_grow,
        )
    final_rfi = (
        stored_cal
        | stored_burst
        | recalculated
        | robust
        | nonfinite_channel
    )

    components: list[BurstRMData] = []
    for burst_idx, region in valid_regions:
        good = np.zeros(freq.size, dtype=bool)
        good[int(region["freq_start"]) : int(region["freq_end"])] = True
        good &= ~final_rfi
        if freq_min is not None:
            good &= freq >= freq_min
        if freq_max is not None:
            good &= freq <= freq_max
        if np.count_nonzero(good) < min_channels:
            warnings.append(
                f"{path.name} burst {burst_idx}: only "
                f"{np.count_nonzero(good)} good channels; skipped"
            )
            continue

        time_mask, time_info = _select_strong_time_samples(
            iquv[0],
            good,
            noise_mask,
            region,
            peak_fraction=time_peak_fraction,
            min_snr=min_time_snr,
        )
        times = np.flatnonzero(time_mask)
        if times.size == 0:
            warnings.append(
                f"{path.name} burst {burst_idx}: empty I-only time gate; skipped"
            )
            continue

        i_on = np.nan_to_num(iquv[0, times][:, good], nan=0.0)
        q_on = np.nan_to_num(iquv[1, times][:, good], nan=0.0)
        u_on = np.nan_to_num(iquv[2, times][:, good], nan=0.0)
        q_noise = np.nan_to_num(iquv[1, noise_mask][:, good], nan=0.0)
        u_noise = np.nan_to_num(iquv[2, noise_mask][:, good], nan=0.0)
        variance_channel = (
            robust_channel_variance(q_noise) + robust_channel_variance(u_noise)
        )
        variance_one_time = float(np.sum(variance_channel))
        if not np.isfinite(variance_one_time) or variance_one_time <= 0:
            warnings.append(
                f"{path.name} burst {burst_idx}: invalid Q/U noise variance; skipped"
            )
            continue

        freq_good = freq[good]
        wave2 = (C_M_S / (freq_good * 1e6)) ** 2
        components.append(
            BurstRMData(
                component_id=component_id(path.name, burst_idx),
                file_name=path.name,
                file_path=path,
                burst_idx=burst_idx,
                peak_snr=float(time_info["peak_snr"]),
                time_indices=times,
                freq_mhz=freq_good,
                wave2_m2=wave2,
                p_on=q_on + 1j * u_on,
                p_noise=q_noise + 1j * u_noise,
                i_total=float(np.sum(i_on)),
                noise_variance_one_time=variance_one_time,
                stored_cal_rfi_count=int(np.count_nonzero(stored_cal)),
                stored_burst_rfi_count=int(np.count_nonzero(stored_burst)),
                recalculated_rfi_count=int(np.count_nonzero(recalculated)),
                robust_rfi_count=int(np.count_nonzero(robust)),
                nonfinite_rfi_count=int(np.count_nonzero(nonfinite_channel)),
                final_rfi_count=int(np.count_nonzero(final_rfi)),
            )
        )
    return components, warnings


def curve_weights(
    bursts: list[BurstRMData],
    mode: str,
    max_weight_ratio: float,
) -> np.ndarray:
    if mode == "equal":
        return np.ones(len(bursts), dtype=np.float64)
    if mode != "peak-snr":
        raise ValueError(f"Unknown curve-weighting mode: {mode}")
    raw = np.asarray([max(burst.peak_snr, np.finfo(float).tiny) for burst in bursts])
    median = float(np.median(raw))
    low = median / max_weight_ratio
    high = median * max_weight_ratio
    return np.clip(raw, low, high) / median


def individual_rm_curves(
    burst: BurstRMData,
    rm_grid: np.ndarray,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    faraday = complex_faraday_transform(
        burst.p_on,
        burst.wave2_m2,
        rm_grid,
        chunk_size=chunk_size,
        output_dtype=np.dtype(np.complex128),
    )
    variance = max(burst.noise_variance_one_time, np.finfo(float).tiny)
    linear_degree = np.sum(np.abs(faraday), axis=0) / max(
        abs(burst.i_total), np.finfo(float).tiny
    )
    time_pa_power = np.sum(np.abs(faraday) ** 2, axis=0) / variance
    return {
        "linear_degree": linear_degree,
        "time_pa_power": time_pa_power,
    }


def combine_curves(
    bursts: list[BurstRMData],
    curves: dict[str, dict[str, np.ndarray]],
    weights: np.ndarray,
) -> dict[str, np.ndarray]:
    time_pa_power = sum(
        curves[burst.component_id]["time_pa_power"] for burst in bursts
    )
    denominator = math.sqrt(float(np.sum(weights**2)))
    linear_degree_stack = sum(
        weight * standardized_curve(curves[burst.component_id]["linear_degree"])
        for burst, weight in zip(bursts, weights, strict=True)
    ) / max(denominator, np.finfo(float).tiny)
    return {
        "time_pa_power": np.asarray(time_pa_power, dtype=np.float64),
        "linear_degree_stack": np.asarray(linear_degree_stack, dtype=np.float64),
    }


def observed_curves_on_grid(
    bursts: list[BurstRMData],
    fine_grid: np.ndarray,
    fine_individual: dict[str, dict[str, np.ndarray]],
    fine_combined: dict[str, np.ndarray],
    target_grid: np.ndarray,
    weights: np.ndarray,
) -> dict[str, np.ndarray]:
    time_pa_power = np.interp(
        target_grid, fine_grid, fine_combined["time_pa_power"]
    )
    denominator = math.sqrt(float(np.sum(weights**2)))
    linear_degree_stack = sum(
        weight
        * standardized_curve(
            np.interp(
                target_grid,
                fine_grid,
                fine_individual[burst.component_id]["linear_degree"],
            )
        )
        for burst, weight in zip(bursts, weights, strict=True)
    ) / max(denominator, np.finfo(float).tiny)
    return {
        "time_pa_power": time_pa_power,
        "linear_degree_stack": linear_degree_stack,
    }


def peak_in_window(
    rm_grid: np.ndarray,
    curve: np.ndarray,
    window: SearchWindow,
) -> tuple[float, float, float]:
    inside = (rm_grid >= window.low) & (rm_grid <= window.high)
    indices = np.flatnonzero(inside)
    if indices.size == 0:
        raise ValueError(f"No RM samples inside test window {window.name}")
    local = np.asarray(curve)[inside]
    local_index = int(np.nanargmax(local))
    index = int(indices[local_index])
    z_curve = standardized_curve(curve)
    return (
        float(rm_grid[index]),
        float(curve[index]),
        float(z_curve[index]),
    )


def precompute_noise_transforms(
    bursts: list[BurstRMData],
    rm_grid: np.ndarray,
    pool_size: int,
    rng: np.random.Generator,
    chunk_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    transforms: dict[str, np.ndarray] = {}
    pool_indices: dict[str, np.ndarray] = {}
    for burst in bursts:
        available = burst.p_noise.shape[0]
        requested = available if pool_size == 0 else min(pool_size, available)
        requested = max(requested, burst.n_time)
        requested = min(requested, available)
        if requested < burst.n_time:
            raise ValueError(
                f"{burst.component_id}: only {available} off-pulse samples for "
                f"a {burst.n_time}-sample burst gate"
            )
        if requested == available:
            chosen = np.arange(available, dtype=int)
        else:
            chosen = np.sort(rng.choice(available, size=requested, replace=False))
        pool_indices[burst.component_id] = chosen
        transforms[burst.component_id] = complex_faraday_transform(
            burst.p_noise[chosen],
            burst.wave2_m2,
            rm_grid,
            chunk_size=chunk_size,
            output_dtype=np.dtype(np.complex64),
        )
        print(
            f"  null pool {burst.component_id}: {requested}/{available} "
            "off-pulse samples transformed",
            flush=True,
        )
    return transforms, pool_indices


def null_trial_curves(
    bursts: list[BurstRMData],
    noise_transforms: dict[str, np.ndarray],
    weights: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    time_pa_power: np.ndarray | None = None
    linear_stack: np.ndarray | None = None
    for burst, weight in zip(bursts, weights, strict=True):
        pool = noise_transforms[burst.component_id]
        chosen = rng.choice(pool.shape[0], size=burst.n_time, replace=False)
        faraday = pool[chosen]
        variance = max(burst.noise_variance_one_time, np.finfo(float).tiny)
        time_curve = np.sum(np.abs(faraday) ** 2, axis=0) / variance
        linear_curve = np.sum(np.abs(faraday), axis=0)
        linear_z = weight * standardized_curve(linear_curve)
        time_pa_power = (
            time_curve if time_pa_power is None else time_pa_power + time_curve
        )
        linear_stack = (
            linear_z if linear_stack is None else linear_stack + linear_z
        )
    assert time_pa_power is not None
    assert linear_stack is not None
    linear_stack /= max(
        math.sqrt(float(np.sum(weights**2))), np.finfo(float).tiny
    )
    return {
        "time_pa_power": np.asarray(time_pa_power, dtype=np.float64),
        "linear_degree_stack": np.asarray(linear_stack, dtype=np.float64),
    }


def run_empirical_null(
    bursts: list[BurstRMData],
    observed_curves: dict[str, np.ndarray],
    rm_grid: np.ndarray,
    windows: list[SearchWindow],
    weights: np.ndarray,
    *,
    n_null: int,
    pool_size: int,
    seed: int,
    chunk_size: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    observed: dict[tuple[str, str], tuple[float, float, float]] = {}
    maxima: dict[tuple[str, str], np.ndarray] = {}
    for method in ALL_METHODS:
        for window in windows:
            key = (method, window.name)
            observed[key] = peak_in_window(
                rm_grid, observed_curves[method], window
            )
            maxima[key] = np.empty(n_null, dtype=np.float64)

    pool_rng = np.random.default_rng(seed)
    noise_transforms, pool_indices = precompute_noise_transforms(
        bursts,
        rm_grid,
        pool_size,
        pool_rng,
        chunk_size,
    )
    trial_rng = np.random.default_rng(seed + 1)
    progress_step = max(1, n_null // 10)
    for trial in range(n_null):
        trial_curves = null_trial_curves(
            bursts, noise_transforms, weights, trial_rng
        )
        for method in ALL_METHODS:
            for window in windows:
                _, statistic, _ = peak_in_window(
                    rm_grid, trial_curves[method], window
                )
                maxima[(method, window.name)][trial] = statistic
        if (trial + 1) % progress_step == 0 or trial + 1 == n_null:
            print(f"  null trials {trial + 1}/{n_null}", flush=True)

    rows: list[dict[str, object]] = []
    for method in ALL_METHODS:
        for window in windows:
            key = (method, window.name)
            observed_rm, observed_statistic, observed_z = observed[key]
            values = maxima[key]
            exceedances = int(np.count_nonzero(values >= observed_statistic))
            rows.append(
                {
                    "method": method,
                    "method_role": (
                        "requested" if method in PRIMARY_METHODS else "diagnostic"
                    ),
                    "window": window.name,
                    "window_rm_min": window.low,
                    "window_rm_max": window.high,
                    "null_grid_peak_rm": observed_rm,
                    "observed_null_grid_peak_statistic": observed_statistic,
                    "observed_null_grid_peak_robust_z": observed_z,
                    "null_exceedances": exceedances,
                    "n_null": n_null,
                    "empirical_p": float((exceedances + 1) / (n_null + 1)),
                    "null_p95_max_statistic": float(np.percentile(values, 95)),
                    "null_p99_max_statistic": float(np.percentile(values, 99)),
                }
            )
    pool_archive = {
        component: indices.astype(np.int32)
        for component, indices in pool_indices.items()
    }
    null_archive = {
        f"{method}__{window}": values
        for (method, window), values in maxima.items()
    }
    return pd.DataFrame(rows), null_archive, pool_archive


def fine_grid_summary(
    rm_grid: np.ndarray,
    combined: dict[str, np.ndarray],
    windows: list[SearchWindow],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in ALL_METHODS:
        for window in windows:
            rm, statistic, robust_z = peak_in_window(
                rm_grid, combined[method], window
            )
            rows.append(
                {
                    "method": method,
                    "method_role": "requested",
                    "window": window.name,
                    "fine_grid_peak_rm": rm,
                    "fine_grid_peak_statistic": statistic,
                    "fine_grid_peak_robust_z": robust_z,
                }
            )
    return pd.DataFrame(rows)


def write_plot(
    output_path: Path,
    rm_grid: np.ndarray,
    bursts: list[BurstRMData],
    individual: dict[str, dict[str, np.ndarray]],
    combined: dict[str, np.ndarray],
    windows: list[SearchWindow],
    null_table: pd.DataFrame | None,
    null_maxima: dict[str, np.ndarray] | None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for burst in bursts:
        axes[0, 0].plot(
            rm_grid,
            standardized_curve(individual[burst.component_id]["linear_degree"]),
            lw=0.8,
            label=f"{burst.component_id} I-S/N={burst.peak_snr:.1f}",
        )
    axes[0, 0].set_title("Individual RM–linear-degree curves")
    axes[0, 0].set_xlabel(r"RM (rad m$^{-2}$)")
    axes[0, 0].set_ylabel("Robust z")
    axes[0, 0].legend(fontsize=7, ncol=2)

    colors = {
        "time_pa_power": "C2",
        "linear_degree_stack": "C1",
    }
    for method in PRIMARY_METHODS:
        axes[0, 1].plot(
            rm_grid,
            standardized_curve(combined[method]),
            color=colors[method],
            label=method,
        )
    axes[0, 1].set_title("Requested common-RM methods")
    axes[0, 1].set_xlabel(r"RM (rad m$^{-2}$)")
    axes[0, 1].set_ylabel("Robust z")
    axes[0, 1].legend(fontsize=8)

    primary_window = windows[0]
    for axis, method in zip(axes[1], PRIMARY_METHODS, strict=True):
        if null_table is not None and null_maxima is not None:
            row = null_table[
                (null_table["method"] == method)
                & (null_table["window"] == primary_window.name)
            ].iloc[0]
            values = null_maxima[f"{method}__{primary_window.name}"]
            axis.hist(values, bins=40, color=colors[method], alpha=0.65)
            axis.axvline(
                float(row["observed_null_grid_peak_statistic"]),
                color="C3",
                lw=2,
                label=(
                    f"RM={float(row['null_grid_peak_rm']):.0f}, "
                    f"p={float(row['empirical_p']):.4g}"
                ),
            )
            axis.set_title(
                f"Off-pulse maximum null: {method} / "
                f"{primary_window.name}"
            )
            axis.set_xlabel("Maximum search statistic")
            axis.set_ylabel("Trials")
            axis.legend(fontsize=8)
        else:
            axis.axis("off")
            axis.text(
                0.5,
                0.5,
                f"{method}\nEmpirical null disabled (--n-null 0)",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )

    for axis in axes[0]:
        for window in windows:
            axis.axvspan(window.low, window.high, color="0.8", alpha=0.08)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    cal_dir = args.cal_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {output_dir}")
    windows = parse_search_windows(args.test_window, args.rm_min, args.rm_max)
    files = discover_h5_files(cal_dir, args.file_list, args.recursive)

    output_dir.mkdir(parents=True)
    print(f"Found {len(files)} calibrated H5 files in {cal_dir}", flush=True)
    bursts: list[BurstRMData] = []
    warnings: list[str] = []
    for path in files:
        components, file_warnings = load_file_components(
            path,
            freq_min=args.freq_min,
            freq_max=args.freq_max,
            time_peak_fraction=args.time_peak_fraction,
            min_time_snr=args.min_time_snr,
            min_channels=args.min_channels,
            stored_masks_only=args.stored_masks_only,
            rfi_fft=args.rfi_fft,
            rfi_channel_sigma=args.rfi_channel_sigma,
            rfi_channel_window=args.rfi_channel_window,
            rfi_channel_grow=args.rfi_channel_grow,
        )
        bursts.extend(components)
        warnings.extend(file_warnings)

    bursts = [
        burst for burst in bursts if burst.peak_snr >= args.min_peak_snr
    ]
    bursts.sort(key=lambda burst: burst.peak_snr, reverse=True)
    if args.max_bursts is not None:
        bursts = bursts[: args.max_bursts]
    if not bursts:
        raise RuntimeError("No components survived the I-only selection criteria")
    disambiguate_component_ids(bursts)

    print(f"Selected {len(bursts)} burst components:", flush=True)
    for burst in bursts:
        print(
            f"  {burst.component_id}: I-peak S/N={burst.peak_snr:.2f}, "
            f"time={burst.n_time}, channels={burst.n_channel}, "
            f"band={burst.freq_mhz.min():.1f}-{burst.freq_mhz.max():.1f} MHz, "
            f"channel-RFI={burst.final_rfi_count}",
            flush=True,
        )
    for warning in warnings:
        print(f"WARNING: {warning}", flush=True)

    lambda2_sets = [burst.wave2_m2 for burst in bursts]
    rm_grid, rm_grid_info = build_automatic_rm_grid(
        args.rm_min,
        args.rm_max,
        lambda2_sets,
    )
    null_grid, null_grid_info = build_automatic_rm_grid(
        args.rm_min,
        args.rm_max,
        lambda2_sets,
        oversample=NULL_RM_GRID_OVERSAMPLE,
    )
    weights = curve_weights(
        bursts, args.curve_weighting, args.max_weight_ratio
    )

    individual: dict[str, dict[str, np.ndarray]] = {}
    print(
        f"Automatic RM grid: {rm_grid.size} points, "
        f"step={rm_grid_info['rm_step']:.3f} rad/m^2, "
        f"RMSF FWHM={rm_grid_info['rmsf_fwhm']:.3f} rad/m^2",
        flush=True,
    )
    print(f"Computing observed curves on {rm_grid.size} RM points...", flush=True)
    for burst in bursts:
        individual[burst.component_id] = individual_rm_curves(
            burst, rm_grid, args.transform_chunk_size
        )
        print(f"  observed {burst.component_id}", flush=True)
    combined = combine_curves(bursts, individual, weights)
    observed_null_grid = observed_curves_on_grid(
        bursts,
        rm_grid,
        individual,
        combined,
        null_grid,
        weights,
    )

    fine_table = fine_grid_summary(rm_grid, combined, windows)
    null_table: pd.DataFrame | None = None
    null_archive: dict[str, np.ndarray] | None = None
    pool_archive: dict[str, np.ndarray] | None = None
    if args.n_null > 0:
        print(
            f"Computing off-pulse null on {null_grid.size} RM points "
            f"for {args.n_null} trials...",
            flush=True,
        )
        null_table, null_archive, pool_archive = run_empirical_null(
            bursts,
            observed_null_grid,
            null_grid,
            windows,
            weights,
            n_null=args.n_null,
            pool_size=args.null_pool_size,
            seed=args.seed,
            chunk_size=args.transform_chunk_size,
        )
        summary_table = fine_table.merge(
            null_table,
            on=["method", "method_role", "window"],
            how="left",
        )
    else:
        summary_table = fine_table

    member_string = ";".join(burst.component_id for burst in bursts)
    summary_table.insert(0, "members", member_string)
    summary_table.insert(0, "n_members", len(bursts))
    summary_table.insert(2, "rm_grid_size", rm_grid_info["n_rm"])
    summary_table.insert(3, "rm_grid_step", rm_grid_info["rm_step"])
    summary_table.insert(4, "rm_rmsf_fwhm", rm_grid_info["rmsf_fwhm"])
    summary_table.insert(
        5,
        "rm_grid_samples_per_fwhm",
        rm_grid_info["samples_per_fwhm"],
    )
    summary_table.insert(6, "null_rm_grid_step", null_grid_info["rm_step"])
    summary_table.to_csv(output_dir / "burst_sync_rm_summary.csv", index=False)

    selection_rows = []
    for rank, (burst, weight) in enumerate(
        zip(bursts, weights, strict=True), start=1
    ):
        selection_rows.append(
            {
                "rank": rank,
                "component_id": burst.component_id,
                "file_name": burst.file_name,
                "file_path": str(burst.file_path),
                "burst_idx": burst.burst_idx,
                "i_peak_snr": burst.peak_snr,
                "curve_weight": float(weight),
                "time_indices": ";".join(
                    str(int(value)) for value in burst.time_indices
                ),
                "n_time_samples": burst.n_time,
                "n_frequency_channels": burst.n_channel,
                "frequency_min_mhz": float(burst.freq_mhz.min()),
                "frequency_max_mhz": float(burst.freq_mhz.max()),
                "stored_cal_rfi_count": burst.stored_cal_rfi_count,
                "stored_burst_rfi_count": burst.stored_burst_rfi_count,
                "recalculated_rfi_count": burst.recalculated_rfi_count,
                "robust_rfi_count": burst.robust_rfi_count,
                "nonfinite_rfi_count": burst.nonfinite_rfi_count,
                "final_rfi_count": burst.final_rfi_count,
                "offpulse_sample_count": int(burst.p_noise.shape[0]),
            }
        )
    pd.DataFrame(selection_rows).to_csv(
        output_dir / "selected_bursts.csv", index=False
    )

    curve_archive: dict[str, np.ndarray] = {"rm_grid": rm_grid}
    for method, curve in combined.items():
        curve_archive[f"combined__{method}"] = curve
    for burst in bursts:
        for name, curve in individual[burst.component_id].items():
            curve_archive[f"{burst.component_id}__{name}"] = curve
    np.savez_compressed(
        output_dir / "burst_sync_rm_curves.npz", **curve_archive
    )
    if null_archive is not None and pool_archive is not None:
        np.savez_compressed(
            output_dir / "offpulse_null_maxima.npz",
            rm_grid=null_grid,
            **null_archive,
        )
        np.savez_compressed(
            output_dir / "offpulse_pool_indices.npz", **pool_archive
        )

    write_plot(
        output_dir / "burst_sync_rm.png",
        rm_grid,
        bursts,
        individual,
        combined,
        windows,
        null_table,
        null_archive,
    )

    script_path = Path(__file__).resolve()
    manifest = {
        "script": str(script_path),
        "script_sha256": sha256_file(script_path),
        "cal_dir": str(cal_dir),
        "output_dir": str(output_dir),
        "input_files": [str(path) for path in files],
        "selected_components": [row["component_id"] for row in selection_rows],
        "pixel_mask": "not read or applied",
        "channel_mask_policy": (
            "stored calibration + stored detection"
            if args.stored_masks_only
            else "stored calibration + stored detection + recalculated all-Stokes "
            "+ robust local channel mask"
        ),
        "time_gate_policy": (
            "Stokes-I only; smoothed profile above peak fraction and minimum S/N"
        ),
        "methods": {
            "time_pa_power": (
                "sum over bursts and selected time samples of |F_bt(RM)|^2 "
                "divided by per-time Q/U noise variance"
            ),
            "linear_degree_stack": (
                "fixed-weight sum of robustly standardized per-burst "
                "RM-versus-linear-degree curves"
            ),
        },
        "curve_weighting": args.curve_weighting,
        "curve_weights": {
            burst.component_id: float(weight)
            for burst, weight in zip(bursts, weights, strict=True)
        },
        "rm_grid": {
            "min": args.rm_min,
            "max": args.rm_max,
            "n": rm_grid_info["n_rm"],
            "step": rm_grid_info["rm_step"],
            "delta_lambda2": rm_grid_info["delta_lambda2"],
            "rmsf_fwhm": rm_grid_info["rmsf_fwhm"],
            "oversample_target": rm_grid_info["oversample_target"],
            "samples_per_fwhm": rm_grid_info["samples_per_fwhm"],
        },
        "test_windows": [
            {"name": window.name, "low": window.low, "high": window.high}
            for window in windows
        ],
        "null": {
            "rm_n": null_grid_info["n_rm"],
            "rm_step": null_grid_info["rm_step"],
            "oversample_target": null_grid_info["oversample_target"],
            "n_null": args.n_null,
            "seed": args.seed,
            "pool_size_limit": args.null_pool_size,
        },
        "selection": {
            "time_peak_fraction": args.time_peak_fraction,
            "min_time_snr": args.min_time_snr,
            "min_peak_snr": args.min_peak_snr,
            "max_bursts": args.max_bursts,
            "freq_min": args.freq_min,
            "freq_max": args.freq_max,
            "min_channels": args.min_channels,
        },
        "warnings": warnings,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    summary_lines = [
        "Joint burst RM search",
        f"cal_dir={cal_dir}",
        f"components={member_string}",
        "pixel_mask=not read or applied",
        f"curve_weighting={args.curve_weighting}",
        "",
        summary_table.to_string(index=False),
    ]
    if warnings:
        summary_lines.extend(["", "Warnings:", *warnings])
    (output_dir / "burst_sync_rm_summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    print(summary_table.to_string(index=False), flush=True)
    print(f"DONE output={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
