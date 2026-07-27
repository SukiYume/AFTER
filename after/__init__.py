"""AFTER single-observation processing package.

Run workflow stages from the repository root with ``python -m after.<module>``.
Batch catalog workflows remain in :mod:`batch_processing`.
"""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent

DEFAULT_GAIN_CSV = REPOSITORY_ROOT / "gain_para.csv"
DEFAULT_CAL_NPZ = REPOSITORY_ROOT / "highcal_20201014_psr_tny.npz"
DEFAULT_DETECTOR_MODEL = REPOSITORY_ROOT / "models" / "best_model_yolo11n_ema.pth"

__all__ = [
    "PACKAGE_ROOT",
    "REPOSITORY_ROOT",
    "DEFAULT_GAIN_CSV",
    "DEFAULT_CAL_NPZ",
    "DEFAULT_DETECTOR_MODEL",
]
