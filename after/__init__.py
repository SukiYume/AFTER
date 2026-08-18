"""AFTER 单次观测处理包。

各处理阶段都可以在仓库根目录用 ``python -m after.<模块名>`` 运行；面向整份
爆发目录或观测表的批处理入口放在 :mod:`batch_processing`。本模块还集中解析仓库
内置的增益表、噪声管温度表和检测模型路径，避免各子模块依赖当前工作目录。
"""

from pathlib import Path


# 所有默认资源都相对源码文件定位。这样从任意目录执行 ``python -m after...`` 时，
# 都不会因为当前工作目录不同而读错模型或定标文件。
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
