# fmt: off

"""Backward-compatible launcher and import shim for :mod:`after.calibration`.

Run this file from the repository root to keep the historical command:

    python calibration.py

Edit the observation-specific constants in ``after/calibration.py``.  Imports
such as ``from calibration import process_one_burst`` remain supported.
"""

if __name__ == "__main__":
    from runpy import run_module

    run_module("after.calibration", run_name="__main__", alter_sys=True)
else:
    from after.calibration import *  # noqa: F401,F403

# fmt: on
