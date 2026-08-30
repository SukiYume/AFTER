# fmt: off

"""Backward-compatible launcher and import shim for :mod:`after.cut_burst_data`.

Run this file from the repository root to keep the historical command:

    python cut_burst_data.py

Edit the observation-specific constants in ``after/cut_burst_data.py``.
Imports such as ``from cut_burst_data import cut_one_burst`` remain supported.
"""

if __name__ == "__main__":
    from runpy import run_module

    run_module("after.cut_burst_data", run_name="__main__", alter_sys=True)
else:
    from after.cut_burst_data import *  # noqa: F401,F403

# fmt: on
