# fmt: off

"""仓库根目录下的定标启动与导入入口。

直接运行时启动 :mod:`after.calibration`：

    python calibration.py

作为模块导入时公开 :mod:`after.calibration` 的接口。
"""

if __name__ == "__main__":
    from runpy import run_module

    run_module("after.calibration", run_name="__main__", alter_sys=True)
else:
    from after.calibration import *  # noqa: F401,F403

# fmt: on
