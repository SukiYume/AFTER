# fmt: off

"""仓库根目录下的爆发切片启动与导入入口。

直接运行时启动 :mod:`after.cut_burst_data`：

    python cut_burst_data.py

作为模块导入时公开 :mod:`after.cut_burst_data` 的接口。
"""

if __name__ == "__main__":
    from runpy import run_module

    run_module("after.cut_burst_data", run_name="__main__", alter_sys=True)
else:
    from after.cut_burst_data import *  # noqa: F401,F403

# fmt: on
