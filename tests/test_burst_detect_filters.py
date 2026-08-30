# fmt: off

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

# 这里测试检测阶段的 RFI 数据流。用最小模块桩隔离未参与测试、且在
# Windows 上加载较慢的 Torch/Ultralytics GPU 栈。
_STUBBED_MODULE_NAMES = (
    "torch",
    "ultralytics",
    "ultralytics.nn",
    "ultralytics.nn.tasks",
    "ultralytics.cfg",
    "seaborn",
    "scipy",
    "scipy.ndimage",
    "after.rfi",
)
_ORIGINAL_MODULES = {name: sys.modules.get(name) for name in _STUBBED_MODULE_NAMES}

torch_stub           = types.ModuleType("torch")
torch_stub.device    = lambda name: name
torch_stub.cuda      = types.SimpleNamespace(is_available=lambda: False)
torch_stub.Tensor    = type("Tensor", (), {})
sys.modules["torch"] = torch_stub

ultralytics_stub                      = types.ModuleType("ultralytics")
ultralytics_nn_stub                   = types.ModuleType("ultralytics.nn")
ultralytics_tasks_stub                = types.ModuleType("ultralytics.nn.tasks")
ultralytics_cfg_stub                  = types.ModuleType("ultralytics.cfg")
ultralytics_tasks_stub.DetectionModel = object
ultralytics_cfg_stub.get_cfg          = lambda: None
sys.modules["ultralytics"]            = ultralytics_stub
sys.modules["ultralytics.nn"]         = ultralytics_nn_stub
sys.modules["ultralytics.nn.tasks"]   = ultralytics_tasks_stub
sys.modules["ultralytics.cfg"]        = ultralytics_cfg_stub

seaborn_stub           = types.ModuleType("seaborn")
sys.modules["seaborn"] = seaborn_stub

scipy_stub         = types.ModuleType("scipy")
scipy_ndimage_stub = types.ModuleType("scipy.ndimage")


def zoom_stub(array, factors, order=1):
    """测试所需的二维最近邻缩放，避免加载完整 SciPy。"""
    del order
    new_shape = tuple(
        max(1, int(round(size * factor))) for size, factor in zip(array.shape, factors)
    )
    row = np.minimum(
        (np.arange(new_shape[0]) / factors[0]).astype(int), array.shape[0] - 1
    )
    column = np.minimum(
        (np.arange(new_shape[1]) / factors[1]).astype(int), array.shape[1] - 1
    )
    return array[row][:, column]


scipy_ndimage_stub.zoom      = zoom_stub
sys.modules["scipy"]         = scipy_stub
sys.modules["scipy.ndimage"] = scipy_ndimage_stub

rfi_stub  = types.ModuleType("after.rfi")
RFI_CALLS = []


def cal_rfi_stub(data, noise_mask, **kwargs):
    """让 I/V 返回不同坏通道和坏像素，便于检查并集逻辑。"""
    del kwargs
    call_index             = len(RFI_CALLS)
    channel_index          = 1 if call_index % 2 == 0 else min(2, data.shape[1] - 1)
    channel                = np.zeros(data.shape[1], dtype=bool)
    pixel                  = np.zeros(data.shape, dtype=bool)
    channel[channel_index] = True
    pixel[min(call_index % 2, data.shape[0] - 1), 0 if call_index % 2 == 0 else -1] = (
        True
    )
    RFI_CALLS.append(np.asarray(noise_mask, dtype=bool).copy())
    return channel, pixel


rfi_stub.cal_rfi         = cal_rfi_stub
sys.modules["after.rfi"] = rfi_stub

from after import burst_detect    # noqa: E402
from after.burst_detect import (  # noqa: E402
    detect_one_file,
    filter_inference_boxes,
    prepare_calibration_display,
    prepare_image_tiles,
    write_detection_results,
)

# burst_detect 已把所需对象绑定到自己的模块命名空间。立刻恢复全局模块缓存，
# 避免这些最小桩污染同一 pytest 进程里随后收集的分析/RM 测试。
for _module_name, _original_module in _ORIGINAL_MODULES.items():
    if _original_module is None:
        sys.modules.pop(_module_name, None)
    else:
        sys.modules[_module_name] = _original_module


class FilterInferenceBoxesTest(unittest.TestCase):
    def test_removes_horizontal_box_and_keeps_largest_overlap(self):
        scores = np.array([0.99, 0.4, 0.95, 0.8], dtype=np.float32)
        boxes = np.array(
            [
                [40, 40, 90, 10],
                [100, 100, 40, 80],
                [100, 100, 10, 20],
                [250, 100, 10, 40],
            ],
            dtype=np.float32,
        )

        kept_scores, kept_boxes = filter_inference_boxes(scores, boxes)

        np.testing.assert_allclose(kept_scores, [0.4, 0.8])
        np.testing.assert_allclose(
            kept_boxes,
            [
                [100, 100, 40, 80],
                [250, 100, 10, 40],
            ],
        )

    def test_keeps_boxes_that_only_touch_edges(self):
        scores = np.array([0.7, 0.6], dtype=np.float32)
        boxes = np.array(
            [
                [10, 10, 10, 20],
                [20, 10, 10, 20],
            ],
            dtype=np.float32,
        )

        kept_scores, kept_boxes = filter_inference_boxes(scores, boxes)

        np.testing.assert_allclose(kept_scores, scores)
        np.testing.assert_allclose(kept_boxes, boxes)

    def test_kept_boxes_have_no_positive_area_overlap(self):
        scores = np.array([0.6, 0.9, 0.7, 0.8], dtype=np.float32)
        boxes = np.array(
            [
                [50, 50, 60, 60],
                [55, 50, 20, 20],
                [110, 50, 40, 40],
                [130, 50, 20, 20],
            ],
            dtype=np.float32,
        )

        _, kept_boxes = filter_inference_boxes(scores, boxes)
        half_size     = kept_boxes[:, 2:] / 2
        xyxy = np.column_stack(
            [
                kept_boxes[:, :2] - half_size,
                kept_boxes[:, :2] + half_size,
            ]
        )

        for first in range(len(xyxy)):
            for second in range(first + 1, len(xyxy)):
                overlap_width = min(xyxy[first, 2], xyxy[second, 2]) - max(
                    xyxy[first, 0], xyxy[second, 0]
                )
                overlap_height = min(xyxy[first, 3], xyxy[second, 3]) - max(
                    xyxy[first, 1], xyxy[second, 1]
                )
                self.assertFalse(overlap_width > 0 and overlap_height > 0)


class DetectionDisplayTest(unittest.TestCase):
    def test_finds_same_directory_calibration_jpg(self):
        with tempfile.TemporaryDirectory() as directory:
            h5_path        = Path(directory) / "sample_cal.h5"
            reference_path = Path(directory) / "sample.jpg"
            h5_path.touch()
            reference_path.touch()

            self.assertEqual(
                burst_detect._find_reference_jpg(str(h5_path)),
                str(reference_path),
            )

    def test_missing_reference_jpg_returns_none(self):
        with tempfile.TemporaryDirectory() as directory:
            h5_path = Path(directory) / "sample_cal.h5"
            h5_path.touch()

            self.assertIsNone(burst_detect._find_reference_jpg(str(h5_path)))

    def test_calibration_display_normalizes_and_downsamples_copy(self):
        data = np.array(
            [
                [1, 10, 100, 1000],
                [3, 12, 102, 1002],
                [5, 14, 104, 1004],
                [7, 16, 106, 1006],
            ],
            dtype=np.float32,
        )
        original = data.copy()
        freq     = np.array([1000, 1100, 1200, 1300], dtype=np.float32)

        display, display_freq, display_time_reso = prepare_calibration_display(
            data, freq, time_reso=0.001, time_factor=2, freq_factor=2
        )

        np.testing.assert_array_equal(data, original)
        normalized = original / np.nanmean(original, axis=0, keepdims=True)
        expected = np.nanmean(
            np.nanmean(normalized.reshape(2, 2, 4), axis=1).reshape(2, 2, 2),
            axis=2,
        )
        np.testing.assert_allclose(display, expected)
        np.testing.assert_allclose(display_freq, [1050, 1250])
        self.assertEqual(display_time_reso, 0.002)

    def test_two_panel_matches_calibration_quicklook_layout(self):
        data = np.arange(16, dtype=np.float32).reshape(4, 4)
        freq = np.linspace(1000, 1500, 4)
        fig  = burst_detect.plt.figure(figsize=burst_detect.TWO_PANEL_FIGSIZE)
        try:
            with mock.patch("matplotlib.axes.Axes.imshow") as imshow_mock:
                burst_detect._render_two_panel(fig, data, freq, time_reso=0.001)
            kwargs = imshow_mock.call_args.kwargs
            self.assertNotIn("interpolation", kwargs)
            self.assertNotIn("resample", kwargs)
            self.assertEqual(kwargs["extent"], [0, 4, 1000, 1500])
            self.assertEqual(
                tuple(fig.get_size_inches()), burst_detect.TWO_PANEL_FIGSIZE
            )
        finally:
            burst_detect.plt.close(fig)

    def test_interactive_title_and_labels_stay_inside_canvas(self):
        data        = np.arange(16, dtype=np.float32).reshape(4, 4)
        freq        = np.linspace(1000, 1500, 4)
        figures     = []
        real_figure = burst_detect.plt.figure
        real_close  = burst_detect.plt.close

        def capture_figure(*args, **kwargs):
            fig = real_figure(*args, **kwargs)
            fig.canvas.start_event_loop = mock.Mock()
            figures.append(fig)
            return fig

        with (
            mock.patch.object(burst_detect.plt, "figure", side_effect=capture_figure),
            mock.patch.object(burst_detect.plt, "show"),
            mock.patch.object(burst_detect.plt, "pause"),
            mock.patch.object(burst_detect.plt, "close"),
            mock.patch.object(burst_detect, "_raise_window"),
            mock.patch("matplotlib.axes.Axes.imshow"),
        ):
            result = burst_detect.review_interactive(data, freq, time_reso=0.001)

        self.assertEqual(result, [])
        fig = figures[0]
        try:
            ax_prof, ax_spec = fig.axes
            title            = ax_prof.title
            fig.canvas.draw()

            renderer = fig.canvas.get_renderer()
            for artist in (
                title,
                ax_prof.yaxis.label,
                ax_spec.yaxis.label,
                ax_spec.xaxis.label,
            ):
                bounds = artist.get_window_extent(renderer)
                self.assertGreaterEqual(bounds.x0, 0)
                self.assertGreaterEqual(bounds.y0, 0)
                self.assertLessEqual(bounds.x1, fig.bbox.x1)
                self.assertLessEqual(bounds.y1, fig.bbox.y1)
            self.assertIn("\n", title.get_text())
        finally:
            real_close(fig)

    def test_interactive_shows_reference_jpg_in_same_window(self):
        data        = np.ones((4, 4), dtype=np.float32)
        freq        = np.linspace(1000, 1500, 4)
        figures     = []
        real_figure = burst_detect.plt.figure
        real_close  = burst_detect.plt.close

        def capture_figure(*args, **kwargs):
            fig = real_figure(*args, **kwargs)
            fig.canvas.start_event_loop = mock.Mock()
            figures.append(fig)
            return fig

        def render_left_panels(fig, *_args, subplot_spec=None, **_kwargs):
            """只构造左侧坐标轴，避免测试替身未注册 mako 色表。"""
            grid = (
                fig.add_gridspec(4, 1, hspace=0)
                if subplot_spec is None
                else subplot_spec.subgridspec(4, 1, hspace=0)
            )
            return fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1:, 0])

        reference = np.zeros((20, 20, 3), dtype=np.uint8)
        with (
            mock.patch.object(burst_detect.plt, "figure", side_effect=capture_figure),
            mock.patch.object(burst_detect.plt, "imread", return_value=reference),
            mock.patch.object(burst_detect.plt, "show"),
            mock.patch.object(burst_detect.plt, "pause"),
            mock.patch.object(burst_detect.plt, "close"),
            mock.patch.object(
                burst_detect,
                "_render_two_panel",
                side_effect=render_left_panels,
            ),
            mock.patch.object(burst_detect, "_raise_window"),
        ):
            result = burst_detect.review_interactive(
                data,
                freq,
                time_reso            = 0.001,
                reference_image_path = "sample.jpg",
            )

        self.assertEqual(result, [])
        fig = figures[0]
        try:
            self.assertEqual(len(fig.axes), 3)
            reference_axis = fig.axes[2]
            self.assertEqual(len(reference_axis.images), 1)
            self.assertFalse(reference_axis.axison)
            self.assertIn("sample.jpg", reference_axis.get_title())
            self.assertEqual(
                tuple(fig.get_size_inches()),
                burst_detect.REFERENCE_REVIEW_FIGSIZE,
            )
        finally:
            real_close(fig)


class DetectionRfiTest(unittest.TestCase):
    def setUp(self):
        RFI_CALLS.clear()

    def test_write_results_uses_nonburst_noise_and_unions_i_v_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            path    = Path(directory) / "sample_cal.h5"
            iquv    = np.arange(4 * 6 * 4, dtype=np.float32).reshape(4, 6, 4)
            regions = [{"time_start": 2, "time_end": 4}]
            with h5py.File(path, "w"):
                pass

            channel, _, plot_i = write_detection_results(str(path), iquv, regions)

            expected_noise = np.array([True, True, False, False, True, True])
            self.assertEqual(len(RFI_CALLS), 2)
            np.testing.assert_array_equal(RFI_CALLS[0], expected_noise)
            np.testing.assert_array_equal(RFI_CALLS[1], expected_noise)
            np.testing.assert_array_equal(channel, [False, True, True, False])
            self.assertTrue(np.isnan(plot_i[:, 1:3]).all())
            with h5py.File(path, "r") as h5:
                self.assertEqual(h5.attrs["burst_rfi_noise_sample_count"], 4)
                self.assertTrue(np.all(h5["burst_rfi_mask"][:, 1:3]))

    def test_detect_uses_original_stokes_i_for_model_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path                    = Path(directory) / "sample_cal.h5"
            data                    = np.ones((4, 512, 512), dtype=np.float32)
            calibration_mask        = np.zeros((512, 512), dtype=bool)
            calibration_mask[:, 0]  = True
            calibration_mask[10, 5] = True
            with h5py.File(path, "w") as h5:
                h5.create_dataset("data", data=data)
                h5.create_dataset("freq", data=np.linspace(1000, 1500, 512))
                h5.create_dataset("rfi_mask", data=calibration_mask)
                h5.attrs["time_reso"]      = 0.000393216
                h5.attrs["down_time"]      = 8
                h5.attrs["plot_down_time"] = 8

            with (
                mock.patch.object(
                    burst_detect, "predict_single", return_value=(None, None)
                ),
                mock.patch.object(
                    burst_detect, "prepare_image_tiles", wraps=prepare_image_tiles
                ) as prepare_mock,
            ):
                result = detect_one_file(str(path), object(), mode="auto")

            self.assertFalse(result["has_burst"])
            self.assertEqual(prepare_mock.call_count, 1)
            model_input = prepare_mock.call_args.args[0]
            np.testing.assert_array_equal(model_input, data[0])
            self.assertEqual(prepare_mock.call_args.kwargs["time_factor"], 1)
            with h5py.File(path, "r") as h5:
                self.assertIn("burst_rfi_mask", h5)
                self.assertIn("burst_rfi_channel", h5)
                self.assertEqual(h5.attrs["bursts"], "[]")

    def test_quit_returns_normally_without_marking_current_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path           = Path(directory) / "sample_cal.h5"
            reference_path = Path(directory) / "sample.jpg"
            reference_path.touch()
            with h5py.File(path, "w") as h5:
                h5.create_dataset("data", data=np.ones((4, 512, 512), dtype=np.float32))
                h5.create_dataset("freq", data=np.linspace(1000, 1500, 512))
                h5.create_dataset("rfi_mask", data=np.zeros((512, 512), dtype=bool))
                h5.attrs["time_reso"]      = 0.000393216
                h5.attrs["down_time"]      = 8
                h5.attrs["plot_down_time"] = 16
                h5.attrs["down_freq"]      = 4
                h5.attrs["plot_down_freq"] = 8

            with (
                mock.patch.object(
                    burst_detect, "predict_single", return_value=(None, None)
                ),
                mock.patch.object(
                    burst_detect, "review_interactive", return_value=None
                ) as review_mock,
            ):
                result = detect_one_file(str(path), object(), mode="semi-auto")

            self.assertIsNone(result)
            np.testing.assert_array_equal(
                review_mock.call_args.args[0],
                np.ones((512, 512), dtype=np.float32),
            )
            self.assertEqual(review_mock.call_args.kwargs["time_factor"], 2)
            self.assertEqual(review_mock.call_args.kwargs["freq_factor"], 2)
            self.assertEqual(
                review_mock.call_args.kwargs["reference_image_path"],
                str(reference_path),
            )
            with h5py.File(path, "r") as h5:
                self.assertNotIn("bursts", h5.attrs)
                self.assertNotIn("burst_rfi_mask", h5)


if __name__ == "__main__":
    unittest.main()

# fmt: on
