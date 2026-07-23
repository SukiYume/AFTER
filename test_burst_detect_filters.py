import unittest
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

# 这里测试检测阶段的 RFI 数据流。用最小模块桩隔离未参与测试、且在
# Windows 上加载较慢的 Torch/Ultralytics GPU 栈。
_STUBBED_MODULE_NAMES = (
    'torch', 'torchvision', 'torchvision.ops',
    'ultralytics', 'ultralytics.nn', 'ultralytics.nn.tasks',
    'ultralytics.cfg', 'seaborn', 'scipy', 'scipy.ndimage', 'rfi_utils',
)
_ORIGINAL_MODULES = {
    name: sys.modules.get(name) for name in _STUBBED_MODULE_NAMES
}

torch_stub = types.ModuleType('torch')
torch_stub.device = lambda name: name
torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
torch_stub.Tensor = type('Tensor', (), {})
sys.modules['torch'] = torch_stub

torchvision_stub = types.ModuleType('torchvision')
torchvision_ops_stub = types.ModuleType('torchvision.ops')
torchvision_ops_stub.nms = lambda *args, **kwargs: None
torchvision_stub.ops = torchvision_ops_stub
sys.modules['torchvision'] = torchvision_stub
sys.modules['torchvision.ops'] = torchvision_ops_stub

ultralytics_stub = types.ModuleType('ultralytics')
ultralytics_nn_stub = types.ModuleType('ultralytics.nn')
ultralytics_tasks_stub = types.ModuleType('ultralytics.nn.tasks')
ultralytics_cfg_stub = types.ModuleType('ultralytics.cfg')
ultralytics_tasks_stub.DetectionModel = object
ultralytics_cfg_stub.get_cfg = lambda: None
sys.modules['ultralytics'] = ultralytics_stub
sys.modules['ultralytics.nn'] = ultralytics_nn_stub
sys.modules['ultralytics.nn.tasks'] = ultralytics_tasks_stub
sys.modules['ultralytics.cfg'] = ultralytics_cfg_stub

seaborn_stub = types.ModuleType('seaborn')
sys.modules['seaborn'] = seaborn_stub

scipy_stub = types.ModuleType('scipy')
scipy_ndimage_stub = types.ModuleType('scipy.ndimage')


def zoom_stub(array, factors, order=1):
    """测试所需的二维最近邻缩放，避免加载完整 SciPy。"""
    del order
    new_shape = tuple(max(1, int(round(size * factor)))
                      for size, factor in zip(array.shape, factors))
    row = np.minimum(
        (np.arange(new_shape[0]) / factors[0]).astype(int),
        array.shape[0] - 1)
    column = np.minimum(
        (np.arange(new_shape[1]) / factors[1]).astype(int),
        array.shape[1] - 1)
    return array[row][:, column]


scipy_ndimage_stub.zoom = zoom_stub
sys.modules['scipy'] = scipy_stub
sys.modules['scipy.ndimage'] = scipy_ndimage_stub

rfi_utils_stub = types.ModuleType('rfi_utils')
RFI_CALLS = []


def cal_rfi_stub(data, noise_mask, **kwargs):
    """让 I/V 返回不同坏通道和坏像素，便于检查并集逻辑。"""
    del kwargs
    call_index = len(RFI_CALLS)
    channel_index = 1 if call_index % 2 == 0 else min(2, data.shape[1] - 1)
    channel = np.zeros(data.shape[1], dtype=bool)
    pixel = np.zeros(data.shape, dtype=bool)
    channel[channel_index] = True
    pixel[min(call_index % 2, data.shape[0] - 1),
          0 if call_index % 2 == 0 else -1] = True
    RFI_CALLS.append(np.asarray(noise_mask, dtype=bool).copy())
    return channel, pixel


rfi_utils_stub.cal_rfi = cal_rfi_stub
sys.modules['rfi_utils'] = rfi_utils_stub

import burst_detect  # noqa: E402
from burst_detect import (  # noqa: E402
    detect_one_file,
    filter_inference_boxes,
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
        boxes = np.array([
            [40, 40, 90, 10],
            [100, 100, 40, 80],
            [100, 100, 10, 20],
            [250, 100, 10, 40],
        ], dtype=np.float32)

        kept_scores, kept_boxes = filter_inference_boxes(scores, boxes)

        np.testing.assert_allclose(kept_scores, [0.4, 0.8])
        np.testing.assert_allclose(kept_boxes, [
            [100, 100, 40, 80],
            [250, 100, 10, 40],
        ])

    def test_keeps_boxes_that_only_touch_edges(self):
        scores = np.array([0.7, 0.6], dtype=np.float32)
        boxes = np.array([
            [10, 10, 10, 20],
            [20, 10, 10, 20],
        ], dtype=np.float32)

        kept_scores, kept_boxes = filter_inference_boxes(scores, boxes)

        np.testing.assert_allclose(kept_scores, scores)
        np.testing.assert_allclose(kept_boxes, boxes)


class DetectionRfiTest(unittest.TestCase):

    def setUp(self):
        RFI_CALLS.clear()

    def test_write_results_uses_nonburst_noise_and_unions_i_v_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample_cal.h5'
            iquv = np.arange(4 * 6 * 4, dtype=np.float32).reshape(4, 6, 4)
            regions = [{'time_start': 2, 'time_end': 4}]
            with h5py.File(path, 'w'):
                pass

            channel, _, plot_i = write_detection_results(
                str(path), iquv, regions)

            expected_noise = np.array([True, True, False, False, True, True])
            self.assertEqual(len(RFI_CALLS), 2)
            np.testing.assert_array_equal(RFI_CALLS[0], expected_noise)
            np.testing.assert_array_equal(RFI_CALLS[1], expected_noise)
            np.testing.assert_array_equal(channel, [False, True, True, False])
            self.assertTrue(np.isnan(plot_i[:, 1:3]).all())
            with h5py.File(path, 'r') as h5:
                self.assertEqual(h5.attrs['burst_rfi_noise_sample_count'], 4)
                self.assertTrue(np.all(h5['burst_rfi_mask'][:, 1:3]))

    def test_detect_uses_original_stokes_i_for_model_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample_cal.h5'
            data = np.ones((4, 512, 512), dtype=np.float32)
            calibration_mask = np.zeros((512, 512), dtype=bool)
            calibration_mask[:, 0] = True
            calibration_mask[10, 5] = True
            with h5py.File(path, 'w') as h5:
                h5.create_dataset('data', data=data)
                h5.create_dataset('freq', data=np.linspace(1000, 1500, 512))
                h5.create_dataset('rfi_mask', data=calibration_mask)
                h5.attrs['time_reso'] = 0.000393216
                h5.attrs['down_time'] = 8
                h5.attrs['plot_down_time'] = 8

            with mock.patch.object(
                    burst_detect, 'predict_single', return_value=(None, None)), \
                    mock.patch.object(
                        burst_detect, 'prepare_image_tiles',
                        wraps=prepare_image_tiles) as prepare_mock:
                result = detect_one_file(str(path), object(), mode='auto')

            self.assertFalse(result['has_burst'])
            self.assertEqual(prepare_mock.call_count, 1)
            model_input = prepare_mock.call_args.args[0]
            np.testing.assert_array_equal(model_input, data[0])
            self.assertEqual(prepare_mock.call_args.kwargs['time_factor'], 1)
            with h5py.File(path, 'r') as h5:
                self.assertIn('burst_rfi_mask', h5)
                self.assertIn('burst_rfi_channel', h5)
                self.assertEqual(h5.attrs['bursts'], '[]')

    def test_quit_returns_normally_without_marking_current_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample_cal.h5'
            with h5py.File(path, 'w') as h5:
                h5.create_dataset(
                    'data', data=np.ones((4, 512, 512), dtype=np.float32))
                h5.create_dataset('freq', data=np.linspace(1000, 1500, 512))
                h5.create_dataset(
                    'rfi_mask', data=np.zeros((512, 512), dtype=bool))
                h5.attrs['time_reso'] = 0.000393216
                h5.attrs['down_time'] = 8
                h5.attrs['plot_down_time'] = 8

            with mock.patch.object(
                    burst_detect, 'predict_single', return_value=(None, None)), \
                    mock.patch.object(
                        burst_detect, 'review_interactive', return_value=None):
                result = detect_one_file(
                    str(path), object(), mode='semi-auto')

            self.assertIsNone(result)
            with h5py.File(path, 'r') as h5:
                self.assertNotIn('bursts', h5.attrs)
                self.assertNotIn('burst_rfi_mask', h5)


if __name__ == '__main__':
    unittest.main()
