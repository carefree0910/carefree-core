import torch
import tempfile
import unittest

import numpy as np

from pathlib import Path
from core.toolkit.array import *


class TestArray(unittest.TestCase):
    def test_is_int(self):
        self.assertTrue(is_int(np.int8(1)))
        self.assertTrue(is_int(np.int16(1)))
        self.assertTrue(is_int(np.int32(1)))
        self.assertTrue(is_int(np.int64(1)))
        self.assertTrue(is_int(np.uint8(1)))
        self.assertTrue(is_int(np.uint16(1)))
        self.assertTrue(is_int(np.uint32(1)))
        self.assertTrue(is_int(np.uint64(1)))
        self.assertFalse(is_int(np.float16(1)))
        self.assertFalse(is_int(np.float32(1)))
        self.assertFalse(is_int(np.float64(1)))
        self.assertTrue(is_int(torch.tensor(1, dtype=torch.int8)))
        self.assertTrue(is_int(torch.tensor(1, dtype=torch.int16)))
        self.assertTrue(is_int(torch.tensor(1, dtype=torch.int32)))
        self.assertTrue(is_int(torch.tensor(1, dtype=torch.int64)))
        self.assertTrue(is_int(torch.tensor(1, dtype=torch.uint8)))
        self.assertFalse(is_int(torch.tensor(1, dtype=torch.float16)))
        self.assertFalse(is_int(torch.tensor(1, dtype=torch.float32)))
        self.assertFalse(is_int(torch.tensor(1, dtype=torch.float64)))

    def test_is_float(self):
        self.assertFalse(is_float(np.int8(1)))
        self.assertFalse(is_float(np.int16(1)))
        self.assertFalse(is_float(np.int32(1)))
        self.assertFalse(is_float(np.int64(1)))
        self.assertFalse(is_float(np.uint8(1)))
        self.assertFalse(is_float(np.uint16(1)))
        self.assertFalse(is_float(np.uint32(1)))
        self.assertFalse(is_float(np.uint64(1)))
        self.assertTrue(is_float(np.float16(1)))
        self.assertTrue(is_float(np.float32(1)))
        self.assertTrue(is_float(np.float64(1)))
        self.assertFalse(is_float(torch.tensor(1, dtype=torch.int8)))
        self.assertFalse(is_float(torch.tensor(1, dtype=torch.int16)))
        self.assertFalse(is_float(torch.tensor(1, dtype=torch.int32)))
        self.assertFalse(is_float(torch.tensor(1, dtype=torch.int64)))
        self.assertFalse(is_float(torch.tensor(1, dtype=torch.uint8)))
        self.assertTrue(is_float(torch.tensor(1, dtype=torch.float16)))
        self.assertTrue(is_float(torch.tensor(1, dtype=torch.float32)))
        self.assertTrue(is_float(torch.tensor(1, dtype=torch.float64)))

    def test_is_string(self):
        self.assertFalse(is_string(np.int8(1)))
        self.assertFalse(is_string(np.int16(1)))
        self.assertFalse(is_string(np.int32(1)))
        self.assertFalse(is_string(np.int64(1)))
        self.assertFalse(is_string(np.uint8(1)))
        self.assertFalse(is_string(np.uint16(1)))
        self.assertFalse(is_string(np.uint32(1)))
        self.assertFalse(is_string(np.uint64(1)))
        self.assertFalse(is_string(np.float16(1)))
        self.assertFalse(is_string(np.float32(1)))
        self.assertFalse(is_string(np.float64(1)))
        self.assertTrue(is_string(np.str_("1")))
        self.assertTrue(is_string(np.array(["1"])))
        self.assertFalse(is_string(torch.tensor(1, dtype=torch.int8)))
        self.assertFalse(is_string(torch.tensor(1, dtype=torch.int16)))
        self.assertFalse(is_string(torch.tensor(1, dtype=torch.int32)))
        self.assertFalse(is_string(torch.tensor(1, dtype=torch.int64)))
        self.assertFalse(is_string(torch.tensor(1, dtype=torch.uint8)))
        self.assertFalse(is_string(torch.tensor(1, dtype=torch.float16)))
        self.assertFalse(is_string(torch.tensor(1, dtype=torch.float32)))
        self.assertFalse(is_string(torch.tensor(1, dtype=torch.float64)))

    def test_functions(self):
        array = np.random.randn(3, 5, 7)
        tensor = torch.randn(3, 5, 7)
        sigmoid(array)
        sigmoid(tensor)
        softmax(array)
        softmax(tensor)
        l2_normalize(array)
        l2_normalize(tensor)
        normalize(array)
        normalize(tensor)
        normalize(array, global_norm=False)
        normalize(tensor, global_norm=False)
        _, array_stats = normalize(array, return_stats=True)
        _, tensor_stats = normalize(tensor, return_stats=True)
        an = normalize_from(array, array_stats)
        tn = normalize_from(tensor, tensor_stats)
        np.testing.assert_allclose(array, recover_normalize_from(an, array_stats))
        torch.testing.assert_close(tensor, recover_normalize_from(tn, tensor_stats))
        _, array_stats = normalize(array, global_norm=False, return_stats=True)
        _, tensor_stats = normalize(tensor, global_norm=False, return_stats=True)
        tensor_stats = {k: torch.tensor(v) for k, v in tensor_stats.items()}
        an = normalize_from(array, array_stats)
        tn = normalize_from(tensor, tensor_stats)
        np.testing.assert_allclose(array, recover_normalize_from(an, array_stats))
        torch.testing.assert_close(tensor, recover_normalize_from(tn, tensor_stats))
        min_max_normalize(array)
        min_max_normalize(tensor)
        min_max_normalize(array, global_norm=False)
        min_max_normalize(tensor, global_norm=False)
        _, array_stats = min_max_normalize(array, return_stats=True)
        _, tensor_stats = min_max_normalize(tensor, return_stats=True)
        an = min_max_normalize_from(array, array_stats)
        tn = min_max_normalize_from(tensor, tensor_stats)
        np.testing.assert_allclose(
            array, recover_min_max_normalize_from(an, array_stats)
        )
        torch.testing.assert_close(
            tensor, recover_min_max_normalize_from(tn, tensor_stats)
        )
        _, array_stats = min_max_normalize(array, global_norm=False, return_stats=True)
        _, tensor_stats = min_max_normalize(
            tensor, global_norm=False, return_stats=True
        )
        tensor_stats = {k: torch.tensor(v) for k, v in tensor_stats.items()}
        an = min_max_normalize_from(array, array_stats)
        tn = min_max_normalize_from(tensor, tensor_stats)
        np.testing.assert_allclose(
            array, recover_min_max_normalize_from(an, array_stats)
        )
        torch.testing.assert_close(
            tensor, recover_min_max_normalize_from(tn, tensor_stats)
        )
        quantile_normalize(array)
        quantile_normalize(tensor)
        quantile_normalize(array, global_norm=False)
        quantile_normalize(tensor, global_norm=False)
        _, array_stats = quantile_normalize(array, return_stats=True)
        _, tensor_stats = quantile_normalize(tensor, return_stats=True)
        an = quantile_normalize_from(array, array_stats)
        tn = quantile_normalize_from(tensor, tensor_stats)
        np.testing.assert_allclose(
            array, recover_quantile_normalize_from(an, array_stats)
        )
        torch.testing.assert_close(
            tensor, recover_quantile_normalize_from(tn, tensor_stats)
        )
        _, array_stats = quantile_normalize(array, global_norm=False, return_stats=True)
        _, tensor_stats = quantile_normalize(
            tensor, global_norm=False, return_stats=True
        )
        tensor_stats = {k: torch.tensor(v) for k, v in tensor_stats.items()}
        an = quantile_normalize_from(array, array_stats)
        tn = quantile_normalize_from(tensor, tensor_stats)
        np.testing.assert_allclose(
            array, recover_quantile_normalize_from(an, array_stats)
        )
        torch.testing.assert_close(
            tensor, recover_quantile_normalize_from(tn, tensor_stats)
        )
        clip_normalize(array)
        clip_normalize(tensor)
        clip_normalize(array.astype(np.uint8))
        clip_normalize(tensor.to(torch.uint8))

        array = np.random.randn(17, 3)
        tensor = torch.randn(17, 3)
        with self.assertRaises(ValueError):
            iou(array, array)
        with self.assertRaises(ValueError):
            iou(tensor, tensor)
        iou(array[:, :2], array[:, :2])
        iou(tensor[:, :2], tensor[:, :2])
        iou(array[:, :1], array[:, :1])
        iou(tensor[:, :1], tensor[:, :1])

        array = np.random.randn(3, 5, 7, 11)
        tensor = torch.randn(3, 5, 7, 11)
        make_grid(array)
        make_grid(tensor)

    def test_squeeze(self):
        array = np.arange(5)
        tensor = torch.arange(5)
        np.testing.assert_allclose(squeeze(array[None, None]), array[None])
        torch.testing.assert_close(squeeze(tensor[None, None]), tensor[None])

    def test_to_standard(self) -> None:
        def _check(src: np.dtype, tgt: np.dtype) -> None:
            self.assertEqual(to_standard(np.array([0], src)).dtype, tgt)

        _check(np.float16, np.float32)
        _check(np.float32, np.float32)
        _check(np.float64, np.float32)
        _check(np.int8, np.int64)
        _check(np.int16, np.int64)
        _check(np.int32, np.int64)
        _check(np.int64, np.int64)

    def test_conversion(self):
        array = np.random.randn(3, 5, 7)
        tensor = torch.randn(3, 5, 7)
        self.assertIsInstance(to_torch(array), torch.Tensor)
        self.assertIsInstance(to_numpy(tensor), np.ndarray)

    def test_to_device(self):
        tensors = {
            "a": torch.randn(3, 5, 7),
            "b": [torch.randn(3, 5, 7)],
            "c": {"d": torch.randn(3, 5, 7)},
        }
        to_device(tensors, None)
        to_device(tensors, "cpu")

    def test_corr(self) -> None:
        pred = np.random.randn(100, 5)
        target = np.random.randn(100, 5)
        weights = np.zeros([100, 1])
        weights[:30] = weights[-30:] = 1.0
        corr00 = corr(pred, pred, weights)
        corr01 = corr(pred, target, weights)
        corr02 = corr(target, pred, weights)
        w_pred = pred[list(range(30)) + list(range(70, 100))]
        w_target = target[list(range(30)) + list(range(70, 100))]
        corr10 = corr(w_pred, w_pred)
        corr11 = corr(w_pred, w_target)
        corr12 = corr(w_target, w_pred)
        self.assertTrue(allclose(corr00, corr10))
        self.assertTrue(allclose(corr01, corr11, corr02.T, corr12.T))
        with self.assertRaises(ValueError):
            corr(pred, target[:, :4], get_diagonal=True)

    def test_get_one_hot(self):
        indices = [1, 4, 2, 3]
        self.assertEqual(
            [[0, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]],
            get_one_hot(indices, 5).tolist(),
        )

    def test_get_indices_from_another(self):
        def _get(is_sorted: bool) -> np.ndarray:
            indices = get_indices_from_another(base, segment, already_sorted=is_sorted)
            return base[np.minimum(indices, len(base) - 1)]

        base, segment = np.random.permutation(100), np.random.permutation(100)[:10]
        self.assertTrue(np.allclose(_get(False), segment))
        self.assertFalse(np.allclose(_get(True), segment))
        base.sort()
        self.assertTrue(np.allclose(_get(True), segment))

    def test_get_unique_indices(self):
        arr = np.array([1, 2, 3, 2, 4, 1, 0, 1], np.int64)
        res = get_unique_indices(arr)
        self.assertTrue(np.allclose(res.unique, np.array([0, 1, 2, 3, 4])))
        self.assertTrue(np.allclose(res.unique_cnt, np.array([1, 3, 2, 1, 1])))
        gt = np.array([6, 0, 5, 7, 1, 3, 2, 4])
        self.assertTrue(np.allclose(res.sorting_indices, gt))
        self.assertTrue(np.allclose(res.split_arr, np.array([1, 4, 6, 7])))
        gt_indices_list = list(map(np.array, [[6], [0, 5, 7], [1, 3], [2], [4]]))
        for rs_indices, gt_indices in zip(res.split_indices, gt_indices_list):
            self.assertTrue(np.allclose(rs_indices, gt_indices))

    def test_counter_from_arr(self):
        arr = np.array([1, 2, 3, 2, 4, 1, 0, 1])
        counter = get_counter_from_arr(arr)
        self.assertTrue(counter[0], 1)
        self.assertTrue(counter[1], 3)
        self.assertTrue(counter[2], 2)
        self.assertTrue(counter[3], 1)
        self.assertTrue(counter[4], 1)

    def test_allclose(self):
        arr = np.random.random(1000)
        self.assertTrue(allclose(*(arr for _ in range(10))))
        self.assertFalse(allclose(*[arr for _ in range(9)] + [arr + 1e-6]))

    def test_stride_array(self):
        arr = StrideArray(np.arange(9).reshape([3, 3]))
        self.assertEqual(str(arr), str(arr.arr))
        self.assertEqual(repr(arr), repr(arr.arr))
        with self.assertRaises(ValueError):
            arr.roll(4, axis=0)
        with self.assertRaises(ValueError):
            arr.patch(4, patch_h=2)
        with self.assertRaises(ValueError):
            arr.patch(2, patch_h=4)
        with self.assertRaises(ValueError):
            StrideArray(np.arange(9)).patch(2)
        self.assertTrue(
            np.allclose(
                arr.roll(2, axis=0),
                np.array([[[0, 1, 2], [3, 4, 5]], [[3, 4, 5], [6, 7, 8]]]),
            )
        )
        self.assertTrue(
            np.allclose(
                arr.roll(2, axis=-2),
                np.array([[[0, 1, 2], [3, 4, 5]], [[3, 4, 5], [6, 7, 8]]]),
            )
        )
        self.assertTrue(
            np.allclose(
                arr.roll(2, axis=1),
                np.array([[[0, 1], [1, 2]], [[3, 4], [4, 5]], [[6, 7], [7, 8]]]),
            )
        )
        patch_gt = np.array(
            [
                [
                    [
                        [0, 1],
                        [3, 4],
                    ],
                    [
                        [1, 2],
                        [4, 5],
                    ],
                ],
                [
                    [
                        [3, 4],
                        [6, 7],
                    ],
                    [
                        [4, 5],
                        [7, 8],
                    ],
                ],
            ]
        )
        self.assertTrue(np.allclose(arr.patch(2), patch_gt))
        arr = StrideArray(np.arange(16).reshape([4, 4]))
        self.assertTrue(
            np.allclose(
                arr.roll(2, axis=0, stride=2),
                np.array(
                    [[[0, 1, 2, 3], [4, 5, 6, 7]], [[8, 9, 10, 11], [12, 13, 14, 15]]]
                ),
            )
        )
        self.assertTrue(
            np.allclose(
                arr.roll(2, axis=1, stride=2),
                np.array(
                    [
                        [[0, 1], [2, 3]],
                        [[4, 5], [6, 7]],
                        [[8, 9], [10, 11]],
                        [[12, 13], [14, 15]],
                    ]
                ),
            )
        )
        self.assertTrue(
            np.allclose(
                arr.patch(2, h_stride=2, w_stride=2),
                np.array(
                    [
                        [[[0, 1], [4, 5]], [[2, 3], [6, 7]]],
                        [[[8, 9], [12, 13]], [[10, 11], [14, 15]]],
                    ]
                ),
            )
        )
        self.assertTrue(
            np.allclose(
                arr.patch(2, h_stride=1, w_stride=2),
                np.array(
                    [
                        [[[0, 1], [4, 5]], [[2, 3], [6, 7]]],
                        [[[4, 5], [8, 9]], [[6, 7], [10, 11]]],
                        [[[8, 9], [12, 13]], [[10, 11], [14, 15]]],
                    ]
                ),
            )
        )
        self.assertTrue(
            np.allclose(
                arr.patch(2, h_stride=2, w_stride=1),
                np.array(
                    [
                        [[[0, 1], [4, 5]], [[1, 2], [5, 6]], [[2, 3], [6, 7]]],
                        [[[8, 9], [12, 13]], [[9, 10], [13, 14]], [[10, 11], [14, 15]]],
                    ]
                ),
            )
        )
        arr = StrideArray(np.arange(9).reshape([3, 3, 1, 1]))
        with self.assertRaises(ValueError):
            arr.repeat(2, axis=0)
        self.assertTrue(np.allclose(arr.patch(2, h_axis=0)[..., 0, 0], patch_gt))
        repeat_gt = np.array(
            [
                [[0, 0], [1, 1], [2, 2]],
                [[3, 3], [4, 4], [5, 5]],
                [[6, 6], [7, 7], [8, 8]],
            ]
        )
        self.assertTrue(np.allclose(arr.repeat(2)[:, :, 0], repeat_gt))
        self.assertTrue(np.allclose(arr.repeat(2, axis=-2)[..., 0], repeat_gt))

    def test_shared_array(self):
        array = SharedArray.from_data(np.random.randn(3, 5, 7))
        array.destroy()

    def test_to_labels(self):
        logits = np.random.randn(17, 2)
        diff = logits[:, [1]] - logits[:, [0]]
        np.testing.assert_allclose(to_labels(logits, 0.123), to_labels(diff, 0.123))
        np.testing.assert_allclose(to_labels(logits), to_labels(diff))
        logits = np.random.randn(17, 7)
        np.testing.assert_allclose(to_labels(logits), logits.argmax(1)[..., None])

    def test_get_full_logits(self):
        logits = np.random.randn(3, 5, 7)
        np.testing.assert_allclose(get_full_logits(logits), logits)
        logits = np.random.randn(3, 5, 1)
        full_logits = get_full_logits(logits)
        np.testing.assert_allclose(logits, full_logits[..., [1]])
        np.testing.assert_allclose(-logits, full_logits[..., [0]])


class TestNpSafeSerializer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp_dir.name)
        self.data = np.array([1, 2, 3, 4, 5])

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save(self):
        NpSafeSerializer.save(self.folder, self.data)
        self.assertTrue((self.folder / NpSafeSerializer.array_file).exists())
        self.assertTrue((self.folder / NpSafeSerializer.size_file).exists())

    def test_load(self):
        NpSafeSerializer.save(self.folder, self.data)
        loaded_data = NpSafeSerializer.load(self.folder)
        np.testing.assert_array_equal(loaded_data, self.data)

    def test_try_load(self):
        NpSafeSerializer.save(self.folder, self.data)
        np.testing.assert_array_equal(NpSafeSerializer.try_load(self.folder), self.data)
        self.assertIsNone(NpSafeSerializer.try_load(self.folder / "invalid"))
        np.save(self.folder / NpSafeSerializer.array_file, self.data[..., :2])
        self.assertIsNone(NpSafeSerializer.try_load(self.folder))
        NpSafeSerializer.save(self.folder, self.data)
        np.testing.assert_array_equal(NpSafeSerializer.try_load(self.folder), self.data)
        (self.folder / NpSafeSerializer.size_file).unlink()
        self.assertIsNone(NpSafeSerializer.try_load(self.folder))

    def test_try_load_no_load(self):
        NpSafeSerializer.save(self.folder, self.data)
        loaded_data = NpSafeSerializer.try_load(self.folder, no_load=True)
        np.testing.assert_array_equal(loaded_data, np.zeros(0))

    def test_try_load_invalid_size(self):
        NpSafeSerializer.save(self.folder, self.data)
        with open(self.folder / NpSafeSerializer.size_file, "w") as f:
            f.write("invalid")
        loaded_data = NpSafeSerializer.try_load(self.folder)
        self.assertIsNone(loaded_data)

    def test_load_with(self):
        def init_fn():
            return np.array([6, 7, 8, 9, 10])

        NpSafeSerializer.cleanup(self.folder)
        loaded_data = NpSafeSerializer.load_with(self.folder, init_fn)
        np.testing.assert_array_equal(loaded_data, init_fn())
        self.assertTrue((self.folder / NpSafeSerializer.array_file).exists())
        self.assertTrue((self.folder / NpSafeSerializer.size_file).exists())


if __name__ == "__main__":
    unittest.main()
