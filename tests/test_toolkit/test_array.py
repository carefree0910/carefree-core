import torch
import pytest
import unittest

import numpy as np

from core.toolkit.array import *
from core.toolkit.array import _postprocess_logits
from pathlib import Path


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

    def test_is_real_numeric(self):
        self.assertTrue(is_real_numeric(np.int64(1)))
        self.assertTrue(is_real_numeric(np.float64(1)))
        self.assertFalse(is_real_numeric(torch.tensor(1.0j)))

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
            "e": "unchanged",
        }
        to_device(tensors, None)
        converted = to_device(tensors, "cpu")
        self.assertEqual(converted["e"], "unchanged")

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
        np.testing.assert_allclose(corr(pred, pred, get_diagonal=True), np.ones(5))
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

    def test_postprocess_logits_binary(self):
        array = np.array([-1.0, 0.0, 1.0], np.float32)
        probabilities = _postprocess_logits(array, return_probabilities=True)
        self.assertEqual(probabilities.shape, (3, 2))
        self.assertEqual(probabilities.dtype, array.dtype)
        np.testing.assert_allclose(probabilities.sum(1), np.ones(3))
        np.testing.assert_array_equal(to_labels(array), np.array([0, 0, 1]))
        np.testing.assert_array_equal(
            _postprocess_logits(array, inclusive=True),
            np.array([0, 1, 1]),
        )
        np.testing.assert_array_equal(
            to_labels(array, class_dim=-1),
            np.array([0, 0, 1]),
        )

        column = array[:, None]
        probabilities = _postprocess_logits(column, return_probabilities=True)
        self.assertEqual(probabilities.shape, (3, 2))
        np.testing.assert_array_equal(to_labels(column).ravel(), np.array([0, 0, 1]))

        logits = np.array([[1.0, 2.0], [2.0, 1.0]], np.float64)
        probabilities = _postprocess_logits(logits, return_probabilities=True)
        self.assertEqual(probabilities.dtype, logits.dtype)
        np.testing.assert_allclose(probabilities.sum(1), np.ones(2))
        np.testing.assert_array_equal(to_labels(logits).ravel(), np.array([1, 0]))

        tensor = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
        probabilities = _postprocess_logits(tensor, return_probabilities=True)
        self.assertEqual(probabilities.shape, (3, 2))
        self.assertEqual(probabilities.dtype, tensor.dtype)
        self.assertEqual(probabilities.device, tensor.device)
        torch.testing.assert_close(
            probabilities.sum(1), torch.ones(3, dtype=tensor.dtype)
        )
        labels = to_labels(tensor)
        self.assertEqual(labels.dtype, torch.int64)
        self.assertEqual(labels.device, tensor.device)
        torch.testing.assert_close(labels, torch.tensor([0, 0, 1]))

        tensor_column = tensor[:, None]
        probabilities = _postprocess_logits(
            tensor_column,
            return_probabilities=True,
        )
        self.assertEqual(probabilities.shape, (3, 2))
        torch.testing.assert_close(to_labels(tensor_column).ravel(), labels)

        tensor_logits = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
        probabilities = _postprocess_logits(
            tensor_logits,
            return_probabilities=True,
        )
        torch.testing.assert_close(probabilities.sum(1), torch.ones(2))
        torch.testing.assert_close(
            to_labels(tensor_logits).ravel(),
            torch.tensor([1, 0]),
        )

    def test_to_labels_numerical_bc(self):
        one_logit = np.array([[1.0e-8]], np.float32)
        two_logits = np.array([[0.0, 1.0e-8]], np.float32)
        np.testing.assert_array_equal(to_labels(one_logit), np.ones((1, 1)))
        np.testing.assert_array_equal(to_labels(two_logits), np.ones((1, 1)))

        one_logit = np.array([[2.2]], np.float16)
        two_logits = np.array([[0.0, 2.2]], np.float16)
        np.testing.assert_array_equal(to_labels(one_logit, 0.9), np.ones((1, 1)))
        np.testing.assert_array_equal(to_labels(two_logits, 0.9), np.ones((1, 1)))

        finite_min = -np.finfo(np.float16).max
        finite_max = np.finfo(np.float16).max
        one_logit = np.array([[finite_min], [finite_max]], np.float16)
        two_logits = np.array([[0.0, finite_min], [0.0, finite_max]], np.float16)
        np.testing.assert_array_equal(to_labels(one_logit, 0.0), np.ones((2, 1)))
        np.testing.assert_array_equal(to_labels(two_logits, 0.0), np.ones((2, 1)))
        np.testing.assert_array_equal(to_labels(one_logit, 1.0), np.zeros((2, 1)))
        np.testing.assert_array_equal(to_labels(two_logits, 1.0), np.zeros((2, 1)))

    def test_postprocess_logits_modes_and_axes(self):
        nchw = np.arange(2 * 3 * 2 * 2, dtype=np.float32).reshape(2, 3, 2, 2)
        probabilities = _postprocess_logits(nchw, return_probabilities=True)
        self.assertEqual(probabilities.shape, nchw.shape)
        np.testing.assert_allclose(probabilities.sum(1), np.ones((2, 2, 2)))
        labels = to_labels(nchw)
        self.assertEqual(labels.shape, (2, 1, 2, 2))
        np.testing.assert_array_equal(labels, nchw.argmax(1, keepdims=True))

        nhwc = np.moveaxis(nchw, 1, -1)
        probabilities = _postprocess_logits(
            nhwc,
            class_dim=-1,
            return_probabilities=True,
        )
        self.assertEqual(probabilities.shape, nhwc.shape)
        np.testing.assert_allclose(probabilities.sum(-1), np.ones((2, 2, 2)))
        labels = to_labels(nhwc, class_dim=-1)
        self.assertEqual(labels.shape, (2, 2, 2, 1))
        np.testing.assert_array_equal(labels, nhwc.argmax(-1, keepdims=True))

        single_nchw = nchw[:, :1]
        single_nhwc = np.moveaxis(single_nchw, 1, -1)
        for class_dim, single_logits in [(1, single_nchw), (-1, single_nhwc)]:
            with self.subTest(num_classes=1, class_dim=class_dim):
                positive = sigmoid(single_logits)
                expected_probabilities = np.concatenate(
                    [1.0 - positive, positive],
                    axis=class_dim,
                )
                probabilities = _postprocess_logits(
                    single_logits,
                    class_dim=class_dim,
                    return_probabilities=True,
                )
                np.testing.assert_allclose(probabilities, expected_probabilities)
                expected_labels = (single_logits > 0.0).astype(int)
                np.testing.assert_array_equal(
                    to_labels(single_logits, class_dim=class_dim),
                    expected_labels,
                )
                tensor_logits = torch.from_numpy(single_logits)
                tensor_probabilities = _postprocess_logits(
                    tensor_logits,
                    class_dim=class_dim,
                    return_probabilities=True,
                )
                torch.testing.assert_close(
                    tensor_probabilities,
                    torch.from_numpy(expected_probabilities),
                )
                torch.testing.assert_close(
                    to_labels(tensor_logits, class_dim=class_dim),
                    torch.from_numpy(expected_labels).long(),
                )

        two_classes = nchw[:, :2]
        two_classes_nhwc = np.moveaxis(two_classes, 1, -1)
        for class_dim, binary_logits in [(1, two_classes), (-1, two_classes_nhwc)]:
            with self.subTest(num_classes=2, class_dim=class_dim):
                positive = np.take(binary_logits, [1], axis=class_dim)
                negative = np.take(binary_logits, [0], axis=class_dim)
                expected_labels = (positive - negative > 0.0).astype(int)
                np.testing.assert_array_equal(
                    to_labels(binary_logits, class_dim=class_dim),
                    expected_labels,
                )
                torch.testing.assert_close(
                    to_labels(
                        torch.from_numpy(binary_logits),
                        class_dim=class_dim,
                    ),
                    torch.from_numpy(expected_labels).long(),
                )
        labels = to_labels(two_classes, prediction_mode="multiclass")
        np.testing.assert_array_equal(labels, two_classes.argmax(1, keepdims=True))

        multilabel = np.array([[-1.0, 0.0, 1.0]], np.float32)
        probabilities = _postprocess_logits(
            multilabel,
            prediction_mode="multilabel",
            return_probabilities=True,
        )
        self.assertEqual(probabilities.shape, multilabel.shape)
        np.testing.assert_array_equal(
            to_labels(multilabel, prediction_mode="multilabel"),
            np.array([[0, 0, 1]]),
        )
        rank_one = _postprocess_logits(
            multilabel.ravel(),
            prediction_mode="multilabel",
            return_probabilities=True,
        )
        self.assertEqual(rank_one.shape, (3,))

        tensor = torch.from_numpy(nchw)
        probabilities = _postprocess_logits(tensor, return_probabilities=True)
        torch.testing.assert_close(
            probabilities.sum(1),
            torch.ones((2, 2, 2)),
        )
        labels = to_labels(tensor)
        torch.testing.assert_close(labels, tensor.argmax(1, keepdim=True))
        tensor_multilabel = torch.from_numpy(multilabel)
        probabilities = _postprocess_logits(
            tensor_multilabel,
            prediction_mode="multilabel",
            return_probabilities=True,
        )
        torch.testing.assert_close(probabilities, torch.sigmoid(tensor_multilabel))
        torch.testing.assert_close(
            to_labels(tensor_multilabel, prediction_mode="multilabel"),
            torch.tensor([[0, 0, 1]]),
        )

    def test_postprocess_logits_validation(self):
        logits = np.zeros((2, 1), np.float32)
        with self.assertRaisesRegex(ValueError, "prediction mode"):
            _postprocess_logits(logits, prediction_mode="invalid")
        with self.assertRaisesRegex(ValueError, "at least one dimension"):
            _postprocess_logits(np.array(0.0))
        with self.assertRaisesRegex(ValueError, "rank-1"):
            _postprocess_logits(np.zeros(2), class_dim=2)
        with self.assertRaisesRegex(ValueError, "out of range"):
            _postprocess_logits(logits, class_dim=2)
        with self.assertRaisesRegex(ValueError, "batch dimension"):
            _postprocess_logits(logits, class_dim=0)
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            _postprocess_logits(np.empty((2, 0)))
        with self.assertRaisesRegex(ValueError, "one or two classes"):
            _postprocess_logits(np.zeros((2, 3)), prediction_mode="binary")
        with self.assertRaisesRegex(ValueError, "at least 2 classes"):
            _postprocess_logits(np.zeros(2), prediction_mode="multiclass")
        with self.assertRaisesRegex(ValueError, "at least 2 classes"):
            _postprocess_logits(logits, prediction_mode="multiclass")

        for threshold in [
            float("nan"),
            float("inf"),
            -float("inf"),
            -0.1,
            1.1,
        ]:
            with self.subTest(threshold=threshold):
                with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
                    to_labels(logits, threshold)
        np.testing.assert_array_equal(to_labels(logits, 0.0), np.ones((2, 1)))
        np.testing.assert_array_equal(to_labels(logits, 1.0), np.zeros((2, 1)))
        _postprocess_logits(
            logits,
            threshold=float("nan"),
            return_probabilities=True,
        )
        to_labels(
            np.zeros((2, 3)),
            float("nan"),
            prediction_mode="multiclass",
        )

    def test_get_full_logits(self):
        logits = np.random.randn(3, 5, 7)
        self.assertIs(get_full_logits(logits), logits)
        np.testing.assert_allclose(get_full_logits(logits), logits)
        logits = np.random.randn(3, 5, 1)
        full_logits = get_full_logits(logits)
        np.testing.assert_allclose(logits, full_logits[..., [1]])
        np.testing.assert_allclose(-logits, full_logits[..., [0]])

    def test_array_from_pointer(self):
        source = np.arange(6, dtype=np.int64).reshape(2, 3)
        typestr = BaseType.INT.to_typestr(source.dtype.itemsize)
        loaded = arr_from_ptr(source.ctypes.data, typestr, list(source.shape))
        np.testing.assert_array_equal(loaded, source)
        self.assertFalse(loaded.flags.writeable)


class TestNpSafeSerializer(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _prepare_folder(self, tmp_path: Path) -> None:
        self.folder = tmp_path
        self.data = np.array([1, 2, 3, 4, 5], dtype="S7")
        self.rawd = dict(dtype=self.data.dtype)

    def test_save(self):
        NpSafeSerializer.save(self.folder, self.data)
        self.assertTrue((self.folder / NpSafeSerializer.array_file).exists())
        self.assertTrue((self.folder / NpSafeSerializer.size_file).exists())
        NpSafeSerializer.cleanup(self.folder)
        NpSafeSerializer.save(self.folder, self.data, to_raw=True)
        self.assertTrue((self.folder / NpSafeSerializer.raw_array_file).exists())
        self.assertTrue((self.folder / NpSafeSerializer.size_file).exists())

    def test_load(self):
        NpSafeSerializer.save(self.folder, self.data)
        loaded_data = NpSafeSerializer.load(self.folder)
        np.testing.assert_array_equal(loaded_data, self.data)
        NpSafeSerializer.cleanup(self.folder)
        NpSafeSerializer.save(self.folder, self.data, to_raw=True)
        loaded_data = NpSafeSerializer.load_raw(self.folder, **self.rawd)
        np.testing.assert_array_equal(loaded_data, self.data)
        rawd = dict(**self.rawd, shape=self.data.shape)
        loaded_data = NpSafeSerializer.load_raw(self.folder, **rawd)
        np.testing.assert_array_equal(loaded_data, self.data)
        loaded_data = NpSafeSerializer.load_raw(self.folder, **rawd, mmap_mode="r")
        try:
            np.testing.assert_array_equal(loaded_data, self.data)
        finally:
            loaded_data._mmap.close()

    def test_try_load(self):
        NpSafeSerializer.save(self.folder, self.data)
        np.testing.assert_array_equal(NpSafeSerializer.try_load(self.folder), self.data)
        NpSafeSerializer.cleanup(self.folder)
        NpSafeSerializer.save(self.folder, self.data, to_raw=True)
        loaded_data = NpSafeSerializer.try_load(self.folder, **self.rawd, from_raw=True)
        np.testing.assert_array_equal(loaded_data, self.data)
        self.assertIsNone(NpSafeSerializer.try_load(self.folder / "invalid"))
        np.save(self.folder / NpSafeSerializer.array_file, self.data[..., :2])
        self.assertIsNone(NpSafeSerializer.try_load(self.folder))
        NpSafeSerializer.save(self.folder, self.data)
        np.testing.assert_array_equal(NpSafeSerializer.try_load(self.folder), self.data)
        (self.folder / NpSafeSerializer.size_file).unlink()
        self.assertIsNone(NpSafeSerializer.try_load(self.folder))

    def test_try_load_raw_validation(self):
        NpSafeSerializer.save(self.folder, self.data, to_raw=True)
        with self.assertRaisesRegex(ValueError, "kwargs"):
            NpSafeSerializer.try_load(
                self.folder,
                dtype=self.data.dtype,
                from_raw=True,
                allow_pickle=False,
            )
        with self.assertRaisesRegex(ValueError, "dtype"):
            NpSafeSerializer.try_load(self.folder, from_raw=True)

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
        def init():
            return np.array([6, 7, 8, 9, 10], dtype=self.data.dtype)

        NpSafeSerializer.cleanup(self.folder)
        arr = NpSafeSerializer.load_with(self.folder, init)
        np.testing.assert_array_equal(arr, init())
        NpSafeSerializer.cleanup(self.folder)
        arr = NpSafeSerializer.load_with(self.folder, init, **self.rawd, use_raw=True)
        np.testing.assert_array_equal(arr, init())
        self.assertTrue((self.folder / NpSafeSerializer.raw_array_file).exists())
        self.assertTrue((self.folder / NpSafeSerializer.size_file).exists())


if __name__ == "__main__":
    unittest.main()
