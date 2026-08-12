import os
import torch
import tempfile
import unittest

import numpy as np

from PIL import Image
from core.toolkit.cv import *


class TestCV(unittest.TestCase):
    @staticmethod
    def _get_rgba_image():
        y, x = np.indices((5, 7))
        array = np.stack(
            [
                (31 * x + 7 * y) % 256,
                (13 * x + 41 * y) % 256,
                (53 * x + 17 * y) % 256,
                (67 * x + 29 * y) % 256,
            ],
            axis=-1,
        ).astype(np.uint8)
        return Image.fromarray(array)

    def test_to_rgb(self):
        for src_mode in ["CMYK", "RGBA", "RGB", "L"]:
            self.assertEqual(to_rgb(Image.new(src_mode, [123, 321])).mode, "RGB")

    def test_uint8(self):
        self.assertEqual(to_uint8(np.array([0.0, 0.5, 1.0])).dtype, np.uint8)
        self.assertEqual(to_uint8(torch.tensor([0.0, 0.5, 1.0])).dtype, torch.uint8)

    def test_to_alpha_channel(self):
        for src_mode in ["CMYK", "RGBA", "RGB", "L"]:
            self.assertEqual(
                to_alpha_channel(Image.new(src_mode, [123, 321])).mode, "L"
            )

    def test_np_to_bytes(self):
        self.assertIsInstance(np_to_bytes(np.array([0.0, 2.0])), bytes)

    def test_restrict_wh(self):
        self.assertEqual(restrict_wh(100, 200, 300), (100, 200))
        self.assertEqual(restrict_wh(300, 200, 100), (100, 67))
        self.assertEqual(restrict_wh(200, 300, 100), (67, 100))

    def test_get_suitalbe_size(self):
        self.assertEqual(get_suitable_size(100, 64), 128)
        self.assertEqual(get_suitable_size(90, 64), 64)
        self.assertEqual(get_suitable_size(23, 64), 64)

    def test_read_image_response_abi(self):
        self.assertTupleEqual(
            ReadImageResponse._fields,
            (
                "image",
                "alpha",
                "original",
                "anchored",
                "to_masked",
                "original_size",
                "anchored_size",
            ),
        )
        values = tuple(object() for _ in ReadImageResponse._fields)
        response = ReadImageResponse(*values)
        self.assertTupleEqual(tuple(response), values)
        with self.assertRaises(TypeError):
            ReadImageResponse(*values[:-1])

    def test_read_image_legacy_guards(self):
        array = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
        normalized = (array.astype(np.float32) / 255.0).transpose(2, 0, 1)
        image = Image.fromarray(array)
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = os.path.join(temp_dir, "test.png")
            image.save(test_path)
            response = read_image(test_path, None, anchor=None)
            np.testing.assert_allclose(response.image[0], normalized)
            self.assertIsNone(response.alpha)
            response = read_image(test_path, None, anchor=None, to_gray=True)
            gt = np.array(Image.fromarray(array).convert("L"))
            np.testing.assert_allclose(response.image[0][0], gt / 255.0)

        image = self._get_rgba_image()
        with self.assertRaises(ValueError):
            read_image(image, None, anchor=None, to_gray=True, to_mask=True)
        response = read_image(image, None, anchor=None, to_mask=True)
        np.testing.assert_allclose(
            response.image[0][0], np.array(image.getchannel("A")) / 255.0
        )

        large = Image.new("RGB", (100, 100))
        response = read_image(large, None, anchor=32)
        self.assertSequenceEqual(response.image.shape, [1, 3, 96, 96])
        self.assertIsNone(response.alpha)
        response = read_image(large, 80, anchor=32)
        self.assertSequenceEqual(response.image.shape, [1, 3, 64, 64])

    def test_read_rgba_resize_anchor_layout_and_normalization(self):
        image = self._get_rgba_image()
        size = (4, 4)
        expected_image = np.array(
            to_rgb(image).resize(size, resample=Image.Resampling.LANCZOS)
        )
        expected_alpha = np.array(
            image.getchannel("A").resize(size, resample=Image.Resampling.LANCZOS)
        )
        for to_torch_fmt in [False, True]:
            for normalize in [False, True]:
                with self.subTest(
                    to_torch_fmt=to_torch_fmt,
                    normalize=normalize,
                ):
                    response = read_image(
                        image,
                        6,
                        anchor=4,
                        to_torch_fmt=to_torch_fmt,
                        normalize=normalize,
                    )
                    image_array = expected_image
                    alpha_array = expected_alpha
                    if normalize:
                        image_array = image_array.astype(np.float32) / 255.0
                        alpha_array = alpha_array.astype(np.float32) / 255.0
                    if to_torch_fmt:
                        image_array = image_array[None].transpose(0, 3, 1, 2)
                        alpha_array = alpha_array[None, None]
                    np.testing.assert_array_equal(response.image, image_array)
                    np.testing.assert_array_equal(response.alpha, alpha_array)
                    self.assertEqual(response.image.dtype, image_array.dtype)
                    self.assertEqual(response.alpha.dtype, alpha_array.dtype)
                    np.testing.assert_array_equal(response.anchored, expected_image)
                    self.assertIs(response.original, image)
                    self.assertTupleEqual(response.original_size, (7, 5))
                    self.assertTupleEqual(response.anchored_size, size)
                    self.assertEqual(response.anchored.size, size)
                    self.assertIsNone(response.to_masked)

    def test_read_mask_resampling_and_metadata(self):
        image = self._get_rgba_image()
        alpha = image.getchannel("A")
        size = (4, 4)
        expected_nearest = np.array(
            alpha.resize(size, resample=Image.Resampling.NEAREST)
        )
        expected_bilinear = np.array(
            alpha.resize(size, resample=Image.Resampling.BILINEAR)
        )
        self.assertFalse(np.array_equal(expected_nearest, expected_bilinear))
        for resample in [Image.Resampling.NEAREST, Image.Resampling.BILINEAR]:
            with self.subTest(resample=resample):
                kwargs = (
                    {}
                    if resample == Image.Resampling.NEAREST
                    else {"resample": resample}
                )
                response = read_image(
                    image,
                    6,
                    anchor=4,
                    to_mask=True,
                    normalize=False,
                    to_torch_fmt=False,
                    **kwargs,
                )
                expected = np.array(alpha.resize(size, resample=resample))
                np.testing.assert_array_equal(response.image, expected)
                np.testing.assert_array_equal(response.alpha, expected)
                np.testing.assert_array_equal(response.anchored, expected)
                self.assertIsNotNone(response.to_masked)
                np.testing.assert_array_equal(response.to_masked, expected)
                self.assertTupleEqual(response.original_size, (7, 5))
                self.assertTupleEqual(response.anchored_size, size)
                self.assertEqual(response.original.size, response.original_size)
                self.assertEqual(response.anchored.size, response.anchored_size)
                self.assertEqual(response.to_masked.size, response.anchored_size)

    def test_save_images(self):
        array = np.random.randn(5, 3, 7, 11)
        tensor = torch.randn(5, 3, 7, 11)
        with tempfile.TemporaryDirectory() as temp_dir:
            save_images(array, os.path.join(temp_dir, "test_array.png"))
            save_images(tensor, os.path.join(temp_dir, "test_tensor.png"))

    def test_base_64(self):
        array = np.random.randint(0, 256, [100, 100, 3]).astype(np.uint8)
        image = Image.fromarray(array)
        recovered = from_base64(to_base64(image))
        np.testing.assert_allclose(recovered, array)

    def test_image_box(self):
        box = ImageBox(3, 5, 7, 11)
        self.assertEqual(box.w, 4)
        self.assertEqual(box.h, 6)
        self.assertEqual(box.wh_ratio, 2 / 3)
        self.assertTupleEqual(box.tuple, (3, 5, 7, 11))
        self.assertEqual(box.matrix, Matrix2D(a=4, b=0, c=0, d=6, e=3, f=5))
        self.assertEqual(box, box.copy())
        self.assertNotEqual(box, "foo")
        array = np.arange(200).reshape(20, 10)
        tensor = torch.arange(200).view(20, 10)
        np.testing.assert_allclose(box.crop(array), array[5:11, 3:7])
        torch.testing.assert_close(box.crop(tensor), tensor[5:11, 3:7])
        self.assertEqual(box.pad(2), ImageBox(1, 3, 9, 13))
        self.assertEqual(box.pad(2, w=7), ImageBox(1, 3, 8, 13))
        self.assertEqual(box.pad(2, h=7), ImageBox(1, 3, 9, 10))
        self.assertEqual(box.to_square(), ImageBox(2, 5, 8, 11))
        self.assertEqual(box.to_square(w=5), ImageBox(2, 5, 7, 11))
        self.assertEqual(box.to_square(h=5), ImageBox(2, 5, 8, 10))
        self.assertEqual(box.to_square().to_square(), ImageBox(2, 5, 8, 11))
        self.assertEqual(box.to_square(expand=False), ImageBox(3, 6, 7, 10))
        self.assertEqual(box.to_square(w=3, expand=False), ImageBox(3, 6, 6, 10))
        self.assertEqual(box.to_square(h=3, expand=False), ImageBox(3, 6, 7, 9))
        box = ImageBox(5, 3, 11, 7)
        self.assertEqual(box.to_square(), ImageBox(5, 2, 11, 8))
        self.assertEqual(box.to_square(expand=False), ImageBox(6, 3, 10, 7))

    def test_image_box_half_open_mask_roundtrip(self):
        mask = np.zeros([20, 10], dtype=np.uint8)
        empty = ImageBox.from_mask(mask)
        self.assertEqual(empty, ImageBox(0, 0, 0, 0))
        self.assertTupleEqual((empty.w, empty.h), (0, 0))
        self.assertTupleEqual(empty.crop(mask).shape, (0, 0))

        mask[5:12, 3:8] = 1
        box = ImageBox.from_mask(mask)
        self.assertEqual(box, ImageBox(3, 5, 8, 12))
        self.assertTupleEqual((box.w, box.h), (5, 7))
        self.assertEqual(box.matrix, Matrix2D(a=5, b=0, c=0, d=7, e=3, f=5))
        np.testing.assert_array_equal(box.crop(mask), mask[5:12, 3:8])
        tensor = torch.from_numpy(mask.copy())
        torch.testing.assert_close(box.crop(tensor), tensor[5:12, 3:8])
        self.assertEqual(ImageBox.from_inclusive(3, 5, 7, 11), box)

        edge_mask = np.zeros_like(mask)
        edge_mask[-1, -1] = 1
        edge = ImageBox.from_mask(edge_mask)
        self.assertEqual(edge, ImageBox(9, 19, 10, 20))
        self.assertTupleEqual((edge.w, edge.h), (1, 1))
        self.assertTupleEqual(edge.crop(edge_mask).shape, (1, 1))

    def test_image_box_mask_threshold_is_strict(self):
        mask = np.zeros([4, 5], dtype=np.uint8)
        mask[0, 0] = 5
        mask[2, 3] = 6
        mask[3, 4] = 9
        self.assertEqual(ImageBox.from_mask(mask, threshold=5), ImageBox(3, 2, 5, 4))
        self.assertEqual(ImageBox.from_mask(mask, threshold=9), ImageBox(0, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
