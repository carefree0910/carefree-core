import os
import torch
import tempfile
import unittest

import numpy as np

from PIL import Image
from core.toolkit.cv import *


class TestCV(unittest.TestCase):
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

    def test_read_image(self):
        array = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
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

        array = np.random.randint(0, 256, [100, 100, 4], dtype=np.uint8)
        image = Image.fromarray(array)
        with self.assertRaises(ValueError):
            read_image(image, None, anchor=None, to_gray=True, to_mask=True)
        response = read_image(image, None, anchor=None)
        np.testing.assert_allclose(response.alpha[0][0], array[..., -1] / 255.0)
        response = read_image(image, None, anchor=None, to_mask=True)
        np.testing.assert_allclose(response.image[0][0], array[..., -1] / 255.0)

        response = read_image(image, None, anchor=32)
        self.assertSequenceEqual(response.image.shape, [1, 3, 96, 96])
        response = read_image(image, 80, anchor=32)
        self.assertSequenceEqual(response.image.shape, [1, 3, 64, 64])
        gt_img = to_rgb(Image.fromarray(array))
        gt_img = gt_img.resize((64, 64), resample=Image.Resampling.LANCZOS)
        gt = np.array(gt_img).transpose(2, 0, 1)
        np.testing.assert_allclose(response.image[0], gt / 255.0)

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
        np.testing.assert_allclose(box.crop(array), array[5:12, 3:8])
        torch.testing.assert_close(box.crop(tensor), tensor[5:12, 3:8])
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
        mask = np.zeros([20, 10], dtype=np.uint8)
        self.assertEqual(ImageBox.from_mask(mask), ImageBox(0, 0, 0, 0))
        mask[5:12, 3:8] = 1
        self.assertEqual(ImageBox.from_mask(mask), box)
        box = ImageBox(5, 3, 11, 7)
        self.assertEqual(box.to_square(), ImageBox(5, 2, 11, 8))
        self.assertEqual(box.to_square(expand=False), ImageBox(6, 3, 10, 7))


if __name__ == "__main__":
    unittest.main()
