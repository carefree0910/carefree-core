import unittest

import numpy as np
import core.learn as cflearn


class TestData(unittest.TestCase):
    def test_array_data(self) -> None:
        input_dim = 11
        output_dim = 7
        num_samples = 123
        batch_size = 17

        x = np.random.randn(num_samples, input_dim)
        y = np.random.randn(num_samples, output_dim)
        data = cflearn.ArrayData.init().fit(x, y)
        data.config.batch_size = batch_size
        loader = data.build_loader(x, y, drop_last=True)

        for batch in loader:
            x_batch = batch[cflearn.INPUT_KEY]
            y_batch = batch[cflearn.LABEL_KEY]
            self.assertEqual(x_batch.shape, (batch_size, input_dim))
            self.assertEqual(y_batch.shape, (batch_size, output_dim))

    def test_array_dict_data(self) -> None:
        input_dim = 11
        output_dim = 7
        num_samples = 123
        batch_size = 17

        x = np.random.randn(num_samples, input_dim)
        y = np.random.randn(num_samples, output_dim)
        d = dict(a=x, b=y)
        data = cflearn.ArrayDictData.init().fit(d)
        data.config.batch_size = batch_size
        loader = data.build_loader(d, drop_last=True)

        for batch in loader:
            x_batch = batch["a"]
            y_batch = batch["b"]
            self.assertEqual(x_batch.shape, (batch_size, input_dim))
            self.assertEqual(y_batch.shape, (batch_size, output_dim))


if __name__ == "__main__":
    unittest.main()
