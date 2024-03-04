import torch
import unittest

import numpy as np
import core.learn as cflearn

from accelerate import Accelerator
from core.toolkit.types import tensor_dict_type


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

        with self.assertRaises(ValueError):
            cflearn.ArrayData.init().fit(None).build_loaders()
        with self.assertRaises(ValueError):
            cflearn.ArrayData.init().fit(x, y[:-1]).build_loaders()

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

        with self.assertRaises(ValueError):
            cflearn.ArrayDictData.init().fit(x).build_loaders()
        with self.assertRaises(ValueError):
            cflearn.ArrayDictData.init().fit(dict(a=x, b=y[:-1])).build_loaders()
        with self.assertRaises(ValueError):
            cflearn.ArrayDictData.init().fit(d, x_valid=x).build_loaders()

    def test_recover(self) -> None:
        @cflearn.IDataBlock.register("foo", allow_duplicate=True)
        class FooBlock(cflearn.IDataBlock):
            def to_info(self) -> dict:
                return {}

            def transform(self, bundle, for_inference) -> cflearn.DataBundle:
                return bundle

            def fit_transform(self, bundle: cflearn.DataBundle) -> cflearn.DataBundle:
                return bundle

            def process_batch(self, batch, *, for_inference) -> tensor_dict_type:
                batch[cflearn.LABEL_KEY] -= 1
                return batch

            def recover_labels(self, key: str, y: torch.Tensor) -> torch.Tensor:
                return y + 1

        @cflearn.register_module("foo", allow_duplicate=True)
        class _(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.ones(1))

            def forward(self, batch):
                return self.param * torch.zeros_like(batch[cflearn.LABEL_KEY])

        input_dim = 11
        output_dim = 7
        num_samples = 23
        batch_size = 17

        x = np.random.randn(num_samples, input_dim)
        y = np.ones([num_samples, output_dim])

        data_config = cflearn.DataConfig()
        data_config.add_blocks(FooBlock)
        data = cflearn.ArrayData.init(data_config).fit(x, y)

        data.config.batch_size = batch_size
        loader = data.build_loader(x, y)
        for batch in loader:
            y_batch = batch[cflearn.LABEL_KEY]
            np.testing.assert_allclose(y_batch, np.zeros_like(y_batch))

        config = cflearn.Config(model="direct", module_name="foo", loss_name="mse")
        config.to_debug()
        p = cflearn.TrainingPipeline.init(config).fit(data)

        outputs = p.evaluate(loader, return_outputs=True)
        self.assertAlmostEqual(outputs.metric_outputs.final_score, 0.0)
        predictions = outputs.forward_results[cflearn.PREDICTIONS_KEY]
        np.testing.assert_allclose(predictions, np.ones_like(predictions))
        outputs = p.evaluate(loader, return_outputs=True, recover_labels=False)
        self.assertAlmostEqual(outputs.metric_outputs.final_score, 0.0)
        predictions = outputs.forward_results[cflearn.PREDICTIONS_KEY]
        np.testing.assert_allclose(predictions, np.zeros_like(predictions))

        accelerator = Accelerator()
        loader = cflearn.prepare_dataloaders(accelerator, loader)[0]
        outputs = p.evaluate(loader, return_outputs=True)
        self.assertAlmostEqual(outputs.metric_outputs.final_score, 0.0)
        predictions = outputs.forward_results[cflearn.PREDICTIONS_KEY]
        np.testing.assert_allclose(predictions, np.ones_like(predictions))
        outputs = p.evaluate(loader, return_outputs=True, recover_labels=False)
        self.assertAlmostEqual(outputs.metric_outputs.final_score, 0.0)
        predictions = outputs.forward_results[cflearn.PREDICTIONS_KEY]
        np.testing.assert_allclose(predictions, np.zeros_like(predictions))


if __name__ == "__main__":
    unittest.main()
