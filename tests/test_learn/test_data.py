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

        data.config.async_prefetch = True
        data.config.async_prefetch_factor = 2
        async_loader = data.build_loader(x, y, drop_last=True)
        self.assertEqual(len(loader), len(async_loader))
        cursor = 0
        for b0, b1 in zip(loader, async_loader):
            cursor += 1
            x0, y0 = b0[cflearn.INPUT_KEY], b0[cflearn.LABEL_KEY]
            x1, y1 = b1[cflearn.INPUT_KEY], b1[cflearn.LABEL_KEY]
            np.testing.assert_allclose(x0, x1)
            np.testing.assert_allclose(y0, y1)
        self.assertEqual(cursor, len(loader))
        data.config.async_prefetch_factor = 1
        async_loader = data.build_loader(x, y, drop_last=True)
        cursor = 0
        for _ in async_loader:
            cursor += 1
        self.assertEqual(cursor, len(async_loader))

        with self.assertRaises(ValueError):
            data.fit(None).build_loaders()
        with self.assertRaises(ValueError):
            data.fit(x, y[:-1]).build_loaders()

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

    def test_process_batch(self) -> None:
        @cflearn.IDataBlock.register("foo", allow_duplicate=True)
        class FooBlock(cflearn.IDataBlock):
            def to_info(self) -> dict:
                return {}

            def transform(self, bundle, for_inference) -> cflearn.DataBundle:
                return bundle

            def fit_transform(self, bundle: cflearn.DataBundle) -> cflearn.DataBundle:
                return bundle

            def process_batch(self, batch, *, for_inference) -> tensor_dict_type:
                batch[cflearn.INPUT_KEY] -= 1
                return batch

        input_dim = 11
        num_samples = 23
        batch_size = 17

        x = np.random.randn(num_samples, input_dim)

        data_config = cflearn.DataConfig()
        data_config.add_blocks(FooBlock)
        data = cflearn.ArrayData.init(data_config).fit(x)
        data.config.batch_size = batch_size
        loader = data.build_loader(x)
        for i, batch in enumerate(loader):
            x_batch = batch[cflearn.INPUT_KEY]
            np.testing.assert_allclose(
                x_batch,
                x[i * batch_size : (i + 1) * batch_size] - 1,
            )

        accelerator = Accelerator()
        loader = cflearn.prepare_dataloaders(accelerator, loader)[0]
        for i, batch in enumerate(loader):
            x_batch = batch[cflearn.INPUT_KEY]
            np.testing.assert_allclose(
                x_batch,
                x[i * batch_size : (i + 1) * batch_size] - 1,
            )

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
        raw_kw = dict(recover_labels=False, recover_predictions=False)
        outputs = p.evaluate(loader, return_outputs=True, **raw_kw)
        self.assertAlmostEqual(outputs.metric_outputs.final_score, 0.0)
        predictions = outputs.forward_results[cflearn.PREDICTIONS_KEY]
        np.testing.assert_allclose(predictions, np.zeros_like(predictions))

        accelerator = Accelerator()
        loader = cflearn.prepare_dataloaders(accelerator, loader)[0]
        outputs = p.evaluate(loader, return_outputs=True)
        self.assertAlmostEqual(outputs.metric_outputs.final_score, 0.0)
        predictions = outputs.forward_results[cflearn.PREDICTIONS_KEY]
        np.testing.assert_allclose(predictions, np.ones_like(predictions))
        outputs = p.evaluate(loader, return_outputs=True, **raw_kw)
        self.assertAlmostEqual(outputs.metric_outputs.final_score, 0.0)
        predictions = outputs.forward_results[cflearn.PREDICTIONS_KEY]
        np.testing.assert_allclose(predictions, np.zeros_like(predictions))

    def test_testing_data(self) -> None:
        data, in_dim, out_dim = cflearn.testing.arange_data()
        config = cflearn.Config(
            module_name="linear",
            module_config=dict(input_dim=in_dim, output_dim=out_dim, bias=False),
            loss_name="mse",
        )
        config.to_debug()
        cflearn.TrainingPipeline.init(config).fit(data)
        data, in_dim, out_dim, _ = cflearn.testing.linear_data(1000, use_async=True)
        config = cflearn.Config(
            module_name="linear",
            module_config=dict(input_dim=in_dim, output_dim=out_dim, bias=False),
            loss_name="mse",
            num_steps=3,
        )
        cflearn.TrainingPipeline.init(config).fit(data)

    def test_async_data(self) -> None:
        data, *_ = cflearn.testing.linear_data(
            100,
            batch_size=4,
            use_validation=True,
            use_async=True,
        )
        train_loader, valid_loader = data.build_loaders()
        for i, _ in enumerate(train_loader):
            if i == 2:
                for _ in valid_loader:
                    break
                for _ in valid_loader:
                    break
                for _ in valid_loader:
                    break
            if i == 4:
                break
        for _ in train_loader:
            break
        for _ in train_loader:
            break

    def test_seeding(self) -> None:
        data = cflearn.testing.arange_data()[0]
        loader = data.build_loaders()[0]
        b0 = next(iter(loader))[cflearn.INPUT_KEY]
        loader = data.build_loaders()[0]
        b1 = next(iter(loader))[cflearn.INPUT_KEY]
        self.assertFalse(np.allclose(b0, b1))
        data.config.loader_seed = 42
        loader = data.build_loaders()[0]
        b0 = next(iter(loader))[cflearn.INPUT_KEY]
        loader = data.build_loaders()[0]
        b1 = next(iter(loader))[cflearn.INPUT_KEY]
        np.testing.assert_allclose(b0, b1)


if __name__ == "__main__":
    unittest.main()
