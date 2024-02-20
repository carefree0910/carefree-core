import torch
import unittest

import numpy as np
import core.learn as cflearn
import torch.nn as nn

from torch import Tensor
from typing import Optional
from core.learn.schema import DataLoader
from core.toolkit.types import np_dict_type


class TestInference(unittest.TestCase):
    def test_inference(self) -> None:
        with self.assertRaises(ValueError):
            cflearn.Inference()
        with self.assertRaises(ValueError):
            cflearn.Inference(onnx=1, model=1)

        input_dim = 11
        output_dim = 7
        num_samples = 123
        batch_size = 17

        x = np.random.randn(num_samples, input_dim).astype(np.float32)
        y = np.random.randn(num_samples, output_dim).astype(np.float32)
        data = cflearn.ArrayData.init().fit(x, y)
        data.config.batch_size = batch_size
        loader = data.build_loader(x, y)

        config = cflearn.Config(
            module_name="linear",
            module_config=dict(input_dim=input_dim, output_dim=output_dim),
            loss_name="mse",
        )
        inference = cflearn.Inference(model=cflearn.IModel.from_config(config))

        model_outputs = inference.get_outputs(loader, use_tqdm=True)
        for v in model_outputs.forward_results.values():
            self.assertEqual(v.shape, (num_samples, output_dim))
        self.assertIsNone(model_outputs.labels.get(cflearn.LABEL_KEY))

        raw_outputs = inference.get_outputs(loader, stack_outputs=False)
        for v in raw_outputs.forward_results.values():
            self.assertIsInstance(v, list)
            for vv in v[:-1]:
                self.assertEqual(vv.shape, (batch_size, output_dim))
        self.assertIsNone(model_outputs.labels.get(cflearn.LABEL_KEY))

        model_outputs = inference.get_outputs(loader, return_labels=True)
        for v in model_outputs.forward_results.values():
            self.assertEqual(v.shape, (num_samples, output_dim))
        labels = model_outputs.labels[cflearn.LABEL_KEY]
        self.assertEqual(labels.shape, (num_samples, output_dim))

        @cflearn.IMetric.register("foo")
        class FooMetric(cflearn.IMetric):
            @property
            def is_positive(self) -> bool:
                return True

            @property
            def requires_all(self) -> bool:
                return True

            def forward(
                self,
                np_batch: np_dict_type,
                np_outputs: np_dict_type,
                loader: Optional[DataLoader] = None,
            ) -> float:
                case.assertIn(cflearn.LABEL_KEY, np_batch)
                if is_multiple:
                    case.assertIn(cflearn.INPUT_KEY, np_batch)
                else:
                    case.assertNotIn(cflearn.INPUT_KEY, np_batch)
                return 0.12

        @cflearn.IMetric.register("bar")
        class BarMetric(cflearn.IMetric):
            @property
            def is_positive(self) -> bool:
                return False

            @property
            def requires_all(self) -> bool:
                return True

            def requires(self, key: str) -> bool:
                return key == cflearn.INPUT_KEY

            def forward(
                self,
                np_batch: np_dict_type,
                np_outputs: np_dict_type,
                loader: Optional[DataLoader] = None,
            ) -> float:
                case.assertIn(cflearn.INPUT_KEY, np_batch)
                if is_multiple:
                    case.assertIn(cflearn.LABEL_KEY, np_batch)
                else:
                    case.assertNotIn(cflearn.LABEL_KEY, np_batch)
                return 3.45

        case = self
        is_multiple = False

        model_outputs = inference.get_outputs(loader, metrics=FooMetric())
        self.assertEqual(model_outputs.metric_outputs.metric_values["foo"], 0.12)
        self.assertEqual(model_outputs.metric_outputs.final_score, 0.12)

        model_outputs = inference.get_outputs(loader, metrics=BarMetric())
        self.assertEqual(model_outputs.metric_outputs.metric_values["bar"], 3.45)
        self.assertEqual(model_outputs.metric_outputs.final_score, -3.45)

        is_multiple = True
        metrics = cflearn.MultipleMetrics.fuse(["foo", "bar"])
        model_outputs = inference.get_outputs(loader, metrics=metrics)
        metric_values = model_outputs.metric_outputs.metric_values
        self.assertEqual(metric_values["foo"], 0.12)
        self.assertEqual(metric_values["bar"], 3.45)
        final_score = model_outputs.metric_outputs.final_score
        self.assertEqual(final_score, 0.5 * (0.12 - 3.45))

    def test_pad(self) -> None:
        @cflearn.register_module("identity", allow_duplicate=True)
        class _(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return torch.from_numpy(x[0])

        x = np.empty(4, dtype=object)
        x[:] = [
            np.array([[0], [1], [2]]),
            np.array([[3, 4], [5, 6], [7, 8]]),
            np.array([[9, 10, 11], [12, 13, 14], [15, 16, 17]]),
            np.array([[18], [19], [20]]),
        ]
        data = cflearn.ArrayData.init().fit(x)
        data.config.batch_size = 1
        loader = data.build_loader(x)

        config = cflearn.Config(module_name="identity", loss_name="mse")
        inference = cflearn.Inference(model=cflearn.IModel.from_config(config))
        with self.assertRaises(ValueError):
            inference.get_outputs(loader)
        outputs = inference.get_outputs(loader, pad_dim=1)
        np.testing.assert_array_equal(
            outputs.forward_results[cflearn.PREDICTIONS_KEY],
            np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [2, 0, 0],
                    [3, 4, 0],
                    [5, 6, 0],
                    [7, 8, 0],
                    [9, 10, 11],
                    [12, 13, 14],
                    [15, 16, 17],
                    [18, 0, 0],
                    [19, 0, 0],
                    [20, 0, 0],
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
