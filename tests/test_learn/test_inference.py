import torch
import unittest

import numpy as np
import core.learn as cflearn
import torch.nn as nn

from torch import Tensor
from typing import Optional
from accelerate import Accelerator
from rich.progress import Progress
from unittest.mock import patch
from unittest.mock import PropertyMock
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
        num_samples = 17 * 31  # `mse` == `stream_mse` iff every batch is full
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

        progress = Progress()
        progress.start()
        model_outputs = inference.get_outputs(loader, progress=progress)
        for v in model_outputs.forward_results.values():
            self.assertEqual(v.shape, (num_samples, output_dim))
        self.assertIsNone(model_outputs.labels.get(cflearn.LABEL_KEY))

        raw_outputs = inference.get_outputs(loader, stack_outputs=False)
        for v in raw_outputs.forward_results.values():
            self.assertIsInstance(v, list)
            for vv in v[:-1]:
                self.assertEqual(vv.shape, (batch_size, output_dim))
        self.assertIsNone(model_outputs.labels.get(cflearn.LABEL_KEY))

        mae = cflearn.MAE()
        model_outputs = inference.get_outputs(loader, return_labels=True)
        model_outputs = inference.get_outputs(loader, metrics=mae, return_labels=True)
        model_outputs = inference.get_outputs(
            loader,
            metrics=cflearn.IMetric.fuse(["mse", "stream_mse"]),
            return_labels=True,
        )
        for v in model_outputs.forward_results.values():
            self.assertEqual(v.shape, (num_samples, output_dim))
        labels = model_outputs.labels[cflearn.LABEL_KEY]
        self.assertEqual(labels.shape, (num_samples, output_dim))
        self.assertAlmostEqual(
            model_outputs.metric_outputs.metric_values["mse"],
            model_outputs.metric_outputs.metric_values["stream_mse"],
        )

        @cflearn.IMetric.register("foo", allow_duplicate=True)
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

        @cflearn.IMetric.register("bar", allow_duplicate=True)
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

        foo = FooMetric()
        model_outputs = inference.get_outputs(loader, metrics=foo, return_labels=True)
        self.assertEqual(model_outputs.metric_outputs.metric_values["foo"], 0.12)
        self.assertEqual(model_outputs.metric_outputs.final_score, 0.12)

        model_outputs = inference.get_outputs(loader, metrics=BarMetric())
        self.assertEqual(model_outputs.metric_outputs.metric_values["bar"], 3.45)
        self.assertEqual(model_outputs.metric_outputs.final_score, -3.45)

        is_multiple = True
        metrics = cflearn.MultipleMetrics.fuse(["foo", "bar"])
        model_outputs = inference.get_outputs(loader, metrics=metrics)
        metric_values = model_outputs.metric_outputs.metric_values
        self.assertAlmostEqual(metric_values["foo"], 0.12)
        self.assertAlmostEqual(metric_values["bar"], 3.45)
        final_score = model_outputs.metric_outputs.final_score
        self.assertAlmostEqual(final_score, 0.5 * (0.12 - 3.45))

        with patch(
            "accelerate.Accelerator.is_main_process",
            new_callable=PropertyMock,
        ) as mock:
            mock.return_value = False
            inference.get_outputs(loader, metrics=metrics, accelerator=Accelerator())

        @cflearn.IMetric.register("ve", allow_duplicate=True)
        class ValueErrorMetric(cflearn.IMetric):
            @property
            def is_positive(self) -> bool:
                return True

            def forward(
                self,
                np_batch: np_dict_type,
                np_outputs: np_dict_type,
                loader: Optional[DataLoader] = None,
            ) -> float:
                raise ValueError

        @cflearn.IModel.register("kim", allow_duplicate=True)
        class KeyboardInterruptModel(cflearn.CommonModel):
            def step(self, *args, **kwargs) -> cflearn.StepOutputs:
                raise KeyboardInterrupt

        @cflearn.IModel.register("vem", allow_duplicate=True)
        class ValueErrorModel(cflearn.CommonModel):
            def step(self, *args, **kwargs) -> cflearn.StepOutputs:
                raise ValueError

        def run(model: str, metrics: cflearn.IMetric) -> None:
            config.model = model
            inference = cflearn.Inference(model=cflearn.IModel.from_config(config))
            inference.get_outputs(loader, metrics=metrics)

        with self.assertRaises(KeyboardInterrupt):
            run("kim", cflearn.MAE())
        with self.assertRaises(ValueError):
            run("vem", cflearn.MAE())
        with self.assertRaises(ValueError):
            run("vem", ValueErrorMetric())

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
        with self.assertRaises(ValueError):
            inference.get_outputs(loader, pad_dim=0)
        with self.assertRaises(ValueError):
            inference.get_outputs(loader, pad_dim={"foo": 0}, accelerator=Accelerator())
        o0 = inference.get_outputs(loader, pad_dim=1)
        o1 = inference.get_outputs(loader, pad_dim=1, accelerator=Accelerator())
        o2 = inference.get_outputs(
            loader,
            pad_dim={cflearn.PREDICTIONS_KEY: 1},
            accelerator=Accelerator(),
        )
        gt = np.array(
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
        )
        np.testing.assert_array_equal(o0.forward_results[cflearn.PREDICTIONS_KEY], gt)
        np.testing.assert_array_equal(o1.forward_results[cflearn.PREDICTIONS_KEY], gt)
        np.testing.assert_array_equal(o2.forward_results[cflearn.PREDICTIONS_KEY], gt)


if __name__ == "__main__":
    unittest.main()
