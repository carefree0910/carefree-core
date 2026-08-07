import torch
import inspect
import unittest

import numpy as np
import torch.nn as nn
import core.learn as cflearn
import core.learn.schema as learn_schema
import core.learn.inference as learn_inference

from torch import Tensor
from typing import Tuple
from typing import Optional
from accelerate import Accelerator
from rich.progress import Progress
from unittest.mock import patch
from unittest.mock import Mock
from unittest.mock import MagicMock
from unittest.mock import PropertyMock
from core.learn.schema import DataLoader
from core.toolkit.types import np_dict_type


def _make_loss_model() -> Tuple[cflearn.Config, cflearn.IModel]:
    cflearn.register_module("$loss_identity", allow_duplicate=True)(nn.Identity)
    config = cflearn.Config(
        module_name="$loss_identity",
        loss_name="mse",
        use_losses_as_metrics=True,
    )
    return config, cflearn.IModel.from_config(config)


class TestInference(unittest.TestCase):
    def test_public_contract(self) -> None:
        self.assertIs(cflearn.Inference, learn_inference.Inference)
        self.assertIs(cflearn.InferenceOutputs, learn_schema.InferenceOutputs)
        self.assertListEqual(learn_inference.__all__, ["Inference"])
        self.assertEqual(
            inspect.signature(cflearn.Inference.get_outputs),
            inspect.signature(learn_schema.IInference.get_outputs),
        )

        forward_results = {cflearn.PREDICTIONS_KEY: torch.ones(2, 1)}
        labels = {cflearn.LABEL_KEY: torch.zeros(2, 1)}
        metric_outputs = cflearn.MetricsOutputs(
            1.0,
            {"score": 1.0},
            {"score": True},
        )
        loss_items = {"loss": 0.5}
        outputs = cflearn.InferenceOutputs(
            forward_results,
            labels,
            metric_outputs,
            loss_items,
        )

        self.assertListEqual(
            list(vars(outputs)),
            [
                "forward_results",
                "labels",
                "metric_outputs",
                "loss_items",
            ],
        )
        self.assertIs(outputs.forward_results, forward_results)
        self.assertIs(outputs.labels, labels)
        self.assertIs(outputs.metric_outputs, metric_outputs)
        self.assertIs(outputs.loss_items, loss_items)

    def test_distributed_helpers(self) -> None:
        def gather(objects):
            self.assertFalse(torch.is_inference_mode_enabled())
            return objects

        with patch.object(learn_inference, "gather_object", side_effect=gather):
            with torch.inference_mode():
                self.assertListEqual(learn_inference._gather_objects([1]), [1])

        accumulator = cflearn.MetricAccumulator()
        stream = cflearn.StreamMSE()
        stream.reset()
        self.assertIs(
            learn_inference._merge_metrics(None, accumulator, [stream]),
            accumulator,
        )
        with patch.object(
            learn_inference,
            "_gather_objects",
            return_value=[
                (cflearn.MetricAccumulator(), [(0.0, 2)]),
                (cflearn.MetricAccumulator(), [(9.0, 1)]),
            ],
        ) as gather:
            merged = learn_inference._merge_metrics(
                MagicMock(),
                accumulator,
                [stream],
            )
        gather.assert_called_once()
        self.assertIsNone(merged.finalize())
        self.assertEqual(stream.finalize(), 3.0)
        with patch.object(
            learn_inference,
            "_gather_objects",
            return_value=[(accumulator, [None])],
        ):
            learn_inference._merge_metrics(MagicMock(), accumulator, [stream])
        self.assertEqual(stream.finalize(), 3.0)

    def test_successful_native_inference_is_single_pass(self) -> None:
        input_dim = 3
        output_dim = 1
        batch_size = 2
        batches = [
            {cflearn.INPUT_KEY: torch.randn(batch_size, input_dim)},
            {cflearn.INPUT_KEY: torch.randn(batch_size, input_dim)},
        ]
        config = cflearn.Config(
            module_name="linear",
            module_config={"input_dim": input_dim, "output_dim": output_dim},
            loss_name="mse",
        )
        model = cflearn.IModel.from_config(config)
        inference = cflearn.Inference(model=model)

        for should_stop_progress in [True, False]:
            with self.subTest(should_stop_progress=should_stop_progress):
                yielded_batches = []

                def iterate():
                    for i, batch in enumerate(batches):
                        yielded_batches.append(i)
                        yield batch

                loader = MagicMock()
                loader.__len__.return_value = len(batches)
                loader.__iter__.side_effect = iterate
                progress = MagicMock()
                progress_task = 7
                progress.add_task.return_value = progress_task
                inject_outputs = Mock()

                with patch.object(model, "step", wraps=model.step) as model_step:
                    outputs = inference.get_outputs(
                        loader,
                        recover_labels=False,
                        recover_predictions=False,
                        inject_outputs_fn=inject_outputs,
                        progress=progress,
                        should_stop_progress=should_stop_progress,
                    )

                self.assertListEqual(yielded_batches, list(range(len(batches))))
                self.assertEqual(loader.__iter__.call_count, 1)
                self.assertEqual(model_step.call_count, len(batches))
                self.assertEqual(inject_outputs.call_count, len(batches))
                self.assertEqual(progress.advance.call_count, len(batches))
                for advance_call in progress.advance.call_args_list:
                    self.assertTupleEqual(advance_call.args, (progress_task,))
                progress.add_task.assert_called_once()
                progress.remove_task.assert_called_once_with(progress_task)
                if should_stop_progress:
                    progress.stop.assert_called_once_with()
                else:
                    progress.stop.assert_not_called()
                self.assertTupleEqual(
                    outputs.forward_results[cflearn.PREDICTIONS_KEY].shape,
                    (len(batches) * batch_size, output_dim),
                )

    def test_native_model_can_be_rebound(self) -> None:
        x = np.random.randn(2, 3).astype(np.float32)
        loader = cflearn.ArrayData.init().fit(x).build_loader(x)
        config = cflearn.Config(
            module_name="linear",
            module_config={"input_dim": 3, "output_dim": 1},
            loss_name="mse",
        )
        original = cflearn.IModel.from_config(config)
        replacement = cflearn.IModel.from_config(config)
        inference = cflearn.Inference(model=original)
        inference.model = replacement

        with patch.object(original, "step", wraps=original.step) as original_step:
            with patch.object(
                replacement,
                "step",
                wraps=replacement.step,
            ) as replacement_step:
                inference.get_outputs(
                    loader,
                    target_outputs=[cflearn.PREDICTIONS_KEY],
                    target_labels=[cflearn.LABEL_KEY],
                    recover_labels=False,
                    recover_predictions=False,
                )

        original_step.assert_not_called()
        self.assertEqual(replacement_step.call_count, len(loader))

    def test_uneven_batch_metric_aggregation(self) -> None:
        self.assertEqual(learn_inference._get_sample_count({}), 1.0)

        batches = [
            {
                cflearn.INPUT_KEY: torch.tensor([[0.0], [0.0]]),
                cflearn.LABEL_KEY: torch.tensor([[0.0], [0.0]]),
            },
            {
                cflearn.INPUT_KEY: torch.tensor([[3.0]]),
                cflearn.LABEL_KEY: torch.tensor([[0.0]]),
            },
        ]
        loader = MagicMock()
        loader.__len__.return_value = len(batches)
        loader.__iter__.side_effect = lambda: iter(batches)
        onnx = Mock()
        onnx.predict.side_effect = lambda batch: {
            cflearn.PREDICTIONS_KEY: batch[cflearn.INPUT_KEY]
        }

        outputs = cflearn.Inference(onnx=onnx).get_outputs(
            loader,
            metrics=cflearn.IMetric.fuse(["mse", "stream_mse"]),
            recover_labels=False,
            recover_predictions=False,
        )

        metric_outputs = outputs.metric_outputs
        self.assertIsNotNone(metric_outputs)
        metric_values = metric_outputs.metric_values
        self.assertEqual(metric_values["mse"], 3.0)
        self.assertEqual(metric_values["stream_mse"], 3.0)
        self.assertEqual(metric_outputs.final_score, -3.0)

        class DiagnosticMSE(cflearn.MSE):
            __identifier__ = "diagnostic_mse"

            @property
            def not_include_in_score(self) -> bool:
                return True

        mixed_outputs = cflearn.Inference(onnx=onnx).get_outputs(
            loader,
            metrics=cflearn.MultipleMetrics(
                [
                    DiagnosticMSE(),
                    cflearn.StreamMSE(),
                ]
            ),
            recover_labels=False,
            recover_predictions=False,
        )
        mixed_metric_outputs = mixed_outputs.metric_outputs
        self.assertIsNotNone(mixed_metric_outputs)
        self.assertEqual(mixed_metric_outputs.metric_values["diagnostic_mse"], 3.0)
        self.assertEqual(mixed_metric_outputs.metric_values["stream_mse"], 3.0)
        self.assertEqual(mixed_metric_outputs.final_score, -3.0)

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

        raw_outputs = inference.get_outputs(loader, concat_outputs=False)
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
        stream_metric = cflearn.StreamMSE()
        stream_outputs = inference.get_outputs(
            loader,
            metrics=stream_metric,
        )
        self.assertAlmostEqual(
            stream_outputs.metric_outputs.metric_values["stream_mse"],
            model_outputs.metric_outputs.metric_values["stream_mse"],
        )

        onnx = Mock()
        onnx.predict.side_effect = lambda batch: {
            cflearn.PREDICTIONS_KEY: batch[cflearn.INPUT_KEY][:, :output_dim]
        }
        inject_outputs = Mock()
        cflearn.Inference(onnx=onnx).get_outputs(
            loader,
            inject_outputs_fn=inject_outputs,
        )
        self.assertEqual(inject_outputs.call_count, len(loader))

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

        @cflearn.IMetric.register("metadata", allow_duplicate=True)
        class MetadataMetric(cflearn.IMetric):
            @property
            def is_positive(self) -> bool:
                return True

            @property
            def requires_all(self) -> bool:
                return True

            def requires(self, key: str) -> bool:
                return key == "metadata"

            def forward(
                self,
                np_batch: np_dict_type,
                np_outputs: np_dict_type,
                loader: Optional[DataLoader] = None,
            ) -> float:
                case.assertListEqual(np_batch["metadata"], ["first", "second"])
                return 0.25

        batches = [
            {
                cflearn.INPUT_KEY: torch.randn(2, input_dim),
                cflearn.LABEL_KEY: torch.randn(2, output_dim),
                "metadata": "first",
            },
            {
                cflearn.INPUT_KEY: torch.randn(3, input_dim),
                cflearn.LABEL_KEY: torch.randn(3, output_dim),
                "metadata": "second",
            },
        ]
        metadata_loader = MagicMock()
        metadata_loader.__len__.return_value = len(batches)
        metadata_loader.__iter__.side_effect = lambda: iter(batches)
        metadata_loader.recover_labels.side_effect = lambda key, value: value
        metadata_outputs = inference.get_outputs(
            metadata_loader,
            metrics=MetadataMetric(),
            target_inputs=[cflearn.INPUT_KEY, "metadata"],
        )
        self.assertEqual(
            metadata_outputs.forward_results["metadata"],
            ["first", "second"],
        )

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

        @cflearn.IMetric.register("empty", allow_duplicate=True)
        class EmptyMetric(cflearn.IMetric):
            @property
            def is_positive(self) -> bool:
                return True

            def forward(
                self,
                np_batch: np_dict_type,
                np_outputs: np_dict_type,
                loader: Optional[DataLoader] = None,
            ) -> float:
                return 0.0

            def evaluate(
                self,
                tensor_batch,
                tensor_outputs,
                loader=None,
            ):
                return None

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
        config.model = "common"
        inference = cflearn.Inference(model=cflearn.IModel.from_config(config))
        with self.assertRaises(RuntimeError):
            inference.get_outputs(loader, metrics=EmptyMetric())

    def test_batch_averaged_loss(self) -> None:
        batches = [
            {
                cflearn.INPUT_KEY: torch.zeros(2, 1),
                cflearn.LABEL_KEY: torch.zeros(2, 1),
            },
            {
                cflearn.INPUT_KEY: torch.tensor([[3.0]]),
                cflearn.LABEL_KEY: torch.zeros(1, 1),
            },
        ]
        config, model = _make_loss_model()
        outputs = model.evaluate(
            config,
            None,
            cflearn.Inference(model=model),
            batches,  # type: ignore
            recover_labels=False,
            recover_predictions=False,
        )

        self.assertIsNotNone(outputs.loss_items)
        self.assertEqual(outputs.loss_items[cflearn.LOSS_KEY], 4.5)

    def test_remainder_aware_loss(self) -> None:
        batches = [
            {
                cflearn.INPUT_KEY: torch.zeros(1, 1),
                cflearn.LABEL_KEY: torch.zeros(1, 1),
            },
            {
                cflearn.INPUT_KEY: torch.tensor([[3.0]]),
                cflearn.LABEL_KEY: torch.zeros(1, 1),
            },
        ]
        accelerator = MagicMock()
        accelerator.gradient_state.remainder = 1

        def gather(losses):
            self.assertFalse(torch.is_inference_mode_enabled())
            return [torch.cat([loss, torch.zeros_like(loss)]) for loss in losses]

        accelerator.gather.side_effect = gather
        config, model = _make_loss_model()
        outputs = model.evaluate(
            config,
            None,
            cflearn.Inference(model=model),
            batches,  # type: ignore
            recover_labels=False,
            recover_predictions=False,
            accelerator=accelerator,
        )

        self.assertIsNotNone(outputs.loss_items)
        self.assertEqual(outputs.loss_items[cflearn.LOSS_KEY], 3.0)

    def test_inference_error_does_not_retry_with_grad(self) -> None:
        run_states = []
        inference_error = RuntimeError("inference failed")

        @cflearn.IModel.register("inference_error", allow_duplicate=True)
        class InferenceErrorModel(cflearn.CommonModel):
            def step(self, *args, **kwargs) -> cflearn.StepOutputs:
                run_states.append(
                    (
                        self.m.training,
                        self.loss.training,
                        torch.is_grad_enabled(),
                        torch.is_inference_mode_enabled(),
                    )
                )
                raise inference_error

        x = np.random.randn(2, 3).astype(np.float32)
        data = cflearn.ArrayData.init().fit(x)
        loader = data.build_loader(x)
        config = cflearn.Config(
            model="inference_error",
            module_name="linear",
            module_config={"input_dim": 3, "output_dim": 1},
            loss_name="mse",
        )
        model = cflearn.IModel.from_config(config)
        model.m.train()
        model.loss.train()
        inference = cflearn.Inference(model=model)
        progress = MagicMock()
        progress.add_task.return_value = 1
        progress.stop.side_effect = RuntimeError("progress cleanup failed")
        progress.remove_task.side_effect = RuntimeError("task cleanup failed")

        with self.assertRaises(RuntimeError) as context:
            inference.get_outputs(loader, progress=progress)

        self.assertIs(context.exception, inference_error)
        self.assertListEqual(run_states, [(False, False, False, True)])
        self.assertTrue(model.m.training)
        self.assertTrue(model.loss.training)
        self.assertFalse(inference.use_grad_in_predict)
        progress.add_task.assert_called_once()
        progress.stop.assert_called_once_with()
        progress.remove_task.assert_called_once_with(1)

        with self.assertRaises(RuntimeError) as context:
            inference.get_outputs(loader, use_grad=True)

        self.assertIs(context.exception, inference_error)
        self.assertListEqual(
            run_states,
            [
                (False, False, False, True),
                (False, False, True, False),
            ],
        )
        self.assertTrue(model.m.training)
        self.assertTrue(model.loss.training)
        self.assertFalse(inference.use_grad_in_predict)

        model = cflearn.IModel.from_config(config)
        model.m.train()
        model.loss.train()
        inference = cflearn.Inference(
            model=model,
            use_grad_in_predict=True,
        )
        with self.assertRaises(RuntimeError) as context:
            inference.get_outputs(loader)

        self.assertIs(context.exception, inference_error)
        self.assertListEqual(
            run_states,
            [
                (False, False, False, True),
                (False, False, True, False),
                (False, False, True, False),
            ],
        )
        self.assertTrue(model.m.training)
        self.assertTrue(model.loss.training)
        self.assertTrue(inference.use_grad_in_predict)

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
        with self.assertRaises(RuntimeError):
            inference.get_outputs(loader)
        with self.assertRaises(RuntimeError):
            inference.get_outputs(loader, pad_dim=0)
        with self.assertRaises(RuntimeError):
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

        x_float = np.empty_like(x)
        x_float[:] = [sample.astype(np.float32) for sample in x]
        float_data = cflearn.ArrayData.init().fit(x_float)
        float_data.config.batch_size = 1
        float_loader = float_data.build_loader(x_float)
        float_outputs = inference.get_outputs(
            float_loader,
            pad_dim=1,
            verbose=False,
        )
        float_predictions = float_outputs.forward_results[cflearn.PREDICTIONS_KEY]
        self.assertTrue(torch.isnan(float_predictions[0, 1:]).all())
        self.assertTrue(torch.isnan(float_predictions[3, 2:]).all())

        same_x = np.empty(2, dtype=object)
        same_x[:] = [
            np.ones((3, 2), dtype=np.float32),
            np.zeros((3, 2), dtype=np.float32),
        ]
        same_data = cflearn.ArrayData.init().fit(same_x)
        same_data.config.batch_size = 1
        same_loader = same_data.build_loader(same_x)
        same_outputs = inference.get_outputs(same_loader, pad_dim=1)
        self.assertEqual(
            same_outputs.forward_results[cflearn.PREDICTIONS_KEY].shape,
            (6, 2),
        )


if __name__ == "__main__":
    # unittest.main()
    TestInference().test_pad()
