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
from contextlib import ExitStack
from rich.progress import Progress
from unittest.mock import patch
from torch.utils.data import DataLoader as TorchDataLoader
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
        accelerator = Accelerator(cpu=True)
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
                accelerator,
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
            learn_inference._merge_metrics(accelerator, accumulator, [stream])
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
        original_loader_iter = TorchDataLoader.__iter__

        for should_stop_progress in [True, False]:
            with self.subTest(should_stop_progress=should_stop_progress):
                loader = TorchDataLoader(batches, batch_size=None)
                progress = Progress(auto_refresh=False, disable=True)
                num_injections = 0

                def inject_outputs(_, __) -> None:
                    nonlocal num_injections
                    num_injections += 1

                with ExitStack() as stack:
                    loader_iter = stack.enter_context(
                        patch.object(
                            TorchDataLoader,
                            "__iter__",
                            autospec=True,
                            side_effect=original_loader_iter,
                        )
                    )
                    model_step = stack.enter_context(
                        patch.object(model, "step", wraps=model.step)
                    )
                    add_task = stack.enter_context(
                        patch.object(progress, "add_task", wraps=progress.add_task)
                    )
                    advance_progress = stack.enter_context(
                        patch.object(progress, "advance", wraps=progress.advance)
                    )
                    remove_task = stack.enter_context(
                        patch.object(
                            progress,
                            "remove_task",
                            wraps=progress.remove_task,
                        )
                    )
                    stop_progress = stack.enter_context(
                        patch.object(progress, "stop", wraps=progress.stop)
                    )
                    outputs = inference.get_outputs(
                        loader,  # type: ignore
                        recover_labels=False,
                        recover_predictions=False,
                        inject_outputs_fn=inject_outputs,
                        progress=progress,
                        should_stop_progress=should_stop_progress,
                    )

                loader_iter.assert_called_once_with(loader)
                self.assertEqual(model_step.call_count, len(batches))
                self.assertEqual(num_injections, len(batches))
                add_task.assert_called_once()
                self.assertEqual(add_task.call_args.kwargs["total"], len(batches))
                remove_task.assert_called_once()
                progress_task = remove_task.call_args.args[0]
                self.assertListEqual(
                    [call.args for call in advance_progress.call_args_list],
                    [(progress_task,)] * len(batches),
                )
                self.assertEqual(
                    stop_progress.call_count,
                    int(should_stop_progress),
                )
                self.assertListEqual(progress.tasks, [])
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

        original_forwards = []
        replacement_forwards = []
        handles = [
            original.m.register_forward_hook(lambda *_: original_forwards.append(None)),
            replacement.m.register_forward_hook(
                lambda *_: replacement_forwards.append(None)
            ),
        ]
        try:
            inference.get_outputs(
                loader,
                target_outputs=[cflearn.PREDICTIONS_KEY],
                target_labels=[cflearn.LABEL_KEY],
                recover_labels=False,
                recover_predictions=False,
            )
        finally:
            for handle in handles:
                handle.remove()

        self.assertListEqual(original_forwards, [])
        self.assertEqual(len(replacement_forwards), len(loader))

    def test_empty_loader(self) -> None:
        loader = TorchDataLoader([], batch_size=None)
        progress = Progress(auto_refresh=False, disable=True)
        _, model = _make_loss_model()

        with ExitStack() as stack:
            model_step = stack.enter_context(
                patch.object(model, "step", wraps=model.step)
            )
            add_task = stack.enter_context(
                patch.object(progress, "add_task", wraps=progress.add_task)
            )
            advance_progress = stack.enter_context(
                patch.object(progress, "advance", wraps=progress.advance)
            )
            remove_task = stack.enter_context(
                patch.object(progress, "remove_task", wraps=progress.remove_task)
            )
            stop_progress = stack.enter_context(
                patch.object(progress, "stop", wraps=progress.stop)
            )
            outputs = cflearn.Inference(model=model).get_outputs(
                loader,  # type: ignore
                progress=progress,
            )

        model_step.assert_not_called()
        self.assertDictEqual(outputs.forward_results, {})
        self.assertDictEqual(outputs.labels, {})
        self.assertIsNone(outputs.metric_outputs)
        self.assertIsNone(outputs.loss_items)
        add_task.assert_called_once()
        self.assertEqual(add_task.call_args.kwargs["total"], 0)
        advance_progress.assert_not_called()
        stop_progress.assert_called_once_with()
        remove_task.assert_called_once()
        self.assertListEqual(progress.tasks, [])

    def test_portion_progress_alignment(self) -> None:
        batches = [{cflearn.INPUT_KEY: torch.tensor([[float(i)]])} for i in range(3)]
        cases = [
            (-1.0, 0),
            (0.0, 0),
            (0.5, 2),
            (2.0, 3),
            (float("nan"), 3),
        ]
        _, model = _make_loss_model()
        inference = cflearn.Inference(model=model)
        for portion, expected in cases:
            with self.subTest(portion=portion):
                loader = TorchDataLoader(batches, batch_size=None)
                progress = Progress(auto_refresh=False, disable=True)
                with ExitStack() as stack:
                    model_step = stack.enter_context(
                        patch.object(model, "step", wraps=model.step)
                    )
                    add_task = stack.enter_context(
                        patch.object(progress, "add_task", wraps=progress.add_task)
                    )
                    advance_progress = stack.enter_context(
                        patch.object(progress, "advance", wraps=progress.advance)
                    )
                    remove_task = stack.enter_context(
                        patch.object(
                            progress,
                            "remove_task",
                            wraps=progress.remove_task,
                        )
                    )
                    stop_progress = stack.enter_context(
                        patch.object(progress, "stop", wraps=progress.stop)
                    )
                    outputs = inference.get_outputs(
                        loader,  # type: ignore
                        portion=portion,
                        recover_labels=False,
                        recover_predictions=False,
                        progress=progress,
                    )

                self.assertEqual(model_step.call_count, expected)
                add_task.assert_called_once()
                self.assertEqual(add_task.call_args.kwargs["total"], expected)
                remove_task.assert_called_once()
                progress_task = remove_task.call_args.args[0]
                self.assertListEqual(
                    [call.args for call in advance_progress.call_args_list],
                    [(progress_task,)] * expected,
                )
                stop_progress.assert_called_once_with()
                self.assertListEqual(progress.tasks, [])
                if expected == 0:
                    self.assertDictEqual(outputs.forward_results, {})
                else:
                    predictions = outputs.forward_results[cflearn.PREDICTIONS_KEY]
                    np.testing.assert_array_equal(
                        predictions,
                        np.arange(expected, dtype=np.float32)[..., None],
                    )

    def test_uneven_padded_batches(self) -> None:
        _, model = _make_loss_model()
        inference = cflearn.Inference(model=model)

        def inference_with(batches, **kwargs):
            return inference.get_outputs(
                TorchDataLoader(batches, batch_size=None),  # type: ignore
                recover_labels=False,
                recover_predictions=False,
                **kwargs,
            )

        plain = inference_with(
            [
                {cflearn.INPUT_KEY: torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
                {cflearn.INPUT_KEY: torch.tensor([[5.0, 6.0]])},
            ]
        )
        np.testing.assert_array_equal(
            plain.forward_results[cflearn.PREDICTIONS_KEY],
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        )

        padded = inference_with(
            [
                {cflearn.INPUT_KEY: torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
                {cflearn.INPUT_KEY: torch.tensor([[5.0, 6.0, 7.0]])},
            ],
            pad_dim=1,
            verbose=False,
        )
        np.testing.assert_allclose(
            padded.forward_results[cflearn.PREDICTIONS_KEY],
            np.array(
                [
                    [1.0, 2.0, np.nan],
                    [3.0, 4.0, np.nan],
                    [5.0, 6.0, 7.0],
                ]
            ),
            equal_nan=True,
        )

    def test_non_tensor_outputs(self) -> None:
        batches = [
            {cflearn.INPUT_KEY: torch.tensor([[1.0]])},
            {cflearn.INPUT_KEY: torch.tensor([[2.0]])},
        ]
        loader = TorchDataLoader(batches, batch_size=None)
        _, model = _make_loss_model()

        def inject_outputs(tensor_batch, tensor_outputs) -> None:
            value = tensor_outputs[cflearn.PREDICTIONS_KEY]
            batch_idx = int(tensor_batch[cflearn.INPUT_KEY].item()) - 1
            tensor_outputs["tag"] = f"batch-{batch_idx}"
            tensor_outputs["metadata"] = {
                "tensor": value + 10.0,
                "items": [value + 20.0],
                "pair": (value + 30.0, f"item-{batch_idx}"),
            }

        outputs = cflearn.Inference(model=model).get_outputs(
            loader,  # type: ignore
            target_outputs=[cflearn.PREDICTIONS_KEY, "tag", "metadata"],
            recover_labels=False,
            recover_predictions=False,
            inject_outputs_fn=inject_outputs,
        )

        np.testing.assert_array_equal(
            outputs.forward_results[cflearn.PREDICTIONS_KEY],
            np.array([[1.0], [2.0]]),
        )
        self.assertListEqual(outputs.forward_results["tag"], ["batch-0", "batch-1"])
        metadata = outputs.forward_results["metadata"]
        self.assertEqual(len(metadata), 2)
        for i, item in enumerate(metadata):
            value = float(i + 1)
            self.assertEqual(item["tensor"].device.type, "cpu")
            self.assertEqual(item["tensor"].item(), value + 10.0)
            self.assertEqual(item["items"][0].device.type, "cpu")
            self.assertEqual(item["items"][0].item(), value + 20.0)
            self.assertIsInstance(item["pair"], tuple)
            self.assertEqual(item["pair"][0].device.type, "cpu")
            self.assertEqual(item["pair"][0].item(), value + 30.0)
            self.assertEqual(item["pair"][1], f"item-{i}")

    def test_raw_outputs_with_labels(self) -> None:
        batches = [
            {
                cflearn.INPUT_KEY: torch.tensor([[1.0], [2.0]]),
                cflearn.LABEL_KEY: torch.tensor([[3.0], [4.0]]),
            },
            {
                cflearn.INPUT_KEY: torch.tensor([[5.0]]),
                cflearn.LABEL_KEY: torch.tensor([[6.0]]),
            },
        ]
        loader = TorchDataLoader(batches, batch_size=None)
        _, model = _make_loss_model()

        outputs = cflearn.Inference(model=model).get_outputs(
            loader,  # type: ignore
            return_labels=True,
            concat_outputs=False,
            recover_labels=False,
            recover_predictions=False,
        )

        predictions = outputs.forward_results[cflearn.PREDICTIONS_KEY]
        self.assertEqual(len(predictions), 2)
        self.assertTupleEqual(predictions[0].shape, (2, 1))
        self.assertTupleEqual(predictions[1].shape, (1, 1))
        np.testing.assert_array_equal(
            outputs.labels[cflearn.LABEL_KEY],
            np.array([[3.0], [4.0], [6.0]]),
        )

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
        loader = TorchDataLoader(batches, batch_size=None)
        _, model = _make_loss_model()
        inference = cflearn.Inference(model=model)

        outputs = inference.get_outputs(
            loader,  # type: ignore
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

        mixed_outputs = inference.get_outputs(
            loader,  # type: ignore
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

        accelerator_outputs = inference.get_outputs(
            loader,
            metrics=metrics,
            accelerator=Accelerator(cpu=True),
        )
        self.assertAlmostEqual(
            accelerator_outputs.metric_outputs.final_score,
            final_score,
        )

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
        metadata_loader = TorchDataLoader(batches, batch_size=None)
        metadata_outputs = inference.get_outputs(
            metadata_loader,  # type: ignore
            metrics=MetadataMetric(),
            target_inputs=[cflearn.INPUT_KEY, "metadata"],
            recover_labels=False,
            recover_predictions=False,
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
            TorchDataLoader(batches, batch_size=None),  # type: ignore
            recover_labels=False,
            recover_predictions=False,
        )

        self.assertIsNotNone(outputs.loss_items)
        self.assertEqual(outputs.loss_items[cflearn.LOSS_KEY], 4.5)

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
        progress = Progress(auto_refresh=False, disable=True)

        with patch.object(
            progress,
            "stop",
            side_effect=RuntimeError("progress cleanup failed"),
        ) as stop_progress:
            with patch.object(
                progress,
                "remove_task",
                side_effect=RuntimeError("task cleanup failed"),
            ) as remove_task:
                with self.assertRaises(RuntimeError) as context:
                    inference.get_outputs(loader, progress=progress)

        self.assertIs(context.exception, inference_error)
        self.assertListEqual(run_states, [(False, False, False, True)])
        self.assertTrue(model.m.training)
        self.assertTrue(model.loss.training)
        self.assertFalse(inference.use_grad_in_predict)
        stop_progress.assert_called_once_with()
        self.assertEqual(len(progress.tasks), 1)
        task_id = progress.tasks[0].id
        remove_task.assert_called_once_with(task_id)
        progress.remove_task(task_id)
        progress.stop()

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
        with self.assertRaises(ValueError):
            inference.get_outputs(loader, pad_dim=2)
        o0 = inference.get_outputs(loader, pad_dim=1)
        o1 = inference.get_outputs(loader, pad_dim=1, accelerator=Accelerator())
        o2 = inference.get_outputs(
            loader,
            pad_dim={cflearn.PREDICTIONS_KEY: 1},
            accelerator=Accelerator(),
        )
        o3 = inference.get_outputs(loader, pad_dim=-1, verbose=False)
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
        np.testing.assert_array_equal(o3.forward_results[cflearn.PREDICTIONS_KEY], gt)

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
