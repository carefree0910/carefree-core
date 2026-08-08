import os
import copy
import torch
import tempfile
import unittest

import numpy as np
import core.learn as cflearn
import core.learn.schema as learn_schema

from core.learn.schema import *
from enum import Enum
from types import SimpleNamespace
from typing import Any
from typing import Dict
from contextlib import nullcontext
from unittest.mock import patch
from unittest.mock import MagicMock


class TestSchema(unittest.TestCase):
    def test_functions(self):
        class Foo(Enum):
            A = 1
            B = 2

        sws = np.random.random([3, 5]), np.random.random([3, 5])
        tsw, vsw = split_sw(sws)
        np.testing.assert_allclose(tsw, norm_sw(sws[0]))
        np.testing.assert_allclose(vsw, norm_sw(sws[1]))
        self.assertFalse(check_data_is_info(Foo.A))
        cfg = cflearn.TrainerConfig()
        self.assertEqual(weighted_loss_score(cfg, {}), 0.0)
        self.assertEqual(weighted_loss_score(cfg, {"foo": 1.0, "bar": 2.0}), -1.5)
        cfg.loss_metrics_weights = dict(foo=1.0, bar=2.0)
        self.assertEqual(weighted_loss_score(cfg, {}), 0.0)
        self.assertEqual(weighted_loss_score(cfg, {"foo": 1.0, "bar": 2.0}), -5)

    def test_data_loader(self):
        class MockArray:
            def __init__(self, value):
                self.value = value

            def __len__(self):
                return 1

            def __getitem__(self, item):
                return self.value

        data = cflearn.testing.linear_data(3)[0]
        x, y = data.bundle.x_train, data.bundle.y_train
        loader = data.build_loader(x, y)
        loader.get_full_batch("cpu")
        x = dict(a=MockArray(0), b=MockArray([]))
        data = cflearn.ArrayDictData.init().fit(x)
        loader = data.build_loaders()[0]
        loader.get_input_sample()
        loader.dataset[0]
        data = cflearn.ArrayDictData.init()
        with self.assertRaises(RuntimeError):
            data.build_loaders()
        data.is_ready = True
        with self.assertRaises(RuntimeError):
            data.build_loaders()
        data.from_npd(dict(x=dict(a=MockArray(0))))
        data_config = cflearn.DataConfig(valid_loader_configs=dict(num_workers=7))
        data = cflearn.ArrayDictData.init(data_config).fit(x, x_valid=x)
        loader = data.to_loader(
            data.to_datasets(cflearn.DataBundle(x, x_valid=x), for_inference=False)[1],
            shuffle=False,
            batch_size=1,
            for_inference=False,
            is_validation=True,
        )
        self.assertEqual(loader.num_workers, 7)

    def test_prepare_async_dataloader(self):
        class PreparedLoader:
            def __iter__(self):
                return iter(())

        base = SimpleNamespace()
        prepared = PreparedLoader()
        prepared.base_dataloader = base
        prepared.device = "cpu"
        prepared.rng_types = ["generator"]
        data = SimpleNamespace(config=SimpleNamespace(loader_seed_sync=False))
        loader = SimpleNamespace(
            data=data,
            for_inference=False,
            recover_labels=None,
            presend_device="cpu",
            async_prefetch=True,
            async_prefetch_factor=2,
        )
        accelerator = MagicMock()
        accelerator.prepare.return_value = prepared

        prepared_loaders = prepare_dataloaders(accelerator, loader)

        self.assertListEqual(prepared_loaders, [prepared])
        self.assertIsNone(prepared.device)
        self.assertIsNone(prepared.rng_types)
        self.assertIs(base.presend_device, loader.presend_device)
        self.assertTrue(base.async_prefetch)
        self.assertEqual(base.async_prefetch_factor, 2)

    def test_metric_contracts(self):
        class StreamMetric(IStreamMetric):
            def __init__(self):
                self.__identifier__ = "stream"

            @property
            def is_positive(self):
                return True

            @property
            def requires_all(self):
                return True

            def reset(self):
                pass

            def update(self, tensor_batch, tensor_outputs, loader=None):
                pass

            def finalize(self):
                return 1.0

        class EmptyMetric(IMetric):
            def __init__(self):
                self.__identifier__ = "empty"

            @property
            def is_positive(self):
                return True

            def forward(self, tensor_batch, tensor_outputs, loader=None):
                return 0.0

            def evaluate(self, tensor_batch, tensor_outputs, loader=None):
                return None

        stream = StreamMetric()
        with self.assertRaisesRegex(RuntimeError, "streaming metric"):
            stream.forward({}, {})
        with self.assertRaisesRegex(RuntimeError, "streaming metric"):
            stream.evaluate({}, {})
        with self.assertRaisesRegex(RuntimeError, "should not `requires_all`"):
            _ = MultipleMetrics([stream]).requires_all

        metrics = MultipleMetrics([EmptyMetric()])
        with self.assertRaisesRegex(RuntimeError, "should not return None"):
            metrics.evaluate({}, {})
        with self.assertRaisesRegex(RuntimeError, "no streaming metrics"):
            metrics.finalize()

    def test_inference_gather(self):
        class MockInference(IInference):
            def get_outputs(self, loader, **kwargs):
                raise NotImplementedError

        class MockAccelerator:
            def __init__(self, is_main_process):
                self.device = torch.device("cpu")
                self.num_processes = 2
                self.is_main_process = is_main_process
                self.pad_calls = []

            def pad_across_processes(self, tensors, dim):
                self.pad_calls.append((tensors, dim))
                return tensors

        inference = MockInference()
        tensors = {"x": torch.tensor([1.0])}
        gathered = inference.gather(None, tensors)
        self.assertEqual(len(gathered["x"]), 1)
        torch.testing.assert_close(gathered["x"][0], tensors["x"])

        def gather(value, gather_list, dst):
            self.assertEqual(dst, 0)
            if gather_list is not None:
                for target in gather_list:
                    target.copy_(value)

        accelerator = MockAccelerator(is_main_process=True)
        with patch.object(learn_schema.dist, "gather", side_effect=gather):
            gathered = inference.gather(accelerator, tensors, pad_dim=0)
        self.assertEqual(len(gathered["x"]), 2)
        self.assertEqual(accelerator.pad_calls, [(tensors, 0)])
        for tensor in gathered["x"]:
            torch.testing.assert_close(tensor, tensors["x"])

        tensors = {
            "plain": torch.tensor([1.0]),
            "padded": torch.tensor([2.0]),
        }
        accelerator = MockAccelerator(is_main_process=False)
        with patch.object(learn_schema.dist, "gather") as gather_mock:
            gathered = inference.gather(
                accelerator,
                tensors,
                pad_dim={"padded": 0},
            )
        self.assertDictEqual(gathered, {"plain": None, "padded": None})
        self.assertEqual(accelerator.pad_calls, [(tensors["padded"], 0)])
        self.assertEqual(gather_mock.call_count, 2)
        for call in gather_mock.call_args_list:
            self.assertIsNone(call.kwargs["gather_list"])

    def test_optimizer_hooks(self):
        class OptimizerHooks:
            def __init__(self):
                self.backward_args = None
                self.skip_args = None

            def get_backward_loss(self, state, loss_res, update):
                self.backward_args = state, loss_res, update
                return loss_res.loss * 2.0

            def will_skip_backward(self, state, update):
                self.skip_args = state, update
                return True

        class HookProxy:
            def __init__(self, hooks):
                self.hooks = hooks

            def __getattr__(self, name):
                return getattr(self.hooks, name)

        state = SimpleNamespace(step=1)
        loss_res = TrainStepLoss(
            torch.tensor(2.0),
            {cflearn.LOSS_KEY: torch.tensor(2.0)},
        )
        hooks = OptimizerHooks()
        optimizers = [
            hooks,
            SimpleNamespace(optimizer=hooks),
            SimpleNamespace(optimizer=HookProxy(hooks)),
        ]
        for optimizer in optimizers:
            backward_loss = get_backward_loss(optimizer, state, loss_res, True)
            torch.testing.assert_close(backward_loss, torch.tensor(4.0))
            self.assertEqual(hooks.backward_args, (state, loss_res, True))
            self.assertTrue(will_skip_backward(optimizer, state, False))
            self.assertEqual(hooks.skip_args, (state, False))

        optimizer = SimpleNamespace()
        self.assertIs(
            get_backward_loss(optimizer, state, loss_res, True), loss_res.loss
        )
        self.assertFalse(will_skip_backward(optimizer, state, False))

    def test_model_runtime_paths(self):
        config = cflearn.Config(
            module_name="linear",
            module_config=dict(input_dim=3, output_dim=2),
            loss_name="mse",
        )
        model = cflearn.IModel.from_config(config)
        batch = {cflearn.INPUT_KEY: torch.ones(2, 3)}

        def checkpoint(function, *args, **kwargs):
            self.assertFalse(kwargs.pop("use_reentrant"))
            return function(*args, **kwargs)

        with patch.object(
            learn_schema,
            "checkpoint",
            side_effect=checkpoint,
        ) as checkpoint_mock:
            forward = model(0, batch, use_checkpoint=True)
        self.assertEqual(forward.shape, (2, 2))
        checkpoint_mock.assert_called_once()

        first_parameter = next(model.m.parameters())
        first_parameter.requires_grad = False
        expected_names = [
            name
            for name, parameter in model.m.named_parameters()
            if parameter.requires_grad
        ]
        param_group = model.param_groups()[0]
        self.assertListEqual(param_group["names"], expected_names)
        self.assertListEqual(
            [id(parameter) for parameter in param_group["params"]],
            [
                id(parameter)
                for parameter in model.m.parameters()
                if parameter.requires_grad
            ],
        )

        def inject_outputs(tensor_batch, tensor_outputs):
            tensor_outputs["injected"] = tensor_batch[cflearn.INPUT_KEY].sum()

        outputs = model.step(0, batch, inject_outputs_fn=inject_outputs)
        self.assertIn("injected", outputs.forward_results)
        torch.testing.assert_close(
            outputs.forward_results["injected"],
            torch.tensor(6.0),
        )

    def test_train_forward_grad_and_closure_paths(self):
        class MockTrainStep(TrainStep):
            def __init__(
                self,
                *,
                skip=False,
                requires_new_forward=False,
                requires_grad_in_forward=True,
            ):
                super().__init__(
                    requires_new_forward=requires_new_forward,
                    requires_grad_in_forward=requires_grad_in_forward,
                    enable_toggle_optimizer=False,
                )
                self.skip = skip

            def should_skip(self, m, state):
                return self.skip

            def loss_fn(self, m, state, batch, forward_results, **kwargs):
                loss = forward_results[cflearn.PREDICTIONS_KEY].sum()
                return TrainStepLoss(loss, {cflearn.LOSS_KEY: loss[None]})

        class MockModel(IModel):
            def __init__(self):
                self.m = torch.nn.Linear(1, 1)
                self._train_steps = []
                self.num_forwards = 0

            @property
            def train_steps(self):
                return self._train_steps

            @property
            def all_modules(self):
                return [self.m]

            def build(self, config):
                pass

            def run(self, batch_idx, batch, state=None, **kwargs):
                self.num_forwards += 1
                return {cflearn.PREDICTIONS_KEY: self.m(batch[cflearn.INPUT_KEY])}

        def run(steps, skip_backward):
            model = MockModel()
            model._train_steps = steps
            optimizer = SimpleNamespace()
            trainer = SimpleNamespace(
                state=SimpleNamespace(step=1),
                config=cflearn.TrainerConfig(
                    grad_accumulate=1,
                    use_closure_pack=True,
                ),
                optimizers={"all": optimizer},
                callbacks=[],
            )
            closure_losses = []

            def update_fn(batch, forward, loss_fn, loss_res, optimizer, update):
                self.assertTrue(update)
                self.assertIsNotNone(loss_fn)
                closure_losses.append(loss_fn().loss)

            with patch.object(learn_schema, "get_update_fn", return_value=update_fn):
                with patch.object(
                    learn_schema,
                    "will_skip_backward",
                    side_effect=skip_backward,
                ):
                    with patch.object(
                        learn_schema,
                        "no_grad_context",
                        side_effect=lambda **kwargs: nullcontext(),
                    ) as no_grad:
                        outputs = model.train(
                            0,
                            {cflearn.INPUT_KEY: torch.ones(1, 1)},
                            trainer,
                            {},
                            {},
                        )
            self.assertTrue(closure_losses)
            self.assertIn(cflearn.LOSS_KEY, outputs.loss_tensors)
            return no_grad

        no_grad = run(
            [
                MockTrainStep(),
                MockTrainStep(skip=True),
            ],
            [True],
        )
        self.assertTrue(no_grad.call_args_list[0].kwargs["enabled"])

        no_grad = run(
            [
                MockTrainStep(),
                MockTrainStep(requires_new_forward=True),
            ],
            [True, False],
        )
        self.assertTrue(no_grad.call_args_list[0].kwargs["enabled"])

        no_grad = run(
            [
                MockTrainStep(),
                MockTrainStep(requires_grad_in_forward=True),
            ],
            [True, False],
        )
        self.assertFalse(no_grad.call_args_list[0].kwargs["enabled"])

    def test_data_config(self):
        config = cflearn.DataConfig()

        @cflearn.IDataBlock.register("foo", allow_duplicate=True)
        class FooDataBlock(cflearn.IDataBlock):
            def transform(self, bundle: DataBundle, for_inference: bool) -> DataBundle:
                return bundle

            def fit_transform(self, bundle: DataBundle) -> DataBundle:
                return bundle

            def to_info(self) -> Dict[str, Any]:
                return {}

        config.add_blocks(FooDataBlock)
        self.assertListEqual(config.block_names, ["foo"])
        config.add_blocks(FooDataBlock)
        self.assertListEqual(config.block_names, ["foo"])
        config.set_blocks(FooDataBlock, FooDataBlock)
        self.assertListEqual(config.block_names, ["foo"])
        foo_config = dict(a=1, b=2)
        config.block_configs = dict(foo=foo_config)
        foo_block = FooDataBlock()
        foo_block.build(config)
        self.assertTrue(foo_block.is_local_rank_0)
        self.assertIs(foo_block.configs, foo_config)

        config.block_configs = None
        self.assertDictEqual(foo_block.configs, {})
        self.assertIsNone(config.block_configs)

        empty_configs: Dict[str, Dict[str, Any]] = {}
        config.block_configs = empty_configs
        self.assertDictEqual(foo_block.configs, {})
        self.assertIs(config.block_configs, empty_configs)

        other_configs = {"other": {"a": 1}}
        config.block_configs = other_configs
        self.assertDictEqual(foo_block.configs, {})
        self.assertDictEqual(config.block_configs, {"other": {"a": 1}})

    def test_data_bundle(self):
        bundle = cflearn.DataBundle("foo")
        info = bundle.to_info()
        self.assertDictEqual(info, dict(x_train="foo"))
        self.assertDictEqual(info, cflearn.DataBundle.empty().from_info(info).to_info())
        bundle = cflearn.DataBundle(np.random.random([13, 31]))
        loaded = cflearn.DataBundle.empty().from_npd(bundle.to_npd())
        np.testing.assert_allclose(bundle.x_train, loaded.x_train)
        bundle = cflearn.DataBundle(torch.randn(13, 31))
        loaded = cflearn.DataBundle.empty().from_npd(bundle.to_npd())
        torch.testing.assert_close(bundle.x_train, loaded.x_train)
        bundle = cflearn.DataBundle(
            dict(
                a=dict(b=np.random.random([13, 31]), c=torch.randn(13, 31)),
                d=np.random.random([13, 31]),
                e=torch.randn(13, 31),
            ),
            "foo",
        )
        loaded = cflearn.DataBundle.empty().from_npd(bundle.to_npd())
        np.testing.assert_allclose(bundle.x_train["a"]["b"], loaded.x_train["a"]["b"])
        np.testing.assert_allclose(bundle.x_train["d"], loaded.x_train["d"])
        torch.testing.assert_close(bundle.x_train["a"]["c"], loaded.x_train["a"]["c"])
        torch.testing.assert_close(bundle.x_train["e"], loaded.x_train["e"])

        data = cflearn.testing.linear_data(3)[0]
        info = data.to_info()
        info["bundle"] = bundle.to_info()
        data.from_info(info)
        self.assertDictEqual(data.bundle.to_info(), dict(y_train="foo"))
        data.bundle = bundle
        loaded = data.from_npd(data.to_npd()).bundle
        np.testing.assert_allclose(bundle.x_train["a"]["b"], loaded.x_train["a"]["b"])
        np.testing.assert_allclose(bundle.x_train["d"], loaded.x_train["d"])
        torch.testing.assert_close(bundle.x_train["a"]["c"], loaded.x_train["a"]["c"])
        torch.testing.assert_close(bundle.x_train["e"], loaded.x_train["e"])

    def test_model(self):
        data, in_dim, out_dim, _ = cflearn.testing.linear_data()
        config = cflearn.Config(
            module_name="fcnn",
            module_config=dict(input_dim=in_dim, output_dim=out_dim),
            loss_name="mse",
        )
        m = cflearn.IModel.from_config(config)
        x, y = data.bundle.x_train, data.bundle.y_train
        loader = data.build_loader(x, y)
        inference = cflearn.Inference(model=m)
        o0 = inference.get_outputs(loader).forward_results[cflearn.PREDICTIONS_KEY]
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "model.pt")
            m.save(path)
            loaded = cflearn.IModel.load(path)
            inference = cflearn.Inference(model=loaded)
            o1 = inference.get_outputs(loader).forward_results[cflearn.PREDICTIONS_KEY]
            np.testing.assert_allclose(o0, o1)

    def test_config(self):
        with self.assertRaises(ValueError):
            cflearn.Config(mixed_precision=1)
        cflearn.Config(module_name="foo").sanity_check()
        with self.assertRaises(ValueError):
            cflearn.Config().sanity_check()
        with self.assertRaises(ValueError):
            cflearn.Config(mixed_precision=1)
        self.assertTrue(cflearn.Config().to_debug().is_debug)
        self.assertEqual(cflearn.Config().trainer_config, cflearn.TrainerConfig())
        legacy_config = cflearn.Config.construct({"callback_names": "foo"})
        self.assertListEqual(legacy_config.callback_names, ["foo"])
        self.assertListEqual(legacy_config.to_info()["callback_names"], ["foo"])
        config = cflearn.Config(mixed_precision=cflearn.PrecisionType.FP16)
        self.assertEqual(config.mixed_precision, "fp16")

    def test_partitioned_config(self):
        config = cflearn.Config(module_name="linear", num_epoch=2)
        sections = [
            config.runtime,
            config.distributed,
            config.build,
            config.optimization,
            config.evaluation,
            config.logging,
            config.persistence,
        ]
        for section in sections:
            self.assertIs(section, config)

        config.runtime.num_epoch = 3
        config.build.module_name = "fcnn"
        config.optimization.lr = 1.0e-3
        config.evaluation.valid_portion = 0.5
        self.assertEqual(config.num_epoch, 3)
        self.assertEqual(config.module_name, "fcnn")
        self.assertEqual(config.lr, 1.0e-3)
        self.assertEqual(config.valid_portion, 0.5)

        expected_keys = [
            "model",
            "model_config",
            "module_name",
            "module_config",
            "num_repeat",
            "loss_name",
            "loss_config",
            "in_loading",
            "cudnn_benchmark",
            "workspace",
            "create_sub_workspace",
            "state_config",
            "num_epoch",
            "num_steps",
            "log_steps",
            "valid_portion",
            "clip_norm",
            "grad_accumulate",
            "metric_names",
            "metric_configs",
            "metric_weights",
            "metric_forward_kwargs",
            "use_losses_as_metrics",
            "use_incrementer_for_train_losses_in_eval",
            "recompute_train_losses_in_eval",
            "loss_metrics_weights",
            "monitor_names",
            "monitor_configs",
            "auto_callback",
            "callback_names",
            "callback_configs",
            "lr",
            "optimizer_name",
            "scheduler_name",
            "optimizer_config",
            "scheduler_config",
            "use_closure_pack",
            "update_scheduler_per_epoch",
            "optimizer_settings",
            "use_zero",
            "sort_ckpt_by",
            "finetune_config",
            "resume_training_from",
            "tqdm_settings",
            "save_pipeline_in_realtime",
            "save_realtime_pipeline_individually",
            "profile",
            "profile_config",
            "profile_schedule_config",
            "split_batches",
            "mixed_precision",
            "dispatch_batches",
            "even_batches",
            "non_blocking",
            "find_unused_parameters",
            "timeout",
            "extra",
        ]
        self.assertListEqual(config.field_names, expected_keys)
        self.assertListEqual(list(config.to_info()), expected_keys)
        for name in [
            "runtime",
            "distributed",
            "build",
            "optimization",
            "evaluation",
            "logging",
            "persistence",
        ]:
            self.assertNotIn(name, config.to_info())

    def test_config_validation(self):
        invalid_configs = [
            {"num_epoch": 0},
            {"num_steps": -1},
            {"log_steps": 0},
            {"valid_portion": 0.0},
            {"valid_portion": 1.1},
            {"valid_portion": float("nan")},
            {"clip_norm": -1.0},
            {"clip_norm": float("inf")},
            {"grad_accumulate": 0},
            {"lr": -1.0},
            {"lr": float("inf")},
            {"sort_ckpt_by": "unknown"},
            {"mixed_precision": "unknown"},
            {"dispatch_batches": "unknown"},
            {"timeout": 0},
            {"num_repeat": 0},
        ]
        for kwargs in invalid_configs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                cflearn.Config(**kwargs)

        config = cflearn.Config(num_steps=0, clip_norm=0.0, lr=0.0)
        self.assertEqual(config.num_steps, 0)
        self.assertEqual(config.clip_norm, 0.0)
        self.assertEqual(config.lr, 0.0)

        array_config = {
            "warmup_steps": [1],
            "cycle_lengths": [10],
            "f_start": [0.0],
            "f_min": [0.0],
            "f_max": [1.0],
        }
        scheduler_config = {
            "op_type": "cosine_warmup",
            "op_config": array_config,
        }
        cflearn.Config(scheduler_name="op")
        cflearn.Config(
            scheduler_name="op",
            scheduler_config={"op_type": "custom"},
        )
        cflearn.Config(
            scheduler_name="op",
            scheduler_config=scheduler_config,
        )
        cflearn.Config(
            optimizer_settings={
                "empty": None,
                "ignored": {"scheduler": 1},
                "valid": {
                    "scheduler": "op",
                    "scheduler_config": {"op_type": "custom"},
                },
            }
        )
        with self.assertRaisesRegex(ValueError, "dictionary"):
            cflearn.Config(
                scheduler_name="op",
                scheduler_config={"op_type": "linear_warmup"},
            )
        with self.assertRaisesRegex(ValueError, "non-empty"):
            cflearn.Config(
                scheduler_name="op",
                scheduler_config={
                    "op_type": "linear_warmup",
                    "op_config": {**array_config, "warmup_steps": []},
                },
            )
        with self.assertRaisesRegex(ValueError, "same length"):
            cflearn.Config(
                scheduler_name="op",
                scheduler_config={
                    "op_type": "linear_warmup",
                    "op_config": {**array_config, "warmup_steps": [1, 2]},
                },
            )
        with self.assertRaisesRegex(ValueError, "same length"):
            cflearn.Config(
                optimizer_settings={
                    "all": {
                        "scheduler": "op",
                        "scheduler_config": {
                            "op_type": "cosine_warmup",
                            "op_config": {
                                **array_config,
                                "cycle_lengths": [10, 20],
                            },
                        },
                    }
                }
            )

    def test_legacy_config_payload(self):
        legacy_payload = {
            "workspace": "_legacy",
            "create_sub_workspace": False,
            "num_epoch": 3,
            "num_steps": 7,
            "log_steps": 2,
            "valid_portion": 0.5,
            "grad_accumulate": 2,
            "metric_names": ["mae"],
            "monitor_names": "basic",
            "callback_names": "legacy.callback",
            "lr": 1.0e-3,
            "optimizer_name": "adam",
            "optimizer_config": {"weight_decay": 1.0e-2},
            "sort_ckpt_by": "latest",
            "resume_training_from": "legacy.ckpt",
            "tqdm_settings": {"use_tqdm": True},
            "split_batches": True,
            "mixed_precision": "fp16",
            "dispatch_batches": "false",
            "even_batches": False,
            "find_unused_parameters": True,
            "timeout": 120,
            "model": "direct",
            "model_config": {"state": "legacy"},
            "module_name": "linear",
            "module_config": {"input_dim": 2, "output_dim": 1},
            "num_repeat": 2,
            "loss_name": "mse",
            "loss_config": {"reduction": "mean"},
            "in_loading": True,
            "cudnn_benchmark": True,
            "extra": {"legacy": True},
        }
        original_payload = copy.deepcopy(legacy_payload)
        loaders = [
            lambda payload: cflearn.Config(**payload),
            cflearn.Config.construct,
            lambda payload: cflearn.Config().from_info(payload),
            lambda payload: cflearn.Config.from_pack(
                {"type": "$base", "info": payload}
            ),
        ]

        self.assertIs(cflearn.Config.get("$base"), cflearn.Config)
        for load in loaders:
            with self.subTest(load=load):
                config = load(legacy_payload)
                self.assertDictEqual(legacy_payload, original_payload)
                for key, expected in legacy_payload.items():
                    actual = getattr(config, key)
                    if key == "callback_names":
                        self.assertListEqual(actual, ["legacy.callback"])
                    elif key == "dispatch_batches":
                        self.assertFalse(actual)
                    else:
                        self.assertEqual(actual, expected)
                self.assertEqual(config.sort_ckpt_by, cflearn.SortMethod.LATEST)

                info = config.to_info()
                expected_info = copy.deepcopy(legacy_payload)
                expected_info["callback_names"] = ["legacy.callback"]
                expected_info["dispatch_batches"] = False
                for key, expected in expected_info.items():
                    self.assertEqual(info[key], expected)

    def test_trainer_state(self):
        state = TrainerState(
            num_epoch=1,
            batch_size=2,
            loader_length=10,
            min_num_sample=9,
            enable_logging=False,
            min_snapshot_epoch_gap=2,
            manual_snapshot_epochs=[1],
        )
        self.assertEqual(state.snapshot_start_step, 5)
        self.assertEqual(state.num_step_per_snapshot, 5)
        self.assertSetEqual(state.manual_snapshot_steps, {10})
        self.assertFalse(state.should_terminate)
        self.assertFalse(state.should_log_losses)
        self.assertFalse(state.should_log_metrics_msg)
        self.assertFalse(state.can_snapshot)
        self.assertEqual(state.last_step, 0)
        state.step = 7
        state.set_terminate()
        self.assertEqual(state.last_step, 7)
        state.epoch = -1
        self.assertTrue(state.can_snapshot)


if __name__ == "__main__":
    unittest.main()
