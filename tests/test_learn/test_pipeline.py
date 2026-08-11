import os
import sys
import json
import torch
import pytest
import inspect
import unittest
import threading
import concurrent.futures

import numpy as np
import torch.nn as nn
import core.learn as cflearn
import torch.nn.functional as F

from pathlib import Path
from accelerate import Accelerator
from unittest.mock import patch
from core.learn.schema import losses_type
from core.toolkit.misc import random_hash
from core.toolkit.misc import get_latest_workspace
from core.toolkit.misc import DDPInfo
from core.learn.pipeline.blocks.basic import StateInfo
from core.learn.pipeline.blocks.basic import OptimizerSettings


class TestPipeline(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _prepare_tmp_path(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path

    def _workspace(self, name: str) -> str:
        return str(self.tmp_path / name)

    def _inference_pipeline(self, config=None):
        if config is None:
            config = cflearn.Config(
                module_name="linear",
                module_config={"input_dim": 2, "output_dim": 1},
                loss_name="mse",
            )
        return cflearn.InferencePipeline.build_with(config)

    def _save_fusion_workspace(self, name, config=None):
        workspace = self._workspace(name)
        pipeline = self._inference_pipeline(config)
        cflearn.PipelineSerializer.save(pipeline, workspace, verbose=False)
        checkpoint_folder = (
            Path(workspace)
            / cflearn.PipelineSerializer.pipeline_folder
            / cflearn.SerializeModelBlock.__identifier__
        )
        checkpoint = cflearn.get_sorted_checkpoints(checkpoint_folder)[0]
        return workspace, checkpoint_folder / checkpoint

    def test_verbose_context_restores_after_exception(self):
        pipeline = self._inference_pipeline()
        block = pipeline.serialize_model
        self.assertIsNotNone(block)
        block.verbose = True

        with self.assertRaisesRegex(RuntimeError, "intentional context failure"):
            with pipeline.verbose_context(False):
                self.assertFalse(block.verbose)
                raise RuntimeError("intentional context failure")

        self.assertTrue(block.verbose)
        with pipeline.verbose_context(False):
            self.assertFalse(block.verbose)
        self.assertTrue(block.verbose)

    def test_verbose_context_keeps_outer_override(self):
        pipeline = self._inference_pipeline()
        block = pipeline.serialize_model
        self.assertIsNotNone(block)
        block.verbose = True

        with pipeline.verbose_context(False):
            self.assertFalse(block.verbose)
            with pipeline.verbose_context(True):
                self.assertFalse(block.verbose)
            self.assertFalse(block.verbose)
        self.assertTrue(block.verbose)

    def test_verbose_context_isolates_nested_pipelines(self):
        first = self._inference_pipeline()
        second = self._inference_pipeline()
        first_block = first.serialize_model
        second_block = second.serialize_model
        self.assertIsNotNone(first_block)
        self.assertIsNotNone(second_block)
        first_block.verbose = second_block.verbose = True

        with first.verbose_context(False):
            self.assertFalse(first_block.verbose)
            with second.verbose_context(False):
                self.assertFalse(second_block.verbose)
            self.assertFalse(first_block.verbose)
        self.assertTrue(first_block.verbose)
        self.assertTrue(second_block.verbose)

    def test_verbose_context_isolates_concurrent_pipelines(self):
        first = self._inference_pipeline()
        second = self._inference_pipeline()
        first_block = first.serialize_model
        second_block = second.serialize_model
        self.assertIsNotNone(first_block)
        self.assertIsNotNone(second_block)
        first_block.verbose = second_block.verbose = True
        first_entered = threading.Event()
        release_first = threading.Event()

        def run_first():
            with first.verbose_context(False):
                first_entered.set()
                if not release_first.wait(5.0):
                    raise TimeoutError("second context did not finish")
                return first_block.verbose

        def run_second():
            if not first_entered.wait(5.0):
                raise TimeoutError("first context did not start")
            try:
                with second.verbose_context(False):
                    return second_block.verbose
            finally:
                release_first.set()

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(run_first)
            second_future = executor.submit(run_second)
            observed = first_future.result(), second_future.result()

        self.assertEqual(observed, (False, False))
        self.assertTrue(first_block.verbose)
        self.assertTrue(second_block.verbose)

    def test_basics(self):
        def build_pipeline(in_dim, out_dim):
            data, *_ = cflearn.testing.linear_data(dim=in_dim, out_dim=out_dim)
            config = cflearn.Config(
                workspace=self._workspace("basics"),
                module_name="fcnn",
                module_config=dict(input_dim=in_dim, output_dim=out_dim),
                loss_name="mse",
            )
            cflearn.TrainingPipeline.init(config).fit(data, only_touch=True)
            config.to_debug()
            p = cflearn.TrainingPipeline.init(config).fit(data)
            return p, data, config

        cflearn.Pipeline().run(None)
        p, data, _ = build_pipeline(11, 2)
        x, y = data.bundle.x_train, data.bundle.y_train
        test_loader = data.build_loader(x, y)
        p.predict(test_loader, return_classes=True)
        p, data, _ = build_pipeline(11, 3)
        x, y = data.bundle.x_train, data.bundle.y_train
        test_loader = data.build_loader(x, y)
        p.predict(test_loader, return_classes=True)
        p0, data, config = build_pipeline(10, 1)
        self.assertEqual(p0.device.type, "cpu")
        states = p0.build_model.model.state_dict()
        p1 = cflearn.InferencePipeline.build_with(config, states)
        x, y = data.bundle.x_train, data.bundle.y_train
        test_loader = data.build_loader(x, y)
        r0 = p0.predict(test_loader)[cflearn.PREDICTIONS_KEY]
        r1 = p1.predict(test_loader)[cflearn.PREDICTIONS_KEY]
        np.testing.assert_allclose(r0, r1)
        # function callings
        p0.to("cpu")
        with self.assertRaises(RuntimeError):
            cflearn.TrainingPipeline.init(config).predict(test_loader)
        with self.assertRaises(ValueError):
            p0.predict(test_loader, return_classes=True, return_probabilities=True)
        malformed_outputs = cflearn.InferenceOutputs(
            forward_results={cflearn.PREDICTIONS_KEY: []},
            labels={},
            metric_outputs=None,
            loss_items=None,
        )
        with patch.object(
            p0.build_inference.inference,
            "get_outputs",
            return_value=malformed_outputs,
        ):
            with self.assertRaisesRegex(RuntimeError, "should be concatenated"):
                p0.predict(test_loader, return_classes=True)
        p0.predict(test_loader, return_classes=True)
        p0.predict(test_loader, return_probabilities=True)

        class FooPipeline(cflearn.TrainingPipeline):
            @property
            def building_blocks(self):
                return [cflearn.BuildModelBlock(), cflearn.SerializeModelBlock()]

        original_workspace = config.workspace
        get_random_workspace = lambda: os.path.join(original_workspace, random_hash())
        random_workspace = get_random_workspace()
        config.workspace = random_workspace
        p = FooPipeline.init(config).fit(data)
        block = p.get_block(cflearn.SerializeModelBlock)
        block.save_extra(random_workspace)
        block.ckpt_folder = random_workspace
        block.ckpt_scores = cflearn.get_scores(random_workspace)
        block.training_workspace = None
        block.save_extra(get_random_workspace())
        (Path(random_workspace) / sorted(block.ckpt_scores)[0]).unlink()
        block.save_extra(get_random_workspace())

    def test_prediction_postprocessing(self):
        config = cflearn.Config(
            module_name="linear",
            module_config={"input_dim": 2, "output_dim": 1},
            loss_name="mse",
        )
        pipeline = cflearn.InferencePipeline.build_with(config)
        data, *_ = cflearn.testing.arange_data(n=1, dim=2, batch_size=1)
        loader = data.build_loader(data.bundle.x_train, batch_size=1)
        preserved = torch.ones(1)

        def predict(predictions, **kwargs):
            outputs = cflearn.InferenceOutputs(
                forward_results={
                    cflearn.PREDICTIONS_KEY: predictions,
                    "preserved": preserved,
                },
                labels={},
                metric_outputs=None,
                loss_items=None,
            )
            with patch.object(
                pipeline.build_inference.inference,
                "get_outputs",
                return_value=outputs,
            ):
                results = pipeline.predict(loader, **kwargs)
            self.assertIs(results["preserved"], preserved)
            return results[cflearn.PREDICTIONS_KEY]

        # No postprocessing means no mode / axis validation and no list scan.
        raw_predictions = []
        raw = predict(
            raw_predictions,
            prediction_mode="unknown",
            class_dim=100,
            binary_threshold=float("nan"),
        )
        self.assertIs(raw, raw_predictions)

        # Rank-1 and one-logit binary predictions both expand to two probabilities.
        rank_1 = torch.tensor([0.0, np.log(3.0)], dtype=torch.float32)
        rank_1_probabilities = predict(rank_1, return_probabilities=True)
        torch.testing.assert_close(
            rank_1_probabilities,
            torch.tensor([[0.5, 0.5], [0.25, 0.75]]),
        )
        self.assertEqual(rank_1_probabilities.dtype, rank_1.dtype)
        self.assertEqual(rank_1_probabilities.device, rank_1.device)

        one_logit = np.zeros((2, 1), dtype=np.float32)
        one_logit_probabilities = predict(one_logit, return_probabilities=True)
        np.testing.assert_array_equal(
            one_logit_probabilities,
            np.full((2, 2), 0.5, dtype=np.float32),
        )
        self.assertEqual(one_logit_probabilities.dtype, one_logit.dtype)

        two_logits = torch.tensor(
            [[0.0, 0.0], [0.0, np.log(3.0)]],
            dtype=torch.float32,
        )
        two_logit_probabilities = predict(two_logits, return_probabilities=True)
        torch.testing.assert_close(
            two_logit_probabilities,
            torch.tensor([[0.5, 0.5], [0.25, 0.75]]),
        )

        # Pipeline keeps its inclusive threshold behavior.
        binary_classes = predict(
            torch.zeros(2, 1),
            return_classes=True,
            binary_threshold=0.5,
        )
        torch.testing.assert_close(binary_classes, torch.ones(2, 1, dtype=torch.long))

        multiclass_logits = torch.tensor([[0.0, 1.0, 2.0]])
        multiclass_probabilities = predict(
            multiclass_logits,
            return_probabilities=True,
        )
        torch.testing.assert_close(
            multiclass_probabilities,
            torch.softmax(multiclass_logits, dim=1),
        )
        self.assertEqual(multiclass_probabilities.shape, multiclass_logits.shape)

        # Explicit modes disambiguate two-class and multilabel predictions.
        ambiguous_logits = torch.tensor([[-1.0, 0.0]])
        auto_classes = predict(
            ambiguous_logits,
            return_classes=True,
            binary_threshold=0.8,
        )
        multiclass_classes = predict(
            ambiguous_logits,
            return_classes=True,
            binary_threshold=float("nan"),
            prediction_mode="multiclass",
        )
        torch.testing.assert_close(auto_classes, torch.zeros(1, 1, dtype=torch.long))
        torch.testing.assert_close(
            multiclass_classes,
            torch.ones(1, 1, dtype=torch.long),
        )

        multilabel_logits = torch.tensor([[-1.0, 0.0, 1.0]])
        multilabel_probabilities = predict(
            multilabel_logits,
            return_probabilities=True,
            prediction_mode="multilabel",
            binary_threshold=float("nan"),
        )
        torch.testing.assert_close(
            multilabel_probabilities,
            torch.sigmoid(multilabel_logits),
        )
        multilabel_classes = predict(
            multilabel_logits,
            return_classes=True,
            prediction_mode="multilabel",
        )
        torch.testing.assert_close(
            multilabel_classes,
            torch.tensor([[0, 1, 1]]),
        )

        # Class axes work for both channel-first and channel-last spatial logits.
        nchw_logits = torch.tensor([0.0, 1.0, 2.0]).reshape(1, 3, 1, 1)
        nchw_classes = predict(nchw_logits, return_classes=True)
        torch.testing.assert_close(
            nchw_classes,
            torch.tensor([[[[2]]]]),
        )
        self.assertEqual(nchw_classes.shape, (1, 1, 1, 1))

        nhwc_logits = np.arange(6.0, dtype=np.float32).reshape(1, 1, 2, 3)
        nhwc_probabilities = predict(
            nhwc_logits,
            return_probabilities=True,
            class_dim=-1,
        )
        np.testing.assert_allclose(nhwc_probabilities.sum(axis=-1), 1.0)
        self.assertEqual(nhwc_probabilities.shape, nhwc_logits.shape)

        # Postprocessing happens before the callback, and no other result is replaced.
        callback_predictions = []

        def callback(results):
            callback_predictions.append(results[cflearn.PREDICTIONS_KEY])
            results["callback"] = True
            return results

        with patch.object(pipeline, "predict_callback", side_effect=callback):
            callback_classes = predict(torch.zeros(1, 1), return_classes=True)
        self.assertIs(callback_predictions[0], callback_classes)

        # Flag conflicts are still reported only after inference has run.
        outputs = cflearn.InferenceOutputs(
            forward_results={
                cflearn.PREDICTIONS_KEY: torch.zeros(1, 1),
            },
            labels={},
            metric_outputs=None,
            loss_items=None,
        )
        with patch.object(
            pipeline.build_inference.inference,
            "get_outputs",
            return_value=outputs,
        ) as get_outputs:
            with self.assertRaises(ValueError):
                pipeline.predict(
                    loader,
                    return_classes=True,
                    return_probabilities=True,
                )
        get_outputs.assert_called_once()

    def test_prepare_distributed_with(self):
        data, in_dim, out_dim = cflearn.testing.arange_data(
            n=4,
            dim=2,
            out_dim=1,
            batch_size=4,
        )
        config = cflearn.Config(
            module_name="linear",
            module_config={"input_dim": in_dim, "output_dim": out_dim},
            loss_name="mse",
        )
        pipeline = cflearn.InferencePipeline.build_with(config)
        build_model = pipeline.build_model
        inference = pipeline.build_inference.inference
        original_model = build_model.model
        original_modules = original_model.all_modules
        runtime_state = object()
        original_model.runtime_state = runtime_state
        loader = data.build_loader(data.bundle.x_train, batch_size=4)
        predictions_before = pipeline.predict(
            loader,
            recover_predictions=False,
        )[cflearn.PREDICTIONS_KEY]
        accelerator = Accelerator(cpu=True)

        with patch.object(
            accelerator,
            "prepare",
            wraps=accelerator.prepare,
        ) as prepare:
            with patch.object(
                original_model,
                "from_accelerator",
                side_effect=RuntimeError("failed to construct prepared model"),
            ):
                with self.assertRaisesRegex(RuntimeError, "failed to construct"):
                    pipeline.prepare_distributed_with(accelerator)
            returned = pipeline.prepare_distributed_with(accelerator)
        self.assertIs(build_model.model, original_model)
        self.assertIs(inference.model, original_model)

        prepared_model = build_model.model
        self.assertIsNone(returned)
        self.assertEqual(prepare.call_count, 2)
        for call in prepare.call_args_list:
            self.assertTupleEqual(call.args, tuple(original_modules))
        self.assertIs(prepared_model, original_model)
        self.assertIs(prepared_model.runtime_state, runtime_state)
        self.assertIs(inference.model, prepared_model)
        predictions_after = pipeline.predict(loader, recover_predictions=False)[
            cflearn.PREDICTIONS_KEY
        ]
        torch.testing.assert_close(predictions_after, predictions_before)

        single_model = cflearn.EnsembleModel(original_model, 1)
        single_model.loss = None
        build_model.model = single_model
        inference.model = single_model
        single_module = single_model.m
        with patch.object(
            accelerator,
            "prepare",
            wraps=accelerator.prepare,
        ) as prepare:
            returned = pipeline.prepare_distributed_with(accelerator)
        self.assertIsNone(returned)
        prepare.assert_called_once_with(single_module)
        self.assertIs(build_model.model, single_model)
        self.assertIs(inference.model, single_model)

    def test_load_training(self):
        data, in_dim, out_dim, _ = cflearn.testing.linear_data()
        config = cflearn.Config(
            workspace=self._workspace("load_training"),
            module_name="fcnn",
            module_config=dict(input_dim=in_dim, output_dim=out_dim),
            loss_name="mse",
            scheduler_name="warmup",
        )
        config.to_debug()
        p0 = cflearn.TrainingPipeline.init(config).fit(data)
        with patch.dict(
            os.environ,
            {"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"},
        ):
            p1 = cflearn.PipelineSerializer.load_training(p0.config.workspace)
            p1.fit(data)

    def test_serializer(self):
        data, in_dim, out_dim, _ = cflearn.testing.linear_data()
        config = cflearn.Config(
            workspace=self._workspace("serializer"),
            module_name="linear",
            module_config=dict(input_dim=in_dim, output_dim=out_dim),
            loss_name="mse",
        )
        config.to_debug()
        p0 = cflearn.TrainingPipeline.init(config).fit(data)
        x, y = data.bundle.x_train, data.bundle.y_train
        test_loader = data.build_loader(x, y)
        r0 = p0.predict(test_loader)[cflearn.PREDICTIONS_KEY]
        save_workspace = self.tmp_path / "serialized"
        save_workspace.mkdir()
        cflearn.PipelineSerializer.save(p0, save_workspace, compress=True)
        p1 = cflearn.PipelineSerializer.load_inference(save_workspace)
        r1 = p1.predict(test_loader)[cflearn.PREDICTIONS_KEY]
        np.testing.assert_allclose(r0, r1)
        workspace = p0.config.workspace
        p1 = cflearn.PipelineSerializer.pack_and_load_inference(workspace)
        r1 = p1.predict(test_loader)[cflearn.PREDICTIONS_KEY]
        np.testing.assert_allclose(r0, r1)
        for pt in [cflearn.PackType.TRAINING, cflearn.PackType.EVALUATION]:
            export_folder = self.tmp_path / f"pack_{pt.value}"
            export_folder.mkdir()
            cflearn.PipelineSerializer.pack(workspace, export_folder, pack_type=pt)
        invalid_pack_folder = self.tmp_path / "invalid_pack"
        invalid_pack_folder.mkdir()
        with self.assertRaises(ValueError):
            cflearn.PipelineSerializer.pack(
                workspace,
                invalid_pack_folder,
                pack_type="bla",
            )
        op = self.tmp_path / "test.onnx"
        with self.assertRaises(ValueError):
            cflearn.PipelineSerializer.pack_onnx(workspace, str(op))
        cflearn.PipelineSerializer.pack_onnx(
            workspace,
            str(op),
            loader_sample=test_loader,
        )
        oi = cflearn.Inference(onnx=str(op))
        ro = oi.get_outputs(test_loader).forward_results[cflearn.PREDICTIONS_KEY]
        np.testing.assert_array_almost_equal(r0, ro)
        update_workspace = self.tmp_path / "update"
        update_workspace.mkdir()
        cflearn.PipelineSerializer.save(p1, update_workspace, compress=True)
        cflearn.PipelineSerializer.update(p1, update_workspace)
        p_folder = update_workspace / cflearn.PipelineSerializer.pipeline_folder
        cflearn.PipelineSerializer._load(p_folder)

        with self.assertRaises(ValueError):
            cflearn.PipelineSerializer._load(p_folder, focuses=[cflearn.TrainingBlock])
        p2 = cflearn.PipelineSerializer.load_inference(update_workspace)
        r2 = p2.predict(test_loader)[cflearn.PREDICTIONS_KEY]
        np.testing.assert_allclose(r1, r2)
        with self.assertRaises(ValueError):
            cflearn.PipelineSerializer.update(p1, self.tmp_path / "missing")

    def test_scripted_serializer_contract(self):
        pipeline = cflearn.InferencePipeline.build_with(
            cflearn.Config(
                module_name="linear",
                module_config={"input_dim": 1, "output_dim": 1},
                loss_name="mse",
            )
        )
        scripted = torch.jit.trace(nn.Linear(1, 1), torch.ones(1, 1))
        export_file = "model.pt"
        with patch.object(
            cflearn.PipelineSerializer,
            "pack_and_load_inference",
            return_value=pipeline,
        ) as pack_and_load, patch.object(
            torch.jit,
            "script",
            return_value=scripted,
        ) as script, patch.object(
            torch.jit,
            "save",
        ) as save:
            returned = cflearn.PipelineSerializer.pack_scripted(
                "workspace",
                export_file,
            )
        self.assertIs(returned, pipeline)
        pack_and_load.assert_called_once_with("workspace")
        script.assert_called_once_with(pipeline.build_model.model.m)
        save.assert_called_once_with(scripted, export_file)

    def test_scripted_serializer(self):
        data, in_dim, out_dim, _ = cflearn.testing.linear_data()
        config = cflearn.Config(
            workspace=self._workspace("scripted_serializer"),
            module_name="linear",
            module_config=dict(input_dim=in_dim, output_dim=out_dim),
            loss_name="mse",
        )
        config.to_debug()
        p0 = cflearn.TrainingPipeline.init(config).fit(data)
        x, y = data.bundle.x_train, data.bundle.y_train
        test_loader = data.build_loader(x, y)
        r0 = p0.predict(test_loader)[cflearn.PREDICTIONS_KEY]
        workspace = p0.config.workspace
        scripted_path = self.tmp_path / "test.pt"
        try:
            cflearn.PipelineSerializer.pack_scripted(workspace, str(scripted_path))
        except AttributeError as error:
            if sys.version_info >= (3, 14) and "__annotations__" in str(error):
                pytest.xfail(
                    "this PyTorch release's TorchScript is incompatible with Python 3.14"
                )
            raise
        p1 = cflearn.PipelineSerializer.pack_and_load_inference(workspace)
        p1.build_model.model.m = torch.jit.load(scripted_path)
        self.assertIsInstance(p1.build_model.model.m, torch.jit.ScriptModule)
        r1 = p1.predict(test_loader)[cflearn.PREDICTIONS_KEY]
        np.testing.assert_allclose(r0, r1)

    def test_fuse_rejects_empty_workspaces(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            cflearn.PipelineSerializer.fuse_inference([])

    def test_fuse_rejects_excessive_num_picked(self):
        first, _ = self._save_fusion_workspace("fuse_excessive_first")
        second, _ = self._save_fusion_workspace("fuse_excessive_second")

        with self.assertRaisesRegex(ValueError, "num_picked"):
            cflearn.PipelineSerializer.fuse_inference(
                [first, second],
                num_picked=3,
            )

    def test_fuse_rejects_incompatible_configs(self):
        common = {
            "input_dim": 2,
            "output_dim": 1,
            "hidden_units": [2],
        }
        first, _ = self._save_fusion_workspace(
            "fuse_config_first",
            cflearn.Config(
                module_name="fcnn",
                module_config={**common, "activation": "ReLU"},
                loss_name="mse",
            ),
        )
        second, _ = self._save_fusion_workspace(
            "fuse_config_second",
            cflearn.Config(
                module_name="fcnn",
                module_config={**common, "activation": "Tanh"},
                loss_name="mse",
            ),
        )

        with self.assertRaisesRegex(ValueError, "config"):
            cflearn.PipelineSerializer.fuse_inference([first, second])

    def test_fuse_rejects_incompatible_state_signatures(self):
        for mismatch in ["keys", "values", "shapes", "dtypes"]:
            first, _ = self._save_fusion_workspace(f"fuse_{mismatch}_first")
            second, checkpoint = self._save_fusion_workspace(f"fuse_{mismatch}_second")
            payload = torch.load(checkpoint, weights_only=False)
            states = payload["states"]
            key = next(
                k
                for k, value in states.items()
                if value.is_floating_point() and value.ndim > 0
            )
            if mismatch == "keys":
                states.pop(key)
            elif mismatch == "values":
                states[key] = None
            elif mismatch == "shapes":
                value = states[key]
                states[key] = value.new_zeros((value.shape[0] + 1, *value.shape[1:]))
            else:
                states[key] = states[key].double()
            torch.save(payload, checkpoint)

            with self.subTest(mismatch=mismatch):
                with self.assertRaisesRegex(ValueError, mismatch.rstrip("s")):
                    cflearn.PipelineSerializer.fuse_inference([first, second])
                if mismatch == "dtypes":
                    fused = cflearn.PipelineSerializer.fuse_inference(
                        [first, second],
                        states_callback=lambda _, state: {
                            name: value.float() if value.is_floating_point() else value
                            for name, value in state.items()
                        },
                    )
                    self.assertEqual(len(fused.build_model.model.m), 2)

    def test_fuse_reports_missing_checkpoint(self):
        first, _ = self._save_fusion_workspace("fuse_missing_first")
        second, checkpoint = self._save_fusion_workspace("fuse_missing_second")
        checkpoint.unlink()

        with self.assertRaises(FileNotFoundError) as context:
            cflearn.PipelineSerializer.fuse_inference([first, second])
        self.assertIn(str(checkpoint), str(context.exception))

        empty, checkpoint = self._save_fusion_workspace("fuse_empty_scores")
        (checkpoint.parent / cflearn.SCORES_FILE).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "no checkpoint"):
            cflearn.PipelineSerializer.fuse_inference([empty])

    def test_fuse_preserves_non_float_buffers_per_member(self):
        config = cflearn.Config(
            module_name="fcnn",
            module_config={
                "input_dim": 2,
                "output_dim": 1,
                "hidden_units": [2],
                "batch_norm": True,
            },
            loss_name="mse",
        )
        workspaces = []
        for name, counter in [("first", 3), ("second", 7)]:
            workspace, checkpoint = self._save_fusion_workspace(
                f"fuse_buffer_{name}",
                config,
            )
            payload = torch.load(checkpoint, weights_only=False)
            states = payload["states"]
            counter_key = next(
                key for key in states if key.endswith("num_batches_tracked")
            )
            states[counter_key].fill_(counter)
            torch.save(payload, checkpoint)
            workspaces.append(workspace)

        fused = cflearn.PipelineSerializer.fuse_inference(workspaces)
        counters = [
            value
            for key, value in fused.build_model.model.state_dict().items()
            if key.endswith("num_batches_tracked")
        ]
        self.assertEqual([counter.item() for counter in counters], [3, 7])
        self.assertTrue(all(counter.dtype == torch.int64 for counter in counters))

    def test_fuse(self):
        data, in_dim, out_dim, _ = cflearn.testing.linear_data()
        config = cflearn.Config(
            workspace=self._workspace("fuse"),
            module_name="linear",
            module_config=dict(input_dim=in_dim, output_dim=out_dim),
            loss_name="mse",
        )
        config.to_debug()
        p0 = cflearn.TrainingPipeline.init(config).fit(data)
        p1 = cflearn.TrainingPipeline.init(config).fit(data)
        x, y = data.bundle.x_train, data.bundle.y_train
        test_loader = data.build_loader(x, y)
        r0 = p0.predict(test_loader)[cflearn.PREDICTIONS_KEY]
        r1 = p1.predict(test_loader)[cflearn.PREDICTIONS_KEY]
        ws = [p0.config.workspace, p1.config.workspace]
        sc = lambda _, d: d
        pf = cflearn.PipelineSerializer.fuse_inference(ws, states_callback=sc)
        rf = pf.predict(test_loader)[cflearn.PREDICTIONS_KEY]
        np.testing.assert_array_almost_equal(0.5 * (r0 + r1), rf)
        p2 = cflearn.TrainingPipeline.init(config).fit(data)
        ws.append(p2.config.workspace)
        pf = cflearn.PipelineSerializer.fuse_evaluation(ws, num_picked=2)
        self.assertEqual(len(pf.build_model.model.m), 2)
        pf = cflearn.PipelineSerializer.fuse_evaluation(ws, num_picked=0.6)
        self.assertEqual(len(pf.build_model.model.m), 2)
        with self.assertRaises(ValueError):
            cflearn.PipelineSerializer.fuse_evaluation(ws, num_picked=0.1)
        with self.assertRaises(ValueError):
            cflearn.PipelineSerializer.fuse_evaluation(ws, num_picked=1.1)
        with self.assertRaises(ValueError):
            cflearn.PipelineSerializer._fuse_multiple(ws, cflearn.PackType.TRAINING)

    def test_fuse_ema(self):
        @cflearn.register_module("fcnn_ema")
        class _(cflearn.FCNN):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.ema = cflearn.EMA.hook(self, 0.71)

        data, in_dim, out_dim, _ = cflearn.testing.linear_data()
        config = cflearn.Config(
            workspace=self._workspace("fuse_ema"),
            module_name="fcnn_ema",
            module_config=dict(input_dim=in_dim, output_dim=out_dim),
            loss_name="mse",
        )
        config.to_debug().num_steps = 5
        p0 = cflearn.TrainingPipeline.init(config).fit(data)
        p1 = cflearn.TrainingPipeline.init(config).fit(data)
        x, y = data.bundle.x_train, data.bundle.y_train
        test_loader = data.build_loader(x, y)
        r0 = p0.predict(test_loader)[cflearn.PREDICTIONS_KEY]
        r1 = p1.predict(test_loader)[cflearn.PREDICTIONS_KEY]
        ws = [p0.config.workspace, p1.config.workspace]
        pf = cflearn.PipelineSerializer.fuse_inference(ws)
        rf = pf.predict(test_loader)[cflearn.PREDICTIONS_KEY]
        np.testing.assert_array_almost_equal(0.5 * (r0 + r1), rf)

    def test_self_ensemble(self):
        self_ensemble = cflearn.PipelineSerializer.self_ensemble_inference
        data, in_dim, out_dim, _ = cflearn.testing.linear_data()
        config = cflearn.Config(
            workspace=self._workspace("self_ensemble"),
            module_name="linear",
            module_config=dict(input_dim=in_dim, output_dim=out_dim),
            loss_name="mse",
        )
        config.to_debug().num_steps = 4
        p = cflearn.TrainingPipeline.init(config).fit(data)
        x, y = data.bundle.x_train, data.bundle.y_train
        test_loader = data.build_loader(x, y)
        workspace = p.config.workspace
        ckpt_folder = os.path.join(workspace, cflearn.CHECKPOINTS_FOLDER)
        r = 0.0
        for ckpt_file in cflearn.get_sorted_checkpoints(ckpt_folder)[:3]:
            ckpt_path = os.path.join(ckpt_folder, ckpt_file)
            states = torch.load(ckpt_path, weights_only=False)["states"]
            p.build_model.model.load_state_dict(states)
            r += p.predict(test_loader)[cflearn.PREDICTIONS_KEY]
        r /= 3
        pe = self_ensemble(3, workspace)
        re = pe.predict(test_loader)[cflearn.PREDICTIONS_KEY]
        np.testing.assert_array_almost_equal(r, re)
        states = {k: 0 for k in p.build_model.model.state_dict()}
        ckpt_files = cflearn.get_sorted_checkpoints(ckpt_folder, sort_by="latest")
        for ckpt_file in ckpt_files[:3]:
            i_ckpt_path = os.path.join(ckpt_folder, ckpt_file)
            i_states = torch.load(i_ckpt_path, weights_only=False)["states"]
            for k, v in i_states.items():
                states[k] += v
        for k in states:
            states[k] /= 3
        p.build_model.model.load_state_dict(states)
        r = p.predict(test_loader)[cflearn.PREDICTIONS_KEY]
        sc = lambda _, d: d
        pe = self_ensemble(
            3,
            workspace,
            ensemble_weights=True,
            states_callback=sc,
            sort_ckpt_by="latest",
        )
        re = pe.predict(test_loader)[cflearn.PREDICTIONS_KEY]
        np.testing.assert_array_almost_equal(r, re)
        cflearn.PipelineSerializer.self_ensemble_evaluation(4, workspace)
        with self.assertRaises(RuntimeError):
            cflearn.PipelineSerializer.self_ensemble_inference(5, workspace)
        with self.assertRaises(RuntimeError):
            cflearn.PipelineSerializer.self_ensemble_evaluation(5, workspace)

    def test_resume(self):
        events = []

        @cflearn.TrainerCallback.register("resume_contract", allow_duplicate=True)
        class ResumeContractCallback(cflearn.TrainerCallback):
            def before_loop(self, trainer):
                events.append(("before_loop", trainer.state.step, trainer.state.epoch))

            def at_step_start(self, batch, trainer):
                events.append(
                    ("at_step_start", trainer.state.step, trainer.state.epoch)
                )

            def after_loop(self, trainer):
                events.append(("after_loop", trainer.state.step, trainer.state.epoch))

            def finalize(self, trainer):
                events.append(
                    (
                        "finalize",
                        trainer.state.step,
                        trainer.state.epoch,
                        trainer.state.last_step,
                    )
                )

        cflearn.seed_everything(142857)
        resume_workspace = self._workspace("_resume")
        data, in_dim, out_dim, _ = cflearn.testing.linear_data(
            n=8,
            batch_size=4,
            use_validation=True,
        )
        config = cflearn.Config(
            workspace=resume_workspace,
            module_name="linear",
            module_config=dict(input_dim=in_dim, output_dim=out_dim, bias=False),
            loss_name="mse",
            num_epoch=2,
            callback_names=["resume_contract"],
            tqdm_settings=cflearn.TqdmSettings(use_tqdm=True),
        )
        config.to_debug()
        cflearn.TrainingPipeline.init(config).fit(data, do_summary=False)
        latest_workspace = get_latest_workspace(resume_workspace)
        assert latest_workspace is not None
        resume_from = latest_workspace / cflearn.PipelineSerializer.pipeline_folder
        state_path = (
            resume_from
            / cflearn.SerializeOptimizerBlock.__identifier__
            / cflearn.SerializeOptimizerBlock.state_file
        )
        with state_path.open("r") as f:
            saved_state = json.load(f)
        self.assertDictEqual(saved_state, {"step": 4, "epoch": 2})

        config.resume_training_from = str(resume_from)
        config.num_steps = saved_state["step"] + 1
        events.clear()
        pipeline = cflearn.TrainingPipeline.init(config).fit(data, do_summary=False)

        saved_step = saved_state["step"]
        saved_epoch = saved_state["epoch"]
        self.assertListEqual(
            events,
            [
                ("before_loop", saved_step, saved_epoch),
                ("at_step_start", saved_step, saved_epoch + 1),
                ("after_loop", saved_step + 1, saved_epoch + 1),
                ("finalize", -1, -1, saved_step + 1),
            ],
        )
        trainer = pipeline.training.build_trainer.trainer
        self.assertEqual(trainer.state.last_step, saved_step + 1)


class TestBlocks(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _prepare_tmp_path(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path

    def _workspace(self, name: str) -> str:
        return str(self.tmp_path / name)

    def test_prepare_workspace_uses_node_local_main(self):
        node_workspaces = [
            self._workspace("node_0"),
            None,
            self._workspace("node_1"),
            None,
        ]
        expected_workspaces = [
            node_workspaces[0],
            node_workspaces[0],
            node_workspaces[2],
            node_workspaces[2],
        ]
        local_ranks = [0, 1, 0, 1]
        for rank, (local_rank, expected) in enumerate(
            zip(local_ranks, expected_workspaces)
        ):
            with self.subTest(rank=rank, local_rank=local_rank):
                initial_workspace = self._workspace(f"rank_{rank}_initial")
                config = cflearn.Config(
                    workspace=initial_workspace,
                    create_sub_workspace=True,
                )
                block = cflearn.PrepareWorkspaceBlock()
                block.training_workspace = self._workspace("training")
                ddp_env = {
                    "RANK": str(rank),
                    "WORLD_SIZE": "4",
                    "LOCAL_RANK": str(local_rank),
                    "LOCAL_WORLD_SIZE": "2",
                }
                gathered_workspace = (
                    node_workspaces[rank] if local_rank == 0 else initial_workspace
                )

                def gather_workspaces(workspaces, workspace):
                    self.assertEqual(workspace, gathered_workspace)
                    workspaces[:] = node_workspaces

                with patch.dict(os.environ, ddp_env), patch(
                    "core.learn.pipeline.blocks.basic.is_dist_initialized",
                    return_value=True,
                ), patch(
                    "core.learn.pipeline.blocks.basic.prepare_workspace_from",
                    return_value=node_workspaces[rank],
                ) as mock_prepare, patch(
                    "torch.distributed.all_gather_object",
                    side_effect=gather_workspaces,
                ) as mock_gather:
                    block.build(config)

                self.assertEqual(config.persistence.workspace, expected)
                mock_gather.assert_called_once()
                if local_rank == 0:
                    mock_prepare.assert_called_once_with(block.training_workspace)
                    self.assertEqual(block._defaults["workspace"], expected)
                else:
                    mock_prepare.assert_not_called()
                    self.assertNotIn("workspace", block._defaults)

    def test_prepare_workspace_can_disable_sub_workspace_creation(self):
        workspace = self._workspace("no_sub_workspace")
        config = cflearn.Config(workspace=workspace, create_sub_workspace=False)
        block = cflearn.PrepareWorkspaceBlock()
        block.training_workspace = workspace
        ddp_env = {
            "RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_RANK": "0",
            "LOCAL_WORLD_SIZE": "1",
        }
        with patch.dict(os.environ, ddp_env), patch(
            "core.learn.pipeline.blocks.basic.is_dist_initialized",
            return_value=True,
        ), patch(
            "core.learn.pipeline.blocks.basic.prepare_workspace_from"
        ) as mock_prepare, patch(
            "torch.distributed.all_gather_object"
        ):
            block.build(config)

        mock_prepare.assert_not_called()
        self.assertEqual(config.persistence.workspace, workspace)
        self.assertNotIn("workspace", block._defaults)

    def test_basics(self):
        data, *_ = cflearn.testing.linear_data()
        config = cflearn.Config(workspace=self._workspace("basics"))
        block = cflearn.PrepareWorkspaceBlock()
        block2 = cflearn.SerializeDataBlock()
        block.training_workspace = self._workspace("_foo")
        ddp_info = DDPInfo(rank=1, world_size=2, local_rank=1)
        ddp_env = {
            "RANK": str(ddp_info.rank),
            "WORLD_SIZE": str(ddp_info.world_size),
            "LOCAL_RANK": str(ddp_info.local_rank),
        }
        with patch.dict(os.environ, ddp_env):
            self.assertFalse(block.is_local_rank_0)
        with patch.dict(os.environ, ddp_env), patch(
            "core.learn.pipeline.blocks.basic.is_dist_initialized"
        ) as mock_dist, patch("torch.distributed.all_gather_object"):
            mock_dist.return_value = True
            with patch("core.learn.pipeline.common.is_ddp") as mock_ddp, patch(
                "core.learn.pipeline.common.get_ddp_info"
            ) as mock_info:
                mock_ddp.return_value = True
                mock_info.return_value = ddp_info
                block.build(config)
                block2.save_extra(self._workspace("data"))
        block = cflearn.ExtractStateInfoBlock()
        block.data = None
        self.assertFalse(block.try_load(self._workspace("_bar")))
        with self.assertRaises(ValueError):
            block.from_scratch(config)
        with patch.dict(os.environ, ddp_env):
            block.data = data
            block.from_scratch(config)
        block = cflearn.BuildMonitorsBlock()
        block.build(config)
        block = cflearn.BuildCallbacksBlock()
        block.build(config)
        block = cflearn.ReportBlock()
        block.training_workspace = None
        block.run(None, None)
        block = cflearn.SerializeScriptBlock()
        with patch("core.learn.pipeline.blocks.basic.inspect") as mock_inspect:
            mock_inspect.currentframe.return_value = None
            block.save_extra(None)
            frame = inspect.currentframe()
            self.assertIsNotNone(frame)
            mock_inspect.currentframe.return_value = frame
            mock_inspect.getsource.return_value = "source = True\n"
            script_folder = self.tmp_path / "script"
            block.save_extra(script_folder)
            mock_inspect.getsource.side_effect = RuntimeError("source unavailable")
            with patch("core.learn.pipeline.blocks.basic.console.warn") as mock_warn:
                block.save_extra(self.tmp_path / "missing_source")
            mock_warn.assert_called_once_with(
                "failed to save source script: source unavailable"
            )
        self.assertEqual(
            (script_folder / block.script_file).read_text(),
            "source = True\n",
        )
        opt_pack = cflearn.OptimizerPack("all", "adamw")
        opt_settings = OptimizerSettings()
        opt_state_info = StateInfo(1, 1, 1, 1, 1)
        new_pack = opt_settings.update_opt_pack(opt_state_info, opt_pack)
        self.assertDictEqual(new_pack.optimizer_config, {"lr": opt_settings.lr})

    def test_set_default(self):
        data, in_dim, out_dim, _ = cflearn.testing.linear_data()
        config = cflearn.Config(
            workspace=self._workspace("set_default"),
            module_name="linear",
            module_config=dict(input_dim=in_dim, output_dim=out_dim),
        )
        config.to_debug()
        with self.assertRaises(ValueError):
            cflearn.TrainingPipeline.init(config).fit(data)

        @cflearn.register_loss("linear")
        class FooLinearLoss(cflearn.ILoss):
            def forward(self, forward_results, batch, state=None) -> losses_type:
                predictions = forward_results[cflearn.PREDICTIONS_KEY]
                labels = batch[cflearn.LABEL_KEY]
                return F.mse_loss(predictions, labels)

        p = cflearn.TrainingPipeline.init(config).fit(data)
        self.assertNotIn("linear", p.config.callback_names)

        @cflearn.TrainerCallback.register("linear")
        class FooLinearCallback(cflearn.TrainerCallback):
            pass

        p = cflearn.TrainingPipeline.init(config).fit(data)
        self.assertIn("linear", p.config.callback_names)

        block = cflearn.SetTrainerDefaultsBlock()
        wandb_config = cflearn.Config(
            module_name="linear",
            callback_names=["wandb"],
            callback_configs={},
        )
        block.build(wandb_config)
        configured_wandb = wandb_config.callback_configs["wandb"]
        self.assertListEqual(configured_wandb["tags"], ["linear"])
        self.assertEqual(configured_wandb["config"]["module_name"], "linear")
        self.assertIn("callback_configs.wandb.tags", block._defaults)
        self.assertIn("callback_configs.wandb.config", block._defaults)

        environ_workspace = self._workspace("_foo")
        cflearn.set_environ_workspace(environ_workspace)
        p = cflearn.TrainingPipeline.init(config).fit(data)
        self.assertEqual(Path(p.config.workspace).parent.name, "_foo")
        cflearn.unset_environ_workspace()

    def test_set_trainer_callback_defaults(self):
        training_loop_id = cflearn.TrainingLoopCallback.__identifier__
        progress_id = cflearn.ProgressCallback.__identifier__
        log_metrics_msg_id = cflearn.LogMetricsMsgCallback.__identifier__
        update_artifacts_id = cflearn.UpdateArtifactsCallback.__identifier__
        default_ids = [
            progress_id,
            log_metrics_msg_id,
            update_artifacts_id,
        ]
        cases = [
            (
                "auto disabled without callbacks",
                False,
                None,
                [training_loop_id],
                [training_loop_id],
            ),
            (
                "auto disabled with empty callbacks",
                False,
                [],
                [training_loop_id],
                [training_loop_id],
            ),
            (
                "auto disabled with custom callback",
                False,
                ["custom"],
                [training_loop_id, "custom"],
                [training_loop_id],
            ),
            (
                "auto disabled with training loop",
                False,
                [training_loop_id, "custom"],
                [training_loop_id, "custom"],
                [],
            ),
            (
                "auto enabled without callbacks",
                True,
                None,
                [
                    training_loop_id,
                    update_artifacts_id,
                    log_metrics_msg_id,
                    progress_id,
                ],
                [training_loop_id, *default_ids],
            ),
            (
                "auto enabled with all defaults",
                True,
                default_ids,
                [training_loop_id, *default_ids],
                [training_loop_id],
            ),
            (
                "duplicate training loops",
                True,
                [
                    "custom",
                    training_loop_id,
                    training_loop_id,
                    *default_ids,
                ],
                [training_loop_id, "custom", *default_ids],
                [],
            ),
        ]
        for (
            name,
            auto_callback,
            callback_names,
            expected_names,
            expected_additional,
        ) in cases:
            with self.subTest(name):
                block = cflearn.SetTrainerDefaultsBlock()
                config = cflearn.Config(
                    auto_callback=auto_callback,
                    callback_names=callback_names,
                )
                block.build(config)

                self.assertListEqual(config.callback_names, expected_names)
                self.assertEqual(config.callback_names.count(training_loop_id), 1)
                self.assertEqual(config.callback_names[0], training_loop_id)
                self.assertListEqual(
                    block._defaults.get("additional_callbacks", []),
                    expected_additional,
                )
                if not auto_callback:
                    self.assertTrue(
                        all(
                            callback_id not in config.callback_names
                            for callback_id in default_ids
                        )
                    )

    def test_set_trainer_progress_config(self):
        progress_id = cflearn.ProgressCallback.__identifier__
        log_metrics_msg_id = cflearn.LogMetricsMsgCallback.__identifier__
        update_artifacts_id = cflearn.UpdateArtifactsCallback.__identifier__
        callback_names = [
            progress_id,
            log_metrics_msg_id,
            update_artifacts_id,
        ]

        tqdm_settings = cflearn.TqdmSettings(
            use_step_tqdm=True,
            desc="from config",
        )
        config = cflearn.Config(
            auto_callback=True,
            callback_names=callback_names,
            tqdm_settings=tqdm_settings,
            callback_configs={
                progress_id: {
                    "settings": {"desc": "ignored"},
                    "tqdm_settings": {"desc": "ignored"},
                },
                "custom": {"preserved": "custom"},
            },
        )
        cflearn.SetTrainerDefaultsBlock().build(config)
        progress_config = config.callback_configs[progress_id]
        self.assertDictEqual(
            progress_config,
            {"settings": tqdm_settings.asdict()},
        )
        self.assertFalse(config.callback_configs[log_metrics_msg_id]["verbose"])
        self.assertDictEqual(
            config.callback_configs["custom"],
            {"preserved": "custom"},
        )
        callbacks_block = cflearn.BuildCallbacksBlock()
        callbacks_block.build(config)
        self.assertEqual(
            [callback.__identifier__ for callback in callbacks_block.callbacks],
            config.callback_names,
        )
        progress = next(
            callback
            for callback in callbacks_block.callbacks
            if isinstance(callback, cflearn.ProgressCallback)
        )
        self.assertDictEqual(progress.settings.asdict(), tqdm_settings.asdict())

        disabled_config = cflearn.Config(
            auto_callback=False,
            callback_names=[progress_id],
            tqdm_settings={"use_tqdm": True, "desc": "disabled auto callback"},
            callback_configs={progress_id: []},
        )
        cflearn.SetTrainerDefaultsBlock().build(disabled_config)
        self.assertDictEqual(
            disabled_config.callback_configs[progress_id],
            {"settings": disabled_config.tqdm_settings},
        )
        callbacks_block = cflearn.BuildCallbacksBlock()
        callbacks_block.build(disabled_config)
        progress = next(
            callback
            for callback in callbacks_block.callbacks
            if isinstance(callback, cflearn.ProgressCallback)
        )
        self.assertTrue(progress.settings.use_tqdm)
        self.assertEqual(progress.settings.desc, "disabled auto callback")

    def test_build_metrics(self):
        data, in_dim, out_dim, _ = cflearn.testing.linear_data()
        config = cflearn.Config(
            workspace=self._workspace("build_metrics"),
            module_name="linear",
            module_config=dict(input_dim=in_dim, output_dim=out_dim),
            loss_name="mse",
        )
        config.to_debug()
        cflearn.TrainingPipeline.init(config).fit(data)
        with self.assertRaises(ValueError):
            c = config.copy()
            c.use_losses_as_metrics = False
            cflearn.TrainingPipeline.init(c).fit(data)
        config.metric_names = "mse"
        config.loss_metrics_weights = {"mse": 1.0}
        cflearn.TrainingPipeline.init(config).fit(data)
        with self.assertRaises(ValueError):
            c = config.copy()
            c.use_losses_as_metrics = False
            cflearn.TrainingPipeline.init(c).fit(data)

    def test_build_optimizers(self):
        data, in_dim, out_dim, _ = cflearn.testing.linear_data()
        config = cflearn.Config(
            workspace=self._workspace("build_optimizers"),
            module_name="linear",
            module_config=dict(input_dim=in_dim, output_dim=out_dim),
            optimizer_config={},
            scheduler_config={},
            optimizer_settings=dict(all={}),
            loss_name="mse",
        )
        config.to_debug()
        with self.assertRaises(ValueError):
            cflearn.TrainingPipeline.init(config).fit(data)
        config.optimizer_settings["all"]["optimizer"] = "adam"
        cflearn.TrainingPipeline.init(config).fit(data)

        @cflearn.register_module("custom_linear", allow_duplicate=True)
        class CustomLinear(nn.Module):
            def __init__(self, input_dim: int, output_dim: int):
                super().__init__()
                self.linear = nn.Linear(input_dim, output_dim)

            def forward(self, net):
                return self.linear(net)

        config.module_name = "custom_linear"
        config.optimizer_settings["linear"] = config.optimizer_settings.pop("all")
        cflearn.TrainingPipeline.init(config).fit(data)

        config.optimizer_settings = {
            "all": {
                "optimizer": "adam",
                "param_groups": [{"scope": "linear"}],
            }
        }
        cflearn.TrainingPipeline.init(config).fit(data)

    def test_training(self):
        data, in_dim, out_dim, _ = cflearn.testing.linear_data(6)
        config = cflearn.Config(
            workspace=self._workspace("training"),
            module_name="linear",
            module_config=dict(input_dim=in_dim, output_dim=out_dim),
            loss_name="mse",
            profile=True,
            profile_schedule_config=dict(skip_first=0, wait=0, warmup=1),
            save_pipeline_in_realtime=True,
        )
        config.to_debug().num_steps = 4
        p = cflearn.TrainingPipeline.init(config).fit(data, sample_weights=np.arange(6))
        with p.verbose_context(True):
            with p.verbose_context(False):
                pass
        with p.training.build_trainer.trainer.state.disable_logging:
            pass
        self.assertIsNone(p.training.local_rank)

        config.optimizer_settings = {"foo": None}
        with self.assertRaises(ValueError):
            cflearn.TrainingPipeline.init(config).fit(data)

        @cflearn.register_module("$test_linear")
        class _(nn.Linear):
            @property
            def bar_params(self):
                return [self.weight, self.bias]

        config.module_name = "$test_linear"
        config.module_config = dict(in_features=in_dim, out_features=out_dim)
        config.optimizer_settings = {"bar_params": None}
        cflearn.TrainingPipeline.init(config).fit(data)

        config.scheduler_name = "warmup"
        config.scheduler_config = {"scheduler_afterwards_base": "warmup"}
        with self.assertRaises(ValueError):
            cflearn.TrainingPipeline.init(config).fit(data)


class TestThirdParty(unittest.TestCase):
    def test_evaluation(self):
        class FooPredictor(cflearn.IPredictor):
            def predict(self, x: np.ndarray) -> np.ndarray:
                return x @ w

        data, _, _, w = cflearn.testing.linear_data()
        config = cflearn.Config()
        predictor = FooPredictor()
        with self.assertRaises(ValueError):
            cflearn.GeneralEvaluationPipeline(config, predictor)
        config.metric_names = "mse"
        p = cflearn.GeneralEvaluationPipeline(config, predictor)
        x, y = data.bundle.x_train, data.bundle.y_train
        test_loader = data.build_loader(x, y)
        metrics = p.evaluate(test_loader).metric_outputs.metric_values
        self.assertIn("mse", metrics)
        self.assertAlmostEqual(metrics["mse"], 0.0)


if __name__ == "__main__":
    unittest.main()
