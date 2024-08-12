import os
import torch
import tempfile
import unittest

import numpy as np
import torch.nn as nn
import core.learn as cflearn
import torch.nn.functional as F

from pathlib import Path
from accelerate import Accelerator
from unittest.mock import patch
from unittest.mock import Mock
from core.toolkit.misc import random_hash
from core.learn.schema import losses_type
from core.learn.pipeline.blocks.basic import StateInfo
from core.learn.pipeline.blocks.basic import OptimizerSettings


class TestPipeline(unittest.TestCase):
    def test_basics(self):
        def build_pipeline(in_dim, out_dim):
            data, *_ = cflearn.testing.linear_data(dim=in_dim, out_dim=out_dim)
            config = cflearn.Config(
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

    def test_load_training(self):
        data, in_dim, out_dim, _ = cflearn.testing.linear_data()
        config = cflearn.Config(
            module_name="fcnn",
            module_config=dict(input_dim=in_dim, output_dim=out_dim),
            loss_name="mse",
            scheduler_name="warmup",
        )
        config.to_debug()
        p0 = cflearn.TrainingPipeline.init(config).fit(data)
        p1 = cflearn.PipelineSerializer.load_training(p0.config.workspace)
        p1.fit(data)

    def test_serializer(self):
        data, in_dim, out_dim, _ = cflearn.testing.linear_data()
        config = cflearn.Config(
            module_name="linear",
            module_config=dict(input_dim=in_dim, output_dim=out_dim),
            loss_name="mse",
        )
        config.to_debug()
        p0 = cflearn.TrainingPipeline.init(config).fit(data)
        x, y = data.bundle.x_train, data.bundle.y_train
        test_loader = data.build_loader(x, y)
        r0 = p0.predict(test_loader)[cflearn.PREDICTIONS_KEY]
        with tempfile.TemporaryDirectory() as tempdir:
            cflearn.PipelineSerializer.save(p0, tempdir, compress=True)
            p1 = cflearn.PipelineSerializer.load_inference(tempdir)
            r1 = p1.predict(test_loader)[cflearn.PREDICTIONS_KEY]
            np.testing.assert_allclose(r0, r1)
        workspace = p0.config.workspace
        p1 = cflearn.PipelineSerializer.pack_and_load_inference(workspace)
        r1 = p1.predict(test_loader)[cflearn.PREDICTIONS_KEY]
        np.testing.assert_allclose(r0, r1)
        for pt in [cflearn.PackType.TRAINING, cflearn.PackType.EVALUATION]:
            with tempfile.TemporaryDirectory() as tempdir:
                cflearn.PipelineSerializer.pack(workspace, tempdir, pack_type=pt)
        with tempfile.TemporaryDirectory() as tempdir:
            with self.assertRaises(ValueError):
                cflearn.PipelineSerializer.pack(workspace, tempdir, pack_type="bla")
        op = "test.onnx"
        with self.assertRaises(ValueError):
            cflearn.PipelineSerializer.pack_onnx(workspace, op)
        cflearn.PipelineSerializer.pack_onnx(workspace, op, loader_sample=test_loader)
        oi = cflearn.Inference(onnx=op)
        ro = oi.get_outputs(test_loader).forward_results[cflearn.PREDICTIONS_KEY]
        np.testing.assert_array_almost_equal(r0, ro)
        scripted_path = "test.pt"
        cflearn.PipelineSerializer.pack_scripted(workspace, scripted_path)
        p1 = cflearn.PipelineSerializer.pack_and_load_inference(workspace)
        p1.build_model.model.m = torch.jit.load(scripted_path)
        self.assertIsInstance(p1.build_model.model.m, torch.jit.ScriptModule)
        r1 = p1.predict(test_loader)[cflearn.PREDICTIONS_KEY]
        np.testing.assert_allclose(r0, r1)
        with tempfile.TemporaryDirectory() as tempdir:
            cflearn.PipelineSerializer.save(p1, tempdir, compress=True)
            cflearn.PipelineSerializer.update(p1, tempdir)
            p_folder = os.path.join(tempdir, cflearn.PipelineSerializer.pipeline_folder)
            cflearn.PipelineSerializer._load(p_folder)
            with self.assertRaises(ValueError):
                m_block = Mock()
                m_block.__identifier__ = "foo"
                cflearn.PipelineSerializer._load(p_folder, focuses=[m_block])
            p2 = cflearn.PipelineSerializer.load_inference(tempdir)
            r2 = p2.predict(test_loader)[cflearn.PREDICTIONS_KEY]
            np.testing.assert_allclose(r1, r2)
        with self.assertRaises(ValueError):
            cflearn.PipelineSerializer.update(p1, tempdir)

    def test_fuse(self):
        data, in_dim, out_dim, _ = cflearn.testing.linear_data()
        config = cflearn.Config(
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
            states = torch.load(os.path.join(ckpt_folder, ckpt_file))["states"]
            p.build_model.model.load_state_dict(states)
            r += p.predict(test_loader)[cflearn.PREDICTIONS_KEY]
        r /= 3
        pe = self_ensemble(3, workspace)
        re = pe.predict(test_loader)[cflearn.PREDICTIONS_KEY]
        np.testing.assert_array_almost_equal(r, re)
        states = {k: 0 for k in p.build_model.model.state_dict()}
        ckpt_files = cflearn.get_sorted_checkpoints(ckpt_folder, sort_by="latest")
        for ckpt_file in ckpt_files[:3]:
            i_states = torch.load(os.path.join(ckpt_folder, ckpt_file))["states"]
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


class TestBlocks(unittest.TestCase):
    def test_basics(self):
        data, *_ = cflearn.testing.linear_data()
        config = cflearn.Config()
        block = cflearn.PrepareWorkspaceBlock()
        block2 = cflearn.SerializeDataBlock()
        block.training_workspace = "_foo"
        mock_ddp_info = Mock()
        mock_ddp_info.local_rank = 1
        mock_ddp_info.world_size = 2
        with patch("core.learn.pipeline.blocks.basic.is_dist_initialized") as mock_dist:
            mock_dist.return_value = True
            with patch("core.learn.pipeline.common.is_ddp") as mock_ddp, patch(
                "core.learn.pipeline.common.get_ddp_info"
            ) as mock_info:
                mock_ddp.return_value = True
                mock_info.return_value = mock_ddp_info
                self.assertFalse(block.is_local_rank_0)
                block.build(config)
                block2.save_extra("")
        block = cflearn.ExtractStateInfoBlock()
        block.data = None
        self.assertFalse(block.try_load("_bar"))
        with self.assertRaises(ValueError):
            block.from_scratch(config)
        with patch("core.learn.pipeline.blocks.basic.get_ddp_info") as mock_info:
            block.data = data
            mock_info.return_value = mock_ddp_info
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
        block.save_extra(None)
        opt_pack = cflearn.OptimizerPack("all", "adamw")
        opt_settings = OptimizerSettings()
        opt_state_info = StateInfo(1, 1, 1, 1, 1)
        new_pack = opt_settings.update_opt_pack(opt_state_info, opt_pack)
        self.assertDictEqual(new_pack.optimizer_config, {"lr": opt_settings.lr})

    def test_set_default(self):
        data, in_dim, out_dim, _ = cflearn.testing.linear_data()
        config = cflearn.Config(
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

        cflearn.set_environ_workspace("_foo")
        p = cflearn.TrainingPipeline.init(config).fit(data)
        self.assertEqual(Path(p.config.workspace).parent.name, "_foo")
        cflearn.unset_environ_workspace()

        @cflearn.IModel.register("$test_foo_model")
        class FooModel(cflearn.CommonModel):
            @property
            def all_modules(self):
                return [self.m]

        class FooPipeline(cflearn.TrainingPipeline):
            @property
            def building_blocks(self):
                return super().building_blocks[:-1]

        config = cflearn.Config(
            model=FooModel.__identifier__,
            module_name="linear",
            module_config=dict(input_dim=in_dim, output_dim=out_dim),
        )
        p = FooPipeline.init(config).fit(data)
        p.prepare_distributed_with(Accelerator())

    def test_build_metrics(self):
        data, in_dim, out_dim, _ = cflearn.testing.linear_data()
        config = cflearn.Config(
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
        config.optimizer_settings["m.linear"] = config.optimizer_settings.pop("all")
        cflearn.TrainingPipeline.init(config).fit(data)

    def test_training(self):
        data, in_dim, out_dim, _ = cflearn.testing.linear_data(6)
        config = cflearn.Config(
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
        config.optimizer_settings = {"m.bar_params": None}
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
