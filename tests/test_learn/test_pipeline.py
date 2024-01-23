import os
import torch
import tempfile
import unittest

import numpy as np
import torch.nn as nn
import core.learn as cflearn
import torch.nn.functional as F

from core.learn.schema import losses_type
from core.toolkit.misc import prepare_workspace_from


class TestPipeline(unittest.TestCase):
    def test_basics(self):
        data, in_dim, out_dim, _ = cflearn.testing.linear_data()
        config = cflearn.Config(
            module_name="fcnn",
            module_config=dict(input_dim=in_dim, output_dim=out_dim),
            loss_name="mse",
        )
        config.to_debug()
        p0 = cflearn.TrainingPipeline.init(config).fit(data)
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
        with self.assertRaises(RuntimeError):
            p1.predict(test_loader, recover_labels=True)
        p0.predict(test_loader, return_classes=True)
        p0.predict(test_loader, return_probabilities=True)

    def test_load_training(self):
        data, in_dim, out_dim, _ = cflearn.testing.linear_data()
        config = cflearn.Config(
            module_name="fcnn",
            module_config=dict(input_dim=in_dim, output_dim=out_dim),
            loss_name="mse",
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
            sub_folder = os.path.join(tempdir, "pipeline")
            cflearn.PipelineSerializer.save(p0, sub_folder, compress=True)
            p1 = cflearn.PipelineSerializer._load_inference(sub_folder)
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
            sub_folder = os.path.join(tempdir, "pipeline")
            cflearn.PipelineSerializer.save(p1, sub_folder, compress=True)
            p2 = cflearn.PipelineSerializer._load_inference(sub_folder)
            r2 = p2.predict(test_loader)[cflearn.PREDICTIONS_KEY]
            np.testing.assert_allclose(r1, r2)

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
        pf = cflearn.PipelineSerializer.fuse_inference(ws)
        rf = pf.predict(test_loader)[cflearn.PREDICTIONS_KEY]
        np.testing.assert_array_almost_equal(0.5 * (r0 + r1), rf)
        p2 = cflearn.TrainingPipeline.init(config).fit(data)
        ws.append(p2.config.workspace)
        pf = cflearn.PipelineSerializer.fuse_inference(ws, num_picked=2)
        self.assertEqual(len(pf.build_model.model.m), 2)
        with self.assertRaises(ValueError):
            cflearn.PipelineSerializer._fuse_multiple(ws, cflearn.PackType.TRAINING)


class TestBlocks(unittest.TestCase):
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

        config = cflearn.Config(
            module_name="linear",
            module_config=dict(input_dim=in_dim, output_dim=out_dim),
        )

        class FooPipeline(cflearn.TrainingPipeline):
            @property
            def building_blocks(self):
                return super().building_blocks[:-1]

        p = FooPipeline.init(config).fit(data)

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
        )
        config.to_debug().num_steps = 2
        p = cflearn.TrainingPipeline.init(config).fit(data, sample_weights=np.arange(6))
        with p.training.build_trainer.trainer.state.disable_logging:
            pass


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
        metrics = p.evaluate(test_loader).metric_values
        self.assertIn("mse", metrics)
        self.assertAlmostEqual(metrics["mse"], 0.0)


if __name__ == "__main__":
    unittest.main()