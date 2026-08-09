import os
import torch
import tempfile
import unittest

import numpy as np
import torch.nn as nn
import core.learn as cflearn

from accelerate import Accelerator
from rich.table import Table
from unittest.mock import patch
from core.learn.schema import losses_type
from core.learn.callbacks.loggers import AutoWrapLine


def make_callback_trainer(
    config: cflearn.Config,
    module: nn.Module,
) -> cflearn.Trainer:
    trainer = cflearn.Trainer(config)
    model = cflearn.IModel.from_config(config)
    model.m = module
    trainer.model = model
    trainer.accelerator = Accelerator(cpu=True)
    return trainer


class TestCallbacks(unittest.TestCase):
    def test_wandb_callback(self) -> None:
        @cflearn.register_module("foo_linear", allow_duplicate=True)
        class _(cflearn.Linear):
            def __init__(self, input_dim: int, output_dim: int, *, bias: bool = True):
                super().__init__(input_dim, output_dim, bias=bias)
                self.register_buffer("param", torch.tensor(0.0))

        data, in_dim, out_dim, _ = cflearn.testing.linear_data(use_validation=True)
        config = cflearn.Config(
            module_name="foo_linear",
            module_config=dict(input_dim=in_dim, output_dim=out_dim, bias=False),
            scheduler_name="warmup",
            loss_name="mse",
            callback_names=["wandb"],
            callback_configs=dict(wandb=dict(anonymous="must", log_artifacts=True)),
            num_steps=20,
            log_steps=4,
        )
        cflearn.TrainingPipeline.init(config).fit(data)

    def test_nan_detector_callback(self) -> None:
        data, in_dim, out_dim, _ = cflearn.testing.linear_data(3)
        data.bundle.y_train[0, 0] = float("nan")
        config = cflearn.Config(
            module_name="linear",
            module_config=dict(input_dim=in_dim, output_dim=out_dim, bias=False),
            loss_name="mse",
            callback_names=["nan_detector"],
            callback_configs=dict(nan_detector=dict(check_parameters=True)),
        )
        config.to_debug()
        with self.assertRaises(RuntimeError):
            cflearn.TrainingPipeline.init(config).fit(data)

    def test_gradient_detector_callback(self) -> None:
        @cflearn.register_loss("inf_loss")
        class InfLoss(cflearn.ILoss):
            def forward(self, forward_results, batch, state=None) -> losses_type:
                predictions = forward_results[cflearn.PREDICTIONS_KEY]
                return 1.0 / (predictions - predictions.detach()).mean()

        @cflearn.register_loss("nan_loss")
        class NaNLoss(cflearn.ILoss):
            def forward(self, forward_results, batch, state=None) -> losses_type:
                predictions = forward_results[cflearn.PREDICTIONS_KEY]
                return 0.0 / (predictions - predictions.detach()).mean()

        @cflearn.register_loss("large_loss")
        class LargeLoss(cflearn.ILoss):
            def forward(self, forward_results, batch, state=None) -> losses_type:
                predictions = forward_results[cflearn.PREDICTIONS_KEY]
                labels = batch[cflearn.LABEL_KEY]
                return 1000 * (predictions - labels).abs().mean()

        data, in_dim, out_dim, _ = cflearn.testing.linear_data(3)
        config = cflearn.Config(
            module_name="linear",
            module_config=dict(input_dim=in_dim, output_dim=out_dim, bias=False),
            callback_names=["grad_detector"],
        )
        config.to_debug()
        for loss in ["inf_loss", "nan_loss"]:
            config.loss_name = loss
            with self.assertRaises(RuntimeError):
                cflearn.TrainingPipeline.init(config).fit(data)
        config.loss_name = "large_loss"
        cflearn.TrainingPipeline.init(config).fit(data)

    def test_nan_detector_warning(self) -> None:
        batch = {"object": np.array([object()], dtype=object)}
        with tempfile.TemporaryDirectory() as workspace:
            config = cflearn.Config(
                workspace=workspace,
                module_name="linear",
                module_config=dict(input_dim=1, output_dim=1),
                loss_name="mse",
            )
            trainer = make_callback_trainer(config, nn.Linear(1, 1))
            step_outputs = cflearn.StepOutputs(
                {},
                {cflearn.LOSS_KEY: torch.tensor(float("nan"))},
            )
            with patch(
                "core.learn.callbacks.detectors.console.warn"
            ) as mock_warn, patch("core.learn.callbacks.detectors.console.error"):
                with self.assertRaises(RuntimeError):
                    cflearn.NaNDetectorCallback().after_train_step(
                        batch,
                        step_outputs,
                        trainer,
                    )
            warning_messages = [call.args[0] for call in mock_warn.call_args_list]
            self.assertTrue(
                any(
                    "failed to calculate nan ratio" in message
                    for message in warning_messages
                )
            )
            checkpoint_folder = os.path.join(
                workspace,
                "debugging",
                f"checkpoints_{trainer.accelerator.process_index}",
            )
            self.assertTrue(
                any(file.endswith(".pt") for file in os.listdir(checkpoint_folder))
            )

    def test_update_artifacts_individually(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            config = cflearn.Config(
                workspace=workspace,
                module_name="linear",
                module_config=dict(input_dim=1, output_dim=1),
                loss_name="mse",
                save_pipeline_in_realtime=True,
                save_realtime_pipeline_individually=True,
            )
            trainer = cflearn.Trainer(config)
            trainer.pipeline = cflearn.TrainingPipeline.init(config)
            trainer.state = cflearn.TrainerState(
                num_epoch=1,
                batch_size=1,
                loader_length=1,
            )
            trainer.state.step = 7
            cflearn.UpdateArtifactsCallback().before_loop(trainer)
            pipeline_folder = os.path.join(workspace, "pipeline_7")
            self.assertTrue(os.path.isdir(pipeline_folder))
            self.assertTrue(os.path.isfile(os.path.join(pipeline_folder, "id.txt")))

    def test_finetune_exclusions_and_ddp_names(self) -> None:
        def assert_load_state_dict(load_state_dict, expected, strict):
            load_state_dict.assert_called_once()
            loaded = load_state_dict.call_args.args[0]
            self.assertSetEqual(set(loaded), set(expected))
            for name, value in expected.items():
                self.assertTrue(torch.equal(loaded[name], value))
            self.assertIs(
                load_state_dict.call_args.kwargs["strict"],
                strict,
            )

        def run(
            finetune_config,
            states,
            *,
            ddp=False,
        ):
            with tempfile.TemporaryDirectory() as folder:
                checkpoint = os.path.join(folder, "checkpoint.pt")
                torch.save(states, checkpoint)
                finetune_config = dict(finetune_config)
                finetune_config["pretrained_ckpt"] = checkpoint
                config = cflearn.Config(
                    module_name="linear",
                    module_config=dict(input_dim=1, output_dim=1),
                    loss_name="mse",
                    finetune_config=finetune_config,
                )
                linear = nn.Linear(1, 1)
                nn.init.zeros_(linear.weight)
                nn.init.zeros_(linear.bias)
                module = nn.Sequential()
                module.add_module("module", linear)
                trainer = make_callback_trainer(config, module)
                with patch.object(
                    trainer.model,
                    "load_state_dict",
                    wraps=trainer.model.load_state_dict,
                ) as load_state_dict, patch(
                    "core.learn.callbacks.defaults.is_ddp",
                    return_value=ddp,
                ), patch(
                    "core.learn.callbacks.defaults.console.warn"
                ), patch(
                    "core.learn.callbacks.defaults.console.log"
                ):
                    cflearn.TrainingLoopCallback().before_summary(trainer)
                return trainer.model, load_state_dict

        states = {
            "module.weight": torch.ones(1, 1),
            "module.bias": torch.full((1,), 2.0),
        }
        unmatched, load_state_dict = run(
            {
                "pretrained_ckpt": "checkpoint.pt",
                "exclude": "missing",
            },
            states,
        )
        self.assertTrue(
            torch.equal(
                unmatched.state_dict()["module.weight"],
                states["module.weight"],
            )
        )
        self.assertTrue(
            torch.equal(unmatched.state_dict()["module.bias"], states["module.bias"])
        )
        assert_load_state_dict(load_state_dict, states, True)
        matched, load_state_dict = run(
            {
                "pretrained_ckpt": "checkpoint.pt",
                "exclude": "module.weight",
            },
            states,
        )
        self.assertTrue(
            torch.equal(
                matched.state_dict()["module.weight"],
                torch.zeros(1, 1),
            )
        )
        self.assertTrue(
            torch.equal(
                matched.state_dict()["module.bias"],
                states["module.bias"],
            )
        )
        assert_load_state_dict(
            load_state_dict,
            {"module.bias": states["module.bias"]},
            False,
        )
        frozen, _ = run(
            {
                "pretrained_ckpt": "checkpoint.pt",
                "freeze": "weight",
            },
            states,
            ddp=True,
        )
        frozen_parameters = dict(frozen.named_parameters())
        weight = frozen_parameters["module.weight"]
        bias = frozen_parameters["module.bias"]
        self.assertFalse(weight.requires_grad)
        self.assertTrue(bias.requires_grad)

        frozen_except, _ = run(
            {
                "pretrained_ckpt": "checkpoint.pt",
                "freeze_except": "weight",
            },
            states,
            ddp=True,
        )
        frozen_except_parameters = dict(frozen_except.named_parameters())
        weight = frozen_except_parameters["module.weight"]
        bias = frozen_except_parameters["module.bias"]
        self.assertTrue(weight.requires_grad)
        self.assertFalse(bias.requires_grad)

    def test_mocked_wandb_callback(self) -> None:
        module = nn.Linear(2, 1)
        module.register_buffer("scale", torch.ones(1))
        module(torch.ones(1, 2)).sum().backward()
        config = cflearn.Config(
            workspace="workspace",
            module_name="linear",
            module_config=dict(input_dim=2, output_dim=1),
            loss_name="mse",
        )
        trainer = make_callback_trainer(config, module)
        trainer.state = cflearn.TrainerState(
            num_epoch=1,
            batch_size=1,
            loader_length=1,
        )
        callback = cflearn.WandBCallback(
            project="project",
            config={"key": "value"},
            entity="entity",
            save_code=True,
            group="group",
            job_type="job",
            tags=["tag"],
            name="name",
            notes="notes",
            relogin=True,
            anonymous="must",
            log_histograms=True,
            log_artifacts=True,
            log_grad_norms=True,
        )
        without_grad_norms = cflearn.WandBCallback(log_grad_norms=False)
        step_outputs = cflearn.StepOutputs(
            {},
            {cflearn.LOSS_KEY: torch.tensor(1.0)},
        )
        metrics_outputs = cflearn.MetricsOutputs(
            0.5,
            {"mse": 1.0},
            {"mse": False},
        )
        loss = torch.tensor(1.0)
        loss_res = cflearn.TrainStepLoss(loss, {cflearn.LOSS_KEY: loss})
        with patch(
            "core.learn.schema.is_local_rank_0",
            return_value=True,
        ), patch("core.learn.callbacks.loggers.wandb") as mock_wandb:
            callback.initialize()
            callback.before_loop(trainer)
            without_grad_norms.before_gradient_update(
                trainer,
                {},
                {},
                loss_res,
                True,
            )
            callback.before_gradient_update(
                trainer,
                {},
                {},
                loss_res,
                True,
            )
            callback.log_lr("lr", 1.0e-3, trainer)
            callback.log_train_step(step_outputs, trainer.state)
            callback.log_metrics(metrics_outputs, trainer.state)
            trainer.state.set_terminate()
            callback.finalize(trainer)
        mock_wandb.login.assert_called_once_with(anonymous="must", relogin=True)
        mock_wandb.init.assert_called_once_with(**callback.init_kwargs)
        mock_wandb.log_artifact.assert_called_with("workspace")
        mock_wandb.finish.assert_called_once_with()


class TestAutoWrapLine(unittest.TestCase):
    def setUp(self):
        self.auto_wrap_line = AutoWrapLine()
        with self.assertRaises(RuntimeError):
            self.auto_wrap_line.get_table()

    def test_add_column(self):
        self.auto_wrap_line.add_column("test_column")
        self.assertEqual(self.auto_wrap_line._col_names, ["test_column"])
        self.assertEqual(self.auto_wrap_line._col_kwargs, [{}])

    def test_add_row(self):
        self.auto_wrap_line.add_row("test_row")
        self.assertEqual(self.auto_wrap_line._row, ("test_row",))
        with self.assertRaises(RuntimeError):
            self.auto_wrap_line.add_row("test_row")

    @patch("shutil.get_terminal_size")
    def test_get_table(self, mock_get_terminal_size):
        n_cols = 20
        mock_get_terminal_size.return_value = os.terminal_size((40, 24))

        for i in range(n_cols):
            self.auto_wrap_line.add_column(f"test_column{i}")
        self.auto_wrap_line.add_row(*[f"test_row{i}" for i in range(n_cols)])

        table = self.auto_wrap_line.get_table()

        self.assertIsInstance(table, Table)
        table = table.columns[0]
        self.assertEqual(table._cells[0].columns[0].header, "")
        self.assertEqual(table._cells[0].columns[1].header, "test_column0")
        self.assertEqual(table._cells[0].columns[1]._cells[0], "test_row0")
        for i in range(1, n_cols // 2):
            self.assertEqual(table._cells[i].columns[0].header, f"test_column{i*2-1}")
            self.assertEqual(table._cells[i].columns[1].header, f"test_column{i*2}")
            self.assertEqual(table._cells[i].columns[0]._cells[0], f"test_row{i*2-1}")
            self.assertEqual(table._cells[i].columns[1]._cells[0], f"test_row{i*2}")


if __name__ == "__main__":
    unittest.main()
