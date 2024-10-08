import torch
import unittest

import core.learn as cflearn

from rich.table import Table
from unittest.mock import patch
from unittest.mock import MagicMock
from core.learn.schema import losses_type
from core.learn.callbacks.loggers import AutoWrapLine


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
            callback_names="wandb",
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
            callback_names="nan_detector",
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
            callback_names="grad_detector",
        )
        config.to_debug()
        for loss in ["inf_loss", "nan_loss"]:
            config.loss_name = loss
            with self.assertRaises(RuntimeError):
                cflearn.TrainingPipeline.init(config).fit(data)
        config.loss_name = "large_loss"
        cflearn.TrainingPipeline.init(config).fit(data)


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
        mock_get_terminal_size.return_value = MagicMock(columns=40)

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
