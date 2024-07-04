import torch
import unittest

import core.learn as cflearn

from core.learn.schema import losses_type


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


if __name__ == "__main__":
    unittest.main()
