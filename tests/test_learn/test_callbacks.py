import unittest

import core.learn as cflearn


class TestCallbacks(unittest.TestCase):
    def test_wandb_callback(self) -> None:
        data, in_dim, out_dim, _ = cflearn.testing.linear_data()
        config = cflearn.Config(
            module_name="linear",
            module_config=dict(input_dim=in_dim, output_dim=out_dim, bias=False),
            scheduler_name="warmup",
            loss_name="mse",
            num_steps=10**4,
            callback_names="wandb",
            callback_configs=dict(wandb=dict(anonymous="must")),
        )
        config.to_debug().num_steps = 10  # comment this line to disable debug mode
        cflearn.TrainingPipeline.init(config).fit(data)

    def test_nan_detector_callback(self) -> None:
        data, in_dim, out_dim, _ = cflearn.testing.linear_data(3)
        data.bundle.y_train[0, 0] = float("nan")
        config = cflearn.Config(
            module_name="linear",
            module_config=dict(input_dim=in_dim, output_dim=out_dim, bias=False),
            loss_name="mse",
            callback_names="nan_detector",
        )
        config.to_debug()
        with self.assertRaises(RuntimeError):
            cflearn.TrainingPipeline.init(config).fit(data)


if __name__ == "__main__":
    unittest.main()
