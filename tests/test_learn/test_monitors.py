import unittest

import core.learn as cflearn


class TestMonitors(unittest.TestCase):
    def test_monitors(self):
        data, in_dim, out_dim, _ = cflearn.testing.linear_data(6, batch_size=4)
        config = cflearn.Config(
            module_name="linear",
            module_config=dict(input_dim=in_dim, output_dim=out_dim),
            monitor_names=["basic", "mean_std", "plateau", "conservative", "lazy"],
            monitor_configs=dict(plateau=dict(window_size=2)),
            scheduler_name="warmup",
            loss_name="mse",
        )
        config.to_debug().num_steps = 10
        config.num_epoch = 2
        cflearn.TrainingPipeline.init(config).fit(data)


if __name__ == "__main__":
    unittest.main()
