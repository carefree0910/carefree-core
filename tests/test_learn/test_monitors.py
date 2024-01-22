import unittest

import numpy as np
import core.learn as cflearn


class TestMonitors(unittest.TestCase):
    def test_monitors(self):
        x = np.random.random([6, 10])
        w = np.random.random([10, 1])
        y = x @ w
        data = cflearn.ArrayData.init().fit(x, y)
        data.config.batch_size = 4
        config = cflearn.Config(
            module_name="linear",
            module_config=dict(input_dim=x.shape[1], output_dim=y.shape[1]),
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
