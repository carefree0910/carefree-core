import unittest

import numpy as np
import core.learn as cflearn


class TestOptimizers(unittest.TestCase):
    def test_optimizers(self):
        x = np.random.random([6, 10])
        w = np.random.random([10, 1])
        y = x @ w
        data = cflearn.ArrayData.init().fit(x, y)
        data.config.batch_size = 4
        for optimizer in ["sgd", "adam", "adamw", "rmsprop", "adamp"]:
            config = cflearn.Config(
                module_name="linear",
                module_config=dict(input_dim=x.shape[1], output_dim=y.shape[1]),
                optimizer_name=optimizer,
                loss_name="mse",
            )
            config.to_debug()
            cflearn.TrainingPipeline.init(config).fit(data)


if __name__ == "__main__":
    unittest.main()
