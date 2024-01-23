import unittest

import numpy as np
import core.learn as cflearn


class TestOptimizers(unittest.TestCase):
    def test_optimizers(self):
        data, in_dim, out_dim, _ = cflearn.testing.linear_data(6, batch_size=4)
        for optimizer in ["sgd", "adam", "adamw", "rmsprop", "adamp"]:
            config = cflearn.Config(
                module_name="linear",
                module_config=dict(input_dim=in_dim, output_dim=out_dim),
                optimizer_name=optimizer,
                loss_name="mse",
            )
            config.to_debug()
            cflearn.TrainingPipeline.init(config).fit(data)


if __name__ == "__main__":
    unittest.main()
