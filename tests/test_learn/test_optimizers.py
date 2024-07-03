import torch
import unittest

import core.learn as cflearn
import torch.nn as nn


class TestOptimizers(unittest.TestCase):
    def test_optimizers(self):
        data, in_dim, out_dim, _ = cflearn.testing.linear_data(6, batch_size=4)
        for optimizer in ["sgd", "adam", "adamw", "rmsprop", "adamp_10", "adamp_0"]:
            if not optimizer.startswith("adamp"):
                optimizer_config = None
            else:
                optimizer, d = optimizer.split("_")
                d = float(d)
                nesterov = d > 0
                optimizer_config = dict(delta=d, weight_decay=0.1, nesterov=nesterov)
            config = cflearn.Config(
                module_name="linear",
                module_config=dict(input_dim=in_dim, output_dim=out_dim),
                optimizer_name=optimizer,
                optimizer_config=optimizer_config,
                loss_name="mse",
            )
            config.to_debug()
            cflearn.TrainingPipeline.init(config).fit(data)

    def test_adamp_closure(self):
        def closure():
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            return loss

        model = nn.Linear(10, 2)
        params = list(model.parameters()) + [torch.tensor(1.0)]
        criterion = nn.MSELoss()
        optimizer = cflearn.optimizers.AdamP(params)
        x = torch.randn(4, 10)
        y = torch.randn(4, 2)
        optimizer.step(closure)


if __name__ == "__main__":
    unittest.main()
