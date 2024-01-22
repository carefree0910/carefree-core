import torch
import unittest

import core.learn as cflearn


class TestModel(unittest.TestCase):
    def test_model(self):
        with self.assertRaises(ValueError):
            cflearn.IModel.from_config(cflearn.Config())
        in_dim = 13
        out_dim = 7
        num_ensemble = 3
        models = [
            cflearn.IModel.from_config(
                cflearn.Config(
                    module_name="fcnn",
                    module_config=dict(input_dim=in_dim, output_dim=out_dim),
                    loss_name="mse",
                )
            )
            for _ in range(3)
        ]
        ensemble = cflearn.EnsembleModel(models[0], num_ensemble)
        self.assertIs(ensemble.all_modules[0], ensemble.m)
        self.assertEqual(len(ensemble.train_steps), 1)
        self.assertEqual(len(ensemble.m), num_ensemble)
        with self.assertRaises(RuntimeError):
            ensemble.build(ensemble.config)
        for i, model in enumerate(models):
            self.assertEqual(model.config, ensemble.config)
            ensemble.m[i].load_state_dict(model.state_dict())
        batch = {cflearn.INPUT_KEY: torch.randn(17, in_dim)}
        naive_out = 0.0
        for model in models:
            naive_out += model.run(0, batch)[cflearn.PREDICTIONS_KEY]
        naive_out /= 3.0
        ensemble_out = ensemble.run(0, batch)[cflearn.PREDICTIONS_KEY]
        torch.testing.assert_close(naive_out, ensemble_out)


if __name__ == "__main__":
    unittest.main()
