import torch
import unittest
import subprocess

import numpy as np
import core.learn as cflearn
import torch.nn as nn

from pathlib import Path
from core.toolkit import console


class TestLinear(unittest.TestCase):
    def test_linear(self) -> None:
        data, in_dim, out_dim, w = cflearn.testing.linear_data()
        config = cflearn.Config(
            module_name="linear",
            module_config=dict(input_dim=in_dim, output_dim=out_dim, bias=False),
            scheduler_name="warmup",
            loss_name="mse",
            num_steps=10**4,
        )
        config.to_debug().num_steps = 10  # comment this line to disable debug mode
        pipeline = cflearn.TrainingPipeline.init(config).fit(data)

        learned_w = pipeline.build_model.model.m.net.weight.view(-1).detach().numpy()
        console.log(f"> evaluation {pipeline.evaluate(data.build_loaders()[0])}")
        console.log(f"> learned weights {learned_w}")
        console.log(f"> ground truth weights {w.ravel()}")

        loaded = cflearn.PipelineSerializer.load_evaluation(pipeline.config.workspace)
        loaded_w = loaded.build_model.model.m.net.weight.view(-1).detach().numpy()
        console.log(f"> loaded evaluation {loaded.evaluate(data.build_loaders()[0])}")
        console.log(f"> loaded weights {loaded_w}")

    def test_linear_custom(self) -> None:
        @cflearn.register_module("custom_linear", allow_duplicate=True)
        class CustomLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Linear(in_dim, out_dim, bias=False)
                self.net.weight.data = torch.from_numpy(w.T.astype(np.float32))

            def forward(self, net: torch.Tensor) -> torch.Tensor:
                return self.net(net)

        data, in_dim, out_dim, w = cflearn.testing.linear_data()
        config = cflearn.Config(module_name="custom_linear", loss_name="mse")
        config.lr = 0.0
        config.to_debug()
        pipeline = cflearn.TrainingPipeline.init(config).fit(data)

        learned_w = pipeline.build_model.model.m.net.weight.view(-1).detach().numpy()
        console.log(f"> evaluation {pipeline.evaluate(data.build_loaders()[0])}")
        console.log(f"> learned weights {learned_w}")
        console.log(f"> ground truth weights {w.ravel()}")

        loaded = cflearn.PipelineSerializer.load_evaluation(pipeline.config.workspace)
        loaded_w = loaded.build_model.model.m.net.weight.view(-1).detach().numpy()
        console.log(f"> loaded evaluation {loaded.evaluate(data.build_loaders()[0])}")
        console.log(f"> loaded weights {loaded_w}")

    def test_linear_ddp(self) -> None:
        ddp_task_path = Path(__file__).parent / "ddp_linear_task.py"
        cmd = ["accelerate", "launch", "--num_processes=2", str(ddp_task_path)]
        # subprocess.run(cmd, check=True)  # uncomment this line to run the test


if __name__ == "__main__":
    unittest.main()
