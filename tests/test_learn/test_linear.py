import os
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
        x = np.random.random([10000, 10])
        w = np.random.random([10, 1])
        y = x @ w
        data = cflearn.ArrayData.init().fit(x, y)
        data.config.batch_size = 100
        config = cflearn.Config(
            module_name="linear",
            module_config=dict(input_dim=x.shape[1], output_dim=y.shape[1], bias=False),
            loss_name="mse",
            num_steps=10**4,
        )
        config.to_debug()  # comment this line to disable debug mode
        pipeline = cflearn.TrainingPipeline.init(config).fit(data)

        learned_w = pipeline.build_model.model.m.net.weight.view(-1).detach().numpy()
        console.log(f"> evaluation {pipeline.evaluate(data.build_loaders()[0])}")
        console.log(f"> learned weights {learned_w}")
        console.log(f"> ground truth weights {w.ravel()}")

        workspace = pipeline.config.workspace
        pipeline_dir = os.path.join(
            workspace, cflearn.PipelineSerializer.pipeline_folder
        )
        loaded = cflearn.PipelineSerializer.load_evaluation(pipeline_dir)
        loaded_w = loaded.build_model.model.m.net.weight.view(-1).detach().numpy()
        console.log(f"> loaded evaluation {loaded.evaluate(data.build_loaders()[0])}")
        console.log(f"> loaded weights {loaded_w}")

    def test_linear_custom(self) -> None:
        @cflearn.register_module("custom_linear")
        class CustomLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Linear(x.shape[1], y.shape[1], bias=False)
                self.net.weight.data = torch.from_numpy(w.T.astype(np.float32))

            def forward(self, net: torch.Tensor) -> torch.Tensor:
                return self.net(net)

        x = np.random.random([10000, 10])
        w = np.random.random([10, 1])
        y = x @ w
        data = cflearn.ArrayData.init().fit(x, y)
        data.config.batch_size = 100
        config = cflearn.Config(
            module_name="custom_linear", loss_name="mse", num_steps=10**4
        )
        config.lr = 0.0
        config.to_debug()
        pipeline = cflearn.TrainingPipeline.init(config).fit(data)

        learned_w = pipeline.build_model.model.m.net.weight.view(-1).detach().numpy()
        console.log(f"> evaluation {pipeline.evaluate(data.build_loaders()[0])}")
        console.log(f"> learned weights {learned_w}")
        console.log(f"> ground truth weights {w.ravel()}")

        workspace = pipeline.config.workspace
        pipeline_dir = os.path.join(
            workspace, cflearn.PipelineSerializer.pipeline_folder
        )
        loaded = cflearn.PipelineSerializer.load_evaluation(pipeline_dir)
        loaded_w = loaded.build_model.model.m.net.weight.view(-1).detach().numpy()
        console.log(f"> loaded evaluation {loaded.evaluate(data.build_loaders()[0])}")
        console.log(f"> loaded weights {loaded_w}")

    def test_linear_ddp(self) -> None:
        ddp_task_path = Path(__file__).parent / "ddp_linear_task.py"
        cmd = ["accelerate", "launch", "--num_processes=2", str(ddp_task_path)]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    unittest.main()
