import time
import torch
import unittest

import numpy as np
import core.learn as cflearn
import core.toolkit.console as console
import torch.nn.functional as F

from typing import Optional
from core.learn.schema import losses_type
from core.learn.schema import train_forward_results_type
from core.learn.schema import TrainerState
from core.toolkit.types import tensor_dict_type


class TestModules(unittest.TestCase):
    def test_moe(self) -> None:
        def naive_moe() -> torch.Tensor:
            moe_outputs = []
            for sample in net:
                sample = sample.unsqueeze(0)
                logits = sample @ moe.router.w_gate.data
                indices = logits.argsort(dim=1, descending=True)[0, :top_k]
                weights = torch.softmax(logits[0][indices], 0)
                sample_out = 0.0
                for i, idx in enumerate(indices.tolist()):
                    i_net = moe.experts[idx](sample)
                    sample_out += i_net * weights[i]
                moe_outputs.append(sample_out)
            return torch.cat(moe_outputs)

        dim = 7
        top_k = 11
        num_experts = 17
        output_dim = 23
        batch_size = 29

        moe = cflearn.MoE(
            "fcnn",
            dict(input_dim=dim, output_dim=output_dim),
            dim=dim,
            top_k=top_k,
            num_experts=num_experts,
        )
        moe.eval()
        moe.router.w_gate.data.normal_()

        net = torch.randn(batch_size, dim)
        naive_moe_net = naive_moe()
        t0 = time.time()
        for _ in range(10):
            naive_moe()
        t1 = time.time()
        moe_net = moe(net)[cflearn.PREDICTIONS_KEY]
        t2 = time.time()
        for _ in range(10):
            moe(net)
        t3 = time.time()
        t_naive_moe = t1 - t0
        t_moe = t3 - t2
        console.log(f"naive_moe : {t_naive_moe:.3f}")
        console.log(f"moe       : {t_moe:.3f}")
        self.assertSequenceEqual(moe_net.shape, (batch_size, output_dim))
        self.assertLess(t_moe, t_naive_moe)
        torch.testing.assert_close(naive_moe_net, moe_net)

    def test_moe_training(self) -> None:
        @cflearn.register_loss("moe")
        class MoELoss(cflearn.ILoss):
            def forward(
                self,
                forward_results: train_forward_results_type,
                batch: tensor_dict_type,
                state: Optional[TrainerState] = None,
            ) -> losses_type:
                predictions = forward_results[cflearn.PREDICTIONS_KEY]
                labels = batch[cflearn.LABEL_KEY]
                mse_loss = F.mse_loss(predictions, labels)
                load_loss = cflearn.get_load_balance_loss(forward_results)
                loss = mse_loss + 0.01 * load_loss
                return {cflearn.LOSS_KEY: loss, "mse": mse_loss, "load": load_loss}

        dim = 17
        top_k = 3
        num_experts = 7
        cflearn.register_module("moe")(cflearn.MoE)
        x = np.random.random([10000, dim])
        w = np.random.random([dim, 1])
        y = x @ w
        data = cflearn.ArrayData.init().fit(x, y)
        data.config.batch_size = 100
        config = cflearn.Config(
            module_name="moe",
            module_config=dict(
                expert_name="linear",
                expert_config=dict(input_dim=dim, output_dim=y.shape[1], bias=False),
                dim=dim,
                top_k=top_k,
                num_experts=num_experts,
            ),
            loss_name="moe",
            num_steps=10**4,
        )
        config.to_debug()  # comment this line to disable debug mode
        cflearn.TrainingPipeline.init(config).fit(data)


if __name__ == "__main__":
    unittest.main()
