import time
import torch
import unittest

import torch.nn as nn
import core.learn as cflearn
import core.toolkit.console as console
import torch.nn.functional as F

from typing import Optional
from core.learn.schema import losses_type
from core.learn.schema import TrainerState
from core.toolkit.types import tensor_dict_type
from core.learn.modules.moe import squared_coef
from core.learn.modules.moe import get_load_balance_loss


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
                if moe.commons is not None:
                    for m in moe.commons:
                        sample_out += m(sample)
                moe_outputs.append(sample_out)
            return torch.cat(moe_outputs)

        dim = 7
        top_k = 29
        num_experts = 47
        output_dim = 23
        batch_size = 29

        with self.assertRaises(RuntimeError):
            get_load_balance_loss({})
        with self.assertRaises(RuntimeError):
            get_load_balance_loss({"importances": "foo"})
        with self.assertRaises(ValueError):
            cflearn.MoE(
                "fcnn",
                dict(input_dim=dim, output_dim=output_dim),
                dim=dim,
                top_k=5,
                num_experts=3,
            )
        torch.testing.assert_close(squared_coef(torch.randn(1)), torch.zeros(1))

        moe = cflearn.MoE(
            "fcnn",
            dict(input_dim=dim, output_dim=output_dim),
            dim=dim,
            top_k=top_k,
            num_experts=num_experts,
            num_common_experts=1,
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
                forward_results: tensor_dict_type,
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
        data, _, out_dim, _ = cflearn.testing.linear_data(dim=dim)
        config = cflearn.Config(
            module_name="moe",
            module_config=dict(
                expert_name="linear",
                expert_config=dict(input_dim=dim, output_dim=out_dim, bias=False),
                dim=dim,
                top_k=top_k,
                num_experts=num_experts,
            ),
            loss_name="moe",
            num_steps=10**4,
        )
        config.to_debug()  # comment this line to disable debug mode
        cflearn.TrainingPipeline.init(config).fit(data)
        config.module_config["router_config"] = dict(noisy_gating=False)
        cflearn.TrainingPipeline.init(config).fit(data)

    def test_prefix_modules(self) -> None:
        foo = cflearn.PrefixModules("$foo")
        A = type("A", (), {})
        B = type("B", (), {})
        foo.register("A")(A)
        foo.register("B")(B)
        self.assertListEqual(foo.all, ["$foo.A", "$foo.B"])
        self.assertIs(foo.get("A"), A)
        self.assertIs(foo.get("B"), B)
        self.assertIsNone(foo.get("C"))

    def test_ema(self) -> None:
        decay = 0.9
        p1 = nn.Parameter(torch.randn(2, 3, 4, 5))
        p2 = nn.Parameter(torch.randn(2, 3, 4, 5))
        p3 = nn.Parameter(torch.randn(2, 3, 4, 5))
        gt = p1.data
        gt = decay * gt + (1.0 - decay) * p2.data
        gt = decay * gt + (1.0 - decay) * p3.data
        for i, p in enumerate([p1, p1.data]):
            p = p.clone()
            ema = cflearn.EMA(decay, [("test", p)])
            p.data = p2.data
            ema()
            p.data = p3.data
            ema()
            ema.eval()
            ema.train()
            ema.eval()
            ema.train()
            ema.eval()
            torch.testing.assert_close(p.data, gt.data if i == 0 else p3.data)
            ema.train()
            ema.eval()
            ema.train()
            ema.eval()
            ema.train()
            torch.testing.assert_close(p.data, p3.data)
            ema.eval()
            ema.eval()
            ema.train()
            ema.train()
            ema.eval()
            ema.eval()
            torch.testing.assert_close(p.data, gt.data if i == 0 else p3.data)
            ema.train()
            ema.train()
            ema.eval()
            ema.eval()
            ema.train()
            ema.train()
            torch.testing.assert_close(p.data, p3.data)
            with cflearn.eval_context(ema):
                torch.testing.assert_close(p.data, gt.data if i == 0 else p3.data)
            torch.testing.assert_close(p.data, p3.data)
            str(ema)
            ema = cflearn.EMA(decay, [("test", p)], use_num_updates=True)
            ema()
            with cflearn.eval_context(ema):
                with self.assertRaises(RuntimeError):
                    ema()

    def test_ema_training(self) -> None:
        dim = 11
        data, _, out_dim, _ = cflearn.testing.linear_data(dim=dim)
        config = cflearn.Config(
            module_name="linear",
            module_config=dict(input_dim=dim, output_dim=out_dim, bias=False),
            loss_name="mse",
        )
        config.to_debug()
        p = cflearn.TrainingPipeline.init(config).fit(data)
        model = p.build_model.model
        x = torch.from_numpy(data.bundle.x_train)
        y0 = model.m(x)
        with model.eval_context():
            y1 = model.m(x)
        torch.testing.assert_close(y0, y1)

        @cflearn.register_module("linear_ema")
        class _(cflearn.Linear):
            def __init__(self) -> None:
                super().__init__(**config.module_config)
                self.ema = cflearn.EMA.hook(self, 0.9)

        config.module_name = "linear_ema"
        p = cflearn.TrainingPipeline.init(config).fit(data)
        model = p.build_model.model
        x = torch.from_numpy(data.bundle.x_train)
        y0 = model.m(x)
        with model.eval_context():
            y1 = model.m(x)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(y0, y1)

    def test_residual(self) -> None:
        dim = 11
        x = torch.randn(7, 5, dim)
        m_config = dict(input_dim=dim, output_dim=dim, batch_norm=True, dropout=0.1)
        m = cflearn.build_module("fcnn", **m_config)
        rm = cflearn.Residual(m)
        with cflearn.eval_context(rm):
            torch.testing.assert_close(m(x) + x, rm(x))
        m_config["highway"] = True
        m = cflearn.build_module("fcnn", **m_config)
        rm = cflearn.Residual(m)
        with cflearn.eval_context(rm):
            torch.testing.assert_close(m(x) + x, rm(x))

    def test_zero_module(self) -> None:
        dim = 11
        m = cflearn.build_module("linear", input_dim=dim, output_dim=dim)
        m.net.weight.data.random_()
        m.net.bias.data.random_()
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(m.net.weight, torch.zeros_like(m.net.weight))
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(m.net.bias, torch.zeros_like(m.net.bias))
        mz = cflearn.zero_module(m)
        torch.testing.assert_close(mz.net.weight, torch.zeros_like(m.net.weight))
        torch.testing.assert_close(mz.net.bias, torch.zeros_like(m.net.bias))

    def test_avg_pool_nd(self) -> None:
        cflearn.avg_pool_nd(1, 3)
        cflearn.avg_pool_nd(2, 3)
        cflearn.avg_pool_nd(3, 3)
        with self.assertRaises(ValueError):
            cflearn.avg_pool_nd(0, 3)
        with self.assertRaises(ValueError):
            cflearn.avg_pool_nd(4, 3)


if __name__ == "__main__":
    unittest.main()
