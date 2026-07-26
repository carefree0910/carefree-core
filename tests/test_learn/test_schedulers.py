import torch
import unittest

import core.learn as cflearn

from tempfile import TemporaryDirectory
from unittest.mock import Mock


class TestSchedulers(unittest.TestCase):
    def _make_cyclic_scheduler(
        self,
        optimizer,
        scheduler_config=None,
    ):
        state_info = Mock(
            num_batches=1,
            snapshot_start_step=1,
            num_step_per_snapshot=1,
        )
        extract_state_info = Mock(state_info=state_info)
        block = cflearn.BuildOptimizersBlock()
        block.previous = {
            cflearn.ExtractStateInfoBlock.__identifier__: extract_state_info,
        }
        block.schedulers = {}
        optimizer_lr = optimizer.param_groups[0]["lr"]
        pack = cflearn.OptimizerPack(
            "all",
            "adam",
            "cyclic",
            {"lr": optimizer_lr},
            scheduler_config,
        )
        block._define_scheduler(optimizer, pack)
        scheduler = block.schedulers["all"]
        self.assertIsInstance(
            scheduler,
            torch.optim.lr_scheduler.CyclicLR,
        )
        return scheduler

    @staticmethod
    def _step_lrs(optimizer, scheduler, num_steps):
        lrs = []
        for _ in range(num_steps):
            optimizer.step()
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])
        return lrs

    @staticmethod
    def _make_serialize_optimizer_block(optimizer, scheduler):
        build_optimizers = Mock(
            optimizers={"all": optimizer},
            schedulers={"all": scheduler},
        )
        build_trainer = Mock(
            trainer=Mock(
                accelerator=None,
                state=None,
            ),
        )
        block = cflearn.SerializeOptimizerBlock()
        block.previous = {
            cflearn.BuildOptimizersBlock.__identifier__: build_optimizers,
            cflearn.BuildTrainerBlock.__identifier__: build_trainer,
        }
        return block

    def test_cyclic_default_bounds(self) -> None:
        optimizer_lr = 0.1
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.Adam([parameter], lr=optimizer_lr)

        scheduler = self._make_cyclic_scheduler(optimizer)

        self.assertEqual(scheduler.base_lrs, [1.0e-8])
        self.assertEqual(scheduler.max_lrs, [optimizer_lr])
        self.assertEqual(scheduler.get_last_lr(), [1.0e-8])

    def test_cyclic_disables_unavailable_momentum(self) -> None:
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.Adam([parameter], lr=0.1)

        scheduler = self._make_cyclic_scheduler(optimizer)

        self.assertFalse(scheduler.cycle_momentum)

    def test_cyclic_default_curve(self) -> None:
        optimizer_lr = 0.1
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.Adam([parameter], lr=optimizer_lr)
        scheduler = self._make_cyclic_scheduler(
            optimizer,
            {
                "step_size_up": 2,
                "step_size_down": 2,
            },
        )

        lrs = scheduler.get_last_lr() + self._step_lrs(optimizer, scheduler, 4)
        expected = [
            1.0e-8,
            0.5 * (optimizer_lr + 1.0e-8),
            optimizer_lr,
            0.5 * (optimizer_lr + 1.0e-8),
            1.0e-8,
        ]

        self.assertEqual(len(lrs), len(expected))
        for lr, expected_lr in zip(lrs, expected):
            self.assertAlmostEqual(lr, expected_lr)

    def test_cyclic_explicit_bounds_override_defaults(self) -> None:
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.Adam([parameter], lr=0.1)
        scheduler = self._make_cyclic_scheduler(
            optimizer,
            {
                "base_lr": 0.02,
                "max_lr": 0.2,
                "step_size_up": 1,
                "step_size_down": 1,
            },
        )

        lrs = scheduler.get_last_lr() + self._step_lrs(optimizer, scheduler, 2)

        self.assertEqual(scheduler.base_lrs, [0.02])
        self.assertEqual(scheduler.max_lrs, [0.2])
        self.assertEqual(lrs, [0.02, 0.2, 0.02])

    def test_cyclic_state_round_trip_preserves_trajectory(self) -> None:
        scheduler_config = {
            "step_size_up": 2,
            "step_size_down": 3,
        }
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.Adam([parameter], lr=0.1)
        scheduler = self._make_cyclic_scheduler(optimizer, scheduler_config)
        self._step_lrs(optimizer, scheduler, 3)
        expected_last_epoch = scheduler.last_epoch
        expected_lr = optimizer.param_groups[0]["lr"]
        serializer = self._make_serialize_optimizer_block(optimizer, scheduler)

        with TemporaryDirectory() as folder:
            serializer.save_extra(folder)
            expected_lrs = self._step_lrs(optimizer, scheduler, 6)

            restored_parameter = torch.nn.Parameter(torch.ones(1))
            restored_optimizer = torch.optim.Adam([restored_parameter], lr=0.1)
            restored_scheduler = self._make_cyclic_scheduler(
                restored_optimizer,
                scheduler_config,
            )
            restored_serializer = self._make_serialize_optimizer_block(
                restored_optimizer,
                restored_scheduler,
            )
            restored_serializer.load_from(folder)
            self.assertEqual(restored_scheduler.last_epoch, expected_last_epoch)
            self.assertEqual(
                restored_optimizer.param_groups[0]["lr"],
                expected_lr,
            )
            restored_lrs = self._step_lrs(
                restored_optimizer,
                restored_scheduler,
                6,
            )

        self.assertEqual(restored_lrs, expected_lrs)

    def test_warmup_removes_unsupported_verbose(self) -> None:
        class NoVerboseScheduler(torch.optim.lr_scheduler.LRScheduler):
            def __init__(self, optimizer):
                super().__init__(optimizer)

            def get_lr(self):
                return self.base_lrs

        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scheduler_config = {"verbose": False}
        scheduler = cflearn.WarmupScheduler(
            optimizer,
            multiplier=2.0,
            warmup_step=1,
            scheduler_afterwards_base=NoVerboseScheduler,
            scheduler_afterwards_config=scheduler_config,
        )
        self.assertIsInstance(scheduler.scheduler_afterwards, NoVerboseScheduler)
        self.assertDictEqual(scheduler_config, {"verbose": False})

    def test_schedulers(self):
        data, in_dim, out_dim, _ = cflearn.testing.linear_data(6, batch_size=4)
        scheduler_config = dict(start_epoch=0, end_epoch=1, warmup_step=2)
        for scheduler in [
            "linear",
            "linear_inverse",
            "step",
            "exponential",
            "plateau",
            "warmup",
            "op.cosine_warmup",
            "op.linear_warmup",
            "op.foo",
            "warmup-step",
        ]:
            if scheduler == "warmup-step":
                scheduler = "warmup"
                scheduler_config["scheduler_afterwards_base"] = "step"
            if scheduler.startswith("op."):
                scheduler, op_type = scheduler.split(".")
                scheduler_config["op_type"] = op_type
                scheduler_config["op_config"] = dict(
                    warmup_steps=[2],
                    cycle_lengths=[1],
                    f_start=[0.0],
                    f_min=[0.0],
                    f_max=[0.1],
                )
            config = cflearn.Config(
                module_name="linear",
                module_config=dict(input_dim=in_dim, output_dim=out_dim),
                scheduler_name=scheduler,
                scheduler_config=scheduler_config,
                loss_name="mse",
            )
            config.to_debug().num_steps = 10
            if scheduler == "op" and op_type == "foo":
                with self.assertRaises(ValueError):
                    cflearn.TrainingPipeline.init(config).fit(data)
            else:
                cflearn.TrainingPipeline.init(config).fit(data)


if __name__ == "__main__":
    unittest.main()
