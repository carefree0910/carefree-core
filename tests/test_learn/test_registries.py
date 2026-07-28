import sys
import torch
import subprocess

import torch.nn as nn
import core.learn as cflearn
import core.learn.schema as learn_schema
import core.learn.optimizers as learn_optimizers
import core.learn.schedulers as learn_schedulers
import core.learn.modules.common as learn_modules

from unittest.mock import patch


def test_public_learn_registries_replace_legacy_dicts() -> None:
    assert cflearn.module_registry is learn_modules.module_registry
    assert cflearn.optimizer_registry is learn_optimizers.optimizer_registry
    assert cflearn.scheduler_registry is learn_schedulers.scheduler_registry
    assert cflearn.scheduler_op_registry is learn_schedulers.scheduler_op_registry
    assert not hasattr(learn_modules, "module_dict")
    assert not hasattr(learn_optimizers, "optimizer_dict")
    assert not hasattr(learn_schedulers, "scheduler_dict")
    assert not hasattr(learn_schedulers, "scheduler_ops")
    assert not hasattr(cflearn, "module_dict")
    assert not hasattr(cflearn, "optimizer_dict")
    assert not hasattr(cflearn, "scheduler_dict")
    assert not hasattr(cflearn, "scheduler_ops")
    assert learn_schema.DataConfig.d is learn_schema.data_configs
    assert learn_schema.Config.d is learn_schema.configs


def test_learn_registry_decorators_and_factories() -> None:
    module_name = "$phase3.module"

    @cflearn.register_module(module_name)
    class CustomModule(nn.Module):
        def __init__(self, value: int = 0) -> None:
            super().__init__()
            self.value = value

    assert learn_modules.module_registry.get(module_name) is CustomModule
    module = cflearn.build_module(module_name, value=1, ignored=True)
    assert isinstance(module, CustomModule)
    assert module.value == 1

    loss_name = "$phase3.loss"

    @cflearn.register_loss(loss_name)
    class CustomLoss(cflearn.ILoss):
        def __init__(self, value: int = 0) -> None:
            super().__init__()
            self.value = value

        def forward(self, forward_results, batch, state=None):
            return torch.tensor(float(self.value))

    assert learn_modules.module_registry.get(f"loss.{loss_name}") is CustomLoss
    assert CustomLoss.__identifier__ == loss_name
    loss = cflearn.build_loss(loss_name, value=2, ignored=True)
    assert isinstance(loss, CustomLoss)
    assert loss.value == 2

    optimizer_name = "$phase3.optimizer"

    @cflearn.register_optimizer(optimizer_name)
    class CustomOptimizer(torch.optim.SGD):
        pass

    parameter = nn.Parameter(torch.ones(1))
    optimizer = learn_optimizers.optimizer_registry.make(
        optimizer_name,
        {"params": [parameter], "lr": 0.1},
    )
    assert learn_optimizers.optimizer_registry.get(optimizer_name) is CustomOptimizer
    assert isinstance(optimizer, CustomOptimizer)

    scheduler_name = "$phase3.scheduler"

    @cflearn.register_scheduler(scheduler_name)
    class CustomScheduler(torch.optim.lr_scheduler.StepLR):
        pass

    scheduler = learn_schedulers.scheduler_registry.make(
        scheduler_name,
        {"optimizer": optimizer, "step_size": 1},
    )
    assert learn_schedulers.scheduler_registry.get(scheduler_name) is CustomScheduler
    assert isinstance(scheduler, CustomScheduler)

    op_name = "$phase3.op"

    @cflearn.register_op(op_name)
    class CustomOp(learn_schedulers.ISchedulerOp):
        def __init__(self, scale: float) -> None:
            self.scale = scale

        def schedule(self, step: int, **kwargs) -> float:
            return self.scale * step

    op = learn_schedulers.scheduler_op_registry.make(op_name, {"scale": 0.5})
    assert learn_schedulers.scheduler_op_registry.get(op_name) is CustomOp
    assert isinstance(op, CustomOp)
    assert op.schedule(4) == 2.0


def test_module_registry_adapter_keeps_hook_and_duplicate_behavior() -> None:
    module_name = "$phase3.hooks"
    events = []

    def before_register(module_cls) -> None:
        events.append(("before", module_cls))

    def after_register(module_cls) -> None:
        events.append(("after", module_cls))

    @cflearn.register_module(
        module_name,
        before_register=before_register,
        after_register=after_register,
    )
    class OriginalModule(nn.Module):
        pass

    with patch("core.learn.modules.common.console.warn") as warn:

        @cflearn.register_module(
            module_name,
            before_register=before_register,
            after_register=after_register,
        )
        class IgnoredDuplicate(nn.Module):
            pass

    warn.assert_called_once()
    assert learn_modules.module_registry.get(module_name) is OriginalModule
    assert events == [
        ("before", OriginalModule),
        ("after", OriginalModule),
        ("before", IgnoredDuplicate),
    ]

    @cflearn.register_module(
        module_name,
        allow_duplicate=True,
        before_register=before_register,
        after_register=after_register,
    )
    class ReplacedModule(nn.Module):
        pass

    assert learn_modules.module_registry.get(module_name) is ReplacedModule
    assert events[-2:] == [("before", ReplacedModule), ("after", ReplacedModule)]


def test_prefix_modules_resolve_registry_aliases() -> None:
    target = "$phase3.alias_target"
    alias = "$phase3.alias"

    class AliasLoss(cflearn.ILoss):
        def forward(self, forward_results, batch, state=None):
            return torch.tensor(0.0)

    with learn_modules.module_registry.isolated():
        learn_modules.module_registry.register(f"loss.{target}", AliasLoss)
        learn_modules.module_registry.register_alias(f"loss.{alias}", f"loss.{target}")
        assert cflearn.losses.has(alias)
        assert cflearn.losses.get(alias) is AliasLoss
        assert isinstance(cflearn.build_loss(alias), AliasLoss)


def test_minimal_import_registers_learn_builtins() -> None:
    code = "\n".join(
        [
            "import core.learn as cflearn",
            "assert 'linear' in cflearn.module_registry",
            "assert 'loss.mse' in cflearn.module_registry",
            "assert 'adam' in cflearn.optimizer_registry",
            "assert 'cyclic' in cflearn.scheduler_registry",
            "assert 'cosine_warmup' in cflearn.scheduler_op_registry",
            "assert not hasattr(cflearn, 'module_dict')",
            "assert not hasattr(cflearn, 'optimizer_dict')",
            "assert not hasattr(cflearn, 'scheduler_dict')",
            "assert not hasattr(cflearn, 'scheduler_ops')",
            "assert cflearn.DataConfig.d['$base'] is cflearn.DataConfig",
            "assert cflearn.Config.d['$base'] is cflearn.Config",
        ]
    )
    subprocess.run([sys.executable, "-c", code], check=True)
