import torch

import torch.nn as nn

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import TypeVar
from typing import Callable
from typing import Iterable
from typing import Optional
from torch.nn import Module

from ...toolkit.misc import update_dict
from ...toolkit.misc import parse_config
from ...toolkit.misc import register_core
from ...toolkit.misc import safe_instantiate
from ...toolkit.misc import shallow_copy_dict
from ...toolkit.array import is_int
from ...toolkit.types import TConfig
from ...toolkit.types import tensor_dict_type


# managements


TModule = TypeVar("TModule", bound=Type[Module])
TParams = Iterable[Tuple[str, Union[Tensor, nn.Parameter]]]

module_dict: Dict[str, Type["Module"]] = {}


def register_module(name: str, **kwargs: Any) -> Callable[[TModule], TModule]:
    return register_core(name, module_dict, **kwargs)  # type: ignore


def merge_config(config: TConfig = None, **kwargs: Any) -> Dict[str, Any]:
    config = parse_config(config)
    return update_dict(shallow_copy_dict(kwargs), config)


def build_module(name: str, *, config: TConfig = None, **kwargs: Any) -> Module:
    kwargs = merge_config(config, **kwargs)
    return safe_instantiate(module_dict[name], kwargs)


class PrefixModules:
    def __init__(self, prefix: str) -> None:
        self._prefix = prefix

    @property
    def all(self) -> List[str]:
        return sorted([k for k in module_dict if k.startswith(self._prefix)])

    def has(self, name: str) -> bool:
        return self.prefix(name) in module_dict

    def get(self, name: str) -> Optional[Type[Module]]:
        return module_dict.get(self.prefix(name))

    def register(self, name: str, **kwargs: Any) -> Callable[[TModule], TModule]:
        return register_module(self.prefix(name), **kwargs)

    def build(self, name: str, *, config: TConfig = None, **kwargs: Any) -> Module:
        return build_module(self.prefix(name), config=config, **kwargs)

    def prefix(self, name: str) -> str:
        return f"{self._prefix}.{name}"


# common building blocks


class Lambda(Module):
    def __init__(self, fn: Callable, name: Optional[str] = None):
        super().__init__()
        self.name = name
        self.fn = fn

    def extra_repr(self) -> str:
        return "" if self.name is None else self.name

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


def get_tgt_params(named_parameters: TParams) -> List[Any]:
    return [
        [name.replace(".", "_"), parameter]
        for name, parameter in named_parameters
        if not is_int(parameter.data)
    ]


class EMA(Module):
    num_updates: Optional[Tensor]

    def __init__(
        self,
        decay: float,
        named_parameters: TParams,
        *,
        use_num_updates: bool = False,
    ):
        super().__init__()
        self._cache: tensor_dict_type = {}
        self._decay = decay
        self.tgt_params = get_tgt_params(named_parameters)
        for name, param in self.tgt_params:
            self.register_buffer(name, param.data.clone())
        if not use_num_updates:
            self.num_updates = None
        else:
            self.register_buffer("num_updates", torch.tensor(0, dtype=torch.int))

    def forward(self) -> None:
        if not self.training:
            raise RuntimeError("should not update `EMA` at inference stage")
        if self.num_updates is None:
            decay = self._decay
        else:
            self.num_updates += 1
            decay = min(self._decay, (1 + self.num_updates) / (10 + self.num_updates))
        for name, param in self.tgt_params:
            ema_attr = getattr(self, name)
            ema = torch.lerp(param.data, ema_attr, decay)
            setattr(self, name, ema)

    def train(self, mode: bool = True) -> "EMA":
        super().train(mode)
        if mode:
            for name, param in self.tgt_params:
                cached = self._cache.pop(name, None)
                if cached is not None:
                    param.data.copy_(cached)
        else:
            for name, param in self.tgt_params:
                if name not in self._cache:
                    self._cache[name] = param.data.clone()
                param.data.copy_(getattr(self, name).clone())
        return self

    def rehook(self, named_parameters: TParams) -> None:
        new_params = get_tgt_params(named_parameters)
        self.tgt_params = [[a[0], b[1]] for a, b in zip(self.tgt_params, new_params)]

    def extra_repr(self) -> str:
        max_str_len = max(len(name) for name, _ in self.tgt_params)
        return "\n".join(
            [f"(0): decay_rate={self._decay}\n(1): Params("]
            + [
                f"  {name:<{max_str_len}s} - Tensor({list(param.shape)})"
                for name, param in self.tgt_params
            ]
            + [")"]
        )

    @classmethod
    def hook(cls, m: Module, decay: float, *, use_num_updates: bool = False) -> "EMA":
        return cls(decay, m.state_dict().items(), use_num_updates=use_num_updates)


# common structures


class Residual(Module):
    def __init__(self, module: Module):
        super().__init__()
        self.module = module

    def forward(self, net: Tensor, **kwargs: Any) -> Tensor:
        return net + self.module(net, **kwargs)


def zero_module(module: Module) -> Module:
    for p in module.parameters():
        p.detach().zero_()
    return module


def avg_pool_nd(n: int, *args: Any, **kwargs: Any) -> Module:
    if n == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif n == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif n == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {n}")


# shortcuts


class BN(nn.BatchNorm1d):
    def forward(self, net: Tensor) -> Tensor:
        if len(net.shape) == 3:
            net = net.transpose(1, 2)
        net = super().forward(net)
        if len(net.shape) == 3:
            net = net.transpose(1, 2)
        return net


__all__ = [
    "module_dict",
    "register_module",
    "build_module",
    "PrefixModules",
    "Lambda",
    "EMA",
    "Residual",
    "zero_module",
    "avg_pool_nd",
    "BN",
]
