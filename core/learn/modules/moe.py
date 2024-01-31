# 2D ([batch_size, dim]) Mixture of Experts (MoE) module
# We restrict inputs to be 2D because indexing will be much faster

import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Any
from typing import List
from typing import Type
from typing import TypeVar
from typing import Callable
from typing import Optional
from typing import NamedTuple
from torch.nn import Module
from torch.distributions.normal import Normal

from .common import build_module
from .common import PrefixModules
from ..constants import PREDICTIONS_KEY
from ...toolkit.types import TConfig
from ...toolkit.types import tensor_dict_type


# router


TRouter = TypeVar("TRouter", bound=Type["IMoERouter"])

moe_routers = PrefixModules("moe_routers")


class IMoERouter(Module):
    dim: int
    num_experts: int


class MoERouterOutputs(NamedTuple):
    logits: Tensor
    clean_logits: Tensor
    noise_std: Optional[Tensor]


def register_moe_router(name: str, **kwargs: Any) -> Callable[[TRouter], TRouter]:
    return moe_routers.register(name, **kwargs)


def build_moe_router(name: str, *, config: TConfig = None, **kwargs: Any) -> IMoERouter:
    return moe_routers.build(name, config=config, **kwargs)


@register_moe_router("basic")
class MoEBasicRouter(IMoERouter):
    def __init__(
        self,
        *,
        dim: int,
        num_experts: int,
        noisy_gating: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.w_gate = nn.Parameter(torch.zeros(dim, num_experts))
        if not noisy_gating:
            self.w_noise = None
        else:
            self.w_noise = nn.Parameter(torch.zeros(dim, num_experts))

    def forward(self, net: Tensor, noise_eps: float = 1e-2) -> MoERouterOutputs:
        logits = clean_logits = net @ self.w_gate
        if not self.training or self.w_noise is None:
            noise_std = None
        else:
            noise_std = net @ self.w_noise
            noise_std = F.softplus(noise_std) + noise_eps
            logits = logits + (torch.randn_like(logits) * noise_std)
        return MoERouterOutputs(logits, clean_logits, noise_std)


# dispatcher


class MoEDispatchOutputs(NamedTuple):
    gates: Tensor
    inputs: List[Tensor]
    indices: Tensor
    gates_gahtered: Tensor
    load: Tensor


class MoEDispatcher(Module):
    mean: Tensor
    std: Tensor

    def __init__(self, *, top_k: int, num_experts: int):
        super().__init__()
        if top_k > num_experts:
            raise ValueError(f"top_k={top_k} must be <= num_experts={num_experts}")
        self.k = top_k
        self.num_experts = num_experts
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

    def dispatch(self, net: Tensor, routings: MoERouterOutputs) -> MoEDispatchOutputs:
        # [B, E]
        logits, clean_logits, noise_std = routings
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[..., : self.k]
        top_k_indices = top_indices[..., : self.k]
        top_k_gates = F.softmax(top_k_logits, dim=1)

        # [B, E]
        zeros = torch.zeros_like(logits)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        # [E]
        part_sizes = (gates > 0).sum(0)

        # [E]
        if noise_std is None or self.k >= self.num_experts or not self.training:
            load = part_sizes.to(net)
        else:
            in_threshold = top_logits[..., self.k : self.k + 1]
            out_threshold = top_logits[..., self.k - 1 : self.k]
            normal = Normal(self.mean, self.std)
            prob_if_in = normal.cdf((clean_logits - in_threshold) / noise_std)
            prob_if_out = normal.cdf((clean_logits - out_threshold) / noise_std)
            prob = torch.where(logits > in_threshold, prob_if_in, prob_if_out)
            load = prob.sum(0)

        # [B * k, 2]
        nonzero_gates = torch.nonzero(gates)
        sorted_gates, sorted_gates_indices = nonzero_gates.sort(0)
        # [B * k]
        sorted_expert_indices = sorted_gates_indices[..., 1]
        batch_indices = nonzero_gates[sorted_expert_indices, 0]
        # E * [?, D]
        inputs = torch.split(net[batch_indices], part_sizes.tolist(), dim=0)
        # [B * k, E]
        gates_expand = gates[batch_indices]
        # [B * k, 1]
        gates_gathered = torch.gather(gates_expand, 1, sorted_gates[..., 1:])
        return MoEDispatchOutputs(gates, inputs, batch_indices, gates_gathered, load)

    def combine(self, outputs: List[Tensor], indices: Tensor, gates: Tensor) -> Tensor:
        # [B * k, D]
        net = torch.cat(outputs, dim=0)
        net = net * gates
        batch_size = indices.shape[0] // self.k
        zeros = net.new_zeros(batch_size, net.shape[1])
        net = zeros.index_add_(0, indices, net)
        return net


# MoE


def squared_coef(net: Tensor) -> Tensor:
    if net.shape[0] == 1:
        return net.new_zeros([1])
    return net.var() / (net.mean() ** 2 + 1.0e-10)


def get_load_balance_loss(forward_results: tensor_dict_type) -> Tensor:
    importances = forward_results.get("importances")
    load = forward_results.get("load")
    if importances is None:
        raise ValueError(
            f"`importances`, which should be `gates.sum(0)`, is not found in "
            f"`forward_results` ({forward_results})"
        )
    if load is None:
        raise ValueError(
            f"`load`, which should be calculated in the `dispatch` method, "
            f"is not found in `forward_results` ({forward_results})"
        )
    return squared_coef(importances) + squared_coef(load)


class MoE(Module):
    def __init__(
        self,
        expert_name: str,
        expert_config: TConfig = None,
        *,
        dim: int,
        top_k: int,
        num_experts: int,
        num_common_experts: int = 0,
        router: str = "basic",
        router_config: TConfig = None,
    ) -> None:
        super().__init__()
        build = lambda: build_module(expert_name, config=expert_config)
        self.experts = nn.ModuleList([build() for _ in range(num_experts)])
        if num_common_experts <= 0:
            self.commons = None
        else:
            self.commons = nn.ModuleList([build() for _ in range(num_common_experts)])
        router_kwargs = dict(dim=dim, num_experts=num_experts)
        self.router = build_moe_router(router, config=router_config, **router_kwargs)
        self.dispatcher = MoEDispatcher(top_k=top_k, num_experts=num_experts)

    def forward(
        self,
        net: Tensor,
        *,
        noise_eps: float = 1e-2,
        **kwargs: Any,
    ) -> tensor_dict_type:
        inp = net
        routings = self.router(net, noise_eps)
        dispatch = self.dispatcher.dispatch(net, routings)
        outs = [m(inp, **kwargs) for m, inp in zip(self.experts, dispatch.inputs)]
        net = self.dispatcher.combine(outs, dispatch.indices, dispatch.gates_gahtered)
        if self.commons is not None:
            for m in self.commons:
                net += m(inp, **kwargs)
        return {
            PREDICTIONS_KEY: net,
            "importances": dispatch.gates.sum(0),
            "load": dispatch.load,
        }


__all__ = [
    "register_moe_router",
    "build_moe_router",
    "get_load_balance_loss",
    "IMoERouter",
    "MoERouterOutputs",
    "MoEDispatchOutputs",
    "MoEDispatcher",
    "MoE",
]
