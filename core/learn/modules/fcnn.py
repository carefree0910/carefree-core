import math
import torch

import torch.nn as nn

from torch import Tensor
from typing import List
from typing import Optional

from .common import register_module
from .common import BN
from .activations import build_activation


class Linear(nn.Linear):
    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.xavier_normal_(self.weight.data, 1.0 / math.sqrt(2.0))
            if self.bias is not None:
                self.bias.data.zero_()


class HighwayBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        bias: Optional[bool] = None,
        batch_norm: bool = True,
        activation: Optional[str] = "ReLU",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.linear = Linear(in_dim, out_dim, bias=True if bias is None else bias)
        self.nonlinear_mapping = nn.Sequential(
            Linear(in_dim, out_dim, bias=True if bias is None else bias),
            BN(out_dim) if batch_norm else nn.Identity(),
            build_activation(activation) if activation is not None else nn.Identity(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )
        self.gate_linear = Linear(in_dim, out_dim, bias=True if bias is None else bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, net: Tensor) -> Tensor:
        linear = self.linear(net)
        nonlinear = self.nonlinear_mapping(net)
        gate = self.sigmoid(self.gate_linear(net))
        return gate * nonlinear + (1.0 - gate) * linear


@register_module("fcnn")
class FCNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_units: Optional[List[int]] = None,
        *,
        bias: bool = True,
        batch_norm: bool = False,
        activation: str = "ReLU",
        dropout: float = 0.0,
        highway: bool = False,
    ):
        super().__init__()
        if hidden_units is None:
            dim = max(32, min(1024, 2 * input_dim))
            hidden_units = 2 * [dim]
        blocks: List[nn.Module] = []
        for hidden_unit in hidden_units:
            if highway:
                blocks.append(
                    HighwayBlock(
                        input_dim,
                        hidden_unit,
                        bias=bias,
                        batch_norm=batch_norm,
                        activation=activation,
                        dropout=dropout,
                    )
                )
            else:
                i_blocks: List[nn.Module] = [nn.Linear(input_dim, hidden_unit, bias)]
                if batch_norm:
                    i_blocks.append(BN(hidden_unit))
                i_blocks.append(build_activation(activation))
                if dropout > 0.0:
                    i_blocks.append(nn.Dropout(dropout))
                blocks.append(nn.Sequential(*i_blocks))
            input_dim = hidden_unit
        blocks.append(nn.Linear(input_dim, output_dim, bias))
        self.hidden_units = hidden_units
        self.net = nn.Sequential(*blocks)

    def forward(self, net: Tensor) -> Tensor:
        return self.net(net)


__all__ = [
    "FCNN",
]
