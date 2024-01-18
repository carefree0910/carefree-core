import torch.nn as nn

from torch import Tensor
from typing import List
from typing import Optional

from .common import register_module
from .activations import build_activation


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
    ):
        super().__init__()
        if hidden_units is None:
            dim = max(32, min(1024, 2 * input_dim))
            hidden_units = 2 * [dim]
        blocks: List[nn.Module] = []
        for hidden_unit in hidden_units:
            i_blocks = [nn.Linear(input_dim, hidden_unit, bias)]
            if batch_norm:
                i_blocks.append(nn.BatchNorm1d(hidden_unit))
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
