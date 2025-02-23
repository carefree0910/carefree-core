import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional
from dataclasses import dataclass

from .schema import build_loss
from .schema import register_loss
from ..schema import ILoss
from ..schema import TrainerState
from ..constants import LOSS_KEY
from ..constants import INPUT_KEY
from ..constants import LABEL_KEY
from ..constants import PREDICTIONS_KEY
from ...toolkit.array import corr
from ...toolkit.types import tensor_dict_type


@register_loss("bce")
class BCELoss(ILoss):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
    ) -> Tensor:
        predictions = forward_results[PREDICTIONS_KEY]
        labels = batch[LABEL_KEY]
        return self.bce(predictions, labels.to(predictions.dtype))


@register_loss("mae")
class MAELoss(ILoss):
    def forward(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
    ) -> Tensor:
        predictions = forward_results[PREDICTIONS_KEY]
        labels = batch[LABEL_KEY]
        return F.l1_loss(predictions, labels)


@register_loss("mse")
class MSELoss(ILoss):
    def forward(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
    ) -> Tensor:
        predictions = forward_results[PREDICTIONS_KEY]
        labels = batch[LABEL_KEY]
        return F.mse_loss(predictions, labels)


@register_loss("corr")
class CorrelationLoss(ILoss):
    def forward(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
    ) -> Tensor:
        predictions = forward_results[PREDICTIONS_KEY]
        labels = batch[LABEL_KEY]
        return -corr(predictions, labels, get_diagonal=True).mean()


def get_stats(predictions: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
    log_prob_mat = F.log_softmax(predictions, dim=1)
    nll_losses = -log_prob_mat.gather(dim=1, index=labels)
    return log_prob_mat, nll_losses


@register_loss("cross_entropy")
class CrossEntropyLoss(ILoss):
    def __init__(self, *, is_auto_regression: bool = False):
        super().__init__()
        self.is_auto_regression = is_auto_regression

    def forward(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
    ) -> Tensor:
        label_key = INPUT_KEY if self.is_auto_regression else LABEL_KEY
        predictions = forward_results[PREDICTIONS_KEY]
        labels = batch[label_key]
        _, nll_losses = get_stats(predictions, labels)
        return nll_losses.mean()


@dataclass
class LossItem:
    name: str
    config: Optional[Dict[str, Any]] = None
    weight: float = 1.0
    tag: Optional[str] = None


@register_loss("multi_loss")
class MultiLoss(ILoss):
    def __init__(self, losses: List[Dict[str, Any]]):
        super().__init__()
        loss_items = [LossItem(**loss) for loss in losses]
        self.losses = nn.ModuleDict(
            {
                loss.tag or loss.name: build_loss(loss.name, config=loss.config)
                for loss in loss_items
            }
        )
        self.weights = {loss.tag or loss.name: loss.weight for loss in loss_items}

    def forward(
        self,
        forward_results: tensor_dict_type,  # type: ignore
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
    ) -> tensor_dict_type:
        loss = 0.0
        losses: tensor_dict_type = {}
        for k, loss_fn in self.losses.items():
            k_losses = loss_fn(forward_results, batch, state)
            if k_losses is None:
                continue
            if isinstance(k_losses, Tensor):
                losses[k] = k_losses
                loss += self.weights[k] * k_losses  # type: ignore
            else:
                k_loss = k_losses.pop(LOSS_KEY)
                loss += self.weights[k] * k_loss
                for kk, vk in k_losses.items():
                    losses[f"{k}_{kk}"] = vk
        losses[LOSS_KEY] = loss
        return losses


__all__ = [
    "BCELoss",
    "MAELoss",
    "MSELoss",
    "CorrelationLoss",
    "CrossEntropyLoss",
    "MultiLoss",
]
