import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional

from .schema import register_loss
from ..schema import ILoss
from ..schema import TrainerState
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
        forward_results: tensor_dict_type,  # type: ignore
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
    ) -> Tensor:
        predictions = forward_results[PREDICTIONS_KEY]
        labels = batch[LABEL_KEY]
        return self.bce(predictions, labels.to(predictions.dtype))


@register_loss("mse")
class MSELoss(ILoss):
    def forward(
        self,
        forward_results: tensor_dict_type,  # type: ignore
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
        forward_results: tensor_dict_type,  # type: ignore
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
    ) -> Tensor:
        predictions = forward_results[PREDICTIONS_KEY]
        labels = batch[LABEL_KEY]
        return -corr(predictions, labels, get_diagonal=True).mean()


__all__ = [
    "BCELoss",
    "MSELoss",
    "CorrelationLoss",
]
