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
from ..schema import normalize_loss_result
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
        if not loss_items:
            raise ValueError("at least one loss should be provided")
        tags = [loss.tag or loss.name for loss in loss_items]
        if LOSS_KEY in tags:
            raise ValueError(f"'{LOSS_KEY}' is reserved for the primary loss")
        if len(tags) != len(set(tags)):
            raise ValueError("loss tags should be unique")
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
    ) -> Optional[tensor_dict_type]:
        loss: Optional[Tensor] = None
        losses: tensor_dict_type = {}
        for k, loss_fn in self.losses.items():
            k_losses = loss_fn(forward_results, batch, state)
            loss_result = normalize_loss_result(k_losses)
            if loss_result is None:
                continue
            weighted = self.weights[k] * loss_result.primary
            loss = weighted if loss is None else loss + weighted
            if isinstance(k_losses, Tensor):
                i_losses = {k: loss_result.primary}
            else:
                i_losses = {
                    f"{k}_{kk}": vk for kk, vk in loss_result.components.items()
                }
            duplicated = set(losses).intersection(i_losses)
            if duplicated:
                raise ValueError(
                    f"duplicated flattened loss keys: {sorted(duplicated)}"
                )
            losses.update(i_losses)
        if loss is None:
            return None
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
