import torch

from typing import Optional

from .schema import IMetric
from .schema import DataLoader
from .schema import IStreamMetric
from .constants import LABEL_KEY
from .constants import PREDICTIONS_KEY
from ..toolkit.array import corr
from ..toolkit.array import to_labels
from ..toolkit.types import tensor_dict_type


@IMetric.register("acc")
class Accuracy(IMetric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    @property
    def is_positive(self) -> bool:
        return True

    def forward(
        self,
        tensor_batch: tensor_dict_type,
        tensor_outputs: tensor_dict_type,
        loader: Optional[DataLoader] = None,
    ) -> float:
        logits = tensor_outputs[PREDICTIONS_KEY].numpy()
        labels = tensor_batch[LABEL_KEY].numpy()
        predictions = to_labels(logits, self.threshold)
        return (predictions == labels).mean().item()


@IMetric.register("mae")
class MAE(IMetric):
    @property
    def is_positive(self) -> bool:
        return False

    def forward(
        self,
        tensor_batch: tensor_dict_type,
        tensor_outputs: tensor_dict_type,
        loader: Optional[DataLoader] = None,
    ) -> float:
        predictions = tensor_outputs[PREDICTIONS_KEY]
        labels = tensor_batch[LABEL_KEY]
        return torch.mean(torch.abs(labels - predictions)).item()


@IMetric.register("mse")
class MSE(IMetric):
    @property
    def is_positive(self) -> bool:
        return False

    def forward(
        self,
        tensor_batch: tensor_dict_type,
        tensor_outputs: tensor_dict_type,
        loader: Optional[DataLoader] = None,
    ) -> float:
        predictions = tensor_outputs[PREDICTIONS_KEY]
        labels = tensor_batch[LABEL_KEY]
        return torch.mean(torch.square(labels - predictions)).item()


@IMetric.register("corr")
class Correlation(IMetric):
    @property
    def is_positive(self) -> bool:
        return True

    def forward(
        self,
        tensor_batch: tensor_dict_type,
        tensor_outputs: tensor_dict_type,
        loader: Optional[DataLoader] = None,
    ) -> float:
        predictions = tensor_outputs[PREDICTIONS_KEY]
        labels = tensor_batch[LABEL_KEY]
        return corr(predictions, labels, get_diagonal=True).mean().item()


@IMetric.register("stream_mse")
class StreamMSE(IStreamMetric):
    error: float
    num: int

    @property
    def is_positive(self) -> bool:
        return False

    def reset(self) -> None:
        self.error = 0.0
        self.num = 0

    def update(
        self,
        tensor_batch: tensor_dict_type,
        tensor_outputs: tensor_dict_type,
        loader: Optional[DataLoader] = None,
    ) -> None:
        predictions = tensor_outputs[PREDICTIONS_KEY]
        labels = tensor_batch[LABEL_KEY]
        self.error += torch.sum(torch.square(labels - predictions)).item()
        self.num += labels.numel()

    def finalize(self) -> float:
        return self.error / self.num if self.num > 0 else 0.0


__all__ = [
    "MAE",
    "MSE",
    "Accuracy",
    "Correlation",
    "StreamMSE",
]
