import numpy as np

from typing import Optional

from .schema import IMetric
from .schema import DataLoader
from .constants import LABEL_KEY
from .constants import PREDICTIONS_KEY
from ..toolkit.array import corr
from ..toolkit.array import to_labels
from ..toolkit.types import np_dict_type


@IMetric.register("acc")
class Accuracy(IMetric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    @property
    def is_positive(self) -> bool:
        return True

    def forward(
        self,
        np_batch: np_dict_type,
        np_outputs: np_dict_type,
        loader: Optional[DataLoader] = None,
    ) -> float:
        logits = np_outputs[PREDICTIONS_KEY]
        labels = np_batch[LABEL_KEY]
        predictions = to_labels(logits, self.threshold)
        return (predictions == labels).mean().item()


@IMetric.register("mae")
class MAE(IMetric):
    @property
    def is_positive(self) -> bool:
        return False

    def forward(
        self,
        np_batch: np_dict_type,
        np_outputs: np_dict_type,
        loader: Optional[DataLoader] = None,
    ) -> float:
        predictions = np_outputs[PREDICTIONS_KEY]
        labels = np_batch[LABEL_KEY]
        return np.mean(np.abs(labels - predictions)).item()


@IMetric.register("mse")
class MSE(IMetric):
    @property
    def is_positive(self) -> bool:
        return False

    def forward(
        self,
        np_batch: np_dict_type,
        np_outputs: np_dict_type,
        loader: Optional[DataLoader] = None,
    ) -> float:
        predictions = np_outputs[PREDICTIONS_KEY]
        labels = np_batch[LABEL_KEY]
        return np.mean(np.square(labels - predictions)).item()


@IMetric.register("corr")
class Correlation(IMetric):
    @property
    def is_positive(self) -> bool:
        return True

    def forward(
        self,
        np_batch: np_dict_type,
        np_outputs: np_dict_type,
        loader: Optional[DataLoader] = None,
    ) -> float:
        predictions = np_outputs[PREDICTIONS_KEY]
        labels = np_batch[LABEL_KEY]
        return corr(predictions, labels, get_diagonal=True).mean().item()


__all__ = [
    "MAE",
    "MSE",
    "Accuracy",
    "Correlation",
]
