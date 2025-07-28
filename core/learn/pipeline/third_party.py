import torch

import numpy as np

from abc import abstractmethod
from abc import ABC

from .schema import IEvaluationPipeline
from .blocks import BuildMetricsBlock
from ..schema import Config
from ..schema import DataLoader
from ..schema import InferenceOutputs
from ..toolkit import tensor_batch_to_np
from ..constants import INPUT_KEY
from ..constants import LABEL_KEY
from ..constants import PREDICTIONS_KEY


class IPredictor(ABC):
    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """return predictions for input `x`"""


class GeneralEvaluationPipeline(IEvaluationPipeline):
    def __init__(self, config: Config, predictor: IPredictor) -> None:
        b_metrics = BuildMetricsBlock()
        b_metrics.build(config)
        if b_metrics.metrics is None:
            raise ValueError(
                "`metrics` should not be `None` for `GeneralPredictor`, "
                "you may try specifying `metric_names` in `config`"
            )
        self.m = predictor
        self.metrics = b_metrics.metrics

    def evaluate(self, loader: DataLoader) -> InferenceOutputs:
        tensor_batch = loader.get_full_batch()
        np_batch = tensor_batch_to_np(tensor_batch)
        predictions = self.m.predict(np_batch[INPUT_KEY])
        np_results = {PREDICTIONS_KEY: predictions}
        tensor_results = {PREDICTIONS_KEY: torch.from_numpy(predictions)}
        metric_outputs = self.metrics.evaluate(tensor_batch, tensor_results, loader)
        return InferenceOutputs(
            np_results,
            {LABEL_KEY: np_batch[LABEL_KEY]},
            metric_outputs,
            loss_items=None,
        )


__all__ = [
    "IPredictor",
    "GeneralEvaluationPipeline",
]
