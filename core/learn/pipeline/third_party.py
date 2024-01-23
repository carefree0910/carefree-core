import numpy as np

from abc import abstractmethod
from abc import ABC

from .schema import IEvaluationPipeline
from .blocks import BuildMetricsBlock
from ..schema import Config
from ..schema import DataLoader
from ..schema import MetricsOutputs
from ..toolkit import tensor_batch_to_np
from ..constants import INPUT_KEY
from ..constants import PREDICTIONS_KEY


class IPredictor(ABC):
    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        pass


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

    def evaluate(self, loader: DataLoader) -> MetricsOutputs:
        full_batch = tensor_batch_to_np(loader.get_full_batch())
        predictions = self.m.predict(full_batch[INPUT_KEY])
        return self.metrics.evaluate(full_batch, {PREDICTIONS_KEY: predictions}, loader)


__all__ = [
    "IPredictor",
    "GeneralEvaluationPipeline",
]
