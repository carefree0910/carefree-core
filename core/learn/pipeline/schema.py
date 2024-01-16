from abc import abstractmethod
from abc import ABC

from ..schema import DataLoader
from ..schema import MetricsOutputs


class IEvaluationPipeline(ABC):
    @abstractmethod
    def evaluate(self, loader: DataLoader) -> MetricsOutputs:
        pass


__all__ = [
    "IEvaluationPipeline",
]
