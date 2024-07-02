from abc import abstractmethod
from abc import ABC

from ..schema import DataLoader
from ..schema import InferenceOutputs


class IEvaluationPipeline(ABC):
    @abstractmethod
    def evaluate(self, loader: DataLoader) -> InferenceOutputs:
        """return evaluation results for `loader`"""


__all__ = [
    "IEvaluationPipeline",
]
