import numpy as np

from typing import List
from typing import Optional

from .schema import TrainerState
from .schema import TrainerMonitor
from ..toolkit.misc import Incrementer


@TrainerMonitor.register("basic")
class BasicMonitor(TrainerMonitor):
    def __init__(self, *, num_keep: int = 25):
        super().__init__()
        if num_keep <= 0:
            raise ValueError("`num_keep` should be a positive integer")
        self.history: List[float] = []
        self.num_keep = num_keep
        self.worst_score: Optional[float] = None
        self.start_terminate = False

    def should_snapshot(self, new_score: float) -> bool:
        if self.worst_score is None:
            self.worst_score = new_score
        else:
            self.worst_score = min(new_score, self.worst_score)
        if len(self.history) < self.num_keep:
            self.history.append(new_score)
            return True
        self.start_terminate = True
        min_idx = np.argmin(self.history).item()
        if new_score >= self.history[min_idx]:
            self.history[min_idx] = new_score
            return True
        return False

    def should_terminate(self, new_score: float) -> bool:
        if not self.start_terminate or self.worst_score is None:
            return False
        return new_score <= self.worst_score


@TrainerMonitor.register("mean_std")
class MeanStdMonitor(BasicMonitor):
    def __init__(
        self,
        *,
        num_keep: int = 25,
        patience: float = 5.0,
        window_size: int = 25,
        overfit_tolerance: float = 25.0,
    ):
        super().__init__(num_keep=num_keep)
        self.patience = patience
        self.overfit_tolerance = overfit_tolerance
        self.overfit_level = 0.0
        self._incrementer = Incrementer(window_size)

    def should_snapshot(self, new_score: float) -> bool:
        if len(self.history) >= self.num_keep:
            mean, std = self._incrementer.mean, self._incrementer.std
            std = max(std, 1.0e-8)
            if new_score < mean - std:
                max_decrease = self.overfit_tolerance / self.patience
                decrease = min(max_decrease, (mean - new_score) / std + 1.0)
                self.overfit_level += decrease
            elif new_score > mean + std:
                improvement = (new_score - mean) / std - 1.0
                self.overfit_level = max(0.0, self.overfit_level - improvement)
        self._incrementer.update(new_score)
        return super().should_snapshot(new_score)

    def should_terminate(self, new_score: float) -> bool:
        if super().should_terminate(new_score):
            return True
        if self.overfit_level >= self.overfit_tolerance:
            return True
        return False


@TrainerMonitor.register("plateau")
class PlateauMonitor(BasicMonitor):
    def __init__(
        self,
        *,
        num_keep: int = 25,
        patience: float = 5.0,
        extension: int = 5,
        window_size: int = 25,
        plateau_tolerance: float = 25.0,
        plateau_threshold: float = 0.2,
    ):
        super().__init__(num_keep=num_keep)
        self.patience = patience
        self.extension = extension
        self.window_size = window_size
        self.plateau_tolerance = plateau_tolerance
        self.plateau_threshold = plateau_threshold
        self.plateau_level = 0.0
        self._incrementer = Incrementer(window_size)

    @property
    def max_plateau_increase(self) -> float:
        return self.plateau_tolerance / self.patience

    def should_terminate(self, new_score: float) -> bool:
        if super().should_terminate(new_score):
            return True
        self._incrementer.update(new_score)
        if self._incrementer.num_record > self.window_size:
            mean, std = self._incrementer.mean, self._incrementer.std
            ratio = max(abs(new_score - mean) / max(std, 1.0e-8), 1.0e-8)
            if ratio < self.plateau_threshold:
                plateau = min(
                    self.max_plateau_increase,
                    1.0 / ratio - 1.0 / self.plateau_threshold,
                )
                self.plateau_level += plateau
        return self.plateau_level >= self.plateau_tolerance

    def punish_extension(self) -> None:
        self.plateau_level += self.max_plateau_increase / 5.0

    def handle_extension(self, state: TrainerState) -> int:
        return self.extension


@TrainerMonitor.register("conservative")
class ConservativeMonitor(TrainerMonitor):
    def should_snapshot(self, new_score: float) -> bool:
        return True

    def should_terminate(self, new_score: float) -> bool:
        return False


@TrainerMonitor.register("lazy")
class LazyMonitor(TrainerMonitor):
    def should_snapshot(self, new_score: float) -> bool:
        return False

    def should_terminate(self, new_score: float) -> bool:
        return False


__all__ = [
    "BasicMonitor",
    "MeanStdMonitor",
    "PlateauMonitor",
    "ConservativeMonitor",
    "LazyMonitor",
]
