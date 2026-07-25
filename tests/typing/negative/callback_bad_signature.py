from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.learn.schema import ITrainer
    from core.learn.schema import TrainerCallback

    class BadCallback(TrainerCallback):
        def before_loop(
            self,
            trainer: int,  # expected-mypy: override
        ) -> None:
            return None
