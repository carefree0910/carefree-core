from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.learn.schema import DataLoader
    from core.learn.pipeline.api import InferencePipeline

    def check_prediction_mode(
        pipeline: InferencePipeline,
        loader: DataLoader,
    ) -> None:
        pipeline.predict(loader, prediction_mode="unknown")  # expected-mypy: arg-type
