from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional
    from core.learn.schema import IMetric
    from core.learn.schema import DataLoader
    from core.toolkit.types import tensor_dict_type

    class BadMetric(IMetric):
        @property
        def is_positive(self) -> bool:
            return True

        def forward(  # expected-mypy: override
            self,
            tensor_batch: tensor_dict_type,
            tensor_outputs: tensor_dict_type,
            loader: Optional[DataLoader] = None,
        ) -> str:
            return ""
