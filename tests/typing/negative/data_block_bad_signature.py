from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from typing import Dict
    from core.learn.schema import DataBundle
    from core.learn.schema import IDataBlock

    class BadDataBlock(IDataBlock["BadDataBlock"]):
        def to_info(self) -> Dict[str, Any]:
            return {}

        def transform(  # expected-mypy: override
            self,
            bundle: DataBundle,
            for_inference: bool,
        ) -> str:
            return ""

        def fit_transform(self, bundle: DataBundle) -> DataBundle:
            return bundle
