from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from core.learn.pipeline.common import Block

    class BadBlock(Block):
        def build(self, config: Any) -> int:  # expected-mypy: override
            return 0
