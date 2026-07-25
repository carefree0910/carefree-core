from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.learn.schema import Config
    from core.learn.pipeline.common import Pipeline

    class BadPipeline(Pipeline["BadPipeline"]):
        @classmethod
        def init(  # expected-mypy: override, override
            cls,
            config: Config,
        ) -> int:
            return 0
