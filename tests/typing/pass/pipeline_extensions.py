from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from typing import Dict
    from typing import Optional
    from collections import OrderedDict
    from typing_extensions import assert_type
    from core.learn.schema import IData
    from core.learn.schema import Config
    from core.learn.pipeline.common import Block
    from core.learn.pipeline.common import Pipeline

    @Block.register("typing.block")
    class ExternalBlock(Block):
        def build(self, config: Any) -> None:
            return None

        def process_defaults(self, _defaults: OrderedDict[Any, Any]) -> None:
            return None

        def run(
            self,
            data: IData,
            _defaults: OrderedDict[Any, Any],
            **kwargs: Any,
        ) -> None:
            return None

    @Pipeline.register("typing.pipeline")
    class ExternalPipeline(Pipeline["ExternalPipeline"]):
        pass

    def check_pipeline(config: Config, data: IData) -> None:
        pipeline = ExternalPipeline.init(config)
        block = ExternalBlock()
        pipeline.build(block)
        pipeline.run(data)
        assert_type(pipeline, ExternalPipeline)
        assert_type(pipeline.get_block(ExternalBlock), ExternalBlock)
        assert_type(pipeline.try_get_block(ExternalBlock), Optional[ExternalBlock])
        assert_type(pipeline.block_mappings, Dict[str, Block])
