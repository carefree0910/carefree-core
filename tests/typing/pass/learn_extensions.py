from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    import torch.nn as nn

    from typing import Any
    from typing import Dict
    from typing import List
    from typing import Type
    from typing import Optional
    from typing_extensions import assert_type
    from core.learn.schema import Config
    from core.learn.schema import IModel
    from core.learn.schema import IMetric
    from core.learn.schema import ITrainer
    from core.learn.schema import TrainStep
    from core.learn.schema import DataBundle
    from core.learn.schema import DataLoader
    from core.learn.schema import IDataBlock
    from core.learn.schema import MetricsOutputs
    from core.learn.schema import TrainerCallback
    from core.toolkit.pipeline import IBlock
    from core.toolkit.types import tensor_dict_type

    @IDataBlock.register("typing.data_block")
    class ExternalDataBlock(IDataBlock["ExternalDataBlock"]):
        def to_info(self) -> Dict[str, Any]:
            return {}

        def transform(
            self,
            bundle: DataBundle,
            for_inference: bool,
        ) -> DataBundle:
            return bundle

        def fit_transform(self, bundle: DataBundle) -> DataBundle:
            return bundle

    @IMetric.register("typing.metric")
    class ExternalMetric(IMetric):
        @property
        def is_positive(self) -> bool:
            return True

        def forward(
            self,
            tensor_batch: tensor_dict_type,
            tensor_outputs: tensor_dict_type,
            loader: Optional[DataLoader] = None,
        ) -> float:
            return 1.0

    @IModel.register("typing.model")
    class ExternalModel(IModel):
        def __init__(self) -> None:
            self.m = nn.Identity()

        @property
        def train_steps(self) -> List[TrainStep]:
            return []

        @property
        def all_modules(self) -> List[nn.Module]:
            return [self.m]

        def build(self, config: Config) -> None:
            self.config = config

    @TrainerCallback.register("typing.callback")
    class ExternalCallback(TrainerCallback):
        def before_loop(self, trainer: ITrainer) -> None:
            return None

    data_block = ExternalDataBlock()
    metric = ExternalMetric()
    model_config = Config()
    model_config.module_name = "typing.model"
    assert_type(model_config.callback_names, Optional[List[str]])
    legacy_config = Config.construct({"callback_names": "typing.callback"})
    assert_type(legacy_config, Config)
    assert_type(legacy_config.from_info({"num_epoch": 1}), Config)
    assert_type(legacy_config.to_info(), Dict[str, Any])
    assert_type(
        Config.from_pack(
            {
                "type": "$base",
                "info": {"callback_names": "typing.callback"},
            }
        ),
        Config,
    )
    assert_type(ExternalDataBlock, Type[ExternalDataBlock])
    assert_type(ExternalDataBlock.make("typing.data_block", {}), IBlock[Any])
    assert_type(data_block.copy(), ExternalDataBlock)
    assert_type(ExternalMetric, Type[ExternalMetric])
    assert_type(IMetric.make("typing.metric", {}), IMetric)
    assert_type(ExternalMetric.make("typing.metric", {}), IMetric)
    assert_type(metric.evaluate({}, {}), Optional[MetricsOutputs])
    assert_type(ExternalModel, Type[ExternalModel])
    assert_type(IModel.make("typing.model", {}), IModel)
    assert_type(ExternalModel.make("typing.model", {}), IModel)
    assert_type(IModel.from_config(model_config), IModel)
    assert_type(ExternalCallback, Type[ExternalCallback])
    assert_type(TrainerCallback.make("typing.callback", {}), TrainerCallback)
    assert_type(
        ExternalCallback.make("typing.callback", {}),
        TrainerCallback,
    )
