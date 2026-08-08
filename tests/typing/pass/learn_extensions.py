from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    import torch.nn as nn

    from typing import Any
    from typing import Dict
    from typing import List
    from typing import Type
    from typing import Tuple
    from typing import Union
    from typing import Optional
    from accelerate import Accelerator
    from typing_extensions import assert_type
    from core.learn.schema import prepare_dataloaders
    from core.learn.schema import normalize_loss_result

    from core.learn import Inference
    from core.learn import StreamMSE
    from core.learn.schema import ILoss
    from core.learn.schema import Config
    from core.learn.schema import IModel
    from core.learn.schema import IMetric
    from core.learn.schema import ITrainer
    from core.learn.schema import TrainStep
    from core.learn.schema import DataBundle
    from core.learn.schema import DataConfig
    from core.learn.schema import DataLoader
    from core.learn.schema import DLSettings
    from core.learn.schema import IDataBlock
    from core.learn.schema import ClosurePack
    from core.learn.schema import MetricResult
    from core.learn.schema import MetricValues
    from core.learn.schema import IAsyncDataset
    from core.learn.schema import IStreamMetric
    from core.learn.schema import TrainStepLoss
    from core.learn.schema import MetricsOutputs
    from core.learn.schema import LoggingSettings
    from core.learn.schema import RuntimeSettings
    from core.learn.schema import TrainerCallback
    from core.learn.schema import InferenceOutputs
    from core.learn.schema import ClosurePackStepFn
    from core.learn.schema import GetBackwardLossFn
    from core.learn.schema import MetricAccumulator
    from core.learn.schema import EvaluationSettings
    from core.learn.schema import WillSkipBackwardFn
    from core.learn.schema import DistributedSettings
    from core.learn.schema import PersistenceSettings
    from core.learn.schema import OptimizationSettings
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

    @IMetric.register("typing.metric_values")
    class ExternalMetricValues(IMetric):
        @property
        def is_positive(self) -> bool:
            return True

        def forward(
            self,
            tensor_batch: tensor_dict_type,
            tensor_outputs: tensor_dict_type,
            loader: Optional[DataLoader] = None,
        ) -> MetricValues:
            return MetricValues(
                {"score": 1.0},
                {"score": True},
            )

    @IMetric.register("typing.stream_metric")
    class ExternalStreamMetric(IStreamMetric[Dict[str, float]]):
        @property
        def is_positive(self) -> bool:
            return False

        def reset(self) -> None:
            return None

        def update(
            self,
            tensor_batch: tensor_dict_type,
            tensor_outputs: tensor_dict_type,
            loader: Optional[DataLoader] = None,
        ) -> None:
            return None

        def finalize(self) -> float:
            return 0.0

    class ExternalMergeableStreamMetric(ExternalStreamMetric):
        def get_distributed_state(self) -> Dict[str, float]:
            return {}

        def merge_distributed_states(
            self,
            states: List[Dict[str, float]],
        ) -> None:
            return None

    class ExternalAsyncDataset(IAsyncDataset):
        def __len__(self) -> int:
            return 1

        def async_reset(self) -> None:
            return None

        def async_submit(self, cursor: int, index: Any) -> bool:
            return True

        def async_fetch(self, cursor: int, index: Any) -> Optional[Any]:
            return None

        def async_finalize(self) -> None:
            return None

        def async_recover(self) -> None:
            return None

    class LegacyStreamMetric(IStreamMetric):
        @property
        def is_positive(self) -> bool:
            return False

        def reset(self) -> None:
            return None

        def update(
            self,
            tensor_batch: tensor_dict_type,
            tensor_outputs: tensor_dict_type,
            loader: Optional[DataLoader] = None,
        ) -> None:
            return None

        def finalize(self) -> float:
            return 0.0

    class OptionalLoss(ILoss):
        def forward(
            self,
            forward_results: tensor_dict_type,
            batch: tensor_dict_type,
            state: Optional[Any] = None,
        ) -> Optional[torch.Tensor]:
            return None

    class OptimizerExtensions:
        def get_backward_loss(
            self,
            state: Any,
            loss_res: TrainStepLoss,
            update: bool,
        ) -> Optional[torch.Tensor]:
            return loss_res.loss

        def will_skip_backward(self, state: Any, update: bool) -> bool:
            return False

        def step(self, pack: ClosurePack) -> None:
            return None

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
    metric_values_metric = ExternalMetricValues()
    stream_metric = ExternalStreamMetric()
    mergeable_stream_metric = ExternalMergeableStreamMetric()
    legacy_stream_metric = LegacyStreamMetric()
    optional_loss = OptionalLoss()
    optimizer_extensions = OptimizerExtensions()
    backward_loss_fn: GetBackwardLossFn = optimizer_extensions.get_backward_loss
    skip_backward_fn: WillSkipBackwardFn = optimizer_extensions.will_skip_backward
    closure_step_fn: ClosurePackStepFn = optimizer_extensions.step
    model_config = Config()
    data_config = DataConfig()
    model_config.module_name = "typing.model"
    assert_type(data_block.configs, Dict[str, Any])
    assert_type(model_config.callback_names, Optional[List[str]])
    assert_type(model_config.dispatch_batches, Optional[bool])
    assert_type(model_config.runtime, RuntimeSettings)
    assert_type(model_config.distributed, DistributedSettings)
    assert_type(model_config.build, DLSettings)
    assert_type(model_config.optimization, OptimizationSettings)
    assert_type(model_config.evaluation, EvaluationSettings)
    assert_type(model_config.logging, LoggingSettings)
    assert_type(model_config.persistence, PersistenceSettings)
    assert_type(data_config.presend_device, Optional[str])
    assert_type(data_config.async_prefetch, bool)
    assert_type(data_config.async_prefetch_factor, int)
    assert_type(data_config.async_prefetch_factor_for_validation, Optional[int])
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
    assert_type(metric_values_metric.evaluate({}, {}), Optional[MetricsOutputs])
    assert_type(stream_metric.evaluate({}, {}), Optional[MetricsOutputs])
    assert_type(stream_metric.report(stream_metric.finalize()), MetricsOutputs)
    assert_type(stream_metric.get_distributed_state(), Optional[Dict[str, float]])
    assert_type(
        mergeable_stream_metric.get_distributed_state(),
        Dict[str, float],
    )
    assert_type(mergeable_stream_metric.merge_distributed_states([{}]), None)
    stream_mse = StreamMSE()
    assert_type(stream_mse.get_distributed_state(), Tuple[float, int])
    assert_type(legacy_stream_metric.get_distributed_state(), Optional[Any])
    assert_type(ExternalModel, Type[ExternalModel])
    assert_type(IModel.make("typing.model", {}), IModel)
    assert_type(ExternalModel.make("typing.model", {}), IModel)
    assert_type(IModel.from_config(model_config), IModel)
    assert_type(normalize_loss_result(None), Optional[TrainStepLoss])
    assert_type(optional_loss.forward({}, {}), Optional[torch.Tensor])
    assert_type(ExternalCallback, Type[ExternalCallback])
    assert_type(TrainerCallback.make("typing.callback", {}), TrainerCallback)
    assert_type(
        ExternalCallback.make("typing.callback", {}),
        TrainerCallback,
    )

    metric_values = MetricValues(
        {"score": 1.0},
        {"score": True},
    )
    metric_outputs = MetricsOutputs(
        1.0,
        {"typing.metric_values": 1.0},
        {"typing.metric_values": True},
    )
    metric_result = MetricResult(metric_outputs, 2.0)
    metric_accumulator = MetricAccumulator()
    metric_accumulator.add(metric_result)
    inference_outputs = InferenceOutputs(
        {},
        {},
        metric_outputs,
        None,
    )
    assert_type(metric_values, MetricValues)
    assert_type(metric_values[0], Dict[str, float])
    assert_type(metric_values[1], Dict[str, bool])
    assert_type(metric_outputs, MetricsOutputs)
    assert_type(metric_outputs[0], float)
    assert_type(metric_outputs[1], Dict[str, float])
    assert_type(metric_outputs[2], Dict[str, bool])
    assert_type(metric_result, MetricResult)
    assert_type(metric_result.value, MetricsOutputs)
    assert_type(metric_result.sample_count, float)
    assert_type(metric_accumulator.merge(MetricAccumulator()), None)
    assert_type(metric_accumulator.finalize(), Optional[MetricsOutputs])
    assert_type(inference_outputs, InferenceOutputs)
    assert_type(
        inference_outputs.forward_results,
        Union[tensor_dict_type, Dict[str, List[torch.Tensor]]],
    )
    assert_type(
        inference_outputs.labels,
        Union[tensor_dict_type, Dict[str, List[torch.Tensor]]],
    )
    assert_type(inference_outputs.metric_outputs, Optional[MetricsOutputs])
    assert_type(inference_outputs.loss_items, Optional[Dict[str, float]])

    def check_inference_outputs(
        inference: Inference,
        loader: DataLoader,
    ) -> None:
        assert_type(inference.get_outputs(loader), InferenceOutputs)

    def check_async_loader(
        accelerator: Accelerator,
        loader: DataLoader,
    ) -> None:
        assert_type(loader.presend_device, Optional[str])
        assert_type(loader.async_prefetch, bool)
        assert_type(loader.async_prefetch_factor, int)
        assert_type(
            prepare_dataloaders(accelerator, loader),
            List[Optional[DataLoader]],
        )
