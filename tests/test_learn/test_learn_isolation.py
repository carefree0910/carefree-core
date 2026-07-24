import core.learn as cflearn
import core.learn.schema as learn_schema
import core.learn.toolkit as learn_toolkit
import core.learn.optimizers as learn_optimizers
import core.learn.schedulers as learn_schedulers
import core.toolkit.pipeline as toolkit_pipeline
import core.learn.modules.common as learn_modules
import core.learn.pipeline.common as learn_pipeline

from typing import Any
from typing import Dict

_REGISTRIES = [
    (learn_schema, "data_dict", learn_schema.data_dict),
    (learn_schema, "data_configs", learn_schema.data_configs),
    (learn_schema, "monitors", learn_schema.monitors),
    (learn_schema, "metrics", learn_schema.metrics),
    (learn_schema, "models", learn_schema.models),
    (learn_schema, "trainer_callbacks", learn_schema.trainer_callbacks),
    (learn_schema, "configs", learn_schema.configs),
    (learn_modules, "module_dict", learn_modules.module_dict),
    (learn_optimizers, "optimizer_dict", learn_optimizers.optimizer_dict),
    (learn_schedulers, "scheduler_ops", learn_schedulers.scheduler_ops),
    (learn_schedulers, "scheduler_dict", learn_schedulers.scheduler_dict),
    (toolkit_pipeline, "pipelines", toolkit_pipeline.pipelines),
    (toolkit_pipeline, "pipeline_blocks", toolkit_pipeline.pipeline_blocks),
]
_REGISTRY_BASELINES = {
    (module.__name__, name): dict(original) for module, name, original in _REGISTRIES
}
_OWNERS = [
    (learn_schema.IData, learn_schema.data_dict),
    (learn_schema.DataConfig, learn_schema.data_configs),
    (learn_schema.TrainerMonitor, learn_schema.monitors),
    (learn_schema.IMetric, learn_schema.metrics),
    (learn_schema.IModel, learn_schema.models),
    (learn_schema.TrainerCallback, learn_schema.trainer_callbacks),
    (learn_schema.Config, learn_schema.configs),
    (toolkit_pipeline.IPipeline, toolkit_pipeline.pipelines),
    (toolkit_pipeline.IBlock, toolkit_pipeline.pipeline_blocks),
    (learn_schema.IDataBlock, toolkit_pipeline.pipeline_blocks),
    (learn_pipeline.Block, toolkit_pipeline.pipeline_blocks),
    (learn_pipeline.Pipeline, toolkit_pipeline.pipelines),
]
_DEFINED_INITIALIZATIONS = learn_toolkit.Initializer.defined_initialization
_DEFINED_INITIALIZATIONS_BASELINE = set(_DEFINED_INITIALIZATIONS)
_CUSTOM_INITIALIZERS = learn_toolkit.Initializer.custom_initializer
_CUSTOM_INITIALIZERS_BASELINE = dict(_CUSTOM_INITIALIZERS)


def test_learn_state_mutation_is_isolated() -> None:
    replacement: Dict[str, Any] = {"$replacement": object()}
    for module, name, original in _REGISTRIES:
        original["$fixture"] = object()
        setattr(module, name, replacement.copy())
    for owner, _ in _OWNERS:
        owner.d = replacement.copy()
    cflearn.optimizer_dict = replacement.copy()
    cflearn.scheduler_dict = replacement.copy()
    cflearn.module_dict = replacement.copy()

    _DEFINED_INITIALIZATIONS.add("$fixture")
    _CUSTOM_INITIALIZERS["$fixture"] = lambda *_: None
    learn_toolkit.Initializer.defined_initialization = {"$replacement"}
    learn_toolkit.Initializer.custom_initializer = {
        "$replacement": lambda *_: None,
    }


def test_learn_state_was_restored() -> None:
    for module, name, original in _REGISTRIES:
        assert getattr(module, name) is original
        assert original == _REGISTRY_BASELINES[(module.__name__, name)]
        assert "$fixture" not in original
        assert "$replacement" not in original
    for owner, original in _OWNERS:
        assert owner.d is original
    assert cflearn.optimizer_dict is learn_optimizers.optimizer_dict
    assert cflearn.scheduler_dict is learn_schedulers.scheduler_dict
    assert cflearn.module_dict is learn_modules.module_dict

    assert learn_toolkit.Initializer.defined_initialization is _DEFINED_INITIALIZATIONS
    assert _DEFINED_INITIALIZATIONS == _DEFINED_INITIALIZATIONS_BASELINE
    assert learn_toolkit.Initializer.custom_initializer is _CUSTOM_INITIALIZERS
    assert _CUSTOM_INITIALIZERS == _CUSTOM_INITIALIZERS_BASELINE
