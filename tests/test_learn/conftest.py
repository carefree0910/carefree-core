import pytest

import core.learn as cflearn
import core.learn.schema as learn_schema
import core.learn.toolkit as learn_toolkit
import core.learn.optimizers as learn_optimizers
import core.learn.schedulers as learn_schedulers
import core.toolkit.pipeline as toolkit_pipeline
import core.learn.pipeline.api as pipeline_api
import core.learn.modules.common as learn_modules
import core.learn.pipeline.common as learn_pipeline
import core.learn.pipeline.blocks.basic as pipeline_blocks

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from pathlib import Path

RegistryState = Tuple[Any, str, Dict[str, Any], Dict[str, Any]]
OwnerState = Tuple[Any, str, Dict[str, Any]]


def _registry_state(module: Any, name: str) -> RegistryState:
    registry = getattr(module, name)
    return module, name, registry, dict(registry)


# Importing `core.learn` above finishes all built-in registrations before this
# baseline is captured. Test modules are collected only after this conftest.
_LEARN_REGISTRIES: List[RegistryState] = [
    _registry_state(learn_schema, "data_dict"),
    _registry_state(learn_schema, "data_configs"),
    _registry_state(learn_schema, "monitors"),
    _registry_state(learn_schema, "metrics"),
    _registry_state(learn_schema, "models"),
    _registry_state(learn_schema, "trainer_callbacks"),
    _registry_state(learn_schema, "configs"),
    _registry_state(learn_modules, "module_dict"),
    _registry_state(learn_optimizers, "optimizer_dict"),
    _registry_state(learn_schedulers, "scheduler_ops"),
    _registry_state(learn_schedulers, "scheduler_dict"),
    _registry_state(toolkit_pipeline, "pipelines"),
    _registry_state(toolkit_pipeline, "pipeline_blocks"),
]
_LEARN_EXPORTS = [
    (name, original)
    for _, name, original, _ in _LEARN_REGISTRIES
    if getattr(cflearn, name, None) is original
]
_LEARN_OWNERS: List[OwnerState] = [
    (learn_schema.IData, "d", learn_schema.data_dict),
    (learn_schema.DataConfig, "d", learn_schema.data_configs),
    (learn_schema.TrainerMonitor, "d", learn_schema.monitors),
    (learn_schema.IMetric, "d", learn_schema.metrics),
    (learn_schema.IModel, "d", learn_schema.models),
    (learn_schema.TrainerCallback, "d", learn_schema.trainer_callbacks),
    (learn_schema.Config, "d", learn_schema.configs),
    (toolkit_pipeline.IPipeline, "d", toolkit_pipeline.pipelines),
    (toolkit_pipeline.IBlock, "d", toolkit_pipeline.pipeline_blocks),
    (learn_schema.IDataBlock, "d", toolkit_pipeline.pipeline_blocks),
    (learn_pipeline.Block, "d", toolkit_pipeline.pipeline_blocks),
    (learn_pipeline.Pipeline, "d", toolkit_pipeline.pipelines),
]
_INITIALIZER_DEFINED = learn_toolkit.Initializer.defined_initialization
_INITIALIZER_DEFINED_BASELINE = set(_INITIALIZER_DEFINED)
_CUSTOM_INITIALIZERS = learn_toolkit.Initializer.custom_initializer
_CUSTOM_INITIALIZERS_BASELINE = dict(_CUSTOM_INITIALIZERS)


def _restore_learn_state() -> None:
    for module, name, original, baseline in _LEARN_REGISTRIES:
        setattr(module, name, original)
        original.clear()
        original.update(baseline)
    for name, original in _LEARN_EXPORTS:
        setattr(cflearn, name, original)
    for owner, name, original in _LEARN_OWNERS:
        setattr(owner, name, original)
    learn_toolkit.Initializer.defined_initialization = _INITIALIZER_DEFINED
    _INITIALIZER_DEFINED.clear()
    _INITIALIZER_DEFINED.update(_INITIALIZER_DEFINED_BASELINE)
    learn_toolkit.Initializer.custom_initializer = _CUSTOM_INITIALIZERS
    _CUSTOM_INITIALIZERS.clear()
    _CUSTOM_INITIALIZERS.update(_CUSTOM_INITIALIZERS_BASELINE)


@pytest.fixture(autouse=True)
def _isolate_learn_state() -> Any:
    _restore_learn_state()
    yield
    _restore_learn_state()


@pytest.fixture(autouse=True)
def _redirect_relative_workspaces(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original = pipeline_api.prepare_workspace_from

    def prepare_workspace_from(workspace: Any, **kwargs: Any) -> Any:
        path = Path(workspace)
        if not path.is_absolute():
            workspace = tmp_path / path
        return original(workspace, **kwargs)

    monkeypatch.setattr(pipeline_api, "prepare_workspace_from", prepare_workspace_from)
    monkeypatch.setattr(
        pipeline_blocks,
        "prepare_workspace_from",
        prepare_workspace_from,
    )
