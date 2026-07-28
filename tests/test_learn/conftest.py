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
from typing import Type
from typing import Tuple
from pathlib import Path
from core.toolkit.registry import Registry

MappingState = Tuple[Any, str, Dict[str, Any], Dict[str, Any]]
RegistryState = Tuple[Any, str, Registry[Any], Dict[str, Type[Any]], Dict[str, str]]
OwnerState = Tuple[Any, str, Dict[str, Any]]


def _mapping_state(module: Any, name: str) -> MappingState:
    mapping = getattr(module, name)
    return module, name, mapping, dict(mapping)


def _registry_state(module: Any, name: str) -> RegistryState:
    registry: Registry[Any] = getattr(module, name)
    return module, name, registry, dict(registry.storage), registry.aliases


# Importing `core.learn` above finishes all built-in registrations before this
# baseline is captured. Test modules are collected only after this conftest.
_LEARN_MAPPINGS: List[MappingState] = [
    _mapping_state(learn_schema, "data_dict"),
    _mapping_state(learn_schema, "data_configs"),
    _mapping_state(learn_schema, "monitors"),
    _mapping_state(learn_schema, "metrics"),
    _mapping_state(learn_schema, "models"),
    _mapping_state(learn_schema, "trainer_callbacks"),
    _mapping_state(learn_schema, "configs"),
    _mapping_state(toolkit_pipeline, "pipelines"),
    _mapping_state(toolkit_pipeline, "pipeline_blocks"),
]
_LEARN_REGISTRIES: List[RegistryState] = [
    _registry_state(learn_modules, "module_registry"),
    _registry_state(learn_optimizers, "optimizer_registry"),
    _registry_state(learn_schedulers, "scheduler_registry"),
    _registry_state(learn_schedulers, "scheduler_op_registry"),
]
_LEARN_EXPORTS = [
    (name, original)
    for _, name, original, _ in _LEARN_MAPPINGS
    if getattr(cflearn, name, None) is original
] + [
    (name, original)
    for _, name, original, _, _ in _LEARN_REGISTRIES
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


def _clear_async_iterators() -> None:
    # Different loaders may share one dataset, so close every worker pool but
    # finalize each dataset at most once.
    iterators = {
        id(iterator): iterator
        for iterator in learn_schema.AsyncIterManager._cur.values()
    }
    learn_schema.AsyncIterManager._cur.clear()
    active_datasets = {}
    finalized_dataset_ids = set()
    for iterator in iterators.values():
        if not iterator._initialized:
            continue
        dataset_id = id(iterator._dataset)
        if iterator._finalized:
            finalized_dataset_ids.add(dataset_id)
            continue
        iterator._pool.shutdown(wait=True)
        iterator._results.clear()
        iterator._finalized = True
        active_datasets[dataset_id] = iterator._dataset
    for dataset_id, dataset in active_datasets.items():
        if dataset_id not in finalized_dataset_ids:
            dataset.async_finalize()


def _restore_learn_state() -> None:
    for module, name, original, baseline in _LEARN_MAPPINGS:
        setattr(module, name, original)
        original.clear()
        original.update(baseline)
    for module, name, original, storage, aliases in _LEARN_REGISTRIES:
        setattr(module, name, original)
        original.reset()
        original.storage.update(storage)
        for alias, target in aliases.items():
            original.register_alias(alias, target)
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
    _clear_async_iterators()
    _restore_learn_state()
    try:
        yield
    finally:
        _clear_async_iterators()
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
