import os
import sys
import copy
import torch
import pytest
import random
import hashlib

import numpy as np
import core.parameters as parameters
import core.toolkit.pipeline as toolkit_pipeline

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Callable
from typing import Optional
from pydantic import BaseModel

_ORIGINAL_ENVIRON = os.environ
_OPT = parameters.OPT
_PIPELINE_REGISTRIES = [
    (
        toolkit_pipeline,
        "pipelines",
        toolkit_pipeline.pipelines,
        dict(toolkit_pipeline.pipelines),
    ),
    (
        toolkit_pipeline,
        "pipeline_blocks",
        toolkit_pipeline.pipeline_blocks,
        dict(toolkit_pipeline.pipeline_blocks),
    ),
]
_PIPELINE_OWNERS = [
    (toolkit_pipeline.IPipeline, "d", toolkit_pipeline.pipelines),
    (toolkit_pipeline.IBlock, "d", toolkit_pipeline.pipeline_blocks),
]


def _restore_dict(
    module: Any,
    name: str,
    original: Dict[str, Any],
    baseline: Dict[str, Any],
) -> None:
    setattr(module, name, original)
    original.clear()
    original.update(baseline)


def _restore_environ(original: Any, baseline: Dict[str, str]) -> None:
    os.environ = original
    original.clear()
    original.update(baseline)


def _capture_model_state(model: BaseModel) -> Tuple[BaseModel, Dict[str, Any]]:
    fields = {}
    for name in model.__class__.model_fields:
        value = getattr(model, name)
        if isinstance(value, BaseModel):
            fields[name] = _capture_model_state(value)
        else:
            fields[name] = copy.deepcopy(value)
    return model, fields


_OPT_STATE = _capture_model_state(_OPT)


def _restore_model_state(
    state: Tuple[BaseModel, Dict[str, Any]],
) -> BaseModel:
    model, fields = state
    for name, value in fields.items():
        if (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], BaseModel)
        ):
            value = _restore_model_state(value)
        else:
            value = copy.deepcopy(value)
        setattr(model, name, value)
    return model


def _restore_opt() -> None:
    parameters.OPT = _restore_model_state(_OPT_STATE)


def _restore_pipeline_registries() -> None:
    for module, name, original, baseline in _PIPELINE_REGISTRIES:
        _restore_dict(module, name, original, baseline)
    for owner, name, original in _PIPELINE_OWNERS:
        setattr(owner, name, original)


def _get_open_figures() -> Dict[int, Any]:
    pyplot = sys.modules.get("matplotlib.pyplot")
    if pyplot is None:
        return {}
    return {
        id(manager.canvas.figure): manager.canvas.figure
        for manager in pyplot._pylab_helpers.Gcf.get_all_fig_managers()
    }


def _close_new_figures(baseline: Dict[int, Any]) -> List[str]:
    pyplot = sys.modules.get("matplotlib.pyplot")
    if pyplot is None:
        return []
    errors = []
    for figure_id, figure in _get_open_figures().items():
        if figure_id in baseline:
            continue
        try:
            pyplot.close(figure)
        except Exception as err:
            errors.append(repr(err))
    return errors


class OwnedResources:
    """Track resources created by a test and clean up only those resources."""

    def __init__(self) -> None:
        self._cleanups: List[Tuple[str, Callable[[], None]]] = []

    def add_cleanup(self, cleanup: Callable[[], None], *, name: str) -> None:
        self._cleanups.append((name, cleanup))

    def track(
        self,
        resource: Any,
        *,
        close: Optional[Callable[[], None]] = None,
        unlink: bool = False,
    ) -> Any:
        if close is None:
            close = getattr(resource, "close", None)
            if close is None:
                mmap = getattr(resource, "_mmap", None)
                close = getattr(mmap, "close", None)
        if unlink:
            unlink_resource = getattr(resource, "unlink", None)
            if unlink_resource is None:
                raise TypeError("tracked resource does not provide `unlink`")
            self.add_cleanup(
                unlink_resource,
                name=f"{type(resource).__name__}.unlink",
            )
        if close is not None:
            self.add_cleanup(close, name=f"{type(resource).__name__}.close")
        if close is None and not unlink:
            raise TypeError("tracked resource does not provide a cleanup operation")
        return resource

    def cleanup(self) -> None:
        errors = []
        while self._cleanups:
            name, cleanup = self._cleanups.pop()
            try:
                cleanup()
            except Exception as err:
                errors.append(f"{name}: {err!r}")
        if errors:
            raise AssertionError(
                "failed to clean up test-owned resources: " + "; ".join(errors)
            )


@pytest.fixture(scope="session", autouse=True)
def _restore_session_environment() -> Any:
    baseline = dict(_ORIGINAL_ENVIRON)
    yield
    _restore_environ(_ORIGINAL_ENVIRON, baseline)


@pytest.fixture
def deterministic_seed(request: pytest.FixtureRequest) -> int:
    digest = hashlib.sha256(request.node.nodeid.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big")


@pytest.fixture
def owned_resources() -> Any:
    resources = OwnedResources()
    yield resources
    resources.cleanup()


@pytest.fixture(autouse=True)
def _isolate_process_state(deterministic_seed: int) -> Any:
    environ_baseline = dict(_ORIGINAL_ENVIRON)
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.random.get_rng_state().clone()
    cuda_rng_states = None
    if torch.cuda.is_available():
        cuda_rng_states = [state.clone() for state in torch.cuda.get_rng_state_all()]
    figures = _get_open_figures()

    _restore_environ(_ORIGINAL_ENVIRON, environ_baseline)
    _restore_opt()
    _restore_pipeline_registries()
    random.seed(deterministic_seed)
    np.random.seed(deterministic_seed)
    torch.manual_seed(deterministic_seed)
    if cuda_rng_states is not None:
        torch.cuda.manual_seed_all(deterministic_seed)

    yield

    cleanup_errors = _close_new_figures(figures)
    random.setstate(python_rng_state)
    np.random.set_state(numpy_rng_state)
    torch.random.set_rng_state(torch_rng_state)
    if cuda_rng_states is not None:
        torch.cuda.set_rng_state_all(cuda_rng_states)
    _restore_pipeline_registries()
    _restore_opt()
    _restore_environ(_ORIGINAL_ENVIRON, environ_baseline)
    if cleanup_errors:
        pytest.fail(
            "failed to close test-owned matplotlib figures: "
            + "; ".join(cleanup_errors)
        )
