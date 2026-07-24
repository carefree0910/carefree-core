import os
import sys
import torch
import random
import subprocess

import numpy as np
import core.parameters as parameters
import core.toolkit.pipeline as toolkit_pipeline

from pathlib import Path

_ENVIRON = os.environ
_ENVIRON_SENTINEL = os.environ.get("CFCORE_FIXTURE_SENTINEL")
_OPT = parameters.OPT
_OPT_BASELINE = _OPT.model_dump()
_FLOW_OPT = _OPT.flow_opt
_LEARN_OPT = _OPT.learn_opt
_PIPELINES = toolkit_pipeline.pipelines
_PIPELINE_BLOCKS = toolkit_pipeline.pipeline_blocks
_CREATED_FIGURE = None


def test_toolkit_fixtures_do_not_import_core_learn() -> None:
    conftest_path = Path(__file__).parents[1] / "conftest.py"
    code = "\n".join(
        [
            "import importlib.util",
            "import sys",
            f"spec = importlib.util.spec_from_file_location('root_conftest', {str(conftest_path)!r})",
            "module = importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(module)",
            "assert 'core.learn' not in sys.modules",
        ]
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_deterministic_seed(
    deterministic_seed: int,
    owned_resources,
) -> None:
    class Resource:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    actual = (random.random(), np.random.random(), torch.rand(1))
    random.seed(deterministic_seed)
    np.random.seed(deterministic_seed)
    torch.manual_seed(deterministic_seed)
    expected = (random.random(), np.random.random(), torch.rand(1))
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2])

    owned = Resource()
    unowned = Resource()
    assert owned_resources.track(owned) is owned
    owned_resources.cleanup()
    assert owned.closed
    assert not unowned.closed


def test_process_state_mutation_is_isolated() -> None:
    import matplotlib.pyplot as plt

    global _CREATED_FIGURE
    _CREATED_FIGURE = plt.figure()

    parameters.OPT.learn_opt.cache_dir = Path("_fixture_cache")
    parameters.OPT.flow_opt = parameters.FlowOpt(focus="$replacement")
    parameters.OPT.learn_opt = parameters.LearnOpt()
    parameters.OPT = parameters.OPTClass()
    toolkit_pipeline.pipelines["$fixture"] = object()
    toolkit_pipeline.pipeline_blocks["$fixture"] = object()
    toolkit_pipeline.pipelines = {"$replacement": object()}
    toolkit_pipeline.pipeline_blocks = {"$replacement": object()}
    toolkit_pipeline.IPipeline.d = toolkit_pipeline.pipelines
    toolkit_pipeline.IBlock.d = toolkit_pipeline.pipeline_blocks
    os.environ["CFCORE_FIXTURE_SENTINEL"] = "mutated"
    os.environ = {"CFCORE_FIXTURE_SENTINEL": "replacement"}


def test_process_state_was_restored(deterministic_seed: int) -> None:
    import matplotlib.pyplot as plt

    assert os.environ is _ENVIRON
    assert os.environ.get("CFCORE_FIXTURE_SENTINEL") == _ENVIRON_SENTINEL
    assert parameters.OPT is _OPT
    assert parameters.OPT.flow_opt is _FLOW_OPT
    assert parameters.OPT.learn_opt is _LEARN_OPT
    assert parameters.OPT.model_dump() == _OPT_BASELINE
    assert toolkit_pipeline.pipelines is _PIPELINES
    assert toolkit_pipeline.pipeline_blocks is _PIPELINE_BLOCKS
    assert toolkit_pipeline.IPipeline.d is _PIPELINES
    assert toolkit_pipeline.IBlock.d is _PIPELINE_BLOCKS
    assert toolkit_pipeline.pipelines == {}
    assert toolkit_pipeline.pipeline_blocks == {}
    if _CREATED_FIGURE is not None:
        assert not plt.fignum_exists(_CREATED_FIGURE.number)

    actual = (random.random(), np.random.random(), torch.rand(1))
    random.seed(deterministic_seed)
    np.random.seed(deterministic_seed)
    torch.manual_seed(deterministic_seed)
    expected = (random.random(), np.random.random(), torch.rand(1))
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2])
