import time
import hashlib

from typing import Any
from typing import Dict
from typing import List
from typing import Type


def _positive_int(params: Dict[str, Any], key: str, default: int) -> int:
    value = params.get(key, default)
    assert type(value) is int
    assert value > 0
    return value


def pipeline_build(params: Dict[str, Any], seed: int) -> Dict[str, Any]:
    from core.toolkit.pipeline import IBlock
    from core.toolkit.pipeline import IPipeline

    assert isinstance(seed, int)
    num_blocks = _positive_int(params, "num_blocks", 8)
    iterations = _positive_int(params, "iterations", 1)
    expected_order = [f"block_{i:04d}" for i in range(num_blocks)]

    class Requirement:
        __identifier__: str

        def __init__(self, identifier: str) -> None:
            self.__identifier__ = identifier

    class BenchmarkBlock(IBlock):
        def __init__(self, identifier: str, requirement: Any = None) -> None:
            self.__identifier__ = identifier
            self._requirements = [] if requirement is None else [requirement]

        @property
        def requirements(self) -> List[Any]:
            return self._requirements

        def build(self, config: Any) -> None:
            config["order"].append(self.__identifier__)

    class BenchmarkPipeline(IPipeline):
        @classmethod
        def init(cls, config: Any) -> "BenchmarkPipeline":
            instance = cls()
            instance.config = config
            return instance

        @property
        def config_base(self) -> Type:
            return dict

        @property
        def block_base(self) -> Type:
            return BenchmarkBlock

    prepared = []
    for _ in range(iterations):
        pipeline = BenchmarkPipeline.init({"order": []})
        blocks = []
        for i, identifier in enumerate(expected_order):
            requirement = None if i == 0 else Requirement(expected_order[i - 1])
            blocks.append(BenchmarkBlock(identifier, requirement))
        prepared.append((pipeline, blocks))

    perf_counter_ns = time.perf_counter_ns
    start = perf_counter_ns()
    for pipeline, blocks in prepared:
        pipeline.build(*blocks)
    elapsed_ns = perf_counter_ns() - start

    expected_previous_count = num_blocks * (num_blocks - 1) // 2
    previous_count = 0
    for pipeline, blocks in prepared:
        assert pipeline.config["order"] == expected_order
        assert [block.__identifier__ for block in pipeline.blocks] == expected_order
        for i, block in enumerate(blocks):
            assert list(block.previous) == expected_order[:i]
            previous_count += len(block.previous)
    assert previous_count == iterations * expected_previous_count

    behavior = {
        "num_blocks": num_blocks,
        "build_count": iterations * num_blocks,
        "dependency_edges": iterations * max(0, num_blocks - 1),
        "previous_entries": previous_count,
        "first_block": expected_order[0],
        "last_block": expected_order[-1],
    }
    return {
        "elapsed_ns": elapsed_ns,
        "iterations": iterations,
        "behavior": behavior,
        "uses_torch": False,
    }


def filter_kw_case(params: Dict[str, Any], seed: int) -> Dict[str, Any]:
    import inspect

    from core.toolkit.misc import filter_kw

    assert isinstance(seed, int)
    num_kwargs = _positive_int(params, "num_kwargs", 8)
    num_accepted = params.get("num_accepted", num_kwargs // 2)
    iterations = _positive_int(params, "iterations", 1_000)
    strict = params.get("strict", False)
    accepts_var_kwargs = params.get("accepts_var_kwargs", False)
    assert type(num_accepted) is int
    assert 0 <= num_accepted <= num_kwargs
    assert type(strict) is bool
    assert type(accepts_var_kwargs) is bool

    keys = [f"key_{i:04d}" for i in range(num_kwargs)]
    accepted_keys = keys[:num_accepted]
    kwargs = {key: i for i, key in enumerate(keys)}

    def target(**target_kwargs: Any) -> None:
        del target_kwargs

    signature_parameters = [
        inspect.Parameter(key, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for key in accepted_keys
    ]
    if accepts_var_kwargs:
        signature_parameters.append(
            inspect.Parameter("extra_kwargs", inspect.Parameter.VAR_KEYWORD)
        )
    target.__signature__ = inspect.Signature(signature_parameters)  # type: ignore
    if accepts_var_kwargs and not strict:
        expected = kwargs
    else:
        expected = {key: kwargs[key] for key in accepted_keys}

    result: Dict[str, Any] = {}
    perf_counter_ns = time.perf_counter_ns
    start = perf_counter_ns()
    for _ in range(iterations):
        result = filter_kw(target, kwargs, strict=strict)
    elapsed_ns = perf_counter_ns() - start

    assert result == expected
    behavior = {
        "num_kwargs": num_kwargs,
        "num_accepted": len(result),
        "strict": strict,
        "accepts_var_kwargs": accepts_var_kwargs,
        "result_keys": list(result),
    }
    return {
        "elapsed_ns": elapsed_ns,
        "iterations": iterations,
        "behavior": behavior,
        "uses_torch": False,
    }


def summary_case(params: Dict[str, Any], seed: int) -> Dict[str, Any]:
    import torch

    import torch.nn as nn

    from core.learn.toolkit import summary
    from core.learn.constants import INPUT_KEY

    assert isinstance(seed, int)
    input_dim = _positive_int(params, "input_dim", 16)
    hidden_dim = _positive_int(params, "hidden_dim", 32)
    output_dim = _positive_int(params, "output_dim", 8)
    depth = _positive_int(params, "depth", 2)
    batch_size = _positive_int(params, "batch_size", 4)
    iterations = _positive_int(params, "iterations", 1)

    torch.manual_seed(seed)
    layers: List[nn.Module] = []
    previous_dim = input_dim
    for _ in range(depth):
        layers.append(nn.Linear(previous_dim, hidden_dim))
        layers.append(nn.ReLU())
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, output_dim))
    model = nn.Sequential(*layers)
    sample_batch = {INPUT_KEY: torch.randn(batch_size, input_dim)}
    summary_forward = lambda batch: model(batch[INPUT_KEY])

    expected_params = input_dim * hidden_dim + hidden_dim
    expected_params += (depth - 1) * (hidden_dim * hidden_dim + hidden_dim)
    expected_params += hidden_dim * output_dim + output_dim
    assert sum(parameter.numel() for parameter in model.parameters()) == expected_params

    result = ""
    perf_counter_ns = time.perf_counter_ns
    start = perf_counter_ns()
    for _ in range(iterations):
        result = summary(
            model,
            sample_batch,
            return_only=True,
            summary_forward=summary_forward,
        )
    elapsed_ns = perf_counter_ns() - start

    hook_count = sum(len(module._forward_hooks) for module in model.modules())
    assert hook_count == 0
    parameter_line = f"Total params: {expected_params:,}"
    assert parameter_line in result

    behavior = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "depth": depth,
        "batch_size": batch_size,
        "parameter_count": expected_params,
        "summary_lines": len(result.splitlines()),
        "summary_sha256": hashlib.sha256(result.encode("utf-8")).hexdigest(),
        "hooks_after": hook_count,
    }
    return {
        "elapsed_ns": elapsed_ns,
        "iterations": iterations,
        "behavior": behavior,
        "uses_torch": True,
    }
