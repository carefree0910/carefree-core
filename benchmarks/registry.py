from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import NamedTuple


class CaseSpec(NamedTuple):
    id: str
    entrypoint: str
    description: str
    target: str
    tags: Tuple[str, ...]
    profiles: Dict[str, Dict[str, Any]]


PROFILE_REPETITIONS = {
    "smoke": 7,
    "baseline": 15,
}

CASES = [
    CaseSpec(
        id="import.core",
        entrypoint="benchmarks.scenarios.imports:cold_import",
        description="Cold import of the top-level core package",
        target="core/__init__.py",
        tags=("cpu", "cold-import"),
        profiles={
            "smoke": {"module": "core", "iterations": 1},
            "baseline": {"module": "core", "iterations": 1},
        },
    ),
    CaseSpec(
        id="import.core_learn",
        entrypoint="benchmarks.scenarios.imports:cold_import",
        description="Cold import of the core.learn public surface",
        target="core/learn/__init__.py",
        tags=("cpu", "cold-import"),
        profiles={
            "smoke": {"module": "core.learn", "iterations": 1},
            "baseline": {"module": "core.learn", "iterations": 1},
        },
    ),
    CaseSpec(
        id="data.async_loader",
        entrypoint="benchmarks.scenarios.learn:async_dataloader",
        description="Drain an asynchronous ArrayData loader",
        target="core/learn/schema.py:AsyncDataLoaderIter",
        tags=("cpu", "torch", "data"),
        profiles={
            "smoke": {
                "num_samples": 64,
                "input_dim": 8,
                "output_dim": 2,
                "batch_size": 16,
                "prefetch_factor": 2,
                "iterations": 1,
            },
            "baseline": {
                "num_samples": 512,
                "input_dim": 32,
                "output_dim": 8,
                "batch_size": 32,
                "prefetch_factor": 4,
                "iterations": 1,
            },
        },
    ),
    CaseSpec(
        id="inference.padding",
        entrypoint="benchmarks.scenarios.learn:inference_padding",
        description="Accumulate and pad variable-width inference outputs",
        target="core/learn/inference.py:Inference.get_outputs",
        tags=("cpu", "torch", "inference"),
        profiles={
            "smoke": {
                "num_samples": 4,
                "rows": 3,
                "max_width": 3,
                "iterations": 1,
            },
            "baseline": {
                "num_samples": 32,
                "rows": 8,
                "max_width": 8,
                "iterations": 3,
            },
        },
    ),
    CaseSpec(
        id="trainer.train_step",
        entrypoint="benchmarks.scenarios.learn:trainer_train_step",
        description="Run optimizer-backed Trainer.train_step calls",
        target="core/learn/trainer.py:Trainer.train_step",
        tags=("cpu", "torch", "training"),
        profiles={
            "smoke": {
                "batch_size": 16,
                "input_dim": 16,
                "hidden_dim": 32,
                "output_dim": 4,
                "depth": 1,
                "iterations": 2,
            },
            "baseline": {
                "batch_size": 64,
                "input_dim": 64,
                "hidden_dim": 128,
                "output_dim": 16,
                "depth": 2,
                "iterations": 16,
            },
        },
    ),
    CaseSpec(
        id="moe.dispatch_combine",
        entrypoint="benchmarks.scenarios.learn:moe_dispatch_combine",
        description="Dispatch, execute, and combine deterministic MoE routes",
        target="core/learn/modules/moe.py:MoEDispatcher",
        tags=("cpu", "torch", "moe"),
        profiles={
            "smoke": {
                "batch_size": 32,
                "dim": 16,
                "num_experts": 4,
                "top_k": 2,
                "output_dim": 8,
                "iterations": 4,
            },
            "baseline": {
                "batch_size": 512,
                "dim": 64,
                "num_experts": 16,
                "top_k": 4,
                "output_dim": 32,
                "iterations": 8,
            },
        },
    ),
    CaseSpec(
        id="ema.update",
        entrypoint="benchmarks.scenarios.learn:ema_update",
        description="Update and swap exponential moving-average parameters",
        target="core/learn/modules/common.py:EMA",
        tags=("cpu", "torch", "ema"),
        profiles={
            "smoke": {
                "num_layers": 2,
                "dim": 64,
                "iterations": 4,
            },
            "baseline": {
                "num_layers": 4,
                "dim": 256,
                "iterations": 16,
            },
        },
    ),
    CaseSpec(
        id="toolkit.pipeline_build",
        entrypoint="benchmarks.scenarios.toolkit:pipeline_build",
        description="Build an in-memory toolkit block pipeline",
        target="core/toolkit/pipeline.py:IPipeline.build",
        tags=("cpu", "toolkit", "pipeline"),
        profiles={
            "smoke": {"num_blocks": 16, "iterations": 1},
            "baseline": {"num_blocks": 512, "iterations": 1},
        },
    ),
    CaseSpec(
        id="toolkit.filter_kw",
        entrypoint="benchmarks.scenarios.toolkit:filter_kw_case",
        description="Filter a fixed keyword mapping against a signature",
        target="core/toolkit/misc.py:filter_kw",
        tags=("cpu", "toolkit", "micro"),
        profiles={
            "smoke": {
                "num_kwargs": 8,
                "num_accepted": 3,
                "iterations": 200,
                "strict": True,
            },
            "baseline": {
                "num_kwargs": 32,
                "num_accepted": 3,
                "iterations": 2000,
                "strict": True,
            },
        },
    ),
    CaseSpec(
        id="learn.summary",
        entrypoint="benchmarks.scenarios.toolkit:summary_case",
        description="Render a deterministic torch module summary",
        target="core/learn/toolkit.py:summary",
        tags=("cpu", "torch", "summary"),
        profiles={
            "smoke": {
                "input_dim": 16,
                "hidden_dim": 32,
                "output_dim": 8,
                "depth": 1,
                "batch_size": 4,
                "iterations": 2,
            },
            "baseline": {
                "input_dim": 128,
                "hidden_dim": 256,
                "output_dim": 64,
                "depth": 3,
                "batch_size": 32,
                "iterations": 8,
            },
        },
    ),
]


def get_cases() -> List[CaseSpec]:
    return list(CASES)


__all__ = [
    "CASES",
    "PROFILE_REPETITIONS",
    "CaseSpec",
    "get_cases",
]
