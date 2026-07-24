import sys
import time

from typing import Any
from typing import Dict


def cold_import(params: Dict[str, Any], seed: int) -> Dict[str, Any]:
    import importlib

    module_name = params.get("module")
    iterations = params.get("iterations", 1)
    assert isinstance(seed, int)
    assert module_name in {"core", "core.learn"}
    assert type(iterations) is int
    assert iterations == 1
    assert module_name not in sys.modules
    if module_name == "core.learn":
        assert "core" not in sys.modules

    modules_before = set(sys.modules)
    import_module = importlib.import_module
    perf_counter_ns = time.perf_counter_ns
    start = perf_counter_ns()
    module = import_module(module_name)
    elapsed_ns = perf_counter_ns() - start

    module_file = getattr(module, "__file__", None)
    expected_suffix = f"{module_name.replace('.', '/')}/__init__.py"
    assert isinstance(module_file, str)
    assert module_file.replace("\\", "/").endswith(expected_suffix)
    assert module_name in sys.modules
    module_delta = len(set(sys.modules).difference(modules_before))
    assert module_delta >= 1

    behavior = {
        "module": module_name,
        "file_suffix": expected_suffix,
    }
    return {
        "elapsed_ns": elapsed_ns,
        "iterations": iterations,
        "behavior": behavior,
        "observations": {
            "imported_module_count": module_delta,
        },
        "uses_torch": module_name == "core.learn",
    }
