import os
import sys
import json
import random
import argparse
import platform
import importlib

from typing import Any
from typing import Dict
from pathlib import Path

from ._protocol import load_json
from ._protocol import canonical_sha256
from ._protocol import atomic_write_json


def _peak_rss() -> Dict[str, Any]:
    status_path = Path("/proc/self/status")
    if status_path.is_file():
        with status_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return {
                            "bytes": int(parts[1]) * 1024,
                            "method": "linux_proc_status_vmhwm",
                        }
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            peak_bytes = int(peak)
        else:
            peak_bytes = int(peak) * 1024
        return {
            "bytes": peak_bytes,
            "method": "resource_ru_maxrss",
        }
    except (ImportError, OSError):
        return {
            "bytes": None,
            "method": "unsupported",
        }


def _load_callable(entrypoint: str) -> Any:
    module_name, separator, attribute = entrypoint.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"invalid benchmark entrypoint: '{entrypoint}'")
    module = importlib.import_module(module_name)
    fn = getattr(module, attribute)
    if not callable(fn):
        raise TypeError(f"benchmark entrypoint is not callable: '{entrypoint}'")
    return fn


def _validate_scenario_result(result: Any) -> Dict[str, Any]:
    if not isinstance(result, dict):
        raise TypeError("a benchmark scenario should return a dictionary")
    elapsed_ns = result.get("elapsed_ns")
    iterations = result.get("iterations")
    behavior = result.get("behavior")
    observations = result.get("observations", {})
    uses_torch = result.get("uses_torch")
    if not isinstance(elapsed_ns, int) or elapsed_ns <= 0:
        raise ValueError("scenario 'elapsed_ns' should be a positive integer")
    if not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("scenario 'iterations' should be a positive integer")
    if not isinstance(behavior, dict):
        raise TypeError("scenario 'behavior' should be a dictionary")
    if not isinstance(observations, dict):
        raise TypeError("scenario 'observations' should be a dictionary")
    if not isinstance(uses_torch, bool):
        raise TypeError("scenario 'uses_torch' should be a boolean")
    canonical_sha256(behavior)
    canonical_sha256(observations)
    result["observations"] = observations
    return result


def run(request_path: Path, output_path: Path) -> None:
    request = load_json(request_path)
    entrypoint = request["entrypoint"]
    params = request["parameters"]
    seed = request["seed"]
    mode = request["mode"]
    if not isinstance(entrypoint, str):
        raise TypeError("'entrypoint' should be a string")
    if not isinstance(params, dict):
        raise TypeError("'parameters' should be a dictionary")
    if not isinstance(seed, int):
        raise TypeError("'seed' should be an integer")
    if mode not in {"validate", "measure"}:
        raise ValueError("'mode' should be either 'validate' or 'measure'")
    random.seed(seed)
    scenario = _load_callable(entrypoint)
    raw_scenario_result = scenario(params, seed)
    rss = _peak_rss()
    scenario_result = _validate_scenario_result(raw_scenario_result)
    uses_torch = scenario_result["uses_torch"]
    if uses_torch:
        torch_memory = {
            "device": "cpu",
            "cpu_peak_bytes": rss["bytes"],
            "cpu_measurement": "process_peak_rss",
            "cuda_peak_allocated_bytes": None,
            "cuda_peak_reserved_bytes": None,
            "synchronization": "not_required_for_cpu",
        }
    else:
        torch_memory = None
    elapsed_ns = scenario_result["elapsed_ns"]
    iterations = scenario_result["iterations"]
    result = {
        "mode": mode,
        "pid": os.getpid(),
        "elapsed_ns": elapsed_ns,
        "iterations": iterations,
        "ns_per_iteration": elapsed_ns / iterations,
        "peak_rss_bytes": rss["bytes"],
        "rss_method": rss["method"],
        "torch_memory": torch_memory,
        "observations": scenario_result["observations"],
        "behavior": scenario_result["behavior"],
        "behavior_sha256": canonical_sha256(scenario_result["behavior"]),
    }
    atomic_write_json(output_path, result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Internal carefree-core benchmark worker"
    )
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.request, args.output)


if __name__ == "__main__":
    main()
