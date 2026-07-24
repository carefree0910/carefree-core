import os
import sys
import time


def _peak_rss() -> tuple:
    status_path = "/proc/self/status"
    if os.path.isfile(status_path):
        with open(status_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024, "linux_proc_status_vmhwm"
    import platform
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = int(peak) if platform.system() == "Darwin" else int(peak) * 1024
    return peak_bytes, "resource_ru_maxrss"


def main() -> None:
    if len(sys.argv) != 4:
        raise RuntimeError("expected: <module> <mode> <output>")
    module_name = sys.argv[1]
    mode = sys.argv[2]
    output_path = sys.argv[3]
    if module_name not in {"core", "core.learn"}:
        raise ValueError(f"unsupported cold import target: '{module_name}'")
    if mode not in {"validate", "measure"}:
        raise ValueError(f"unsupported worker mode: '{mode}'")
    if module_name in sys.modules:
        raise RuntimeError(f"'{module_name}' was imported before cold-import timing")
    if module_name == "core.learn" and "core" in sys.modules:
        raise RuntimeError("'core' was imported before cold-import timing")

    modules_before = set(sys.modules)
    start = time.perf_counter_ns()
    __import__(module_name)
    elapsed_ns = time.perf_counter_ns() - start
    peak_rss_bytes, rss_method = _peak_rss()
    module = sys.modules[module_name]

    module_file = getattr(module, "__file__", None)
    expected_suffix = f"{module_name.replace('.', '/')}/__init__.py"
    if not isinstance(module_file, str):
        raise TypeError(f"'{module_name}' does not expose a source path")
    if not module_file.replace("\\", "/").endswith(expected_suffix):
        raise RuntimeError(
            f"'{module_name}' resolved to an unexpected source: '{module_file}'"
        )
    imported_module_count = len(set(sys.modules).difference(modules_before))
    if imported_module_count < 1:
        raise RuntimeError(f"'{module_name}' did not add an imported module")

    import json
    import hashlib

    behavior = {
        "module": module_name,
        "file_suffix": expected_suffix,
    }
    canonical_behavior = json.dumps(
        behavior,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    behavior_sha256 = hashlib.sha256(canonical_behavior.encode("utf-8")).hexdigest()
    uses_torch = module_name == "core.learn"
    if uses_torch:
        torch_memory = {
            "device": "cpu",
            "cpu_peak_bytes": peak_rss_bytes,
            "cpu_measurement": "process_peak_rss",
            "cuda_peak_allocated_bytes": None,
            "cuda_peak_reserved_bytes": None,
            "synchronization": "not_required_for_cpu",
        }
    else:
        torch_memory = None
    result = {
        "mode": mode,
        "pid": os.getpid(),
        "elapsed_ns": elapsed_ns,
        "iterations": 1,
        "ns_per_iteration": float(elapsed_ns),
        "peak_rss_bytes": peak_rss_bytes,
        "rss_method": rss_method,
        "torch_memory": torch_memory,
        "observations": {
            "imported_module_count": imported_module_count,
            "worker_module_count_before_import": len(modules_before),
        },
        "behavior": behavior,
        "behavior_sha256": behavior_sha256,
    }
    temporary_path = f"{output_path}.tmp.{os.getpid()}"
    try:
        with open(temporary_path, "w", encoding="utf-8") as f:
            json.dump(
                result,
                f,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


if __name__ == "__main__":
    main()
