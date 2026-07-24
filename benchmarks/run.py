import os
import sys
import json
import time
import signal
import hashlib
import argparse
import platform
import tempfile
import subprocess

import importlib.metadata as importlib_metadata

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from pathlib import Path
from datetime import datetime
from datetime import timezone

from . import SCHEMA_VERSION
from .registry import get_cases
from .registry import CaseSpec
from .registry import PROFILE_REPETITIONS
from ._protocol import load_json
from ._protocol import summarize
from ._protocol import file_sha256
from ._protocol import canonical_sha256
from ._protocol import atomic_write_json

DEFAULT_SEED = 20240724
STATISTICS_ALGORITHM = "median_raw_mad_r7_v1"
THREAD_SETTINGS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
PROFILE_TIMEOUT_SECONDS = {
    "smoke": 45,
    "baseline": 120,
}
PROFILE_TOTAL_TIMEOUT_SECONDS = {
    "smoke": 12 * 60,
    "baseline": 60 * 60,
}


class BenchmarkRunError(RuntimeError):
    pass


class BenchmarkTerminationRequested(BaseException):
    pass


def _raise_termination_requested(signum: int, frame: Any) -> None:
    del frame
    raise BenchmarkTerminationRequested(f"received signal {signum}")


def _repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _safe_case_id(case_id: str) -> str:
    return case_id.replace(".", "_").replace("/", "_")


def _tail(path: Path, limit: int = 4000) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-limit:]


def _process_group_exists(process_group_id: int) -> bool:
    if os.name != "posix":
        return False
    try:
        os.killpg(process_group_id, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _terminate_process_group(
    process: subprocess.Popen, grace_seconds: float = 5.0
) -> None:
    if os.name != "posix":
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        return
    process_group_id = process.pid
    if _process_group_exists(process_group_id):
        try:
            os.killpg(process_group_id, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_seconds
    while _process_group_exists(process_group_id) and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.05)
    if _process_group_exists(process_group_id):
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        kill_deadline = time.monotonic() + grace_seconds
        while (
            _process_group_exists(process_group_id) and time.monotonic() < kill_deadline
        ):
            process.poll()
            time.sleep(0.05)
    if process.poll() is None:
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _worker_environment(
    repository_root: Path,
    matplotlib_cache: Path,
    workspace: Path,
) -> Dict[str, str]:
    environment = dict(os.environ)
    for name in [
        "RANK",
        "GROUP_RANK",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "TORCHELASTIC_RUN_ID",
        "WORLD_SIZE",
    ]:
        environment.pop(name, None)
    environment.update(THREAD_SETTINGS)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "MPLCONFIGDIR": str(matplotlib_cache),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": str(repository_root),
            "PYTHONUNBUFFERED": "1",
            "TMP": str(workspace),
            "TEMP": str(workspace),
            "TMPDIR": str(workspace),
            "WANDB_MODE": "disabled",
            "WANDB_SILENT": "true",
        }
    )
    return environment


def _bounded_timeout(timeout_seconds: int, deadline: Optional[float]) -> float:
    if deadline is None:
        return float(timeout_seconds)
    remaining = deadline - time.monotonic()
    if remaining <= 0.0:
        raise BenchmarkRunError("benchmark suite exceeded its total time budget")
    return min(float(timeout_seconds), remaining)


def _prepare_matplotlib_cache(
    cache: Path,
    temporary_root: Path,
    timeout_seconds: int,
    deadline: Optional[float],
) -> None:
    repository_root = _repository_root()
    with tempfile.TemporaryDirectory(
        prefix="matplotlib-cache-",
        dir=str(temporary_root),
    ) as temporary_directory:
        workspace = Path(temporary_directory)
        environment = _worker_environment(repository_root, cache, workspace)
        command = [
            sys.executable,
            "-c",
            "import matplotlib.font_manager",
        ]
        process = subprocess.Popen(
            command,
            cwd=str(workspace),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=os.name == "posix",
        )
        try:
            output, _ = process.communicate(
                timeout=_bounded_timeout(timeout_seconds, deadline)
            )
        except subprocess.TimeoutExpired as error:
            _terminate_process_group(process)
            raise BenchmarkRunError(
                "Matplotlib cache preparation exceeded its timeout"
            ) from error
        except BaseException:
            _terminate_process_group(process)
            raise
        leaked_processes = _process_group_exists(process.pid)
        if leaked_processes:
            _terminate_process_group(process)
        if process.returncode != 0:
            raise BenchmarkRunError(
                "Matplotlib cache preparation failed"
                f"{' and leaked a child process' if leaked_processes else ''}:\n"
                f"{output[-4000:]}"
            )
        if leaked_processes:
            raise BenchmarkRunError(
                "Matplotlib cache preparation left a child process running"
            )


def _run_worker(
    spec: CaseSpec,
    parameters: Dict[str, Any],
    seed: int,
    mode: str,
    label: str,
    timeout_seconds: float,
    logs_root: Path,
    temporary_root: Path,
    matplotlib_cache: Path,
) -> Dict[str, Any]:
    repository_root = _repository_root()
    safe_id = _safe_case_id(spec.id)
    log_folder = logs_root / safe_id
    log_folder.mkdir(parents=True, exist_ok=True)
    stdout_path = log_folder / f"{label}.stdout.log"
    stderr_path = log_folder / f"{label}.stderr.log"
    with tempfile.TemporaryDirectory(
        prefix=f"{safe_id}-{label}-",
        dir=str(temporary_root),
    ) as temporary_directory:
        workspace = Path(temporary_directory)
        request_path = workspace / "request.json"
        result_path = workspace / "result.json"
        atomic_write_json(
            request_path,
            {
                "entrypoint": spec.entrypoint,
                "parameters": parameters,
                "seed": seed,
                "mode": mode,
            },
        )
        if "cold-import" in spec.tags:
            command = [
                sys.executable,
                "-m",
                "benchmarks._import_worker",
                parameters["module"],
                mode,
                str(result_path),
            ]
        else:
            command = [
                sys.executable,
                "-m",
                "benchmarks._worker",
                "--request",
                str(request_path),
                "--output",
                str(result_path),
            ]
        environment = _worker_environment(
            repository_root,
            matplotlib_cache,
            workspace,
        )
        with stdout_path.open("wb") as stdout_file:
            with stderr_path.open("wb") as stderr_file:
                process = subprocess.Popen(
                    command,
                    cwd=str(workspace),
                    env=environment,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    start_new_session=os.name == "posix",
                )
                try:
                    return_code = process.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired as error:
                    _terminate_process_group(process)
                    stdout_file.flush()
                    stderr_file.flush()
                    raise BenchmarkRunError(
                        f"'{spec.id}' {label} exceeded {timeout_seconds:.1f}s; "
                        f"stderr tail:\n{_tail(stderr_path)}"
                    ) from error
                except BaseException:
                    _terminate_process_group(process)
                    raise
        leaked_processes = _process_group_exists(process.pid)
        if leaked_processes:
            _terminate_process_group(process)
        if return_code != 0:
            leak_message = (
                "; a leaked child process was terminated" if leaked_processes else ""
            )
            raise BenchmarkRunError(
                f"'{spec.id}' {label} exited with code {return_code}{leak_message}; "
                f"stderr tail:\n{_tail(stderr_path)}"
            )
        if leaked_processes:
            raise BenchmarkRunError(f"'{spec.id}' {label} left a child process running")
        if not result_path.is_file():
            raise BenchmarkRunError(
                f"'{spec.id}' {label} did not produce a result file"
            )
        result = load_json(result_path)
    if stdout_path.stat().st_size == 0:
        stdout_path.unlink()
    if stderr_path.stat().st_size == 0:
        stderr_path.unlink()
    return result


def _sample_from_worker(result: Dict[str, Any], index: int) -> Dict[str, Any]:
    return {
        "index": index,
        "pid": result["pid"],
        "elapsed_ns": result["elapsed_ns"],
        "iterations": result["iterations"],
        "ns_per_iteration": result["ns_per_iteration"],
        "peak_rss_bytes": result["peak_rss_bytes"],
        "rss_method": result["rss_method"],
        "torch_memory": result["torch_memory"],
        "observations": result["observations"],
    }


def run_case(
    spec: CaseSpec,
    profile: str,
    seed: int,
    repeats: int,
    timeout_seconds: int,
    logs_root: Path,
    temporary_root: Path,
    matplotlib_cache: Path,
    deadline: Optional[float] = None,
) -> Dict[str, Any]:
    parameters = spec.profiles[profile]
    validation = _run_worker(
        spec,
        parameters,
        seed,
        "validate",
        "validation",
        _bounded_timeout(timeout_seconds, deadline),
        logs_root,
        temporary_root,
        matplotlib_cache,
    )
    behavior = validation["behavior"]
    behavior_sha256 = validation["behavior_sha256"]
    samples = []
    for index in range(repeats):
        result = _run_worker(
            spec,
            parameters,
            seed,
            "measure",
            f"sample-{index:02d}",
            _bounded_timeout(timeout_seconds, deadline),
            logs_root,
            temporary_root,
            matplotlib_cache,
        )
        if result["behavior_sha256"] != behavior_sha256:
            raise BenchmarkRunError(
                f"'{spec.id}' behavior changed between validation and sample {index}"
            )
        if result["behavior"] != behavior:
            raise BenchmarkRunError(
                f"'{spec.id}' behavior hash matched but summaries differed"
            )
        samples.append(_sample_from_worker(result, index))
    iterations = {sample["iterations"] for sample in samples}
    if len(iterations) != 1:
        raise BenchmarkRunError(
            f"'{spec.id}' used inconsistent iteration counts: {sorted(iterations)}"
        )
    peak_rss_values = [sample["peak_rss_bytes"] for sample in samples]
    if any(value is None for value in peak_rss_values):
        raise BenchmarkRunError(
            f"'{spec.id}' could not measure peak RSS on this runtime"
        )
    return {
        "id": spec.id,
        "variant": profile,
        "description": spec.description,
        "target": spec.target,
        "tags": list(spec.tags),
        "parameters": parameters,
        "behavior": {
            "status": "passed",
            "summary": behavior,
            "canonical_sha256": behavior_sha256,
        },
        "protocol": {
            "repeats": repeats,
            "behavior_validation_runs": 1,
            "warmups": 0,
            "iterations_per_sample": next(iter(iterations)),
            "process_model": "fresh_exec_per_validation_and_sample",
            "synchronization": "not_required_for_cpu",
        },
        "samples": samples,
        "statistics": {
            "time_ns_per_iteration": summarize(
                [float(sample["ns_per_iteration"]) for sample in samples]
            ),
            "peak_rss_bytes": summarize(
                [float(value) for value in peak_rss_values if value is not None]
            ),
        },
    }


def _run_git(repository_root: Path, arguments: List[str]) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=str(repository_root),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _dependency_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _cpu_model() -> str:
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.is_file():
        with cpu_info.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.partition(":")[2].strip()
    return platform.processor() or "unknown"


def _optional_hardware_info() -> Dict[str, Any]:
    try:
        import psutil

        physical_count = psutil.cpu_count(logical=False)
        total_ram_bytes = psutil.virtual_memory().total
    except ImportError:
        physical_count = None
        total_ram_bytes = None
    try:
        affinity_count = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity_count = None
    return {
        "model": _cpu_model(),
        "physical_count": physical_count,
        "logical_count": os.cpu_count(),
        "affinity_count": affinity_count,
        "total_ram_bytes": total_ram_bytes,
    }


def _installed_distributions_sha256() -> str:
    distributions = []
    for distribution in importlib_metadata.distributions():
        name = distribution.metadata.get("Name")
        if name is not None:
            distributions.append(f"{name.lower()}=={distribution.version}")
    payload = "\n".join(sorted(distributions)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _benchmark_suite_sha256(repository_root: Path) -> str:
    benchmark_root = repository_root / "benchmarks"
    entries = []
    for path in sorted(benchmark_root.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative_path = path.relative_to(repository_root).as_posix()
        entries.append(f"{relative_path}:{file_sha256(path)}")
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()


def collect_environment(repository_root: Path) -> Dict[str, Any]:
    constraints_path = repository_root / "requirements" / "constraints-ci.txt"
    commit = _run_git(repository_root, ["rev-parse", "HEAD"])
    status = _run_git(repository_root, ["status", "--porcelain"])
    return {
        "source": {
            "git_commit": commit,
            "dirty": None if status is None else bool(status),
            "constraints_path": constraints_path.relative_to(
                repository_root
            ).as_posix(),
            "constraints_sha256": (
                file_sha256(constraints_path) if constraints_path.is_file() else None
            ),
            "benchmark_suite_sha256": _benchmark_suite_sha256(repository_root),
        },
        "environment": {
            "python": {
                "implementation": platform.python_implementation(),
                "version": platform.python_version(),
                "executable": sys.executable,
            },
            "operating_system": {
                "system": platform.system(),
                "release": platform.release(),
                "kernel": platform.version(),
                "architecture": platform.machine(),
            },
            "cpu": _optional_hardware_info(),
            "dependencies": {
                "torch": _dependency_version("torch"),
                "accelerate": _dependency_version("accelerate"),
                "numpy": _dependency_version("numpy"),
                "psutil": _dependency_version("psutil"),
            },
            "installed_distributions_sha256": _installed_distributions_sha256(),
        },
    }


def build_report(
    profile: str,
    seed: int,
    repeats: int,
    timeout_seconds: int,
    total_timeout_seconds: int,
    cases: List[Dict[str, Any]],
    matplotlib_cache: Path,
) -> Dict[str, Any]:
    repository_root = _repository_root()
    collected = collect_environment(repository_root)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": collected["source"],
        "environment": collected["environment"],
        "configuration": {
            "profile": profile,
            "seed": seed,
            "repeats": repeats,
            "worker_timeout_seconds": timeout_seconds,
            "suite_timeout_seconds": total_timeout_seconds,
            "process_model": "fresh_exec_per_validation_and_sample",
            "statistics_algorithm": STATISTICS_ALGORITHM,
            "timing_clock": "time.perf_counter_ns",
            "rss_scope": "whole_worker_through_measurement",
            "torch_cpu_memory_measurement": "process_peak_rss",
            "cuda_policy": "disabled_for_cpu_suite",
            "matplotlib_cache": str(matplotlib_cache),
            "matplotlib_cache_policy": "shared_prewarmed_before_validation",
            "thread_environment": dict(THREAD_SETTINGS),
        },
        "cases": cases,
    }


def validate_semantics(report: Dict[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {report.get('schema_version')}")
    configuration = report["configuration"]
    expected_repeats = configuration["repeats"]
    if expected_repeats < 7:
        raise ValueError("benchmark reports require at least seven samples per case")
    case_ids = set()
    for case in report["cases"]:
        case_id = case["id"]
        if case_id in case_ids:
            raise ValueError(f"duplicate benchmark case id: '{case_id}'")
        case_ids.add(case_id)
        if case["variant"] != configuration["profile"]:
            raise ValueError(f"'{case_id}' variant is inconsistent with the profile")
        samples = case["samples"]
        if case["protocol"]["repeats"] != expected_repeats:
            raise ValueError(f"'{case_id}' protocol repeat count is inconsistent")
        if len(samples) != expected_repeats:
            raise ValueError(
                f"'{case_id}' has {len(samples)} samples, expected {expected_repeats}"
            )
        sample_indices = [sample["index"] for sample in samples]
        if sample_indices != list(range(expected_repeats)):
            raise ValueError(f"'{case_id}' sample indices are not consecutive")
        behavior = case["behavior"]
        expected_behavior_sha256 = canonical_sha256(behavior["summary"])
        if behavior["canonical_sha256"] != expected_behavior_sha256:
            raise ValueError(f"'{case_id}' has a stale behavior digest")
        expected_iterations = case["protocol"]["iterations_per_sample"]
        if any(sample["iterations"] != expected_iterations for sample in samples):
            raise ValueError(f"'{case_id}' has inconsistent sample iterations")
        for sample in samples:
            expected_ns_per_iteration = sample["elapsed_ns"] / sample["iterations"]
            if sample["ns_per_iteration"] != expected_ns_per_iteration:
                raise ValueError(
                    f"'{case_id}' sample {sample['index']} has inconsistent timing"
                )
        expected_time = summarize(
            [float(sample["ns_per_iteration"]) for sample in samples]
        )
        expected_rss = summarize(
            [float(sample["peak_rss_bytes"]) for sample in samples]
        )
        if case["statistics"]["time_ns_per_iteration"] != expected_time:
            raise ValueError(f"'{case_id}' has stale time statistics")
        if case["statistics"]["peak_rss_bytes"] != expected_rss:
            raise ValueError(f"'{case_id}' has stale RSS statistics")


def validate_report(report: Dict[str, Any]) -> None:
    validate_semantics(report)
    try:
        import jsonschema
    except ImportError as error:
        raise RuntimeError(
            "JSON schema validation requires the 'benchmark' optional dependencies"
        ) from error
    schema_path = Path(__file__).resolve().with_name("report-v1.schema.json")
    schema = load_json(schema_path)
    jsonschema.Draft202012Validator(schema).validate(report)


def run_suite(
    profile: str,
    output: Path,
    *,
    seed: int = DEFAULT_SEED,
    completed_cases: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if profile not in PROFILE_REPETITIONS:
        raise ValueError(f"unsupported profile: '{profile}'")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    repeats = PROFILE_REPETITIONS[profile]
    timeout_seconds = PROFILE_TIMEOUT_SECONDS[profile]
    total_timeout_seconds = PROFILE_TOTAL_TIMEOUT_SECONDS[profile]
    deadline = time.monotonic() + total_timeout_seconds
    logs_root = output.parent / f"{output.stem}.logs"
    matplotlib_cache = output.parent / f"{output.stem}.matplotlib"
    logs_root.mkdir(parents=True, exist_ok=True)
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    cases = [] if completed_cases is None else completed_cases
    specs = get_cases()
    with tempfile.TemporaryDirectory(
        prefix=f".{output.stem}.workers.",
        dir=str(output.parent),
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        _prepare_matplotlib_cache(
            matplotlib_cache,
            temporary_root,
            timeout_seconds,
            deadline,
        )
        for index, spec in enumerate(specs, start=1):
            print(f"[{index}/{len(specs)}] {spec.id}", flush=True)
            case = run_case(
                spec,
                profile,
                seed,
                repeats,
                timeout_seconds,
                logs_root,
                temporary_root,
                matplotlib_cache,
                deadline,
            )
            cases.append(case)
            median = case["statistics"]["time_ns_per_iteration"]["median"]
            print(f"  median {median / 1.0e6:.3f} ms/iteration", flush=True)
    report = build_report(
        profile,
        seed,
        repeats,
        timeout_seconds,
        total_timeout_seconds,
        cases,
        matplotlib_cache,
    )
    validate_semantics(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run reproducible carefree-core CPU performance baselines"
    )
    parser.add_argument("--profile", required=True, choices=sorted(PROFILE_REPETITIONS))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    partial_path = output.with_name(f"{output.stem}.partial{output.suffix}")
    completed_cases: List[Dict[str, Any]] = []
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _raise_termination_requested)
    try:
        report = run_suite(
            args.profile,
            output,
            completed_cases=completed_cases,
        )
        if args.validate:
            validate_report(report)
        atomic_write_json(output, report)
        if partial_path.is_file():
            partial_path.unlink()
    except BaseException as error:
        partial = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "completed_cases": completed_cases,
        }
        atomic_write_json(partial_path, partial)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
