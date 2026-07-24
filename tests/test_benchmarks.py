import os
import sys
import copy
import time
import psutil
import pytest
import tempfile
import subprocess

from benchmarks.run import run_case
from benchmarks.registry import get_cases
from benchmarks._protocol import summarize
from benchmarks.run import _run_worker
from benchmarks._protocol import quantile_r7
from benchmarks.run import build_report
from benchmarks.run import validate_report
from benchmarks.run import validate_semantics
from pathlib import Path
from benchmarks.registry import CaseSpec
from jsonschema.exceptions import ValidationError
from benchmarks.run import BenchmarkRunError


def test_benchmark_statistics() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 100.0]
    assert quantile_r7(values, 0.25) == 2.0
    assert quantile_r7(values, 0.75) == 4.0
    assert summarize(values) == {
        "median": 3.0,
        "mad": 1.0,
        "q1": 2.0,
        "q3": 4.0,
        "iqr": 2.0,
        "min": 1.0,
        "max": 100.0,
    }


def test_benchmark_registry_boundaries() -> None:
    cases = get_cases()
    ids = [case.id for case in cases]
    assert len(ids) == len(set(ids))
    assert set(ids) == {
        "import.core",
        "import.core_learn",
        "data.async_loader",
        "inference.padding",
        "trainer.train_step",
        "moe.dispatch_combine",
        "ema.update",
        "toolkit.pipeline_build",
        "toolkit.filter_kw",
        "learn.summary",
    }
    assert all(set(case.profiles) == {"smoke", "baseline"} for case in cases)
    assert all("flow" not in case.target.lower() for case in cases)
    assert all("serial" not in case.target.lower() for case in cases)


@pytest.mark.timeout(30)
def test_benchmark_worker_and_report_protocol() -> None:
    spec = get_cases()[0]
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        logs_root = root / "logs"
        workers_root = root / "workers"
        matplotlib_cache = root / "matplotlib"
        workers_root.mkdir()
        matplotlib_cache.mkdir()
        case = run_case(
            spec,
            "smoke",
            20240724,
            7,
            10,
            logs_root,
            workers_root,
            matplotlib_cache,
        )
        assert len(case["samples"]) == 7
        assert all(sample["pid"] != os.getpid() for sample in case["samples"])
        report = build_report(
            "smoke",
            20240724,
            7,
            10,
            60,
            [case],
            matplotlib_cache,
        )
        validate_semantics(report)
        validate_report(report)
        corrupted = copy.deepcopy(report)
        corrupted_sample = corrupted["cases"][0]["samples"][0]
        corrupted_sample["elapsed_ns"] += corrupted_sample["iterations"]
        corrupted_sample["ns_per_iteration"] += 1.0
        with pytest.raises(ValueError, match="stale time statistics"):
            validate_semantics(corrupted)
        corrupted = copy.deepcopy(report)
        corrupted["cases"][0]["protocol"]["repeats"] = 8
        with pytest.raises(ValueError, match="protocol repeat count"):
            validate_semantics(corrupted)
        corrupted = copy.deepcopy(report)
        corrupted["cases"][0]["samples"][0]["index"] = 1
        with pytest.raises(ValueError, match="sample indices"):
            validate_semantics(corrupted)
        corrupted = copy.deepcopy(report)
        corrupted["cases"][0]["samples"][0]["elapsed_ns"] += 1
        with pytest.raises(ValueError, match="inconsistent timing"):
            validate_semantics(corrupted)
        corrupted = copy.deepcopy(report)
        corrupted["cases"][0]["variant"] = "baseline"
        with pytest.raises(ValueError, match="variant"):
            validate_semantics(corrupted)
        corrupted = copy.deepcopy(report)
        corrupted.pop("source")
        with pytest.raises(ValidationError):
            validate_report(corrupted)


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


@pytest.mark.timeout(15)
def test_benchmark_worker_errors_and_reaps_process_group() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        logs_root = root / "logs"
        workers_root = root / "workers"
        matplotlib_cache = root / "matplotlib"
        workers_root.mkdir()
        matplotlib_cache.mkdir()
        failure_spec = CaseSpec(
            id="testing.failure",
            entrypoint="benchmarks.scenarios._testing:fail_case",
            description="failure protocol test",
            target="benchmarks/scenarios/_testing.py:fail_case",
            tags=("testing",),
            profiles={"smoke": {}, "baseline": {}},
        )
        with pytest.raises(BenchmarkRunError, match="failure exception marker"):
            _run_worker(
                failure_spec,
                {},
                20240724,
                "measure",
                "failure",
                2,
                logs_root,
                workers_root,
                matplotlib_cache,
            )
        failure_stderr = (
            logs_root / "testing_failure" / "failure.stderr.log"
        ).read_text(encoding="utf-8")
        assert "benchmark failure stderr marker" in failure_stderr

        child_pid_path = root / "child.pid"
        timeout_spec = CaseSpec(
            id="testing.timeout",
            entrypoint="benchmarks.scenarios._testing:hang_with_child",
            description="timeout protocol test",
            target="benchmarks/scenarios/_testing.py:hang_with_child",
            tags=("testing",),
            profiles={
                "smoke": {"pid_path": str(child_pid_path)},
                "baseline": {"pid_path": str(child_pid_path)},
            },
        )
        with pytest.raises(BenchmarkRunError, match="exceeded"):
            _run_worker(
                timeout_spec,
                timeout_spec.profiles["smoke"],
                20240724,
                "measure",
                "timeout",
                0.5,
                logs_root,
                workers_root,
                matplotlib_cache,
            )
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 2.0
        while _pid_exists(child_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _pid_exists(child_pid)
        timeout_stderr = (
            logs_root / "testing_timeout" / "timeout.stderr.log"
        ).read_text(encoding="utf-8")
        assert f"benchmark timeout child pid {child_pid}" in timeout_stderr


@pytest.mark.timeout(20)
def test_benchmark_runner_sigterm_reaps_active_worker() -> None:
    repository_root = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory() as temporary_directory:
        output = Path(temporary_directory) / "sigterm.json"
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(repository_root)
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "benchmarks.run",
                "--profile",
                "smoke",
                "--output",
                str(output),
            ],
            cwd=str(repository_root),
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        parent = psutil.Process(process.pid)
        children = []
        worker_modules = {
            "benchmarks._import_worker",
            "benchmarks._worker",
        }
        deadline = time.monotonic() + 10.0
        while not children and time.monotonic() < deadline:
            for child in parent.children(recursive=True):
                try:
                    command = child.cmdline()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                if any(argument in worker_modules for argument in command):
                    children.append(child)
            time.sleep(0.01)
        assert children
        child_pids = [child.pid for child in children]
        process.terminate()
        assert process.wait(timeout=10) != 0
        for child_pid in child_pids:
            child_deadline = time.monotonic() + 3.0
            while psutil.pid_exists(child_pid) and time.monotonic() < child_deadline:
                time.sleep(0.05)
            assert not psutil.pid_exists(child_pid)
        assert not output.is_file()
        assert output.with_name("sigterm.partial.json").is_file()
