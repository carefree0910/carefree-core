import re
import sys
import json
import argparse
import subprocess

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from pathlib import Path

TRACKED_DIAGNOSTICS = (
    "no-any-return",
    "type-arg",
    "unused-ignore",
)
TRACKED_SOURCE_DEBT = (
    "no_type_check",
    "type_ignore",
)
MYPY_FLAGS = (
    "--config-file=pyproject.toml",
    "--python-version=3.8",
    "--no-incremental",
    "--no-site-packages",
    "--warn-unused-ignores",
    "--disallow-any-generics",
    "--warn-return-any",
    "--show-error-codes",
    "--no-error-summary",
    "--no-pretty",
    "--no-color-output",
)

ERROR_PATTERN = re.compile(
    r"^(?P<path>.*?):\d+(?::\d+)?: error: .* \[(?P<code>[a-z0-9-]+)\]$"
)
NO_TYPE_CHECK_PATTERN = re.compile(r"^[ \t]*@no_type_check\b", re.MULTILINE)
TYPE_IGNORE_PATTERN = re.compile(r"#\s*type:\s*ignore\b")


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/")


def _is_excluded(path: str, exclusions: List[str]) -> bool:
    normalized = _normalize_path(path)
    return any(
        normalized == exclusion or normalized.startswith(f"{exclusion}/")
        for exclusion in exclusions
    )


def _load_baseline(path: Path) -> Dict[str, Any]:
    try:
        baseline = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"baseline does not exist: {path}")
    except json.JSONDecodeError as err:
        raise ValueError(f"invalid baseline JSON: {err}")
    if not isinstance(baseline, dict):
        raise ValueError("baseline must be a JSON object")
    if baseline.get("schema_version") != 1:
        raise ValueError("baseline schema_version must be 1")
    measurement = baseline.get("measurement")
    if not isinstance(measurement, dict):
        raise ValueError("baseline measurement must be an object")
    if measurement.get("mode") != "combined-no-site-packages":
        raise ValueError("baseline measurement.mode must be combined-no-site-packages")
    if measurement.get("mypy_version") != "1.11.2":
        raise ValueError("baseline measurement.mypy_version must be 1.11.2")
    if measurement.get("runtime_python") != ["3.8", "3.14"]:
        raise ValueError("baseline measurement.runtime_python must be ['3.8', '3.14']")
    if measurement.get("target_python") != "3.8":
        raise ValueError("baseline measurement.target_python must be 3.8")
    if measurement.get("platform_independent") is not True:
        raise ValueError("baseline measurement.platform_independent must be true")
    if measurement.get("flags") != list(MYPY_FLAGS):
        raise ValueError("baseline measurement.flags do not match the checker")
    scope = baseline.get("scope")
    if not isinstance(scope, dict):
        raise ValueError("baseline scope must be an object")
    for key in ["include", "exclude"]:
        values = scope.get(key)
        if not isinstance(values, list) or not all(
            isinstance(value, str) for value in values
        ):
            raise ValueError(f"baseline scope.{key} must be a list of strings")
    limits = baseline.get("limits")
    if not isinstance(limits, dict):
        raise ValueError("baseline limits must be an object")
    expected_categories = {
        "diagnostics": TRACKED_DIAGNOSTICS,
        "source": TRACKED_SOURCE_DEBT,
    }
    for group, categories in expected_categories.items():
        group_limits = limits.get(group)
        if not isinstance(group_limits, dict):
            raise ValueError(f"baseline limits.{group} must be an object")
        if set(group_limits) != set(categories):
            raise ValueError(
                f"baseline limits.{group} must contain exactly {sorted(categories)}"
            )
        for category, file_limits in group_limits.items():
            if not isinstance(file_limits, dict) or set(file_limits) != {
                "total",
                "by_file",
            }:
                raise ValueError(
                    f"baseline limits.{group}.{category} must contain "
                    "total and by_file"
                )
            total = file_limits["total"]
            by_file = file_limits["by_file"]
            if not isinstance(by_file, dict):
                raise ValueError(
                    f"baseline limits.{group}.{category}.by_file must be an object"
                )
            for file, limit in by_file.items():
                if not isinstance(file, str):
                    raise ValueError(
                        f"baseline limits.{group}.{category} paths must be strings"
                    )
                if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
                    raise ValueError(
                        f"baseline limit for {group}.{category}:{file} "
                        "must be a non-negative integer"
                    )
            if not isinstance(total, int) or isinstance(total, bool) or total < 0:
                raise ValueError(
                    f"baseline total for {group}.{category} "
                    "must be a non-negative integer"
                )
            measured_total = sum(by_file.values())
            if total != measured_total:
                raise ValueError(
                    f"baseline total for {group}.{category} is {total}, "
                    f"but by_file sums to {measured_total}"
                )
    return baseline


def _collect_python_files(
    root: Path,
    inclusions: List[str],
    exclusions: List[str],
) -> List[str]:
    files = []
    normalized_exclusions = [
        _normalize_path(exclusion).rstrip("/") for exclusion in exclusions
    ]
    for inclusion in inclusions:
        path = (root / inclusion).resolve()
        if root != path and root not in path.parents:
            raise ValueError(f"scope escapes repository root: {inclusion}")
        if not path.exists():
            raise ValueError(f"scope does not exist: {inclusion}")
        candidates = [path] if path.is_file() else path.rglob("*.py")
        for candidate in candidates:
            if candidate.suffix != ".py":
                continue
            relative = candidate.relative_to(root).as_posix()
            if not _is_excluded(relative, normalized_exclusions):
                files.append(relative)
    return sorted(set(files))


def _scan_source_debt(
    root: Path,
    files: List[str],
) -> Dict[str, Dict[str, int]]:
    debt: Dict[str, Dict[str, int]] = {category: {} for category in TRACKED_SOURCE_DEBT}
    for file in files:
        source = (root / file).read_text(encoding="utf-8")
        counts = {
            "no_type_check": len(NO_TYPE_CHECK_PATTERN.findall(source)),
            "type_ignore": len(TYPE_IGNORE_PATTERN.findall(source)),
        }
        for category, count in counts.items():
            if count:
                debt[category][file] = count
    return debt


def _parse_mypy_output(
    output: str,
) -> Tuple[Dict[str, Dict[str, int]], List[str]]:
    diagnostics: Dict[str, Dict[str, int]] = {
        category: {} for category in TRACKED_DIAGNOSTICS
    }
    unexpected = []
    for line in output.splitlines():
        match = ERROR_PATTERN.match(line)
        if match is None:
            if ": error:" in line:
                unexpected.append(line)
            continue
        code = match.group("code")
        if code not in diagnostics:
            unexpected.append(line)
            continue
        file = _normalize_path(match.group("path"))
        diagnostics[code][file] = diagnostics[code].get(file, 0) + 1
    return diagnostics, unexpected


def _run_command(command: List[str], root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _check_mypy_version(root: Path, expected: str) -> None:
    completed = _run_command([sys.executable, "-m", "mypy", "--version"], root)
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout.strip() or "could not run mypy")
    match = re.match(r"^mypy (?P<version>\S+)", completed.stdout)
    actual = match.group("version") if match is not None else ""
    if actual != expected:
        raise RuntimeError(f"expected mypy {expected}, found {actual or 'unknown'}")


def _run_mypy(
    root: Path,
    files: List[str],
) -> Tuple[Dict[str, Dict[str, int]], List[str]]:
    command = [
        sys.executable,
        "-m",
        "mypy",
        *MYPY_FLAGS,
        *files,
    ]
    completed = _run_command(command, root)
    diagnostics, unexpected = _parse_mypy_output(completed.stdout)
    if completed.returncode not in [0, 1]:
        unexpected.append(
            completed.stdout.strip()
            or f"mypy exited with status {completed.returncode}"
        )
    elif completed.returncode == 1 and not any(diagnostics.values()):
        unexpected.append(completed.stdout.strip() or "mypy failed without diagnostics")
    return diagnostics, unexpected


def _compare_limits(
    group: str,
    current: Dict[str, Dict[str, int]],
    limits: Dict[str, Dict[str, Any]],
) -> List[str]:
    failures = []
    for category, file_counts in current.items():
        category_limits = limits[category]
        file_limits = category_limits["by_file"]
        for file, count in file_counts.items():
            limit = file_limits.get(file, 0)
            if count > limit:
                failures.append(
                    f"{group}.{category}:{file} increased from {limit} to {count}"
                )
        current_total = sum(file_counts.values())
        if current_total > category_limits["total"]:
            failures.append(
                f"{group}.{category} total increased from "
                f"{category_limits['total']} to {current_total}"
            )
    return failures


def _print_summary(
    diagnostics: Dict[str, Dict[str, int]],
    source_debt: Dict[str, Dict[str, int]],
    limits: Dict[str, Dict[str, Any]],
) -> None:
    for group, current in [
        ("diagnostics", diagnostics),
        ("source", source_debt),
    ]:
        for category in sorted(current):
            current_total = sum(current[category].values())
            limit_total = limits[group][category]["total"]
            print(f"{group}.{category}: {current_total}/{limit_total}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reject increases to the committed mypy and source debt baseline."
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="baseline JSON path, relative to the repository root",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    baseline_path = Path(args.baseline)
    if not baseline_path.is_absolute():
        baseline_path = root / baseline_path
    try:
        baseline = _load_baseline(baseline_path)
        scope = baseline["scope"]
        files = _collect_python_files(root, scope["include"], scope["exclude"])
        measurement = baseline["measurement"]
        runtime_python = f"{sys.version_info.major}.{sys.version_info.minor}"
        if runtime_python not in measurement["runtime_python"]:
            raise RuntimeError(
                f"unsupported measurement runtime Python {runtime_python}"
            )
        _check_mypy_version(root, measurement["mypy_version"])
        diagnostics, unexpected = _run_mypy(root, files)
    except (RuntimeError, ValueError) as err:
        print(f"type debt check failed: {err}", file=sys.stderr)
        return 2

    source_debt = _scan_source_debt(root, files)
    limits = baseline["limits"]
    failures = unexpected
    failures.extend(_compare_limits("diagnostics", diagnostics, limits["diagnostics"]))
    failures.extend(_compare_limits("source", source_debt, limits["source"]))
    _print_summary(diagnostics, source_debt, limits)
    if failures:
        print("type debt check failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("type debt is within the committed baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
