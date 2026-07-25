import re
import sys
import subprocess

from typing import Dict
from typing import List
from typing import Tuple
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
NEGATIVE_FOLDER = Path(__file__).resolve().parent / "negative"
EXPECTED_PATTERN = re.compile(
    r"# expected-mypy: "
    r"(?:(?P<line>\d+):)?"
    r"(?P<codes>[a-z0-9-]+(?:, [a-z0-9-]+)*)$",
)
ERROR_PATTERN = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+)(?::\d+)?: " r"error: .+ \[(?P<code>[a-z0-9-]+)\]$"
)


def get_expected_errors(path: Path) -> List[Tuple[int, str]]:
    expected_errors: List[Tuple[int, str]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        match = EXPECTED_PATTERN.search(line)
        if match is None:
            continue
        expected_line = match.group("line")
        if expected_line is None:
            expected_line_number = line_number
        else:
            expected_line_number = int(expected_line)
        expected_errors.extend(
            (expected_line_number, code) for code in match.group("codes").split(", ")
        )
    if not expected_errors:
        raise ValueError(f"cannot find expected mypy codes in '{path}'")
    return expected_errors


def normalize_path(path_string: str) -> str:
    path = Path(path_string)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve().relative_to(ROOT).as_posix()


def get_actual_errors(output: str) -> Dict[str, List[Tuple[int, str]]]:
    actual_errors: Dict[str, List[Tuple[int, str]]] = {}
    for line in output.splitlines():
        if ": error:" not in line:
            continue
        match = ERROR_PATTERN.search(line)
        if match is None:
            raise ValueError(f"mypy error has no error code: {line}")
        path = normalize_path(match.group("path"))
        error = (int(match.group("line")), match.group("code"))
        actual_errors.setdefault(path, []).append(error)
    return actual_errors


def main() -> int:
    paths = sorted(NEGATIVE_FOLDER.glob("*.py"))
    if not paths:
        print("no negative typing fixtures found", file=sys.stderr)
        return 1
    expected_errors = {
        path.relative_to(ROOT).as_posix(): get_expected_errors(path) for path in paths
    }
    with TemporaryDirectory(prefix="cfcore-mypy-") as cache_dir:
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "mypy",
                "--config-file",
                "pyproject.toml",
                "--show-error-codes",
                "--no-error-summary",
                "--no-pretty",
                "--no-site-packages",
                "--cache-dir",
                cache_dir,
                *(str(path.relative_to(ROOT)) for path in paths),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    output = process.stdout + process.stderr
    if process.returncode != 1:
        print(
            f"expected mypy exit 1, got {process.returncode}\n{output}",
            file=sys.stderr,
        )
        return 1
    try:
        actual_errors = get_actual_errors(output)
    except ValueError as err:
        print(f"{err}\n{output}", file=sys.stderr)
        return 1
    failures: List[str] = []
    for path in sorted(set(expected_errors) | set(actual_errors)):
        expected = sorted(expected_errors.get(path, []))
        actual = sorted(actual_errors.get(path, []))
        if actual != expected:
            failures.append(f"{path}: expected {expected}, got {actual}")
    if failures:
        failure_output = "\n".join(failures)
        print(f"{failure_output}\n{output}", file=sys.stderr)
        return 1
    print(f"checked {len(paths)} negative typing fixtures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
