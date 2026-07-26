import sys
import json
import argparse

from typing import Any
from typing import Dict
from pathlib import Path


def _load_report(path: Path) -> Dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"coverage report does not exist: {path}")
    except json.JSONDecodeError as err:
        raise ValueError(f"invalid coverage JSON: {err}")
    if not isinstance(report, dict):
        raise ValueError("coverage report must be a JSON object")
    return report


def _get_object(report: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = report.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"coverage report {key!r} must be an object")
    return value


def _get_count(totals: Dict[str, Any], key: str) -> int:
    value = totals.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"coverage total {key!r} must be a non-negative integer")
    return value


def _validate_report(report: Dict[str, Any]) -> str:
    meta = _get_object(report, "meta")
    if meta.get("branch_coverage") is not True:
        raise ValueError("coverage report was not collected in branch mode")

    totals = _get_object(report, "totals")
    covered_lines = _get_count(totals, "covered_lines")
    missing_lines = _get_count(totals, "missing_lines")
    num_statements = _get_count(totals, "num_statements")
    num_branches = _get_count(totals, "num_branches")
    if num_statements == 0:
        raise ValueError("coverage report contains no statements")
    if num_branches == 0:
        raise ValueError("coverage report contains no branches")
    if covered_lines + missing_lines != num_statements:
        raise ValueError("coverage line totals are inconsistent")
    if missing_lines:
        raise ValueError(
            f"line coverage must remain 100%: {missing_lines} statements are missing"
        )
    return (
        f"line coverage: {covered_lines}/{num_statements}; "
        f"branch mode: {num_branches} branches"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keep line coverage at 100% while coverage.py checks the total."
    )
    parser.add_argument("report", type=Path, help="coverage.py JSON report")
    args = parser.parse_args()
    try:
        message = _validate_report(_load_report(args.report))
    except ValueError as err:
        print(f"coverage check failed: {err}", file=sys.stderr)
        raise SystemExit(1)
    print(f"coverage check passed: {message}")


if __name__ == "__main__":
    main()
