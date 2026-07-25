import json
import pytest

from pathlib import Path

from scripts.check_type_debt import MYPY_FLAGS
from scripts.check_type_debt import _load_baseline
from scripts.check_type_debt import _compare_limits
from scripts.check_type_debt import _scan_source_debt
from scripts.check_type_debt import _parse_mypy_output
from scripts.check_type_debt import _collect_python_files


def test_type_debt_source_scope(tmp_path: Path) -> None:
    included = tmp_path / "core" / "included.py"
    excluded = tmp_path / "core" / "flow" / "excluded.py"
    included.parent.mkdir()
    excluded.parent.mkdir()
    included.write_text(
        "@no_type_check\n"
        "def f():\n"
        "    return value  # type: ignore[no-any-return]\n",
        encoding="utf-8",
    )
    excluded.write_text(
        "value = 1  # type: ignore\n",
        encoding="utf-8",
    )

    files = _collect_python_files(tmp_path, ["core"], ["core/flow"])
    assert files == ["core/included.py"]
    assert _scan_source_debt(tmp_path, files) == {
        "no_type_check": {"core/included.py": 1},
        "type_ignore": {"core/included.py": 1},
    }


def test_type_debt_diagnostic_parser() -> None:
    diagnostics, unexpected = _parse_mypy_output(
        "core/a.py:3: error: Returning Any [no-any-return]\n"
        "core/a.py:4:5: error: Missing type arguments [type-arg]\n"
        "core/a.py:5: error: Unused ignore [unused-ignore]\n"
        "core/a.py:6: error: Other error [assignment]\n"
    )
    assert diagnostics == {
        "no-any-return": {"core/a.py": 1},
        "type-arg": {"core/a.py": 1},
        "unused-ignore": {"core/a.py": 1},
    }
    assert unexpected == ["core/a.py:6: error: Other error [assignment]"]


def test_type_debt_limit_is_a_per_file_ratchet() -> None:
    limits = {
        "category": {
            "total": 2,
            "by_file": {"core/a.py": 2},
        }
    }
    assert not _compare_limits(
        "group",
        {"category": {"core/a.py": 1}},
        limits,
    )
    assert _compare_limits(
        "group",
        {"category": {"core/a.py": 3, "core/new.py": 1}},
        limits,
    ) == [
        "group.category:core/a.py increased from 2 to 3",
        "group.category:core/new.py increased from 0 to 1",
        "group.category total increased from 2 to 4",
    ]


def test_type_debt_baseline_totals_are_self_consistent(tmp_path: Path) -> None:
    baseline = {
        "schema_version": 1,
        "measurement": {
            "mode": "combined-no-site-packages",
            "mypy_version": "1.11.2",
            "runtime_python": ["3.8", "3.14"],
            "target_python": "3.8",
            "platform_independent": True,
            "flags": list(MYPY_FLAGS),
        },
        "scope": {
            "include": ["core"],
            "exclude": [],
        },
        "limits": {
            "diagnostics": {
                "no-any-return": {"total": 1, "by_file": {}},
                "type-arg": {"total": 0, "by_file": {}},
                "unused-ignore": {"total": 0, "by_file": {}},
            },
            "source": {
                "no_type_check": {"total": 0, "by_file": {}},
                "type_ignore": {"total": 0, "by_file": {}},
            },
        },
    }
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(baseline), encoding="utf-8")
    with pytest.raises(ValueError, match="by_file sums to 0"):
        _load_baseline(path)
