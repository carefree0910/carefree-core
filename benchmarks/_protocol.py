import os
import json
import math
import hashlib
import tempfile
import statistics

from typing import Any
from typing import Dict
from typing import List
from typing import Union
from pathlib import Path

JSONValue = Union[
    None, bool, int, float, str, List["JSONValue"], Dict[str, "JSONValue"]
]


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    encoded = canonical_json(value).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def quantile_r7(values: List[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot calculate a quantile from an empty sequence")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability should be between 0 and 1")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    median = float(statistics.median(values))
    deviations = [abs(value - median) for value in values]
    q1 = quantile_r7(values, 0.25)
    q3 = quantile_r7(values, 0.75)
    return {
        "median": median,
        "mad": float(statistics.median(deviations)),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"expected a JSON object in '{path}'")
    return loaded


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


__all__ = [
    "JSONValue",
    "atomic_write_json",
    "canonical_json",
    "canonical_sha256",
    "file_sha256",
    "load_json",
    "quantile_r7",
    "summarize",
]
