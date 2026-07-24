import os
import sys
import time
import subprocess

from typing import Any
from typing import Dict
from pathlib import Path


def fail_case(params: Dict[str, Any], seed: int) -> Dict[str, Any]:
    del params
    del seed
    print("benchmark failure stdout marker", flush=True)
    print("benchmark failure stderr marker", file=sys.stderr, flush=True)
    raise RuntimeError("benchmark failure exception marker")


def hang_with_child(params: Dict[str, Any], seed: int) -> Dict[str, Any]:
    del seed
    pid_path = Path(params["pid_path"])
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(300)",
        ]
    )
    pid_path.write_text(str(child.pid), encoding="utf-8")
    print(
        f"benchmark timeout child pid {child.pid} from worker {os.getpid()}",
        file=sys.stderr,
        flush=True,
    )
    while True:
        time.sleep(1.0)


__all__ = [
    "fail_case",
    "hang_with_child",
]
