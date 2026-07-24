from __future__ import annotations

import os
import pytest

from typing import Iterator
from pathlib import Path
from ddp_harness import DDPProcessError
from ddp_harness import DDPTimeoutError
from ddp_harness import CPUDistributedHarness


@pytest.fixture
def cpu_ddp(tmp_path: Path) -> Iterator[CPUDistributedHarness]:
    worker_path = Path(__file__).with_name("ddp_harness_task.py")
    yield CPUDistributedHarness(tmp_path, worker_path)


def test_cpu_ddp_minimal_success(cpu_ddp: CPUDistributedHarness) -> None:
    result = cpu_ddp.run("success")

    assert set(result.rank_logs) == {0, 1}
    assert "rank 0: all_reduce success" in result.rank_logs[0]
    assert "rank 1: all_reduce success" in result.rank_logs[1]


def test_cpu_ddp_rank_failure_is_propagated(
    cpu_ddp: CPUDistributedHarness,
) -> None:
    with pytest.raises(DDPProcessError) as error:
        cpu_ddp.run("rank_failure")

    diagnostics = str(error.value)
    assert "[rank 1]" in diagnostics
    assert "intentional failure from rank 1" in diagnostics
    assert "waiting for rank 1 failure propagation" in diagnostics


def test_cpu_ddp_only_global_main_writes_shared_target(
    cpu_ddp: CPUDistributedHarness,
    tmp_path: Path,
) -> None:
    shared_target = tmp_path / "shared.txt"
    result = cpu_ddp.run("global_main_write", shared_target=shared_target)

    assert shared_target.read_text(encoding="utf-8") == "rank=0\n"
    assert "rank 0: observed the global-main artifact" in result.rank_logs[0]
    assert "rank 1: observed the global-main artifact" in result.rank_logs[1]


def test_cpu_ddp_timeout_collects_logs_and_reaps_process_group(
    cpu_ddp: CPUDistributedHarness,
) -> None:
    with pytest.raises(DDPTimeoutError) as error:
        cpu_ddp.run("timeout", timeout=15.0)

    result = error.value.result
    assert set(result.rank_logs) == {0, 1}
    assert "rank 0: waiting for the parent timeout" in result.rank_logs[0]
    assert "rank 1: waiting for the parent timeout" in result.rank_logs[1]
    with pytest.raises(ProcessLookupError):
        os.killpg(result.process_group_id, 0)
