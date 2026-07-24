from __future__ import annotations

import os
import sys
import signal
import subprocess

from typing import Dict
from typing import List
from typing import Tuple
from typing import Mapping
from typing import Optional
from pathlib import Path
from dataclasses import dataclass

_DISTRIBUTED_ENV_KEYS = (
    "GROUP_RANK",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "TORCHELASTIC_ERROR_FILE",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_RUN_ID",
    "WORLD_SIZE",
)


@dataclass(frozen=True)
class DDPResult:
    command: Tuple[str, ...]
    process_group_id: int
    returncode: int
    console_output: str
    rank_logs: Mapping[int, str]

    @property
    def diagnostics(self) -> str:
        sections = []
        if self.console_output.strip():
            sections.append("[launcher]\n" + self.console_output.rstrip())
        for rank, output in sorted(self.rank_logs.items()):
            sections.append(f"[rank {rank}]\n{output.rstrip()}")
        if not sections:
            return "<no subprocess output>"
        return "\n\n".join(sections)


class DDPProcessError(RuntimeError):
    def __init__(self, result: DDPResult) -> None:
        self.result = result
        super().__init__(
            f"CPU DDP subprocess exited with code {result.returncode}\n"
            f"{result.diagnostics}"
        )


class DDPTimeoutError(TimeoutError):
    def __init__(self, timeout: float, result: DDPResult) -> None:
        self.timeout = timeout
        self.result = result
        super().__init__(
            f"CPU DDP subprocess exceeded {timeout:.1f}s; "
            "the subprocess group was terminated\n"
            f"{result.diagnostics}"
        )


class CPUDistributedHarness:
    world_size = 2

    def __init__(
        self,
        workspace: Path,
        worker_path: Path,
        *,
        python_executable: Optional[str] = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.worker_path = worker_path.resolve()
        self.python_executable = python_executable or sys.executable

    def run(
        self,
        scenario: str,
        *,
        timeout: float = 30.0,
        shared_target: Optional[Path] = None,
    ) -> DDPResult:
        run_workspace = self.workspace / scenario
        log_dir = run_workspace / "logs"
        run_workspace.mkdir(parents=True, exist_ok=False)
        log_dir.mkdir()

        command = self._build_command(scenario, log_dir, shared_target)
        process = subprocess.Popen(
            command,
            cwd=str(run_workspace),
            env=self._build_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        timed_out = False
        try:
            console_output, _ = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            console_output = self._terminate_process_group(process)

        result = DDPResult(
            tuple(command),
            process.pid,
            process.returncode,
            console_output,
            self._collect_rank_logs(log_dir),
        )
        if timed_out:
            raise DDPTimeoutError(timeout, result)
        if result.returncode != 0:
            raise DDPProcessError(result)
        return result

    def _build_command(
        self,
        scenario: str,
        log_dir: Path,
        shared_target: Optional[Path],
    ) -> List[str]:
        command = [
            self.python_executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc-per-node={self.world_size}",
            "--max-restarts=0",
            "--monitor-interval=0.1",
            f"--log-dir={log_dir}",
            "--tee=3",
            str(self.worker_path),
            "--scenario",
            scenario,
        ]
        if shared_target is not None:
            command.extend(["--shared-target", str(shared_target.resolve())])
        return command

    def _build_environment(self) -> Dict[str, str]:
        environment = os.environ.copy()
        for key in _DISTRIBUTED_ENV_KEYS:
            environment.pop(key, None)
        # pytest-cov injects coverage into every descendant process. The timeout
        # scenario intentionally kills its workers, which can leave a partial
        # `.coverage.*` file in the repository. These test workers do not execute
        # production code whose coverage belongs in the parent report.
        for key in tuple(environment):
            if key.startswith("COV_CORE_"):
                environment.pop(key)
        repository_root = str(self.worker_path.parents[2])
        previous_pythonpath = environment.get("PYTHONPATH")
        pythonpath_parts = [repository_root]
        if previous_pythonpath:
            pythonpath_parts.append(previous_pythonpath)
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "",
                "GLOO_SOCKET_IFNAME": "lo",
                "MKL_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONHASHSEED": "0",
                "PYTHONPATH": os.pathsep.join(pythonpath_parts),
                "PYTHONUNBUFFERED": "1",
            }
        )
        return environment

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen) -> str:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            output, _ = process.communicate(timeout=5.0)
            return output
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            output, _ = process.communicate()
            return output

    @staticmethod
    def _collect_rank_logs(log_dir: Path) -> Dict[int, str]:
        streams: Dict[int, List[str]] = {}
        for log_path in sorted(log_dir.rglob("*.log")):
            rank_name = log_path.parent.name
            if not rank_name.isdigit():
                continue
            rank = int(rank_name)
            output = log_path.read_text(encoding="utf-8", errors="replace")
            if not output:
                continue
            streams.setdefault(rank, []).append(f"[{log_path.stem}]\n{output.rstrip()}")
        return {rank: "\n".join(outputs) for rank, outputs in streams.items()}
