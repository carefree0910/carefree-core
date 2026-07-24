import pytest
import asyncio
import threading

from typing import Set
from typing import AsyncIterator


def _non_daemon_threads() -> Set[threading.Thread]:
    return {
        thread
        for thread in threading.enumerate()
        if thread.is_alive() and not thread.daemon
    }


def _is_anyio_runner_task(task: "asyncio.Task[object]") -> bool:
    coroutine_name = getattr(task.get_coro(), "__qualname__", "")
    return coroutine_name == "TestRunner._call_in_runner_task"


@pytest.fixture
def anyio_backend() -> str:
    """Keep AnyIO tests on the asyncio backend used by the production helpers."""

    return "asyncio"


@pytest.fixture
async def no_async_resource_leaks(anyio_backend: str) -> AsyncIterator[None]:
    """Fail after an async test if it leaves a task or non-daemon thread behind."""

    # AnyIO keeps one runner task alive across fixture setup, the test body and
    # fixture teardown, so it belongs to the baseline rather than to the test.
    tasks_before = set(asyncio.all_tasks())
    threads_before = _non_daemon_threads()

    yield

    # Give callbacks scheduled by cancellation and task-group teardown one loop turn.
    await asyncio.sleep(0)
    current_task = asyncio.current_task()
    leaked_tasks = {
        task
        for task in asyncio.all_tasks()
        if task is not current_task
        and task not in tasks_before
        and not task.done()
        and not _is_anyio_runner_task(task)
    }
    leaked_threads = _non_daemon_threads() - threads_before

    # Do not let a failed assertion poison the AnyIO runner used by later tests.
    for task in leaked_tasks:
        task.cancel()
    if leaked_tasks:
        await asyncio.gather(*leaked_tasks, return_exceptions=True)

    task_names = sorted(task.get_name() for task in leaked_tasks)
    thread_names = sorted(thread.name for thread in leaked_threads)
    assert not task_names, f"unfinished asyncio tasks: {task_names}"
    assert not thread_names, f"leaked non-daemon threads: {thread_names}"
