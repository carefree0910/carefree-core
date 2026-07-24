import torch
import pytest

from types import SimpleNamespace
from typing import Any
from typing import Dict
from typing import Optional
from unittest.mock import patch
from core.learn.schema import DataLoader
from core.learn.schema import IAsyncDataset
from core.learn.schema import AsyncIterManager
from core.learn.schema import AsyncDataLoaderIter
from core.learn.schema import AsyncDataLoaderIterCallbacks


class _IdentityData:
    def process_batch(self, batch: Dict[str, torch.Tensor], *, for_inference: bool):
        return batch


class _RecoveringAsyncDataset(IAsyncDataset):
    def __init__(self, failure: Optional[str] = None):
        self.failure = failure
        self.failed = False
        self.events = []

    def __len__(self) -> int:
        return 1

    def async_reset(self) -> None:
        self.events.append("reset")
        self.submitted = {}

    def async_submit(self, cursor: int, index: Any) -> bool:
        self.events.append("submit")
        if self.failure == "submit" and not self.failed:
            self.failed = True
            return False
        self.submitted[cursor] = index
        return True

    def async_fetch(self, cursor: int, index: Any) -> Dict[str, torch.Tensor]:
        self.events.append("fetch")
        if self.failure == "fetch" and not self.failed:
            self.failed = True
            raise RuntimeError("fetch failed")
        return {"value": torch.as_tensor(self.submitted.pop(cursor))}

    def async_finalize(self) -> None:
        self.events.append("finalize")
        self.submitted.clear()

    def async_recover(self) -> None:
        self.events.append("recover")


def _make_loader(
    dataset: _RecoveringAsyncDataset,
    *,
    async_prefetch_factor: int,
) -> DataLoader:
    loader = DataLoader(dataset, batch_size=1)
    loader.data = _IdentityData()
    loader.for_inference = False
    loader.presend_device = None
    loader.async_prefetch = True
    loader.async_prefetch_factor = async_prefetch_factor
    return loader


def test_async_iterator_manager_remove_delegates_cleanup(monkeypatch) -> None:
    iterator = object()
    monkeypatch.setattr(AsyncIterManager, "_cur", {123: iterator})

    with patch.object(AsyncIterManager, "cleanup") as cleanup:
        AsyncIterManager.remove(iterator)

    cleanup.assert_called_once_with(123)


def test_async_iterator_callbacks_lifecycle() -> None:
    callbacks = AsyncDataLoaderIterCallbacks()
    deleted = []
    cleaned = []
    cpu_data = torch.arange(3).numpy()

    def delete(data) -> None:
        deleted.append(("always", data))

    def delete_once(data) -> bool:
        deleted.append(("once", data))
        return True

    def keep_deleting(data) -> bool:
        deleted.append(("keep", data))
        return False

    def cleanup() -> None:
        cleaned.append("always")

    def cleanup_once() -> None:
        cleaned.append("once")

    callbacks.register_del(delete)
    callbacks.register_del_once(delete_once)
    callbacks.register_del_once(keep_deleting)
    callbacks.call_del(cpu_data)
    callbacks.call_del(cpu_data)

    assert [name for name, _ in deleted] == [
        "always",
        "once",
        "keep",
        "always",
        "keep",
    ]
    assert callbacks.del_once_callbacks == [keep_deleting]

    callbacks.unregister_del(delete)
    with pytest.raises(RuntimeError, match="is not registered"):
        callbacks.unregister_del(delete)

    callbacks.register_cleanup(cleanup)
    callbacks.register_cleanup_once(cleanup_once)
    callbacks.call_cleanup()
    callbacks.call_cleanup()
    assert cleaned == ["always", "once", "always"]

    callbacks.unregister_cleanup(cleanup)
    callbacks.register_del(delete)
    callbacks.register_cleanup(cleanup)
    callbacks.register_cleanup_once(cleanup_once)
    callbacks.unregister_all()
    assert callbacks.del_callbacks == []
    assert callbacks.del_once_callbacks == []
    assert callbacks.cleanup_callbacks == []
    assert callbacks.cleanup_once_callbacks == []


def test_async_submit_presends_to_local_ddp_device() -> None:
    dataset = _RecoveringAsyncDataset()
    dataset.async_reset()
    iterator = object.__new__(AsyncDataLoaderIter)
    iterator._dataset = dataset
    iterator._results = {}
    iterator._pin_memory = False
    iterator.presend_device = "cuda"
    iterator.async_prefetch_factor = 2
    sent = {"value": torch.tensor([-1])}

    with patch(
        "core.learn.schema.get_ddp_info",
        return_value=SimpleNamespace(local_rank=3),
    ), patch(
        "core.learn.schema.send_to_device",
        return_value=sent,
    ) as send_to_device:
        iterator._async_submit(0, [0])

    cpu_data = iterator._results["cpu_0"]
    assert torch.equal(cpu_data["value"], torch.tensor([0]))
    assert iterator._results[0] is sent
    send_to_device.assert_called_once_with(
        cpu_data,
        "cuda:3",
        non_blocking=False,
    )


def test_async_loader_handles_partial_initial_prefetch() -> None:
    dataset = _RecoveringAsyncDataset()
    loader = _make_loader(dataset, async_prefetch_factor=2)
    iterator = iter(loader)

    try:
        batch = next(iterator)
        assert torch.equal(batch["value"], torch.tensor([0]))
        with pytest.raises(StopIteration):
            next(iterator)
        assert dataset.events == ["reset", "submit", "fetch", "finalize"]
    finally:
        AsyncIterManager.cleanup(id(loader))


@pytest.mark.parametrize("failure", ["submit", "fetch"])
def test_async_loader_recovers_from_failure(failure: str) -> None:
    dataset = _RecoveringAsyncDataset(failure)
    loader = _make_loader(dataset, async_prefetch_factor=1)
    iterator = iter(loader)

    try:
        with patch("core.learn.schema.console.error") as error:
            batch = next(iterator)
        assert torch.equal(batch["value"], torch.tensor([0]))
        with pytest.raises(StopIteration):
            next(iterator)
        assert dataset.events.count("recover") == 1
        assert dataset.events.count("reset") == 2
        assert dataset.events.count("finalize") == 2
        error.assert_called_once()
        if failure == "submit":
            assert "async submit failed" in error.call_args.args[0]
        else:
            assert "fetch failed" in error.call_args.args[0]
    finally:
        AsyncIterManager.cleanup(id(loader))
