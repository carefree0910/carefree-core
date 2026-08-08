import torch
import pytest
import threading

import core.learn as cflearn

from types import SimpleNamespace
from typing import Any
from typing import Dict
from typing import Optional
from unittest.mock import patch
from core.learn.schema import prepare_dataloaders

from core.learn.schema import DataLoader
from core.learn.schema import IAsyncDataset
from core.learn.schema import ADLI_CALLBACKS
from core.learn.schema import AsyncIterManager
from core.learn.schema import AsyncExceptionPack
from core.learn.schema import AsyncDataLoaderIter
from core.learn.schema import AsyncDataLoaderIterCallbacks
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter


class _IdentityData:
    def process_batch(self, batch: Dict[str, torch.Tensor], *, for_inference: bool):
        return batch


class _RecoveringAsyncDataset(IAsyncDataset):
    def __init__(self, failure: Optional[str] = None, size: int = 1):
        self.failure = failure
        self.size = size
        self.failed = False
        self.events = []

    def __len__(self) -> int:
        return self.size

    def async_reset(self) -> None:
        self.events.append("reset")
        self.submitted = {}

    def async_submit(self, cursor: int, index: Any) -> bool:
        self.events.append("submit")
        if self.failure == "submit" and cursor == 0 and not self.failed:
            self.failed = True
            return False
        self.submitted[cursor] = index
        return True

    def async_fetch(self, cursor: int, index: Any) -> Dict[str, torch.Tensor]:
        self.events.append("fetch")
        if self.failure == "fetch" and cursor == 0 and not self.failed:
            self.failed = True
            raise RuntimeError("fetch failed")
        return {"value": torch.as_tensor(self.submitted.pop(cursor))}

    def async_finalize(self) -> None:
        self.events.append("finalize")
        self.submitted.clear()

    def async_recover(self) -> None:
        self.events.append("recover")


class _OutOfOrderAsyncDataset(IAsyncDataset):
    def __init__(self, size: int):
        self.size = size
        self.lock = threading.Lock()
        self.submitted_condition = threading.Condition(self.lock)
        self.second_completed = threading.Event()
        self.submitted = {}
        self.submitted_cursors = []
        self.completed_cursors = []
        self.num_finalize = 0

    def __len__(self) -> int:
        return self.size

    def async_reset(self) -> None:
        with self.lock:
            self.submitted.clear()
            self.submitted_cursors.clear()
            self.completed_cursors.clear()
        self.second_completed.clear()

    def async_submit(self, cursor: int, index: Any) -> bool:
        with self.submitted_condition:
            self.submitted[cursor] = index
            self.submitted_cursors.append(cursor)
            self.submitted_condition.notify_all()
        return True

    def wait_for_submissions(self, count: int) -> bool:
        with self.submitted_condition:
            return self.submitted_condition.wait_for(
                lambda: len(self.submitted_cursors) >= count,
                timeout=5.0,
            )

    def async_fetch(self, cursor: int, index: Any) -> Dict[str, torch.Tensor]:
        if cursor == 0:
            self.second_completed.wait(5.0)
        with self.lock:
            value = self.submitted.pop(cursor)
            self.completed_cursors.append(cursor)
        if cursor == 1:
            self.second_completed.set()
        return {"value": torch.as_tensor(value)}

    def async_finalize(self) -> None:
        self.num_finalize += 1

    def async_recover(self) -> None:
        pass


class _PollingAsyncDataset(_RecoveringAsyncDataset):
    def __init__(self):
        super().__init__()
        self.num_fetches = 0

    def async_fetch(self, cursor: int, index: Any) -> Optional[Dict[str, torch.Tensor]]:
        self.num_fetches += 1
        if self.num_fetches < 3:
            return None
        return super().async_fetch(cursor, index)


def _make_loader(
    dataset: IAsyncDataset,
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


def _configure_fake_prepared(prepared: Any) -> None:
    prepared.base_dataloader = SimpleNamespace()
    prepared.device = "cpu"
    prepared.rng_types = []


def _make_fake_loader(
    process_batch: Any,
    *,
    for_inference: bool,
) -> Any:
    data = SimpleNamespace(
        config=SimpleNamespace(loader_seed_sync=True),
        process_batch=process_batch,
    )
    return SimpleNamespace(
        data=data,
        for_inference=for_inference,
        recover_labels=None,
        presend_device=None,
        async_prefetch=False,
        async_prefetch_factor=2,
    )


def _prepare_fake_loader(
    prepared: Any,
    process_batch: Any,
    *,
    for_inference: bool,
) -> Any:
    _configure_fake_prepared(prepared)
    loader = _make_fake_loader(process_batch, for_inference=for_inference)
    accelerator = SimpleNamespace(prepare=lambda *_: prepared)
    return prepare_dataloaders(accelerator, loader)[0]


def test_async_loader_public_contract() -> None:
    assert cflearn.DataLoader is DataLoader
    assert cflearn.IAsyncDataset is IAsyncDataset
    assert cflearn.prepare_dataloaders is prepare_dataloaders


def test_async_loader_extension_contract() -> None:
    pack = AsyncExceptionPack(1, [2], "failed")
    assert (pack.cursor, pack.index, pack.e) == (1, [2], "failed")
    assert isinstance(ADLI_CALLBACKS, AsyncDataLoaderIterCallbacks)
    assert issubclass(AsyncDataLoaderIter, _SingleProcessDataLoaderIter)
    for name in [
        "_initialize",
        "_cleanup",
        "_finalize",
        "_async_submit",
        "_handle_exception",
    ]:
        assert callable(getattr(AsyncDataLoaderIter, name))


def test_async_loader_config_serialization_contract() -> None:
    config = cflearn.DataConfig(
        presend_device="cpu",
        async_prefetch=True,
        async_prefetch_factor=3,
        async_prefetch_factor_for_validation=2,
    )
    restored = cflearn.DataConfig().from_info(config.to_info())
    assert restored.presend_device == "cpu"
    assert restored.async_prefetch
    assert restored.async_prefetch_factor == 3
    assert restored.async_prefetch_factor_for_validation == 2

    legacy = cflearn.DataConfig().from_info({"batch_size": 2})
    assert legacy.presend_device is None
    assert not legacy.async_prefetch
    assert legacy.async_prefetch_factor == 4
    assert legacy.async_prefetch_factor_for_validation is None


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


def test_async_loader_preserves_order_with_bounded_prefetch() -> None:
    dataset = _OutOfOrderAsyncDataset(4)
    loader = _make_loader(dataset, async_prefetch_factor=2)
    iterator = iter(loader)

    try:
        first = next(iterator)
        assert first["value"].item() == 0
        assert sorted(dataset.submitted_cursors) == [0, 1]

        second = next(iterator)
        assert dataset.wait_for_submissions(3)
        assert sorted(dataset.submitted_cursors) == [0, 1, 2]

        values = [first, second, *list(iterator)]
        assert [batch["value"].item() for batch in values] == [0, 1, 2, 3]
        assert dataset.completed_cursors[:2] == [1, 0]
        assert dataset.num_finalize == 1
    finally:
        AsyncIterManager.cleanup(id(loader))


def test_async_loader_keeps_extension_hooks_live(monkeypatch) -> None:
    initialized = []
    submitted = []
    original_initialize = AsyncDataLoaderIter._initialize

    def initialize(iterator: AsyncDataLoaderIter) -> None:
        initialized.append(iterator)
        original_initialize(iterator)

    def submit(iterator: AsyncDataLoaderIter, cursor: int, index: Any) -> None:
        submitted.append((cursor, index))
        iterator._results[cursor] = {"value": torch.as_tensor(index) + 10}

    monkeypatch.setattr(AsyncDataLoaderIter, "_initialize", initialize)
    monkeypatch.setattr(AsyncDataLoaderIter, "_async_submit", submit)
    dataset = _RecoveringAsyncDataset(size=3)
    loader = _make_loader(dataset, async_prefetch_factor=2)

    try:
        assert [batch["value"].item() for batch in loader] == [10, 11, 12]
        assert len(initialized) == 1
        assert sorted(cursor for cursor, _ in submitted) == [0, 1, 2]
    finally:
        AsyncIterManager.cleanup(id(loader))


def test_async_loader_waits_for_submitted_future() -> None:
    waited = []
    iterator = object.__new__(AsyncDataLoaderIter)
    iterator._futures = {0: SimpleNamespace(result=lambda: waited.append(True))}
    iterator._results = {0: "data"}

    assert iterator._poll(0) == "data"
    assert waited == [True]


def test_async_loader_propagates_unhandled_worker_exception(monkeypatch) -> None:
    def submit(_iterator: AsyncDataLoaderIter, _cursor: int, _index: Any) -> None:
        raise RuntimeError("worker failed")

    monkeypatch.setattr(AsyncDataLoaderIter, "_async_submit", submit)
    dataset = _RecoveringAsyncDataset()
    loader = _make_loader(dataset, async_prefetch_factor=1)

    try:
        with pytest.raises(RuntimeError, match="worker failed"):
            list(loader)
    finally:
        AsyncIterManager.cleanup(id(loader))


def test_async_loader_polls_until_data_is_ready() -> None:
    dataset = _PollingAsyncDataset()
    loader = _make_loader(dataset, async_prefetch_factor=1)

    try:
        values = [batch["value"].item() for batch in loader]
        assert values == [0]
        assert dataset.num_fetches == 3
    finally:
        AsyncIterManager.cleanup(id(loader))


def test_async_loader_restarts_after_early_exit() -> None:
    dataset = _RecoveringAsyncDataset(size=4)
    loader = _make_loader(dataset, async_prefetch_factor=2)
    first_iterator = iter(loader)

    try:
        assert next(first_iterator)["value"].item() == 0
        for _ in range(2):
            values = [batch["value"].item() for batch in loader]
            assert values == [0, 1, 2, 3]
        assert dataset.events.count("reset") == 3
        assert dataset.events.count("finalize") == 3
    finally:
        AsyncIterManager.cleanup(id(loader))


@pytest.mark.parametrize("failure", ["submit", "fetch"])
def test_async_loader_recovers_from_failure(failure: str) -> None:
    dataset = _RecoveringAsyncDataset(failure, size=5)
    loader = _make_loader(dataset, async_prefetch_factor=3)

    try:
        with patch("core.learn.schema.console.error") as error:
            values = [batch["value"].item() for batch in loader]
        assert values == [0, 1, 2, 3, 4]
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


def test_get_input_sample_restores_prefetch_factor() -> None:
    dataset = _RecoveringAsyncDataset(size=2)
    loader = _make_loader(dataset, async_prefetch_factor=3)

    try:
        sample = loader.get_input_sample()
        assert sample["value"].item() == 0
        assert loader.async_prefetch_factor == 3
    finally:
        AsyncIterManager.cleanup(id(loader))


def test_prepare_dataloader_adapts_builtin_iteration() -> None:
    events = []

    class Dataset:
        def reset(self, *, for_inference: bool) -> None:
            events.append(("reset", for_inference))

    class PreparedLoader:
        def __init__(self):
            self.dataset = Dataset()

        def __iter__(self):
            events.append(("iterate", None))
            return iter(
                [
                    {"value": torch.tensor([1])},
                    {"value": torch.tensor([2])},
                ]
            )

    def process_batch(batch, *, for_inference: bool):
        value = batch["value"]
        events.append(("process", for_inference, value.item()))
        return {"value": value + 10}

    target = PreparedLoader()
    prepared = _prepare_fake_loader(
        target,
        process_batch,
        for_inference=True,
    )
    assert prepared is target

    iterator = iter(prepared)
    assert events == []
    assert next(iterator)["value"].item() == 11
    assert events == [
        ("reset", True),
        ("iterate", None),
        ("process", True, 1),
    ]
    assert next(iterator)["value"].item() == 12
    with pytest.raises(StopIteration):
        next(iterator)

    batches = list(iter(prepared))
    assert [batch["value"].item() for batch in batches] == [11, 12]
    assert events == [
        ("reset", True),
        ("iterate", None),
        ("process", True, 1),
        ("process", True, 2),
        ("reset", True),
        ("iterate", None),
        ("process", True, 1),
        ("process", True, 2),
    ]


def test_prepare_dataloaders_keep_target_contexts_separate() -> None:
    process_events = []

    class Dataset:
        def __init__(self):
            self.reset_calls = []

        def reset(self, *, for_inference: bool) -> None:
            self.reset_calls.append(for_inference)

    class PreparedLoader:
        def __init__(self, value: int):
            self.value = value
            self.dataset = Dataset()

        def __iter__(self):
            return iter([{"value": torch.tensor([self.value])}])

    def make_process(name: str, offset: int):
        def process_batch(batch, *, for_inference: bool):
            process_events.append((name, for_inference))
            return {"value": batch["value"] + offset}

        return process_batch

    first = PreparedLoader(1)
    second = PreparedLoader(2)
    _configure_fake_prepared(first)
    _configure_fake_prepared(second)
    accelerator = SimpleNamespace(prepare=lambda *_: (first, second))
    prepared_first, prepared_second = prepare_dataloaders(
        accelerator,
        _make_fake_loader(
            make_process("first", 10),
            for_inference=False,
        ),
        _make_fake_loader(
            make_process("second", 100),
            for_inference=True,
        ),
    )

    assert list(iter(prepared_first))[0]["value"].item() == 11
    assert list(iter(prepared_second))[0]["value"].item() == 102
    assert list(iter(prepared_first))[0]["value"].item() == 11
    assert first.dataset.reset_calls == [False, False]
    assert second.dataset.reset_calls == [True]
    assert first.rng_types == []
    assert second.rng_types == []
    assert process_events == [
        ("first", False),
        ("second", True),
        ("first", False),
    ]


def test_prepare_dataloader_does_not_stack_iteration_adapters() -> None:
    reset_calls = []
    process_calls = []

    class Dataset:
        def reset(self, *, for_inference: bool) -> None:
            reset_calls.append(for_inference)

    class PreparedLoader:
        def __init__(self):
            self.dataset = Dataset()

        def __iter__(self):
            return iter([{"value": torch.tensor([1])}])

    def process_batch(batch, *, for_inference: bool):
        process_calls.append(for_inference)
        return {"value": batch["value"] + 1}

    target = PreparedLoader()
    first = _prepare_fake_loader(target, process_batch, for_inference=False)
    second = _prepare_fake_loader(first, process_batch, for_inference=False)

    assert first is target
    assert second is target
    assert list(iter(second))[0]["value"].item() == 2
    assert reset_calls == [False]
    assert process_calls == [False]


def test_empty_async_loader_stops_normally() -> None:
    dataset = _RecoveringAsyncDataset(size=0)
    loader = _make_loader(dataset, async_prefetch_factor=2)

    try:
        assert list(loader) == []
        assert dataset.events == ["reset", "finalize"]
    finally:
        AsyncIterManager.cleanup(id(loader))


def test_empty_async_iterator_finalizes_once() -> None:
    dataset = _RecoveringAsyncDataset(size=0)
    loader = _make_loader(dataset, async_prefetch_factor=2)
    iterator = AsyncDataLoaderIter(loader)

    try:
        for _ in range(2):
            with pytest.raises(StopIteration):
                next(iterator)
        assert dataset.events == ["reset", "finalize"]
    finally:
        AsyncIterManager.cleanup(id(loader))


@pytest.mark.xfail(
    strict=True,
    reason="P0-05 phase 4: preparing one loader should not affect its siblings",
)
def test_prepare_dataloader_does_not_affect_unprepared_instances() -> None:
    class Dataset:
        def __init__(self):
            self.reset_calls = []

        def reset(self, *, for_inference: bool) -> None:
            self.reset_calls.append(for_inference)

    class PreparedLoader:
        def __init__(self, value: int):
            self.value = value
            self.dataset = Dataset()
            self.for_inference = True
            self.process_calls = []
            self.num_iterations = 0
            self.data = SimpleNamespace(process_batch=self.process_batch)

        def process_batch(self, batch, *, for_inference: bool):
            self.process_calls.append(for_inference)
            return {"value": batch["value"] + 100}

        def __iter__(self):
            self.num_iterations += 1
            return iter([{"value": torch.tensor([self.value])}])

    existing = PreparedLoader(1)
    _prepare_fake_loader(
        PreparedLoader(0),
        lambda batch, *, for_inference: batch,
        for_inference=False,
    )

    for untouched, expected in [(existing, 1), (PreparedLoader(2), 2)]:
        batches = list(iter(untouched))
        assert len(batches) == 1
        assert batches[0]["value"].item() == expected
        assert untouched.num_iterations == 1
        assert untouched.dataset.reset_calls == []
        assert untouched.process_calls == []


@pytest.mark.xfail(
    strict=True,
    reason="P0-05 phase 5: temporary prefetch settings should be transactional",
)
def test_get_input_sample_restores_prefetch_factor_after_failure() -> None:
    dataset = _RecoveringAsyncDataset()
    loader = _make_loader(dataset, async_prefetch_factor=3)

    with patch.object(dataset, "pseudo_batch", side_effect=RuntimeError("failed")):
        with pytest.raises(RuntimeError, match="failed"):
            loader.get_input_sample()

    assert loader.async_prefetch_factor == 3
