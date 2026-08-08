import os
import sys
import torch
import pytest
import subprocess

from types import SimpleNamespace
from typing import Any
from typing import Dict
from typing import List
from typing import Iterator
from pathlib import Path
from core.learn.schema import prepare_dataloaders

from torch.utils.data import Dataset
from core.learn.schema import DataLoader
from accelerate import Accelerator
from core.learn.schema import IAsyncDataset
from accelerate.data_loader import DataLoaderShard
from core.learn.schema import AsyncIterManager
from core.learn.schema import AsyncDataLoaderIter
from accelerate.data_loader import DataLoaderDispatcher
from accelerate import DataLoaderConfiguration

_MISSING = object()
_ACCELERATE_LOADER_STATES = []
for _loader_type in [DataLoaderShard, DataLoaderDispatcher]:
    _marker = _loader_type.__dict__.get("_iter_prepared_", _MISSING)
    _ACCELERATE_LOADER_STATES.append((_loader_type, _loader_type.__iter__, _marker))


class _TrackingDataset(Dataset):
    reset_flags: List[bool]

    def __init__(self, size: int):
        self.size = size
        self.reset_flags = []

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {"value": torch.tensor(index)}

    def reset(self, *, for_inference: bool) -> None:
        self.reset_flags.append(for_inference)


class _TrackingAsyncDataset(IAsyncDataset):
    submitted: Dict[int, Any]
    reset_flags: List[bool]
    num_async_resets: int
    num_finalizes: int

    def __init__(self, size: int):
        self.size = size
        self.submitted = {}
        self.reset_flags = []
        self.num_async_resets = 0
        self.num_finalizes = 0

    def __len__(self) -> int:
        return self.size

    def reset(self, *, for_inference: bool) -> None:
        self.reset_flags.append(for_inference)

    def async_reset(self) -> None:
        self.num_async_resets += 1
        self.submitted = {}

    def async_submit(self, cursor: int, index: Any) -> bool:
        self.submitted[cursor] = index
        return True

    def async_fetch(self, cursor: int, index: Any) -> Dict[str, torch.Tensor]:
        return {"value": torch.as_tensor(self.submitted.pop(cursor))}

    def async_finalize(self) -> None:
        self.num_finalizes += 1
        self.submitted.clear()

    def async_recover(self) -> None:
        pass


class _TrackingData:
    config = SimpleNamespace(loader_seed_sync=True)

    process_flags: List[bool]
    num_processed: int

    def __init__(self):
        self.process_flags = []
        self.num_processed = 0

    def process_batch(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        for_inference: bool,
    ) -> Dict[str, torch.Tensor]:
        self.process_flags.append(for_inference)
        self.num_processed += 1
        return {"value": batch["value"] + 10}


def _make_loader(
    dataset: Any,
    data: _TrackingData,
    *,
    for_inference: bool,
    async_prefetch: bool,
) -> DataLoader:
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    loader.data = data
    loader.for_inference = for_inference
    loader.presend_device = None
    loader.async_prefetch = async_prefetch
    loader.async_prefetch_factor = 2
    return loader


@pytest.fixture(autouse=True)
def _restore_accelerate_loader_classes() -> Iterator[None]:
    def restore() -> None:
        for loader_type, original_iter, marker in _ACCELERATE_LOADER_STATES:
            loader_type.__iter__ = original_iter
            if marker is _MISSING:
                if "_iter_prepared_" in loader_type.__dict__:
                    delattr(loader_type, "_iter_prepared_")
            else:
                loader_type._iter_prepared_ = marker

    restore()
    yield
    restore()


@pytest.mark.parametrize("dispatch_batches", [False, True])
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="P0-05 phase 4: preparing a core loader should not affect Accelerate siblings",
)
def test_prepare_dataloader_keeps_accelerate_siblings_isolated(
    tmp_path: Path,
    dispatch_batches: bool,
) -> None:
    task_path = Path(__file__).with_name("prepare_dataloader_isolation_task.py")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    repo_root = task_path.parents[2]
    python_path = env.get("PYTHONPATH")
    if python_path is None:
        env["PYTHONPATH"] = str(repo_root)
    else:
        env["PYTHONPATH"] = os.pathsep.join([str(repo_root), python_path])
    command = [sys.executable, str(task_path)]
    if dispatch_batches:
        command.append("--dispatch-batches")
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    details = f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    if completed.returncode == 42:
        raise AssertionError(f"Accelerate sibling class was patched\n{details}")
    if completed.returncode != 0:
        raise RuntimeError(details)


@pytest.mark.parametrize(
    "dispatch_batches,expected_type",
    [
        (False, DataLoaderShard),
        (True, DataLoaderDispatcher),
    ],
)
def test_prepare_dataloader_preserves_accelerate_iteration(
    dispatch_batches: bool,
    expected_type: Any,
) -> None:
    dataset = _TrackingDataset(4)
    data = _TrackingData()
    loader = _make_loader(
        dataset,
        data,
        for_inference=True,
        async_prefetch=False,
    )
    accelerator = Accelerator(
        cpu=True,
        dataloader_config=DataLoaderConfiguration(
            dispatch_batches=dispatch_batches,
        ),
    )

    prepared = prepare_dataloaders(accelerator, loader)[0]

    assert type(prepared) is expected_type
    assert len(prepared) == 2
    assert prepared.iteration == 0
    for iteration in range(1, 3):
        values = [batch["value"].tolist() for batch in prepared]
        assert values == [[10, 11], [12, 13]]
        assert data.num_processed == 2 * iteration
        assert dataset.reset_flags == [True] * iteration
        assert prepared.iteration == iteration
        assert prepared.end_of_dataloader
    assert data.process_flags == [True] * 4


@pytest.mark.parametrize(
    "dispatch_batches,expected_type",
    [
        (False, DataLoaderShard),
        (True, DataLoaderDispatcher),
    ],
)
def test_prepare_async_dataloader_preserves_repeated_iteration(
    dispatch_batches: bool,
    expected_type: Any,
) -> None:
    dataset = _TrackingAsyncDataset(5)
    data = _TrackingData()
    loader = _make_loader(
        dataset,
        data,
        for_inference=False,
        async_prefetch=True,
    )
    accelerator = Accelerator(
        cpu=True,
        dataloader_config=DataLoaderConfiguration(
            dispatch_batches=dispatch_batches,
        ),
    )
    prepared = prepare_dataloaders(accelerator, loader)[0]
    base = prepared.base_dataloader

    try:
        assert type(prepared) is expected_type
        assert base.async_prefetch
        assert base.async_prefetch_factor == 2
        base_iterator = base.__iter__()
        assert isinstance(base_iterator, AsyncDataLoaderIter)
        for iteration in range(1, 3):
            values = [batch["value"].tolist() for batch in prepared]
            assert values == [[10, 11], [12, 13], [14]]
            assert data.num_processed == 3 * iteration
            assert dataset.reset_flags == [False] * iteration
            assert dataset.num_async_resets == iteration
            assert dataset.num_finalizes == iteration
            assert prepared.iteration == iteration
            assert prepared.end_of_dataloader
        assert data.process_flags == [False] * 6
    finally:
        AsyncIterManager.cleanup(id(base))
