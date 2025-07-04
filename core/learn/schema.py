import json
import math
import onnx
import time
import torch
import traceback

import numpy as np
import torch.nn as nn

from abc import abstractmethod
from abc import ABC
from abc import ABCMeta
from torch import device
from torch import Tensor
from typing import Any
from typing import Set
from typing import Dict
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import Generic
from typing import TypeVar
from typing import Callable
from typing import Iterator
from typing import Optional
from typing import NamedTuple
from typing import ContextManager
from datetime import timedelta
from accelerate import Accelerator
from accelerate import InitProcessGroupKwargs
from contextlib import nullcontext
from dataclasses import dataclass
from rich.progress import Progress
from torch.optim import Optimizer
from torch.profiler import profile
from pydantic.dataclasses import dataclass as pydantic_dataclass
from accelerate.utils import PrecisionType
from accelerate.utils import send_to_device
from accelerate.utils import extract_model_from_parallel
from concurrent.futures import ThreadPoolExecutor
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.checkpoint import checkpoint
from torch.utils.data.dataloader import _utils
from torch.utils.data.dataloader import _BaseDataLoaderIter
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter

from .toolkit import device_type
from .toolkit import get_device
from .toolkit import get_torch_device
from .toolkit import fix_denormal_states
from .toolkit import ONNX
from .toolkit import eval_context
from .toolkit import no_grad_context
from .toolkit import toggle_optimizer
from .constants import LOSS_KEY
from .constants import INPUT_KEY
from .constants import LABEL_KEY
from .constants import PREDICTIONS_KEY
from ..toolkit import console
from ..toolkit.misc import is_fsdp
from ..toolkit.misc import update_dict
from ..toolkit.misc import get_ddp_info
from ..toolkit.misc import safe_execute
from ..toolkit.misc import get_world_size
from ..toolkit.misc import is_local_rank_0
from ..toolkit.misc import shallow_copy_dict
from ..toolkit.misc import init_process_group
from ..toolkit.misc import WithRegister
from ..toolkit.misc import DataClassBase
from ..toolkit.misc import ISerializable
from ..toolkit.misc import ISerializableArrays
from ..toolkit.misc import ISerializableDataClass
from ..toolkit.array import to_numpy
from ..toolkit.array import to_torch
from ..toolkit.array import to_device
from ..toolkit.types import TPath
from ..toolkit.types import np_dict_type
from ..toolkit.types import tensor_dict_type
from ..toolkit.pipeline import IBlock
from ..toolkit.pipeline import IPipeline


# types


td_type = tensor_dict_type
data_type = Optional[Union[str, np.ndarray, np_dict_type, td_type, Any]]
configs_type = Optional[Union[List[Dict[str, Any]], Dict[str, Any]]]
sample_weights_type = Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]
raw_forward_results_type = Union[Tensor, td_type]
losses_type = Union[Tensor, td_type]

T_d = TypeVar("T_d", bound="IData", covariant=True)
TData = TypeVar("TData", bound="IData", covariant=True)
TDataset = TypeVar("TDataset", bound="IDataset", covariant=True)
TSplitSW = Tuple[Optional[np.ndarray], Optional[np.ndarray]]
TDs = Tuple["IDataset", Optional["IDataset"]]
TL = Optional["DataLoader"]
TLs = List[TL]
TDataLoaders = Tuple["DataLoader", Optional["DataLoader"]]
T_db = TypeVar("T_db", bound="IDataBlock", covariant=True)
TDataBlock = TypeVar("TDataBlock", bound="IDataBlock", covariant=True)


# collections


data_dict: Dict[str, Type["IData"]] = {}
data_configs: Dict[str, Type["DataConfig"]] = {}
monitors: Dict[str, Type["TrainerMonitor"]] = {}
metrics: Dict[str, Type["IMetric"]] = {}
models: Dict[str, Type["IModel"]] = {}
trainer_callbacks: Dict[str, Type["TrainerCallback"]] = {}
configs: Dict[str, Type["Config"]] = {}


# data


"""

Design of the `IData` system:

* `IData` is an `IPipeline` constructed by a series of `IDataBlock`, it may also
hold some data - which are constructed into a `DataBundle` - temporarily, in case
we need to use the data immediately (e.g. use them for training), or need to serialize them.
* Complicated logics are maintained in each `IDataBlock`.
* An `IDataBlock` need to do four jobs:
  * `transform`: transform a `DataBundle` into a new `DataBundle`.
  * `fit_transform`: collect necessary info and perform `transform`.
  * `process_batch` (optional): process an incoming batch at runtime.
  * `recover_labels` (optional): recover labels to their original format.


Typical workflows are:

* Training : raw data -> `fit_transform` -> transformed data
             -> data_loader -> `process_batch` -> model -> predictions
* Inference: raw data -> `transform` -> transformed data
             -> data_loader -> `process_batch` -> model -> predictions -> `recover_labels`

> When serializing, a property called `bundle` (the `DataBundle`) will be saved, which holds
the 'transformed data'. So after the serialization, we don't need to run `fit_transform` or
`transform` anymore, and can reuse the `bundle` property directly.
> However we can also serialize `IData` without saving `bundle` (which is a better choice when
we only want to serialize it for inference). In this case, we need to run `transform` on new datasets.


The above statements indicate that:
* `transform` / `fit_transform` are at the 'pre-calculation' stage.
* `process_batch` / `recover_labels` are at the 'runtime' stage.


Common use cases are:

* ML datasets: will mostly utilize `transform` / `fit_transform`, because most ML datasets
can be transfered into a numpy-based datasets, which should be calculated beforehand
because directly indexing numpy arrays is very fast while streaming them will be slow.

* CV / NLP datasets: will mostly utilize `process_batch`, because most CV / NLP datasets
are very large, which means it is impossible to be pre-calculated because that will
cost too much RAM. Instead, common practice is to 'stream' the datasets, which means many
calculations must be done at runtime.

* `recover_labels` might be used across all kinds of datasets, because labels may always need
to be transformed.

"""


def norm_sw(sample_weights: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if sample_weights is None:
        return None
    return sample_weights / sample_weights.sum()


def split_sw(sample_weights: sample_weights_type) -> TSplitSW:
    if sample_weights is None:
        train_weights = valid_weights = None
    else:
        if not isinstance(sample_weights, np.ndarray):
            train_weights, valid_weights = sample_weights
        else:
            train_weights, valid_weights = sample_weights, None
    train_weights, valid_weights = map(norm_sw, [train_weights, valid_weights])
    return train_weights, valid_weights


class IDataset(Dataset):
    """
    A thin wrapper of the `torch.utils.data.Dataset` class, which forces the user
    to implement the `__getitems__` method, which is used to get a collated batch of data
    with high performance.

    > It is also OK to return a torch-friendly-uncollated batch, and leave collation works
    to the default `collate_fn` given by PyTorch.
    > If you want to use the default `collate_fn`, you should set the `bypass_collate_fn`
    attribute of the `DataConfig` object to `False` (default is `True`, which means we want
    you to collate the batch by yourself in the `__getitems__` method).

    """

    @abstractmethod
    def __len__(self) -> int:
        """this method should return the number of samples of the dataset"""

    @abstractmethod
    def __getitems__(self, indices: List[int]) -> Any:
        """this method should return a collated batch of data"""

    # optional callbacks

    def __getitem__(self, index: int) -> Any:
        return self.__getitems__([index])

    def pseudo_batch(self, device: device_type = None) -> Optional[tensor_dict_type]:
        return None

    def reset(self, *, for_inference: bool) -> None:
        """this will be called everytime the `DataLoader` enters `__iter__`"""


@dataclass
class AsyncPack:
    cursor: int
    index: Any


@dataclass
class AsyncExceptionPack(AsyncPack):
    e: Union[str, Exception]


class IAsyncDataset(IDataset):
    """
    An async version of `IDataset`

    Notice that the APIs here are all synchronous, so it is designed to interact with other
    programming languages that can do the 'real' async I/O stuffs.
    """

    def __getitems__(self, indices: List[int]) -> Any:  # pragma: no cover
        raise NotImplementedError("should not call `__getitems__` of an async dataset")

    @abstractmethod
    def async_reset(self) -> None:
        """reset the dataset at the beginning of each epoch"""

    @abstractmethod
    def async_submit(self, cursor: int, index: Any) -> bool:
        """return whether the submission is successful"""

    @abstractmethod
    def async_fetch(self, cursor: int, index: Any) -> Optional[Any]:
        """fetch the data after submission, return None if not ready"""

    @abstractmethod
    def async_finalize(self) -> None:
        """finalize the dataset at the end of each epoch"""

    @abstractmethod
    def async_recover(self) -> None:
        """recover from exceptions, it is common to reset the related resources here"""

    def poll(self, cursor: int, index: Any) -> Any:
        while True:
            fetched = self.async_fetch(cursor, index)
            if fetched is not None:
                return fetched
            time.sleep(0.01)  # pragma: no cover


class AsyncIterManager:
    _cur: Dict[int, "AsyncDataLoaderIter"] = {}

    @classmethod
    def new(
        cls,
        id: int,
        fn: Callable[[], "AsyncDataLoaderIter"],
    ) -> "AsyncDataLoaderIter":
        cls.cleanup(id)
        cls._cur[id] = fn()
        return cls._cur[id]

    @classmethod
    def remove(cls, iter: "AsyncDataLoaderIter") -> None:
        for id, v in cls._cur.items():
            if v is iter:
                cls.cleanup(id)

    @classmethod
    def cleanup(cls, id: int) -> None:
        cur = cls._cur.pop(id, None)
        if cur is not None:
            if not cur._finalized:
                cur._cleanup()


class AsyncDataLoaderIter(_SingleProcessDataLoaderIter):
    _pool: ThreadPoolExecutor
    _queue: Optional[List[AsyncPack]]
    _drained: bool
    _queue_cursor: int
    _dataset: IAsyncDataset
    _finalized: bool
    _results: Dict[int, Any]

    def __init__(self, loader: "DataLoader"):
        super().__init__(loader)
        self.enabled = loader.async_prefetch
        self.presend_device = loader.presend_device
        self.async_prefetch_factor = loader.async_prefetch_factor
        if self.enabled and not isinstance(loader.dataset, IAsyncDataset):
            raise RuntimeError(
                "async prefetch is only available for `IAsyncDataset`"
            )  # pragma: no cover
        self._finalized = True
        self._initialized = False

    def __del__(self) -> None:
        AsyncIterManager.remove(self)

    def _initialize(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=self.async_prefetch_factor)
        self._queue = None
        self._drained = False
        self._queue_cursor = 0
        self._results = {}
        self._dataset.async_reset()
        self._finalized = False
        self._initialized = True

    def _cleanup(self) -> None:
        self._pool.shutdown()
        self._results.clear()
        self._dataset.async_finalize()

    def _finalize(self) -> None:
        self._cleanup()
        self._finalized = True
        raise StopIteration

    def _async_submit(self, cursor: int, index: Any) -> None:
        try:  # pragma: no cover
            if not self._dataset.async_submit(cursor, index):
                err_msg = "async submit failed"
                self._results[cursor] = AsyncExceptionPack(cursor, index, err_msg)
                return None
        except Exception as e:  # pragma: no cover
            self._results[cursor] = AsyncExceptionPack(cursor, index, e)
            return None
        try:
            data = self._dataset.poll(cursor, index)
        except KeyboardInterrupt:  # pragma: no cover
            raise
        except Exception as e:  # pragma: no cover
            self._results[cursor] = AsyncExceptionPack(cursor, index, e)
            return None
        if self._pin_memory:  # pragma: no cover
            data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)
        presend_device = self.presend_device
        if presend_device is not None:
            ddp_info = get_ddp_info()
            if presend_device == "cuda" and ddp_info is not None:
                presend_device = f"cuda:{ddp_info.local_rank}"
            data = send_to_device(data, presend_device, non_blocking=self._pin_memory)
        self._results[cursor] = data

    def _submit_next(self) -> None:
        cursor = self._queue_cursor
        index = self._next_index()
        self._queue.append(AsyncPack(cursor, index))  # type: ignore
        self._queue_cursor = cursor + 1
        self._pool.submit(self._async_submit, cursor, index)

    def _next_data(self) -> Any:
        if not self.enabled:
            return super()._next_data()
        if not self._initialized:
            self._initialize()
        if self._queue is None:
            self._queue = []
            try:
                for _ in range(self.async_prefetch_factor):
                    self._submit_next()
            except StopIteration:
                self._drained = True
        else:
            if not self._drained:
                try:
                    self._submit_next()
                except StopIteration:
                    self._drained = True
            if len(self._queue) == 0:
                self._finalize()
        return self._poll(self._queue.pop(0).cursor)

    def _poll(self, cursor: int) -> Any:
        while True:
            data = self._results.pop(cursor, None)
            if data is not None:
                if isinstance(data, AsyncExceptionPack):
                    return self._handle_exception(data)
                return data
            time.sleep(0.001)

    def _handle_exception(self, pack: AsyncExceptionPack) -> Any:
        if isinstance(pack.e, str):
            err_msg = pack.e
        else:
            err_msg = f"{pack.e}\n{''.join(traceback.format_tb(pack.e.__traceback__))}"
        console.error(f"trying to recover from error: {err_msg}")
        queue = self._queue or []
        queue_cursor = self._queue_cursor
        to_re_submit = queue + [pack]
        self._cleanup()
        self._dataset.async_recover()
        self._initialize()
        for re_submit in to_re_submit:
            self._pool.submit(self._async_submit, re_submit.cursor, re_submit.index)
        self._queue = queue
        self._queue_cursor = queue_cursor
        return self._poll(pack.cursor)


class DataLoader(TorchDataLoader):
    """
    A thin wrapper of the `torch.utils.data.DataLoader` class, which forces the user
    to assign the `data` attribute and the `for_inference` attribute, which are used
    to process the collated batch from `IDataset` objects.

    > If `async_prefetch` is set to `True`, the `DataLoader` will use the `AsyncDataLoaderIter`
    > to prefetch the data asynchronously.
    > It is not achieved by multi-processing, not even `asyncio` stuffs - just a set of synchronous
    > APIs with 'asynchronous design'.
    > And since python can hardly achieve real async I/O under the GIL, it is only useful when you can
    > offload the data loading to other programming languages.
    """

    data: "IData"
    dataset: IDataset
    for_inference: bool

    presend_device: Optional[str]
    async_prefetch: bool
    async_prefetch_factor: int

    def _get_iterator(self) -> _BaseDataLoaderIter:
        if self.num_workers == 0:
            return AsyncIterManager.new(id(self), lambda: AsyncDataLoaderIter(self))
        return super()._get_iterator()  # pragma: no cover

    def __iter__(self) -> Iterator[tensor_dict_type]:  # type: ignore
        self.dataset.reset(for_inference=self.for_inference)
        for batch in super().__iter__():
            yield self.data.process_batch(batch, for_inference=self.for_inference)

    def recover_labels(self, key: str, y: Tensor) -> Tensor:
        return self.data.recover_labels(key, y)

    def get_one_batch(self, device: device_type = None) -> tensor_dict_type:
        batch = next(iter(self))
        if device is not None:
            batch = to_device(batch, get_torch_device(device))
        return batch

    def get_full_batch(self, device: device_type = None) -> tensor_dict_type:
        full_batch: Dict[str, List[Tensor]] = {}
        for batch in self:
            for k, v in batch.items():
                if v is not None:
                    full_batch.setdefault(k, []).append(v)
        concat = {k: torch.cat(v, dim=0) for k, v in full_batch.items()}
        if device is not None:
            concat = to_device(concat, get_torch_device(device))
        return concat

    def get_input_sample(self, device: device_type = None) -> tensor_dict_type:
        prev_factor = self.async_prefetch_factor
        self.async_prefetch_factor = 1
        pseudo_batch = self.dataset.pseudo_batch(device)
        if pseudo_batch is None:
            pseudo_batch = self.get_one_batch(device)
        for k, v in pseudo_batch.items():
            if isinstance(v, Tensor):
                pseudo_batch[k] = v[:1]
            elif isinstance(v, list):
                pseudo_batch[k] = [vv[:1] if isinstance(vv, Tensor) else vv for vv in v]
            else:
                pseudo_batch[k] = v
        self.async_prefetch_factor = prev_factor
        return pseudo_batch


def prepare_dataloaders(accelerator: Accelerator, *loaders: TL) -> TLs:
    prepared_loaders = accelerator.prepare(*loaders)
    if len(loaders) == 1:
        prepared_loaders = [prepared_loaders]
    for loader, prepared in zip(loaders, prepared_loaders):
        if loader is not None:

            def _iter_factory(original_iter: Callable) -> Callable:
                def _iter(self: DataLoader) -> Iterator[tensor_dict_type]:
                    process_fn = self.data.process_batch
                    self.dataset.reset(for_inference=self.for_inference)
                    for batch in original_iter(self):
                        yield process_fn(batch, for_inference=self.for_inference)

                return _iter

            d = prepared
            d.data = loader.data
            d.for_inference = loader.for_inference
            d.recover_labels = loader.recover_labels
            base = d.base_dataloader
            base.presend_device = loader.presend_device
            base.async_prefetch = loader.async_prefetch
            base.async_prefetch_factor = loader.async_prefetch_factor
            if base.presend_device is not None:
                d.device = None
            if base.async_prefetch:
                get_iterator = (lambda ins: lambda: DataLoader._get_iterator(ins))(base)
                base._get_iterator = get_iterator
            td = type(d)
            iter_prepared = getattr(td, "_iter_prepared_", False)
            if not iter_prepared:
                td.__iter__ = _iter_factory(td.__iter__)
                td._iter_prepared_ = True
            if not loader.data.config.loader_seed_sync:
                prepared.rng_types = None
    return prepared_loaders


@pydantic_dataclass
class DataConfig(ISerializableDataClass["DataConfig"]):
    batch_size: int = 1
    valid_batch_size: Optional[int] = None
    shuffle_train: bool = True
    shuffle_valid: bool = False
    drop_train_last: bool = False
    block_names: Optional[List[str]] = None
    block_configs: Optional[Dict[str, Dict[str, Any]]] = None
    loader_configs: Optional[Dict[str, Any]] = None
    valid_loader_configs: Optional[Dict[str, Any]] = None
    loader_seed: Optional[int] = None
    loader_seed_sync: bool = True
    bypass_collate_fn: bool = True
    presend_device: Optional[str] = None
    # async prefetch configs
    async_prefetch: bool = False
    async_prefetch_factor: int = 4
    ## this will be used when `for_inference=True` or `is_validation=True`
    async_prefetch_factor_for_validation: Optional[int] = None

    def add_blocks(self, *blocks: Type["IDataBlock"]) -> None:
        if self.block_names is None:
            self.block_names = []
        for b in blocks:
            b_id = b.__identifier__
            if b_id in self.block_names:
                console.warn(f"block `{b_id}` already exists, it will be skipped")
                continue
            self.block_names.append(b_id)

    def set_blocks(self, *blocks: Type["IDataBlock"]) -> None:
        self.block_names = []
        self.block_configs = {}
        self.add_blocks(*blocks)


def check_data_is_info(data: data_type) -> bool:
    if (
        data is None
        or isinstance(data, dict)
        or isinstance(data, np.ndarray)
        or isinstance(data, Tensor)
    ):
        return False
    try:
        json.dumps([data])
        return True
    except:
        return False


@dataclass
class DataBundle(DataClassBase):
    x_train: data_type
    y_train: data_type = None
    x_valid: data_type = None
    y_valid: data_type = None

    def to_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        for k, v in self.asdict().items():
            if check_data_is_info(v):
                info[k] = v
        return info

    def from_info(self, info: Dict[str, Any]) -> "DataBundle":
        for k, v in info.items():
            setattr(self, k, v)
        return self

    def to_npd(self) -> np_dict_type:
        def _to_npd(d: Dict[str, Any]) -> np_dict_type:
            npd: np_dict_type = {}
            tensor_keys: List[str] = []
            for k, v in d.items():
                if isinstance(v, dict):
                    npd[k] = _to_npd(v)
                elif isinstance(v, np.ndarray):
                    npd[k] = v
                elif isinstance(v, Tensor):
                    tensor_keys.append(k)
                    npd[k] = to_numpy(v)
            if tensor_keys:
                npd["__tensor_keys__"] = np.array(tensor_keys)
            return npd

        return _to_npd(self.asdict())

    def from_npd(self, npd: np_dict_type) -> "DataBundle":
        def _convert(d: np_dict_type) -> None:
            tensor_keys = set(d.pop("__tensor_keys__", np.array([])).tolist())
            for k, v in d.items():
                if isinstance(v, dict):
                    _convert(v)
                elif k in tensor_keys:
                    d[k] = to_torch(v)

        _convert(npd)
        for k, v in npd.items():
            setattr(self, k, v)
        return self

    @classmethod
    def empty(cls) -> "DataBundle":
        return cls(None)


class IDataBlock(  # type: ignore
    Generic[TDataBlock],
    IBlock,
    ISerializable[TDataBlock],
    metaclass=ABCMeta,
):
    config: DataConfig

    @property
    def configs(self) -> Dict[str, Any]:
        """
        This property extract the corresponding config from the `block_configs`
        attribute of the `DataConfig` object.
        """

        return (self.config.block_configs or {}).setdefault(self.__identifier__, {})

    @property
    def is_local_rank_0(self) -> bool:
        return is_local_rank_0()

    def build(self, config: DataConfig) -> None:
        self.config = config

    def from_info(self: T_db, info: Dict[str, Any]) -> T_db:
        for k, v in info.items():
            setattr(self, k, v)
        return self

    # abstract

    @abstractmethod
    def transform(self, bundle: DataBundle, for_inference: bool) -> DataBundle:
        """
        This method should not utilize `config`, because when it is deserialized
        from a serialization file, the `config` will not be available and only
        `from_info` will be called, which only has access to the info returned
        from `to_info`.

        Changes can happen inplace.
        """

    @abstractmethod
    def fit_transform(self, bundle: DataBundle) -> DataBundle:
        """
        This method should prepare necessary info, which might be used
        in the `to_info` method.

        If any necessary info comes from `config`, this method should extract
        them and assign them to the corresponding properties.

        This method will NOT be called in a loading procedure, and the
        necessary info should be loaded in the `from_info` method.

        This method will always assume `for_inference=False`.

        Changes can happen inplace.
        """

    # optional callbacks

    ## changes can happen inplace
    def process_batch(self, batch: td_type, *, for_inference: bool) -> td_type:
        return batch

    ## changes can happen inplace
    def recover_labels(self, key: str, y: Tensor) -> Tensor:
        return y


def collate_placeholder(x: Any) -> Any:
    return x


class IData(  # type: ignore
    Generic[TData, TDataset],
    ISerializableArrays[TData],
    IPipeline[IDataBlock, DataConfig, TData],
    metaclass=ABCMeta,
):
    d = data_dict  # type: ignore

    train_dataset: TDataset
    valid_dataset: Optional[TDataset]

    train_weights: Optional[np.ndarray]
    valid_weights: Optional[np.ndarray]

    bundle: Optional[DataBundle]
    is_ready: bool = False

    def __init__(self) -> None:
        super().__init__()
        self.train_weights = None
        self.valid_weights = None

    # constructor
    ## this class is not supposed to be instantiated directly, but to be constructed
    ## with this `init` method.

    @classmethod
    def init(cls: Type[TData], config: Optional[DataConfig] = None) -> TData:
        self: TData = cls()
        self.bundle = None
        self.config = config or DataConfig()
        if self.config.block_names is not None:
            self.build(*(IDataBlock.get(name)() for name in self.config.block_names))  # type: ignore
        return self

    # abstract

    @abstractmethod
    def to_datasets(self, bundle: DataBundle, *, for_inference: Optional[bool]) -> TDs:
        """this method should return a tuple of datasets (train dataset, Optional[valid dataset])"""

    # inheritance

    @property
    def config_base(self) -> Type[DataConfig]:
        return DataConfig

    @property
    def block_base(self) -> Type[IDataBlock]:
        return IDataBlock

    def to_info(self) -> Dict[str, Any]:
        self._check_ready("to_info")
        info = super().to_info()
        info["bundle"] = None if self.bundle is None else self.bundle.to_info()
        return info

    def from_info(self: T_d, info: Dict[str, Any]) -> T_d:
        super().from_info(info)
        bundle_info = info["bundle"]
        if not bundle_info:
            self.bundle = None
        else:
            self.bundle = DataBundle.empty()
            self.bundle.from_info(bundle_info)
        return self

    def to_npd(self) -> np_dict_type:
        return {} if self.bundle is None else self.bundle.to_npd()

    def from_npd(self: T_d, npd: np_dict_type) -> T_d:
        if npd:
            if self.bundle is None:
                self.bundle = DataBundle.empty()
            self.bundle.from_npd(npd)
        return self

    def after_load(self) -> None:
        self.is_ready = True

    # optional callbacks

    def to_loader(
        self,
        dataset: IDataset,
        *,
        shuffle: bool,
        batch_size: int,
        for_inference: bool,
        is_validation: bool = False,
        sample_weights: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> DataLoader:
        if sample_weights is None or not shuffle:
            sampler = None
        else:
            shuffle = False
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))  # type: ignore
        loader_configs = shallow_copy_dict(self.config.loader_configs or {})
        if is_validation and self.config.valid_loader_configs is not None:
            valid_loader_configs = shallow_copy_dict(self.config.valid_loader_configs)
            loader_configs = update_dict(valid_loader_configs, loader_configs)
        kwargs = update_dict(kwargs, loader_configs)
        if self.config.loader_seed is not None:
            seed = self.config.loader_seed
        else:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)
        loader = DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            sampler=sampler,
            generator=generator,
            **kwargs,
        )
        loader.data = self
        loader.for_inference = for_inference
        loader.presend_device = self.config.presend_device
        loader.async_prefetch = self.config.async_prefetch
        if not is_validation and not for_inference:
            async_prefetch_factor = self.config.async_prefetch_factor
        else:
            async_prefetch_factor = (
                self.config.async_prefetch_factor_for_validation
                or self.config.async_prefetch_factor
            )
        loader.async_prefetch_factor = async_prefetch_factor
        if self.config.bypass_collate_fn:
            # this is useful when collation is already done in the `__getitems__` method
            loader.collate_fn = collate_placeholder
        return loader

    def get_bundle(
        self,
        x_train: data_type,
        y_train: Optional[data_type] = None,
        x_valid: Optional[data_type] = None,
        y_valid: Optional[data_type] = None,
        *args: Any,
        **kwargs: Any,
    ) -> DataBundle:
        return DataBundle(x_train, y_train, x_valid, y_valid)

    def set_sample_weights(self: TData, sample_weights: sample_weights_type) -> TData:
        self.train_weights, self.valid_weights = split_sw(sample_weights)
        return self

    # api

    def fit(
        self: TData,
        x_train: data_type,
        y_train: Optional[data_type] = None,
        x_valid: Optional[data_type] = None,
        y_valid: Optional[data_type] = None,
        *args: Any,
        **kwargs: Any,
    ) -> TData:
        args = x_train, y_train, x_valid, y_valid, *args
        bundle = self.get_bundle(*args, **kwargs)
        bundle = self._run("fit_transform", bundle, False)
        self.bundle = bundle
        self.is_ready = True
        return self

    def transform(
        self,
        x_train: data_type,
        y_train: Optional[data_type] = None,
        x_valid: Optional[data_type] = None,
        y_valid: Optional[data_type] = None,
        *args: Any,
        **kwargs: Any,
    ) -> DataBundle:
        self._check_ready("transform")
        bundle = self.get_bundle(x_train, y_train, x_valid, y_valid, *args, **kwargs)
        bundle = self._run("transform", bundle, for_inference=True)
        return bundle

    def build_loader(
        self,
        x: data_type,
        y: Optional[data_type] = None,
        *,
        shuffle: bool = False,
        batch_size: Optional[int] = None,
        for_inference: bool = True,
        sample_weights: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> DataLoader:
        self._check_ready("build_loader")
        bundle = self.transform(x, y)
        dataset = self.to_datasets(bundle, for_inference=for_inference)[0]
        loader = self.to_loader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size
            or self.config.valid_batch_size
            or self.config.batch_size,
            for_inference=for_inference,
            sample_weights=sample_weights,
            **kwargs,
        )
        return loader

    def build_loaders(self, *, for_inference: Optional[bool] = None) -> TDataLoaders:
        self._check_ready("build_loaders")
        if self.bundle is None:
            raise RuntimeError(
                "`bundle` property is not initialized, "
                "did you forget to call the `fit` method first?"
            )
        datasets = self.to_datasets(self.bundle, for_inference=for_inference)
        self.train_dataset, self.valid_dataset = datasets  # type: ignore
        train_loader = self.to_loader(
            self.train_dataset,
            shuffle=self.config.shuffle_train,
            batch_size=self.config.batch_size,
            for_inference=False if for_inference is None else for_inference,
            sample_weights=self.train_weights,
            drop_last=self.config.drop_train_last,
        )
        if self.valid_dataset is None:
            valid_loader = None
        else:
            valid_loader = self.to_loader(
                self.valid_dataset,
                shuffle=self.config.shuffle_valid,
                batch_size=self.config.valid_batch_size or self.config.batch_size,
                for_inference=True if for_inference is None else for_inference,
                is_validation=True,
                sample_weights=self.valid_weights,
            )
        return train_loader, valid_loader

    ## changes can happen inplace
    def process_batch(self, batch: td_type, *, for_inference: bool) -> td_type:
        for block in self.blocks:
            batch = block.process_batch(batch, for_inference=for_inference)
        return batch

    ## changes can happen inplace
    def recover_labels(self, key: str, y: Tensor) -> Tensor:
        for block in self.blocks[::-1]:
            y = block.recover_labels(key, y)
        return y

    # internal

    def _run(self, fn: str, bundle: DataBundle, for_inference: bool) -> DataBundle:
        kw = dict(bundle=bundle.copy(), for_inference=for_inference)
        previous: Dict[str, IDataBlock] = {}
        for block in self.blocks:
            block.previous = shallow_copy_dict(previous)
            kw["bundle"] = safe_execute(getattr(block, fn), kw)
            previous[block.__identifier__] = block
        return kw["bundle"]  # type: ignore

    def _check_ready(self, method_name: str) -> None:
        if not self.is_ready:
            raise RuntimeError(
                f"`{self.__class__.__name__}` should be ready before calling "
                f"`{method_name}`, did you forget to call the `fit` method first?"
            )


# loss


class ILoss(nn.Module, metaclass=ABCMeta):
    __identifier__: str

    @abstractmethod
    def forward(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional["TrainerState"] = None,
    ) -> losses_type:
        """this method should return the loss, it could be a tensor or a tensor dict"""


# metrics


class MetricsOutputs(NamedTuple):
    final_score: float
    metric_values: Dict[str, float]
    is_positive: Dict[str, bool]

    @classmethod
    def reduce(cls, outputs: List["MetricsOutputs"]) -> "MetricsOutputs":
        """
        reduce a list of `MetricsOutputs` to a single `MetricsOutputs` object.

        `MetricsOutputs` in `outputs` is considered to be the metrics of each 'batch',
        so after we evaluate all batches, we need to reduce them to a single `MetricsOutputs`
        for later use.
        """

        scores = []
        metric_values: Dict[str, List[float]] = {}
        for o in outputs:
            scores.append(o.final_score)
            for k, v in o.metric_values.items():
                metric_values.setdefault(k, []).append(v)
        return MetricsOutputs(
            sum(scores) / len(scores),
            {k: sum(vl) / len(vl) for k, vl in metric_values.items()},
            outputs[0].is_positive,
        )

    def union(self, other: "MetricsOutputs") -> "MetricsOutputs":
        """
        union two `MetricsOutputs` objects, which means we will combine the metric values
        and the `is_positive` values.
        """

        return MetricsOutputs(
            (self.final_score + other.final_score) / 2,
            {**self.metric_values, **other.metric_values},
            {**self.is_positive, **other.is_positive},
        )


class IMetric(WithRegister["IMetric"], metaclass=ABCMeta):
    d = metrics

    # abstract

    @property
    @abstractmethod
    def is_positive(self) -> bool:
        """
        Specify whether this Metric is a 'positive' metric.

        > A 'positive' metric means that the larger the better (e.g., accuracy),
        and a 'negative' metric means that the smaller the better (e.g., loss).

        """

    @abstractmethod
    def forward(
        self,
        np_batch: np_dict_type,
        np_outputs: np_dict_type,
        loader: Optional[DataLoader] = None,
    ) -> float:
        """this method should return the metric value, which should be a float"""

    # optional callbacks

    @property
    def not_include_in_score(self) -> bool:
        """
        Sometimes we may want to get some statistics from the predictions, but
        don't want to include them in the final score. In this case, we can
        set this property to `True`.
        """

        return False

    @property
    def requires_all(self) -> bool:
        """
        Specify whether this Metric needs 'all' data.

        > Typical metrics often does not need to evaluate itself on the entire dataset,
        but some does need to avoid corner cases. (for instance, the AUC metrics may
        fail to evaluate itself on only a batch, because the labels in this batch may
        be all the same, which breaks the calculation of AUC).
        """

        return False

    def requires(self, key: str) -> bool:
        """
        This method is only useful when `requires_all` returns `True`, and should tell
        the inference stage whether this metric needs a specific data key from either
        the `np_batch` or the `np_outputs`.

        > For instance, most of the metrics only needs the labels - which is often the
        value of `LABEL_KEY`, so it should return `False` when other keys are passed in.
        """

        return key == LABEL_KEY

    # api

    @classmethod
    def fuse(
        cls,
        names: Union[str, List[str]],
        configs: configs_type = None,
        *,
        metric_weights: Optional[Dict[str, float]] = None,
    ) -> "MultipleMetrics":
        metrics = IMetric.make_multiple(names, configs)
        if isinstance(metrics, IMetric):
            metrics = [metrics]
        return MultipleMetrics(metrics, weights=metric_weights)

    def evaluate(
        self,
        np_batch: np_dict_type,
        np_outputs: np_dict_type,
        loader: Optional[DataLoader] = None,
    ) -> MetricsOutputs:
        k = self.__identifier__
        metric = self.forward(np_batch, np_outputs, loader)
        score = metric * (1.0 if self.is_positive else -1.0)
        return MetricsOutputs(score, {k: metric}, {k: self.is_positive})


class IStreamMetric(IMetric):
    """
    an interface for metrics that support streaming calculation.

    this type of metrics are designed to run as follows:

    ```python
    metric = IStreamMetric()
    metric.reset()
    for batch in dataloader:
        outputs = model(batch)
        metric.update(batch, outputs)
    score = metric.finalize()
    ```
    """

    @abstractmethod
    def reset(self) -> None:
        """reset the streaming context"""

    @abstractmethod
    def update(
        self,
        np_batch: np_dict_type,
        np_outputs: np_dict_type,
        loader: Optional[DataLoader] = None,
    ) -> None:
        """update the streaming context with new batch & outputs"""

    @abstractmethod
    def finalize(self) -> float:
        """finalize the streaming context and return the final metric"""

    def forward(
        self,
        np_batch: np_dict_type,
        np_outputs: np_dict_type,
        loader: Optional[DataLoader] = None,
    ) -> float:
        raise RuntimeError(
            f"should not call `forward` of {self.__class__.__name__} directly, "
            "as it is a streaming metric"
        )

    def evaluate(
        self,
        np_batch: np_dict_type,
        np_outputs: np_dict_type,
        loader: Optional[DataLoader] = None,
    ) -> MetricsOutputs:
        raise RuntimeError(
            f"should not call `evaluate` of {self.__class__.__name__} directly, "
            "as it is a streaming metric"
        )

    def report(self, metric: float) -> MetricsOutputs:
        k = self.__identifier__
        score = metric * (1.0 if self.is_positive else -1.0)
        return MetricsOutputs(score, {k: metric}, {k: self.is_positive})


class MultipleMetrics(IMetric):
    def __init__(
        self,
        metric_list: List[IMetric],
        *,
        weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.metrics = metric_list
        self.weights = weights or {}
        self.__identifier__ = " | ".join(m.__identifier__ for m in metric_list)

    @property
    def is_positive(self) -> bool:
        raise NotImplementedError

    @property
    def has_streaming(self) -> bool:
        return any(isinstance(metric, IStreamMetric) for metric in self.metrics)

    @property
    def requires_all(self) -> bool:
        requires_all = any(m.requires_all for m in self.metrics)
        any_streaming = any(isinstance(m, IStreamMetric) for m in self.metrics)
        if requires_all and any_streaming:
            raise RuntimeError("streaming metrics should not `requires_all`")
        return requires_all

    def requires(self, key: str) -> bool:
        return any(metric.requires(key) for metric in self.metrics)

    def forward(
        self,
        np_batch: np_dict_type,
        np_outputs: np_dict_type,
        loader: Optional[DataLoader] = None,
    ) -> float:
        raise NotImplementedError

    def evaluate(
        self,
        np_batch: np_dict_type,
        np_outputs: np_dict_type,
        loader: Optional[DataLoader] = None,
        for_streaming: bool = False,
    ) -> MetricsOutputs:
        scores: List[float] = []
        weights: List[float] = []
        metrics_values: Dict[str, float] = {}
        is_positive: Dict[str, bool] = {}
        for metric in self.metrics:
            if isinstance(metric, IStreamMetric):
                if not for_streaming:
                    continue
                metric_outputs = metric.report(metric.finalize())
            else:
                if for_streaming:
                    continue
                metric_outputs = metric.evaluate(np_batch, np_outputs, loader)
            metrics_values.update(metric_outputs.metric_values)
            is_positive.update(metric_outputs.is_positive)
            if not metric.not_include_in_score:
                w = self.weights.get(metric.__identifier__, 1.0)
                weights.append(w)
                scores.append(metric_outputs.final_score * w)
        if not scores:
            final_score = 0.0
        else:
            final_score = sum(scores) / (sum(weights) + 1.0e-12)
        return MetricsOutputs(final_score, metrics_values, is_positive)

    def reset(self) -> None:
        for metric in self.metrics:
            if isinstance(metric, IStreamMetric):
                metric.reset()

    def update(
        self,
        np_batch: np_dict_type,
        np_outputs: np_dict_type,
        loader: Optional[DataLoader] = None,
    ) -> None:
        for metric in self.metrics:
            if isinstance(metric, IStreamMetric):
                metric.update(np_batch, np_outputs, loader)

    def finalize(self) -> MetricsOutputs:
        return self.evaluate({}, {}, for_streaming=True)


# inference


@dataclass
class InferenceOutputs:
    forward_results: np_dict_type
    labels: np_dict_type
    metric_outputs: Optional[MetricsOutputs]
    loss_items: Optional[Dict[str, float]]


class IInference(ABC):
    onnx: Optional[ONNX]
    model: Optional["IModel"]
    use_grad_in_predict = False

    @abstractmethod
    def get_outputs(
        self,
        loader: DataLoader,
        *,
        portion: float = 1.0,
        metrics: Optional[IMetric] = None,
        use_losses_as_metrics: bool = False,
        return_outputs: bool = True,
        target_outputs: Union[str, List[str]] = PREDICTIONS_KEY,
        recover_labels: bool = True,
        recover_predictions: bool = True,
        return_labels: bool = False,
        target_labels: Union[str, List[str]] = LABEL_KEY,
        stack_outputs: bool = True,
        progress: Optional[Progress] = None,
        progress_kwargs: Optional[Dict[str, Any]] = None,
        use_inference_mode: Optional[bool] = None,
        accelerator: Optional[Accelerator] = None,
        pad_dim: Optional[Union[int, Dict[str, int]]] = None,
        only_hold_data_on_rank_0: bool = False,
        verbose: bool = True,
        **kwargs: Any,
    ) -> InferenceOutputs:
        """this is an internal interface and will be implemented internally"""


# general model


def weighted_loss_score(config: "TrainerConfig", loss_items: Dict[str, float]) -> float:
    if not config.loss_metrics_weights:
        if not loss_items:
            return 0.0
        loss = loss_items.get(LOSS_KEY)
        if loss is not None:
            return -loss
        return -sum(loss_items.values()) / len(loss_items)
    score = 0.0
    for k, w in config.loss_metrics_weights.items():
        v = loss_items.get(k)
        if v is None:
            continue
        score -= v * w
    return score


def get_update_fn(
    trainer: "ITrainer",
) -> Callable[
    [tensor_dict_type, tensor_dict_type, "TrainStepLoss", Optimizer, bool], None
]:
    def update_fn(
        batch: tensor_dict_type,
        forward: tensor_dict_type,
        loss_res: TrainStepLoss,
        optimizer: Optimizer,
        update: bool,
    ) -> None:
        trainer.accelerator.backward(loss_res.loss)
        for c in trainer.callbacks:
            c.before_gradient_update(trainer, batch, forward, loss_res, update)
        if update:
            optimizer.step()
            optimizer.zero_grad()

    return update_fn


def no_sync_context(update: bool, trainer: "ITrainer") -> ContextManager:
    return nullcontext() if update else trainer.accelerator.no_sync(trainer.model.m)


def get_inputs(model: onnx.ModelProto) -> List[onnx.ValueInfoProto]:
    initializer_names = [x.name for x in model.graph.initializer]
    return [inp for inp in model.graph.input if inp.name not in initializer_names]


def get_input_names(model: onnx.ModelProto) -> List[str]:
    input_names = [inp.name for inp in get_inputs(model)]
    return input_names


@dataclass
class StepOutputs:
    forward_results: tensor_dict_type
    loss_tensors: tensor_dict_type

    @property
    def loss_items(self) -> Dict[str, float]:
        return {k: v.item() for k, v in self.loss_tensors.items()}


class TrainStepLoss(NamedTuple):
    loss: Tensor
    loss_tensors: tensor_dict_type


class TrainStep(ABC):
    def __init__(
        self,
        scope: str = "all",
        *,
        grad_accumulate: Optional[int] = None,
        requires_new_forward: bool = False,
        requires_grad_in_forward: bool = True,
        enable_toggle_optimizer: bool = True,
    ) -> None:
        self.scope = scope
        self.grad_accumulate = grad_accumulate
        self.requires_new_forward = requires_new_forward
        self.requires_grad_in_forward = requires_grad_in_forward
        self.enable_toggle_optimizer = enable_toggle_optimizer

    @abstractmethod
    def loss_fn(
        self,
        m: "IModel",
        state: Optional["TrainerState"],
        batch: tensor_dict_type,
        forward_results: tensor_dict_type,
        **kwargs: Any,
    ) -> TrainStepLoss:
        """this method should return the `TrainStepLoss` object based on the given inputs"""

    # optional callbacks

    def get_default_optimizer_settings(self) -> Optional[Dict[str, Any]]:
        return None

    def should_skip(self, m: "IModel", state: Optional["TrainerState"]) -> bool:
        return False

    def callback(
        self,
        m: "IModel",
        trainer: "ITrainer",
        batch: tensor_dict_type,
        forward_results: tensor_dict_type,
    ) -> None:
        pass


class IModel(WithRegister["IModel"], metaclass=ABCMeta):
    d = models

    def __str__(self) -> str:
        module_str = str(nn.ModuleList(self.all_modules))
        module_str = "\n".join(["["] + module_str.split("\n")[1:-1] + ["]"])
        return f"{self.__class__.__name__}({module_str})"

    __repr__ = __str__

    def __call__(self, *args: Any, **kwargs: Any) -> raw_forward_results_type:
        use_checkpoint = kwargs.pop("use_checkpoint", False)
        with self.checkpoint_context(enable=use_checkpoint):
            return self.forward(*args, **kwargs)

    def checkpoint_context(self, *, enable: bool = True) -> ContextManager:

        class ctx:
            m: nn.Module

            def __enter__(self) -> None:
                def ckpt_fw(*args: Any, **kwargs: Any) -> raw_forward_results_type:
                    kwargs.setdefault("use_reentrant", False)
                    return checkpoint(self.m, *args, **kwargs)

                if not enable:
                    return None
                self.m = model.m
                model.m = ckpt_fw  # type: ignore

            def __exit__(self, *args: Any) -> None:
                if not enable:
                    return None
                model.m = self.m

        model = self
        return ctx()

    # constructors
    ## this class is not supposed to be instantiated directly, but to be constructed
    ## by one of the following methods.

    @classmethod
    def load(cls, path: TPath, strict: bool = True) -> "IModel":
        full = torch.load(path, weights_only=False)
        self = cls.from_config(Config(**full["config"]))
        self.load_state_dict(full["states"], strict)
        return self

    @classmethod
    def from_config(cls, config: "Config") -> "IModel":
        self = cls.make(config.model, {})
        self.config = config.copy()
        self.build(config)
        return self

    # abstract

    m: nn.Module
    config: "Config"

    @property
    @abstractmethod
    def train_steps(self) -> List[TrainStep]:
        """this property should return a list of `TrainStep` objects, which will be used in the training loop"""

    @property
    @abstractmethod
    def all_modules(self) -> List[nn.Module]:
        """this property should return a list of all modules in the model, so we can manage the states (e.g., .eval()) of them"""

    @abstractmethod
    def build(self, config: "Config") -> None:
        """this method should build the model based on the given config"""

    # optional callbacks

    def from_accelerator(self, *args: nn.Module) -> "IModel":
        cloned = IModel.make(self.config.model, {})
        cloned.config = self.config.copy()
        for i, k in enumerate(self.all_module_names):
            setattr(cloned, k, args[i])
        return cloned

    def params_groups(self) -> List[Dict[str, Any]]:
        return [{"params": [p for p in self.parameters() if p.requires_grad]}]

    def init_with_trainer(self, trainer: "ITrainer") -> None:
        pass

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> raw_forward_results_type:
        return self.m(batch[INPUT_KEY], **kwargs)

    def onnx_forward(self, batch: tensor_dict_type, **kwargs: Any) -> Any:
        return self.run(0, batch, **kwargs)

    def summary_forward(self, batch: tensor_dict_type, **kwargs: Any) -> None:
        kwargs.setdefault("use_checkpoint", True)
        self.onnx_forward(batch, **kwargs)

    def postprocess(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        forward_results: raw_forward_results_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        if isinstance(forward_results, dict):
            return forward_results
        if isinstance(forward_results, Tensor):
            return {PREDICTIONS_KEY: forward_results}
        raise ValueError(f"unrecognized forward results occurred: {forward_results}")

    def step(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        forward_kwargs: Optional[Dict[str, Any]] = None,
        *,
        get_losses: bool = False,
        detach_losses: bool = True,
        loss_kwargs: Optional[Dict[str, Any]] = None,
        recover_predictions_fn: Optional[Callable[[td_type], td_type]] = None,
    ) -> StepOutputs:
        loss_tensors = {}
        loss_kwargs = loss_kwargs or {}
        forward_kwargs = forward_kwargs or {}
        get_fw = lambda: self.run(batch_idx, batch, None, **forward_kwargs)
        train_steps = self.train_steps
        if not train_steps:
            return StepOutputs(get_fw(), {})
        fw = None
        for train_step in self.train_steps:
            if train_step.should_skip(self, None):
                continue
            if fw is None or train_step.requires_new_forward:
                fw = get_fw()
                if recover_predictions_fn is not None:
                    fw = recover_predictions_fn(fw)
            if get_losses:
                loss_res = train_step.loss_fn(self, None, batch, fw, **loss_kwargs)
                if not detach_losses:
                    i_losses = loss_res.loss_tensors
                else:
                    i_losses = {k: v.detach() for k, v in loss_res.loss_tensors.items()}
                loss_tensors.update(i_losses)
        if fw is None:
            fw = get_fw()
        return StepOutputs(fw, loss_tensors)

    def train(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        trainer: "ITrainer",
        forward_kwargs: Dict[str, Any],
        loss_kwargs: Dict[str, Any],
    ) -> StepOutputs:
        """
        Runs a series of custom training steps on a batch of data.

        Parameters
        ----------
        batch_idx : int
            The current batch index.
        batch : tensor_dict_type
            The batch of data to use for training.
        trainer : ITrainer
            The trainer object used to train the model.
        forward_kwargs : Dict[str, Any]
            Additional arguments to pass to the forward pass of the model.
        loss_kwargs : Dict[str, Any]
            Additional arguments to pass to the loss function of each training step.

        Returns
        -------
        StepOutputs
            An object containing the outputs of the forward pass and the calculated loss values of the training steps.

        Step by step explanation
        ------------------------
        1. Initialize variables: `forward` (an empty dictionary), `loss_tensors` (an empty dictionary), `any_update`
        (a bool flag set to `False`), and `update_fn` (a function returned by the `get_update_fn` function defined above).
        2. Check whether the forward pass should have gradients (`fw_has_grad`) and which training step to use for the
        forward pass (`fw_train_step`). This is done by looping through each training step and checking its
        `requires_new_forward` and `requires_grad_in_forward` attributes.
        3. If `fw_has_grad` is `False` and a subsequent training step requires gradients in the forward pass, raise a
        RuntimeError with a message indicating which training steps have conflicting requirements.
        4. Loop through each training step and execute the following steps for each:
          1) Check whether the current training step should be skipped. If so, move on to the next training step.
          2) If this is the first training step, or if `requires_new_forward` is `True` for the current training step,
          execute the forward pass of the model and store the output in `forward`. The `no_grad_context` context manager
          is used to prevent gradients from being calculated if `requires_grad_in_forward` is `False`.
          3) Get the optimizer to be used for this training step.
          4) If `enable_toggle_optimizer` is `True` for this training step, temporarily switch to the optimizer associated
          with this training step using the `toggle_optimizer` context manager.
          5) Calculate the loss for this training step using the model, state, batch, and forward pass outputs.
          6) Update the optimizer if `train_step.grad_accumulate` is a factor of the current `state.step`.
          7) Update the `loss_tensors` with the loss values for this training step.
        5. Loop through each callback in the trainer and call its `after_gradient_update` method with the trainer, batch,
        forward pass outputs, loss values, and `any_update` flag.
        6. Loop through each training step and call its callback function with the model, trainer, batch, and forward pass outputs.
        7. Return the `StepOutputs` object containing the forward pass outputs and loss values.
        """

        state = trainer.state
        # sanity check
        fw_has_grad = True
        fw_train_step: Optional[TrainStep] = None
        for train_step in self.train_steps:
            if train_step.should_skip(self, state):
                continue
            if fw_train_step is None or train_step.requires_new_forward:
                fw_has_grad = train_step.requires_grad_in_forward
                fw_train_step = train_step
            if not fw_has_grad and train_step.requires_grad_in_forward:
                fw_name = fw_train_step.__class__.__name__
                current_name = train_step.__class__.__name__
                raise RuntimeError(
                    f"current forward pass comes from '{fw_name}' and has no grad, "
                    f"but '{current_name}' requires grad in forward. You can either set "
                    f"`requires_grad_in_forward` of '{fw_name}' to True, or set "
                    f"`requires_new_forward` of '{current_name}' to True."
                )
        # run train steps
        forward: Optional[tensor_dict_type] = None
        loss_tensors = {}
        update_fn = get_update_fn(trainer)
        any_update = False
        get_fw = lambda: self.run(batch_idx, batch, state, **forward_kwargs)
        for train_step in self.train_steps:
            if train_step.should_skip(self, state):
                continue
            update = (
                state.step
                % (train_step.grad_accumulate or trainer.config.grad_accumulate)
                == 0
            )
            with no_sync_context(update, trainer):
                if forward is None or train_step.requires_new_forward:
                    no_grad = not train_step.requires_grad_in_forward
                    with no_grad_context(enabled=no_grad):
                        forward = get_fw()
                optimizer = trainer.optimizers[train_step.scope]
                should_toggle = train_step.enable_toggle_optimizer
                with toggle_optimizer(self.m, optimizer, enabled=should_toggle):
                    loss_args = self, state, batch, forward
                    loss_res = train_step.loss_fn(*loss_args, **loss_kwargs)
                    update_fn(batch, forward, loss_res, optimizer, update)
                    i_losses = {k: v.detach() for k, v in loss_res.loss_tensors.items()}
                    loss_tensors.update(i_losses)
            if update:
                any_update = True
        if forward is None:
            forward = get_fw()
        for c in trainer.callbacks:
            c.after_gradient_update(trainer, batch, forward, loss_tensors, any_update)
        # train step callbacks
        for train_step in self.train_steps:
            train_step.callback(self, trainer, batch, forward)
        return StepOutputs(forward, loss_tensors)

    def evaluate(
        self,
        config: "TrainerConfig",
        metrics: Optional[IMetric],
        inference: IInference,
        loader: DataLoader,
        *,
        portion: float = 1.0,
        state: Optional["TrainerState"] = None,
        return_outputs: bool = False,
        target_outputs: Union[str, List[str]] = PREDICTIONS_KEY,
        recover_labels: bool = True,
        recover_predictions: bool = True,
        return_labels: bool = False,
        target_labels: Union[str, List[str]] = LABEL_KEY,
        progress: Optional[Progress] = None,
        progress_kwargs: Optional[Dict[str, Any]] = None,
        use_inference_mode: Optional[bool] = None,
        accelerator: Optional[Accelerator] = None,
        pad_dim: Optional[Union[int, Dict[str, int]]] = None,
        only_hold_data_on_rank_0: bool = False,
        verbose: bool = True,
        **kwargs: Any,
    ) -> InferenceOutputs:
        outputs = inference.get_outputs(
            loader,
            portion=portion,
            metrics=metrics,
            use_losses_as_metrics=config.use_losses_as_metrics,  # type: ignore
            return_outputs=return_outputs,
            target_outputs=target_outputs,
            recover_labels=recover_labels,
            recover_predictions=recover_predictions,
            return_labels=return_labels,
            target_labels=target_labels,
            progress=progress,
            progress_kwargs=progress_kwargs,
            use_inference_mode=use_inference_mode,
            accelerator=accelerator,
            pad_dim=pad_dim,
            only_hold_data_on_rank_0=only_hold_data_on_rank_0,
            verbose=verbose,
            **kwargs,
        )
        metric_values = {}
        is_positive = {}
        final_scores = []
        loss_items = outputs.loss_items
        metric_outputs = outputs.metric_outputs
        if loss_items is not None:
            metric_values.update(loss_items)
            is_positive.update({k: False for k in loss_items})
            final_scores.append(weighted_loss_score(config, loss_items))
        if metric_outputs is not None:
            metric_values.update(metric_outputs.metric_values)
            is_positive.update(metric_outputs.is_positive)
            final_score = metric_outputs.final_score
            # `0` often means that user wants to skip this metric
            # but if no other scores are available, we should still use it
            if final_score != 0 or not final_scores:
                final_scores.append(final_score)
        final_score = sum(final_scores) / len(final_scores)
        outputs.metric_outputs = MetricsOutputs(final_score, metric_values, is_positive)
        return outputs

    # api

    @property
    def device(self) -> device:
        return get_device(self.m)

    @property
    def all_module_names(self) -> List[str]:
        names = []
        for m in self.all_modules:
            for k, v in self.__dict__.items():
                if v is m:
                    names.append(k)
                    break
        return names

    def to(self, device: device_type) -> "IModel":
        self.m.to(get_torch_device(device))
        return self

    def state_dict(self, **kwargs: Any) -> tensor_dict_type:
        if is_fsdp():
            return self.m.state_dict(**kwargs)
        return extract_model_from_parallel(self.m).state_dict(**kwargs)

    def parameters(self) -> Iterator[nn.Parameter]:
        return self.m.parameters()

    def named_parameters(self) -> Iterator[Tuple[str, nn.Parameter]]:
        return self.m.named_parameters()

    def load_state_dict(self, d: tensor_dict_type, strict: bool = True) -> None:
        if is_fsdp():
            self.m.load_state_dict(d, strict)
        else:
            extract_model_from_parallel(self.m).load_state_dict(d, strict)

    def run(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        forward_results = self(batch_idx, batch, state, **kwargs)
        outputs = self.postprocess(batch_idx, batch, forward_results, state, **kwargs)
        return outputs

    def eval_context(self, **kwargs: Any) -> ContextManager:
        return eval_context(nn.ModuleList(self.all_modules), **kwargs)

    def to_onnx(
        self,
        export_file: str,
        input_sample: tensor_dict_type,
        dynamic_axes: Optional[Union[List[int], Dict[int, str]]] = None,
        *,
        opset: int = 11,
        simplify: bool = True,
        forward_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        output_names: Optional[List[str]] = None,
        num_samples: Optional[int] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> "IModel":
        # prepare
        device = self.device
        model = self.to("cpu")
        if num_samples is not None:
            input_sample = {k: v[:num_samples] for k, v in input_sample.items()}
        onnx_forward = forward_fn or model.onnx_forward
        input_names = sorted(input_sample.keys())
        if output_names is None:
            with model.eval_context():
                forward_results = onnx_forward(shallow_copy_dict(input_sample))
            if not isinstance(forward_results, dict):
                forward_results = {PREDICTIONS_KEY: forward_results}
            output_names = sorted(forward_results.keys())
        # setup
        kwargs = shallow_copy_dict(kwargs)
        kwargs["input_names"] = input_names
        kwargs["output_names"] = output_names
        kwargs["opset_version"] = opset
        kwargs["export_params"] = True
        kwargs["do_constant_folding"] = True
        if dynamic_axes is None:
            dynamic_axes = {}
        elif isinstance(dynamic_axes, list):
            dynamic_axes = {axis: f"axis.{axis}" for axis in dynamic_axes}
        if num_samples is None:
            dynamic_axes[0] = "batch_size"
        dynamic_axes_settings = {}
        for name in input_names + output_names:
            dynamic_axes_settings[name] = dynamic_axes
        kwargs["dynamic_axes"] = dynamic_axes_settings
        kwargs["verbose"] = verbose
        # export

        class ONNXWrapper(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.m = model.m

            def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
                res = onnx_forward(batch)
                if isinstance(res, Tensor):
                    return {k: res for k in output_names}  # type: ignore
                return {k: res[k] for k in output_names}  # type: ignore

        m_onnx = ONNXWrapper()
        original_states = model.state_dict()
        fixed_states = fix_denormal_states(original_states, verbose=verbose)
        with eval_context(m_onnx):
            model.load_state_dict(fixed_states)
            torch.onnx.export(
                m_onnx,
                ({k: input_sample[k] for k in input_names}, {}),
                export_file,
                **shallow_copy_dict(kwargs),
            )
            model.load_state_dict(original_states)
            if not simplify:
                return self.to(device)
            try:
                from onnxsim import simplify as onnx_simplify

                onnx_model = onnx.load(export_file)
                final_input_names = get_input_names(onnx_model)
                model_simplified, check = onnx_simplify(
                    onnx_model,
                    test_input_shapes={
                        name: tensor.shape
                        for name, tensor in input_sample.items()
                        if name in final_input_names
                    },
                )
            except Exception as err:  # pragma: no cover
                if verbose:
                    console.warn(f"failed to simplify ONNX model ({err})")
                model_simplified = None
                check = False
            if verbose:
                tag = " " if check else " not "
                console.log(f"simplified ONNX model is{tag}validated!")
            if check and model_simplified is not None:
                onnx.save(model_simplified, export_file)
        return self.to(device)

    def save(self, path: TPath, *, do_save: bool = True, **kwargs: Any) -> None:
        full = dict(
            config=self.config.asdict(),
            states=self.state_dict(**kwargs),
        )
        if do_save:
            torch.save(full, path)


# trainer


class TrainerState:
    def __init__(
        self,
        *,
        num_epoch: int,
        batch_size: int,
        loader_length: int,
        num_steps: Optional[int] = None,
        enable_logging: bool = True,
        min_num_sample: int = 3000,
        snapshot_start_step: Optional[int] = None,
        max_snapshot_file: int = 25,
        num_snapshot_per_epoch: float = 2.0,
        num_step_per_snapshot: Optional[int] = None,
        max_step_per_snapshot: int = 1000,
        min_snapshot_epoch_gap: int = 0,
        manual_snapshot_steps: Optional[List[int]] = None,
        manual_snapshot_epochs: Optional[List[int]] = None,
    ):
        self.step = self.epoch = 0
        self.batch_size = batch_size * get_world_size()
        self.num_step_per_epoch = loader_length
        self.num_epoch = num_epoch
        self.num_steps = num_steps
        self.enable_logging = enable_logging
        self.min_num_sample = min_num_sample
        if snapshot_start_step is None:
            snapshot_start_step = math.ceil(min_num_sample / self.batch_size)
        self.snapshot_start_step = snapshot_start_step
        self.max_snapshot_file = max_snapshot_file
        self.num_snapshot_per_epoch = num_snapshot_per_epoch
        if num_step_per_snapshot is None:
            num_step_per_snapshot = int(round(loader_length / num_snapshot_per_epoch))
            num_step_per_snapshot = max(1, num_step_per_snapshot)
            num_step_per_snapshot = min(max_step_per_snapshot, num_step_per_snapshot)
        self.num_step_per_snapshot = num_step_per_snapshot
        self.max_step_per_snapshot = max_step_per_snapshot
        self.min_snapshot_epoch_gap = min_snapshot_epoch_gap
        self.manual_snapshot_steps = set(manual_snapshot_steps or [])
        if manual_snapshot_epochs is not None:
            snapshot_steps = [e * loader_length for e in manual_snapshot_epochs]
            self.manual_snapshot_steps.update(snapshot_steps)
        self._last_step: Optional[int] = None
        self._previous_snapshot_epoch = 0

    def set_terminate(self) -> None:
        self._last_step = self.step
        self.step = self.epoch = -1

    def update_snapshot_epoch(self) -> None:
        self._previous_snapshot_epoch = self.epoch

    @property
    def last_step(self) -> int:
        return self.step if self._last_step is None else self._last_step

    @property
    def is_terminate(self) -> bool:
        return self.epoch == -1

    @property
    def should_train(self) -> bool:
        if self.num_steps is not None:
            return self.step < self.num_steps
        return self.epoch < self.num_epoch

    @property
    def should_terminate(self) -> bool:
        if self.num_steps is None:
            return False
        return self.step == self.num_steps

    @property
    def should_monitor(self) -> bool:
        return (
            self.step % self.num_step_per_snapshot == 0
            or self.step in self.manual_snapshot_steps
        )

    @property
    def should_log_lr(self) -> bool:
        return self.should_log_losses

    @property
    def should_log_losses(self) -> bool:
        if not self.enable_logging:
            return False
        patience = max(4, int(round(self.num_step_per_snapshot / 50.0)))
        denominator = min(self.num_step_per_snapshot, patience)
        return self.step % denominator == 0

    @property
    def should_log_artifacts(self) -> bool:
        return self.should_log_metrics_msg

    @property
    def should_log_metrics_msg(self) -> bool:
        if not self.enable_logging:
            return False
        if self.is_terminate:
            return True
        return self.should_monitor

    @property
    def can_snapshot(self) -> bool:
        if self.is_terminate:
            return True
        return self.epoch - self._previous_snapshot_epoch >= self.min_snapshot_epoch_gap

    @property
    def should_start_snapshot(self) -> bool:
        return self.step >= self.snapshot_start_step

    @property
    def should_extend_epoch(self) -> bool:
        return self.epoch == self.num_epoch

    @property
    def disable_logging(self) -> ContextManager:
        class _:
            def __init__(self, state: TrainerState):
                self.state = state
                self.enabled = state.enable_logging

            def __enter__(self) -> None:
                self.state.enable_logging = False

            def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                self.state.enable_logging = self.enabled

        return _(self)


class TrainerMonitor(WithRegister["TrainerMonitor"], metaclass=ABCMeta):
    """
    This is the base class for monitoring the model, it should decide whether to save the checkpoint, to early stop,
    or to extend the training process. Here are some brief introductions:

    - `should_snapshot` is called when a new score is obtained, and it should return whether to save the checkpoint.
    - `should_terminate` is called when a new score is obtained, and it should return whether to terminate the training.
    - `get_extension` is called when the training is about to be extended, and it should return the number of epochs
    to extend. `0` means no extension.
    - `punish_extension` is called when the training is about to be extended, and it should do some punishment if needed.

    > Examples could be found at `core/learn/monitors.py`.

    """

    d = monitors

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    # abstract

    @abstractmethod
    def should_snapshot(self, new_score: float) -> bool:
        """return `True` if you want to save the checkpoint"""

    @abstractmethod
    def should_terminate(self, new_score: float) -> bool:
        """return `True` if you want to terminate the training process"""

    # optional callbacks

    def get_extension(self, state: TrainerState) -> int:
        """
        Returns
        -------
        int
            The number of epochs to extend. `0` means no extension.

        """

        return 0

    def punish_extension(self) -> None:
        pass


class MonitorResults(NamedTuple):
    terminate: bool
    save_checkpoint: bool
    metric_outputs: Optional[MetricsOutputs]


@dataclass
class OptimizerPack(DataClassBase):
    scope: str
    optimizer_name: str
    scheduler_name: Optional[str] = None
    optimizer_config: Optional[Dict[str, Any]] = None
    scheduler_config: Optional[Dict[str, Any]] = None


class TrainerCallback(WithRegister["TrainerCallback"]):
    """
    This is the base class for various callbacks used in the training process, here are some brief introductions:

    - `initialize` is called before the trainer is constructed.
    - `before_loop` / `before_loop_with_loaders` is called right before the training loop starts.
    - `mutate_forward_kwargs` is used to mutate the `forward_kwargs` of the model.
    > `forward_kwargs` will be used in `IModel.run` method, which will affect both `m.forward` & `self.postprocess`.
    - `mutate_loss_kwargs` is used to mutate the `loss_kwargs` of the model.
    > `loss_kwargs` will be used in each `train_step.loss_fn` method.
    - `log_*` methods will be triggered when the `log` event is triggered, so you can use them to do custom logging.
    > these methods will only be triggered when `is_local_rank_0` is `True`.
    - `after_train_step` is called after each training step.
    - `after_monitor` is called after each monitor step.
    - `after_save_checkpoint` is called once a new checkpoint is saved.
    - `finalize` is called at the end of the `fit` method of the trainer.

    And here's the lifecycle of the callbacks:

    overall:

       `initialize` -> `after_workspace_prepared` -> `before_summary` -> `before_loop` -> `before_loop_with_loaders`
    -> training loop -> `after_loop` -> `finalize`

    * training loop:

       `at_epoch_start` -> `at_step_start`
    ->  train step -> `log_train_step` -> `after_train_step`
    ->  monitor step -> `after_monitor`
    ->  save checkpoint -> `after_save_checkpoint` -> `at_step_end`
    ->  (if terminate) -> `at_terminate` -> `at_epoch_end`

    * train step:

       `mutate_forward_kwargs`
    -> `mutate_loss_kwargs`
    ->  IModel.run
    ->  train_step.loss_fn
    ->  accelerator.backward
    -> `before_gradient_update`
    ->  optimizer.step & zero_grad
    -> `after_gradient_update`

    * monitor step:

        get metrics -> logging (`log_metrics` / `log_metrics_msg` / `log_artifacts`)

    > Examples could be found under `core/learn/callbacks`.

    """

    d = trainer_callbacks

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    @property
    def is_local_rank_0(self) -> bool:
        return is_local_rank_0()

    def initialize(self) -> None:
        pass

    def after_workspace_prepared(self, trainer: "ITrainer") -> None:
        pass

    def before_summary(self, trainer: "ITrainer") -> None:
        pass

    def before_loop(self, trainer: "ITrainer") -> None:
        pass

    def before_loop_with_loaders(
        self,
        trainer: "ITrainer",
        train_loader: DataLoader,
        valid_loader: Optional[DataLoader],
    ) -> None:
        pass

    def mutate_forward_kwargs(self, kw: Dict[str, Any], trainer: "ITrainer") -> None:
        pass

    def mutate_loss_kwargs(self, kw: Dict[str, Any], trainer: "ITrainer") -> None:
        pass

    def log_lr(self, key: str, lr: float, trainer: "ITrainer") -> None:
        pass

    def log_train_step(self, step_outputs: StepOutputs, state: TrainerState) -> None:
        pass

    def before_monitor_logging(self, trainer: "ITrainer") -> None:
        pass

    def log_metrics(self, metric_outputs: MetricsOutputs, state: TrainerState) -> None:
        pass

    def log_metrics_msg(
        self,
        trainer: "ITrainer",
        metrics_outputs: MetricsOutputs,
    ) -> None:
        pass

    def log_artifacts(self, trainer: "ITrainer") -> None:
        pass

    def before_gradient_update(
        self,
        trainer: "ITrainer",
        batch: tensor_dict_type,
        forward: tensor_dict_type,
        loss_res: TrainStepLoss,
        update: bool,
    ) -> None:
        pass

    def after_gradient_update(
        self,
        trainer: "ITrainer",
        batch: tensor_dict_type,
        forward: tensor_dict_type,
        loss_tensors: tensor_dict_type,
        any_update: bool,
    ) -> None:
        pass

    def after_train_step(
        self,
        batch: tensor_dict_type,
        step_outputs: StepOutputs,
        trainer: "ITrainer",
    ) -> None:
        pass

    def after_monitor(self, monitored: MonitorResults, trainer: "ITrainer") -> None:
        pass

    def after_save_checkpoint(self, trainer: "ITrainer") -> None:
        pass

    def at_epoch_start(self, trainer: "ITrainer", train_loader: DataLoader) -> None:
        pass

    def at_step_start(self, batch: tensor_dict_type, trainer: "ITrainer") -> None:
        pass

    def at_step_end(self, trainer: "ITrainer") -> None:
        pass

    def at_epoch_end(self, trainer: "ITrainer") -> None:
        pass

    def at_terminate(self, trainer: "ITrainer") -> None:
        pass

    def after_loop(self, trainer: "ITrainer") -> None:
        pass

    def finalize(self, trainer: "ITrainer") -> None:
        pass


class ITrainerPipeline(IPipeline):
    verbose_context: Callable[[bool], ContextManager]


class ITrainer(ABC):
    config: "TrainerConfig"

    model: IModel
    metrics: Optional[IMetric]
    monitors: List[TrainerMonitor]
    callbacks: List[TrainerCallback]
    optimizers: Dict[str, Optimizer]
    schedulers: Dict[str, Optional[LRScheduler]]
    accelerator: Accelerator
    pipeline: ITrainerPipeline

    state: TrainerState
    inference: IInference

    intermediate: Optional[MetricsOutputs]
    tqdm_settings: "TqdmSettings"
    metrics_log_path: str
    schedulers_requires_metric: Set[str]

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """return the device of the trainer"""

    @property
    @abstractmethod
    def workspace(self) -> TPath:
        """return the workspace of the trainer"""

    @property
    @abstractmethod
    def checkpoint_folder(self) -> str:
        """return the checkpoint folder of the trainer"""

    @abstractmethod
    def fit(
        self,
        data: IData,
        model: IModel,
        metrics: Optional[IMetric],
        inference: IInference,
        optimizers: Dict[str, Optimizer],
        schedulers: Dict[str, Optional[LRScheduler]],
        monitors: List[TrainerMonitor],
        callbacks: List[TrainerCallback],
        schedulers_requires_metric: Set[str],
        *,
        show_summary: bool = True,
        loaded_state: Optional[Dict[str, Any]] = None,
        skip_final_evaluation: bool = False,
        only_touch: bool = False,
        device: device_type = None,
        p: Optional[profile] = None,
    ) -> "ITrainer":
        """the training loop"""

    @abstractmethod
    def save_checkpoint(
        self,
        score: float,
        folder: Optional[TPath] = None,
        *,
        no_history: bool = False,
        check_rank_0: bool = True,
    ) -> None:
        """method to save the checkpoint"""

    @abstractmethod
    def restore_checkpoint(
        self,
        folder: Optional[TPath] = None,
        strict: bool = True,
        state_dict_callback: Optional[Callable[[tensor_dict_type], None]] = None,
    ) -> bool:
        """method to restore the checkpoint"""

    @abstractmethod
    def get_metrics(self, loader: DataLoader, portion: float = 1.0) -> MetricsOutputs:
        """method to get the metrics from the given loader"""


# configs


@pydantic_dataclass
class TqdmSettings(DataClassBase):
    use_tqdm: bool = False
    use_step_tqdm: bool = False
    use_tqdm_in_validation: bool = False
    desc: str = "running epoch"


@pydantic_dataclass
class TrainerConfig:
    workspace: TPath = "_logs"
    create_sub_workspace: bool = True
    state_config: Optional[Dict[str, Any]] = None
    num_epoch: int = 40
    num_steps: Optional[int] = None
    log_steps: Optional[int] = None
    valid_portion: float = 1.0
    clip_norm: float = 0.0
    grad_accumulate: int = 1
    metric_names: Optional[Union[str, List[str]]] = None
    metric_configs: configs_type = None
    metric_weights: Optional[Dict[str, float]] = None
    metric_forward_kwargs: Optional[Dict[str, Any]] = None
    use_losses_as_metrics: Optional[bool] = None
    use_incrementer_for_train_losses_in_eval: bool = True
    recompute_train_losses_in_eval: bool = True
    loss_metrics_weights: Optional[Dict[str, float]] = None
    monitor_names: Optional[Union[str, List[str]]] = None
    monitor_configs: Optional[Dict[str, Any]] = None
    auto_callback: bool = True
    callback_names: Optional[Union[str, List[str]]] = None
    callback_configs: Optional[Dict[str, Any]] = None
    lr: Optional[float] = None
    optimizer_name: Optional[str] = None
    scheduler_name: Optional[str] = None
    optimizer_config: Optional[Dict[str, Any]] = None
    scheduler_config: Optional[Dict[str, Any]] = None
    update_scheduler_per_epoch: bool = False
    optimizer_settings: Optional[Dict[str, Optional[Dict[str, Any]]]] = None
    use_zero: bool = False
    finetune_config: Optional[Dict[str, Any]] = None
    resume_training_from: Optional[str] = None
    tqdm_settings: Optional[Union[Dict[str, Any], TqdmSettings]] = None
    save_pipeline_in_realtime: bool = False
    # profile settings
    profile: bool = False
    profile_config: Optional[Dict[str, Any]] = None
    profile_schedule_config: Optional[Dict[str, Any]] = None
    # `accelerator` attributes
    split_batches: bool = False
    mixed_precision: Union[str, PrecisionType] = "no"
    dispatch_batches: Optional[str] = None
    even_batches: bool = True
    non_blocking: bool = False
    find_unused_parameters: bool = False
    timeout: int = 2400

    def init_process_group(self, *, cpu: bool) -> None:
        timeout = timedelta(seconds=self.timeout)
        init_process_group(cpu=cpu, handler=InitProcessGroupKwargs(timeout=timeout))


@pydantic_dataclass
class DLSettings:
    model: str = "common"
    model_config: Optional[Dict[str, Any]] = None
    module_name: str = ""
    module_config: Optional[Dict[str, Any]] = None
    num_repeat: Optional[int] = None
    loss_name: Optional[str] = None
    loss_config: Optional[Dict[str, Any]] = None
    in_loading: bool = False
    cudnn_benchmark: bool = False


@pydantic_dataclass
class Config(TrainerConfig, DLSettings, ISerializableDataClass["Config"]):  # type: ignore
    def __post_init__(self) -> None:
        if isinstance(self.tqdm_settings, TqdmSettings):
            self.tqdm_settings = self.tqdm_settings.asdict()
        if isinstance(self.mixed_precision, PrecisionType):
            self.mixed_precision = str(self.mixed_precision)

    def to_debug(self) -> "Config":
        self.num_steps = 1
        self.log_steps = 1
        self.valid_portion = 1.0e-4
        return self

    def sanity_check(self) -> None:
        if not self.module_name:
            raise ValueError("`module_name` should be provided")

    def get_external_configs(self, excluded: Set[str]) -> Dict[str, Any]:
        original = self.__class__().asdict()
        external_configs: Dict[str, Any] = {}
        for k, v in self.asdict().items():
            if k in excluded:
                continue
            ov = original[k]
            if v != ov:
                external_configs[k] = v
        return external_configs

    @property
    def is_debug(self) -> bool:
        return self.num_steps == 1

    @property
    def trainer_config(self) -> TrainerConfig:
        return safe_execute(TrainerConfig, self.asdict())  # type: ignore


# registrations


DataConfig.d = data_configs  # type: ignore
DataConfig.register("$base")(DataConfig)
Config.d = configs  # type: ignore
Config.register("$base")(Config)
