import os
import sys
import dill
import json
import math
import time
import random
import shutil
import asyncio
import decimal
import inspect
import hashlib
import operator
import importlib
import unicodedata

import numpy as np

from abc import abstractmethod
from abc import ABC
from abc import ABCMeta
from tqdm import tqdm
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
from typing import Iterable
from typing import Optional
from typing import Protocol
from typing import Coroutine
from typing import ContextManager
from pathlib import Path
from argparse import Namespace
from datetime import datetime
from datetime import timedelta
from functools import reduce
from collections import OrderedDict
from dataclasses import asdict
from dataclasses import fields
from dataclasses import dataclass
from dataclasses import is_dataclass
from dataclasses import Field
from concurrent.futures import ThreadPoolExecutor

from . import console
from .types import TConfig
from .types import arr_type
from .types import np_dict_type
from .constants import TIME_FORMAT


dill._dill._reverse_typemap["ClassType"] = type


# util functions


T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
TDict = TypeVar("TDict", bound="dict")
TRetryResponse = TypeVar("TRetryResponse")
TFutureResponse = TypeVar("TFutureResponse")


class Fn(Protocol[T_co]):
    def __call__(self, *args: Any, **kwargs: Any) -> T_co:
        pass


def walk(
    root: str,
    hierarchy_callback: Callable[[List[str], str], None],
    filter_extensions: Optional[Set[str]] = None,
) -> None:
    walked = list(os.walk(root))
    for folder, _, files in tqdm(walked, desc="folders", position=0, mininterval=1):
        for file in tqdm(files, desc="files", position=1, leave=False, mininterval=1):
            if filter_extensions is not None:
                if not any(file.endswith(ext) for ext in filter_extensions):
                    continue
            hierarchy = folder.split(os.path.sep) + [file]
            hierarchy_callback(hierarchy, os.path.join(folder, file))


def parse_config(config: TConfig) -> Dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, (str, Path)):
        with open(config, "r") as f:
            return json.load(f)
    return shallow_copy_dict(config)


def check_requires(fn: Any, name: str, strict: bool = True) -> bool:
    if isinstance(fn, type):
        fn = fn.__init__  # type: ignore
    signature = inspect.signature(fn)
    for k, param in signature.parameters.items():
        if not strict and param.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if k == name:
            if param.kind is inspect.Parameter.VAR_POSITIONAL:
                return False
            return True
    return False


def get_requirements(fn: Any) -> List[str]:
    remove_first = False
    if isinstance(fn, type):
        fn = fn.__init__  # type: ignore
        remove_first = True  # remove `self`
    requirements = []
    signature = inspect.signature(fn)
    for k, param in signature.parameters.items():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        if param.default is not inspect.Parameter.empty:
            continue
        requirements.append(k)
    if remove_first:
        requirements = requirements[1:]
    return requirements


def filter_kw(
    fn: Callable,
    kwargs: Dict[str, Any],
    *,
    strict: bool = False,
) -> Dict[str, Any]:
    kw = {}
    for k, v in kwargs.items():
        if check_requires(fn, k, strict):
            kw[k] = v
    return kw


def safe_execute(fn: Fn[T], kw: Dict[str, Any], *, strict: bool = False) -> T:
    return fn(**filter_kw(fn, kw, strict=strict))


def safe_instantiate(cls: Type[T], kw: Dict[str, Any], *, strict: bool = False) -> T:
    return cls(**filter_kw(cls, kw, strict=strict))


def get_num_positional_args(fn: Callable) -> Union[int, float]:
    signature = inspect.signature(fn)
    counter = 0
    for param in signature.parameters.values():
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            return math.inf
        if param.kind is inspect.Parameter.POSITIONAL_ONLY:
            counter += 1
        elif param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
            counter += 1
    return counter


def prepare_workspace_from(
    workspace: str,
    *,
    timeout: timedelta = timedelta(30),
    make: bool = True,
) -> str:
    current_time = datetime.now()
    if os.path.isdir(workspace):
        for stuff in os.listdir(workspace):
            if not os.path.isdir(os.path.join(workspace, stuff)):
                continue
            try:
                stuff_time = datetime.strptime(stuff, TIME_FORMAT)
                stuff_delta = current_time - stuff_time
                if stuff_delta > timeout:
                    console.warn(f"{stuff} will be removed (already {stuff_delta} ago)")
                    shutil.rmtree(os.path.join(workspace, stuff))
            except:
                pass
    workspace = os.path.join(workspace, current_time.strftime(TIME_FORMAT))
    if make:
        os.makedirs(workspace)
    return workspace


def get_latest_workspace(root: str) -> Optional[str]:
    if not os.path.isdir(root):
        return None
    all_workspaces = []
    for stuff in os.listdir(root):
        if not os.path.isdir(os.path.join(root, stuff)):
            continue
        try:
            datetime.strptime(stuff, TIME_FORMAT)
            all_workspaces.append(stuff)
        except:
            pass
    if not all_workspaces:
        return None
    return os.path.join(root, sorted(all_workspaces)[-1])


def sort_dict_by_value(d: Dict[Any, Any], *, reverse: bool = False) -> OrderedDict:
    sorted_items = sorted([(v, k) for k, v in d.items()], reverse=reverse)
    return OrderedDict({item[1]: item[0] for item in sorted_items})


def parse_args(args: Any) -> Namespace:
    return Namespace(**{k: None if not v else v for k, v in args.__dict__.items()})


def get_arguments(
    *,
    num_back: int = 0,
    pop_class_attributes: bool = True,
) -> Dict[str, Any]:
    frame = inspect.currentframe()
    if frame is None:
        raise ValueError("`get_arguments` should be called in a frame")
    frame = frame.f_back
    for i in range(num_back):
        if frame is None:
            raise ValueError(f"`get_arguments` failed at {i}th frame backword")
        frame = frame.f_back
    if frame is None:
        raise ValueError(f"`get_arguments` failed at {num_back}th frame backword")
    arguments = inspect.getargvalues(frame)[-1]
    if pop_class_attributes:
        arguments.pop("self", None)
        arguments.pop("__class__", None)
    return arguments


def timestamp(simplify: bool = False, ensure_different: bool = False) -> str:
    """
    Return current timestamp.

    Parameters
    ----------
    simplify : bool. If True, format will be simplified to 'year-month-day'.
    ensure_different : bool. If True, format will include millisecond.

    Returns
    -------
    timestamp : str

    """

    now = datetime.now()
    if simplify:
        return now.strftime("%Y-%m-%d")
    if ensure_different:
        return now.strftime("%Y-%m-%d_%H-%M-%S-%f")
    return now.strftime("%Y-%m-%d_%H-%M-%S")


def prod(iterable: Iterable) -> float:
    """Return cumulative production of an iterable."""

    return float(reduce(operator.mul, iterable, 1))


def hash_code(code: str) -> str:
    """Return hash code for a string."""

    return hashlib.md5(code.encode()).hexdigest()


def hash_dict(d: Dict[str, Any]) -> str:
    """Return a consistent hash code for an arbitrary dict."""

    def _hash(_d: Dict[str, Any]) -> str:
        sorted_keys = sorted(_d)
        hashes = []
        for k in sorted_keys:
            v = _d[k]
            if isinstance(v, dict):
                hashes.append(_hash(v))
            elif isinstance(v, set):
                hashes.append(hash_code(str(sorted(v))))
            else:
                hashes.append(hash_code(str(v)))
        return hash_code("".join(hashes))

    return _hash(d)


def random_hash() -> str:
    return hash_code(str(random.random()))


def prefix_dict(d: TDict, prefix: str) -> TDict:
    """Prefix every key in dict `d` with `prefix`."""

    return {f"{prefix}_{k}": v for k, v in d.items()}  # type: ignore


def shallow_copy_dict(d: TDict) -> TDict:
    def _copy(d_: T) -> T:
        if isinstance(d_, list):
            return [_copy(item) for item in d_]  # type: ignore
        if isinstance(d_, dict):
            return {k: _copy(v) for k, v in d_.items()}  # type: ignore
        return d_

    return _copy(d)


def update_dict(src_dict: dict, tgt_dict: dict) -> dict:
    """
    Update tgt_dict with src_dict.
    * Notice that changes will happen only on keys which src_dict holds.

    Parameters
    ----------
    src_dict : dict
    tgt_dict : dict

    Returns
    -------
    tgt_dict : dict

    """

    for k, v in src_dict.items():
        tgt_v = tgt_dict.get(k)
        if tgt_v is None:
            tgt_dict[k] = v
        elif not isinstance(v, dict):
            tgt_dict[k] = v
        else:
            update_dict(v, tgt_v)
    return tgt_dict


def fix_float_to_length(num: float, length: int) -> str:
    """Change a float number to string format with fixed length."""

    ctx = decimal.Context()
    ctx.prec = 2 * length
    d = ctx.create_decimal(repr(num))
    str_num = format(d, "f").lower()
    if str_num == "nan":
        return f"{str_num:^{length}s}"
    idx = str_num.find(".")
    if idx == -1:
        diff = length - len(str_num)
        if diff <= 0:
            return str_num
        if diff == 1:
            return f"{str_num}."
        return f"{str_num}.{'0' * (diff - 1)}"
    length = max(length, idx)
    return str_num[:length].ljust(length, "0")


def truncate_string_to_length(string: str, length: int) -> str:
    """Truncate a string to make sure its length not exceeding a given length."""

    if len(string) <= length:
        return string
    half_length = int(0.5 * length) - 1
    head = string[:half_length]
    tail = string[-half_length:]
    return f"{head}{'.' * (length - 2 * half_length)}{tail}"


def grouped(iterable: Iterable, n: int, *, keep_tail: bool = False) -> List[tuple]:
    """Group an iterable every `n` elements."""

    if not keep_tail:
        return list(zip(*[iter(iterable)] * n))
    with batch_manager(iterable, batch_size=n, max_batch_size=n) as manager:
        return [tuple(batch) for batch in manager]


def grouped_into(iterable: Iterable, n: int) -> List[tuple]:
    """Group an iterable into `n` groups."""

    elements = list(iterable)
    num_elements = len(elements)
    num_elem_per_group = int(math.ceil(num_elements / n))
    results: List[tuple] = []
    split_idx = num_elements + n - n * num_elem_per_group
    start = 0
    for _ in range(split_idx):
        end = start + num_elem_per_group
        results.append(tuple(elements[start:end]))
        start = end
    for _ in range(split_idx, n):
        end = start + num_elem_per_group - 1
        results.append(tuple(elements[start:end]))
        start = end
    return results


def is_numeric(s: Any) -> bool:
    """Check whether `s` is a number."""

    try:
        s = float(s)
        return True
    except (TypeError, ValueError):
        try:
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            return False


def register_core(
    name: str,
    global_dict: Dict[str, T],
    *,
    allow_duplicate: bool = False,
    before_register: Optional[Callable] = None,
    after_register: Optional[Callable] = None,
) -> Callable[[T], T]:
    def _register(cls: T) -> T:
        if before_register is not None:
            before_register(cls)
        registered = global_dict.get(name)
        if registered is not None and not allow_duplicate:
            console.warn(
                f"'{name}' has already registered "
                f"in the given global dict ({global_dict})"
            )
            return cls
        global_dict[name] = cls
        if after_register is not None:
            after_register(cls)
        return cls

    return _register


def get_err_msg(err: Exception) -> str:
    return " | ".join(map(repr, sys.exc_info()[:2] + (str(err),)))


async def retry(
    fn: Callable[[], Coroutine[None, None, TRetryResponse]],
    num_retry: Optional[int] = None,
    *,
    health_check: Optional[Callable[[TRetryResponse], bool]] = None,
    error_verbose_fn: Optional[Callable[[TRetryResponse], None]] = None,
) -> TRetryResponse:
    counter = 0
    if num_retry is None:
        num_retry = 1
    while counter < num_retry:
        try:
            res = await fn()
            if health_check is None or health_check(res):
                if counter > 0:
                    console.log(f"succeeded after {counter} retries")
                return res
            if error_verbose_fn is not None:
                error_verbose_fn(res)
            else:
                raise ValueError("response did not pass health check")
        except Exception as e:
            console.warn(f"{e}, retrying ({counter + 1})")
        finally:
            counter += 1
    raise ValueError(f"failed after {num_retry} retries")


async def offload(future: Coroutine[Any, Any, TFutureResponse]) -> TFutureResponse:
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        return await loop.run_in_executor(
            executor,
            lambda new_loop, f: new_loop.run_until_complete(f),
            asyncio.new_event_loop(),
            future,
        )


def compress(absolute_folder: str, remove_original: bool = True) -> None:
    shutil.make_archive(absolute_folder, "zip", absolute_folder)
    if remove_original:
        shutil.rmtree(absolute_folder)


# util modules


TRegister = TypeVar("TRegister", bound="WithRegister", covariant=True)
TTRegister = TypeVar("TTRegister", bound=Type["WithRegister"])
TSerializable = TypeVar("TSerializable", bound="ISerializable", covariant=True)
TSArrays = TypeVar("TSArrays", bound="ISerializableArrays", covariant=True)
TSDataClass = TypeVar("TSDataClass", bound="ISerializableDataClass", covariant=True)
TDataClass = TypeVar("TDataClass", bound="DataClassBase")


class DataClassBase:
    """
    To use this base class, you should not only inherit from `DataClassBase`,
    but also decorate your class with `@dataclass`.
    """

    @property
    def fields(self) -> List[Field]:
        return fields(self)  # type: ignore

    @property
    def field_names(self) -> List[str]:
        return [f.name for f in self.fields]

    @property
    def attributes(self) -> List[Any]:
        return [getattr(self, name) for name in self.field_names]

    def asdict(self) -> Dict[str, Any]:
        def _to_item(ins: Any) -> Any:
            if isinstance(ins, DataClassBase):
                return ins.asdict()
            if isinstance(ins, dict):
                return {k: _to_item(v) for k, v in ins.items()}
            if isinstance(ins, list):
                return [_to_item(item) for item in ins]
            if is_dataclass(ins):
                return asdict(ins)
            return ins

        return {k: _to_item(v) for k, v in zip(self.field_names, self.attributes)}

    def copy(self: TDataClass) -> TDataClass:
        return self.__class__(**self.asdict())

    def update_with(self: TDataClass, other: TDataClass) -> TDataClass:
        d = update_dict(other.asdict(), self.asdict())
        return self.__class__(**d)


class WithRegister(Generic[TRegister]):
    d: Dict[str, Type[TRegister]]
    __identifier__: str

    @classmethod
    def get(cls: Type[TRegister], name: str) -> Type[TRegister]:
        return cls.d[name]

    @classmethod
    def has(cls, name: str) -> bool:
        return name in cls.d

    @classmethod
    def make(
        cls: Type[TRegister],
        name: str,
        config: Dict[str, Any],
        *,
        ensure_safe: bool = False,
    ) -> TRegister:
        base = cls.get(name)
        if not ensure_safe:
            return base(**config)  # type: ignore
        return safe_instantiate(base, config)

    @classmethod
    def make_multiple(
        cls: Type[TRegister],
        names: Union[str, List[str]],
        configs: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
        *,
        ensure_safe: bool = False,
    ) -> List[TRegister]:
        if configs is None:
            configs = {}
        if isinstance(names, str):
            assert isinstance(configs, dict)
            return cls.make(names, configs, ensure_safe=ensure_safe)  # type: ignore
        if not isinstance(configs, list):
            configs = [configs.get(name, {}) for name in names]
        return [
            cls.make(name, shallow_copy_dict(config), ensure_safe=ensure_safe)
            for name, config in zip(names, configs)
        ]

    @classmethod
    def register(
        cls,
        name: str,
        *,
        allow_duplicate: bool = False,
    ) -> Callable[[TTRegister], TTRegister]:
        def before(cls_: TTRegister) -> None:
            cls_.__identifier__ = name

        return register_core(  # type: ignore
            name,
            cls.d,
            allow_duplicate=allow_duplicate,
            before_register=before,
        )

    @classmethod
    def check_subclass(cls, name: str) -> bool:
        return issubclass(cls.d[name], cls)


@dataclass
class JsonPack(DataClassBase):
    type: str
    info: Dict[str, Any]


class ISerializable(
    Generic[TSerializable],
    WithRegister[TSerializable],
    metaclass=ABCMeta,
):
    # abstract

    @abstractmethod
    def to_info(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def from_info(self, info: Dict[str, Any]) -> None:
        pass

    # optional callbacks

    def after_load(self) -> None:
        pass

    # api

    def to_pack(self) -> JsonPack:
        return JsonPack(self.__identifier__, self.to_info())

    @classmethod
    def from_pack(cls: Type[TSerializable], pack: Dict[str, Any]) -> TSerializable:
        obj: TSerializable = cls.make(pack["type"], {})
        obj.from_info(pack["info"])
        obj.after_load()
        return obj

    def to_json(self) -> str:
        return json.dumps(self.to_pack().asdict())

    @classmethod
    def from_json(cls: Type[TSerializable], json_string: str) -> TSerializable:
        return cls.from_pack(json.loads(json_string))

    def copy(self: TSerializable) -> TSerializable:
        copied = self.__class__()
        copied.from_info(shallow_copy_dict(self.to_info()))
        return copied


class PureFromInfoMixin:
    def from_info(self, info: Dict[str, Any]) -> None:
        for k, v in info.items():
            setattr(self, k, v)


class ISerializableArrays(
    Generic[TSArrays],
    ISerializable[TSArrays],
    metaclass=ABCMeta,
):
    @abstractmethod
    def to_npd(self) -> np_dict_type:
        pass

    @abstractmethod
    def from_npd(self, npd: np_dict_type) -> TSArrays:
        pass

    def copy(self: TSArrays) -> TSArrays:
        copied: TSArrays = super().copy()
        copied.from_npd(shallow_copy_dict(self.to_npd()))
        return copied


class ISerializableDataClass(
    Generic[TSDataClass],
    PureFromInfoMixin,
    ISerializable[TSDataClass],
    DataClassBase,
):
    def to_info(self) -> Dict[str, Any]:
        return self.asdict()


class Serializer:
    id_file: str = "id.txt"
    info_file: str = "info.json"
    npd_folder: str = "npd"

    @classmethod
    def save_info(
        cls,
        folder: str,
        *,
        info: Optional[Dict[str, Any]] = None,
        serializable: Optional[ISerializable] = None,
    ) -> None:
        os.makedirs(folder, exist_ok=True)
        if info is None and serializable is None:
            raise ValueError("either `info` or `serializable` should be provided")
        if info is None:
            info = serializable.to_info()  # type: ignore
        with open(os.path.join(folder, cls.info_file), "w") as f:
            json.dump(info, f)

    @classmethod
    def load_info(cls, folder: str) -> Dict[str, Any]:
        return cls.try_load_info(folder, strict=True)  # type: ignore

    @classmethod
    def try_load_info(
        cls,
        folder: str,
        *,
        strict: bool = False,
    ) -> Optional[Dict[str, Any]]:
        info_path = os.path.join(folder, cls.info_file)
        if not os.path.isfile(info_path):
            if not strict:
                return None
            raise ValueError(f"'{info_path}' does not exist")
        with open(info_path, "r") as f:
            info = json.load(f)
        return info

    @classmethod
    def save_npd(
        cls,
        folder: str,
        *,
        npd: Optional[np_dict_type] = None,
        serializable: Optional[ISerializableArrays] = None,
    ) -> None:
        os.makedirs(folder, exist_ok=True)
        if npd is None and serializable is None:
            raise ValueError("either `npd` or `serializable` should be provided")
        if npd is None:
            npd = serializable.to_npd()  # type: ignore
        npd_folder = os.path.join(folder, cls.npd_folder)
        os.makedirs(npd_folder, exist_ok=True)
        for k, v in npd.items():
            np.save(os.path.join(npd_folder, f"{k}.npy"), v)

    @classmethod
    def load_npd(cls, folder: str) -> np_dict_type:
        os.makedirs(folder, exist_ok=True)
        npd_folder = os.path.join(folder, cls.npd_folder)
        if not os.path.isdir(npd_folder):
            raise ValueError(f"'{npd_folder}' does not exist")
        npd = {}
        for file in os.listdir(npd_folder):
            key = os.path.splitext(file)[0]
            npd[key] = np.load(os.path.join(npd_folder, file))
        return npd

    @classmethod
    def save(
        cls,
        folder: str,
        serializable: ISerializable,
        *,
        save_npd: bool = True,
    ) -> None:
        cls.save_info(folder, serializable=serializable)
        if save_npd and isinstance(serializable, ISerializableArrays):
            cls.save_npd(folder, serializable=serializable)
        with open(os.path.join(folder, cls.id_file), "w") as f:
            f.write(serializable.__identifier__)

    @classmethod
    def load(
        cls,
        folder: str,
        base: Type[TSerializable],
        *,
        swap_id: Optional[str] = None,
        swap_info: Optional[Dict[str, Any]] = None,
        load_npd: bool = True,
    ) -> TSerializable:
        serializable = cls.load_empty(folder, base, swap_id=swap_id)
        serializable.from_info(swap_info or cls.load_info(folder))
        if load_npd and isinstance(serializable, ISerializableArrays):
            serializable.from_npd(cls.load_npd(folder))
        serializable.after_load()
        return serializable

    @classmethod
    def load_empty(
        cls,
        folder: str,
        base: Type[TSerializable],
        *,
        swap_id: Optional[str] = None,
    ) -> TSerializable:
        if swap_id is not None:
            s_type = swap_id
        else:
            id_path = os.path.join(folder, cls.id_file)
            if not os.path.isfile(id_path):
                raise ValueError(f"cannot find '{id_path}'")
            with open(id_path, "r") as f:
                s_type = f.read().strip()
        return base.make(s_type, {})


class Incrementer:
    """
    Util class which can calculate running mean & running std efficiently.

    Parameters
    ----------
    window_size : {int, None}, window size of running statistics.
    * If None, then all history records will be used for calculation.

    Examples
    ----------
    >>> incrementer = Incrementer(window_size=5)
    >>> for i in range(10):
    >>>     incrementer.update(i)
    >>>     if i >= 4:
    >>>         print(incrementer.mean)  # will print 2.0, 3.0, ..., 6.0, 7.0

    """

    def __init__(self, window_size: Optional[int] = None):
        if window_size is not None:
            if not isinstance(window_size, int):
                msg = f"window size should be integer, {type(window_size)} found"
                raise ValueError(msg)
            if window_size < 2:
                msg = f"window size should be at least 2, {window_size} found"
                raise ValueError(msg)
        self.previous: List[float] = []
        self.num_record = 0.0
        self.window_size = window_size
        self.running_sum = self.running_square_sum = 0.0

    @property
    def mean(self) -> float:
        return self.running_sum / self.num_record

    @property
    def std(self) -> float:
        return math.sqrt(
            max(
                0.0,
                self.running_square_sum / self.num_record - self.mean**2,
            )
        )

    def update(self, new_value: float) -> None:
        self.num_record += 1
        self.running_sum += new_value
        self.running_square_sum += new_value**2
        if self.window_size is not None:
            self.previous.append(new_value)
            if self.num_record == self.window_size + 1:
                self.num_record -= 1
                previous = self.previous.pop(0)
                self.running_sum -= previous
                self.running_square_sum -= previous**2


class OPTBase(ABC):
    def __init__(self) -> None:
        self._opt = self.defaults
        self.update_from_env()

    def __getattr__(self, __name: str) -> Any:
        return self._opt[__name]

    # abstract

    @property
    @abstractmethod
    def env_key(self) -> str:
        pass

    @property
    @abstractmethod
    def defaults(self) -> Dict[str, Any]:
        pass

    # optional callbacks

    def update_from_env(self) -> None:
        env_opt_json = os.environ.get(self.env_key)
        if env_opt_json is not None:
            update_dict(json.loads(env_opt_json), self._opt)

    # api

    def opt_context(self, increment: Dict[str, Any]) -> ContextManager:
        class _:
            def __init__(self) -> None:
                self._increment = increment
                self._backup = shallow_copy_dict(instance._opt)

            def __enter__(self) -> None:
                update_dict(self._increment, instance._opt)

            def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                instance._opt = self._backup

        instance = self
        return _()

    def opt_env_context(self, increment: Dict[str, Any]) -> ContextManager:
        class _:
            def __init__(self) -> None:
                self._increment = increment
                self._backup = os.environ.get(instance.env_key)

            def __enter__(self) -> None:
                os.environ[instance.env_key] = json.dumps(self._increment)

            def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                if self._backup is None:
                    del os.environ[instance.env_key]
                else:
                    os.environ[instance.env_key] = self._backup

        instance = self
        return _()


# contexts


class timeit:
    """
    Timing context manager.

    Examples
    --------
    >>> with timeit("something"):
    >>>     # do something here
    >>> # will print ">  [ info ] timing for    something     : x.xxxx"

    """

    t: float

    def __init__(self, message: str, *, precision: int = 6):
        self.p = precision
        self.message = message

    def __enter__(self) -> None:
        self.t = time.time()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        console.log(
            f"timing for {self.message:^16s} : "
            f"{time.time() - self.t:{self.p}.{self.p-2}f}"
        )


class batch_manager:
    """
    Process data in batch.

    Parameters
    ----------
    inputs : tuple(np.ndarray), auxiliary array inputs.
    num_elem : {int, float}, indicates how many elements will be processed in a batch.
    > `element` here means every single entry of the `inputs`.
    batch_size : int, indicates the batch_size; if None, batch_size will be
                      calculated by `num_elem`.

    Examples
    --------
    >>> with batch_manager(np.arange(5), np.arange(1, 6), batch_size=2) as manager:
    >>>     for arr, tensor in manager:
    >>>         print(arr, tensor)
    >>>         # Will print:
    >>>         #   [0 1], [1 2]
    >>>         #   [2 3], [3 4]
    >>>         #   [4]  , [5]

    """

    start: int
    end: int

    def __init__(
        self,
        *inputs: arr_type,
        num_elem: Union[int, float] = 1e6,
        batch_size: Optional[int] = None,
        max_batch_size: int = 1024,
    ):
        if not inputs:
            raise ValueError("inputs should be provided in general_batch_manager")
        input_lengths = list(map(len, inputs))
        self.num_samples, self.inputs = input_lengths[0], inputs
        assert_msg = "inputs should be of same length"
        assert all(length == self.num_samples for length in input_lengths), assert_msg
        if batch_size is not None:
            self.batch_size = batch_size
        else:
            self.batch_size = int(
                int(num_elem) / sum(map(lambda arr: prod(arr.shape[1:]), inputs))
            )
        self.batch_size = min(max_batch_size, min(self.num_samples, self.batch_size))
        self.num_epoch = int(self.num_samples / self.batch_size)
        self.num_epoch += int(self.num_epoch * self.batch_size < self.num_samples)

    def __enter__(self) -> "batch_manager":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def __iter__(self) -> "batch_manager":
        self.start, self.end = 0, self.batch_size
        return self

    def __next__(self) -> Union[Tuple[arr_type, ...], arr_type]:
        if self.start >= self.num_samples:
            raise StopIteration
        batched_data = tuple(
            map(
                lambda arr: arr[self.start : self.end],
                self.inputs,
            )
        )
        self.start, self.end = self.end, self.end + self.batch_size
        if len(batched_data) == 1:
            return batched_data[0]
        return batched_data

    def __len__(self) -> int:
        return self.num_epoch
