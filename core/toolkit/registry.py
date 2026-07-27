from contextlib import contextmanager
from typing import overload
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import Generic
from typing import Literal
from typing import Mapping
from typing import TypeVar
from typing import Iterator
from typing import Optional
from typing import Sequence
from typing import Generator
from typing import ContextManager
from typing import MutableMapping

__all__ = ["DuplicatePolicy", "Registry"]
TRegistry = TypeVar("TRegistry")
DuplicatePolicy = Literal["raise", "keep", "replace"]
RegistryConfig = Mapping[str, Any]
RegistryConfigMap = Mapping[str, RegistryConfig]
RegistryNames = Union[str, Sequence[str]]
RegistryConfigs = Union[Sequence[RegistryConfig], RegistryConfigMap]


class Registry(Generic[TRegistry]):
    """A typed, ordered registry of classes and their aliases."""

    def __init__(
        self,
        storage: Optional[MutableMapping[str, Type[TRegistry]]] = None,
        *,
        base_type: Optional[Type[TRegistry]] = None,
        duplicate: DuplicatePolicy = "raise",
    ) -> None:
        if base_type is not None and not isinstance(base_type, type):
            raise TypeError("base_type must be a class")
        self._base_type = base_type
        self._duplicate = self._validate_duplicate(duplicate)
        self._storage: MutableMapping[str, Type[TRegistry]]
        self._storage = {} if storage is None else storage
        self._aliases: Dict[str, str] = {}
        for name, factory in self._storage.items():
            self._validate_name(name)
            self._validate_factory(factory)

    def __len__(self) -> int:
        return len(self._storage)

    def __iter__(self) -> Iterator[str]:
        return iter(self._storage)

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return name in self._storage or name in self._aliases

    @property
    def storage(self) -> MutableMapping[str, Type[TRegistry]]:
        return self._storage

    @property
    def aliases(self) -> Dict[str, str]:
        return dict(self._aliases)

    @staticmethod
    def _validate_name(name: str) -> None:
        if not isinstance(name, str):
            raise TypeError("registry names must be strings")
        if not name:
            raise ValueError("registry names must not be empty")

    @staticmethod
    def _validate_duplicate(duplicate: DuplicatePolicy) -> DuplicatePolicy:
        if duplicate not in ("raise", "keep", "replace"):
            raise ValueError(f"unsupported duplicate policy: {duplicate!r}")
        return duplicate

    def _resolve_duplicate(
        self,
        duplicate: Optional[DuplicatePolicy],
    ) -> DuplicatePolicy:
        if duplicate is None:
            return self._duplicate
        return self._validate_duplicate(duplicate)

    def _validate_factory(self, factory: Type[TRegistry]) -> None:
        if not isinstance(factory, type):
            raise TypeError("registry factories must be classes")
        if self._base_type is not None and not issubclass(factory, self._base_type):
            raise TypeError(f"{factory!r} must be a subclass of {self._base_type!r}")

    def keys(self) -> Iterator[str]:
        return iter(self._storage)

    def values(self) -> Iterator[Type[TRegistry]]:
        return iter(self._storage.values())

    def items(self) -> Iterator[Tuple[str, Type[TRegistry]]]:
        return iter(self._storage.items())

    def has(self, name: str) -> bool:
        return name in self

    def resolve_name(self, name: str) -> str:
        if name in self._storage:
            return name
        try:
            return self._aliases[name]
        except KeyError:
            raise KeyError(name) from None

    def get(self, name: str) -> Type[TRegistry]:
        return self._storage[self.resolve_name(name)]

    def register(
        self,
        name: str,
        factory: Type[TRegistry],
        *,
        duplicate: Optional[DuplicatePolicy] = None,
    ) -> Type[TRegistry]:
        self._validate_name(name)
        self._validate_factory(factory)
        policy = self._resolve_duplicate(duplicate)
        if name in self._storage:
            if policy == "raise":
                raise ValueError(f"registry name {name!r} is already registered")
            if policy == "keep":
                return factory
        elif name in self._aliases:
            if policy == "raise":
                raise ValueError(f"registry name {name!r} is already an alias")
            if policy == "keep":
                return factory
            del self._aliases[name]
        self._storage[name] = factory
        return factory

    def register_alias(
        self,
        alias: str,
        target: str,
        *,
        duplicate: Optional[DuplicatePolicy] = None,
    ) -> str:
        self._validate_name(alias)
        self._validate_name(target)
        policy = self._resolve_duplicate(duplicate)
        canonical = self.resolve_name(target)
        if alias in self._storage:
            if policy == "keep":
                return alias
            raise ValueError(
                f"alias {alias!r} cannot replace a registered canonical name"
            )
        if alias in self._aliases:
            if policy == "raise":
                raise ValueError(f"registry alias {alias!r} is already registered")
            if policy == "keep":
                return self._aliases[alias]
        self._aliases[alias] = canonical
        return canonical

    @staticmethod
    def _copy_config(config: RegistryConfig) -> Dict[str, Any]:
        if not isinstance(config, Mapping):
            raise TypeError("registry configs must be mappings")
        return dict(config)

    def make(
        self,
        name: str,
        config: Optional[RegistryConfig] = None,
    ) -> TRegistry:
        kwargs = {} if config is None else self._copy_config(config)
        return self.get(name)(**kwargs)

    @classmethod
    def _normalize_names(cls, names: RegistryNames) -> Tuple[List[str], bool]:
        if isinstance(names, str):
            cls._validate_name(names)
            return [names], True
        if not isinstance(names, Sequence):
            raise TypeError("registry names must be a string or a sequence")
        normalized = list(names)
        for name in normalized:
            cls._validate_name(name)
        return normalized, False

    @classmethod
    def _normalize_positional_configs(
        cls,
        configs: Sequence[RegistryConfig],
        expected: int,
    ) -> List[Dict[str, Any]]:
        if isinstance(configs, (str, bytes)):
            raise TypeError("registry configs must be mappings")
        if len(configs) != expected:
            raise ValueError(
                f"expected {expected} registry configs, got {len(configs)}"
            )
        return [cls._copy_config(config) for config in configs]

    @classmethod
    def _normalize_configs(
        cls,
        names: List[str],
        scalar_name: bool,
        configs: Optional[Union[RegistryConfig, RegistryConfigs]],
    ) -> List[Dict[str, Any]]:
        if configs is None:
            return [{} for _ in names]
        if scalar_name and isinstance(configs, Mapping):
            return [cls._copy_config(configs)]
        if isinstance(configs, Mapping):
            normalized = {}
            for name, config in configs.items():
                cls._validate_name(name)
                normalized[name] = cls._copy_config(config)
            return [normalized.get(name, {}) for name in names]
        if not isinstance(configs, Sequence):
            raise TypeError(
                "registry configs must be a mapping or a sequence of mappings"
            )
        return cls._normalize_positional_configs(configs, len(names))

    @overload
    def make_many(
        self,
        names: str,
        configs: Optional[Union[RegistryConfig, Sequence[RegistryConfig]]] = None,
    ) -> List[TRegistry]: ...

    @overload
    def make_many(
        self,
        names: Sequence[str],
        configs: Optional[RegistryConfigs] = None,
    ) -> List[TRegistry]: ...

    def make_many(
        self,
        names: RegistryNames,
        configs: Optional[Union[RegistryConfig, RegistryConfigs]] = None,
    ) -> List[TRegistry]:
        normalized_names, scalar_name = self._normalize_names(names)
        normalized_configs = self._normalize_configs(
            normalized_names,
            scalar_name,
            configs,
        )
        factories = [self.get(name) for name in normalized_names]
        return [
            factory(**config) for factory, config in zip(factories, normalized_configs)
        ]

    def reset(self) -> None:
        self._storage.clear()
        self._aliases.clear()

    @contextmanager
    def isolated(
        self,
        *,
        reset: bool = False,
    ) -> Generator["Registry[TRegistry]", None, None]:
        storage = dict(self._storage)
        aliases = dict(self._aliases)
        if reset:
            self.reset()
        try:
            yield self
        finally:
            self._storage.clear()
            self._storage.update(storage)
            self._aliases.clear()
            self._aliases.update(aliases)

    def scope(
        self,
        *,
        reset: bool = False,
    ) -> ContextManager["Registry[TRegistry]"]:
        return self.isolated(reset=reset)
