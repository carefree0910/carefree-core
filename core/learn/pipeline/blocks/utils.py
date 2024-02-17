import os

from abc import abstractmethod
from abc import ABCMeta
from typing import OrderedDict as OrderedDictType
from collections import OrderedDict

from ..common import Block
from ...schema import Config
from ....toolkit.misc import to_path
from ....toolkit.types import TPath


class InjectDefaultsMixin:
    _defaults: OrderedDictType

    def __init__(self) -> None:
        self._defaults = OrderedDict()

    def process_defaults(self, _defaults: OrderedDictType) -> None:
        for k, v in self._defaults.items():
            _defaults[k] = v


class TryLoadBlock(Block, metaclass=ABCMeta):
    # abstract

    @abstractmethod
    def try_load(self, folder: TPath) -> bool:
        pass

    @abstractmethod
    def from_scratch(self, config: Config) -> None:
        pass

    @abstractmethod
    def dump_to(self, folder: TPath) -> None:
        pass

    # inheritance

    def build(self, config: Config) -> None:
        if self.serialize_folder is not None:
            serialize_folder = self.serialize_folder / self.__identifier__
            if self.try_load(serialize_folder):
                return
        self.from_scratch(config)

    def save_extra(self, folder: TPath) -> None:
        folder = to_path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.dump_to(folder)


__all__ = [
    "InjectDefaultsMixin",
    "TryLoadBlock",
]
