from typing import Any
from typing import Type
from typing import TypeVar
from typing import Callable

from ..schema import ILoss
from ..modules.common import PrefixModules
from ...toolkit.types import TConfig


TLoss = TypeVar("TLoss", bound=Type[ILoss])

losses = PrefixModules("loss")


def register_loss(name: str, **kwargs: Any) -> Callable[[TLoss], TLoss]:
    def before_register(cls: TLoss) -> TLoss:
        cls.__identifier__ = name
        return cls

    kwargs.setdefault("before_register", before_register)
    return losses.register(name, **kwargs)


def build_loss(name: str, *, config: TConfig = None, **kwargs: Any) -> ILoss:
    return losses.build(name, config=config, **kwargs)  # type: ignore


__all__ = [
    "losses",
    "register_loss",
    "build_loss",
]
