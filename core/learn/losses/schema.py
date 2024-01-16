from typing import Any
from typing import Dict
from typing import Type
from typing import Union
from typing import TypeVar
from typing import Callable
from typing import Optional

from ..schema import ILoss
from ..modules.common import PrefixModules


TLoss = TypeVar("TLoss", bound=Type[ILoss])

losses = PrefixModules("loss")


def register_loss(name: str, **kwargs: Any) -> Callable[[TLoss], TLoss]:
    def before_register(cls: TLoss) -> TLoss:
        cls.__identifier__ = name
        return cls

    kwargs.setdefault("before_register", before_register)
    return losses.register(name, **kwargs)


def build_loss(
    name: str,
    *,
    config: Optional[Union[str, Dict[str, Any]]] = None,
    **kwargs: Any,
) -> ILoss:
    return losses.build(name, config=config, **kwargs)


__all__ = [
    "losses",
    "register_loss",
    "build_loss",
]
