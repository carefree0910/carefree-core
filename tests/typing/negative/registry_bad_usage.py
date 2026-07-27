from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.toolkit.registry import Registry

    class Plugin:
        pass

    class NotAPlugin:
        pass

    registry = Registry[Plugin](base_type=Plugin)
    registry.register("typing.bad", NotAPlugin)  # expected-mypy: arg-type
    Registry[Plugin](base_type=Plugin, duplicate="invalid")  # expected-mypy: arg-type
    registry.make_many(["typing.bad"], {"value": 1})  # expected-mypy: dict-item
