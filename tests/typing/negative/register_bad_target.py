from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict
    from typing import Type
    from core.toolkit.misc import WithRegister

    class Plugin(WithRegister["Plugin"]):
        d: Dict[str, Type["Plugin"]] = {}

    class NotAPlugin:
        pass

    Plugin.register("typing.bad")(NotAPlugin)  # expected-mypy: type-var
