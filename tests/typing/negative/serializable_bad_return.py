from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from typing import Dict
    from typing import Type
    from core.toolkit.misc import ISerializable

    class BadSerializable(ISerializable["BadSerializable"]):
        d: Dict[str, Type["BadSerializable"]] = {}

        def to_info(self) -> Dict[str, Any]:
            return {}

        def from_info(  # expected-mypy: override
            self,
            info: Dict[str, Any],
        ) -> int:
            return 0
