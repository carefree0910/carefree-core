from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.toolkit.misc import offload

    offload(1)  # expected-mypy: arg-type, unused-coroutine
    offload(lambda: 1)  # expected-mypy: arg-type, unused-coroutine
