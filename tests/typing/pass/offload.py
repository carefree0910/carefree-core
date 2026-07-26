from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from typing import Coroutine
    from typing_extensions import assert_type
    from core.toolkit.misc import offload

    async def check_offload_types() -> None:
        async def get_number() -> int:
            return 1

        coroutine: Coroutine[Any, Any, int] = get_number()
        assert_type(await offload(coroutine), int)
        assert_type(await offload(future=get_number()), int)
