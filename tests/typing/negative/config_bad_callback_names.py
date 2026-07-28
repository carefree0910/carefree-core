from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.learn.schema import Config

    config = Config()
    config.callback_names = "typing.callback"  # expected-mypy: assignment
    config.dispatch_batches = "true"  # expected-mypy: assignment
