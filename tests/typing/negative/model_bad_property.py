from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn

    from typing import List
    from core.learn.schema import Config
    from core.learn.schema import IModel

    class BadModel(IModel):
        def __init__(self) -> None:
            self.m = nn.Identity()

        @property
        def train_steps(self) -> List[int]:  # expected-mypy: override
            return []

        @property
        def all_modules(self) -> List[nn.Module]:
            return [self.m]

        def build(self, config: Config) -> None:
            self.config = config
