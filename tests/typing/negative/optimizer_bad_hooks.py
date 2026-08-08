from core.learn.schema import ClosurePackStepFn
from core.learn.schema import GetBackwardLossFn
from core.learn.schema import WillSkipBackwardFn


class BadOptimizerHooks:
    def get_backward_loss(self, state: int, loss_res: int, update: bool) -> None:
        return None

    def will_skip_backward(self, state: object, update: bool) -> str:
        return "no"

    def step(self, pack: int) -> None:
        return None


hooks = BadOptimizerHooks()
backward: GetBackwardLossFn = hooks.get_backward_loss  # expected-mypy: assignment
skip: WillSkipBackwardFn = hooks.will_skip_backward  # expected-mypy: assignment
step: ClosurePackStepFn = hooks.step  # expected-mypy: assignment
