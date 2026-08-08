import torch
import pytest

import core.learn as cflearn

from unittest.mock import patch
from contextlib import nullcontext
from unittest.mock import MagicMock
from types import SimpleNamespace


class _Step(cflearn.TrainStep):
    def __init__(
        self,
        scope="all",
        *,
        grad_accumulate=None,
        skip_fn=None,
        events=None,
    ):
        super().__init__(
            scope,
            grad_accumulate=grad_accumulate,
            enable_toggle_optimizer=False,
        )
        self.skip_fn = skip_fn
        self.events = events
        self.skip_states = []
        self.loss_calls = []
        self.callback_calls = []

    def should_skip(self, m, state):
        self.skip_states.append(state)
        return False if self.skip_fn is None else self.skip_fn(len(self.skip_states))

    def loss_fn(self, m, state, batch, forward_results, **kwargs):
        if self.events is not None:
            self.events.append("loss")
        loss = forward_results[cflearn.PREDICTIONS_KEY].sum()
        loss_res = cflearn.TrainStepLoss(loss, {cflearn.LOSS_KEY: loss[None]})
        self.loss_calls.append(loss_res)
        return loss_res

    def callback(self, m, trainer, batch, forward_results):
        if self.events is not None:
            self.events.append("train_step_callback")
        self.callback_calls.append(forward_results)


class _Model(cflearn.IModel):
    def __init__(self, provider, events=None):
        self.m = torch.nn.Identity()
        self.provider = provider
        self.events = events
        self.train_steps_calls = 0
        self.provided_steps = []

    @property
    def train_steps(self):
        self.train_steps_calls += 1
        train_steps = self.provider()
        self.provided_steps.extend(train_steps)
        return train_steps

    @property
    def all_modules(self):
        return [self.m]

    def build(self, config):
        pass

    def run(self, batch_idx, batch, state=None, **kwargs):
        if self.events is not None:
            self.events.append("forward")
        return {cflearn.PREDICTIONS_KEY: self.m(batch[cflearn.INPUT_KEY])}


def _trainer(optimizers, *, state=None, callbacks=None, closure=False):
    accelerator = SimpleNamespace(
        backward=lambda loss: None,
        no_sync=lambda module: nullcontext(),
    )
    return SimpleNamespace(
        state=state or SimpleNamespace(step=1),
        config=cflearn.TrainerConfig(
            grad_accumulate=1,
            use_closure_pack=closure,
        ),
        optimizers=optimizers,
        callbacks=callbacks or [],
        accelerator=accelerator,
    )


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="P0-06: capture dynamic train steps once",
)
def test_model_operations_capture_dynamic_steps_once():
    models = []
    batch = {cflearn.INPUT_KEY: torch.ones(1, 1)}
    for operation in ["step", "train"]:
        model = _Model(lambda: [_Step()])
        if operation == "step":
            model.step(0, batch, get_losses=True)
        else:
            with patch("core.learn.schema.get_update_fn", return_value=MagicMock()):
                model.train(0, batch, _trainer({"all": SimpleNamespace()}), {}, {})
        models.append(model)

    assert [model.train_steps_calls for model in models] == [1, 1]
    assert all(len(model.provided_steps) == 1 for model in models)
    assert all(len(model.provided_steps[0].loss_calls) == 1 for model in models)
    assert len(models[1].provided_steps[0].callback_calls) == 1


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="P0-06: evaluate should_skip once",
)
def test_train_evaluates_stateful_skip_once():
    step = _Step(skip_fn=lambda num_calls: num_calls > 1)
    model = _Model(lambda: [step])
    state = SimpleNamespace(step=1)
    with patch("core.learn.schema.get_update_fn", return_value=MagicMock()):
        model.train(
            0,
            {cflearn.INPUT_KEY: torch.ones(1, 1)},
            _trainer({"all": SimpleNamespace()}, state=state),
            {},
            {},
        )
    assert step.skip_states == [state]
    assert len(step.loss_calls) == 1


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="P0-06: bind each deferred closure",
)
def test_train_binds_deferred_closures_to_steps():
    steps = [_Step("first"), _Step("second")]
    packs = []
    optimizer = lambda: SimpleNamespace(
        step=lambda pack: packs.append(pack),
        zero_grad=lambda: None,
    )
    trainer = _trainer(
        {"first": optimizer(), "second": optimizer()},
        closure=True,
    )
    _Model(lambda: steps).train(
        0,
        {cflearn.INPUT_KEY: torch.ones(1, 1)},
        trainer,
        {},
        {},
    )
    for step in steps:
        step.loss_calls.clear()
    for pack in packs:
        pack.loss_fn()
    assert [len(step.loss_calls) for step in steps] == [1, 1]


def test_closure_optimizer_hooks_and_callback_identity():
    events = []
    step = _Step(events=events)
    model = _Model(lambda: [step], events)
    packs = []
    hooks = SimpleNamespace(
        will_skip_backward=MagicMock(
            side_effect=lambda state, update: events.append("will_skip_backward")
            or False
        ),
        get_backward_loss=MagicMock(
            side_effect=lambda state, loss_res, update: events.append(
                "get_backward_loss"
            )
            or loss_res.loss
        ),
    )

    def optimizer_step(pack):
        events.append("optimizer_step")
        packs.append(pack)
        pack.loss_fn()

    optimizer = SimpleNamespace(
        optimizer=hooks,
        step=optimizer_step,
        zero_grad=lambda: events.append("zero_grad"),
    )
    before = MagicMock(
        side_effect=lambda *args: events.append("before_gradient_update")
    )
    after = MagicMock(side_effect=lambda *args: events.append("after_gradient_update"))
    callback = SimpleNamespace(
        before_gradient_update=before,
        after_gradient_update=after,
    )
    trainer = _trainer({"all": optimizer}, callbacks=[callback], closure=True)
    trainer.accelerator.backward = lambda loss: events.append("backward")
    batch = {cflearn.INPUT_KEY: torch.ones(1, 1)}
    outputs = model.train(0, batch, trainer, {}, {})

    assert events == [
        "will_skip_backward",
        "forward",
        "loss",
        "get_backward_loss",
        "backward",
        "before_gradient_update",
        "optimizer_step",
        "forward",
        "loss",
        "zero_grad",
        "after_gradient_update",
        "train_step_callback",
    ]
    pack = packs[0]
    assert pack._fields == (
        "state",
        "loss_fn",
        "batch",
        "forward",
        "loss_res",
        "accelerator",
    )
    assert pack.state is trainer.state
    assert pack.batch is batch
    assert pack.forward is outputs.forward_results
    assert pack.loss_res is step.loss_calls[0]
    assert pack.accelerator is trainer.accelerator
    assert step.callback_calls[0] is outputs.forward_results
    forward = outputs.forward_results
    expected_before = trainer, batch, forward, pack.loss_res, True
    expected_after = trainer, batch, forward, outputs.loss_tensors, True
    assert list(map(id, before.call_args.args)) == list(map(id, expected_before))
    assert list(map(id, after.call_args.args)) == list(map(id, expected_after))


def test_train_updates_each_optimizer_scope():
    steps = [_Step("a"), _Step("b")]
    optimizers = {
        step.scope: SimpleNamespace(step=MagicMock(), zero_grad=MagicMock())
        for step in steps
    }
    _Model(lambda: steps).train(
        0,
        {cflearn.INPUT_KEY: torch.ones(1, 1)},
        _trainer(optimizers),
        {},
        {},
    )
    for optimizer in optimizers.values():
        optimizer.step.assert_called_once_with(None)
        optimizer.zero_grad.assert_called_once_with()


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="P0-06: advance only updated schedulers",
)
def test_schedulers_follow_updated_optimizer_scopes():
    steps = [_Step("a"), _Step("b", grad_accumulate=2)]
    model = _Model(lambda: steps)
    ema = cflearn.EMA(0.9, [])
    ema.forward = MagicMock()
    model.m.ema = ema
    loop = cflearn.TrainingLoopCallback()
    loop.get_scheduler_settings = MagicMock(return_value=(False, {}))
    optimizer = lambda: SimpleNamespace(
        step=lambda pack: None,
        zero_grad=lambda: None,
    )
    trainer = _trainer(
        {"a": optimizer(), "b": optimizer()},
        callbacks=[loop],
    )
    trainer.model = model
    trainer.schedulers = {"a": MagicMock(), "b": MagicMock()}
    model.train(0, {cflearn.INPUT_KEY: torch.ones(1, 1)}, trainer, {}, {})

    trainer.schedulers["a"].step.assert_called_once_with()
    trainer.schedulers["b"].step.assert_not_called()
    ema.forward.assert_called_once_with()
