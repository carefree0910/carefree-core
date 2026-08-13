import torch

import core.learn as cflearn

from accelerate import Accelerator
from unittest.mock import patch
from unittest.mock import Mock


class Step(cflearn.TrainStep):
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
        loss_res = cflearn.TrainStepLoss(loss, {cflearn.LOSS_KEY: loss})
        self.loss_calls.append(loss_res)
        return loss_res

    def callback(self, m, trainer, batch, forward_results):
        if self.events is not None:
            self.events.append("train_step_callback")
        self.callback_calls.append(forward_results)


class Model(cflearn.IModel):
    def __init__(self, provider, events=None):
        self.m = torch.nn.Linear(1, 1)
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


class UpdateCallback(cflearn.TrainerCallback):
    def __init__(self):
        self.calls = []
        self.scopes = []

    def after_gradient_update(
        self,
        trainer,
        batch,
        forward,
        loss_tensors,
        updated_scopes,
    ):
        self.calls.append((trainer, batch, forward, loss_tensors, updated_scopes))
        self.scopes.append(updated_scopes)


def make_trainer(optimizers, *, state=None, callbacks=None, closure=False):
    trainer = cflearn.Trainer(
        cflearn.TrainerConfig(
            grad_accumulate=1,
            use_closure_pack=closure,
        )
    )
    if state is None:
        state = cflearn.TrainerState(
            num_epoch=1,
            batch_size=1,
            loader_length=1,
        )
        state.step = 1
    trainer.state = state
    trainer.optimizers = optimizers
    trainer.callbacks = callbacks or []
    trainer.accelerator = Accelerator(cpu=True)
    return trainer


def test_model_operations_capture_dynamic_steps_once():
    models = []
    batch = {cflearn.INPUT_KEY: torch.ones(1, 1)}
    for operation in ["step", "train"]:
        model = Model(lambda: [Step()])
        if operation == "step":
            model.step(0, batch, get_losses=True)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            model.train(0, batch, make_trainer({"all": optimizer}), {}, {})
        models.append(model)

    assert [model.train_steps_calls for model in models] == [1, 1]
    assert all(len(model.provided_steps) == 1 for model in models)
    assert all(len(model.provided_steps[0].loss_calls) == 1 for model in models)
    assert len(models[1].provided_steps[0].callback_calls) == 1


def test_train_evaluates_stateful_skip_once():
    step = Step(skip_fn=lambda num_calls: num_calls > 1)
    model = Model(lambda: [step])
    state = cflearn.TrainerState(
        num_epoch=1,
        batch_size=1,
        loader_length=1,
    )
    state.step = 1
    model.train(
        0,
        {cflearn.INPUT_KEY: torch.ones(1, 1)},
        make_trainer(
            {"all": torch.optim.SGD(model.parameters(), lr=0.1)},
            state=state,
        ),
        {},
        {},
    )
    assert step.skip_states == [state]
    assert len(step.loss_calls) == 1


def test_model_operations_preserve_repeated_step_positions():
    batch = {cflearn.INPUT_KEY: torch.ones(1, 1)}
    shared_step = Step()
    model = Model(lambda: [shared_step, shared_step])
    model.step(0, batch, get_losses=True)
    assert shared_step.skip_states == [None, None]
    assert len(shared_step.loss_calls) == 2

    shared_step = Step()
    shared_step.requires_new_forward = True
    model = Model(lambda: [shared_step, shared_step])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = make_trainer({"all": optimizer})

    model.train(0, batch, trainer, {}, {})

    assert shared_step.skip_states == [trainer.state, trainer.state]
    assert len(shared_step.loss_calls) == 2
    assert len(shared_step.callback_calls) == 2


def test_all_skipped_train_keeps_callback_contract():
    skipped_step = Step(skip_fn=lambda _: True)
    model = Model(lambda: [skipped_step])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    callback = cflearn.TrainingLoopCallback()
    trainer = make_trainer({"all": optimizer}, callbacks=[callback])
    batch = {cflearn.INPUT_KEY: torch.ones(1, 1)}

    with patch.object(
        model,
        "run",
        wraps=model.run,
    ) as run, patch.object(
        trainer.accelerator,
        "backward",
        wraps=trainer.accelerator.backward,
    ) as backward, patch.object(
        optimizer,
        "step",
        wraps=optimizer.step,
    ) as optimizer_step, patch.object(
        optimizer,
        "zero_grad",
        wraps=optimizer.zero_grad,
    ) as zero_grad, patch.object(
        callback,
        "after_gradient_update",
        wraps=callback.after_gradient_update,
    ) as after_gradient_update:
        outputs = model.train(0, batch, trainer, {}, {})

    run.assert_called_once_with(0, batch, trainer.state)
    backward.assert_not_called()
    optimizer_step.assert_not_called()
    zero_grad.assert_not_called()
    after_gradient_update.assert_called_once()
    callback_args = after_gradient_update.call_args.args
    assert callback_args[0] is trainer
    assert callback_args[1] is batch
    assert callback_args[2] is outputs.forward_results
    assert callback_args[3:] == ({}, set())
    assert skipped_step.skip_states == [trainer.state]
    assert skipped_step.loss_calls == []
    assert len(skipped_step.callback_calls) == 1
    assert skipped_step.callback_calls[0] is outputs.forward_results


def test_accumulating_step_only_runs_backward_and_no_sync():
    train_step = Step(grad_accumulate=2)
    model = Model(lambda: [train_step])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ema = cflearn.EMA(0.9, model.m.named_parameters(), use_num_updates=True)
    model.m.ema = ema
    callback = cflearn.TrainingLoopCallback()
    update_callback = UpdateCallback()
    trainer = make_trainer({"all": optimizer}, callbacks=[callback, update_callback])
    trainer.model = model
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    trainer.schedulers = {"all": scheduler}
    trainer.schedulers_requires_metric = set()
    batch = {cflearn.INPUT_KEY: torch.ones(1, 1)}
    initial_scheduler_epoch = scheduler.last_epoch
    initial_ema_updates = ema.num_updates.item()

    with patch.object(
        trainer.accelerator,
        "backward",
        wraps=trainer.accelerator.backward,
    ) as backward, patch.object(
        trainer.accelerator,
        "no_sync",
        wraps=trainer.accelerator.no_sync,
    ) as no_sync, patch.object(
        optimizer,
        "step",
        wraps=optimizer.step,
    ) as optimizer_step, patch.object(
        optimizer,
        "zero_grad",
        wraps=optimizer.zero_grad,
    ) as zero_grad, patch.object(
        callback,
        "before_gradient_update",
        wraps=callback.before_gradient_update,
    ) as before_gradient_update, patch.object(
        callback,
        "after_gradient_update",
        wraps=callback.after_gradient_update,
    ) as after_gradient_update:
        model.train(0, batch, trainer, {}, {})

    backward.assert_called_once()
    no_sync.assert_called_once_with(model.m)
    optimizer_step.assert_not_called()
    zero_grad.assert_not_called()
    before_gradient_update.assert_called_once()
    after_gradient_update.assert_called_once()
    assert before_gradient_update.call_args.args[-1] is False
    assert after_gradient_update.call_args.args[-1] == set()
    assert scheduler.last_epoch == initial_scheduler_epoch
    assert ema.num_updates.item() == initial_ema_updates
    assert update_callback.scopes == [set()]


def test_closure_optimizer_hooks_and_callback_identity():
    events = []
    step = Step(events=events)
    model = Model(lambda: [step], events)
    packs = []
    hooks = Mock(spec=["will_skip_backward", "get_backward_loss"])
    hooks.will_skip_backward.side_effect = (
        lambda state, update: events.append("will_skip_backward") or False
    )
    hooks.get_backward_loss.side_effect = (
        lambda state, loss_res, update: events.append("get_backward_loss")
        or loss_res.loss
    )
    optimizer = Mock(spec=["optimizer", "step", "zero_grad"])
    optimizer.optimizer = hooks
    optimizer.step.side_effect = lambda pack: (
        events.append("optimizer_step"),
        packs.append(pack),
        pack.loss_fn(),
    )
    optimizer.zero_grad.side_effect = lambda: events.append("zero_grad")
    callback = Mock(spec=cflearn.TrainerCallback)
    callback.before_gradient_update.side_effect = lambda *args: events.append(
        "before_gradient_update"
    )
    callback.after_gradient_update.side_effect = lambda *args: events.append(
        "after_gradient_update"
    )
    trainer = make_trainer({"all": optimizer}, callbacks=[callback], closure=True)
    batch = {cflearn.INPUT_KEY: torch.ones(1, 1)}
    with patch.object(
        trainer.accelerator,
        "backward",
        side_effect=lambda loss: events.append("backward"),
    ):
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
    updated_scopes = callback.after_gradient_update.call_args.args[-1]
    assert updated_scopes == {"all"}
    expected_after = trainer, batch, forward, outputs.loss_tensors, updated_scopes
    assert list(map(id, callback.before_gradient_update.call_args.args)) == list(
        map(id, expected_before)
    )
    assert list(map(id, callback.after_gradient_update.call_args.args)) == list(
        map(id, expected_after)
    )


def test_train_updates_each_optimizer_scope():
    steps = [Step("a"), Step("b")]
    model = Model(lambda: steps)
    optimizers = {
        step.scope: torch.optim.SGD(model.parameters(), lr=0.1) for step in steps
    }
    trainer = make_trainer(optimizers)
    with patch.object(
        trainer.accelerator,
        "backward",
    ), patch.object(
        optimizers["a"],
        "step",
        wraps=optimizers["a"].step,
    ) as step_a, patch.object(
        optimizers["a"],
        "zero_grad",
        wraps=optimizers["a"].zero_grad,
    ) as zero_grad_a, patch.object(
        optimizers["b"],
        "step",
        wraps=optimizers["b"].step,
    ) as step_b, patch.object(
        optimizers["b"],
        "zero_grad",
        wraps=optimizers["b"].zero_grad,
    ) as zero_grad_b:
        model.train(0, {cflearn.INPUT_KEY: torch.ones(1, 1)}, trainer, {}, {})
    for step, zero_grad in [(step_a, zero_grad_a), (step_b, zero_grad_b)]:
        step.assert_called_once_with(None)
        zero_grad.assert_called_once_with()


def test_schedulers_follow_updated_optimizer_scopes():
    steps = [Step("a"), Step("b", grad_accumulate=2)]
    steps[1].requires_new_forward = True
    model = Model(lambda: steps)
    ema = cflearn.EMA(0.9, model.m.named_parameters(), use_num_updates=True)
    model.m.ema = ema
    loop = cflearn.TrainingLoopCallback()
    callback = UpdateCallback()
    optimizers = {
        scope: torch.optim.SGD(model.parameters(), lr=0.1) for scope in ["a", "b"]
    }
    trainer = make_trainer(
        optimizers,
        callbacks=[loop, callback],
    )
    trainer.model = model
    trainer.schedulers = {
        key: torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        for key, optimizer in optimizers.items()
    }
    trainer.schedulers_requires_metric = set()
    initial_epochs = {
        key: scheduler.last_epoch for key, scheduler in trainer.schedulers.items()
    }
    initial_updates = ema.num_updates.item()
    batch = {cflearn.INPUT_KEY: torch.ones(1, 1)}

    outputs = model.train(0, batch, trainer, {}, {})

    assert trainer.schedulers["a"].last_epoch == initial_epochs["a"] + 1
    assert trainer.schedulers["b"].last_epoch == initial_epochs["b"]
    assert ema.num_updates.item() == initial_updates + 1
    assert callback.scopes == [{"a"}]
    expected = (
        trainer,
        batch,
        outputs.forward_results,
        outputs.loss_tensors,
        callback.scopes[0],
    )
    assert list(map(id, callback.calls[0])) == list(map(id, expected))

    trainer.state.step = 2
    outputs = model.train(0, batch, trainer, {}, {})

    assert trainer.schedulers["a"].last_epoch == initial_epochs["a"] + 2
    assert trainer.schedulers["b"].last_epoch == initial_epochs["b"] + 1
    assert ema.num_updates.item() == initial_updates + 2
    assert callback.scopes[-1] == {"a", "b"}
    expected = (
        trainer,
        batch,
        outputs.forward_results,
        outputs.loss_tensors,
        callback.scopes[-1],
    )
    assert list(map(id, callback.calls[-1])) == list(map(id, expected))


def test_repeated_optimizer_scope_updates_scheduler_and_ema_once():
    steps = [Step(), Step()]
    steps[1].requires_new_forward = True
    model = Model(lambda: steps)
    ema = cflearn.EMA(0.9, model.m.named_parameters(), use_num_updates=True)
    model.m.ema = ema
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loop = cflearn.TrainingLoopCallback()
    callback = UpdateCallback()
    trainer = make_trainer(
        {"all": optimizer},
        callbacks=[loop, callback],
    )
    trainer.model = model
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    trainer.schedulers = {"all": scheduler}
    trainer.schedulers_requires_metric = set()
    initial_epoch = scheduler.last_epoch
    initial_updates = ema.num_updates.item()

    with patch.object(
        optimizer,
        "step",
        wraps=optimizer.step,
    ) as optimizer_step:
        model.train(
            0,
            {cflearn.INPUT_KEY: torch.ones(1, 1)},
            trainer,
            {},
            {},
        )

    assert optimizer_step.call_count == 2
    assert scheduler.last_epoch == initial_epoch + 1
    assert ema.num_updates.item() == initial_updates + 1
    assert callback.scopes == [{"all"}]


def test_per_epoch_schedulers_follow_staggered_scopes():
    steps = [Step("a", grad_accumulate=2), Step("b", grad_accumulate=3)]
    steps[1].requires_new_forward = True
    model = Model(lambda: steps)
    loop = cflearn.TrainingLoopCallback()
    callback = UpdateCallback()
    optimizers = {
        scope: torch.optim.SGD(model.parameters(), lr=0.1) for scope in ["a", "b"]
    }
    trainer = make_trainer(
        optimizers,
        callbacks=[loop, callback],
    )
    trainer.model = model
    trainer.config.update_scheduler_per_epoch = True
    trainer.state.epoch = 0
    trainer.schedulers = {
        key: torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        for key, optimizer in optimizers.items()
    }
    trainer.schedulers_requires_metric = set()
    initial_epochs = {
        key: scheduler.last_epoch for key, scheduler in trainer.schedulers.items()
    }
    batch = {cflearn.INPUT_KEY: torch.ones(1, 1)}

    trainer.state.step = 2
    model.train(0, batch, trainer, {}, {})
    assert trainer.schedulers["a"].last_epoch == initial_epochs["a"] + 1
    assert trainer.schedulers["b"].last_epoch == initial_epochs["b"]

    trainer.state.step = 3
    model.train(0, batch, trainer, {}, {})
    assert trainer.schedulers["a"].last_epoch == initial_epochs["a"] + 1
    assert trainer.schedulers["b"].last_epoch == initial_epochs["b"] + 1

    trainer.state.step = 6
    model.train(0, batch, trainer, {}, {})
    assert trainer.schedulers["a"].last_epoch == initial_epochs["a"] + 1
    assert trainer.schedulers["b"].last_epoch == initial_epochs["b"] + 1

    trainer.state.epoch = 1
    model.train(0, batch, trainer, {}, {})
    assert trainer.schedulers["a"].last_epoch == initial_epochs["a"] + 2
    assert trainer.schedulers["b"].last_epoch == initial_epochs["b"] + 2
    assert callback.scopes == [
        {"a"},
        {"b"},
        {"a", "b"},
        {"a", "b"},
    ]
    assert loop.stepped_scopes == {"a", "b"}


def test_training_loop_callback_direct_call_uses_explicit_scopes():
    model = Model(lambda: [Step()])
    ema = cflearn.EMA(0.9, model.m.named_parameters(), use_num_updates=True)
    model.m.ema = ema
    optimizers = {
        scope: torch.optim.SGD(model.parameters(), lr=0.1) for scope in ["a", "b"]
    }
    loop = cflearn.TrainingLoopCallback()
    trainer = make_trainer(optimizers, callbacks=[loop])
    trainer.model = model
    trainer.schedulers = {
        key: torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        for key, optimizer in optimizers.items()
    }
    for optimizer in optimizers.values():
        optimizer.step()
    trainer.schedulers_requires_metric = set()
    initial_epochs = {
        key: scheduler.last_epoch for key, scheduler in trainer.schedulers.items()
    }
    initial_updates = ema.num_updates.item()
    batch = {cflearn.INPUT_KEY: torch.ones(1, 1)}
    forward = model.run(0, batch, trainer.state)

    loop.after_gradient_update(trainer, batch, forward, {}, {"a"})

    assert trainer.schedulers["a"].last_epoch == initial_epochs["a"] + 1
    assert trainer.schedulers["b"].last_epoch == initial_epochs["b"]
    assert ema.num_updates.item() == initial_updates + 1
