import time

from typing import Any
from typing import Dict


def async_dataloader(params: Dict[str, Any], seed: int) -> Dict[str, Any]:
    import torch

    import numpy as np
    import core.learn as cflearn

    from core.learn.schema import AsyncIterManager

    num_samples = int(params.get("num_samples", params.get("samples", 64)))
    input_dim = int(params.get("input_dim", params.get("dim", 8)))
    output_dim = int(params.get("output_dim", 2))
    batch_size = int(params.get("batch_size", 16))
    prefetch_factor = int(params.get("prefetch_factor", 2))
    iterations = int(params.get("iterations", 1))
    assert num_samples > 0
    assert input_dim > 0
    assert output_dim > 0
    assert batch_size > 0
    assert prefetch_factor > 0
    assert iterations > 0

    np.random.seed(seed)
    torch.manual_seed(seed)
    x = np.arange(num_samples * input_dim, dtype=np.float32).reshape(
        num_samples,
        input_dim,
    )
    y = np.arange(num_samples * output_dim, dtype=np.float32).reshape(
        num_samples,
        output_dim,
    )
    config = cflearn.DataConfig(
        batch_size=batch_size,
        loader_seed=seed,
        async_prefetch=True,
        async_prefetch_factor=prefetch_factor,
        async_prefetch_factor_for_validation=prefetch_factor,
    )
    data = cflearn.ArrayData.init(config).fit(x, y)
    loader = data.build_loader(
        x,
        y,
        shuffle=False,
        batch_size=batch_size,
        for_inference=False,
    )

    final_batches = []
    total_batches = 0
    try:
        started = time.perf_counter_ns()
        for _ in range(iterations):
            final_batches = list(loader)
            total_batches += len(final_batches)
        finished = time.perf_counter_ns()
    finally:
        AsyncIterManager.cleanup(id(loader))

    full_x = torch.cat([batch[cflearn.INPUT_KEY] for batch in final_batches])
    full_y = torch.cat([batch[cflearn.LABEL_KEY] for batch in final_batches])
    expected_x = torch.from_numpy(x)
    expected_y = torch.from_numpy(y)
    expected_batches = (num_samples + batch_size - 1) // batch_size
    order_preserved = torch.equal(full_x, expected_x)
    labels_preserved = torch.equal(full_y, expected_y)
    assert len(final_batches) == expected_batches
    assert total_batches == iterations * expected_batches
    assert order_preserved
    assert labels_preserved

    behavior = {
        "batch_sizes": [
            int(batch[cflearn.INPUT_KEY].shape[0]) for batch in final_batches
        ],
        "input_checksum": float(full_x.double().sum().item()),
        "input_shape": list(full_x.shape),
        "label_checksum": float(full_y.double().sum().item()),
        "label_shape": list(full_y.shape),
        "labels_preserved": labels_preserved,
        "num_batches": len(final_batches),
        "order_preserved": order_preserved,
    }
    return {
        "elapsed_ns": finished - started,
        "iterations": iterations,
        "behavior": behavior,
        "uses_torch": True,
    }


def inference_padding(params: Dict[str, Any], seed: int) -> Dict[str, Any]:
    import torch

    import numpy as np
    import torch.nn as nn
    import core.learn as cflearn

    from core.learn.schema import AsyncIterManager

    num_samples = int(params.get("num_samples", params.get("samples", 4)))
    rows = int(params.get("rows", 3))
    max_width = int(params.get("max_width", params.get("width", 3)))
    batch_size = int(params.get("batch_size", 1))
    iterations = int(params.get("iterations", 1))
    assert num_samples > 0
    assert rows > 0
    assert max_width > 0
    assert batch_size == 1
    assert iterations > 0

    np.random.seed(seed)
    torch.manual_seed(seed)
    module_name = f"$benchmark_padding_{seed}"

    @cflearn.register_module(module_name, allow_duplicate=True)
    class PaddingIdentity(nn.Module):
        def forward(self, x: Any) -> Any:
            return torch.from_numpy(x[0])

    x = np.empty(num_samples, dtype=object)
    expected = np.zeros((num_samples * rows, max_width), dtype=np.int64)
    cursor = 0
    for i in range(num_samples):
        width = i % max_width + 1
        size = rows * width
        sample = np.arange(cursor, cursor + size, dtype=np.int64).reshape(rows, width)
        x[i] = sample
        expected[i * rows : (i + 1) * rows, :width] = sample
        cursor += size

    data = cflearn.ArrayData.init().fit(x)
    data.config.batch_size = batch_size
    loader = data.build_loader(x, batch_size=batch_size)
    config = cflearn.Config(module_name=module_name, loss_name="mse")
    inference = cflearn.Inference(model=cflearn.IModel.from_config(config))

    try:
        started = time.perf_counter_ns()
        for _ in range(iterations):
            outputs = inference.get_outputs(loader, pad_dim=1, verbose=False)
        finished = time.perf_counter_ns()
    finally:
        AsyncIterManager.cleanup(id(loader))

    padded = outputs.forward_results[cflearn.PREDICTIONS_KEY]
    assert isinstance(padded, torch.Tensor)
    expected_tensor = torch.from_numpy(expected)
    exact_match = torch.equal(padded, expected_tensor)
    assert list(padded.shape) == [num_samples * rows, max_width]
    assert exact_match

    behavior = {
        "checksum": int(padded.sum().item()),
        "dtype": str(padded.dtype).replace("torch.", ""),
        "exact_match": exact_match,
        "output_shape": list(padded.shape),
        "zero_count": int((padded == 0).sum().item()),
    }
    return {
        "elapsed_ns": finished - started,
        "iterations": iterations,
        "behavior": behavior,
        "uses_torch": True,
    }


def trainer_train_step(params: Dict[str, Any], seed: int) -> Dict[str, Any]:
    import torch

    import core.learn as cflearn

    from accelerate import Accelerator

    batch_size = int(params.get("batch_size", 16))
    input_dim = int(params.get("input_dim", params.get("dim", 16)))
    output_dim = int(params.get("output_dim", 4))
    hidden_units_param = params.get("hidden_units")
    hidden_dim = int(params.get("hidden_dim", params.get("hidden", 32)))
    depth = int(params.get("depth", 1))
    iterations = int(params.get("iterations", params.get("steps", 2)))
    learning_rate = float(params.get("learning_rate", params.get("lr", 1.0e-3)))
    if hidden_units_param is None:
        hidden_units = [hidden_dim] * depth
    else:
        hidden_units = [int(unit) for unit in hidden_units_param]
    assert batch_size > 0
    assert input_dim > 0
    assert output_dim > 0
    assert hidden_dim > 0
    assert depth > 0
    assert hidden_units
    assert all(unit > 0 for unit in hidden_units)
    assert iterations > 0
    assert learning_rate > 0.0

    torch.manual_seed(seed)
    config = cflearn.Config(
        module_name="fcnn",
        module_config={
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_units": hidden_units,
        },
        loss_name="mse",
        grad_accumulate=1,
        use_closure_pack=False,
    )
    model = cflearn.IModel.from_config(config)
    trainer = cflearn.Trainer(config)
    trainer.accelerator = Accelerator(cpu=True)
    trainer.callbacks = []
    trainer.model = model
    trainer.state = cflearn.TrainerState(
        num_epoch=1,
        num_steps=iterations,
        batch_size=batch_size,
        loader_length=iterations,
        enable_logging=False,
        min_num_sample=0,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    trainer.optimizers = {"all": optimizer}
    model.init_with_trainer(trainer)

    x = torch.linspace(
        -1.0,
        1.0,
        steps=batch_size * input_dim,
        dtype=torch.float32,
    ).reshape(batch_size, input_dim)
    y = torch.linspace(
        0.5,
        -0.5,
        steps=batch_size * output_dim,
        dtype=torch.float32,
    ).reshape(batch_size, output_dim)
    batch = {
        cflearn.INPUT_KEY: x,
        cflearn.LABEL_KEY: y,
    }
    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]

    started = time.perf_counter_ns()
    for batch_idx in range(iterations):
        trainer.state.step += 1
        step_outputs = trainer.train_step(batch_idx, batch)
    finished = time.perf_counter_ns()

    predictions = step_outputs.forward_results[cflearn.PREDICTIONS_KEY]
    loss = step_outputs.loss_tensors[cflearn.LOSS_KEY]
    parameter_changed = any(
        not torch.equal(before, after.detach())
        for before, after in zip(parameters_before, model.parameters())
    )
    assert list(predictions.shape) == [batch_size, output_dim]
    assert torch.isfinite(predictions).all().item()
    assert torch.isfinite(loss).all().item()
    assert parameter_changed
    assert trainer.state.step == iterations

    behavior = {
        "final_loss": float(loss.item()),
        "parameter_changed": parameter_changed,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "prediction_checksum": float(predictions.double().sum().item()),
        "prediction_shape": list(predictions.shape),
        "steps": trainer.state.step,
    }
    return {
        "elapsed_ns": finished - started,
        "iterations": iterations,
        "behavior": behavior,
        "uses_torch": True,
    }


def moe_dispatch_combine(params: Dict[str, Any], seed: int) -> Dict[str, Any]:
    import torch

    from core.learn.modules.moe import MoEDispatcher
    from core.learn.modules.moe import MoERouterOutputs

    batch_size = int(params.get("batch_size", params.get("batch", 32)))
    input_dim = int(params.get("input_dim", params.get("dim", 16)))
    output_dim = int(params.get("output_dim", 8))
    num_experts = int(params.get("num_experts", params.get("experts", 4)))
    top_k = int(params.get("top_k", 2))
    iterations = int(params.get("iterations", 1))
    assert batch_size > 0
    assert input_dim > 0
    assert output_dim > 0
    assert num_experts > 0
    assert 0 < top_k <= num_experts
    assert iterations > 0

    torch.manual_seed(seed)
    net = torch.randn(batch_size, input_dim)
    logits = torch.randn(batch_size, num_experts)
    expert_weights = [torch.randn(input_dim, output_dim) for _ in range(num_experts)]
    dispatcher = MoEDispatcher(top_k=top_k, num_experts=num_experts).eval()
    routings = MoERouterOutputs(logits, logits, None)

    with torch.inference_mode():
        started = time.perf_counter_ns()
        for _ in range(iterations):
            dispatch = dispatcher.dispatch(net, routings)
            expert_outputs = [
                expert_input @ expert_weight
                for expert_input, expert_weight in zip(
                    dispatch.inputs,
                    expert_weights,
                )
            ]
            combined = dispatcher.combine(
                expert_outputs,
                dispatch.indices,
                dispatch.gates_gahtered,
            )
        finished = time.perf_counter_ns()

        reference = torch.zeros_like(combined)
        for i in range(num_experts):
            reference += dispatch.gates[:, i : i + 1] * (net @ expert_weights[i])
        gate_sums = dispatch.gates.sum(1)
        max_gate_error = (gate_sums - 1.0).abs().max()
        matches_reference = torch.allclose(
            combined,
            reference,
            rtol=1.0e-5,
            atol=1.0e-5,
        )
        assert list(combined.shape) == [batch_size, output_dim]
        assert torch.isfinite(combined).all().item()
        assert matches_reference
        assert max_gate_error.item() < 1.0e-6
        assert int(dispatch.load.sum().item()) == batch_size * top_k

    behavior = {
        "checksum": float(combined.double().sum().item()),
        "expert_load": [int(value) for value in dispatch.load.tolist()],
        "load_sum": int(dispatch.load.sum().item()),
        "matches_reference": matches_reference,
        "max_gate_error": float(max_gate_error.item()),
        "output_shape": list(combined.shape),
    }
    return {
        "elapsed_ns": finished - started,
        "iterations": iterations,
        "behavior": behavior,
        "uses_torch": True,
    }


def ema_update(params: Dict[str, Any], seed: int) -> Dict[str, Any]:
    import torch

    import torch.nn as nn

    from core.learn.modules.common import EMA

    width = int(params.get("width", params.get("dim", 64)))
    depth = int(
        params.get(
            "num_layers",
            params.get("depth", params.get("layers", 2)),
        )
    )
    iterations = int(params.get("iterations", params.get("updates", 4)))
    decay = float(params.get("decay", 0.999))
    assert width > 0
    assert depth > 0
    assert iterations > 0
    assert 0.0 <= decay <= 1.0

    torch.manual_seed(seed)
    model = nn.Sequential(*[nn.Linear(width, width) for _ in range(depth)])
    ema = EMA(decay, model.named_parameters(), use_num_updates=True)

    elapsed_ns = 0
    for update in range(iterations):
        delta = float(update + 1) * 1.0e-4
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(delta)
        started = time.perf_counter_ns()
        ema()
        elapsed_ns += time.perf_counter_ns() - started

    target_parameters = list(ema.tgt_params.values())
    raw_snapshot = [parameter.detach().clone() for parameter in target_parameters]
    ema_snapshot = [
        getattr(ema, name).detach().clone() for name in ema.tgt_params.keys()
    ]
    ema.eval()
    eval_swap_ok = all(
        torch.equal(parameter.detach(), expected)
        for parameter, expected in zip(target_parameters, ema_snapshot)
    )
    ema.train()
    restore_ok = all(
        torch.equal(parameter.detach(), expected)
        for parameter, expected in zip(target_parameters, raw_snapshot)
    )
    assert ema.num_updates is not None
    num_updates = int(ema.num_updates.item())
    assert num_updates == iterations
    assert eval_swap_ok
    assert restore_ok

    behavior = {
        "ema_checksum": float(
            sum(parameter.double().sum().item() for parameter in ema_snapshot)
        ),
        "eval_swap_ok": eval_swap_ok,
        "num_parameters": sum(parameter.numel() for parameter in target_parameters),
        "num_updates": num_updates,
        "restore_ok": restore_ok,
    }
    return {
        "elapsed_ns": elapsed_ns,
        "iterations": iterations,
        "behavior": behavior,
        "uses_torch": True,
    }
