from __future__ import annotations

import os
import time
import torch
import argparse

import core.learn as cflearn
import torch.distributed as dist

from typing import List
from typing import Tuple
from typing import Optional
from pathlib import Path
from datetime import timedelta
from accelerate import Accelerator
from unittest.mock import patch
from torch.utils.data import DataLoader
from core.toolkit.misc import is_rank_0
from core.toolkit.misc import is_local_rank_0
from core.toolkit.misc import only_execute_on_rank0
from core.toolkit.misc import only_execute_on_local_rank0
from core.toolkit.misc import wait_for_everyone_at_end
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.elastic.multiprocessing.errors import record


class _DistributedGuardFailure(ValueError):
    pass


class _WorkspaceFailure(RuntimeError):
    pass


class _RealtimeSaveFailure(RuntimeError):
    pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=(
            "success",
            "rank_failure",
            "coordinated_decorators",
            "workspace_failure",
            "realtime_initial_save",
            "global_main_write",
            "prepared_pipeline",
            "prepared_remainder",
            "callback_lifecycle",
            "uneven_metric_state",
            "inference_gather",
            "timeout",
        ),
        required=True,
    )
    parser.add_argument("--shared-target", type=Path)
    return parser.parse_args()


def _make_identity_model() -> Tuple[cflearn.Config, cflearn.IModel]:
    config = cflearn.Config(
        module_name="linear",
        module_config={
            "input_dim": 1,
            "output_dim": 1,
            "bias": False,
        },
        loss_name="mse",
        use_losses_as_metrics=True,
    )
    model = cflearn.IModel.from_config(config)
    with torch.no_grad():
        next(model.m.parameters()).fill_(1.0)
    return config, model


def _run_success(rank: int) -> None:
    value = torch.tensor([rank + 1], dtype=torch.int64)
    dist.all_reduce(value)
    assert value.item() == 3
    print(f"rank {rank}: all_reduce success", flush=True)


def _run_rank_failure(rank: int) -> None:
    dist.barrier()
    if rank == 1:
        raise RuntimeError("intentional failure from rank 1")
    print("rank 0: waiting for rank 1 failure propagation", flush=True)
    time.sleep(60.0)


def _assert_process_group_usable(rank: int) -> None:
    value = torch.tensor([rank + 1], dtype=torch.int64)
    dist.all_reduce(value)
    assert value.item() == 3


def _assert_coordinated_failure(
    rank: int,
    origin_rank: int,
    original: Exception,
    error: Exception,
) -> None:
    if rank == origin_rank:
        assert error is original
        return
    assert type(error) is RuntimeError
    message = str(error)
    assert f"rank {origin_rank}" in message.lower()
    assert type(original).__name__ in message
    assert str(original) in message


def _run_coordinated_decorators(rank: int) -> None:
    assert is_rank_0() is (rank == 0)
    assert is_local_rank_0() is (rank == 0)

    @wait_for_everyone_at_end
    def identity(value: int) -> int:
        return value

    assert identity(rank) == rank

    calls = [0, 0]

    @only_execute_on_rank0
    def global_main() -> None:
        calls[0] += 1

    @only_execute_on_local_rank0
    def local_main() -> None:
        calls[1] += 1

    global_main()
    local_main()
    reduced_calls = torch.tensor(calls, dtype=torch.int64)
    dist.all_reduce(reduced_calls)
    assert reduced_calls.tolist() == [1, 1]

    generic_error = _DistributedGuardFailure("generic failure from rank 1")

    @wait_for_everyone_at_end
    def fail_on_rank_1() -> None:
        if rank == 1:
            raise generic_error

    try:
        fail_on_rank_1()
    except Exception as error:
        _assert_coordinated_failure(rank, 1, generic_error, error)
    else:
        raise AssertionError("the coordinated rank-1 failure was not propagated")
    _assert_process_group_usable(rank)

    rank_0_error = _DistributedGuardFailure("rank-0-only failure")

    @only_execute_on_rank0
    def fail_on_rank_0() -> None:
        raise rank_0_error

    try:
        fail_on_rank_0()
    except Exception as error:
        _assert_coordinated_failure(rank, 0, rank_0_error, error)
    else:
        raise AssertionError("the rank-0 failure was not propagated")
    _assert_process_group_usable(rank)
    print(f"rank {rank}: coordinated decorators success", flush=True)


def _run_workspace_failure(rank: int) -> None:
    workspace = Path.cwd() / "workspace_failure"
    shared_workspace = workspace / "rank_0"
    success_config = cflearn.Config(
        workspace=str(workspace),
        create_sub_workspace=True,
    )
    success_block = cflearn.PrepareWorkspaceBlock()
    success_block.training_workspace = workspace

    def prepare_success(*args, **kwargs):
        assert rank == 0
        return shared_workspace

    with patch(
        "core.learn.pipeline.blocks.basic.prepare_workspace_from",
        side_effect=prepare_success,
    ):
        success_block.build(success_config)
    assert Path(success_config.persistence.workspace) == shared_workspace
    _assert_process_group_usable(rank)

    config = cflearn.Config(workspace=str(workspace), create_sub_workspace=True)
    block = cflearn.PrepareWorkspaceBlock()
    block.training_workspace = workspace
    original = _WorkspaceFailure("workspace creation failed on rank 0")

    def fail_prepare(*args, **kwargs):
        assert rank == 0
        raise original

    with patch(
        "core.learn.pipeline.blocks.basic.prepare_workspace_from",
        side_effect=fail_prepare,
    ):
        try:
            block.build(config)
        except Exception as error:
            _assert_coordinated_failure(rank, 0, original, error)
        else:
            raise AssertionError("the workspace failure was not propagated")
    _assert_process_group_usable(rank)
    print(f"rank {rank}: workspace failure propagated", flush=True)


def _run_realtime_initial_save(rank: int, shared_target: Optional[Path]) -> None:
    if shared_target is None:
        raise ValueError("'--shared-target' is required")
    shared_target.mkdir(parents=True, exist_ok=True)
    config = cflearn.Config(
        workspace=str(Path.cwd()),
        save_pipeline_in_realtime=True,
    )
    trainer = cflearn.Trainer(config)
    trainer.pipeline = None  # type: ignore
    callback = cflearn.UpdateArtifactsCallback()

    def record(stage: str) -> None:
        (shared_target / f"{stage}_rank_{rank}").touch()

    def save(*args: object, **kwargs: object) -> None:
        record("success")

    with patch("core.learn.pipeline.PipelineSerializer.save", side_effect=save):
        callback.before_loop(trainer)

    original = _RealtimeSaveFailure("initial pipeline save failed")

    def fail_save(*args: object, **kwargs: object) -> None:
        record("failure")
        raise original

    with patch(
        "core.learn.pipeline.PipelineSerializer.save",
        side_effect=fail_save,
    ):
        try:
            callback.before_loop(trainer)
        except Exception as error:
            _assert_coordinated_failure(rank, 0, original, error)
        else:
            raise AssertionError("the initial-save failure was not propagated")

    _assert_process_group_usable(rank)
    assert {path.name for path in shared_target.iterdir()} == {
        "success_rank_0",
        "failure_rank_0",
    }
    print(f"rank {rank}: realtime initial save coordinated", flush=True)


def _run_global_main_write(rank: int, shared_target: Optional[Path]) -> None:
    if shared_target is None:
        raise ValueError("'--shared-target' is required")
    expected_is_main = rank == 0
    assert is_rank_0() is expected_is_main
    if is_rank_0():
        shared_target.write_text(f"rank={rank}\n", encoding="utf-8")
    dist.barrier()
    assert shared_target.read_text(encoding="utf-8") == "rank=0\n"
    print(f"rank {rank}: observed the global-main artifact", flush=True)


def _run_timeout(rank: int) -> None:
    dist.barrier()
    print(f"rank {rank}: waiting for the parent timeout", flush=True)
    time.sleep(60.0)


def _run_callback_lifecycle(rank: int) -> None:
    @cflearn.IModel.register("$ddp_callback_model")
    class CallbackModel(cflearn.CommonModel):
        @property
        def all_modules(self) -> List[torch.nn.Module]:
            return [self.m]

    @cflearn.TrainerCallback.register("$ddp_callback_lifecycle")
    class CallbackLifecycle(cflearn.TrainerCallback):
        events: List[str]

        def __init__(self) -> None:
            self.events = []

        def initialize(self) -> None:
            self.events.append("initialize")

        def after_workspace_prepared(self, trainer: cflearn.ITrainer) -> None:
            self.events.append("after_workspace_prepared")

        def before_summary(self, trainer: cflearn.ITrainer) -> None:
            self.events.append("before_summary")

        def before_loop(self, trainer: cflearn.ITrainer) -> None:
            self.events.append("before_loop")

        def before_loop_with_loaders(
            self,
            trainer: cflearn.ITrainer,
            train_loader: DataLoader,
            valid_loader: Optional[DataLoader],
        ) -> None:
            self.events.append("before_loop_with_loaders")

        def log_lr(self, key: str, lr: float, trainer: cflearn.ITrainer) -> None:
            self.events.append("log_lr")

        def log_train_step(
            self,
            step_outputs: cflearn.StepOutputs,
            state: cflearn.TrainerState,
        ) -> None:
            self.events.append("log_train_step")

        def after_train_step(
            self,
            batch: cflearn.tensor_dict_type,
            step_outputs: cflearn.StepOutputs,
            trainer: cflearn.ITrainer,
        ) -> None:
            self.events.append("after_train_step")

        def log_metrics(
            self,
            metric_outputs: cflearn.MetricsOutputs,
            state: cflearn.TrainerState,
        ) -> None:
            self.events.append("log_metrics")

        def after_save_checkpoint(self, trainer: cflearn.ITrainer) -> None:
            self.events.append("after_save_checkpoint")

        def at_terminate(self, trainer: cflearn.ITrainer) -> None:
            self.events.append("at_terminate")

        def after_loop(self, trainer: cflearn.ITrainer) -> None:
            self.events.append("after_loop")

        def finalize(self, trainer: cflearn.ITrainer) -> None:
            self.events.append("finalize")

    data, in_dim, out_dim = cflearn.testing.arange_data(
        n=2,
        dim=2,
        out_dim=1,
        batch_size=2,
    )
    config = cflearn.Config(
        workspace=str(Path.cwd() / "callback_lifecycle"),
        create_sub_workspace=False,
        model=CallbackModel.__identifier__,
        module_name="linear",
        module_config={"input_dim": in_dim, "output_dim": out_dim},
        loss_name="mse",
        scheduler_name="warmup",
        scheduler_config={"warmup_step": 2},
        monitor_names="basic",
        callback_names=[CallbackLifecycle.__identifier__],
        auto_callback=False,
        num_steps=1,
        state_config={
            "max_snapshot_file": 1,
            "snapshot_start_step": 1,
            "num_step_per_snapshot": 1,
        },
    )
    pipeline = cflearn.TrainingPipeline.init(config).fit(
        data,
        do_summary=False,
        skip_final_evaluation=True,
    )
    assert dist.is_initialized()
    assert "gloo" in str(dist.get_backend()).lower()
    callbacks = pipeline.training.build_callbacks.callbacks
    callback = next(c for c in callbacks if isinstance(c, CallbackLifecycle))
    trainer = pipeline.training.build_trainer.trainer
    assert (
        next(c for c in trainer.callbacks if isinstance(c, CallbackLifecycle))
        is callback
    )
    if rank == 0:
        expected_events = [
            "initialize",
            "after_workspace_prepared",
            "before_summary",
            "before_loop",
            "before_loop_with_loaders",
            "log_lr",
            "log_train_step",
            "after_train_step",
            "log_metrics",
            "after_save_checkpoint",
            "at_terminate",
            "after_loop",
            "log_metrics",
            "finalize",
        ]
    else:
        expected_events = [
            "initialize",
            "before_summary",
            "before_loop",
            "before_loop_with_loaders",
            "after_train_step",
            "at_terminate",
            "after_loop",
            "finalize",
        ]
    assert callback.events == expected_events, callback.events
    print(f"rank {rank}: callback lifecycle success", flush=True)


def _run_prepared_pipeline(rank: int) -> None:
    @cflearn.IModel.register("$ddp_prepared_model")
    class PreparedModel(cflearn.IModel):
        @property
        def train_steps(self):
            return []

        @property
        def all_modules(self):
            return [self.m, self.aux]

        def build(self, config):
            self.m = cflearn.build_module(
                config.module_name,
                config=config.module_config,
            )
            self.aux = torch.nn.Linear(1, 1, bias=False)
            self.runtime_state = object()

    data, in_dim, out_dim = cflearn.testing.arange_data(
        n=2,
        dim=2,
        out_dim=1,
        batch_size=2,
    )
    config = cflearn.Config(
        model=PreparedModel.__identifier__,
        module_name="linear",
        module_config={"input_dim": in_dim, "output_dim": out_dim},
    )
    pipeline = cflearn.InferencePipeline.build_with(config)
    original_model = pipeline.build_model.model
    runtime_state = original_model.runtime_state
    accelerator = Accelerator(cpu=True)
    pipeline.prepare_distributed_with(accelerator)
    prepared_model = pipeline.build_model.model
    assert prepared_model is original_model
    assert prepared_model.runtime_state is runtime_state
    assert pipeline.build_inference.inference.model is prepared_model
    assert isinstance(prepared_model.m, DistributedDataParallel)
    assert isinstance(prepared_model.aux, DistributedDataParallel)

    num_forwards = 0

    def count_forward(*args: object) -> None:
        nonlocal num_forwards
        num_forwards += 1

    handle = prepared_model.m.register_forward_hook(count_forward)
    try:
        loader = data.build_loader(data.bundle.x_train, batch_size=2)
        predictions = pipeline.predict(
            loader,
            recover_predictions=False,
            accelerator=accelerator,
            verbose=False,
        )[cflearn.PREDICTIONS_KEY]
    finally:
        handle.remove()
    assert num_forwards == 1
    assert predictions.shape == (2, 1)
    assert torch.isfinite(predictions).all()
    print(f"rank {rank}: prepared pipeline prediction success", flush=True)


def _run_prepared_remainder(rank: int) -> None:
    config, model = _make_identity_model()
    values = torch.tensor([[0.0], [0.0], [3.0]])
    dataset = [
        {
            cflearn.INPUT_KEY: value,
            cflearn.LABEL_KEY: torch.zeros_like(value),
        }
        for value in values
    ]
    accelerator = Accelerator(cpu=True)
    loader = accelerator.prepare(
        DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
        )
    )
    gather_inference_modes = []
    original_gather = accelerator.gather

    def gather(tensors):
        gather_inference_modes.append(torch.is_inference_mode_enabled())
        return original_gather(tensors)

    with patch.object(accelerator, "gather", side_effect=gather):
        outputs = model.evaluate(
            config,
            None,
            cflearn.Inference(model=model),
            loader,
            return_outputs=False,
            recover_labels=False,
            recover_predictions=False,
            concat_outputs=False,
            accelerator=accelerator,
            verbose=False,
        )

    assert outputs.loss_items is not None
    assert accelerator.gradient_state.remainder == 1
    assert gather_inference_modes == [False]
    assert outputs.loss_items[cflearn.LOSS_KEY] == 3.0
    print(f"rank {rank}: prepared remainder success", flush=True)


def _run_inference_gather(rank: int) -> None:
    _, model = _make_identity_model()
    accelerator = Accelerator(cpu=True)
    gathered = cflearn.Inference(model=model).gather(
        accelerator,
        {
            "plain": torch.tensor([float(rank + 1)]),
            "padded": torch.arange(
                rank + 1,
                dtype=torch.float32,
            )
            + 10.0 * (rank + 1),
        },
        pad_dim={"padded": 0},
    )

    if rank == 0:
        assert [tensor.tolist() for tensor in gathered["plain"]] == [[1.0], [2.0]]
        assert [tensor.tolist() for tensor in gathered["padded"]] == [
            [10.0, 0.0],
            [20.0, 21.0],
        ]
    else:
        assert gathered == {"plain": None, "padded": None}
    print(f"rank {rank}: inference gather success", flush=True)


def _run_uneven_metric_state(rank: int) -> None:
    config, model = _make_identity_model()
    values = [0.0, 0.0] if rank == 0 else [3.0]
    inputs = torch.tensor(values, dtype=torch.float32).reshape(-1, 1)
    loader = [
        {
            cflearn.INPUT_KEY: inputs,
            cflearn.LABEL_KEY: torch.zeros_like(inputs),
        }
    ]
    outputs = model.evaluate(
        config,
        cflearn.IMetric.fuse(["mse", "stream_mse"]),
        cflearn.Inference(model=model),
        loader,  # type: ignore
        return_outputs=False,
        recover_labels=False,
        recover_predictions=False,
        concat_outputs=False,
        accelerator=Accelerator(cpu=True),
        verbose=False,
    )

    expected_metric = 3.0
    assert outputs.loss_items is not None
    assert outputs.loss_items[cflearn.LOSS_KEY] == 4.5
    metric_outputs = outputs.metric_outputs
    assert metric_outputs is not None
    for key in ["mse", "stream_mse"]:
        assert metric_outputs.metric_values[key] == expected_metric
    print(f"rank {rank}: uneven metric state success", flush=True)


@record
def main() -> None:
    args = _parse_args()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2
    assert os.environ["LOCAL_RANK"] == str(rank)
    assert not torch.cuda.is_available()

    if args.scenario == "callback_lifecycle":
        try:
            _run_callback_lifecycle(rank)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
        return

    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        timeout=timedelta(seconds=15.0),
    )
    try:
        assert dist.get_backend() == "gloo"
        if args.scenario == "success":
            _run_success(rank)
        elif args.scenario == "rank_failure":
            _run_rank_failure(rank)
        elif args.scenario == "coordinated_decorators":
            _run_coordinated_decorators(rank)
        elif args.scenario == "workspace_failure":
            _run_workspace_failure(rank)
        elif args.scenario == "realtime_initial_save":
            _run_realtime_initial_save(rank, args.shared_target)
        elif args.scenario == "global_main_write":
            _run_global_main_write(rank, args.shared_target)
        elif args.scenario == "prepared_pipeline":
            _run_prepared_pipeline(rank)
        elif args.scenario == "prepared_remainder":
            _run_prepared_remainder(rank)
        elif args.scenario == "uneven_metric_state":
            _run_uneven_metric_state(rank)
        elif args.scenario == "inference_gather":
            _run_inference_gather(rank)
        else:
            _run_timeout(rank)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
