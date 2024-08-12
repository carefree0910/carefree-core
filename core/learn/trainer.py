import os
import json
import math
import torch

from enum import Enum
from typing import Any
from typing import Set
from typing import Dict
from typing import List
from typing import Callable
from typing import Optional
from pathlib import Path
from accelerate import Accelerator
from accelerate import DataLoaderConfiguration
from tqdm.autonotebook import tqdm
from torch.optim import Optimizer
from torch.profiler import profile
from torch.optim.lr_scheduler import LRScheduler

from .schema import device_type
from .schema import prepare_dataloaders
from .schema import weighted_loss_score
from .schema import IData
from .schema import IModel
from .schema import IMetric
from .schema import ITrainer
from .schema import IInference
from .schema import DataLoader
from .schema import StepOutputs
from .schema import TqdmSettings
from .schema import TrainerState
from .schema import TrainerConfig
from .schema import MetricsOutputs
from .schema import MonitorResults
from .schema import TrainerMonitor
from .schema import TrainerCallback
from .toolkit import summary
from .toolkit import get_torch_device
from .callbacks import TrainingLoopCallback
from .constants import PT_PREFIX
from .constants import SCORES_FILE
from .constants import CHECKPOINTS_FOLDER
from ..toolkit import console
from ..toolkit.misc import is_ddp
from ..toolkit.misc import to_path
from ..toolkit.misc import safe_execute
from ..toolkit.misc import shallow_copy_dict
from ..toolkit.misc import sort_dict_by_value
from ..toolkit.misc import is_dist_initialized
from ..toolkit.misc import Incrementer
from ..toolkit.types import TPath
from ..toolkit.types import tensor_dict_type


T_Lo = Optional[DataLoader]


class SortMethod(str, Enum):
    BEST = "best"
    LATEST = "latest"


def get_scores(checkpoint_folder: TPath) -> Dict[str, float]:
    scores_path = to_path(checkpoint_folder) / SCORES_FILE
    if not scores_path.is_file():
        return {}
    with scores_path.open("r") as f:
        return json.load(f)


def get_sorted_checkpoints(
    checkpoint_folder: TPath,
    *,
    sort_by: SortMethod = SortMethod.BEST,
    target_ckpt_step: Optional[int] = None,
) -> List[str]:
    """
    'better' checkpoints will be placed earlier, which means `checkpoints[0]` is the 'best' checkpoint
    > which checkpoint is 'better' is determined by the `sort_by` parameter
    """

    scores = get_scores(checkpoint_folder)
    if not scores:
        return []
    if target_ckpt_step is not None:
        target_file = f"{PT_PREFIX}{target_ckpt_step}.pt"
        if target_file in scores:
            return [target_file]
        raise RuntimeError(
            f"checkpoint '{target_file}' is not found under '{checkpoint_folder}' "
            f"(available: {', '.join(sorted(scores))})"
        )
    if sort_by == SortMethod.BEST:
        return list(sort_dict_by_value(scores, reverse=True).keys())
    return sorted(scores.keys(), key=lambda k: int(Path(k).stem.split("_")[1]))[::-1]


def get_metrics_path(workspace: TPath) -> Path:
    return to_path(workspace) / Trainer.metrics_log_file


def is_started_workspace(workspace: TPath) -> bool:
    return get_metrics_path(workspace).is_file()


def is_finished_workspace(workspace: TPath) -> bool:
    if not is_started_workspace(workspace):
        return False
    with get_metrics_path(workspace).open("r") as f:
        metrics = [line.strip() for line in f]
    metrics = [line for line in metrics if line]
    return len(metrics) > 0 and "epoch  -1" in metrics[-1]


def is_crashed_workspace(workspace: TPath) -> bool:
    return is_started_workspace(workspace) and not is_finished_workspace(workspace)


class Trainer(ITrainer):
    model_log_file = "model.txt"
    metrics_log_file = "metrics.txt"
    summary_log_file = "summary.txt"

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.tqdm_settings = safe_execute(TqdmSettings, config.tqdm_settings or {})  # type: ignore
        self.accelerator: Accelerator = None
        self.loss_incrementers: Dict[str, Incrementer] = {}
        self.intermediate: Optional[MetricsOutputs] = None
        self.final_results: Optional[MetricsOutputs] = None
        self.checkpoint_scores: Dict[str, float] = {}

    @property
    def device(self) -> torch.device:
        return self.accelerator.device

    @property
    def is_rank_0(self) -> bool:
        return self.accelerator.is_main_process

    @property
    def is_local_rank_0(self) -> bool:
        return self.accelerator.is_local_main_process

    @property
    def use_tqdm_in_validation(self) -> bool:
        if not self.is_local_rank_0:
            return False
        return self.tqdm_settings.use_tqdm_in_validation or self.state.is_terminate

    @property
    def has_checkpoint_folder(self) -> bool:
        return os.path.isdir(self.checkpoint_folder)

    @property
    def workspace(self) -> str:
        return self.config.workspace

    @property
    def checkpoint_folder(self) -> str:
        return os.path.join(self.workspace, CHECKPOINTS_FOLDER)

    @property
    def should_autocast(self) -> bool:
        return self.config.mixed_precision != "no"

    # api

    def fit(
        self,
        data: IData,
        model: IModel,
        metrics: Optional[IMetric],
        inference: IInference,
        optimizers: Dict[str, Optimizer],
        schedulers: Dict[str, Optional[LRScheduler]],
        monitors: List[TrainerMonitor],
        callbacks: List[TrainerCallback],
        schedulers_requires_metric: Set[str],
        *,
        show_summary: bool = True,
        skip_final_evaluation: bool = False,
        only_touch: bool = False,
        device: device_type = None,
        p: Optional[profile] = None,
    ) -> "Trainer":
        # accelerator
        cpu = False
        if not is_ddp():
            device = get_torch_device(device)
            if device.type == "cpu":
                cpu = True
            else:
                torch.cuda.set_device(device)  # pragma: no cover
        self.config.init_process_group(cpu=cpu)
        self.accelerator = Accelerator(
            cpu=cpu,
            mixed_precision=self.config.mixed_precision,
            dataloader_config=DataLoaderConfiguration(
                split_batches=self.config.split_batches,
                dispatch_batches=self.config.dispatch_batches,
                even_batches=self.config.even_batches,
            ),
        )
        self.accelerator.wait_for_everyone()
        # initialize artifact structure
        if self.is_local_rank_0:
            os.makedirs(self.workspace, exist_ok=True)
            for callback in callbacks:
                callback.after_workspace_prepared(self)
            self.metrics_log_path = os.path.join(self.workspace, self.metrics_log_file)
            with open(self.metrics_log_path, "w"):
                pass
            os.makedirs(self.checkpoint_folder, exist_ok=True)
        # initialize
        self.metrics = metrics
        self.monitors = monitors
        self.callbacks = callbacks
        if not any(isinstance(c, TrainingLoopCallback) for c in self.callbacks):
            console.warn(  # pragma: no cover
                "`TrainingLoopCallback` is not found in the callbacks, "
                "some features may not work as expected"
            )
        self.schedulers_requires_metric = schedulers_requires_metric
        if self.is_local_rank_0:
            with open(os.path.join(self.workspace, self.model_log_file), "w") as f:
                f.write(str(model))
        self.inference = inference
        # accelerator prepare
        n_optim = len(optimizers)
        optim_keys = sorted(optimizers)
        train_loader, valid_loader = data.build_loaders()
        prepared_lds = prepare_dataloaders(self.accelerator, train_loader, valid_loader)
        distributed_train_loader = prepared_lds[0]
        distributed_valid_loader = prepared_lds[1]
        assert distributed_train_loader is not None
        prepared = self.accelerator.prepare(
            *model.all_modules,
            *[optimizers[k] for k in optim_keys],
        )
        self.state = TrainerState(
            num_epoch=self.config.num_epoch,
            num_steps=self.config.num_steps,
            batch_size=train_loader.batch_size,  # type: ignore
            loader_length=len(distributed_train_loader),
            **(self.config.state_config or {}),
        )
        self.model = model.from_accelerator(*prepared[:-n_optim])
        self.inference.model = self.model
        self.optimizers = {k: prepared[-n_optim + i] for i, k in enumerate(optim_keys)}
        self.schedulers = schedulers
        for sch in schedulers.values():
            if sch is not None:
                sch.load_state_dict(sch.state_dict())
        # summary
        for callback in self.callbacks:
            callback.before_summary(self)
        ## should always summary to sync the statuses in distributed training
        input_sample = train_loader.get_input_sample(self.device)
        summary_msg = summary(
            self.model.m,
            input_sample,
            return_only=not show_summary or not self.is_local_rank_0 or only_touch,
            summary_forward=self.model.summary_forward,
        )
        if self.is_local_rank_0:
            with open(os.path.join(self.workspace, self.summary_log_file), "w") as f:
                f.write(summary_msg)
        # tqdm
        step_tqdm = None
        self.epoch_tqdm: Optional[tqdm] = None
        if self.is_local_rank_0 and self.tqdm_settings.use_tqdm:
            self.epoch_tqdm = tqdm(
                list(range(self.state.num_epoch)),
                position=self.tqdm_settings.position,
                desc=self.tqdm_settings.desc,
                leave=False,
            )
        # train
        has_ckpt = terminate = False
        for callback in self.callbacks:
            callback.before_loop(self)
        self.accelerator.wait_for_everyone()
        if self.is_local_rank_0 and self.epoch_tqdm is None:
            console.debug("entered training loop")
        while self.state.should_train and not only_touch:
            try:
                for callback in self.callbacks:
                    callback.at_epoch_start(self)
                self.state.epoch += 1
                if not self.is_local_rank_0 or not self.tqdm_settings.use_step_tqdm:
                    step_iterator = distributed_train_loader
                else:
                    step_tqdm = step_iterator = tqdm(
                        distributed_train_loader,
                        total=len(distributed_train_loader),
                        position=self.tqdm_settings.position
                        + int(self.tqdm_settings.use_tqdm),
                        leave=False,
                    )
                for i, batch in enumerate(step_iterator):
                    if i == 0:
                        self.accelerator.wait_for_everyone()
                    for callback in self.callbacks:
                        callback.at_step_start(batch, self)
                    self.state.step += 1
                    step_outputs = self.train_step(i, batch)
                    if self.is_local_rank_0:
                        for callback in self.callbacks:
                            callback.log_train_step(step_outputs, self.state)
                    for callback in self.callbacks:
                        callback.after_train_step(batch, step_outputs, self)
                    monitored = self.monitor(
                        distributed_train_loader,
                        distributed_valid_loader,
                        step_outputs,
                    )
                    if self.state.should_monitor:
                        for callback in self.callbacks:
                            callback.after_monitor(monitored, self)
                    if monitored.save_checkpoint:
                        metric_outputs = monitored.metric_outputs
                        assert metric_outputs is not None
                        self.save_checkpoint(metric_outputs.final_score)
                        if self.is_local_rank_0:
                            for callback in self.callbacks:
                                callback.after_save_checkpoint(self)
                    for callback in self.callbacks:
                        callback.at_step_end(self)
                    terminate = monitored.terminate or self.state.should_terminate
                    if terminate:
                        for callback in self.callbacks:
                            callback.at_terminate(self)
                        break
                    if p is not None and self.is_local_rank_0:
                        p.step()
                for callback in self.callbacks:
                    callback.at_epoch_end(self)
            except KeyboardInterrupt:
                if is_dist_initialized():
                    raise
                console.error("keyboard interrupted")
                terminate = True
            if terminate:
                break
            if self.epoch_tqdm is not None:
                self.epoch_tqdm.total = self.state.num_epoch
                self.epoch_tqdm.update()
        if self.epoch_tqdm is not None:
            if step_tqdm is not None:
                step_tqdm.close()
            self.epoch_tqdm.close()
        # restore
        self.accelerator.wait_for_everyone()
        if self.has_checkpoint_folder and not only_touch:
            if self.is_local_rank_0:
                console.debug("rolling back to the best checkpoint")
            has_ckpt = self.restore_checkpoint()
        # finalize
        self.state.set_terminate()
        if only_touch or skip_final_evaluation:
            self.final_results = self.intermediate
        else:
            loader = distributed_valid_loader or distributed_train_loader
            self.final_results = self.get_metrics(loader, self.config.valid_portion)
        if self.final_results is not None:
            self.log_with(self.final_results)
        if not has_ckpt:
            if self.final_results is None:
                final_score = 0.0
            else:
                final_score = self.final_results.final_score
            self.save_checkpoint(final_score)
        for callback in self.callbacks:
            callback.finalize(self)
        return self

    ## checkpointing

    def save_checkpoint(
        self,
        score: float,
        folder: Optional[TPath] = None,
        *,
        no_history: bool = False,
        check_rank_0: bool = True,
    ) -> None:
        """
        FSDP requires all ranks to call `state_dict()`, so we need to
        enter this method on all ranks, but only rank 0 should `do_save`.
        """

        if folder is None:
            folder = self.checkpoint_folder
        folder = to_path(folder)
        state: Optional[TrainerState] = getattr(self, "state", None)
        pt_file = f"{PT_PREFIX}{-1 if state is None else state.step}.pt"
        do_save = self.is_local_rank_0 or not check_rank_0
        if state is None:
            self.model.save(folder / pt_file, do_save=do_save)
            if self.is_local_rank_0:
                console.warn(
                    "`state` is not initialized, "
                    "latest model will be saved and the recorded score will always be 0"
                )
                with (folder / SCORES_FILE).open("w") as f:
                    json.dump({pt_file: 0.0}, f)
            return
        # leave top_k snapshots only
        if self.is_local_rank_0 and state.max_snapshot_file > 0:
            checkpoints = get_sorted_checkpoints(folder)
            if len(checkpoints) >= state.max_snapshot_file:
                for file in checkpoints[state.max_snapshot_file - 1 :]:
                    self.checkpoint_scores.pop(file)
                    (folder / file).unlink()
        # pt
        self.model.save(folder / pt_file, do_save=do_save)
        # scores
        if self.is_local_rank_0:
            scores = {} if no_history else self.checkpoint_scores
            scores[pt_file] = score
            with (folder / SCORES_FILE).open("w") as f:
                json.dump(sort_dict_by_value(scores, reverse=True), f)

    def restore_checkpoint(
        self,
        folder: Optional[TPath] = None,
        strict: bool = True,
        state_dict_callback: Optional[Callable[[tensor_dict_type], None]] = None,
    ) -> bool:
        if folder is None:
            folder = self.checkpoint_folder
        folder = to_path(folder)
        checkpoints = get_sorted_checkpoints(folder)
        if not checkpoints:
            if self.is_local_rank_0:
                console.warn(f"no model file found in {folder}")
            return False
        success = False
        for checkpoint in checkpoints:
            model_file = folder / checkpoint
            if not os.path.isfile(model_file):
                continue
            if self.is_local_rank_0:
                console.debug(f"restoring from '{model_file}'")
            states = torch.load(model_file, map_location=self.device)["states"]
            if state_dict_callback is not None:
                state_dict_callback(states)
            self.model.load_state_dict(states, strict)
            success = True
            break
        return success

    ## internal

    def train_step(self, batch_idx: int, batch: tensor_dict_type) -> StepOutputs:
        forward_kw: Dict[str, Any] = {}
        for callback in self.callbacks:
            callback.mutate_forward_kwargs(forward_kw, self)
        loss_kw: Dict[str, Any] = {}
        for callback in self.callbacks:
            callback.mutate_loss_kwargs(loss_kw, self)
        return self.model.train(batch_idx, batch, self, forward_kw, loss_kw)

    # `loader` is distributed loader
    def get_metrics(self, loader: DataLoader, portion: float = 1.0) -> MetricsOutputs:
        if not self.use_tqdm_in_validation:
            use_tqdm = False
            tqdm_kwargs = None
        else:
            use_tqdm = True
            tqdm_kwargs = dict(
                total=math.ceil(len(loader) * portion),
                position=self.tqdm_settings.position
                + int(self.tqdm_settings.use_tqdm)
                + int(self.tqdm_settings.use_step_tqdm),
                leave=False,
            )
        kw = shallow_copy_dict(self.config.metric_forward_kwargs or {})
        kw["return_outputs"] = False
        outputs = self.model.evaluate(
            self.config,
            self.metrics,
            self.inference,
            loader,
            portion=portion,
            state=self.state,
            use_tqdm=use_tqdm,
            tqdm_kwargs=tqdm_kwargs,
            accelerator=self.accelerator,
            **kw,
        )
        return outputs.metric_outputs  # type: ignore

    def log_with(self, metrics_outputs: MetricsOutputs) -> None:
        if not self.is_local_rank_0:
            return None
        if self.epoch_tqdm is not None:
            metric_values = shallow_copy_dict(metrics_outputs.metric_values)
            metric_values["score"] = metrics_outputs.final_score
            self.epoch_tqdm.set_postfix(metric_values)
        if self.state.should_log_metrics_msg:
            for c in self.callbacks:
                c.log_metrics_msg(metrics_outputs, self.metrics_log_path, self.state)
        if self.is_rank_0:
            for c in self.callbacks:
                c.log_metrics(metrics_outputs, self.state)
            if self.state.should_log_artifacts:
                for c in self.callbacks:
                    c.log_artifacts(self)

    # `*_loader`s are distributed loaders
    def monitor(
        self,
        train_loader: DataLoader,
        valid_loader: T_Lo,
        step_outputs: StepOutputs,
    ) -> MonitorResults:
        extension = 0
        terminate = False
        save_checkpoint = False
        for monitor in self.monitors:
            if self.state.should_extend_epoch:
                monitor.punish_extension()
                extension = max(extension, monitor.get_extension(self.state))
        if extension:
            self.state.num_epoch += extension
        if (
            valid_loader is None
            and self.config.use_incrementer_for_train_losses_in_eval
        ):
            window = max(3, self.state.num_step_per_snapshot)
            for k, v in step_outputs.loss_tensors.items():
                k_inc = self.loss_incrementers.setdefault(k, Incrementer(window))
                k_inc.update(v)  # type: ignore
        if self.state.should_monitor:
            # get metrics
            valid_portion = self.config.valid_portion
            if valid_loader is not None:
                self.intermediate = self.get_metrics(valid_loader, valid_portion)
            elif (
                not self.config.use_incrementer_for_train_losses_in_eval
                and self.config.recompute_train_losses_in_eval
            ):
                self.intermediate = self.get_metrics(train_loader, valid_portion)
            else:
                if not self.config.use_incrementer_for_train_losses_in_eval:
                    loss_tensors = shallow_copy_dict(step_outputs.loss_tensors)
                else:
                    loss_tensors = {
                        k: incrementer.mean
                        for k, incrementer in self.loss_incrementers.items()
                    }
                loss_items = {k: v.item() for k, v in loss_tensors.items()}
                is_positive = {k: False for k in loss_items}
                loss_score = weighted_loss_score(self.config, loss_items)
                self.intermediate = MetricsOutputs(loss_score, loss_items, is_positive)
            for callback in self.callbacks:
                callback.before_monitor_logging(self)
            # logging
            self.log_with(self.intermediate)
            # check terminate
            if self.state.should_start_snapshot:
                score = self.intermediate.final_score
                if any(monitor.should_snapshot(score) for monitor in self.monitors):
                    if self.state.can_snapshot:
                        self.state.update_snapshot_epoch()
                        save_checkpoint = True
                # should not terminate if DDP is enabled, otherwise the processes may hang
                if not is_ddp():
                    if any(m.should_terminate(score) for m in self.monitors):
                        terminate = True
        return MonitorResults(terminate, save_checkpoint, self.intermediate)


__all__ = [
    "get_scores",
    "get_sorted_checkpoints",
    "Trainer",
]
