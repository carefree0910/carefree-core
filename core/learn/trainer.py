import os
import re
import json
import math
import torch

import torch.distributed as dist

from typing import Any
from typing import Set
from typing import Dict
from typing import List
from typing import Tuple
from typing import Callable
from typing import Optional
from accelerate import Accelerator
from tqdm.autonotebook import tqdm
from torch.optim import Optimizer
from torch.profiler import profile
from torch.optim.lr_scheduler import _LRScheduler

from .schema import device_type
from .schema import weighted_loss_score
from .schema import IData
from .schema import IModel
from .schema import IMetric
from .schema import ITrainer
from .schema import IInference
from .schema import DataLoader
from .schema import TqdmSettings
from .schema import TrainerState
from .schema import TrainerConfig
from .schema import MetricsOutputs
from .schema import MonitorResults
from .schema import TrainerMonitor
from .schema import MultipleMetrics
from .schema import TrainerCallback
from .schema import TrainStepOutputs
from .toolkit import summary
from .toolkit import get_ddp_info
from .toolkit import get_torch_device
from .constants import PT_PREFIX
from .constants import SCORES_FILE
from .constants import CHECKPOINTS_FOLDER
from .schedulers import WarmupScheduler
from ..toolkit import console
from ..toolkit.misc import safe_execute
from ..toolkit.misc import shallow_copy_dict
from ..toolkit.misc import sort_dict_by_value
from ..toolkit.misc import Incrementer
from ..toolkit.types import tensor_dict_type


T_Lo = Optional[DataLoader]


def get_scores(folder: str) -> Dict[str, float]:
    scores_path = os.path.join(folder, SCORES_FILE)
    if not os.path.isfile(scores_path):
        return {}
    with open(scores_path, "r") as f:
        return json.load(f)


def get_sorted_checkpoints(checkpoint_folder: str) -> List[str]:
    """
    better checkpoints will be placed earlier,
    which means `checkpoints[0]` is the best checkpoint
    """

    scores = get_scores(checkpoint_folder)
    if not scores:
        return []
    return list(sort_dict_by_value(scores, reverse=True).keys())


class Trainer(ITrainer):
    model_log_file = "model.txt"
    metrics_log_file = "metrics.txt"
    summary_log_file = "summary.txt"

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.tqdm_settings = safe_execute(TqdmSettings, config.tqdm_settings or {})  # type: ignore
        self.accelerator: Accelerator = None
        self._current_scheduler_epoch = -1
        self.lr_metrics_updated = False
        self.loss_incrementers: Dict[str, Incrementer] = {}
        self.intermediate: Optional[MetricsOutputs] = None
        self.final_results: Optional[MetricsOutputs] = None
        self.checkpoint_scores: Dict[str, float] = {}

    @property
    def export_config(self) -> Dict[str, Any]:
        ddp_info = get_ddp_info()
        ddp_d = None if ddp_info is None else ddp_info.asdict()
        return {
            "state_config": self.state.config,
            "valid_portion": self.config.valid_portion,
            "mixed_precision": self.config.mixed_precision,
            "clip_norm": self.config.clip_norm,
            "metrics": (
                None
                if self.metrics is None
                else (
                    self.metrics.__identifier__
                    if not isinstance(self.metrics, MultipleMetrics)
                    else [metric.__identifier__ for metric in self.metrics.metrics]
                )
            ),
            "loss_metrics_weights": self.config.loss_metrics_weights,
            "monitors": [monitor.__identifier__ for monitor in self.monitors],
            "callbacks": [callback.__identifier__ for callback in self.callbacks],
            "optimizer_settings": self.config.optimizer_settings,
            "ddp_info": ddp_d,
            "finetune_config": self.config.finetune_config,
            "tqdm_settings": self.tqdm_settings.asdict(),
        }

    @property
    def device(self) -> torch.device:
        return self.accelerator.device

    @property
    def is_local_rank_0(self) -> bool:
        return self.accelerator.is_local_main_process

    @property
    def use_tqdm_in_validation(self) -> bool:
        if not self.is_local_rank_0:
            return False
        if self.tqdm_settings.in_distributed:
            return False
        return self.tqdm_settings.use_tqdm_in_validation or self.state.is_terminate

    @property
    def has_checkpoint_folder(self) -> bool:
        if self.checkpoint_folder is None:
            return False
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

    # inheritance

    def clip_norm_step(self) -> None:
        if self.config.clip_norm > 0.0:
            if self.accelerator.sync_gradients:
                self._gradient_norm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.clip_norm,
                )

    def scheduler_step(self) -> None:
        if self.config.update_scheduler_per_epoch:
            if self.state.epoch == self._current_scheduler_epoch:
                return
        lr_metric_logged = False
        for key, scheduler in self.schedulers.items():
            if scheduler is not None:
                should_log_lr, kwargs = self._get_scheduler_settings(key, scheduler)
                if should_log_lr or self.config.update_scheduler_per_epoch:
                    lr_metric_logged = True
                    if self.is_local_rank_0:
                        for callback in self.callbacks:
                            callback.log_lr(
                                f"lr-{key}",
                                scheduler.get_last_lr()[0],
                                self.state,
                            )
                scheduler.step(**shallow_copy_dict(kwargs))
        if lr_metric_logged:
            self.lr_metrics_updated = False
        if self.config.update_scheduler_per_epoch:
            self._current_scheduler_epoch = self.state.epoch

    # api

    def fit(
        self,
        data: IData,
        model: IModel,
        metrics: Optional[IMetric],
        inference: IInference,
        optimizers: Dict[str, Optimizer],
        schedulers: Dict[str, Optional[_LRScheduler]],
        monitors: List[TrainerMonitor],
        callbacks: List[TrainerCallback],
        schedulers_requires_metric: Set[str],
        *,
        config_export_file: Optional[str] = None,
        show_summary: Optional[bool] = None,
        device: device_type = None,
        p: Optional[profile] = None,
    ) -> "Trainer":
        # accelerator
        cpu = False
        if get_ddp_info() is None:
            device = get_torch_device(device)
            if device.type == "cpu":
                cpu = True
        self.accelerator = Accelerator(
            cpu=cpu,
            split_batches=self.config.split_batches,
            mixed_precision=self.config.mixed_precision,
            dispatch_batches=self.config.dispatch_batches,
            even_batches=self.config.even_batches,
        )
        # initialize artifact structure
        if self.is_local_rank_0:
            os.makedirs(self.workspace, exist_ok=True)
            self.metrics_log_path = os.path.join(self.workspace, self.metrics_log_file)
            with open(self.metrics_log_path, "w"):
                pass
            os.makedirs(self.checkpoint_folder, exist_ok=True)
        # initialize
        self.metrics = metrics
        self.monitors = monitors
        self.callbacks = callbacks
        self.schedulers_requires_metric = schedulers_requires_metric
        if self.is_local_rank_0:
            with open(os.path.join(self.workspace, self.model_log_file), "w") as f:
                f.write(str(model))
        self.inference = inference
        # accelerator prepare
        n_optim = len(optimizers)
        optim_keys = sorted(optimizers)
        train_loader, valid_loader = data.build_loaders()
        ## 'valid_loader' is often only used at 'rank 0', so it should not be
        ## 'prepared' for distributed behaviors
        prepared = self.accelerator.prepare(
            train_loader,
            *model.all_modules,
            *[optimizers[k] for k in optim_keys],
        )
        distributed_train_loader = prepared[0]
        self.state = TrainerState(
            num_epoch=self.config.num_epoch,
            num_steps=self.config.num_steps,
            batch_size=train_loader.batch_size,
            loader_length=len(train_loader),
            **(self.config.state_config or {}),
        )
        self.model = model.from_accelerator(*prepared[1:-n_optim])
        self.optimizers = {k: prepared[-n_optim + i] for i, k in enumerate(optim_keys)}
        self.schedulers = schedulers
        for sch in schedulers.values():
            if sch is not None:
                sch.load_state_dict(sch.state_dict())
        # callback
        self.model.init_with_trainer(self)
        # finetune
        self._init_finetune()
        # verbose
        if show_summary is None:
            show_summary = not self.tqdm_settings.in_distributed
        ## should always summary to sync the statuses in distributed training
        input_sample = train_loader.get_input_sample(self.device)
        summary_msg = summary(
            self.model.m,
            input_sample,
            return_only=not show_summary or not self.is_local_rank_0,
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
        if self.is_local_rank_0 and self.epoch_tqdm is None:
            console.debug("entered training loop")
        if self.is_local_rank_0 and config_export_file is not None:
            config_export_path = os.path.join(self.workspace, config_export_file)
            with open(config_export_path, "w") as f:
                json.dump(self.export_config, f)
        for callback in self.callbacks:
            callback.before_loop(self)
        while self.state.should_train:
            try:
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
                    self.state.step += 1
                    train_stepped = self._step(i, batch)
                    for callback in self.callbacks:
                        callback.after_train_step(train_stepped, self.state)
                    monitored = self._monitor(train_loader, valid_loader, train_stepped)
                    if self.state.should_monitor:
                        for callback in self.callbacks:
                            callback.after_monitor(monitored, self.state)
                    if self.is_local_rank_0 and monitored.save_checkpoint:
                        metric_outputs = monitored.metric_outputs
                        assert metric_outputs is not None
                        self.save_checkpoint(metric_outputs.final_score)
                    terminate = monitored.terminate or self.state.should_terminate
                    if terminate:
                        break
                    if p is not None:
                        p.step()
            except KeyboardInterrupt:
                if dist.is_initialized():
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
        if self.has_checkpoint_folder:
            if not self.tqdm_settings.in_distributed:
                console.debug("rolling back to the best checkpoint")
            has_ckpt = self.restore_checkpoint()
        # finalize
        self.state.set_terminate()
        if self.is_local_rank_0:
            # should use the 'non-distributed' loaders here to avoid
            # unwanted distributed behaviors (e.g., hang when `dispatch_batches=True`)
            loader = valid_loader or train_loader
            self.final_results = self._get_metrics(loader, self.config.valid_portion)
            self._logging(self.final_results)
            if not has_ckpt:
                self.save_checkpoint(self.final_results.final_score)
        for callback in self.callbacks:
            callback.finalize(self)
        return self

    ## checkpointing

    def save_checkpoint(
        self,
        score: float,
        folder: Optional[str] = None,
        *,
        no_history: bool = False,
    ) -> None:
        if not self.is_local_rank_0:
            msg = "`save_checkpoint` should not be called when not `is_local_rank_0`"
            raise ValueError(msg)
        if folder is None:
            if self.checkpoint_folder is None:
                msg = "either `folder` or `checkpoint_folder` should be provided"
                raise ValueError(msg)
            folder = self.checkpoint_folder
        state: Optional[TrainerState] = getattr(self, "state", None)
        pt_file = f"{PT_PREFIX}{-1 if state is None else state.step}.pt"
        if state is None:
            console.warn(
                "`state` is not initialized, "
                "latest model will be saved and the recorded score will always be 0"
            )
            self.model.save(os.path.join(folder, pt_file))
            with open(os.path.join(folder, SCORES_FILE), "w") as f:
                json.dump({pt_file: 0.0}, f)
            return
        # leave top_k snapshots only
        if state.max_snapshot_file > 0:
            checkpoints = get_sorted_checkpoints(folder)
            if len(checkpoints) >= state.max_snapshot_file:
                for file in checkpoints[state.max_snapshot_file - 1 :]:
                    self.checkpoint_scores.pop(file)
                    os.remove(os.path.join(folder, file))
        # pt
        self.model.save(os.path.join(folder, pt_file))
        # scores
        scores = {} if no_history else self.checkpoint_scores
        scores[pt_file] = score
        with open(os.path.join(folder, SCORES_FILE), "w") as f:
            json.dump(sort_dict_by_value(scores, reverse=True), f)

    def restore_checkpoint(
        self,
        folder: Optional[str] = None,
        strict: bool = True,
        state_dict_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> bool:
        if folder is None:
            if self.checkpoint_folder is None:
                msg = "either `folder` or `checkpoint_folder` should be provided"
                raise ValueError(msg)
            folder = self.checkpoint_folder
        checkpoints = get_sorted_checkpoints(folder)
        if not checkpoints:
            if not self.tqdm_settings.in_distributed:
                console.warn(f"no model file found in {folder}")
            return False
        success = False
        for checkpoint in checkpoints:
            model_file = os.path.join(folder, checkpoint)
            if not os.path.isfile(model_file):
                continue
            if not self.tqdm_settings.in_distributed:
                console.debug(f"restoring from '{model_file}'")
            states = torch.load(model_file, map_location=self.device)["states"]
            if state_dict_callback is not None:
                state_dict_callback(states)
            self.model.load_state_dict(states, strict)
            success = True
            break
        return success

    # internal

    def _init_finetune(self) -> None:
        finetune_config = self.config.finetune_config
        if finetune_config is None:
            return None
        pretrained_ckpt = finetune_config.get("pretrained_ckpt")
        if pretrained_ckpt is None:
            raise ValueError("`rank` should be provided when `finetune` is triggered")
        console.log(f"loading pretrained checkpoint from '{pretrained_ckpt}'...")
        states = torch.load(pretrained_ckpt, map_location=self.device)["states"]
        self.model.load_state_dict(states)
        freeze = finetune_config.get("freeze", "")
        freeze_except = finetune_config.get("freeze_except", "")
        if not freeze and not freeze_except:
            return None
        if freeze and freeze_except:
            msg = "`freeze` & `freeze_except` should not be provided simultaneously"
            raise ValueError(msg)
        msg_fmt = f"-> {'{}'} parameter(s) will be {'{}'} under '{'{}'}'"
        param_names = []
        if freeze:
            num_frozen = 0
            for name, param in self.model.named_parameters():
                if re.match(freeze, name):
                    num_frozen += 1
                    param.requires_grad_(False)
                    param_names.append(name)
            msg = msg_fmt.format(num_frozen, "frozen", freeze)
        elif freeze_except:
            num_trainable = 0
            for name, param in self.model.named_parameters():
                if not re.match(freeze_except, name):
                    param.requires_grad_(False)
                else:
                    num_trainable += 1
                    param_names.append(name)
            msg = msg_fmt.format(num_trainable, "trainable", freeze_except)
        console.log("\n".join(["=" * 100, msg, "-" * 100] + param_names + ["-" * 100]))

    def _get_metrics(self, loader: DataLoader, portion: float = 1.0) -> MetricsOutputs:
        if self.use_tqdm_in_validation:
            loader = tqdm(
                loader,
                total=math.ceil(len(loader) * portion),
                position=self.tqdm_settings.position
                + int(self.tqdm_settings.use_tqdm)
                + int(self.tqdm_settings.use_step_tqdm),
                leave=False,
            )
        return self.model.evaluate(
            self.config,
            self.metrics,
            self.inference,
            loader,
            portion=portion,
            state=self.state,
            forward_kwargs=self.config.metric_forward_kwargs,
        )

    def _get_scheduler_settings(
        self,
        key: str,
        scheduler: Any,
    ) -> Tuple[bool, Dict[str, Any]]:
        kwargs = {}
        should_log_lr = self.state.should_log_lr
        is_warmup = isinstance(scheduler, WarmupScheduler)
        requires_metric = key in self.schedulers_requires_metric
        if requires_metric and not (is_warmup and not scheduler.finished_warmup):
            if self.intermediate is None:
                kwargs["metrics"] = -math.inf
            else:
                kwargs["metrics"] = self.intermediate.final_score
            should_log_lr &= self.lr_metrics_updated
        return should_log_lr, kwargs

    def _logging(self, metrics_outputs: MetricsOutputs) -> None:
        if not self.is_local_rank_0:
            return None
        if self.epoch_tqdm is not None:
            metric_values = shallow_copy_dict(metrics_outputs.metric_values)
            metric_values["score"] = metrics_outputs.final_score
            self.epoch_tqdm.set_postfix(metric_values)
        for callback in self.callbacks:
            callback.log_metrics(metrics_outputs, self.state)
        if self.state.should_log_artifacts:
            for callback in self.callbacks:
                callback.log_artifacts(self)
        if self.state.should_log_metrics_msg:
            for callback in self.callbacks:
                callback.log_metrics_msg(
                    metrics_outputs,
                    self.metrics_log_path,
                    self.state,
                )

    def _monitor(
        self,
        train_loader: DataLoader,
        valid_loader: T_Lo,
        step_outputs: TrainStepOutputs,
    ) -> MonitorResults:
        extension = 0
        terminate = False
        save_checkpoint = False
        for monitor in self.monitors:
            if self.state.should_extend_epoch:
                monitor.punish_extension()
                extension = max(extension, monitor.handle_extension(self.state))
        if extension:
            self.state.num_epoch += extension
        if self.config.use_incrementer_for_train_losses_in_eval:
            window = max(3, self.state.num_step_per_snapshot)
            for k, v in step_outputs.loss_dict.items():
                k_inc = self.loss_incrementers.setdefault(k, Incrementer(window))
                k_inc.update(v)
        if self.state.should_monitor:
            # get metrics
            valid_portion = self.config.valid_portion
            if valid_loader is not None:
                self.intermediate = self._get_metrics(valid_loader, valid_portion)
            elif (
                not self.config.use_incrementer_for_train_losses_in_eval
                and self.config.recompute_train_losses_in_eval
            ):
                self.intermediate = self._get_metrics(train_loader, valid_portion)
            else:
                if not self.config.use_incrementer_for_train_losses_in_eval:
                    loss_dict = shallow_copy_dict(step_outputs.loss_dict)
                else:
                    loss_dict = {
                        k: incrementer.mean
                        for k, incrementer in self.loss_incrementers.items()
                    }
                is_positive = {k: False for k in loss_dict}
                loss_score = weighted_loss_score(self.config, loss_dict)
                self.intermediate = MetricsOutputs(loss_score, loss_dict, is_positive)
            self.lr_metrics_updated = True
            # logging
            self._logging(self.intermediate)
            # check terminate
            if self.state.should_start_snapshot:
                score = self.intermediate.final_score
                if any(monitor.should_snapshot(score) for monitor in self.monitors):
                    if self.state.can_snapshot:
                        self.state.update_snapshot_epoch()
                        save_checkpoint = True
                # should not terminate if DDP is enabled, otherwise the processes may hang
                if get_ddp_info() is None:
                    if any(m.should_terminate(score) for m in self.monitors):
                        terminate = True
        return MonitorResults(terminate, save_checkpoint, self.intermediate)

    def _step(self, batch_idx: int, batch: tensor_dict_type) -> TrainStepOutputs:
        forward_kw: Dict[str, Any] = {}
        for callback in self.callbacks:
            callback.mutate_forward_kwargs(forward_kw, self)
        loss_kw: Dict[str, Any] = {}
        for callback in self.callbacks:
            callback.mutate_loss_kwargs(loss_kw, self)
        return self.model.train(batch_idx, batch, self, forward_kw, loss_kw)


__all__ = [
    "get_scores",
    "get_sorted_checkpoints",
    "Trainer",
]
