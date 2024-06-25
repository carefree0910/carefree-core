import re
import math
import time
import torch

from torch import Tensor
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Optional

from ..schema import ITrainer
from ..schema import TrainerState
from ..schema import TrainStepLoss
from ..schema import MetricsOutputs
from ..schema import TrainerCallback
from ..schedulers import WarmupScheduler
from ..modules.common import EMA
from ...toolkit import console
from ...toolkit.misc import shallow_copy_dict
from ...toolkit.misc import fix_float_to_length
from ...toolkit.types import tensor_dict_type


@TrainerCallback.register("update_artifacts")
class UpdateArtifactsCallback(TrainerCallback):
    def before_loop(self, trainer: ITrainer) -> None:
        self._save(trainer, update=False)

    def after_save_checkpoint(self, trainer: ITrainer) -> None:
        self._save(trainer, update=True)

    def _save(self, trainer: ITrainer, *, update: bool) -> None:
        if trainer.config.save_pipeline_in_realtime:
            from ..pipeline import PipelineSerializer

            fn = PipelineSerializer.update if update else PipelineSerializer.save
            fn(trainer.pipeline, trainer.workspace, verbose=False)  # type: ignore


@TrainerCallback.register("log_metrics_msg")
class LogMetricsMsgCallback(TrainerCallback):
    def __init__(self, verbose: bool = True) -> None:
        super().__init__()
        self.verbose = verbose
        self.timer = time.time()
        self.metrics_log_path: Optional[str] = None

    @staticmethod
    def _step_str(state: TrainerState) -> str:
        total_step = state.num_step_per_epoch
        if state.step == -1:
            current_step = -1
        else:
            current_step = state.step % total_step
            if current_step == 0:
                current_step = total_step if state.step > 0 else 0
        length = len(str(total_step))
        return f"[{current_step:{length}d} / {total_step}]"

    def log_lr(self, key: str, lr: float, state: TrainerState) -> None:
        if self.metrics_log_path is None:
            return None
        with open(self.metrics_log_path, "a") as f:
            f.write(f" lr : {fix_float_to_length(lr, 12)} |\n")
        self.metrics_log_path = None

    def log_metrics_msg(
        self,
        metrics_outputs: MetricsOutputs,
        metrics_log_path: str,
        state: TrainerState,
        *,
        prefix: Optional[str] = None,
    ) -> None:
        final_score = metrics_outputs.final_score
        metric_values = metrics_outputs.metric_values
        core = " | ".join(
            [
                f"{k} : {fix_float_to_length(metric_values[k], 8)}"
                for k in sorted(metric_values)
            ]
        )
        step_str = self._step_str(state)
        timer_str = f"[{time.time() - self.timer:.3f}s]"
        prefix_str = "" if prefix is None else f" \['{prefix}']"
        msg = (
            f"| epoch {state.epoch:^4d} {step_str} {timer_str}{prefix_str} | {core} | "
            f"score : {fix_float_to_length(final_score, 8)} |"
        )
        if self.verbose:
            console.log(msg)
        with open(metrics_log_path, "a") as f:
            if self.metrics_log_path is not None:
                msg = f"\n{msg}"
            f.write(msg)
        self.timer = time.time()
        self.metrics_log_path = metrics_log_path


@TrainerCallback.register("training_loop")
class TrainingLoopCallback(TrainerCallback):
    """
    this callback will ALWAYS be used, it gathers many common operations in the training loop.
    although it is OK to implement these operations in the `Trainer` itself,
    we put them here to make the codes more modular.
    """

    def __init__(self) -> None:
        self.gradient_norm: Optional[Tensor] = None
        self.lr_metrics_updated = False
        self.current_scheduler_epoch = -1

    def before_summary(self, trainer: ITrainer) -> None:
        model = trainer.model
        # register `trainer` to the model
        model.init_with_trainer(trainer)
        # handle finetune stuffs
        finetune_config = trainer.config.finetune_config
        if finetune_config is not None:
            pretrained_ckpt = finetune_config.get("pretrained_ckpt")
            if pretrained_ckpt is None:
                raise ValueError(
                    "`pretrained_ckpt` should be provided when `finetune` is triggered"
                )
            console.log(f"loading pretrained checkpoint from '{pretrained_ckpt}'...")
            states = torch.load(pretrained_ckpt, map_location=trainer.device)["states"]
            model.load_state_dict(states)
            freeze = finetune_config.get("freeze", "")
            freeze_except = finetune_config.get("freeze_except", "")
            if freeze or freeze_except:
                if freeze and freeze_except:
                    raise ValueError(
                        "`freeze` & `freeze_except` should not be provided simultaneously"
                    )
                msg_fmt = f"-> {'{}'} parameter(s) will be {'{}'} under '{'{}'}'"
                param_names = []
                if freeze:
                    num_frozen = 0
                    for name, param in model.named_parameters():
                        if re.match(freeze, name):
                            num_frozen += 1
                            param.requires_grad_(False)
                            param_names.append(name)
                    msg = msg_fmt.format(num_frozen, "frozen", freeze)
                elif freeze_except:
                    num_trainable = 0
                    for name, param in model.named_parameters():
                        if not re.match(freeze_except, name):
                            param.requires_grad_(False)
                        else:
                            num_trainable += 1
                            param_names.append(name)
                    msg = msg_fmt.format(num_trainable, "trainable", freeze_except)
                console.log(
                    "\n".join(["=" * 100, msg, "-" * 100] + param_names + ["-" * 100])
                )

    def before_gradient_update(
        self,
        trainer: ITrainer,
        batch: tensor_dict_type,
        forward: tensor_dict_type,
        loss_res: TrainStepLoss,
        update: bool,
    ) -> None:
        # clip norm
        config = trainer.config
        if update and config.clip_norm > 0.0:
            if trainer.accelerator.sync_gradients:
                self.gradient_norm = trainer.accelerator.clip_grad_norm_(
                    trainer.model.parameters(),
                    max_norm=config.clip_norm,
                )

    def after_gradient_update(
        self,
        trainer: ITrainer,
        batch: tensor_dict_type,
        forward: tensor_dict_type,
        loss_tensors: tensor_dict_type,
        any_update: bool,
    ) -> None:
        state = trainer.state
        config = trainer.config
        # ema
        if any_update:
            for module in trainer.model.m.modules():
                if isinstance(module, EMA):
                    module()
        # scheduler
        if any_update and not (
            config.update_scheduler_per_epoch
            and state.epoch == self.current_scheduler_epoch
        ):
            lr_metrics_logged = False
            for k, sch in trainer.schedulers.items():
                if sch is not None:
                    should_log_lr, kwargs = self.get_scheduler_settings(k, trainer, sch)
                    if should_log_lr or config.update_scheduler_per_epoch:
                        lr_metrics_logged = True
                        if self.is_local_rank_0:
                            for callback in trainer.callbacks:
                                callback.log_lr(
                                    f"lr-{k}",
                                    sch.get_last_lr()[0],
                                    state,
                                )
                    sch.step(**shallow_copy_dict(kwargs))
            if lr_metrics_logged:
                self.lr_metrics_updated = False
            if config.update_scheduler_per_epoch:
                self.current_scheduler_epoch = state.epoch

    def before_monitor_logging(self, trainer: ITrainer) -> None:
        self.lr_metrics_updated = True

    # internal

    def get_scheduler_settings(
        self,
        key: str,
        trainer: ITrainer,
        scheduler: Any,
    ) -> Tuple[bool, Dict[str, Any]]:
        kwargs = {}
        should_log_lr = trainer.state.should_log_lr
        is_warmup = isinstance(scheduler, WarmupScheduler)
        requires_metric = key in trainer.schedulers_requires_metric
        if requires_metric and not (is_warmup and not scheduler.finished_warmup):
            if trainer.intermediate is None:
                kwargs["metrics"] = -math.inf
            else:
                kwargs["metrics"] = trainer.intermediate.final_score
            should_log_lr &= self.lr_metrics_updated
        return should_log_lr, kwargs


__all__ = [
    "LogMetricsMsgCallback",
    "UpdateArtifactsCallback",
    "TrainingLoopCallback",
]
