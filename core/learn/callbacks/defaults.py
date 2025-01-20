import re
import math
import torch

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional
from accelerate.utils import broadcast_object_list

from ..schema import ITrainer
from ..schema import DataLoader
from ..schema import TrainStepLoss
from ..schema import TrainerCallback
from ..schedulers import WarmupScheduler
from ..modules.common import EMA
from ...toolkit import console
from ...toolkit.misc import is_ddp
from ...toolkit.misc import shallow_copy_dict
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


@TrainerCallback.register("training_loop")
class TrainingLoopCallback(TrainerCallback):
    """
    this callback will ALWAYS be used, it gathers many common operations in the training loop.
    although it is OK to implement these operations in the `Trainer` itself,
    we put them here to make the codes more modular.
    """

    def __init__(self) -> None:
        self.gradient_norm: Optional[Tensor] = None
        self.should_log_lr = False
        self.current_scheduler_epoch = -1

    def before_summary(self, trainer: ITrainer) -> None:
        model = trainer.model
        # register `trainer` to the model
        model.init_with_trainer(trainer)
        # handle finetune stuffs
        finetune_config = trainer.config.finetune_config
        if finetune_config is not None:
            ckpt = finetune_config.get("pretrained_ckpt")
            device = trainer.device
            if ckpt is None:
                raise ValueError(
                    "`pretrained_ckpt` should be provided when `finetune` is triggered"
                )
            if self.is_local_rank_0:
                console.log(f"loading pretrained checkpoint from '{ckpt}'...")
            full_states = torch.load(ckpt, weights_only=False, map_location=device)
            states: tensor_dict_type = full_states["states"]
            exclude = finetune_config.get("exclude", "")
            exclude_names: List[str]
            if not exclude:
                exclude_names = []
                model.load_state_dict(states)
            else:
                exclude_names = []
                for name in states.keys():
                    if re.match(exclude, name):
                        exclude_names.append(name)
                if not exclude_names:
                    if self.is_local_rank_0:
                        console.warn(
                            f"`exclude` pattern '{exclude}' does not match any "
                            "parameter, please make sure this is as expected!"
                        )
                    strict = finetune_config.get("strict", True)
                else:
                    if self.is_local_rank_0:
                        console.log(
                            f"{len(exclude_names)} parameters will be excluded:",
                            exclude_names,
                        )
                    states = {k: v for k, v in states.items() if k not in exclude_names}
                    strict = finetune_config.get("strict", False)
                model.load_state_dict(states, strict=strict)
            freeze = finetune_config.get("freeze", "")
            freeze_except = finetune_config.get("freeze_except", "")
            if freeze or freeze_except:
                if freeze and freeze_except:
                    raise ValueError(
                        "`freeze` & `freeze_except` should not be provided simultaneously"
                    )
                msg_fmt = f"{'{}'} parameter(s) will be {'{}'} under '{'{}'}':"
                param_names = []
                is_ddp_mode = is_ddp()
                if freeze:
                    num_frozen = 0
                    for name, param in model.named_parameters():
                        if is_ddp_mode:
                            name = name[len("module.") :]
                        if re.match(freeze, name):
                            num_frozen += 1
                            param.requires_grad_(False)
                            param_names.append(name)
                    msg = msg_fmt.format(num_frozen, "frozen", freeze)
                elif freeze_except:
                    num_trainable = 0
                    for name, param in model.named_parameters():
                        if is_ddp_mode:
                            name = name[len("module.") :]
                        if not re.match(freeze_except, name):
                            param.requires_grad_(False)
                        else:
                            num_trainable += 1
                            param_names.append(name)
                    msg = msg_fmt.format(num_trainable, "trainable", freeze_except)
                if self.is_local_rank_0:
                    console.log(msg, param_names)

    def before_loop_with_loaders(
        self,
        trainer: ITrainer,
        train_loader: DataLoader,
        valid_loader: Optional[DataLoader],
    ) -> None:
        if trainer.config.resume_training_from is None:
            return None
        if valid_loader is not None:
            loader = valid_loader
        else:  # pragma: no cover
            answer = None
            if self.is_local_rank_0:
                answer = console.ask(
                    "no validation loader found, do you want to calculate resumed-metrics from the training loader?",
                    ["y", "n"],
                    default="n",
                )
            answer = broadcast_object_list([answer])[0]
            if answer == "n":
                return None
            loader = train_loader
        resumed_results = trainer.get_metrics(loader, trainer.config.valid_portion)
        if self.is_local_rank_0:
            console.log("resumed metrics:", resumed_results)

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
            lr_logged = False
            for k, sch in trainer.schedulers.items():
                if sch is not None:
                    should_log_lr, kwargs = self.get_scheduler_settings(k, trainer, sch)
                    if should_log_lr or config.update_scheduler_per_epoch:
                        lr_logged = True
                        if self.is_local_rank_0:
                            for callback in trainer.callbacks:
                                callback.log_lr(
                                    f"lr-{k}",
                                    sch.get_last_lr()[0],
                                    trainer,
                                )
                    sch.step(**shallow_copy_dict(kwargs))
            if lr_logged:
                self.should_log_lr = False
            if config.update_scheduler_per_epoch:
                self.current_scheduler_epoch = state.epoch

    def before_monitor_logging(self, trainer: ITrainer) -> None:
        self.should_log_lr = True

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
            should_log_lr &= self.should_log_lr
        return should_log_lr, kwargs


__all__ = [
    "UpdateArtifactsCallback",
    "TrainingLoopCallback",
]
