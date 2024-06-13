import json
import math
import time
import torch
import wandb

import numpy as np

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from pathlib import Path

from ..schema import ITrainer
from ..schema import StepOutputs
from ..schema import TrainerState
from ..schema import MetricsOutputs
from ..schema import TrainerCallback
from ..toolkit import tensor_batch_to_np
from ...toolkit import console
from ...toolkit.misc import prefix_dict
from ...toolkit.misc import get_ddp_info
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


@TrainerCallback.register("nan_detector")
class NaNDetectorCallback(TrainerCallback):
    def __init__(self, check_parameters: bool = False):
        super().__init__()
        self.check_parameters = check_parameters

    def after_train_step(
        self,
        batch: tensor_dict_type,
        step_outputs: StepOutputs,
        trainer: ITrainer,
    ) -> None:
        is_nan = [k for k, v in step_outputs.loss_items.items() if math.isnan(v)]
        if self.check_parameters:
            for k, p in trainer.model.m.named_parameters():
                if torch.isnan(p).any().item():
                    is_nan.append(k)
        if is_nan:
            np_batch = tensor_batch_to_np(batch)
            nan_ratios = {k: np.isnan(v).mean().item() for k, v in np_batch.items()}
            debug_folder = Path(trainer.workspace) / "debugging"
            debug_folder.mkdir(exist_ok=True)
            ddp_info = get_ddp_info()
            appendix = "" if ddp_info is None else f"_{ddp_info.rank}"
            batch_paths = {}
            for k, v in np_batch.items():
                if isinstance(v, np.ndarray):
                    v_path = debug_folder / f"{k}{appendix}.npy"
                    batch_paths[k] = v_path
                    np.save(v_path, v)
            for k, v in step_outputs.forward_results.items():
                if isinstance(v, torch.Tensor):
                    v_path = debug_folder / f"{k}{appendix}.pt"
                    batch_paths[k] = v_path
                    torch.save(v, v_path)
            ckpt_dir = debug_folder / f"checkpoints_{trainer.accelerator.process_index}"
            ckpt_dir.mkdir(exist_ok=True)
            trainer.save_checkpoint(0.0, ckpt_dir, no_history=True, check_rank_0=False)
            losses_path = debug_folder / f"losses{appendix}.json"
            with losses_path.open("w") as f:
                json.dump(step_outputs.loss_items, f, indent=2)
            console.error(
                f"following stuffs are NaN: {sorted(is_nan)}, nan ratios of the batch "
                f"are {nan_ratios}. Current batch / states will be saved to "
                f"{batch_paths} / {ckpt_dir} for further investigation"
            )
        trainer.accelerator.wait_for_everyone()
        if is_nan:
            raise RuntimeError("NaN detected")


@TrainerCallback.register("wandb")
class WandBCallback(TrainerCallback):
    def __init__(
        self,
        project: str = "carefree-core",
        *,
        config: Optional[Dict[str, Any]] = None,
        entity: Optional[str] = None,
        save_code: Optional[bool] = None,
        group: Optional[str] = None,
        job_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        name: Optional[str] = None,
        notes: Optional[str] = None,
        relogin: Optional[bool] = None,
        anonymous: str = "allow",
        log_histograms: bool = True,
        log_artifacts: bool = False,
    ):
        super().__init__()
        self.init_kwargs = dict(
            project=project,
            config=config,
            entity=entity,
            save_code=save_code,
            group=group,
            job_type=job_type,
            tags=tags,
            name=name,
            notes=notes,
        )
        self._relogin = relogin
        self._anonymous = anonymous
        self._log_histograms = log_histograms
        self._log_artifacts = log_artifacts

    def initialize(self) -> None:
        if self.is_local_rank_0:
            wandb.login(anonymous=self._anonymous, relogin=self._relogin)
            wandb.init(**self.init_kwargs)

    def _wandb_step(self, state: TrainerState) -> int:
        step = state.last_step
        if state.is_terminate:
            step += 1
        return step

    def before_loop(self, trainer: ITrainer) -> None:
        if self.is_local_rank_0:
            self.log_artifacts(trainer)

    def log_lr(self, key: str, lr: float, state: TrainerState) -> None:
        wandb.log({key: lr}, step=self._wandb_step(state))

    def log_train_step(self, step_outputs: StepOutputs, state: TrainerState) -> None:
        if state.should_log_losses:
            wandb.log(prefix_dict(step_outputs.loss_items, "tr"), step=state.step)

    def log_metrics(
        self,
        metric_outputs: MetricsOutputs,
        state: TrainerState,
        *,
        prefix: Optional[str] = None,
    ) -> None:
        metrics = shallow_copy_dict(metric_outputs.metric_values)
        metrics["score"] = metric_outputs.final_score
        if prefix is not None:
            metrics = prefix_dict(metrics, prefix)
        wandb.log(metrics, step=self._wandb_step(state))

    def log_artifacts(self, trainer: ITrainer) -> None:
        if self._log_histograms:
            m = trainer.model.m
            hists = {
                k: wandb.Histogram(v.detach().cpu().numpy())
                for k, v in m.named_parameters()
            }
            for k, v in m.named_buffers():
                hists[k] = wandb.Histogram(v.detach().cpu().numpy())
            wandb.log(hists, step=self._wandb_step(trainer.state))
        if self._log_artifacts:
            wandb.log_artifact(trainer.workspace)

    def finalize(self, trainer: ITrainer) -> None:
        if self.is_local_rank_0:
            self.log_artifacts(trainer)
            wandb.finish()


__all__ = [
    "LogMetricsMsgCallback",
    "UpdateArtifactsCallback",
    "NaNDetectorCallback",
    "WandBCallback",
]
