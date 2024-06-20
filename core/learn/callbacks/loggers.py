import wandb

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from ..schema import ITrainer
from ..schema import StepOutputs
from ..schema import TrainerState
from ..schema import MetricsOutputs
from ..schema import TrainerCallback
from ...toolkit.misc import prefix_dict
from ...toolkit.misc import shallow_copy_dict


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
    "WandBCallback",
]
