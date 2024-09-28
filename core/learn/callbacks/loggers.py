import time
import wandb

from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from datetime import datetime
from rich.table import Column
from rich.progress import Task
from rich.progress import TaskID
from rich.progress import Progress
from rich.progress import BarColumn
from rich.progress import TextColumn
from rich.progress import SpinnerColumn
from rich.progress import TimeRemainingColumn

from ..schema import ITrainer
from ..schema import DataLoader
from ..schema import StepOutputs
from ..schema import TqdmSettings
from ..schema import TrainerState
from ..schema import MetricsOutputs
from ..schema import TrainerCallback
from ...toolkit import console
from ...toolkit.misc import prefix_dict
from ...toolkit.misc import format_float
from ...toolkit.misc import shallow_copy_dict
from ...toolkit.misc import fix_float_to_length
from ...toolkit.console import LOG_TIME_FORMAT


class MetricsFormatter:
    @staticmethod
    def format(task: Task) -> str:
        fields = task.fields
        if not fields:
            return ""
        return (
            "[yellow]| "
            + " | ".join([f"{k}: {format_float(v)}" for k, v in fields.items()])
            + " |"
        )


@TrainerCallback.register("progress")
class ProgressCallback(TrainerCallback):
    """
    we use the `TqdmSettings` for BC, since previously this project
    is built on top of the `tqdm` package for progress bar rendering.
    """

    def __init__(self, settings: Dict[str, Any]) -> None:
        super().__init__()
        self.settings = TqdmSettings(**settings)
        self.time_column = TextColumn("", "log.time")
        self.progress_table = Column()
        self.progress = Progress(
            self.time_column,
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn(
                "[progress.percentage]{task.completed}/{task.total}",
                table_column=self.progress_table,
            ),
            TimeRemainingColumn(),
            TextColumn(MetricsFormatter),  # type: ignore
        )
        self.step_progress: Optional[TaskID] = None
        self.epoch_progress: Optional[TaskID] = None
        self.enabled = self.is_local_rank_0 and (
            self.settings.use_tqdm or self.settings.use_step_tqdm
        )
        if self.enabled:
            self.progress.start()

    def before_loop(self, trainer: ITrainer) -> None:
        if self.is_local_rank_0 and self.settings.use_tqdm:
            now = datetime.now().strftime(LOG_TIME_FORMAT)
            self.time_column.text_format = now
            self.epoch_progress = self.progress.add_task(
                f"[green]{self.settings.desc}",
                total=trainer.state.num_epoch,
                start=True,
            )

    def at_epoch_start(self, trainer: ITrainer, train_loader: DataLoader) -> None:
        num_steps = len(train_loader)
        self.progress_table.width = len(str(num_steps)) * 2 + 1
        if self.is_local_rank_0 and self.settings.use_step_tqdm:
            self.step_progress = self.progress.add_task(
                "[cyan]running step",
                total=num_steps,
                start=True,
            )

    def at_step_end(self, trainer: ITrainer) -> None:
        if self.step_progress is not None:
            self.progress.update(self.step_progress, advance=1)

    def log_metrics_msg(
        self,
        trainer: ITrainer,
        metrics_outputs: MetricsOutputs,
    ) -> None:
        if self.epoch_progress is not None:
            metric_values = shallow_copy_dict(metrics_outputs.metric_values)
            metric_values["score"] = metrics_outputs.final_score
            self.progress.update(self.epoch_progress, **metric_values)  # type: ignore

    def at_epoch_end(self, trainer: ITrainer) -> None:
        if self.epoch_progress is not None:
            self.progress.update(
                self.epoch_progress,
                total=trainer.state.num_epoch,
                advance=1,
            )
        if self.step_progress is not None:
            self.progress.remove_task(self.step_progress)

    def after_loop(self, trainer: ITrainer) -> None:
        if self.enabled:
            self.progress.stop()


@TrainerCallback.register("log_metrics_msg")
class LogMetricsMsgCallback(TrainerCallback):
    """
    this is also a default callback, it logs the metrics messages to the log file
    (and to the console, if `verbose` is True).
    """

    def __init__(self, verbose: bool = True) -> None:
        super().__init__()
        self.verbose = verbose
        self.timer = time.time()
        self.logged = False
        self.recorded_lr: Optional[float] = None

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

    def log_lr(self, key: str, lr: float, trainer: "ITrainer") -> None:
        self.recorded_lr = lr

    def log_metrics_msg(
        self,
        trainer: ITrainer,
        metrics_outputs: MetricsOutputs,
    ) -> None:
        state = trainer.state
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
        msg = (
            f"| epoch {state.epoch:^4d} {step_str} {timer_str} | {core} | "
            f"score : {fix_float_to_length(final_score, 8)} |"
        )
        if self.recorded_lr is not None:
            msg += f" lr : {fix_float_to_length(self.recorded_lr, 12)} |"
        if self.verbose:
            console.log(msg)
        with open(trainer.metrics_log_path, "a") as f:
            if self.logged:
                f.write("\n")
            f.write(msg)
        self.timer = time.time()
        self.logged = True


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
        anonymous: Literal["must", "allow", "never"] = "allow",
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
            wandb.init(**self.init_kwargs)  # type: ignore

    def _wandb_step(self, state: TrainerState) -> int:
        step = state.last_step
        if state.is_terminate:
            step += 1
        return step

    def before_loop(self, trainer: ITrainer) -> None:
        if self.is_local_rank_0:
            self.log_artifacts(trainer)

    def log_lr(self, key: str, lr: float, trainer: "ITrainer") -> None:
        wandb.log({key: lr}, step=self._wandb_step(trainer.state))

    def log_train_step(self, step_outputs: StepOutputs, state: TrainerState) -> None:
        if state.should_log_losses:
            wandb.log(prefix_dict(step_outputs.loss_items, "tr"), step=state.step)

    def log_metrics(self, metric_outputs: MetricsOutputs, state: TrainerState) -> None:
        metrics = shallow_copy_dict(metric_outputs.metric_values)
        metrics["score"] = metric_outputs.final_score
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
    "ProgressCallback",
    "LogMetricsMsgCallback",
    "WandBCallback",
]
