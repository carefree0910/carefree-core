import math
import time
import wandb
import shutil

import numpy as np

from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import NamedTuple
from datetime import datetime
from rich import box
from rich.style import Style
from rich.table import Table
from rich.progress import Task
from rich.progress import TaskID
from rich.progress import Progress
from rich.progress import TextColumn

from ..schema import ITrainer
from ..schema import DataLoader
from ..schema import StepOutputs
from ..schema import TqdmSettings
from ..schema import TrainerState
from ..schema import MetricsOutputs
from ..schema import TrainerCallback
from ..constants import INFERENCE_COLOR
from ...toolkit import console
from ...toolkit.misc import prefix_dict
from ...toolkit.misc import format_float
from ...toolkit.misc import make_progress
from ...toolkit.misc import shallow_copy_dict
from ...toolkit.misc import fix_float_to_length
from ...toolkit.misc import get_console_datetime
from ...toolkit.types import tensor_dict_type
from ...toolkit.console import LOG_TIME_FORMAT


class MetricsFormatter:
    @staticmethod
    def format(task: Task) -> str:
        metrics = task.fields.get("metrics")
        if metrics is None:
            return ""
        return (
            f"[{INFERENCE_COLOR}]| "
            + " | ".join([f"{k}: {format_float(v)}" for k, v in metrics.items()])
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
        self.enabled = False
        self.progress: Optional[Progress] = None
        self.time_column: Optional[TextColumn] = None
        self.step_progress: Optional[TaskID] = None
        self.epoch_progress: Optional[TaskID] = None
        self.settings = TqdmSettings(**settings)

    def init(self) -> None:
        self.progress = make_progress(
            use_spinner=True,
            custom_columns=[TextColumn(MetricsFormatter)],  # type: ignore
        )
        self.time_column: TextColumn = self.progress.columns[0]  # type: ignore
        self.enabled = self.is_local_rank_0 and (
            self.settings.use_tqdm
            or self.settings.use_step_tqdm
            or self.settings.use_tqdm_in_validation
        )
        if self.enabled:
            self.progress.start()

    def before_loop(self, trainer: ITrainer) -> None:
        if self.is_local_rank_0:
            self.init()
            if self.settings.use_tqdm and self.progress is not None:
                self._update_time_column()
                self.epoch_progress = self.progress.add_task(
                    f"[green]{self.settings.desc}",
                    total=trainer.state.num_epoch,
                    start=True,
                )
                self.progress.update(self.epoch_progress, completed=trainer.state.epoch)

    def at_epoch_start(self, trainer: ITrainer, train_loader: DataLoader) -> None:
        num_steps = len(train_loader)
        if (
            self.is_local_rank_0
            and self.settings.use_step_tqdm
            and self.progress is not None
        ):
            self.step_progress = self.progress.add_task(
                "[cyan]running step",
                total=num_steps,
                start=True,
            )
        if self.epoch_progress is None:
            self._update_time_column()

    def after_train_step(
        self,
        batch: tensor_dict_type,
        step_outputs: StepOutputs,
        trainer: ITrainer,
    ) -> None:
        if self.progress is not None and self.step_progress is not None:
            self.progress.update(self.step_progress, advance=1)

    def log_metrics_msg(
        self,
        trainer: ITrainer,
        metrics_outputs: MetricsOutputs,
    ) -> None:
        if self.progress is not None and self.epoch_progress is not None:
            metric_values = shallow_copy_dict(metrics_outputs.metric_values)
            metric_values["score"] = metrics_outputs.final_score
            self.progress.update(self.epoch_progress, metrics=metric_values)  # type: ignore

    def at_epoch_end(self, trainer: ITrainer) -> None:
        if self.progress is not None and self.epoch_progress is not None:
            self.progress.update(
                self.epoch_progress,
                total=trainer.state.num_epoch,
                advance=1,
            )
        if self.progress is not None and self.step_progress is not None:
            self.progress.remove_task(self.step_progress)

    def finalize(self, trainer: ITrainer) -> None:
        if self.enabled and self.progress is not None:
            self.progress.stop()

    def _update_time_column(self) -> None:
        if self.time_column is not None:
            self.time_column.text_format = datetime.now().strftime(LOG_TIME_FORMAT)


class AutoWrapLine:
    def __init__(self, **table_kw: Any) -> None:
        self._table_kw = table_kw
        self._col_names: List[str] = []
        self._col_kwargs: List[Dict[str, Any]] = []
        self._row: Optional[Any] = None

    def add_column(self, name: str, **kwargs: Any) -> None:
        self._col_names.append(name)
        self._col_kwargs.append(kwargs)

    def add_row(self, *row: Any) -> None:
        if self._row is not None:
            raise RuntimeError("should not add row more than once")
        self._row = row

    def get_table(self) -> Table:
        if self._row is None:
            raise RuntimeError("should add row before getting table")
        row = get_console_datetime(), *self._row
        col_names = [""] + self._col_names
        col_kwargs = [dict(style="log.time")] + self._col_kwargs

        terminal_w = shutil.get_terminal_size().columns
        pad_tw = lambda nc: terminal_w - 2 * nc
        cell_widths = np.array(
            [
                max(len(str(row[i])), len(col_name)) + 2
                for i, col_name in enumerate(col_names)
            ]
        )
        ws_cumsum = np.cumsum(cell_widths)
        # if total width is less than terminal width (with 1 pixel padding
        # for each column), no need to wrap
        num_total_cols = len(col_names)
        cumsum_mask = ws_cumsum > pad_tw(num_total_cols)
        wrap_index = np.argmax(cumsum_mask).item()
        if wrap_index == 0 and not cumsum_mask[0]:
            table = Table(**self._table_kw)
            for name, kwargs in zip(col_names, col_kwargs):
                table.add_column(name, **kwargs)
            table.add_row(*row)
            return table
        # iteratively decide how many columns can we have, until only 1 column left
        num_wrapped_rows = 2
        while True:
            num_wrapped_cols = math.ceil(num_total_cols / num_wrapped_rows)
            remainder = num_total_cols % num_wrapped_cols
            padding = num_wrapped_cols - remainder
            padded_cell_widths = np.concatenate([cell_widths, np.zeros(padding)])
            padded_cell_widths = padded_cell_widths.reshape(-1, num_wrapped_cols)
            padded_row_widths = padded_cell_widths.sum(axis=1)
            if padded_row_widths.max() <= pad_tw(num_wrapped_cols):
                break
            num_wrapped_rows += 1
        # add multiple tables
        table = Table(**self._table_kw, show_header=False)
        for i in range(num_wrapped_rows):
            it = Table(**self._table_kw)
            for j in range(num_wrapped_cols):
                idx = i * num_wrapped_cols + j
                if idx >= num_total_cols:
                    break
                name = col_names[idx]
                kwargs = col_kwargs[idx]
                it.add_column(name, **kwargs)
            i_row = row[i * num_wrapped_cols : (i + 1) * num_wrapped_cols]
            it.add_row(*i_row)
            table.add_row(it)
        return table


class Metrics(NamedTuple):
    t: float
    state: TrainerState
    recorded_lr: Optional[float]
    metrics_outputs: MetricsOutputs

    def to_msg(self) -> str:
        final_score = self.metrics_outputs.final_score
        metric_values = self.metrics_outputs.metric_values
        core = " | ".join(
            [
                f"{k} : {fix_float_to_length(metric_values[k], 10)}"
                for k in sorted(metric_values)
            ]
        )
        total_step = self.state.num_step_per_epoch
        if self.state.step == -1:
            current_step = -1
        else:
            current_step = self.state.step % total_step
            if current_step == 0:
                current_step = total_step if self.state.step > 0 else 0
        length = len(str(total_step))
        step_str = f"[{current_step:{length}d} / {total_step}]"
        timer_str = f"[{self.t:.3f}s]"
        msg = (
            f"| epoch {self.state.epoch:^4d} {step_str} {timer_str} | {core} | "
            f"score : {fix_float_to_length(final_score, 10)} |"
        )
        if self.recorded_lr is not None:
            msg += f" lr : {fix_float_to_length(self.recorded_lr, 12)} |"
        return msg

    def verbose(self) -> None:
        columns_kw = dict(justify="center", no_wrap=True)
        inference_kw = dict(style=Style(color=INFERENCE_COLOR, bold=True))
        metrics_kw = dict(
            style=Style(color="turquoise2"),
            header_style=Style(bold=False, color="sea_green1"),
        )

        final_score = self.metrics_outputs.final_score
        metric_values = self.metrics_outputs.metric_values
        line = AutoWrapLine(box=box.SIMPLE_HEAD)
        line.add_column("epoch", **columns_kw, **inference_kw)
        line.add_column("step", **columns_kw, **inference_kw)
        line.add_column("elapsed", **columns_kw, style=Style(color="bright_magenta"))
        sorted_keys = sorted(metric_values)
        for k in sorted_keys:
            line.add_column(k, **columns_kw, **metrics_kw)
        line.add_column("score", **columns_kw, **inference_kw)
        row = [
            str(self.state.epoch),
            str(self.state.step),
            f"{self.t:.4f}",
            *[format_float(metric_values[k], 4) for k in sorted_keys],
            format_float(final_score, 4),
        ]
        if self.recorded_lr is not None:
            line.add_column("lr", **columns_kw, **inference_kw)
            row.append(format_float(self.recorded_lr))
        line.add_row(*row)
        console.print(line.get_table())


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

    def log_lr(self, key: str, lr: float, trainer: "ITrainer") -> None:
        self.recorded_lr = lr

    def log_metrics_msg(
        self,
        trainer: ITrainer,
        metrics_outputs: MetricsOutputs,
    ) -> None:
        metrics = Metrics(
            time.time() - self.timer,
            trainer.state,
            self.recorded_lr,
            metrics_outputs,
        )
        if self.verbose:
            metrics.verbose()
        with open(trainer.metrics_log_path, "a") as f:
            if self.logged:
                f.write("\n")
            f.write(metrics.to_msg())
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
