import os
import json
import math
import torch
import shutil
import inspect

from torch import nn
from typing import Any
from typing import Set
from typing import Dict
from typing import List
from typing import Type
from typing import Tuple
from typing import Optional
from typing import OrderedDict as OrderedDictType
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass
from torch.optim import Optimizer
from torch.profiler import profile
from torch.profiler import schedule
from accelerate.utils import gather_object
from accelerate.utils import wait_for_everyone
from torch.optim.lr_scheduler import _LRScheduler

from .utils import TryLoadBlock
from .utils import InjectDefaultsMixin
from ..common import Block
from ...schema import device_type
from ...schema import trainer_callbacks
from ...schema import IData
from ...schema import IModel
from ...schema import Config
from ...schema import IMetric
from ...schema import ITrainer
from ...schema import IInference
from ...schema import OptimizerPack
from ...schema import TrainerMonitor
from ...schema import TrainerCallback
from ...losses import losses
from ...models import EnsembleModel
from ...toolkit import _get_environ_workspace
from ...toolkit import scheduler_requires_metric
from ...trainer import get_scores
from ...trainer import get_sorted_checkpoints
from ...trainer import Trainer
from ...monitors import BasicMonitor
from ...callbacks import NaNDetectorCallback
from ...callbacks import LogMetricsMsgCallback
from ...callbacks import UpdateArtifactsCallback
from ...constants import PT_PREFIX
from ...constants import SCORES_FILE
from ...constants import CHECKPOINTS_FOLDER
from ...inference import Inference
from ...optimizers import optimizer_dict
from ...schedulers import scheduler_dict
from ...schedulers import WarmupScheduler
from ....toolkit import console
from ....toolkit.misc import to_path
from ....toolkit.misc import filter_kw
from ....toolkit.misc import update_dict
from ....toolkit.misc import shallow_copy_dict
from ....toolkit.misc import sort_dict_by_value
from ....toolkit.misc import prepare_workspace_from
from ....toolkit.misc import truncate_string_to_length
from ....toolkit.misc import Serializer
from ....toolkit.misc import DataClassBase
from ....toolkit.types import TPath


# static blocks


@Block.register("set_defaults")
class SetDefaultsBlock(InjectDefaultsMixin, Block):
    def build(self, config: Config) -> None:
        loss_name = config.loss_name
        module_name = config.module_name
        state_config = config.state_config
        callback_names = config.callback_names
        if loss_name is None:
            if losses.has(module_name):
                loss_name = module_name
                self._defaults["loss_name"] = loss_name
        if state_config is None:
            state_config = {}
        if "max_snapshot_file" not in state_config:
            state_config["max_snapshot_file"] = 25
            self._defaults["max_snapshot_file"] = 25
        if callback_names is None:
            if module_name in trainer_callbacks:
                callback_names = module_name
                self._defaults["callback_names"] = callback_names
        environ_workspace = _get_environ_workspace()
        if environ_workspace:
            config.workspace = environ_workspace
        config.loss_name = loss_name
        config.module_name = module_name
        config.state_config = state_config
        config.callback_names = callback_names
        torch.backends.cudnn.benchmark = config.cudnn_benchmark
        # tqdm settings
        tqdm_settings = config.tqdm_settings
        if tqdm_settings is None:
            tqdm_settings = {}
        use_tqdm = tqdm_settings.setdefault("use_tqdm", False)
        tqdm_settings.setdefault("use_step_tqdm", use_tqdm)
        tqdm_settings.setdefault("use_tqdm_in_validation", False)
        tqdm_settings.setdefault("tqdm_position", 0)
        tqdm_settings.setdefault("tqdm_desc", "epoch")
        config.tqdm_settings = tqdm_settings


@Block.register("prepare_workspace")
class PrepareWorkplaceBlock(InjectDefaultsMixin, Block):
    def build(self, config: Config) -> None:
        if self.training_workspace is None:
            return
        if self.is_local_rank_0 and config.create_sub_workspace:
            workspace = prepare_workspace_from(self.training_workspace)
            config.workspace = workspace
            self._defaults["workspace"] = workspace
        wait_for_everyone()
        workspaces = gather_object([config.workspace])
        if not self.is_local_rank_0:
            # use the workspace from local rank 0
            config.workspace = workspaces[0]


@dataclass
class StateInfo(DataClassBase):
    batch_size: int
    num_batches: int
    num_samples: int
    snapshot_start_step: int
    num_step_per_snapshot: int


@Block.register("extract_state_info")
class ExtractStateInfoBlock(TryLoadBlock):
    config: Config
    state_info: StateInfo

    def try_load(self, folder: TPath) -> bool:
        info = Serializer.try_load_info(folder)
        if info is None:
            return False
        self.state_info = StateInfo(**info)
        return True

    def from_scratch(self, config: Config) -> None:
        if self.data is None:
            raise ValueError(f"`data` should be provided for `ExtractStateInfoBlock`")
        # from loader
        loader = self.data.build_loaders()[0]
        batch_size = loader.batch_size
        num_batches = len(loader)
        num_samples = len(loader.dataset)
        # from config
        log_steps = config.log_steps
        state_config = config.state_config or {}
        # check log_steps
        if log_steps is not None:
            state_config.setdefault("snapshot_start_step", log_steps)
            state_config.setdefault("num_step_per_snapshot", log_steps)
        # check snapshot_start_step
        snapshot_start_step = state_config.get("snapshot_start_step")
        if snapshot_start_step is None:
            min_num_sample = state_config.get("min_num_sample", 3000)
            snapshot_start_step = math.ceil(min_num_sample / batch_size)
        # check num_step_per_snapshot
        num_step_per_snapshot = state_config.get("num_step_per_snapshot")
        if num_step_per_snapshot is None:
            num_snapshot_per_epoch = state_config.get("num_snapshot_per_epoch", 2)
            max_step_per_snapshot = state_config.get("max_step_per_snapshot", 1000)
            num_step_per_snapshot = max(1, int(num_batches / num_snapshot_per_epoch))
            num_step_per_snapshot = min(
                max_step_per_snapshot,
                num_step_per_snapshot,
            )
        # construct
        state_config["num_step_per_snapshot"] = num_step_per_snapshot
        state_config["snapshot_start_step"] = snapshot_start_step
        self.state_info = StateInfo(
            batch_size=batch_size,
            num_batches=num_batches,
            num_samples=num_samples,
            snapshot_start_step=snapshot_start_step,
            num_step_per_snapshot=num_step_per_snapshot,
        )
        config.state_config = state_config

    def dump_to(self, folder: TPath) -> None:
        if self.is_local_rank_0:
            Serializer.save_info(folder, info=self.state_info.asdict())


@Block.register("build_model")
class BuildModelBlock(Block):
    model: IModel

    def build(self, config: Config) -> None:
        num_repeat = config.num_repeat
        m = IModel.from_config(config)
        if num_repeat is None:
            self.model = m
        else:
            self.model = EnsembleModel(m, num_repeat)


@Block.register("build_metrics")
class BuildMetricsBlock(Block):
    metrics: Optional[IMetric]

    def build(self, config: Config) -> None:
        # build metrics
        metric_names = config.metric_names
        metric_configs = config.metric_configs
        metric_weights = config.metric_weights
        if metric_names is None:
            self.metrics = None
        else:
            self.metrics = IMetric.fuse(
                metric_names,
                metric_configs,
                metric_weights=metric_weights,
            )
        # check losses-as-metrics
        loss_metrics_weights = config.loss_metrics_weights
        use_losses_as_metrics = config.use_losses_as_metrics
        if self.metrics is None:
            if use_losses_as_metrics is None:
                use_losses_as_metrics = True
            if not use_losses_as_metrics:
                msg = "`metrics` should be provided when not `use_losses_as_metrics`"
                raise ValueError(msg)
        if loss_metrics_weights is not None:
            if use_losses_as_metrics is None:
                use_losses_as_metrics = True
            elif not use_losses_as_metrics:
                raise ValueError(
                    "`use_losses_as_metrics` should not be False "
                    "when `loss_metrics_weights` is provided"
                )
        config.use_losses_as_metrics = use_losses_as_metrics


@Block.register("build_inference")
class BuildInferenceBlock(Block):
    inference: IInference

    def build(self, config: Config) -> None:
        self.inference = Inference(model=self.build_model.model)

    @property
    def requirements(self) -> List[Type[Block]]:
        return [BuildModelBlock]

    @property
    def build_model(self) -> BuildModelBlock:
        return self.get_previous(BuildModelBlock)


@Block.register("set_trainer_defaults")
class SetTrainerDefaultsBlock(InjectDefaultsMixin, Block):
    def build(self, config: Config) -> None:
        # set some trainer defaults to deep learning tasks which work well in practice
        if config.monitor_names is None:
            config.monitor_names = "basic"
            self._defaults["monitor_names"] = "basic"
        tqdm_settings = config.tqdm_settings
        callback_names = config.callback_names
        callback_configs = config.callback_configs
        if callback_names is None:
            callback_names = []
        if callback_configs is None:
            callback_configs = {}
        if isinstance(callback_names, str):
            callback_names = [callback_names]
        auto_callback = config.auto_callback
        default_callbacks = [
            NaNDetectorCallback.__identifier__,
            LogMetricsMsgCallback.__identifier__,
            UpdateArtifactsCallback.__identifier__,
        ]
        if any(c not in callback_names for c in default_callbacks) and auto_callback:
            additional_callbacks = []
            for c in default_callbacks:
                if c not in callback_names:
                    additional_callbacks.append(c)
                    callback_names.insert(0, c)
            self._defaults["additional_callbacks"] = additional_callbacks
            verbose = False
            if tqdm_settings is None or (
                not tqdm_settings.get("use_tqdm", False)
                and not tqdm_settings.get("use_step_tqdm", False)
            ):
                verbose = True
            log_metrics_msg_cfg = callback_configs.setdefault(default_callbacks[0], {})
            if "verbose" not in log_metrics_msg_cfg:
                log_metrics_msg_cfg["verbose"] = verbose
                self._defaults["log_metrics_msg_verbose"] = verbose
        if "wandb" in callback_names and auto_callback:
            module_name = config.module_name
            wandb_config = callback_configs.setdefault("wandb", {})
            if "tags" not in wandb_config:
                tags_str = json.dumps([module_name])
                wandb_config["tags"] = [module_name]
                self._defaults["callback_configs.wandb.tags"] = tags_str
            if "config" not in wandb_config:
                module_configs = shallow_copy_dict(config.module_config or {})
                config_str = json.dumps(module_configs)
                wandb_config["config"] = module_configs
                self._defaults["callback_configs.wandb.config"] = config_str
        config.tqdm_settings = tqdm_settings
        config.callback_names = callback_names
        config.callback_configs = callback_configs


@Block.register("build_monitors")
class BuildMonitorsBlock(Block):
    monitors: List[TrainerMonitor]

    def build(self, config: Config) -> None:
        monitor_names = config.monitor_names
        monitor_configs = config.monitor_configs
        if isinstance(monitor_names, str):
            monitor_names = [monitor_names]
        if monitor_names is None:
            self.monitors = [BasicMonitor()]
        else:
            self.monitors = TrainerMonitor.make_multiple(monitor_names, monitor_configs)


@Block.register("build_callbacks")
class BuildCallbacksBlock(Block):
    callbacks: List[TrainerCallback]

    def build(self, config: Config) -> None:
        cb_names = config.callback_names
        cb_configs = config.callback_configs
        use_tqdm = (config.tqdm_settings or {}).get("use_tqdm", False)
        if cb_names is not None:
            self.callbacks = TrainerCallback.make_multiple(cb_names, cb_configs)
        else:
            self.callbacks = [
                LogMetricsMsgCallback(not use_tqdm),
                UpdateArtifactsCallback(),
            ]
        for callback in self.callbacks:
            callback.initialize()


@dataclass
class OptimizerSettings(DataClassBase):
    lr: float = 1.0e-3
    optimizer_name: str = "adam"
    scheduler_name: Optional[str] = None
    optimizer_config: Optional[Dict[str, Any]] = None
    scheduler_config: Optional[Dict[str, Any]] = None

    def get_opt_pack(self, state_info: Optional[StateInfo]) -> OptimizerPack:
        optimizer_config = shallow_copy_dict(self.optimizer_config or {})
        scheduler_config = shallow_copy_dict(self.scheduler_config or {})
        if self.scheduler_name != "warmup":
            optimizer_config.setdefault("lr", self.lr)
        else:
            multiplier = scheduler_config.setdefault("multiplier", 3)
            optimizer_config.setdefault("lr", self.lr / multiplier)
            if state_info is None:
                scheduler_config.setdefault("warmup_step", 1000)
            else:
                default_max_warmup_step = int(round(3.0e5 / state_info.batch_size))
                scheduler_config.setdefault(
                    "warmup_step",
                    min(default_max_warmup_step, 10 * state_info.num_batches),
                )
        return OptimizerPack(
            "all",
            self.optimizer_name,
            self.scheduler_name,
            optimizer_config,
            scheduler_config,
        )

    def update_opt_pack(
        self,
        state_info: Optional[StateInfo],
        pack: OptimizerPack,
    ) -> OptimizerPack:
        self_pack = self.get_opt_pack(state_info)
        opt_name = pack.optimizer_name
        sch_name = pack.scheduler_name
        opt_config = shallow_copy_dict(pack.optimizer_config or {})
        sch_config = shallow_copy_dict(pack.scheduler_config or {})
        if self_pack.optimizer_name != opt_name:
            opt_config.setdefault("lr", self.lr)
        else:
            opt_config = update_dict(opt_config, self_pack.optimizer_config or {})
        if self_pack.scheduler_name == sch_name:
            sch_config = update_dict(sch_config, self_pack.scheduler_config or {})
        return OptimizerPack(pack.scope, opt_name, sch_name, opt_config, sch_config)


@Block.register("build_optimizers")
class BuildOptimizersBlock(InjectDefaultsMixin, Block):
    config: Config
    optimizers: Dict[str, Optimizer]
    schedulers: Dict[str, Optional[_LRScheduler]]
    schedulers_requires_metric: Set[str]

    def build(self, config: Config) -> None:
        self.config = config
        state_info = self.extract_state_info.state_info
        # default settings
        settings: Dict[str, Any] = {}
        if config.lr is not None:
            settings["lr"] = config.lr
        if config.optimizer_name is not None:
            settings["optimizer_name"] = config.optimizer_name
        if config.scheduler_name is not None:
            settings["scheduler_name"] = config.scheduler_name
        if config.optimizer_config is not None:
            settings["optimizer_config"] = config.optimizer_config
        if config.scheduler_config is not None:
            settings["scheduler_config"] = config.scheduler_config
        default_opt_settings = OptimizerSettings(**settings)
        ## inject defaults from each train step
        injected_defaults = set()
        optimizer_settings = config.optimizer_settings or {}
        model = self.build_model.model
        for step in model.train_steps:
            scope = step.scope
            if scope not in optimizer_settings:
                injected_defaults.add(scope)
                optimizer_settings[scope] = step.get_default_optimizer_settings()
        # build
        optimizer_packs = []
        converted_defaults = {}
        for scope, sub_settings in optimizer_settings.items():
            if sub_settings is None:
                scope_pack = default_opt_settings.get_opt_pack(state_info)
                scope_pack.scope = scope
            else:
                optimizer = sub_settings.get("optimizer")
                if optimizer is None:
                    raise ValueError(f"optimizer must be provided (scope={scope})")
                scope_pack = OptimizerPack(
                    scope,
                    optimizer,
                    sub_settings.get("scheduler"),
                    sub_settings.get("optimizer_config"),
                    sub_settings.get("scheduler_config"),
                )
            if scope in injected_defaults:
                converted_defaults[scope] = str(scope_pack)
            optimizer_packs.append(scope_pack)
        self._defaults["default_optimizer_settings"] = converted_defaults
        # initialize
        self.optimizers = {}
        self.schedulers = {}
        for pack in optimizer_packs:
            pack = default_opt_settings.update_opt_pack(state_info, pack)
            opt = self._define_optimizer(pack)
            self._define_scheduler(opt, pack)
        # check requires metric
        self.schedulers_requires_metric = set()
        for key, scheduler in self.schedulers.items():
            if scheduler is None:
                continue
            if isinstance(scheduler, WarmupScheduler):
                scheduler = scheduler.scheduler_afterwards
            if scheduler is not None and scheduler_requires_metric(scheduler):
                self.schedulers_requires_metric.add(key)

    @property
    def requirements(self) -> List[Type[Block]]:
        return [ExtractStateInfoBlock, BuildModelBlock]

    @property
    def build_model(self) -> BuildModelBlock:
        return self.get_previous(BuildModelBlock)

    @property
    def extract_state_info(self) -> ExtractStateInfoBlock:
        return self.get_previous(ExtractStateInfoBlock)

    def default_lr_configs(
        self,
        optimizer: Optimizer,
        optimizer_config: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        state_info = self.extract_state_info.state_info
        opt_lr = optimizer_config["lr"]
        # step
        step_default_cfg = {"step_size": 10 * state_info.num_batches}
        # exponential
        exp_gamma = (0.1**0.1) ** (1.0 / state_info.num_batches)
        exp_default_cfg = {"gamma": exp_gamma}
        # cyclic
        cyclic_default_cfg = {
            "base_lr": opt_lr,
            "max_lr": 1.0e-8,
            "step_size_up": 10 * state_info.num_batches,
            "gamma": exp_gamma,
        }
        if "momentum" not in optimizer.defaults:
            cyclic_default_cfg["cycle_momentum"] = False
        # cosine
        cosine_default_cfg = {
            "eta_min": 1.0e-8,
            "T_max": 10 * state_info.num_batches,
        }
        # cosine restarts
        cosine_restarts_default_cfg = {
            "eta_min": 1.0e-8,
            "T_0": 10 * state_info.num_batches,
        }
        # plateau
        plateau_default_cfg = {
            "mode": "max",
            "min_lr": 1.0e-8,
            "verbose": False,
            "patience": max(
                10 * state_info.num_step_per_snapshot,
                state_info.snapshot_start_step,
            ),
        }
        return {
            "step": step_default_cfg,
            "exponential": exp_default_cfg,
            "cyclic": cyclic_default_cfg,
            "cosine": cosine_default_cfg,
            "cosine_restarts": cosine_restarts_default_cfg,
            "plateau": plateau_default_cfg,
        }

    def _define_optimizer(self, pack: OptimizerPack) -> Optimizer:
        model = self.build_model.model
        parameters: Any
        if pack.scope == "all":
            parameters = model.params_groups()
        else:
            attr = model
            scopes = pack.scope.split(".")
            for scope in scopes:
                new_attr = getattr(attr, scope, None)
                if new_attr is None:
                    raise ValueError(f"'{attr}' has no scope '{scope}'")
                attr = new_attr
            if not isinstance(attr, nn.Module):
                parameters = attr
            else:
                parameters = attr.parameters()
        optimizer_base = optimizer_dict[pack.optimizer_name]
        opt_config = pack.optimizer_config or {}
        opt = optimizer_base(parameters, **opt_config)
        self.optimizers[pack.scope] = opt
        return opt

    def _define_scheduler(self, optimizer: Optimizer, pack: OptimizerPack) -> None:
        if pack.scheduler_name is None:
            self.schedulers[pack.scope] = None
        else:
            scheduler = pack.scheduler_name
            opt_config = pack.optimizer_config or {}
            scheduler_config = pack.scheduler_config or {}
            default_lr_configs = self.default_lr_configs(optimizer, opt_config)
            default_lr_config = default_lr_configs.get(scheduler)
            if default_lr_config is not None:
                scheduler_config = update_dict(scheduler_config, default_lr_config)
            if scheduler == "warmup":
                sab = scheduler_config.get("scheduler_afterwards_base", "plateau")
                if sab == "warmup":
                    raise ValueError("warmup should not be used inside a warmup")
                sac = scheduler_config.get("scheduler_afterwards_config", {})
                default_lr_config = default_lr_configs.get(sab)
                sac = update_dict(sac, default_lr_config or {})
                sab = scheduler_dict[sab]
                scheduler_config["scheduler_afterwards_base"] = sab
                scheduler_config["scheduler_afterwards_config"] = sac
            scheduler_base = scheduler_dict[scheduler]
            scheduler_config = filter_kw(scheduler_base, scheduler_config)
            self.schedulers[pack.scope] = scheduler_base(optimizer, **scheduler_config)


@Block.register("build_trainer")
class BuildTrainerBlock(Block):
    trainer: ITrainer

    def build(self, config: Config) -> None:
        self.trainer = Trainer(config)
        self.trainer.pipeline = self.pipeline  # type: ignore


# runtime blocks


@Block.register("record_num_samples")
class RecordNumSamplesBlock(Block):
    def build(self, config: Config) -> None:
        pass

    def run(self, data: IData, _defaults: OrderedDict, **kwargs: Any) -> None:
        _defaults["train_samples"] = len(data.train_dataset)
        if data.valid_dataset is None:
            _defaults["valid_samples"] = None
        else:
            _defaults["valid_samples"] = len(data.valid_dataset)
        _defaults.move_to_end("valid_samples", last=False)
        _defaults.move_to_end("train_samples", last=False)


@Block.register("report")
class ReportBlock(Block):
    config: Config
    report_file = "report.txt"

    def build(self, config: Config) -> None:
        self.config = config

    def run(self, data: IData, _defaults: OrderedDict, **kwargs: Any) -> None:
        if not self.is_local_rank_0 or self.training_workspace is None:
            return
        self._report_messages(
            "Internal Default Configurations Used by `carefree-learn`",
            _defaults,
            self.training_workspace,
        )
        self._report_messages(
            "External Configurations",
            self.config.get_external_configs(set(_defaults)),
            self.training_workspace,
        )

    def _report_messages(
        self,
        title: str,
        messages: Dict[str, Any],
        report_folder: str,
    ) -> None:
        def _stringify_item(
            item: Tuple[str, Any],
            prefix: Optional[str] = None,
            depth: int = 0,
        ) -> str:
            key, value = item
            if prefix is not None:
                key = f"{prefix}{key}"
            if not isinstance(value, dict) or not value or depth >= 2:
                key = truncate_string_to_length(key, span)
                return f"{key:>{span}s}   |   {value}"
            prefix = f"{key}."
            items = [
                _stringify_item((vk, vv), prefix, depth=depth + 1)
                for vk, vv in value.items()
            ]
            return "\n".join(items)

        span = 64
        length = 2 * span
        msg = "\n".join(
            [
                "=" * length,
                f"{title:^{length}s}",
                "-" * length,
                "\n".join(map(_stringify_item, messages.items())),
                "-" * length,
            ]
        )
        console.log(msg)
        if report_folder is not None:
            with open(os.path.join(report_folder, self.report_file), "a") as f:
                f.write(msg + "\n")


@Block.register("training")
class TrainingBlock(Block):
    config: Config
    trainer_config_file = "trainer_config.json"

    def build(self, config: Config) -> None:
        self.config = config

    @property
    def requirements(self) -> List[Type[Block]]:
        return [
            BuildModelBlock,
            BuildMetricsBlock,
            BuildInferenceBlock,
            BuildOptimizersBlock,
            BuildMonitorsBlock,
            BuildCallbacksBlock,
            BuildTrainerBlock,
        ]

    @property
    def build_model(self) -> BuildModelBlock:
        return self.get_previous(BuildModelBlock)

    @property
    def build_metrics(self) -> BuildMetricsBlock:
        return self.get_previous(BuildMetricsBlock)

    @property
    def build_inference(self) -> BuildInferenceBlock:
        return self.get_previous(BuildInferenceBlock)

    @property
    def build_optimizers(self) -> BuildOptimizersBlock:
        return self.get_previous(BuildOptimizersBlock)

    @property
    def build_monitors(self) -> BuildMonitorsBlock:
        return self.get_previous(BuildMonitorsBlock)

    @property
    def build_callbacks(self) -> BuildCallbacksBlock:
        return self.get_previous(BuildCallbacksBlock)

    @property
    def build_trainer(self) -> BuildTrainerBlock:
        return self.get_previous(BuildTrainerBlock)

    def run(
        self,
        data: IData,
        _defaults: OrderedDictType,
        *,
        device: device_type = None,
        **kwargs: Any,
    ) -> None:
        def fit(p: Optional[profile] = None) -> None:
            self.build_trainer.trainer.fit(
                data,
                self.build_model.model,
                self.build_metrics.metrics,
                self.build_inference.inference,
                self.build_optimizers.optimizers,
                self.build_optimizers.schedulers,
                self.build_monitors.monitors,
                self.build_callbacks.callbacks,
                self.build_optimizers.schedulers_requires_metric,
                config_export_file=self.trainer_config_file,
                device=device,
                p=p,
            )

        def trace_handler(p: profile) -> None:
            workspace = self.training_workspace
            if workspace is None:
                return
            trace_folder = os.path.join(workspace, "traces")
            trace_path = os.path.join(trace_folder, f"trace_{p.step_num}.json")
            os.makedirs(trace_folder, exist_ok=True)
            p.export_chrome_trace(trace_path)

        if not self.config.profile:
            fit()
        else:
            os.environ["KINETO_LOG_LEVEL"] = "5"
            schedule_config = self.config.profile_schedule_config or {}
            schedule_config.setdefault("skip_first", 5)
            schedule_config.setdefault("wait", 3)
            schedule_config.setdefault("warmup", 3)
            schedule_config.setdefault("active", 5)
            schedule_config.setdefault("repeat", 5)
            profile_config = self.config.profile_config or {}
            profile_config["schedule"] = schedule(**schedule_config)
            profile_config["on_trace_ready"] = trace_handler
            profile_config.setdefault("record_shapes", True)
            profile_config.setdefault("profile_memory", False)
            profile_config.setdefault("with_stack", True)
            profile_config.setdefault("with_flops", True)
            profile_config.setdefault("with_modules", True)
            with profile():
                console.debug("running dummy profiler warmup for CUPTI")
            with profile(**profile_config) as p:
                fit(p)


# serialization blocks


@Block.register("serialize_data")
class SerializeDataBlock(Block):
    data: Optional[IData]
    config: Config
    package_folder: str = "data_module"

    def build(self, config: Config) -> None:
        self.data = None
        self.config = config

    def save_extra(self, folder: TPath) -> None:
        folder = to_path(folder)
        if not self.is_local_rank_0:
            return
        if self.training_workspace is not None:
            data_folder = os.path.join(self.training_workspace, self.package_folder)
            shutil.copytree(data_folder, folder)
        elif self.data is not None:
            Serializer.save(folder, self.data, save_npd=False)

    def load_from(self, folder: TPath) -> None:
        folder = to_path(folder)
        if folder.is_dir():
            self.data = Serializer.load(folder, IData, load_npd=False)  # type: ignore


@Block.register("serialize_model")
class SerializeModelBlock(Block):
    config: Config

    verbose: bool = True
    ckpt_folder: Optional[Path] = None
    ckpt_scores: Optional[Dict[str, float]] = None

    def build(self, config: Config) -> None:
        self.config = config
        self.best_score = 0.0

    @property
    def requirements(self) -> List[Type[Block]]:
        return [BuildModelBlock]

    @property
    def build_model(self) -> BuildModelBlock:
        return self.get_previous(BuildModelBlock)

    def save_extra(self, folder: TPath) -> None:
        if not self.is_local_rank_0:
            return
        folder = to_path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        warn_msg = "no checkpoints found at {}, current model states will be saved"
        if self.training_workspace is not None:
            ckpt_folder = os.path.join(self.training_workspace, CHECKPOINTS_FOLDER)
            sorted_ckpts = get_sorted_checkpoints(ckpt_folder)
            if sorted_ckpts:
                best_ckpt = sorted_ckpts[0]
                src_path = os.path.join(ckpt_folder, sorted_ckpts[0])
                dst_path = os.path.join(folder, best_ckpt)
                shutil.copyfile(src_path, dst_path)
                scores = get_scores(ckpt_folder)
                with open(os.path.join(folder, SCORES_FILE), "w") as f:
                    json.dump({best_ckpt: scores.get(best_ckpt, 0.0)}, f)
                if self.verbose:
                    console.debug(f"best checkpoint '{best_ckpt}' saved")
            else:
                if self.verbose:
                    console.warn(warn_msg.format(ckpt_folder))
                self._save_current(folder)
            return
        if self.ckpt_folder is None or self.ckpt_scores is None:
            if self.verbose:
                console.warn("current model states will be saved")
            self._save_current(folder)
        else:
            any_saved = False
            filtered_scores = {}
            sorted_ckpt_dict = sort_dict_by_value(self.ckpt_scores, reverse=True)
            for file, score in sorted_ckpt_dict.items():
                ckpt_path = os.path.join(self.ckpt_folder, file)
                if not os.path.isfile(ckpt_path):
                    if self.verbose:
                        msg = f"cannot find checkpoint at '{ckpt_path}', did you delete it?"
                        console.warn(msg)
                    continue
                any_saved = True
                filtered_scores[file] = score
                shutil.copyfile(ckpt_path, os.path.join(folder, file))
                break
            if any_saved:
                with open(os.path.join(folder, SCORES_FILE), "w") as f:
                    json.dump(filtered_scores, f)
            else:
                if self.verbose:
                    console.warn(warn_msg.format(self.ckpt_folder))
                self._save_current(folder)

    def load_from(self, folder: TPath) -> None:
        model = self.build_model.model
        folder = to_path(folder)
        best_file = get_sorted_checkpoints(folder)[0]
        states = torch.load(folder / best_file, map_location="cpu")
        # check if the loaded `states` is a full state dict
        # this check makes it compatible with 'pure' states, for which
        # we don't need to extract from the 'states' key
        if set(states) == {"config", "states"}:
            states = states["states"]
        model.load_state_dict(states)
        scores = get_scores(folder)
        self.ckpt_folder = folder
        self.ckpt_scores = scores

    def _save_current(self, folder: TPath) -> None:
        folder = to_path(folder)
        latest_file = f"{PT_PREFIX}-1.pt"
        latest_path = folder / latest_file
        new_scores_path = folder / SCORES_FILE
        self.build_model.model.save(latest_path)
        with new_scores_path.open("w") as f:
            json.dump({latest_file: 0.0}, f)


@Block.register("serialize_optimizer")
class SerializeOptimizerBlock(Block):
    optimizer_file = "optimizers.pt"
    scheduler_file = "schedulers.pt"

    def build(self, config: Config) -> None:
        pass

    @property
    def requirements(self) -> List[Type[Block]]:
        return [BuildOptimizersBlock]

    @property
    def build_optimizers(self) -> BuildOptimizersBlock:
        return self.get_previous(BuildOptimizersBlock)

    def save_extra(self, folder: TPath) -> None:
        optims = self.build_optimizers.optimizers
        scheds = self.build_optimizers.schedulers
        opt_d = {k: v.state_dict() for k, v in optims.items()}
        sch_d = {k: None if v is None else v.state_dict() for k, v in scheds.items()}
        os.makedirs(folder, exist_ok=True)
        torch.save(opt_d, os.path.join(folder, self.optimizer_file))
        torch.save(sch_d, os.path.join(folder, self.scheduler_file))

    def load_from(self, folder: TPath) -> None:
        folder = to_path(folder)
        optimizers = self.build_optimizers.optimizers
        schedulers = self.build_optimizers.schedulers
        opt_d = torch.load(folder / self.optimizer_file)
        sch_d = torch.load(folder / self.scheduler_file)
        for k, states in opt_d.items():
            optimizers[k].load_state_dict(states)
        for k, states in sch_d.items():
            k_sch = schedulers[k]
            if k_sch is not None:
                k_sch.load_state_dict(states)


@Block.register("serialize_script")
class SerializeScriptBlock(Block):
    script_file = "script.py"

    def build(self, config: Config) -> None:
        pass

    def save_extra(self, folder: TPath) -> None:
        folder = to_path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        frame = inspect.currentframe()
        if frame is None:
            return None
        while frame.f_back is not None:
            frame = frame.f_back
        with (folder / self.script_file).open("w") as f:
            f.write(inspect.getsource(frame))


__all__ = [
    "SetDefaultsBlock",
    "PrepareWorkplaceBlock",
    "ExtractStateInfoBlock",
    "BuildModelBlock",
    "BuildMetricsBlock",
    "BuildInferenceBlock",
    "SetTrainerDefaultsBlock",
    "BuildMonitorsBlock",
    "BuildCallbacksBlock",
    "BuildOptimizersBlock",
    "BuildTrainerBlock",
    "RecordNumSamplesBlock",
    "ReportBlock",
    "TrainingBlock",
    "SerializeDataBlock",
    "SerializeModelBlock",
    "SerializeOptimizerBlock",
    "SerializeScriptBlock",
]
