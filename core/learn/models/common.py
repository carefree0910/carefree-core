import torch

import torch.nn as nn

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Protocol

from ..losses import build_loss
from ..schema import raw_forward_results_type
from ..schema import ILoss
from ..schema import Config
from ..schema import IModel
from ..schema import TrainStep
from ..schema import TrainerState
from ..schema import TrainStepLoss
from ..modules import build_module
from ..modules import EMA
from ..toolkit import get_clones
from ..constants import LOSS_KEY
from ..modules.common import get_tgt_params
from ...toolkit.misc import safe_execute
from ...toolkit.types import tensor_dict_type


class CommonTrainStep(TrainStep):
    def __init__(self, loss: ILoss):
        super().__init__()
        self.loss = loss

    def loss_fn(
        self,
        m: IModel,
        state: Optional[TrainerState],
        batch: tensor_dict_type,
        forward_results: tensor_dict_type,
        **kwargs: Any,
    ) -> TrainStepLoss:
        losses = self.loss(forward_results, batch, state)
        if isinstance(losses, Tensor):
            losses = {LOSS_KEY: losses}
        return TrainStepLoss(losses[LOSS_KEY], {k: v[None] for k, v in losses.items()})


@IModel.register("common")
class CommonModel(IModel):
    loss: ILoss

    @property
    def train_steps(self) -> List[TrainStep]:
        return [CommonTrainStep(self.loss)]

    @property
    def all_modules(self) -> List[nn.Module]:
        return [self.m, self.loss]

    def build(self, config: Config) -> None:
        if config.loss_name is None:
            raise ValueError("`loss_name` should be specified for `CommonModel`")
        self.m = build_module(config.module_name, config=config.module_config)
        self.loss = build_loss(config.loss_name, config=config.loss_config)


@IModel.register("direct")
class DirectModel(CommonModel):
    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> raw_forward_results_type:
        return self.m(batch, **kwargs)


@IModel.register("with_state")
class WithStateModel(CommonModel):
    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> raw_forward_results_type:
        return self.m(batch, state, **kwargs)


class EnsembleFn(Protocol):
    def __call__(self, key: str, tensors: List[Tensor]) -> Tensor:
        pass


class EnsembleModule(nn.ModuleList):
    def forward(self, *args: Any, **kwargs: Any) -> List[raw_forward_results_type]:
        return [m(*args, **kwargs) for m in self]


class EnsembleModel(CommonModel):
    m: EnsembleModule
    ensemble_fn: Optional[EnsembleFn]

    def __init__(self, m: IModel, num_repeat: int) -> None:
        super().__init__()
        self.m = EnsembleModule(get_clones(m.m, num_repeat))
        self.loss = m.loss if isinstance(m, CommonModel) else None
        self.model = m
        self.t_model = m.__class__
        self.config = m.config.copy()
        self.ensemble_fn = None
        self.__identifier__ = m.__identifier__

    @property
    def train_steps(self) -> List[TrainStep]:
        return self.model.train_steps

    @property
    def all_modules(self) -> List[nn.Module]:
        if self.loss is None:
            return [self.m]
        return [self.m, self.loss]

    def build(self, config: Config) -> None:
        raise RuntimeError("`build` should not be called for `EnsembleModel`")

    def rehook_ema(self) -> None:
        def _fetch(m: nn.Module, prefix: str) -> None:
            for k, child in list(m.named_children()):
                if not isinstance(child, EMA):
                    _fetch(child, f"{prefix}.{k}" if prefix else k)
                else:
                    emas[f"{prefix}.{k}"] = child

        emas: Dict[str, EMA] = {}
        _fetch(self.m, "")
        for k, ema in emas.items():
            idx = int(k.split(".")[0])
            idx_module = self.m[idx]
            idx_named_parameters = idx_module.named_parameters()
            ema.tgt_params = get_tgt_params(idx_named_parameters)

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> raw_forward_results_type:
        return self.t_model.forward(self, batch_idx, batch, state, **kwargs)

    def postprocess(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        forward_results: List[raw_forward_results_type],  # type: ignore
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        outputs: Dict[str, List[Tensor]] = {}
        for rs in forward_results:
            i_rs = self.t_model.postprocess(self, batch_idx, batch, rs, state, **kwargs)
            for k, v in i_rs.items():
                outputs.setdefault(k, []).append(v)
        final_results: tensor_dict_type = {}
        for k in sorted(outputs):
            if self.ensemble_fn is None:
                v = torch.stack(outputs[k]).mean(0)
            else:
                v = safe_execute(self.ensemble_fn, dict(key=k, tensors=outputs[k]))
            final_results[k] = v
        return final_results


__all__ = [
    "CommonTrainStep",
    "CommonModel",
    "DirectModel",
    "WithStateModel",
    "EnsembleModel",
]
