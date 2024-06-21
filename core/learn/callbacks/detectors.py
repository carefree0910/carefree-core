import json
import math
import torch

import numpy as np

from typing import Dict
from pathlib import Path

from ..schema import ITrainer
from ..schema import StepOutputs
from ..schema import TrainStepLoss
from ..schema import TrainerCallback
from ..toolkit import tensor_batch_to_np
from ...toolkit import console
from ...toolkit.misc import get_ddp_info
from ...toolkit.types import np_dict_type
from ...toolkit.types import tensor_dict_type


def dump_problematic(
    np_batch: np_dict_type,
    forward_results: tensor_dict_type,
    debug_folder: Path,
    batch_paths: Dict[str, Path],
) -> str:
    ddp_info = get_ddp_info()
    appendix = "" if ddp_info is None else f"_{ddp_info.rank}"
    for k, v in np_batch.items():
        if isinstance(v, np.ndarray):
            v_path = debug_folder / f"{k}{appendix}.npy"
            batch_paths[k] = v_path
            np.save(v_path, v)
    for k, v in forward_results.items():
        if isinstance(v, torch.Tensor):
            v_path = debug_folder / f"{k}{appendix}.pt"
            batch_paths[k] = v_path
            torch.save(v, v_path)
    return appendix


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
            nan_ratios = {
                k: np.isnan(v).mean().item()
                for k, v in np_batch.items()
                if isinstance(v, np.ndarray)
            }
            debug_folder = Path(trainer.workspace) / "debugging"
            debug_folder.mkdir(exist_ok=True)
            batch_paths: Dict[str, Path] = {}
            appendix = dump_problematic(
                np_batch, step_outputs.forward_results, debug_folder, batch_paths
            )
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


@TrainerCallback.register("grad_detector")
class GradientDetectorCallback(TrainerCallback):
    def __init__(self, threshold: float = 1.0):
        super().__init__()
        self.threshold = threshold

    def before_gradient_update(
        self,
        trainer: ITrainer,
        batch: tensor_dict_type,
        forward: tensor_dict_type,
        loss_res: TrainStepLoss,
        update: bool,
    ) -> None:
        def record(msg: str) -> None:
            errors[k] = msg
            err_grads[k] = p.grad
            err_parameters[k] = p

        m = trainer.model.m
        errors: Dict[str, str] = {}
        err_grads: Dict[str, torch.Tensor] = {}
        err_parameters: Dict[str, torch.nn.Parameter] = {}
        batch_paths: Dict[str, Path] = {}
        debug_folder = Path(trainer.workspace) / "debugging" / str(trainer.state.step)
        need_raise = False
        for k, p in m.named_parameters():
            if torch.isnan(p.grad).any().item():
                record("NaN")
                need_raise = True
            elif torch.isinf(p.grad).any().item():
                record("Inf")
                need_raise = True
            else:
                max_grad = torch.abs(p.grad).max().item()
                if max_grad >= self.threshold:
                    record(f"Too Large ({max_grad})")
        if errors:
            debug_folder.mkdir(exist_ok=True, parents=True)
            appendix = dump_problematic(
                tensor_batch_to_np(batch), forward, debug_folder, batch_paths
            )
            for k, g in err_grads.items():
                grad_path = debug_folder / f"{k}_grad{appendix}.pt"
                batch_paths[k] = grad_path
                torch.save(g, grad_path)
            for k, p in err_parameters.items():
                param_path = debug_folder / f"{k}{appendix}.pt"
                batch_paths[k] = param_path
                torch.save(p, param_path)
        if need_raise:
            console.error(
                f"following errors occurred: {errors}, current batch / states / grads "
                f"will be saved to {batch_paths} for further investigation"
            )
            raise RuntimeError("Gradient Error Detected")
        if errors:
            console.warn(
                f"following warnings occurred: {errors}, current batch / states / grads "
                f"will be saved to {batch_paths} for further investigation"
            )


__all__ = [
    "NaNDetectorCallback",
    "GradientDetectorCallback",
]
