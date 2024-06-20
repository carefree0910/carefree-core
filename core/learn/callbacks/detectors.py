import json
import math
import torch

import numpy as np

from pathlib import Path

from ..schema import ITrainer
from ..schema import StepOutputs
from ..schema import TrainerCallback
from ..toolkit import tensor_batch_to_np
from ...toolkit import console
from ...toolkit.misc import get_ddp_info
from ...toolkit.types import tensor_dict_type


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


__all__ = [
    "NaNDetectorCallback",
]
