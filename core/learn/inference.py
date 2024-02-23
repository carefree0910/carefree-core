import math
import torch

import numpy as np

from tqdm import tqdm
from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Optional
from accelerate import Accelerator

from .schema import IModel
from .schema import IMetric
from .schema import IInference
from .schema import DataLoader
from .schema import MetricsOutputs
from .schema import InferenceOutputs
from .toolkit import get_device
from .toolkit import is_local_rank_0
from .toolkit import tensor_batch_to_np
from .toolkit import ONNX
from .constants import LABEL_KEY
from ..toolkit import console
from ..toolkit.misc import shallow_copy_dict
from ..toolkit.array import to_device
from ..toolkit.types import np_dict_type
from ..toolkit.types import tensor_dict_type


TArrays = Dict[str, List[Union[np.ndarray, Any]]]


class Inference(IInference):
    def __init__(
        self,
        *,
        onnx: Optional[Union[str, ONNX]] = None,
        model: Optional[IModel] = None,
        use_grad_in_predict: bool = False,
    ):
        if onnx is None and model is None:
            raise ValueError("either `onnx` or `model` should be provided")
        if onnx is not None and model is not None:
            raise ValueError("only one of `onnx` and `model` should be provided")
        if isinstance(onnx, str):
            onnx = ONNX(onnx)
        self.onnx = onnx
        self.model = model
        self.use_grad_in_predict = use_grad_in_predict

    def get_outputs(
        self,
        loader: DataLoader,
        *,
        portion: float = 1.0,
        metrics: Optional[IMetric] = None,
        use_losses_as_metrics: bool = False,
        return_outputs: bool = True,
        return_labels: bool = False,
        stack_outputs: bool = True,
        use_tqdm: bool = False,
        use_inference_mode: Optional[bool] = None,
        accelerator: Optional[Accelerator] = None,
        pad_dim: Optional[int] = None,
        **kwargs: Any,
    ) -> InferenceOutputs:
        def pad(k: str, arrays: List[np.ndarray]) -> List[np.ndarray]:
            if pad_dim is None:
                return arrays
            padded = []
            max_shape = max([array.shape[pad_dim] for array in arrays])
            if all(array.shape[pad_dim] == max_shape for array in arrays):
                return arrays
            if is_local_rank_0():
                console.warn(
                    f"padding '{k}' at dim {pad_dim} to {max_shape}, please perform "
                    "post-processing to remove the paddings if necessary."
                )
            for array in arrays:
                i_paddings = [[0, 0] for _ in range(array.ndim)]
                i_paddings[pad_dim][1] = max_shape - array.shape[pad_dim]
                padded.append(np.pad(array, i_paddings))
            return padded

        def stack(arrays: TArrays, return_arrays: bool, should_stack: bool) -> Any:
            if not return_arrays:
                return {k: None for k in arrays}
            if not should_stack:
                return arrays
            return {
                k: np.vstack(pad(k, v)) if isinstance(v[0], np.ndarray) else v
                for k, v in arrays.items()
            }

        def to_np_batch(tensors: tensor_dict_type) -> np_dict_type:
            if accelerator is not None:
                if pad_dim is not None:
                    tensors = accelerator.pad_across_processes(tensors, dim=pad_dim)
                tensors = accelerator.gather_for_metrics(tensors)
            return tensor_batch_to_np(tensors)

        def run() -> InferenceOutputs:
            all_np_outputs: TArrays = {}
            all_labels: TArrays = {}
            all_metrics_requires: TArrays = {}
            metric_outputs_list: List[MetricsOutputs] = []
            loss_tensors_lists: Dict[str, List[Tensor]] = {}

            device = None if self.model is None else get_device(self.model)
            iterator = enumerate(loader)
            if use_tqdm:
                total = math.floor(len(loader) * portion)
                iterator = tqdm(iterator, "inference", total)
            gather_np = return_outputs or (metrics is not None and metrics.requires_all)
            for i, tensor_batch in iterator:
                if i / len(loader) >= portion:
                    break
                if i == 0 and accelerator is not None:
                    accelerator.wait_for_everyone()
                np_batch = None
                np_outputs = None
                tensor_outputs = None
                if self.onnx is not None:
                    # will not consider distributed stuffs at onnx inference
                    np_batch = tensor_batch_to_np(tensor_batch)
                    np_outputs = self.onnx.predict(np_batch)
                elif self.model is not None:
                    # accelerator will handle the device stuffs
                    if accelerator is None:
                        tensor_batch = to_device(tensor_batch, device)
                    Flag.in_step = True
                    step_outputs = self.model.step(
                        i,
                        tensor_batch,
                        shallow_copy_dict(kwargs),
                        use_grad=use_grad,
                        get_losses=use_losses_as_metrics,
                        use_inference_mode=use_inference_mode,
                    )
                    Flag.in_step = False
                    tensor_outputs = step_outputs.forward_results
                    if use_losses_as_metrics:
                        for k, v in step_outputs.loss_tensors.items():
                            loss_tensors_lists.setdefault(k, []).append(v)
                assert np_outputs is not None or tensor_outputs is not None
                # metrics
                if metrics is not None and not metrics.requires_all:
                    if np_batch is None:
                        np_batch = to_np_batch(tensor_batch)
                    if np_outputs is None:
                        np_outputs = to_np_batch(tensor_outputs)  # type: ignore
                    metric_outputs = metrics.evaluate(np_batch, np_outputs)
                    metric_outputs_list.append(metric_outputs)
                # gather
                if gather_np:
                    if np_outputs is None:
                        np_outputs = to_np_batch(tensor_outputs)  # type: ignore
                    for k, v in np_outputs.items():
                        if v is not None:
                            all_np_outputs.setdefault(k, []).append(v)
                if return_labels:
                    if np_batch is None:
                        np_batch = to_np_batch(tensor_batch)
                    for k, v in np_batch.items():
                        if v is not None and k.endswith(LABEL_KEY):
                            all_labels.setdefault(k, []).append(v)
                if metrics is not None and metrics.requires_all:
                    if np_batch is None:
                        np_batch = to_np_batch(tensor_batch)
                    for k, v in np_batch.items():
                        if v is not None and metrics.requires(k):
                            all_metrics_requires.setdefault(k, []).append(v)

            # stack
            stacked_np_outputs = stack(all_np_outputs, gather_np, stack_outputs)
            stacked_labels = stack(all_labels, return_labels, stack_outputs)
            # gather metric outputs
            if metrics is None:
                final_metric_outputs = None
            elif metrics.requires_all:
                final_metric_outputs = metrics.evaluate(
                    stack(all_metrics_requires, True, True),
                    stacked_np_outputs,
                    loader,
                )
            else:
                scores = []
                metric_values: Dict[str, List[float]] = {}
                is_positive: Dict[str, bool] = {}
                for metric_outputs in metric_outputs_list:
                    scores.append(metric_outputs.final_score)
                    for k, v in metric_outputs.metric_values.items():
                        metric_values.setdefault(k, []).append(v)
                        existing_is_positive = is_positive.get(k)
                        k_is_positive = metric_outputs.is_positive[k]
                        if (
                            existing_is_positive is not None
                            and existing_is_positive != k_is_positive
                        ):
                            raise ValueError(
                                f"the `is_positive` property of '{k}' collides: "
                                f"{existing_is_positive} (previous) != {k_is_positive}"
                            )
                        is_positive[k] = k_is_positive
                final_metric_outputs = MetricsOutputs(
                    sum(scores) / len(scores),
                    {k: sum(vl) / len(vl) for k, vl in metric_values.items()},
                    is_positive,
                )
            # handle accelerator stuffs
            if accelerator is not None:
                for k, vl in loss_tensors_lists.items():
                    loss_tensors_lists[k] = accelerator.gather_for_metrics(vl)
                accelerator.wait_for_everyone()

            return InferenceOutputs(
                stacked_np_outputs,
                stacked_labels,
                final_metric_outputs,
                (
                    None
                    if not use_losses_as_metrics
                    else {
                        k: torch.cat(v).mean().item()
                        for k, v in loss_tensors_lists.items()
                    }
                ),
            )

        class Flag:
            in_step = False

        use_grad = kwargs.pop("use_grad", self.use_grad_in_predict)
        try:
            return run()
        except KeyboardInterrupt:
            raise
        except:
            if not Flag.in_step:
                raise
            use_grad = self.use_grad_in_predict = True
            return run()


__all__ = [
    "Inference",
]
