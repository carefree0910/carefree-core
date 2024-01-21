import numpy as np

from tqdm import tqdm
from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Optional

from .schema import IModel
from .schema import IMetric
from .schema import IInference
from .schema import DataLoader
from .schema import MetricsOutputs
from .schema import InferenceOutputs
from .toolkit import get_device
from .toolkit import tensor_batch_to_np
from .toolkit import ONNX
from .constants import LABEL_KEY
from ..toolkit.misc import shallow_copy_dict
from ..toolkit.array import to_device


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
        **kwargs: Any,
    ) -> InferenceOutputs:
        def stack(arrays: TArrays, return_nones: bool, should_stack: bool) -> Any:
            if not return_nones:
                return {k: None for k in arrays}
            if not should_stack:
                return arrays
            return {
                k: np.vstack(v) if isinstance(v[0], np.ndarray) else v
                for k, v in arrays.items()
            }

        def run() -> InferenceOutputs:
            all_np_outputs: TArrays = {}
            all_labels: TArrays = {}
            all_metrics_requires: TArrays = {}
            metric_outputs_list: List[MetricsOutputs] = []
            loss_items: Dict[str, List[float]] = {}

            iterator = enumerate(loader)
            if use_tqdm:
                iterator = tqdm(iterator, "inference", len(loader))
            for i, tensor_batch in iterator:
                if i / len(loader) >= portion:
                    break
                np_batch = None
                np_outputs = None
                if self.onnx is not None:
                    np_batch = tensor_batch_to_np(tensor_batch)
                    np_outputs = self.onnx.predict(np_batch)
                elif self.model is not None:
                    tensor_batch = to_device(tensor_batch, get_device(self.model))
                    step_outputs = self.model.step(
                        i,
                        tensor_batch,
                        shallow_copy_dict(kwargs),
                        use_grad=use_grad,
                        get_losses=use_losses_as_metrics,
                    )
                    np_outputs = tensor_batch_to_np(step_outputs.forward_results)
                    if use_losses_as_metrics:
                        for k, vl in step_outputs.loss_dict.items():
                            loss_items.setdefault(k, []).append(vl)
                assert np_outputs is not None
                # metrics
                if metrics is not None and not metrics.requires_all:
                    if np_batch is None:
                        np_batch = tensor_batch_to_np(tensor_batch)
                    metric_outputs = metrics.evaluate(np_batch, np_outputs)
                    metric_outputs_list.append(metric_outputs)
                # gather
                if return_outputs:
                    for k, v in np_outputs.items():
                        if v is not None:
                            all_np_outputs.setdefault(k, []).append(v)
                if return_labels:
                    if np_batch is None:
                        np_batch = tensor_batch_to_np(tensor_batch)
                    for k, v in np_batch.items():
                        if v is not None and k.endswith(LABEL_KEY):
                            all_labels.setdefault(k, []).append(v)
                if metrics is not None and metrics.requires_all:
                    if np_batch is None:
                        np_batch = tensor_batch_to_np(tensor_batch)
                    for k, v in np_batch.items():
                        if v is not None and metrics.requires(k):
                            all_metrics_requires.setdefault(k, []).append(v)

            # stack
            stacked_np_outputs = stack(all_np_outputs, return_outputs, stack_outputs)
            stacked_labels = stack(all_labels, return_labels, stack_outputs)
            # gather metric outputs
            if metrics is None:
                final_metric_outputs = None
            elif metrics.requires_all:
                final_metric_outputs = metrics.evaluate(
                    stack(all_metrics_requires, False, True),
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

            return InferenceOutputs(
                stacked_np_outputs,
                stacked_labels,
                final_metric_outputs,
                None
                if not use_losses_as_metrics
                else {k: sum(v) / len(v) for k, v in loss_items.items()},
            )

        use_grad = kwargs.pop("use_grad", self.use_grad_in_predict)
        try:
            return run()
        except:
            use_grad = self.use_grad_in_predict = True
            return run()


__all__ = [
    "Inference",
]
