import math
import torch

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Optional
from typing import ContextManager
from accelerate import Accelerator
from contextlib import nullcontext
from rich.progress import TaskID
from rich.progress import Progress
from accelerate.utils import gather_object
from accelerate.utils import broadcast_object_list

from .schema import IModel
from .schema import IMetric
from .schema import IInference
from .schema import DataLoader
from .schema import IStreamMetric
from .schema import MetricsOutputs
from .schema import MultipleMetrics
from .schema import InferenceOutputs
from .toolkit import get_device
from .toolkit import np_batch_to_tensor
from .toolkit import tensor_batch_to_np
from .toolkit import ONNX
from .constants import LABEL_KEY
from .constants import INFERENCE_COLOR
from .constants import PREDICTIONS_KEY
from ..toolkit import console
from ..toolkit.misc import is_local_rank_0
from ..toolkit.misc import shallow_copy_dict
from ..toolkit.array import is_int
from ..toolkit.array import to_device
from ..toolkit.types import tensor_dict_type


TTensors = Dict[str, List[Union[Tensor, Any]]]


def no_sync_context(accelerator: Accelerator, model: IModel) -> ContextManager:
    if accelerator is None:
        return nullcontext()
    return accelerator.no_sync(model.m)


class Flags:
    in_step = False
    progress_task: Optional[TaskID] = None


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
        target_outputs: Union[str, List[str]] = PREDICTIONS_KEY,
        recover_labels: bool = True,
        recover_predictions: bool = True,
        return_labels: bool = False,
        target_labels: Union[str, List[str]] = LABEL_KEY,
        concat_outputs: bool = True,
        progress: Optional[Progress] = None,
        progress_kwargs: Optional[Dict[str, Any]] = None,
        use_inference_mode: Optional[bool] = None,
        accelerator: Optional[Accelerator] = None,
        pad_dim: Optional[Union[int, Dict[str, int]]] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> InferenceOutputs:
        def get_pad_dim(k: str) -> Optional[int]:
            return (
                None
                if pad_dim is None
                else pad_dim if isinstance(pad_dim, int) else pad_dim.get(k)
            )

        def concat(tensors: TTensors) -> Any:
            concated: tensor_dict_type = {}
            for k, v in tensors.items():
                if not isinstance(v[0], Tensor):
                    concated[k] = v
                    continue
                k_pad_dim = get_pad_dim(k)
                if k_pad_dim is None:
                    concated[k] = torch.cat(v)
                    continue
                max_shape = max([tensor.shape[k_pad_dim] for tensor in v])
                if all(tensor.shape[k_pad_dim] == max_shape for tensor in v):
                    concated[k] = torch.cat(v)
                    continue
                if verbose:
                    rank = 0 if accelerator is None else accelerator.process_index
                    console.warn(
                        f"\[rank {rank}] padding '{k}' at dim {k_pad_dim} to {max_shape}, please perform "
                        "post-processing to remove the paddings if necessary."
                    )
                shapes = [len(v), *v[0].shape]
                shapes[k_pad_dim + 1] = max_shape
                if is_int(v[0]):
                    new = v[0].new_zeros(shapes)
                else:
                    new = v[0].new_full(shapes, torch.nan)
                for i, tensor in enumerate(v):
                    i_slices = [slice(None)] * len(shapes)
                    i_slices[0] = slice(i, i + 1)
                    i_slices[k_pad_dim + 1] = slice(0, tensor.shape[k_pad_dim])
                    new[tuple(i_slices)] = tensor
                concated[k] = new.view(shapes[0] * shapes[1], *shapes[2:])
            return concated

        def recover_labels_of(tensors: tensor_dict_type) -> tensor_dict_type:
            if recover_labels:
                tensors = shallow_copy_dict(tensors)
                for k, v in tensors.items():
                    if v is not None and k in target_labels:
                        tensors[k] = loader.recover_labels(k, v)
            return tensors

        def recover_predictions_of(tensors: tensor_dict_type) -> tensor_dict_type:
            if recover_predictions:
                tensors = shallow_copy_dict(tensors)
                for k, v in tensors.items():
                    if v is not None and isinstance(v, Tensor):
                        tensors[k] = loader.recover_labels(k, v)
            return tensors

        def cleanup_progress() -> None:
            if progress is not None and flags.progress_task is not None:
                progress.stop()
                progress.remove_task(flags.progress_task)
                flags.progress_task = None

        def _run() -> InferenceOutputs:
            all_inputs: TTensors = {}
            all_labels: TTensors = {}
            all_outputs: TTensors = {}
            metric_outputs_list: List[MetricsOutputs] = []
            loss_tensors_lists: TTensors = {}

            device = None if self.model is None else get_device(self.model.m)
            iterator = enumerate(loader)
            if progress is not None:
                progress_kw = shallow_copy_dict(progress_kwargs or {})
                progress_kw.setdefault("total", math.floor(len(loader) * portion))
                progress_kw.setdefault("description", f"[{INFERENCE_COLOR}]inference")
                flags.progress_task = progress.add_task(**progress_kw)
            is_stream_metric = isinstance(metrics, IStreamMetric) or (
                isinstance(metrics, MultipleMetrics) and metrics.has_streaming
            )
            metrics_requires_all = metrics is not None and metrics.requires_all
            if metrics_requires_all and (
                accelerator is None or accelerator.is_local_main_process
            ):
                console.warn(
                    "detected `requires_all` metrics, it is recommended to implement "
                    "an `IStreamMetric` version to reduce memory footprint."
                )
            gather_outputs = return_outputs or metrics_requires_all
            remainder = -1
            if is_stream_metric:
                metrics.reset()  # type: ignore
            for i, tensor_batch in iterator:
                if i / len(loader) >= portion:
                    break
                if i == 0 and accelerator is not None:
                    remainder = accelerator.gradient_state.remainder
                tensor_outputs = None
                if self.onnx is not None:
                    # will not consider distributed stuffs at onnx inference
                    tensor_batch = recover_labels_of(tensor_batch)
                    np_batch = tensor_batch_to_np(tensor_batch)
                    np_outputs = self.onnx.predict(np_batch)
                    tensor_outputs = np_batch_to_tensor(np_outputs)
                    tensor_outputs = recover_predictions_of(tensor_outputs)
                elif self.model is not None:
                    # accelerator will handle the device stuffs
                    if accelerator is None:
                        tensor_batch = to_device(tensor_batch, device)
                    tensor_batch = recover_labels_of(tensor_batch)
                    flags.in_step = True
                    with no_sync_context(accelerator, self.model):
                        step_outputs = self.model.step(
                            i,
                            tensor_batch,
                            shallow_copy_dict(kwargs),
                            get_losses=use_losses_as_metrics,
                            recover_predictions_fn=recover_predictions_of,
                        )
                    flags.in_step = False
                    tensor_outputs = step_outputs.forward_results
                    if use_losses_as_metrics:
                        for k, v in step_outputs.loss_tensors.items():
                            loss_tensors_lists.setdefault(k, []).append(v)
                assert tensor_outputs is not None
                # metrics
                if metrics is not None and not metrics.requires_all:
                    metric_outputs = metrics.evaluate(tensor_batch, tensor_outputs)
                    if metric_outputs is not None:
                        metric_outputs_list.append(metric_outputs)
                    if is_stream_metric:
                        metrics.update(tensor_batch, tensor_outputs)  # type: ignore
                # gather
                batch_inputs: tensor_dict_type = {}
                if gather_outputs:
                    if metrics_requires_all:
                        for k, v in tensor_batch.items():
                            if v is not None and metrics.requires(k):  # type: ignore
                                v_cpu = v.cpu()
                                batch_inputs[k] = v_cpu
                                all_inputs.setdefault(k, []).append(v_cpu)
                    for k, v in tensor_outputs.items():
                        if v is not None and (
                            k in target_outputs
                            or (metrics_requires_all and metrics.requires(k))  # type: ignore
                        ):
                            all_outputs.setdefault(k, []).append(v.cpu())
                if return_labels:
                    for k, v in tensor_batch.items():
                        if v is not None and k in target_labels:
                            v_cpu = batch_inputs.get(k)
                            if v_cpu is None:
                                v_cpu = v.cpu()
                            all_labels.setdefault(k, []).append(v_cpu)
                # progress
                if progress is not None and flags.progress_task is not None:
                    progress.advance(flags.progress_task)
            cleanup_progress()

            # gather
            is_rank_0 = accelerator is None or accelerator.is_main_process
            need_concat = concat_outputs or metrics_requires_all
            if not need_concat:
                concated_inputs = concated_outputs = concated_labels = None
            else:
                if not metrics_requires_all:
                    concated_inputs = None
                else:
                    concated_inputs = concat(all_inputs)
                concated_outputs = concat(all_outputs)
                concated_labels = concat(all_labels)
            # gather metric outputs
            if metrics is None:
                final_metric_outputs = None
            else:
                to_be_broadcasted: List[Optional[MetricsOutputs]]
                if metrics_requires_all:
                    if not is_rank_0:
                        to_be_broadcasted = [None]
                    else:
                        assert concated_inputs is not None
                        assert concated_outputs is not None
                        to_be_broadcasted = [
                            metrics.evaluate(concated_inputs, concated_outputs, loader)
                        ]
                else:
                    reduced: Optional[MetricsOutputs] = None
                    if accelerator is not None:
                        metric_outputs_lists = gather_object([metric_outputs_list])
                        metric_outputs_list = []
                        for mol in metric_outputs_lists:
                            metric_outputs_list.extend(mol)
                    if metric_outputs_list:
                        reduced = MetricsOutputs.reduce(metric_outputs_list)
                    if is_stream_metric:
                        if isinstance(metrics, MultipleMetrics):
                            stream_outputs = metrics.finalize()
                        else:
                            stream_outputs = metrics.report(metrics.finalize())  # type: ignore
                        if reduced is None:
                            reduced = stream_outputs
                        else:
                            reduced = reduced.union(stream_outputs)
                    if reduced is None:
                        raise RuntimeError("no metric outputs found")
                    if not is_rank_0:
                        to_be_broadcasted = [None]
                    else:
                        to_be_broadcasted = [reduced]
                to_be_broadcasted = broadcast_object_list(to_be_broadcasted)
                final_metric_outputs = to_be_broadcasted[0]
            # handle accelerator stuffs
            if accelerator is not None:
                accelerator.wait_for_everyone()
                for k, vl in loss_tensors_lists.items():
                    vg = accelerator.gather(vl)
                    if remainder > 0:
                        vg[-1] = vg[-1][:remainder]
                    loss_tensors_lists[k] = vg

            return InferenceOutputs(
                concated_outputs if concat_outputs else all_outputs,  # type: ignore
                concated_labels if return_labels else all_labels,  # type: ignore
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

        def run() -> InferenceOutputs:
            ctx: ContextManager
            if self.model is None:
                ctx = nullcontext()
            else:
                ctx_kw = dict(use_grad=use_grad, use_inference=use_inference_mode)
                ctx = self.model.eval_context(**ctx_kw)
            with ctx:
                return _run()

        flags = Flags()
        use_grad = kwargs.pop("use_grad", self.use_grad_in_predict)
        if isinstance(target_outputs, str):
            target_outputs = [target_outputs]
        if isinstance(target_labels, str):
            target_labels = [target_labels]
        try:
            return run()
        except KeyboardInterrupt:
            raise
        except:
            if not flags.in_step:
                raise
            use_grad = self.use_grad_in_predict = True
            cleanup_progress()
            return run()


__all__ = [
    "Inference",
]
