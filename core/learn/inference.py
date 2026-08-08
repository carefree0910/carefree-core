import math
import torch

from abc import abstractmethod
from abc import ABC
from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Callable
from typing import Optional
from typing import Sequence
from typing import ContextManager
from itertools import islice
from accelerate import Accelerator
from contextlib import nullcontext
from dataclasses import field
from dataclasses import dataclass
from rich.progress import TaskID
from rich.progress import Progress
from accelerate.utils import gather_object

from .schema import IModel
from .schema import IMetric
from .schema import InjectFn
from .schema import DataLoader
from .schema import IInference
from .schema import StepOutputs
from .schema import MetricResult
from .schema import IStreamMetric
from .schema import MultipleMetrics
from .schema import InferenceOutputs
from .schema import MetricAccumulator
from .toolkit import get_device
from .toolkit import np_batch_to_tensor
from .toolkit import tensor_batch_to_np
from .toolkit import ONNX
from .constants import LABEL_KEY
from .constants import INFERENCE_COLOR
from .constants import PREDICTIONS_KEY
from ..toolkit import console
from ..toolkit.misc import shallow_copy_dict
from ..toolkit.array import is_int
from ..toolkit.array import to_device
from ..toolkit.types import tensor_dict_type

TTensors = Dict[str, List[Union[Tensor, Any]]]
RecoverPredictionsFn = Callable[[tensor_dict_type], tensor_dict_type]


def _get_sample_count(tensor_batch: tensor_dict_type) -> float:
    candidates = [
        tensor_batch.get(LABEL_KEY),
        *tensor_batch.values(),
    ]
    for candidate in candidates:
        if isinstance(candidate, Tensor) and candidate.ndim > 0:
            return float(candidate.shape[0])
    return 1.0


def _get_num_batches(loader_length: int, portion: float) -> int:
    if loader_length == 0 or portion <= 0.0:
        return 0
    if portion >= 1.0 or math.isnan(portion):
        return loader_length
    return math.ceil(loader_length * portion)


def _gather_objects(objects: List[Any]) -> List[Any]:
    # Object collectives mutate their internal tensors, which is not allowed when
    # those tensors are created by an enclosing `inference_mode` context.
    with torch.inference_mode(False):
        gathered: List[Any] = gather_object(objects)
    return gathered


def _merge_metrics(
    accelerator: Optional[Accelerator],
    accumulator: MetricAccumulator,
    stream_metrics: List[IStreamMetric[Any]],
) -> MetricAccumulator:
    if accelerator is None:
        return accumulator
    states = [metric.get_distributed_state() for metric in stream_metrics]
    gathered = _gather_objects([(accumulator, states)])
    merged = MetricAccumulator()
    for other, _ in gathered:
        merged.merge(other)
    for i, metric in enumerate(stream_metrics):
        metric_states = [rank_states[i] for _, rank_states in gathered]
        if all(state is not None for state in metric_states):
            metric.merge_distributed_states(metric_states)
    return merged


def no_sync_context(
    accelerator: Optional[Accelerator],
    model: IModel,
) -> ContextManager:
    if accelerator is None:
        return nullcontext()
    return accelerator.no_sync(model.m)


@dataclass
class InferenceRequest:
    loader: DataLoader
    portion: float = 1.0
    metrics: Optional[IMetric] = None
    use_losses_as_metrics: bool = False
    return_outputs: bool = True
    target_outputs: Sequence[str] = (PREDICTIONS_KEY,)
    target_inputs: Optional[List[str]] = None
    recover_labels: bool = True
    recover_predictions: bool = True
    return_labels: bool = False
    target_labels: Sequence[str] = (LABEL_KEY,)
    inject_outputs_fn: Optional[InjectFn] = None
    concat_outputs: bool = True
    progress: Optional[Progress] = None
    progress_kwargs: Optional[Dict[str, Any]] = None
    should_stop_progress: bool = True
    use_grad: Optional[bool] = False
    use_inference_mode: Optional[bool] = None
    accelerator: Optional[Accelerator] = None
    pad_dim: Optional[Union[int, Dict[str, int]]] = None
    verbose: bool = True
    forward_kwargs: Dict[str, Any] = field(default_factory=dict)


class OutputAccumulator:
    request: InferenceRequest
    values: Dict[str, List[Any]]

    def __init__(self, request: InferenceRequest) -> None:
        self.request = request
        self.values = {}

    @staticmethod
    def _to_cpu(value: Any) -> Any:
        if isinstance(value, Tensor):
            return value.cpu()
        if isinstance(value, dict):
            return {k: OutputAccumulator._to_cpu(v) for k, v in value.items()}
        if isinstance(value, list):
            return [OutputAccumulator._to_cpu(v) for v in value]
        if isinstance(value, tuple):
            return tuple(OutputAccumulator._to_cpu(v) for v in value)
        return value

    def add(self, key: str, value: Any) -> Any:
        value = self._to_cpu(value)
        self.values.setdefault(key, []).append(value)
        return value

    def _get_pad_dim(self, key: str) -> Optional[int]:
        pad_dim = self.request.pad_dim
        if pad_dim is None:
            return None
        return pad_dim if isinstance(pad_dim, int) else pad_dim.get(key)

    def _pad(self, key: str, tensors: List[Tensor]) -> List[Tensor]:
        pad_dim = self._get_pad_dim(key)
        if pad_dim is None:
            return tensors
        num_dim = tensors[0].ndim
        if not -num_dim <= pad_dim < num_dim:
            raise ValueError(
                f"invalid `pad_dim` {pad_dim} for '{key}' with {num_dim} dimensions"
            )
        dim = pad_dim % num_dim
        if dim == 0:
            return tensors
        max_shape = max(tensor.shape[dim] for tensor in tensors)
        if all(tensor.shape[dim] == max_shape for tensor in tensors):
            return tensors
        if self.request.verbose:
            accelerator = self.request.accelerator
            rank = 0 if accelerator is None else accelerator.process_index
            console.warn(
                rf"\[rank {rank}] padding '{key}' at dim {dim} to {max_shape}, please perform "
                "post-processing to remove the paddings if necessary."
            )
        padded = []
        for tensor in tensors:
            shape = list(tensor.shape)
            shape[dim] = max_shape
            if is_int(tensor):
                new = tensor.new_zeros(shape)
            else:
                new = tensor.new_full(shape, torch.nan)
            slices = [slice(None)] * tensor.ndim
            slices[dim] = slice(0, tensor.shape[dim])
            new[tuple(slices)] = tensor
            padded.append(new)
        return padded

    def finalize(self) -> tensor_dict_type:
        outputs: tensor_dict_type = {}
        for key, values in self.values.items():
            if not all(isinstance(value, Tensor) for value in values):
                outputs[key] = values
                continue
            tensors = [value for value in values if isinstance(value, Tensor)]
            outputs[key] = torch.cat(self._pad(key, tensors))
        return outputs


class InferenceExecutor(ABC):
    def context(self, request: InferenceRequest) -> ContextManager[Any]:
        return nullcontext()

    def setup(self) -> None:
        pass

    def prepare_batch(
        self,
        tensor_batch: tensor_dict_type,
        request: InferenceRequest,
    ) -> tensor_dict_type:
        return tensor_batch

    @abstractmethod
    def execute(
        self,
        batch_idx: int,
        tensor_batch: tensor_dict_type,
        request: InferenceRequest,
        recover_predictions_fn: RecoverPredictionsFn,
    ) -> StepOutputs:
        """execute one inference step"""


class NativeExecutor(InferenceExecutor):
    model: IModel
    device: torch.device

    def __init__(self, model: IModel) -> None:
        self.model = model

    def context(self, request: InferenceRequest) -> ContextManager[Any]:
        return self.model.eval_context(
            use_grad=request.use_grad,
            use_inference=request.use_inference_mode,
        )

    def setup(self) -> None:
        self.device = get_device(self.model.m)

    def prepare_batch(
        self,
        tensor_batch: tensor_dict_type,
        request: InferenceRequest,
    ) -> tensor_dict_type:
        if request.accelerator is None:
            return to_device(tensor_batch, self.device)
        return tensor_batch

    def execute(
        self,
        batch_idx: int,
        tensor_batch: tensor_dict_type,
        request: InferenceRequest,
        recover_predictions_fn: RecoverPredictionsFn,
    ) -> StepOutputs:
        with no_sync_context(request.accelerator, self.model):
            return self.model.step(
                batch_idx,
                tensor_batch,
                shallow_copy_dict(request.forward_kwargs),
                get_losses=request.use_losses_as_metrics,
                inject_outputs_fn=request.inject_outputs_fn,
                recover_predictions_fn=recover_predictions_fn,
            )


class ONNXExecutor(InferenceExecutor):
    onnx: ONNX

    def __init__(self, onnx: ONNX) -> None:
        self.onnx = onnx

    def execute(
        self,
        batch_idx: int,
        tensor_batch: tensor_dict_type,
        request: InferenceRequest,
        recover_predictions_fn: RecoverPredictionsFn,
    ) -> StepOutputs:
        np_batch = tensor_batch_to_np(tensor_batch)
        np_outputs = self.onnx.predict(np_batch)
        tensor_outputs = np_batch_to_tensor(np_outputs)
        if request.inject_outputs_fn is not None:
            request.inject_outputs_fn(tensor_batch, tensor_outputs)
        return StepOutputs(recover_predictions_fn(tensor_outputs), {})


class InferenceRunner:
    request: InferenceRequest
    executor: InferenceExecutor
    progress_task: Optional[TaskID]

    def __init__(
        self,
        request: InferenceRequest,
        executor: InferenceExecutor,
    ) -> None:
        self.request = request
        self.executor = executor
        self.progress_task = None

    def _recover_labels(self, tensors: tensor_dict_type) -> tensor_dict_type:
        if self.request.recover_labels:
            tensors = shallow_copy_dict(tensors)
            for k, v in tensors.items():
                if v is not None and k in self.request.target_labels:
                    tensors[k] = self.request.loader.recover_labels(k, v)
        return tensors

    def _recover_predictions(self, tensors: tensor_dict_type) -> tensor_dict_type:
        if self.request.recover_predictions:
            tensors = shallow_copy_dict(tensors)
            for k, v in tensors.items():
                if v is not None and isinstance(v, Tensor):
                    tensors[k] = self.request.loader.recover_labels(k, v)
        return tensors

    def _cleanup_progress(self) -> None:
        progress = self.request.progress
        if progress is None or self.progress_task is None:
            return
        progress_task = self.progress_task
        self.progress_task = None
        if self.request.should_stop_progress:
            try:
                progress.stop()
            except:
                pass
        try:
            progress.remove_task(progress_task)
        except:
            pass

    def _run(self) -> InferenceOutputs:
        request = self.request
        self.executor.setup()
        loader = request.loader
        metrics = request.metrics
        accelerator = request.accelerator
        inputs = OutputAccumulator(request)
        labels = OutputAccumulator(request)
        outputs = OutputAccumulator(request)
        accumulator = MetricAccumulator()
        losses: TTensors = {}

        num_batches = _get_num_batches(len(loader), request.portion)
        iterator = islice(enumerate(loader), num_batches)
        if request.progress is not None:
            progress_kw = shallow_copy_dict(request.progress_kwargs or {})
            progress_kw.setdefault("total", num_batches)
            progress_kw.setdefault("description", f"[{INFERENCE_COLOR}]inference")
            self.progress_task = request.progress.add_task(**progress_kw)
        stream_metrics: List[IStreamMetric[Any]]
        if isinstance(metrics, IStreamMetric):
            stream_metrics = [metrics]
        elif isinstance(metrics, MultipleMetrics):
            stream_metrics = [
                m for m in metrics.metrics if isinstance(m, IStreamMetric)
            ]
        else:
            stream_metrics = []
        metrics_requires_all = metrics is not None and metrics.requires_all
        if metrics_requires_all and (
            accelerator is None or accelerator.is_local_main_process
        ):
            console.warn(
                "detected `requires_all` metrics, it is recommended to implement "
                "an `IStreamMetric` version to reduce memory footprint."
            )
        collect_outputs = request.return_outputs or metrics_requires_all
        remainder = -1
        if stream_metrics:
            metrics.reset()  # type: ignore
        for i, tensor_batch in iterator:
            if i == 0 and accelerator is not None:
                remainder = accelerator.gradient_state.remainder
            tensor_batch = self.executor.prepare_batch(tensor_batch, request)
            tensor_batch = self._recover_labels(tensor_batch)
            step_outputs = self.executor.execute(
                i,
                tensor_batch,
                request,
                self._recover_predictions,
            )
            tensor_outputs = step_outputs.forward_results
            if request.use_losses_as_metrics:
                for k, v in step_outputs.loss_tensors.items():
                    losses.setdefault(k, []).append(v)
            # metrics
            if metrics is not None and not metrics.requires_all:
                if not isinstance(metrics, IStreamMetric):
                    metric_outputs = metrics.evaluate(tensor_batch, tensor_outputs)
                    if metric_outputs is not None:
                        accumulator.add(
                            MetricResult(
                                metric_outputs,
                                _get_sample_count(tensor_batch),
                            )
                        )
                if stream_metrics:
                    metrics.update(tensor_batch, tensor_outputs)  # type: ignore
            # collect
            batch_inputs: tensor_dict_type = {}
            if collect_outputs:
                if metrics_requires_all:
                    for k, v in tensor_batch.items():
                        if v is not None and metrics.requires(k):  # type: ignore
                            v_cpu = inputs.add(k, v)
                            batch_inputs[k] = v_cpu
                for k, v in tensor_outputs.items():
                    if v is not None and (
                        k in request.target_outputs
                        or (metrics_requires_all and metrics.requires(k))  # type: ignore
                    ):
                        outputs.add(k, v)
            if request.target_inputs is not None:
                for k in request.target_inputs:
                    outputs.add(k, tensor_batch[k])
            if request.return_labels:
                for k, v in tensor_batch.items():
                    if v is not None and k in request.target_labels:
                        v_cpu = batch_inputs.get(k)
                        if v_cpu is None:
                            v_cpu = v
                        labels.add(k, v_cpu)
            # progress
            if request.progress is not None and self.progress_task is not None:
                request.progress.advance(self.progress_task)
        self._cleanup_progress()

        # finalize
        need_concat = request.concat_outputs or metrics_requires_all
        joined_inputs = inputs.finalize() if metrics_requires_all else None
        joined_outputs = outputs.finalize() if need_concat else None
        joined_labels = labels.finalize() if request.return_labels else None
        # gather metric outputs
        final_metric_outputs = None
        if metrics is not None:
            if metrics_requires_all:
                assert joined_inputs is not None
                assert joined_outputs is not None
                mo = metrics.evaluate(joined_inputs, joined_outputs, loader)
                if mo is not None:
                    accumulator.add(MetricResult(mo))
            accumulator = _merge_metrics(accelerator, accumulator, stream_metrics)
            reduced = accumulator.finalize()
            if stream_metrics:
                if isinstance(metrics, MultipleMetrics):
                    stream_outputs = metrics.finalize()
                else:
                    stream_outputs = metrics.report(metrics.finalize())  # type: ignore
                if reduced is None:
                    reduced = stream_outputs
                else:
                    assert isinstance(metrics, MultipleMetrics)
                    reduced = reduced.union(
                        stream_outputs,
                        weight=metrics._get_score_weight(for_streaming=False),
                        other_weight=metrics._get_score_weight(for_streaming=True),
                    )
            if reduced is None:
                raise RuntimeError("no metric outputs found")
            final_metric_outputs = reduced
        # handle accelerator stuffs
        if accelerator is not None:
            accelerator.wait_for_everyone()
            with torch.inference_mode(False):
                for k, vl in losses.items():
                    vg = accelerator.gather([torch.atleast_1d(v) for v in vl])
                    if remainder > 0:
                        vg[-1] = vg[-1][:remainder]
                    losses[k] = vg

        final_outputs: Union[tensor_dict_type, TTensors]
        if request.concat_outputs:
            assert joined_outputs is not None
            final_outputs = joined_outputs
        else:
            final_outputs = outputs.values
        final_labels: Union[tensor_dict_type, TTensors]
        if request.return_labels:
            assert joined_labels is not None
            final_labels = joined_labels
        else:
            final_labels = labels.values
        return InferenceOutputs(
            final_outputs,
            final_labels,
            final_metric_outputs,
            (
                None
                if not request.use_losses_as_metrics
                else {k: torch.hstack(v).mean().item() for k, v in losses.items()}
            ),
        )

    def run(self) -> InferenceOutputs:
        try:
            with self.executor.context(self.request):
                return self._run()
        except BaseException:
            self._cleanup_progress()
            raise


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
        target_inputs: Optional[List[str]] = None,
        recover_labels: bool = True,
        recover_predictions: bool = True,
        return_labels: bool = False,
        target_labels: Union[str, List[str]] = LABEL_KEY,
        inject_outputs_fn: Optional[InjectFn] = None,
        concat_outputs: bool = True,
        progress: Optional[Progress] = None,
        progress_kwargs: Optional[Dict[str, Any]] = None,
        should_stop_progress: bool = True,
        use_inference_mode: Optional[bool] = None,
        accelerator: Optional[Accelerator] = None,
        pad_dim: Optional[Union[int, Dict[str, int]]] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> InferenceOutputs:
        use_grad = kwargs.pop("use_grad", self.use_grad_in_predict)
        request = InferenceRequest(
            loader,
            portion=portion,
            metrics=metrics,
            use_losses_as_metrics=use_losses_as_metrics,
            return_outputs=return_outputs,
            target_outputs=(
                [target_outputs] if isinstance(target_outputs, str) else target_outputs
            ),
            target_inputs=target_inputs,
            recover_labels=recover_labels,
            recover_predictions=recover_predictions,
            return_labels=return_labels,
            target_labels=(
                [target_labels] if isinstance(target_labels, str) else target_labels
            ),
            inject_outputs_fn=inject_outputs_fn,
            concat_outputs=concat_outputs,
            progress=progress,
            progress_kwargs=progress_kwargs,
            should_stop_progress=should_stop_progress,
            use_grad=use_grad,
            use_inference_mode=use_inference_mode,
            accelerator=accelerator,
            pad_dim=pad_dim,
            verbose=verbose,
            forward_kwargs=kwargs,
        )
        executor: InferenceExecutor
        if self.onnx is not None:
            executor = ONNXExecutor(self.onnx)
        else:
            assert self.model is not None
            executor = NativeExecutor(self.model)
        return InferenceRunner(request, executor).run()


__all__ = [
    "Inference",
]
