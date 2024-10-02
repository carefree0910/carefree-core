import os
import torch
import shutil

import numpy as np

from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Union
from typing import TypeVar
from typing import Callable
from typing import Optional
from pathlib import Path
from tempfile import mkdtemp
from tempfile import TemporaryDirectory
from accelerate import Accelerator
from collections import OrderedDict
from rich.progress import Progress

from .common import Block
from .common import Pipeline
from .blocks import SetDefaultsBlock
from .blocks import PrepareWorkspaceBlock
from .blocks import ExtractStateInfoBlock
from .blocks import BuildModelBlock
from .blocks import BuildMetricsBlock
from .blocks import BuildInferenceBlock
from .blocks import SetTrainerDefaultsBlock
from .blocks import BuildMonitorsBlock
from .blocks import BuildCallbacksBlock
from .blocks import BuildOptimizersBlock
from .blocks import BuildTrainerBlock
from .blocks import RecordNumSamplesBlock
from .blocks import ReportBlock
from .blocks import TrainingBlock
from .blocks import SerializeDataBlock
from .blocks import SerializeModelBlock
from .blocks import SerializeOptimizerBlock
from .blocks import SerializeScriptBlock
from .schema import IEvaluationPipeline
from ..models import EnsembleModel
from ..schema import device_type
from ..schema import sample_weights_type
from ..schema import IData
from ..schema import Config
from ..schema import DataLoader
from ..schema import InferenceOutputs
from ..toolkit import get_device
from ..toolkit import get_torch_device
from ..trainer import SortMethod
from ..trainer import get_scores
from ..trainer import get_sorted_checkpoints
from ..constants import LABEL_KEY
from ..constants import PREDICTIONS_KEY
from ..constants import CHECKPOINTS_FOLDER
from ...toolkit import console
from ...toolkit.misc import compress as compress_folder
from ...toolkit.misc import track
from ...toolkit.misc import to_path
from ...toolkit.misc import safe_execute
from ...toolkit.misc import is_local_rank_0
from ...toolkit.misc import shallow_copy_dict
from ...toolkit.misc import prepare_workspace_from
from ...toolkit.misc import Serializer
from ...toolkit.array import sigmoid
from ...toolkit.array import softmax
from ...toolkit.array import is_float
from ...toolkit.types import TPath
from ...toolkit.types import np_dict_type
from ...toolkit.types import tensor_dict_type
from ...toolkit.pipeline import get_folder


TInferPipeline = TypeVar("TInferPipeline", bound="InferencePipeline", covariant=True)

states_callback_type = Optional[Callable[[Any, Dict[str, Any]], Dict[str, Any]]]


# internal mixins


class _DeviceMixin:
    build_model: BuildModelBlock

    @property
    def device(self) -> torch.device:
        return self.build_model.model.device


class _InferenceMixin:
    focuses: List[Type[Block]]
    is_built: bool

    data: Optional[IData]
    get_block: Callable[[Type[Block]], Any]
    try_get_block: Callable[[Type[Block]], Any]

    # optional callbacks

    def predict_callback(self, results: np_dict_type) -> np_dict_type:
        """changes can happen inplace"""
        return results

    # api

    @property
    def build_model(self) -> BuildModelBlock:
        return self.get_block(BuildModelBlock)

    @property
    def build_inference(self) -> BuildInferenceBlock:
        return self.get_block(BuildInferenceBlock)

    @property
    def serialize_data(self) -> SerializeDataBlock:
        return self.get_block(SerializeDataBlock)

    @property
    def serialize_model(self) -> Optional[SerializeModelBlock]:
        return self.try_get_block(SerializeModelBlock)

    @classmethod
    def build_with(  # type: ignore
        cls: Type[TInferPipeline],
        config: Config,
        states: Optional[tensor_dict_type] = None,
        *,
        data: Optional[IData] = None,
    ) -> TInferPipeline:
        self = cls.init(config)
        # last focus will be the serialization block
        self.build(*[Block.make(b.__identifier__, {}) for b in cls.focuses])  # type: ignore
        if states is not None:
            self.build_model.model.load_state_dict(states)
        if self.serialize_model is not None:
            self.serialize_model.verbose = False
        self.serialize_data.data = self.data = data
        self.is_built = True
        return self  # type: ignore

    def to(  # type: ignore
        self: TInferPipeline,
        device: Union[int, str, torch.device],
    ) -> TInferPipeline:
        self.build_model.model.to(device)
        return self

    def predict(
        self,
        loader: DataLoader,
        *,
        return_classes: bool = False,
        binary_threshold: float = 0.5,
        return_probabilities: bool = False,
        target_outputs: Union[str, List[str]] = PREDICTIONS_KEY,
        recover_labels: bool = True,
        accelerator: Optional[Accelerator] = None,
        pad_dim: Optional[Union[int, Dict[str, int]]] = None,
        **kwargs: Any,
    ) -> np_dict_type:
        if not self.is_built:
            raise RuntimeError(
                f"`{self.__class__.__name__}` should be built beforehand, please use "
                "`PipelineSerializer.load_inference/evaluation` or `build_with` "
                "to get a built one!"
            )
        kw = shallow_copy_dict(kwargs)
        kw["loader"] = loader
        kw["target_outputs"] = target_outputs
        kw["recover_labels"] = recover_labels
        kw["accelerator"] = accelerator
        kw["pad_dim"] = pad_dim
        outputs = safe_execute(self.build_inference.inference.get_outputs, kw)
        results = outputs.forward_results
        # handle predict flags
        if return_classes and return_probabilities:
            raise ValueError(
                "`return_classes` & `return_probabilities`"
                "should not be True at the same time"
            )
        elif not return_classes and not return_probabilities:
            pass
        else:
            predictions = results[PREDICTIONS_KEY]
            if predictions.shape[1] > 2 and return_classes:
                results[PREDICTIONS_KEY] = predictions.argmax(1, keepdims=True)  # type: ignore
            else:
                if predictions.shape[1] == 2:
                    probabilities = softmax(predictions)
                else:
                    pos = sigmoid(predictions)
                    probabilities = np.hstack([1.0 - pos, pos])
                if return_probabilities:
                    results[PREDICTIONS_KEY] = probabilities
                else:
                    classes = (probabilities[..., [1]] >= binary_threshold).astype(int)
                    results[PREDICTIONS_KEY] = classes
        # optional callback
        results = self.predict_callback(results)
        # return
        return results

    def prepare_distributed_with(self, accelerator: Accelerator) -> None:
        all_modules = self.build_model.model.all_modules
        ms = accelerator.prepare(*all_modules)
        if len(all_modules) == 1:
            ms = [ms]
        self.build_model.model.from_accelerator(*ms)


class _EvaluationMixin(_InferenceMixin, IEvaluationPipeline):
    config: Config

    @property
    def build_metrics(self) -> BuildMetricsBlock:
        return self.get_block(BuildMetricsBlock)

    def evaluate(
        self,
        loader: DataLoader,
        *,
        portion: float = 1.0,
        return_outputs: bool = False,
        target_outputs: Union[str, List[str]] = PREDICTIONS_KEY,
        recover_labels: bool = True,
        return_labels: bool = False,
        target_labels: Union[str, List[str]] = LABEL_KEY,
        progress: Optional[Progress] = None,
        progress_kwargs: Optional[Dict[str, Any]] = None,
        use_inference_mode: Optional[bool] = None,
        accelerator: Optional[Accelerator] = None,
        pad_dim: Optional[Union[int, Dict[str, int]]] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> InferenceOutputs:
        return self.build_model.model.evaluate(
            self.config,
            self.build_metrics.metrics,
            self.build_inference.inference,
            loader,
            portion=portion,
            return_outputs=return_outputs,
            target_outputs=target_outputs,
            recover_labels=recover_labels,
            return_labels=return_labels,
            target_labels=target_labels,
            progress=progress,
            progress_kwargs=progress_kwargs,
            use_inference_mode=use_inference_mode,
            accelerator=accelerator,
            pad_dim=pad_dim,
            verbose=verbose,
            **kwargs,
        )


# apis


class PipelineTypes(str, Enum):
    TRAINING = "training"
    INFERENCE = "inference"
    EVALUATION = "evaluation"


@Pipeline.register(PipelineTypes.TRAINING)
class TrainingPipeline(Pipeline["TrainingPipeline"], _DeviceMixin, _EvaluationMixin):  # type: ignore
    is_built = False

    @property
    def building_blocks(self) -> List[Block]:
        return [
            SetDefaultsBlock(),
            PrepareWorkspaceBlock(),
            ExtractStateInfoBlock(),
            BuildModelBlock(),
            BuildMetricsBlock(),
            BuildInferenceBlock(),
            SetTrainerDefaultsBlock(),
            BuildMonitorsBlock(),
            BuildCallbacksBlock(),
            BuildOptimizersBlock(),
            BuildTrainerBlock(),
            RecordNumSamplesBlock(),
            ReportBlock(),
            SerializeDataBlock(),
            SerializeModelBlock(),
            SerializeOptimizerBlock(),
            SerializeScriptBlock(),
            TrainingBlock(),
        ]

    @property
    def training(self) -> TrainingBlock:
        return self.get_block(TrainingBlock)

    def after_load(self) -> None:
        self.is_built = True
        workspace = prepare_workspace_from("_logs")
        self.config.workspace = workspace

    def prepare(self, data: IData, sample_weights: sample_weights_type = None) -> None:
        self.data = data.set_sample_weights(sample_weights)
        self.training_workspace = self.config.workspace
        if not self.is_built:
            self.build(*self.building_blocks)
            self.is_built = True
        else:
            for block in self.blocks:
                block.training_workspace = self.training_workspace

    def fit(
        self,
        data: IData,
        *,
        sample_weights: sample_weights_type = None,
        skip_final_evaluation: bool = False,
        only_touch: bool = False,
        device: device_type = None,
    ) -> "TrainingPipeline":
        # build pipeline
        self.prepare(data, sample_weights)
        # check rank 0
        workspace = self.config.workspace if is_local_rank_0() else None
        # save data info
        if workspace is not None:
            Serializer.save(
                os.path.join(workspace, SerializeDataBlock.package_folder),
                data,
                save_npd=False,
            )
        # run pipeline
        self.run(
            data,
            skip_final_evaluation=skip_final_evaluation,
            only_touch=only_touch,
            device=device,
        )
        # save / update pipeline serialization
        if workspace is not None:
            if not self.config.save_pipeline_in_realtime:
                PipelineSerializer.save(self, workspace)
            else:
                PipelineSerializer.update(self, workspace)
        # return
        return self


@Pipeline.register(PipelineTypes.INFERENCE)
class InferencePipeline(Pipeline["InferencePipeline"], _DeviceMixin, _InferenceMixin):
    is_built = False

    focuses = [
        BuildModelBlock,
        BuildInferenceBlock,
        SerializeDataBlock,
        SerializeModelBlock,
    ]

    def after_load(self) -> None:
        self.is_built = True
        self.data = self.serialize_data.data
        if self.serialize_model is not None:
            self.serialize_model.verbose = False


@Pipeline.register(PipelineTypes.EVALUATION)
class EvaluationPipeline(InferencePipeline, _EvaluationMixin):  # type: ignore
    focuses = [
        BuildModelBlock,
        BuildMetricsBlock,
        BuildInferenceBlock,
        SerializeDataBlock,
        SerializeModelBlock,
    ]


class PackType(str, Enum):
    TRAINING = "training"
    INFERENCE = "inference"
    EVALUATION = "evaluation"


class PipelineSerializer:
    id_file = "id.txt"
    config_file = "config.json"
    blocks_file = "blocks.json"
    pipeline_folder = "pipeline"

    # api

    @classmethod
    def save(
        cls,
        pipeline: Pipeline,
        workspace: str,
        *,
        compress: bool = False,
        verbose: bool = True,
    ) -> None:
        folder = os.path.join(workspace, cls.pipeline_folder)
        cls._save(pipeline, folder, compress=compress, verbose=verbose)

    @classmethod
    def update(cls, p: Pipeline, workspace: str, verbose: bool = True) -> None:
        folder = os.path.join(workspace, cls.pipeline_folder)
        if os.path.isdir(folder):
            compress = False
            shutil.rmtree(folder)
        elif os.path.isfile(f"{folder}.zip"):
            compress = True
            os.remove(f"{folder}.zip")
        else:
            raise ValueError(f"neither `{folder}` nor `{folder}.zip` exists")
        cls._save(p, folder, compress=compress, verbose=verbose)

    @classmethod
    def pack(
        cls,
        workspace: str,
        export_folder: str,
        *,
        pack_type: PackType = PackType.INFERENCE,
        compress: bool = True,
    ) -> None:
        excludes: Optional[List[Type[Block]]]
        if pack_type == PackType.TRAINING:
            swap_id = None
            focuses = None
            excludes = [PrepareWorkspaceBlock]
        elif pack_type == PackType.INFERENCE:
            swap_id = InferencePipeline.__identifier__
            focuses = InferencePipeline.focuses
            excludes = None
        elif pack_type == PackType.EVALUATION:
            swap_id = EvaluationPipeline.__identifier__
            focuses = EvaluationPipeline.focuses
            excludes = None
        else:
            raise ValueError(f"unrecognized `pack_type` '{pack_type}' occurred")
        pipeline_folder = os.path.join(workspace, cls.pipeline_folder)
        pipeline = cls._load(
            pipeline_folder,
            swap_id=swap_id,
            focuses=focuses,
            excludes=excludes,
        )
        cls._save(pipeline, export_folder, compress=compress)

    @classmethod
    def pack_and_load_inference(cls, workspace: str) -> InferencePipeline:
        with TemporaryDirectory() as tmp_folder:
            cls.pack(
                workspace,
                export_folder=tmp_folder,
                pack_type=PackType.INFERENCE,
                compress=False,
            )
            return cls._load_inference(tmp_folder)

    @classmethod
    def pack_onnx(
        cls,
        workspace: str,
        export_file: str = "model.onnx",
        dynamic_axes: Optional[Union[List[int], Dict[int, str]]] = None,
        *,
        input_sample: Optional[tensor_dict_type] = None,
        loader_sample: Optional[DataLoader] = None,
        opset: int = 11,
        simplify: bool = True,
        num_samples: Optional[int] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> InferencePipeline:
        p = cls.pack_and_load_inference(workspace)
        model = p.build_model.model
        if input_sample is None:
            if loader_sample is None:
                msg = "either `input_sample` or `loader_sample` should be provided"
                raise ValueError(msg)
            input_sample = loader_sample.get_input_sample(get_device(model.m))
        model.to_onnx(
            export_file,
            input_sample,
            dynamic_axes,
            opset=opset,
            simplify=simplify,
            num_samples=num_samples,
            verbose=verbose,
            **kwargs,
        )
        return p

    @classmethod
    def pack_scripted(
        cls, workspace: str, export_file: str = "model.pt"
    ) -> InferencePipeline:
        p = cls.pack_and_load_inference(workspace)
        model = torch.jit.script(p.build_model.model.m)
        torch.jit.save(model, export_file)
        return p

    @classmethod
    def fuse_inference(
        cls,
        workspaces: List[str],
        *,
        device: device_type = None,
        num_picked: Optional[Union[int, float]] = None,
        states_callback: states_callback_type = None,
        sort_ckpt_by: SortMethod = SortMethod.BEST,
        target_ckpt_step: Optional[int] = None,
    ) -> InferencePipeline:
        src_folders = [os.path.join(w, cls.pipeline_folder) for w in workspaces]
        return cls._fuse_multiple(
            src_folders,
            PackType.INFERENCE,
            device,
            num_picked,
            states_callback,
            sort_ckpt_by,
            target_ckpt_step,
        )

    @classmethod
    def fuse_evaluation(
        cls,
        workspaces: List[str],
        *,
        device: device_type = None,
        num_picked: Optional[Union[int, float]] = None,
        states_callback: states_callback_type = None,
        sort_ckpt_by: SortMethod = SortMethod.BEST,
        target_ckpt_step: Optional[int] = None,
    ) -> EvaluationPipeline:
        src_folders = [os.path.join(w, cls.pipeline_folder) for w in workspaces]
        return cls._fuse_multiple(  # type: ignore
            src_folders,
            PackType.EVALUATION,
            device,
            num_picked,
            states_callback,
            sort_ckpt_by,
            target_ckpt_step,
        )

    @classmethod
    def load_training(cls, workspace: str) -> TrainingPipeline:
        folder = os.path.join(workspace, cls.pipeline_folder)
        swap_id = TrainingPipeline.__identifier__
        return cls._load(folder, swap_id=swap_id)  # type: ignore

    @classmethod
    def load_inference(cls, workspace: str) -> InferencePipeline:
        folder = os.path.join(workspace, cls.pipeline_folder)
        return cls._load_inference(folder)

    @classmethod
    def load_evaluation(cls, workspace: str) -> EvaluationPipeline:
        folder = os.path.join(workspace, cls.pipeline_folder)
        return cls._load_evaluation(folder)

    @classmethod
    def self_ensemble_inference(
        cls,
        n: int,
        workspace: TPath,
        *,
        ensemble_weights: bool = False,
        device: device_type = None,
        states_callback: states_callback_type = None,
        sort_ckpt_by: SortMethod = SortMethod.BEST,
        target_ckpt_step: Optional[int] = None,
        verbose: bool = True,
    ) -> InferencePipeline:
        return cls._self_ensemble(
            workspace,
            PackType.INFERENCE,
            n,
            ensemble_weights,
            device,
            states_callback,
            sort_ckpt_by,
            target_ckpt_step,
            verbose,
        )

    @classmethod
    def self_ensemble_evaluation(
        cls,
        n: int,
        workspace: TPath,
        *,
        ensemble_weights: bool = False,
        device: device_type = None,
        states_callback: states_callback_type = None,
        sort_ckpt_by: SortMethod = SortMethod.BEST,
        target_ckpt_step: Optional[int] = None,
        verbose: bool = True,
    ) -> EvaluationPipeline:
        return cls._self_ensemble(  # type: ignore
            workspace,
            PackType.EVALUATION,
            n,
            ensemble_weights,
            device,
            states_callback,
            sort_ckpt_by,
            target_ckpt_step,
            verbose,
        )

    # internal

    @classmethod
    def _save(
        cls,
        pipeline: Pipeline,
        folder: TPath,
        *,
        compress: bool = False,
        verbose: bool = True,
    ) -> None:
        folder = to_path(folder)
        original_folder = None
        if compress:
            original_folder = folder
            folder = Path(mkdtemp())
        Serializer.save(folder, pipeline)
        with pipeline.verbose_context(verbose):
            for block in pipeline.blocks:
                block.save_extra(folder / block.__identifier__)
        if compress and original_folder is not None:
            absolute_folder = folder.absolute()
            absolute_original = original_folder.absolute()
            compress_folder(absolute_folder)
            shutil.move(f"{absolute_folder}.zip", f"{absolute_original}.zip")

    @classmethod
    def _load(
        cls,
        folder: TPath,
        *,
        swap_id: Optional[str] = None,
        focuses: Optional[List[Type[Block]]] = None,
        excludes: Optional[List[Type[Block]]] = None,
    ) -> Pipeline:
        with get_folder(folder) as folder_path:
            # handle info
            info = Serializer.load_info(folder_path)
            if focuses is not None or excludes is not None:
                if focuses is None:
                    focuses_set = None
                else:
                    focuses_set = {b.__identifier__ for b in focuses}
                block_types = info["blocks"]
                if focuses_set is not None:
                    block_types = [b for b in block_types if b in focuses_set]
                    left = sorted(focuses_set - set(block_types))
                    if left:
                        raise ValueError(
                            "following blocks are specified in `focuses` "
                            f"but not found in the loaded blocks: {', '.join(left)}"
                        )
                if excludes is not None:
                    excludes_set = {b.__identifier__ for b in excludes}
                    block_types = [b for b in block_types if b not in excludes_set]
                info["blocks"] = block_types
            # load
            pipeline = Serializer.load_empty(folder_path, Pipeline, swap_id=swap_id)
            pipeline.serialize_folder = folder_path
            pipeline.from_info(info)
            for block in pipeline.blocks:
                block.load_from(folder_path / block.__identifier__)
            pipeline.after_load()
            # hijacks
            serialize_model = pipeline.try_get_block(SerializeModelBlock)
            if serialize_model is not None:
                serialize_model.ckpt_folder = None
                serialize_model.ckpt_scores = None
        return pipeline

    @classmethod
    def _load_inference(
        cls,
        folder: TPath,
        excludes: Optional[List[Type[Block]]] = None,
    ) -> InferencePipeline:
        return cls._load(  # type: ignore
            folder,
            swap_id=InferencePipeline.__identifier__,
            focuses=InferencePipeline.focuses,
            excludes=excludes,
        )

    @classmethod
    def _load_evaluation(
        cls,
        folder: TPath,
        excludes: Optional[List[Type[Block]]] = None,
    ) -> EvaluationPipeline:
        return cls._load(  # type: ignore
            folder,
            swap_id=EvaluationPipeline.__identifier__,
            focuses=EvaluationPipeline.focuses,
            excludes=excludes,
        )

    @classmethod
    def _build_ensemble_pipeline(
        cls,
        folder: Path,
        pack_type: PackType,
        num_repeat: int,
    ) -> Union[InferencePipeline, EvaluationPipeline]:
        info = Serializer.load_info(folder)
        config = Config.from_pack(info["config"])
        config.num_repeat = num_repeat
        info["config"] = config.to_pack().asdict()
        Serializer.save_info(folder, info=info)
        fn = (
            cls._load_inference
            if pack_type == PackType.INFERENCE
            else cls._load_evaluation
        )
        # avoid loading model because the ensembled model has different states
        p = fn(folder, excludes=[SerializeModelBlock])
        # but we need to build the SerializeModelBlock again for further save/load
        b_serialize_model = SerializeModelBlock()
        b_serialize_model.verbose = False
        p.build(b_serialize_model)
        return p

    @classmethod
    def _get_merged_states(
        cls,
        p: Pipeline,
        device: torch.device,
        ckpt_paths: List[TPath],
        states_callback: states_callback_type = None,
    ) -> OrderedDict:
        merged_states = OrderedDict()
        for i, ckpt_path in enumerate(track(ckpt_paths, description="merge states")):
            states = torch.load(ckpt_path, weights_only=False, map_location=device)
            states = states["states"]
            current_keys = list(states.keys())
            for k, v in list(states.items()):
                states[f"{i}.{k}"] = v
            for k in current_keys:
                states.pop(k)
            if states_callback is not None:
                states = states_callback(p, states)
            merged_states.update(states)
        return merged_states

    @classmethod
    def _fuse_multiple(
        cls,
        src_folders: List[str],
        pack_type: PackType,
        device: device_type = None,
        num_picked: Optional[Union[int, float]] = None,
        states_callback: states_callback_type = None,
        sort_ckpt_by: SortMethod = SortMethod.BEST,
        target_ckpt_step: Optional[int] = None,
    ) -> Union[InferencePipeline, EvaluationPipeline]:
        if pack_type == PackType.TRAINING:
            raise ValueError("should not pack to training pipeline when fusing")
        device = get_torch_device(device)
        # get num picked
        num_total = num_repeat = len(src_folders)
        if num_picked is not None:
            if isinstance(num_picked, float):
                if num_picked < 0.0 or num_picked > 1.0:
                    raise ValueError("`num_picked` should ∈ [0, 1] when set to float")
                num_picked = round(num_total * num_picked)
            if num_picked < 1:
                raise ValueError("calculated `num_picked` should be at least 1")
            scores = []
            for i, folder in enumerate(src_folders):
                ckpt_folder = os.path.join(folder, SerializeModelBlock.__identifier__)
                folder_scores = get_scores(ckpt_folder)
                scores.append(max(folder_scores.values()))
            scores_array = np.array(scores)
            picked_indices = np.argsort(scores)[::-1][:num_picked]
            src_folders = [src_folders[i] for i in picked_indices]
            original_score = scores_array.mean().item()
            picked_score = scores_array[picked_indices].mean().item()
            console.log(
                f"picked {num_picked} / {num_total}, "
                f"score: {original_score} -> {picked_score}"
            )
            num_repeat = num_picked
        # get empty pipeline
        with get_folder(src_folders[0], force_new=True) as src_folder:
            p = cls._build_ensemble_pipeline(src_folder, pack_type, num_repeat)
        # merge state dict
        ckpt_paths = []
        for folder in src_folders:
            with get_folder(folder) as i_folder:
                i_ckpt_dir = i_folder / SerializeModelBlock.__identifier__
                checkpoints = get_sorted_checkpoints(
                    i_ckpt_dir,
                    sort_by=sort_ckpt_by,
                    target_ckpt_step=target_ckpt_step,
                )
                ckpt_paths.append(i_ckpt_dir / checkpoints[0])
        merged_states = cls._get_merged_states(p, device, ckpt_paths, states_callback)
        # load state dict
        model = p.build_model.model
        if isinstance(model, EnsembleModel):
            model.rehook_ema()
        model.to(device)
        model.load_state_dict(merged_states)
        return p

    @classmethod
    def _self_ensemble(
        cls,
        workspace: TPath,
        pack_type: PackType,
        num_ensemble: int,
        ensemble_weights: bool = False,
        device: device_type = None,
        states_callback: states_callback_type = None,
        sort_ckpt_by: SortMethod = SortMethod.BEST,
        target_ckpt_step: Optional[int] = None,
        verbose: bool = True,
    ) -> Union[InferencePipeline, EvaluationPipeline]:
        if pack_type == PackType.TRAINING:  # pragma: no cover
            raise ValueError("should not pack to training pipeline when fusing")
        device = get_torch_device(device)
        workspace = to_path(workspace)
        # pick ckpts
        ckpt_folder = workspace / CHECKPOINTS_FOLDER
        sorted_ckpt_files = get_sorted_checkpoints(
            ckpt_folder,
            sort_by=sort_ckpt_by,
            target_ckpt_step=target_ckpt_step,
        )
        if num_ensemble > len(sorted_ckpt_files):
            raise RuntimeError(
                f"only {len(sorted_ckpt_files)} checkpoints are available, "
                f"but `num_ensemble` is set to {num_ensemble}"
            )
        picked = sorted_ckpt_files[:num_ensemble]
        ckpts = [ckpt_folder / f for f in picked]
        if verbose:
            console.log(f"follwing checkpoints are picked: {', '.join(picked)}")
        # get empty pipeline
        with get_folder(workspace / cls.pipeline_folder, force_new=True) as p_folder:
            if not ensemble_weights:
                p = cls._build_ensemble_pipeline(p_folder, pack_type, num_ensemble)
                merged_states = cls._get_merged_states(p, device, ckpts, states_callback)  # type: ignore
                if isinstance(p.build_model.model, EnsembleModel):
                    p.build_model.model.rehook_ema()
            else:
                fn = (
                    cls._load_inference
                    if pack_type == PackType.INFERENCE
                    else cls._load_evaluation
                )
                p = fn(p_folder)
                merged_states = OrderedDict()
                for ckpt in ckpts:
                    states = torch.load(ckpt, weights_only=False, map_location=device)
                    states = states["states"]
                    if states_callback is not None:
                        states = states_callback(p, states)
                    for k, v in states.items():
                        mv = merged_states.get(k)
                        if mv is None:
                            merged_states[k] = v
                        else:
                            merged_states[k] = mv + v
                for k, v in merged_states.items():
                    if is_float(v):
                        merged_states[k] /= num_ensemble
        # load state dict
        model = p.build_model.model
        model.to(device)
        model.load_state_dict(merged_states)
        return p


__all__ = [
    "PipelineTypes",
    "TrainingPipeline",
    "InferencePipeline",
    "EvaluationPipeline",
    "PackType",
    "PipelineSerializer",
]
