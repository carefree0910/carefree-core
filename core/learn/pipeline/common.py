from typing import Any
from typing import Dict
from typing import Type
from typing import TypeVar
from typing import Generic
from typing import Optional
from typing import ContextManager
from pathlib import Path
from collections import OrderedDict

from ..schema import IData
from ..schema import Config
from ...toolkit import console
from ...toolkit.misc import is_ddp
from ...toolkit.misc import get_ddp_info
from ...toolkit.misc import safe_execute
from ...toolkit.misc import shallow_copy_dict
from ...toolkit.types import TPath
from ...toolkit.pipeline import IBlock
from ...toolkit.pipeline import IPipeline
from ...toolkit.pipeline import TPipeline


T_p = TypeVar("T_p", bound="Pipeline", covariant=True)


class Block(IBlock):
    """
    Building block of a `Pipeline`.

    * data: only available in `TrainingPipeline`.
    * training_workspace: only available in `TrainingPipeline`, identical to `config.workspace`.
    * serialize_folder: only available in `load` process.
    * previous: previous building blocks in `build` process. Will be ALL building blocks in `run` process.
    * pipeline: the `Pipeline` that this block belongs to.

    """

    data: Optional[IData]
    training_workspace: Optional[str]
    serialize_folder: Optional[Path]
    previous: Dict[str, "Block"]
    pipeline: "Pipeline"

    # optional callbacks

    def process_defaults(self, _defaults: OrderedDict) -> None:
        pass

    def run(self, data: IData, _defaults: OrderedDict, **kwargs: Any) -> None:
        pass

    def save_extra(self, folder: TPath) -> None:
        pass

    def load_from(self, folder: TPath) -> None:
        pass

    # api

    @property
    def ddp(self) -> bool:
        return is_ddp()

    @property
    def local_rank(self) -> Optional[int]:
        ddp_info = get_ddp_info()
        return None if ddp_info is None else ddp_info.local_rank

    @property
    def is_local_rank_0(self) -> bool:
        return not self.ddp or self.local_rank == 0


class verbose_context:
    is_in: bool = False
    previous_verboses: Dict[str, bool]

    def __init__(self, p: "Pipeline", verbose: bool) -> None:
        self.p = p
        self.verbose = verbose
        self.is_null = False

    def __enter__(self) -> None:
        if verbose_context.is_in:
            self.is_null = True
            return
        verbose_context.is_in = True
        self.previous_verboses = {}
        for block in self.p.blocks:
            b_verbose = getattr(block, "verbose", None)
            if b_verbose is not None:
                self.previous_verboses[block.__identifier__] = b_verbose
                block.verbose = self.verbose  # type: ignore

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.is_null:
            return
        verbose_context.is_in = False
        for k, v in self.previous_verboses.items():
            self.p.get_block(k).verbose = v  # type: ignore


class Pipeline(Generic[TPipeline], IPipeline[Block, Config, TPipeline]):
    _defaults: OrderedDict

    data: Optional[IData] = None
    training_workspace: Optional[str] = None
    serialize_folder: Optional[Path] = None
    config_file = "config.json"

    # inheritance

    @classmethod
    def init(cls, config: Config) -> TPipeline:
        config.sanity_check()
        self: Pipeline = cls()
        self.config = config.copy()
        self._defaults = OrderedDict()
        return self  # type: ignore

    @property
    def config_base(self) -> Type[Config]:
        return Config

    @property
    def block_base(self) -> Type[Block]:
        return Block

    def to_info(self) -> Dict[str, Any]:
        info = super().to_info()
        info["_defaults"] = [[k, v] for k, v in self._defaults.items()]
        return info

    def from_info(self: T_p, info: Dict[str, Any]) -> T_p:
        self._defaults = OrderedDict()
        for k, v in info["_defaults"]:
            self._defaults[k] = v
        return super().from_info(info)

    def before_block_build(self, block: Block) -> None:
        block.data = self.data
        block.pipeline = self
        block.training_workspace = self.training_workspace
        if self.serialize_folder is None:
            block.serialize_folder = None
        else:
            block.serialize_folder = self.serialize_folder

    def after_block_build(self, block: Block) -> None:
        block.process_defaults(self._defaults)
        if self.training_workspace is not None:
            if self.training_workspace != self.config.workspace:
                self.training_workspace = self.config.workspace

    # api

    def run(self, data: IData, **kwargs: Any) -> None:
        if not self.blocks:
            console.warn("no blocks are built, nothing will happen")
            return
        kw = shallow_copy_dict(kwargs)
        kw["data"] = data
        kw["_defaults"] = self._defaults
        for block in self.blocks:
            safe_execute(block.run, kw)

    def verbose_context(self, verbose: bool) -> ContextManager:
        return verbose_context(self, verbose)


__all__ = [
    "Block",
    "Pipeline",
]
