from pathlib import Path
from pydantic import Field
from pydantic import BaseModel

from .toolkit.misc import OPTBase


class FlowOpt(BaseModel):
    focus: str = ""
    verbose: bool = True


class LearnOpt(BaseModel):
    cache_dir: Path = Path.home() / ".cache" / "carefree-core" / "learn"
    data_cache_dir: Path = Path.home() / ".cache" / "carefree-core" / "learn" / "data"
    external_dir: Path = Path.home() / ".cache" / "carefree-core" / "learn" / "external"


class OPTClass(OPTBase):
    flow_opt: FlowOpt = Field(default_factory=FlowOpt)
    learn_opt: LearnOpt = Field(default_factory=LearnOpt)

    @property
    def env_key(self) -> str:
        return "CFCORE_ENV"


OPT = OPTClass()


__all__ = [
    "OPT",
]
