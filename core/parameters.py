from typing import Any
from typing import Dict
from pathlib import Path
from dataclasses import dataclass

from .toolkit.misc import update_dict
from .toolkit.misc import OPTBase


@dataclass
class FlowOPT:
    focus: str
    verbose: bool


@dataclass
class LearnOPT:
    cache_dir: Path
    data_cache_dir: Path
    external_dir: Path
    meta_settings: Dict[str, Any]


class OPTClass(OPTBase):
    flow_opt: FlowOPT
    learn_opt: LearnOPT

    @property
    def env_key(self) -> str:
        return "CFCORE_ENV"

    @property
    def defaults(self) -> Dict[str, Any]:
        user_dir = Path.home()
        return dict(
            flow_opt=FlowOPT(focus="", verbose=True),
            learn_opt=LearnOPT(
                cache_dir=user_dir / ".cache" / "carefree-core" / "learn",
                data_cache_dir=user_dir / ".cache" / "carefree-core" / "learn" / "data",
                external_dir=user_dir
                / ".cache"
                / "carefree-core"
                / "learn"
                / "external",
                meta_settings={},
            ),
        )

    def update_from_env(self) -> None:
        super().update_from_env()
        defaults = self.defaults
        flow_opt = self._opt["flow_opt"]
        learn_opt = self._opt["learn_opt"]
        if isinstance(flow_opt, dict):
            updated = update_dict(flow_opt, defaults["flow_opt"].asdict())
            self._opt["flow_opt"] = FlowOPT(**updated)
        if isinstance(learn_opt, dict):
            updated = update_dict(learn_opt, defaults["learn_opt"].asdict())
            if "cache_dir" in updated:
                updated["cache_dir"] = Path(updated["cache_dir"])
            if "external_dir" in updated:
                updated["external_dir"] = Path(updated["external_dir"])
            self._opt["learn_opt"] = LearnOPT(**updated)


OPT = OPTClass()


__all__ = [
    "OPT",
]
