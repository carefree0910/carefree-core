from typing import Any
from typing import Dict
from dataclasses import dataclass

from .toolkit.misc import OPTBase


@dataclass
class FlowOPT:
    focus = ""
    verbose = True


class OPTClass(OPTBase):
    flow_opt: FlowOPT

    @property
    def env_key(self) -> str:
        return "CFCORE_ENV"

    @property
    def defaults(self) -> Dict[str, Any]:
        return dict(
            flow_opt=FlowOPT(),
        )

    def update_from_env(self) -> None:
        super().update_from_env()
        if isinstance(self._opt["flow_opt"], dict):
            self._opt["flow_opt"] = FlowOPT(**self._opt["flow_opt"])


OPT = OPTClass()


__all__ = [
    "OPT",
]
