# Common Nodes

import asyncio

from PIL import Image
from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Optional
from pydantic import Field
from pydantic import BaseModel
from dataclasses import dataclass

from .schema import TImage
from .schema import ImageModel
from .schema import IImageNode
from .schema import EmptyOutput
from .schema import IWithImageNode
from ..core import LOOP_NODE
from ..core import GATHER_NODE
from ..core import extract_from
from ..core import Node
from ..core import Flow
from ..core import Schema
from ..core import Injection
from ...toolkit import console
from ...toolkit.misc import shallow_copy_dict


# functional nodes


class LoopBackInjectionModel(BaseModel):
    """Data model of `LoopBackInjection`"""

    src_hierarchy: Optional[str] = Field(
        ...,
        description="""The 'src_hierarchy' of the dependent node's results that the current node depends on.
- `src_hierarchy` can be very complex:
  - use `int` as `list` index, and `str` as `dict` key.
  - use `.` to represent nested structure.
  - for example, you can use `a.0.b` to indicate `results["a"][0]["b"]`.
- If `None`, all results of the dependent node will be used.""",
    )
    dst_hierarchy: str = Field(
        ...,
        description="""The 'dst_hierarchy' of the current node's `data`.
- `dst_hierarchy` can be very complex:
  - use `int` as `list` index, and `str` as `dict` key.
  - use `.` to represent nested structure.
  - for example, if you want to inject to `data["a"][0]["b"]`, you can use `a.0.b`
  as the `dst_hierarchy`.""",
    )


class LoopInput(BaseModel):
    base_node: str = Field(..., description="The node to be looped.")
    base_data: Dict[str, Any] = Field(default_factory=dict, description="Base data.")
    loop_values: Dict[str, List[Any]] = Field(
        ...,
        description="""The values to be looped.
> - The keys should be the 'target hierarchy' of the `data`
> - The values should be a list of values to be looped & injectedinto the 'target hierarchy'.
> - All values should have the same length.

For example, if you want to loop `data["a"]` with values `[1, 2]`, and loop `data["b"][0]["c"]` with values `[3, 4]`, you can use:
```python
{
    "a": [1, 2],
    "b.0.c": [3, 4],
}
```
""",
    )
    loop_back_injections: Optional[List[LoopBackInjectionModel]] = Field(
        None,
        description="The loop back injections.\n"
        "> - If this is set, the results from the previous step in the loop will be "
        "injected into the current node's `data`.\n"
        "> - If `None`, no injection will be performed, and all nodes will be "
        "executed in parallel.",
    )
    extract_hierarchy: Optional[str] = Field(
        None,
        description="The hierarchy of the results to be extracted.\n"
        "> - If `None`, all results will be preserved.",
    )
    verbose: bool = Field(False, description="Whether to print debug logs.")


class LoopOutput(BaseModel):
    results: List[Any] = Field(..., description="The results of the loop.")


@Node.register(LOOP_NODE)
class LoopNode(Node):
    @classmethod
    def get_schema(cls) -> Schema:
        return Schema(
            LoopInput,
            LoopOutput,
            description="A node that represents a loop of another node.",
        )

    async def execute(self) -> Dict[str, List[Dict[str, Any]]]:
        t_node = Node.get(self.data["base_node"])
        if t_node is None:
            raise ValueError(f"node `{self.data['base_node']}` is not defined")
        base_data = self.data["base_data"]
        loop_values = self.data["loop_values"]
        loop_back_injections = self.data["loop_back_injections"]
        loop_keys = list(loop_values)
        lengths = [len(loop_values[k]) for k in loop_keys]
        if len(set(lengths)) != 1:
            raise ValueError(
                "all loop values should have the same length, "
                f"but lengths are {lengths}"
            )
        n = lengths[0]
        flow = Flow()
        for i in range(n):
            i_data = shallow_copy_dict(base_data)
            for k in loop_keys:
                i_data[k] = loop_values[k][i]
            if loop_back_injections is None or i == 0:
                i_injections = []
            else:
                i_injections = list(map(shallow_copy_dict, loop_back_injections))
                i_injections = [Injection(str(i - 1), **d) for d in i_injections]
            flow.push(t_node(str(i), i_data, i_injections))
        target = flow.gather(*map(str, range(n)))
        results = await flow.execute(target, verbose=self.data["verbose"])
        extracted = [results[str(i)] for i in range(n)]
        extract_hierarchy = self.data["extract_hierarchy"]
        if extract_hierarchy is not None:
            extracted = [extract_from(rs, extract_hierarchy) for rs in extracted]
        return {"results": extracted}


@Node.register(GATHER_NODE)
class GatherNode(Node):
    flow: Optional[Flow] = None

    @classmethod
    def get_schema(cls) -> Schema:
        return Schema(
            description="A node that is used to gather other nodes' results.\n"
            "> - This is useful when you have multiple targets to collect results from.\n"
            "> - If you are programming in Python, you can use `flow.gather` to make things easier.",
        )

    @classmethod
    async def get_api_response(cls, results: Dict[str, Any]) -> Dict[str, Any]:
        if cls.flow is None:
            console.warn(
                "`flow` is not provided for `GatherNode`, raw results will be returned "
                "and `get_api_response` might not work as expected"
            )
            return results
        keys = list(results)
        nodes = [cls.flow.get(k) for k in keys]
        if any(n is None for n in nodes):
            raise ValueError(
                "internal error: some nodes are not found when getting api response: "
                f"{[k for k, n in zip(keys, nodes) if n is None]}"
            )
        node_names = [node.data.__identifier__ for node in nodes]  # type: ignore
        t_nodes = list(map(Node.get, node_names))
        tasks = [t.get_api_response(results[k]) for k, t in zip(keys, t_nodes)]
        converted = await asyncio.gather(*tasks)
        return {k: v for k, v in zip(keys, converted)}

    async def initialize(self, flow: Flow) -> None:
        await super().initialize(flow)
        self.flow = flow

    async def execute(self) -> Dict[str, Any]:
        return self.data

    def from_info(self, info: Dict[str, Any]) -> None:
        super().from_info(info)
        for injection in self.injections:
            if injection.src_hierarchy is not None:
                raise ValueError(
                    "`GatherNode` should always use `src_hierarchy=None` "
                    f"for injections, but `{injection}` is found"
                )
            if injection.src_key != injection.dst_hierarchy:
                raise ValueError(
                    "`GatherNode` should always use `src_key=dst_hierarchy` "
                    f"for injections, but `{injection}` is found"
                )


# common nodes


class ParametersModel(BaseModel):
    params: Dict[str, Any] = Field(default_factory=dict, description="The parameters.")


@Node.register("common.parameters")
class ParametersNode(Node):
    @classmethod
    def get_schema(cls) -> Schema:
        return Schema(
            ParametersModel,
            ParametersModel,
            description="Setup parameters.\n"
            "> - This is often used in a pre-defined workflow JSON to decide "
            "which parameters to be exposed to the user.\n"
            "> - See [examples](https://github.com/carefree0910/carefree-workflow/tree/main/examples/workflows) for reference.",
        )

    async def execute(self) -> Dict[str, Any]:
        return self.data


class EchoModel(BaseModel):
    messages: Union[str, List[str]]


@Node.register("common.echo")
class EchoNode(Node):
    @classmethod
    def get_schema(cls) -> Schema:
        return Schema(EchoModel, EchoModel, description="Echo the given message(s).")

    async def execute(self) -> Dict[str, Union[str, List[str]]]:
        messages = self.data["messages"]
        if isinstance(messages, str):
            messages = [messages]
        for message in messages:
            console.log(message)
        return self.data


@dataclass
@Node.register("common.download_image")
class DownloadImageNode(IImageNode):
    offload: bool = True

    @classmethod
    def get_schema(cls) -> Schema:
        schema = super().get_schema()
        schema.description = "Download an image from the given url."
        return schema

    async def execute(self) -> Dict[str, Image.Image]:
        image = await self.get_image_from("url")
        return {"image": image}


class SaveImageInput(ImageModel):
    path: str = Field("debug.png", description="The path to save the image.")


@Node.register("debug.save_image")
class SaveImageNode(IWithImageNode):
    @classmethod
    def get_schema(cls) -> Schema:
        return Schema(
            SaveImageInput,
            output_model=EmptyOutput,
            description="Save an image from the given url to disk, mainly for debugging.",
        )

    async def execute(self) -> dict:
        image = await self.get_image_from("url")
        image.save(self.data["path"])
        return {}


class SaveImagesInput(BaseModel):
    urls: List[TImage] = Field(..., description="The urls of the images.")
    prefix: str = Field("debug", description="The prefix to save the images.")


@Node.register("debug.save_images")
class SaveImagesNode(IWithImageNode):
    @classmethod
    def get_schema(cls) -> Schema:
        return Schema(
            SaveImagesInput,
            output_model=EmptyOutput,
            description="Save images from the given urls to disk, mainly for debugging.",
        )

    async def execute(self) -> dict:
        tasks = [self.fetch_image(str(v), v) for v in self.data["urls"]]
        images = await asyncio.gather(*tasks)
        prefix = self.data["prefix"]
        for i, image in enumerate(images):
            image.save(f"{prefix}_{i}.png")
        return {}


# common node utils


def to_endpoint(name: str) -> str:
    split = name.split(".")
    return f"/{'/'.join(split)}"


__all__ = [
    "LoopBackInjectionModel",
    "LoopNode",
    "GatherNode",
    "EchoNode",
    "DownloadImageNode",
    "SaveImageNode",
    "SaveImagesNode",
    "to_endpoint",
]
