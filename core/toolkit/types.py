from typing import Any
from typing import Dict
from typing import Tuple
from typing import Union
from typing import Literal
from typing import TypeVar
from typing import Optional
from typing import TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    import torch

    import numpy as np
    import numpy.typing as npt

arr_type = Union["np.ndarray", "torch.Tensor"]
TArray = TypeVar("TArray", bound=arr_type)
PredictionMode = Literal["auto", "binary", "multiclass", "multilabel"]

ArrayDict = Dict[
    str,
    Union["npt.NDArray[np.generic]", "torch.Tensor"],
]
NumpyDict = Dict[str, "npt.NDArray[np.generic]"]
TensorDict = Dict[str, "torch.Tensor"]
DataValue = Union[
    str,
    "npt.NDArray[np.generic]",
    "torch.Tensor",
    NumpyDict,
    TensorDict,
]
OptionalDataValue = Optional[DataValue]

# Legacy aliases keep accepting heterogeneous metadata for backward compatibility.
np_dict_type = Dict[str, Union["np.ndarray", Any]]
tensor_dict_type = Dict[str, Union["torch.Tensor", Any]]

TPath = Union[str, Path]
TConfig = Optional[Union[TPath, Dict[str, Any]]]
TNumberPair = Optional[Union[int, Tuple[int, int]]]
