import torch

import numpy as np

from typing import Any
from typing import Dict
from typing import Tuple
from typing import Union
from typing import Optional

arr_type = Union[np.ndarray, torch.Tensor]
np_dict_type = Dict[str, Union[np.ndarray, Any]]
tensor_dict_type = Dict[str, Union[torch.Tensor, Any]]

TConfig = Optional[Union[str, Dict[str, Any]]]
TNumberPair = Optional[Union[int, Tuple[int, int]]]
