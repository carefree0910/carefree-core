import torch

import numpy as np

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional


configs_type = Optional[Union[List[Dict[str, Any]], Dict[str, Any]]]
general_config_type = Optional[Union[str, Dict[str, Any]]]

arr_type = Union[np.ndarray, torch.Tensor]
np_dict_type = Dict[str, Union[np.ndarray, Any]]
tensor_dict_type = Dict[str, Union[torch.Tensor, Any]]

TNumberPair = Optional[Union[int, Tuple[int, int]]]
