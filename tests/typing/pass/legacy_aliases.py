from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    import numpy as np

    from core.learn import td_type
    from core.learn import data_type
    from core.learn.schema import data_type as schema_data_type
    from core.toolkit.types import arr_type
    from core.toolkit.types import ArrayDict
    from core.toolkit.types import DataValue
    from core.toolkit.types import NumpyDict
    from core.toolkit.types import TensorDict
    from core.toolkit.types import np_dict_type
    from core.toolkit.types import tensor_dict_type
    from core.toolkit.types import OptionalDataValue

    legacy_array: arr_type
    legacy_numpy: np_dict_type
    legacy_tensor: tensor_dict_type
    legacy_td: td_type
    legacy_data: data_type
    legacy_schema_data: schema_data_type

    strict_arrays: ArrayDict
    strict_numpy: NumpyDict
    strict_tensor: TensorDict
    strict_data: DataValue
    optional_strict_data: OptionalDataValue

    strict_arrays = {"array": np.ones(1)}
    strict_numpy = {"array": np.ones(1)}
    strict_tensor = {"tensor": torch.ones(1)}
