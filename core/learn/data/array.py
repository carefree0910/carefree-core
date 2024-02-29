from typing import Dict
from typing import List
from typing import Optional

from ..schema import IData
from ..schema import IDataset
from ..schema import DataBundle
from ..schema import TDs
from ..toolkit import np_batch_to_tensor
from ..constants import INPUT_KEY
from ..constants import LABEL_KEY
from ...toolkit.types import arr_type
from ...toolkit.types import tensor_dict_type


class ArrayDataset(IDataset):
    def __init__(self, x: arr_type, y: Optional[arr_type] = None, *others: arr_type):
        if x is None:
            raise ValueError("`x` cannot be `None`")
        self.x = x
        self.arrays = (x, y) + others
        lengths = [len(a) for a in self.arrays if a is not None]
        if len(set(lengths)) != 1:
            raise ValueError(f"all arrays must have the same length, got {lengths}")

    def __len__(self) -> int:
        return len(self.x)

    def __getitems__(self, indices: List[int]) -> tensor_dict_type:
        keys = (
            [INPUT_KEY]
            + [LABEL_KEY]
            + [f"others_{i}" for i in range(len(self.arrays) - 2)]
        )
        fetched = [None if a is None else a[indices] for a in self.arrays]
        batch = {k: v for k, v in zip(keys, fetched)}
        batch = np_batch_to_tensor(batch)
        return batch


class ArrayDictDataset(IDataset):
    def __init__(self, arrays: Dict[str, arr_type]):
        self.arrays = arrays
        lengths = [len(a) for a in self.arrays.values()]
        if len(set(lengths)) != 1:
            raise ValueError(f"all arrays must have the same length, got {lengths}")

    def __len__(self) -> int:
        return len(list(self.arrays.values())[0])

    def __getitems__(self, indices: List[int]) -> tensor_dict_type:
        batch = {k: v[indices] for k, v in self.arrays.items()}
        batch = np_batch_to_tensor(batch)
        return batch


@IData.register("array")
class ArrayData(IData["ArrayData", ArrayDataset]):
    def to_datasets(self, bundle: DataBundle, *, for_inference: Optional[bool]) -> TDs:
        train_dataset = ArrayDataset(bundle.x_train, bundle.y_train)
        if bundle.x_valid is None:
            valid_dataset = None
        else:
            valid_dataset = ArrayDataset(bundle.x_valid, bundle.y_valid)
        return train_dataset, valid_dataset


@IData.register("array_dict")
class ArrayDictData(IData["ArrayDictData", ArrayDictDataset]):
    def to_datasets(self, bundle: DataBundle, *, for_inference: Optional[bool]) -> TDs:
        if not isinstance(bundle.x_train, dict):
            msg = f"`bundle.x_train` must be a `dict`, got {type(bundle.x_train)}"
            raise ValueError(msg)
        train_dataset = ArrayDictDataset(bundle.x_train)
        if bundle.x_valid is None:
            valid_dataset = None
        else:
            if not isinstance(bundle.x_valid, dict):
                msg = f"`bundle.x_valid` must be a `dict`, got {type(bundle.x_valid)}"
                raise ValueError(msg)
            valid_dataset = ArrayDictDataset(bundle.x_valid)
        return train_dataset, valid_dataset


__all__ = [
    "ArrayDataset",
    "ArrayDictDataset",
    "ArrayData",
    "ArrayDictData",
]
