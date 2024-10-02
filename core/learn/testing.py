import numpy as np

from typing import Tuple
from typing import Optional

from .data import ArrayData


def arange_data(
    n: int = 10,
    dim: int = 3,
    *,
    out_dim: int = 1,
    batch_size: int = 4,
) -> Tuple[ArrayData, int, int]:
    x = np.arange(n * dim).reshape([n, dim]).astype(np.float32)
    y = np.arange(n * out_dim).reshape([n, out_dim]).astype(np.float32)
    data = ArrayData.init().fit(x, y)
    data.config.batch_size = batch_size
    return data, dim, out_dim


def linear_data(
    n: int = 10000,
    dim: int = 10,
    *,
    out_dim: int = 1,
    batch_size: int = 100,
    use_validation: bool = False,
    x_noise_scale: Optional[float] = None,
    y_noise_scale: Optional[float] = None,
) -> Tuple[ArrayData, int, int, np.ndarray]:
    x = np.random.random([n, dim])
    w = np.random.random([dim, out_dim])
    y = x @ w
    if x_noise_scale is not None:
        x += np.random.random(x.shape) * x_noise_scale
    if y_noise_scale is not None:
        y += np.random.random(y.shape) * y_noise_scale
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    if not use_validation:
        data = ArrayData.init().fit(x, y)
    else:
        data = ArrayData.init().fit(x, y, x, y)
    data.config.batch_size = batch_size
    return data, dim, out_dim, w
