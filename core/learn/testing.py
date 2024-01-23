import numpy as np

from typing import Tuple
from typing import Optional

from .data import ArrayData


def linear_data(
    n: int = 10000,
    dim: int = 10,
    *,
    out_dim: int = 1,
    batch_size: int = 100,
    use_validation: bool = True,
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
    if not use_validation:
        data = ArrayData.init().fit(x, y)
    else:
        data = ArrayData.init().fit(x, y, x, y)
    data.config.batch_size = batch_size
    return data, dim, out_dim, w
