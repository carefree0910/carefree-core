# Contributing

## Developing

### `toolkit` package

If you want to add some functions / classes that are likely to be used by any other packages, you should put them in the `toolkit` package.

There are already some modules in the `toolkit` package and they have covered most of the use cases, so it is likely that you only need to add your functions / classes to one of them:

- `array.py`: implementations related to `numpy.ndarray` or `torch.Tensor`.
- `console.py`: implementations related to console printing.
- `constants.py`: constants.
- `cv.py`: implementations related to computer vision.
- `data_structures.py`: implementations related to data structures.
- `geometry.py`: implementations related to geometry stuffs (matrices, vectors, etc.).
- `types.py`: implementations related to type aliases.
- `web.py`: implementations related to web stuffs (get, post, downloads, etc.).
- `misc.py`: miscellaneous implementations. If you don't know where to put your functions / classes, you can put them here.

### `learn` package

This package may be where most customizations happen, since the implemetations of deep learning stuffs are put here. Here are some brief development guides for this package:

> Base classes listed below are all defined at `core/learn/schema.py`.

#### Training Process

A training process is constructed by following components:

- **`IData`** - This is where data preprocessing happens.
  - In most cases `ArrayData` / `ArrayDictData` at `core/learn/data/array.py` is enough, and what you need to do is to define custom `IDataBlock`s to convert the raw data into `np.ndarray` / `Dict[str, np.ndarray]` format.
  - `tests/test_learn/ddp_titanic_task.py` shows how we define a custom `IDataBlock` to handle csv files.
- **`IModel`** - A model that wraps some `nn.Module`s and logics.
  - The most common case is it contains a single `nn.Module` (`m`) and a single loss function (`loss`), see `CommonModel` at `core/learn/models/common.py` for more details.
- **`IMetric`** - If you want to evaluate your model with your own metrics, you can inherit this class.
  - Examples are at `core/learn/metrics.py`.

And a brief lifecycle of a training process is:

1. Initialize an `IData` instance (based on `DataConfig`) and `fit` some data into it.
2. Initialize a `TrainingPipeline` (based on `Config`) and `fit` the above `IData` instance into it.

Inside the `TrainingPipeline`, following precedures will be executed:

1. Initialize `IModel`, data loaders, optimizers, schedulers, etc.
2. Enter the training loop, monitor the `IModel` with `IMetric`s, and save checkpoints / early stop / extend training, ..., if necessary.
3. After training, serialize the entire `TrainingPipeline` into a folder and save it to disk.

#### Customizations

If you want to:

- **define your own building blocks**, you may refer to:
  - `core/learn/modules/linear.py` / `core/learn/modules/fcnn.py`, for customizing **modules**.
  - `core/learn/losses/basic.py`, for customizing **losses**.
  - `core/learn/optimizers.py`, for customizing **optimizers**.
  - `core/learn/schedulers.py`, for customizing **schedulers**.
- **monitor the training process** (e.g., early stop, extend training, etc.), you may inherit the **`TrainerMonitor`** class.
  - I've provided docstrings for it and also implement some examples at `core/learn/monitors.py`.
- **hook into the training loop**, you may inherit the **`TrainerCallback`** class.
  - I've provided docstrings for it and also implement an example at `core/learn/callbacks/common.py`.

### `flow` package

This package is relatively stable, so it is not recommended to add new features to it. Instead, it is often used to write business logics by inheriting the `Node` class and build your own workflows with `Flow`.

## Style Guide

`carefree-core` adopted [`black`](https://github.com/psf/black) and [`mypy`](https://github.com/python/mypy) to stylize its codes, so you may need to check the format, coding style and type hint with them before your codes could actually be merged.

Besides, there are a few more principles that I'm using for sorting imports:
- One import at a time (no `from typing import Any, Dict, List, ...`).
- From functions to classes.
- From short to long (for both naming and path).
- From *a* to *z* (alphabetically).
- Divided into four sections:
  1. `import ...`
  2. `import ... as ...`
  3. `from ... import ...`
  4. relative imports
- From general to specific (a `*` will always appear at the top of each section)

Here's an example to illustrate these principles:

```python
import os
import re
import json
import math
import torch

import torch.distributed as dist

from typing import Any
from typing import Set
from typing import Dict
from typing import Optional
from accelerate import Accelerator
from tqdm.autonotebook import tqdm
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from .schema import device_type
from .schema import weighted_loss_score
from .schema import IData
from .schema import IModel
from .toolkit import summary
from .toolkit import get_ddp_info
from .constants import PT_PREFIX
from .constants import SCORES_FILE
from .schedulers import WarmupScheduler
from ..toolkit import console
from ..toolkit.misc import safe_execute
from ..toolkit.misc import shallow_copy_dict
from ..toolkit.misc import Incrementer
from ..toolkit.types import tensor_dict_type
```

But after all, this is not a strict constraint so everything will be fine as long as it 'looks good' 🤣

## Creating a Pull Request

When you are ready to create a pull request, please try to keep the following in mind.

### Title

The title of your pull request should

- briefly describe and reflect the changes
- wrap any code with backticks

### Description

The description of your pull request should

- describe the motivation
- describe the changes
- if still work-in-progress, describe remaining tasks
