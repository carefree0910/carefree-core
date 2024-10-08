import io
import os
import copy
import torch
import random
import argparse

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from torch import device
from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import TypeVar
from typing import Callable
from typing import Optional
from typing import NamedTuple
from typing import ContextManager
from typing import no_type_check
from pathlib import Path
from contextlib import nullcontext
from collections import defaultdict
from collections import OrderedDict
from onnxruntime import InferenceSession
from torch.optim import Optimizer
from matplotlib.figure import Figure
from safetensors.torch import load_file

from .constants import INPUT_KEY
from .constants import WORKSPACE_ENVIRON_KEY
from ..toolkit import console
from ..toolkit.misc import prod
from ..toolkit.misc import check_requires
from ..toolkit.misc import shallow_copy_dict
from ..toolkit.misc import truncate_string_to_length
from ..toolkit.array import to_standard
from ..toolkit.array import is_real_numeric
from ..toolkit.types import TPath
from ..toolkit.types import TArray
from ..toolkit.types import np_dict_type
from ..toolkit.types import tensor_dict_type


param_type = Union[Tensor, nn.Parameter]
device_type = Optional[Union[int, str, device]]


# general


min_seed_value = np.iinfo(np.uint32).min
max_seed_value = np.iinfo(np.uint32).max


def new_seed() -> int:
    """
    Generates a new random seed.

    Returns
    -------
    int
        A new random seed.

    Examples
    --------
    >>> seed = new_seed()
    >>> print(seed)
    42

    """

    return random.randint(min_seed_value, max_seed_value)


def seed_everything(seed: int) -> int:
    """
    Seeds all random number generators.

    Parameters
    ----------
    seed : int
        The seed value to use.

    Returns
    -------
    int
        The seed value used.

    Notes
    -----
    This function seeds the random number generators for NumPy, Python's built-in random module,
    and PyTorch. It ensures reproducibility of random number generation across different runs.

    If the provided seed is not within the valid range, a new random seed will be generated within
    the valid range and used instead. A warning will be printed to notify the user.

    Examples
    --------
    >>> seed = 42
    >>> seed_everything(seed)
    42

    """

    if not min_seed_value <= seed <= max_seed_value:
        seed, old_seed = new_seed(), seed
        console.warn(
            f"{old_seed} is not in bounds, numpy accepts from {min_seed_value} to "
            f"{max_seed_value}, will use {seed} instead."
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed


def get_environ_workspace() -> Optional[str]:
    """
    Get the workspace from the environment variable.
    This is used internally to set the default workspace.

    Returns
    -------
    Optional[str]
        The workspace path if available, None otherwise.

    """

    return os.environ.get(WORKSPACE_ENVIRON_KEY)


def set_environ_workspace(workspace: str) -> None:
    """
    Set the (default) workspace in the environment variable.

    Parameters
    ----------
    workspace : str
        The workspace path.

    """

    os.environ[WORKSPACE_ENVIRON_KEY] = workspace


def unset_environ_workspace() -> None:
    """unset the (default) workspace in the environment variable"""

    if WORKSPACE_ENVIRON_KEY in os.environ:
        del os.environ[WORKSPACE_ENVIRON_KEY]


def check_is_ci() -> bool:
    """
    Check if the code is running in a continuous integration (CI) environment.

    Returns
    -------
    bool
        True if running in a CI environment, False otherwise.

    Examples
    --------
    >>> is_ci = check_is_ci()
    >>> print(is_ci)
    False

    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--ci", type=int, default=0)
    args = parser.parse_args()
    return bool(args.ci)


def show_or_save(
    export_path: Optional[str],
    fig: Optional[Figure] = None,
    **kwargs: Any,
) -> None:
    """
    Utility function to deal with figure.

    Parameters
    ----------
    export_path : {None, str}
    * If None, the figure will be shown.
    * If str, it represents the path where the figure should be saved to.
    fig : {None, Figure}
    * If None, default figure contained in plt will be executed.
    * If plt.figure, it will be executed

    """

    if export_path is None:
        fig.show(**kwargs) if fig is not None else plt.show(**kwargs)
    else:
        if fig is not None:
            fig.savefig(export_path)
        else:
            plt.savefig(export_path, **kwargs)
    plt.close()


def show_or_return(return_canvas: bool) -> Union[None, np.ndarray]:
    """
    Utility function to deal with current plt.

    Parameters
    ----------
    return_canvas : bool, whether return canvas or not.

    """

    if not return_canvas:
        plt.show()
        return None

    buffer_ = io.BytesIO()
    plt.savefig(buffer_, format="png")
    plt.close()
    buffer_.seek(0)
    image = Image.open(buffer_)
    canvas = np.asarray(image)[..., :3]
    buffer_.close()
    return canvas


class WeightsStrategy:
    """
    A strategy for generating sample weights.

    Parameters
    ----------
    strategy : Optional[str]
        The name of the strategy to use. Should be one of "linear_decay", "radius_decay",
        "log_decay", "sigmoid_decay".
        If None, no weights are generated.

    Examples
    --------
    >>> ws = WeightsStrategy("linear_decay")
    >>> ws(10)
    array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ])

    """

    def __init__(self, strategy: Optional[str]):
        self.strategy = strategy

    def __call__(self, num: int) -> Optional[np.ndarray]:
        """
        Generate sample weights.

        Parameters
        ----------
        num : int
            The number of data samples.

        Returns
        -------
        np.ndarray
            The generated sample weights.
        """

        if self.strategy is None:
            return None
        return getattr(self, self.strategy)(num)

    def linear_decay(self, num: int) -> np.ndarray:
        """
        Generate sample weights using a linear decay strategy.

        Parameters
        ----------
        num : int
            The number of data samples.

        Returns
        -------
        np.ndarray
            The generated sample weights.
        """

        return np.linspace(0, 1, num + 1)[1:]

    def radius_decay(self, num: int) -> np.ndarray:
        """
        Generate sample weights using a radius decay strategy.

        Parameters
        ----------
        num : int
            The number of data samples.

        Returns
        -------
        np.ndarray
            The generated sample weights.
        """

        return np.sin(np.arccos(1.0 - np.linspace(0, 1, num + 1)[1:]))

    def log_decay(self, num: int) -> np.ndarray:
        """
        Generate sample weights using a log decay strategy.

        Parameters
        ----------
        num : int
            The number of data samples.

        Returns
        -------
        np.ndarray
            The generated sample weights.
        """

        return np.log(np.arange(num) + np.e)

    def sigmoid_decay(self, num: int) -> np.ndarray:
        """
        Generate sample weights using a sigmoid decay strategy.

        Parameters
        ----------
        num : int
            The number of data samples.

        Returns
        -------
        np.ndarray
            The generated sample weights.
        """

        return 1.0 / (1.0 + np.exp(-np.linspace(-5.0, 5.0, num)))

    def visualize(self, export_path: str = "weights_strategy.png") -> None:
        """
        Visualize the weights strategy.

        Parameters
        ----------
        export_path : str
            The path to save the visualization. If None, the visualization is shown instead of being saved.

        Examples
        --------
        >>> ws = WeightsStrategy("linear_decay")
        >>> ws.visualize("weights_strategy.png")

        """

        n = 1000
        x = np.linspace(0, 1, n)
        y = self(n)
        if y is None:
            raise RuntimeError("no strategy is set")
        plt.figure()
        plt.plot(x, y)
        show_or_save(export_path)


# dl


warnings = set()
GenericM = TypeVar("GenericM", bound=nn.Module)


def warn_once(message: str, *, key: Optional[str] = None) -> None:
    """
    Print a warning message once.

    Parameters
    ----------
    message : str
        The warning message to print.
    key : Optional[str]
        The key associated with the warning. If None, the message is used as the key.

    Examples
    --------
    >>> warn_once("This is a warning message")
    > [warning] This is a warning message
    >>> warn_once("This is a warning message")
    # nothing is printed

    """

    key = key or message
    if key not in warnings:
        console.warn(message)
        warnings.add(key)


def get_tensors(inp: Union[TPath, tensor_dict_type]) -> tensor_dict_type:
    """
    Get tensors from input.

    Parameters
    ----------
    inp : d_inp_type
        The input from which to get the tensors. This could be a path to a file or a dictionary.

    Returns
    -------
    tensor_dict_type
        A dictionary of tensors.

    Examples
    --------
    >>> get_tensors("example.safetensors")
    {'tensor1': tensor([...]), 'tensor2': tensor([...])}

    """

    if isinstance(inp, Path):
        inp = str(inp)
    if isinstance(inp, str):
        if inp.endswith(".safetensors"):
            inp = load_file(inp)
        else:
            inp = torch.load(inp, weights_only=True, map_location="cpu")
    if not isinstance(inp, dict):
        raise ValueError(f"unrecognized input type ({type(inp)})")
    if "state_dict" in inp:
        inp = inp["state_dict"]
    return shallow_copy_dict(inp)  # type: ignore


def get_dtype(m: nn.Module) -> torch.dtype:
    """
    Get the data type of the parameters of a module.

    Parameters
    ----------
    m : nn.Module
        The module.

    Returns
    -------
    torch.dtype
        The data type of the parameters of the module.

    Examples
    --------
    >>> m = nn.Linear(10, 2)
    >>> get_dtype(m)
    torch.float32

    """

    params = list(m.parameters())
    return torch.float32 if not params else params[0].dtype


def get_device(m: nn.Module) -> torch.device:
    """
    Get the device of the parameters of a module.

    Parameters
    ----------
    m : nn.Module
        The module.

    Returns
    -------
    torch.device
        The device of the parameters of the module.

    Examples
    --------
    >>> m = nn.Linear(10, 2)
    >>> get_device(m)
    device(type='cpu')

    """

    params = list(m.parameters())
    return torch.device("cpu") if not params else params[0].device


def get_clones(
    module: nn.Module,
    n: int,
    *,
    return_list: bool = False,
) -> Union[nn.ModuleList, List[nn.Module]]:
    """
    Get clones of a module.

    Parameters
    ----------
    module : nn.Module
        The module to clone.
    n : int
        The number of clones to create.
    return_list : bool, optional
        Whether to return the clones as a list. If False, the clones are returned as a ModuleList.

    Returns
    -------
    Union[nn.ModuleList, List[nn.Module]]
        The clones of the module.

    Examples
    --------
    >>> m = nn.Linear(10, 2)
    >>> get_clones(m, 3)
    ModuleList(
      (0-2): 3 x Linear(in_features=10, out_features=2, bias=True)
    )

    """

    module_list = [module]
    for _ in range(n - 1):
        module_list.append(copy.deepcopy(module))
    if return_list:
        return module_list
    return nn.ModuleList(module_list)


def get_torch_device(device: device_type) -> torch.device:
    """
    Get a torch device.

    Parameters
    ----------
    device : device_type
        The device to get. This could be None, an integer, a string, or a torch.device.

    Returns
    -------
    torch.device
        The torch device.

    Examples
    --------
    >>> get_torch_device("cuda:0")
    device(type='cuda', index=0)

    """

    if device is None:
        return torch.device("cpu")
    if isinstance(device, (int, str)):
        try:
            device = int(device)
        except:
            pass
        finally:
            device = torch.device(device)
    return device


def empty_cuda_cache(device: device_type) -> None:
    """
    Empty the CUDA cache on a specific device.

    Parameters
    ----------
    device : device_type
        The device on which to empty the CUDA cache.

    Examples
    --------
    >>> empty_cuda_cache("cuda:0")

    """

    device = get_torch_device(device)
    if device.type != "cuda":
        return
    with torch.cuda.device(device):  # pragma: no cover
        torch.cuda.empty_cache()


def is_cpu(device: device_type) -> bool:
    """
    Check if a device is a CPU.

    Parameters
    ----------
    device : device_type
        The device to check.

    Returns
    -------
    bool
        True if the device is a CPU, False otherwise.

    Examples
    --------
    >>> is_cpu("cuda:0")
    False

    """

    return get_torch_device(device).type == "cpu"


def np_batch_to_tensor(np_batch: np_dict_type) -> tensor_dict_type:
    """
    Convert a batch of numpy arrays to tensors.

    Parameters
    ----------
    np_batch : np_dict_type
        The batch of numpy arrays to convert.

    Returns
    -------
    tensor_dict_type
        The batch of tensors.

    Examples
    --------
    >>> np_batch_to_tensor({"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])})
    {'a': tensor([1, 2, 3]), 'b': tensor([4, 5, 6])}

    """

    return {
        key: (
            array
            if not isinstance(array, np.ndarray) or not is_real_numeric(array)
            else torch.from_numpy(array)
        )
        for key, array in np_batch.items()
    }


def tensor_batch_to_np(tensor_batch: np_dict_type) -> np_dict_type:
    """
    Convert a batch of tensors to numpy arrays.

    Parameters
    ----------
    tensor_batch : np_dict_type
        The batch of tensors to convert.

    Returns
    -------
    np_dict_type
        The batch of numpy arrays.

    Examples
    --------
    >>> tensor_batch_to_np({"a": torch.tensor([1, 2, 3]), "b": torch.tensor([4, 5, 6])})
    {'a': array([1, 2, 3], dtype=int64), 'b': array([4, 5, 6], dtype=int64)}

    """

    return {
        k: v if not isinstance(v, Tensor) else v.cpu().numpy()
        for k, v in tensor_batch.items()
    }


def safe_clip_(net: Tensor) -> None:
    """
    Clip the values of a tensor in-place to the valid range for its data type.

    Parameters
    ----------
    net : Tensor
        The tensor to clip.

    Examples
    --------
    >>> net = torch.tensor([1.0, 2.0, 3.0])
    >>> safe_clip_(net)
    >>> net
    tensor([1., 2., 3.])

    """

    finfo = torch.finfo(net.dtype)
    net.clamp_(finfo.min, finfo.max)


@no_type_check
def insert_intermediate_dims(net: TArray, ref: TArray) -> TArray:
    """
    Insert intermediate dimensions into a tensor or numpy array.

    Parameters
    ----------
    net : TArray
        The tensor or numpy array to insert dimensions into, which should have 2 dimensions.
    ref : TArray
        The reference tensor or numpy array. The output will have the
        same number of dimensions as this array.

    Returns
    -------
    TArray
        The tensor or numpy array with inserted dimensions.

    Examples
    --------
    >>> net = torch.tensor([[1.0, 2.0, 3.0]])
    >>> ref = torch.tensor([[[1.0, 2.0, 3.0]]])
    >>> insert_intermediate_dims(net, ref)
    tensor([[[1., 2., 3.]]])

    """

    net_dim = len(net.shape)
    if net_dim != 2:
        raise ValueError(f"only 2-dim tensor is supported, but got {net_dim}")
    dim_diff = len(ref.shape) - net_dim
    if dim_diff == 0:
        return net
    new_shape = net.shape[0], *((1,) * dim_diff), net.shape[1]
    if isinstance(net, Tensor):
        return net.view(*new_shape)
    return net.reshape(new_shape)


@no_type_check
def fix_denormal_states(
    states: tensor_dict_type,
    *,
    eps: float = 1.0e-32,
    verbose: bool = False,
) -> tensor_dict_type:
    """
    Fix denormal states in a dictionary of tensors.

    Parameters
    ----------
    states : tensor_dict_type
        The dictionary of tensors to fix.
    eps : float, optional
        The threshold below which a value is considered denormal. Default is 1.0e-32.
    verbose : bool, optional
        Whether to print information about the denormal ratio. Default is False.

    Returns
    -------
    tensor_dict_type
        The dictionary of tensors with denormal states fixed.

    Examples
    --------
    >>> states = {"a": torch.tensor([1.0, 2.0, 1.0e-33]), "b": torch.tensor([4.0, 5.0, 6.0])}
    >>> fix_denormal_states(states)
    {'a': tensor([1., 2., 0.]), 'b': tensor([4., 5., 6.])}

    """

    new_states = shallow_copy_dict(states)
    num_total = num_denormal_total = 0
    for k, v in states.items():
        if not v.is_floating_point():
            continue
        num_total += v.numel()
        denormal = (v != 0) & (v.abs() < eps)
        num_denormal = denormal.sum().item()
        num_denormal_total += num_denormal
        if num_denormal > 0:
            new_states[k][denormal] = v.new_zeros(num_denormal)
    if verbose:
        console.log(f"denormal ratio : {num_denormal_total / num_total:8.6f}")
    return new_states


def has_batch_norms(m: nn.Module) -> bool:
    """
    Check if a module has any batch normalization layers.

    Parameters
    ----------
    m : nn.Module
        The module to check.

    Returns
    -------
    bool
        True if the module has any batch normalization layers, False otherwise.

    Examples
    --------
    >>> m = nn.Sequential(nn.Linear(10, 2), nn.BatchNorm1d(2))
    >>> has_batch_norms(m)
    True

    """

    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in m.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


def inject_parameters(
    src: nn.Module,
    tgt: nn.Module,
    *,
    strict: Optional[bool] = None,
    src_filter_fn: Optional[Callable[[str], bool]] = None,
    tgt_filter_fn: Optional[Callable[[str], bool]] = None,
    custom_mappings: Optional[Dict[str, str]] = None,
    states_callback: Optional[Callable[[tensor_dict_type], tensor_dict_type]] = None,
) -> None:
    """
    Inject parameters from one module into another.

    Parameters
    ----------
    src : nn.Module
        The source module.
    tgt : nn.Module
        The target module.
    strict : Optional[bool], optional
        Whether to strictly enforce that the keys in the src and tgt state dicts match.
        Default is None, which means strict is True if tgt_filter_fn is None, and False otherwise.
    src_filter_fn : Optional[Callable[[str], bool]], optional
        A function that takes a key and returns True if the corresponding parameter should
        be included from the src state dict. Default is None, which means all parameters are included.
    tgt_filter_fn : Optional[Callable[[str], bool]], optional
        A function that takes a key and returns True if the corresponding parameter should
        be included from the tgt state dict. Default is None, which means all parameters are included.
    custom_mappings : Optional[Dict[str, str]], optional
        A dictionary mapping keys in the src state dict to keys in the tgt state dict.
        Default is None, which means the keys are assumed to match exactly.
    states_callback : Optional[Callable[[tensor_dict_type], tensor_dict_type]], optional
        A function that takes the src state dict and returns a new state dict.
        Default is None, which means the src state dict is used as is.

    Examples
    --------
    >>> src = nn.Linear(10, 2)
    >>> tgt = nn.Linear(10, 2)
    >>> inject_parameters(src, tgt)
    >>> assert torch.allclose(src.weight, tgt.weight)
    >>> assert torch.allclose(src.bias, tgt.bias)

    """

    if strict is None:
        strict = tgt_filter_fn is None
    src_states = src.state_dict()
    tgt_states = tgt.state_dict()
    if src_filter_fn is not None:
        pop_keys = [key for key in src_states if not src_filter_fn(key)]
        for key in pop_keys:
            src_states.pop(key)
    if tgt_filter_fn is not None:
        pop_keys = [key for key in tgt_states if not tgt_filter_fn(key)]
        for key in pop_keys:
            tgt_states.pop(key)
    if states_callback is not None:
        src_states = states_callback(shallow_copy_dict(src_states))
    if len(src_states) != len(tgt_states):
        raise ValueError(f"lengths of states are not identical between {src} and {tgt}")
    new_states = OrderedDict()
    if custom_mappings is not None:
        for src_k, tgt_k in custom_mappings.items():
            new_states[tgt_k] = src_states.pop(src_k)
            tgt_states.pop(tgt_k)
    for (src_k, src_v), (tgt_k, tgt_v) in zip(src_states.items(), tgt_states.items()):
        if src_v.shape != tgt_v.shape:
            raise ValueError(
                f"shape of {src_k} ({list(src_v.shape)}) is not identical with "
                f"shape of {tgt_k} ({list(tgt_v.shape)})"
            )
        new_states[tgt_k] = src_v
    tgt.load_state_dict(new_states, strict=strict)


class Diffs(NamedTuple):
    """
    A named tuple for storing the differences between parameters of two modules.

    Attributes
    ----------
    names1 : List[str]
        The names of the parameters in the first module.
    names2 : List[str]
        The names of the parameters in the second module.
    diffs : List[Tensor]
        The differences between the parameters of the two modules.

    """

    names1: List[str]
    names2: List[str]
    diffs: List[Tensor]


def sorted_param_diffs(m1: nn.Module, m2: nn.Module) -> Diffs:
    """
    Compute the sorted differences between the parameters of two modules,
    often used to check if two modules are identical.

    Parameters
    ----------
    m1 : nn.Module
        The first module.
    m2 : nn.Module
        The second module.

    Returns
    -------
    Diffs
        The differences between the parameters of the two modules.

    Raises
    ------
    ValueError
        If the lengths of the parameters of the two modules are not identical.

    Examples
    --------
    >>> m1 = nn.Linear(10, 2)
    >>> m2 = nn.Linear(10, 2)
    >>> sorted_param_diffs(m1, m2)
    Diffs(names1=['weight', 'bias'], names2=['weight', 'bias'], diffs=[tensor([...]), tensor([...])])

    """

    names1, params1 = zip(*m1.named_parameters())
    names2, params2 = zip(*m2.named_parameters())
    if len(params1) != len(params2):
        raise ValueError(f"lengths of params are not identical between {m1} and {m2}")
    diffs = []
    for p1, p2 in zip(params1, params2):
        (p1, _), (p2, _) = map(torch.sort, [p1.view(-1), p2.view(-1)])
        diffs.append(torch.abs(p1.data - p2.data))
    return Diffs(list(names1), list(names2), diffs)


def get_gradient(
    y: Tensor,
    x: Tensor,
    retain_graph: bool = False,
    create_graph: bool = False,
) -> Union[Tensor, Tuple[Tensor, ...]]:
    """
    Compute the gradient of y with respect to x.

    Parameters
    ----------
    y : Tensor
        The tensor to compute the gradient of.
    x : Tensor
        The tensor to compute the gradient with respect to.
    retain_graph : bool, optional
        Whether to retain the computation graph. Default is False.
    create_graph : bool, optional
        Whether to create a computation graph. Default is False.

    Returns
    -------
    Union[Tensor, Tuple[Tensor, ...]]
        The gradient of y with respect to x.

    Examples
    --------
    >>> x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    >>> y = x ** 2
    >>> get_gradient(y, x)
    tensor([2., 4., 6.])

    """

    grads = torch.autograd.grad(y, x, torch.ones_like(y), retain_graph, create_graph)
    return grads[0] if len(grads) == 1 else grads


def set_requires_grad(module: nn.Module, requires_grad: bool = False) -> None:
    """
    Set the `requires_grad` attribute of all parameters of a module.

    Parameters
    ----------
    module : nn.Module
        The module.
    requires_grad : bool, optional
        The value to set the `requires_grad` attribute to. Default is False.

    Examples
    --------
    >>> m = nn.Linear(10, 2)
    >>> set_requires_grad(m, True)

    """

    for param in module.parameters():
        param.requires_grad = requires_grad


def to_eval(module: GenericM) -> GenericM:
    """
    Set a module to evaluation mode and disable gradient computation for its parameters.

    Parameters
    ----------
    module : GenericM
        The module.

    Returns
    -------
    GenericM
        The module.

    Examples
    --------
    >>> m = nn.Linear(10, 2)
    >>> m = to_eval(m)

    """

    module.eval()
    set_requires_grad(module, False)
    return module


def scheduler_requires_metric(scheduler: Any) -> bool:
    """
    Check if a scheduler requires a metric.

    Parameters
    ----------
    scheduler : Any
        The scheduler.

    Returns
    -------
    bool
        True if the scheduler requires a metric, False otherwise.

    Examples
    --------
    >>> scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(torch.optim.SGD(m.parameters(), lr=0.1))
    >>> scheduler_requires_metric(scheduler)
    True

    """

    return check_requires(scheduler.step, "metrics")


# This is a modified version of https://github.com/sksq96/pytorch-summary
#  So it can summary `carefree-learn` model structures better
def summary(
    m: nn.Module,
    sample_batch: tensor_dict_type,
    *,
    return_only: bool = False,
    summary_forward: Optional[Callable[[tensor_dict_type], None]] = None,
) -> str:
    """
    Print a summary of a module.

    Parameters
    ----------
    m : nn.Module
        The module.
    sample_batch : tensor_dict_type
        A sample batch of input to the module.
    return_only : bool, optional
        Whether to return the summary as a string instead of printing it. Default is False.
    summary_forward : Optional[Callable[[tensor_dict_type], None]], optional
        A function that takes a batch of input and passes it through the module. If None, the module's forward method is used.

    Returns
    -------
    str
        The summary of the module.

    Examples
    --------
    >>> m = nn.Linear(10, 2)
    >>> sample_batch = {"input": torch.randn(1, 10)}
    >>> print(summary(m, sample_batch, return_only=True, summary_forward=lambda x: m(x["input"])))
    ========================================================================================================================
    Layer (type)                             Input Shape                             Output Shape    Trainable Param #
    ------------------------------------------------------------------------------------------------------------------------
    Linear                                      [-1, 10]                                  [-1, 2]                   22
    ========================================================================================================================
    Total params: 22
    Trainable params: 22
    Non-trainable params: 0
    ------------------------------------------------------------------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.00
    Params size (MB): 0.00
    Estimated Total Size (MB): 0.00
    ------------------------------------------------------------------------------------------------------------------------

    """

    def _get_param_counts(m: nn.Module) -> Tuple[int, int]:
        num_params = 0
        num_trainable_params = 0
        for p in m.parameters():
            local_num_params = int(round(prod(p.data.shape)))
            num_params += local_num_params
            if p.requires_grad:
                num_trainable_params += local_num_params
        return num_params, num_trainable_params

    def register_hook(m: nn.Module) -> None:
        def inject_output_shape(o: Any, r: Dict[str, Any], prefix: str = "") -> None:
            if isinstance(o, Tensor):
                r[prefix] = list(o.shape)
            elif isinstance(o, (list, tuple)):
                for i, elem in enumerate(o):
                    inject_output_shape(elem, r, f"{prefix}{i}.")
            elif isinstance(o, dict):
                for k, v in o.items():
                    inject_output_shape(v, r, f"{prefix}{k}.")

        def hook(m: nn.Module, inp: Any, output: Any) -> None:
            m_name = module_names.get(m)
            if m_name is None:  # pragma: no cover
                return

            if not inp:  # pragma: no cover
                return
            inp = inp[0]
            if not isinstance(inp, Tensor):
                return

            m_dict: Dict[str, Any] = OrderedDict()
            m_dict["input_shape"] = list(inp.shape)
            output_shape_res = m_dict["output_shape"] = {}
            inject_output_shape(output, output_shape_res)

            num_params_, num_trainable_params_ = _get_param_counts(m)
            m_dict["num_params"] = num_params_
            m_dict["num_trainable_params"] = num_trainable_params_
            raw_summary_dict[m_name] = m_dict

        if not isinstance(m, torch.jit.ScriptModule):
            hooks.append(m.register_forward_hook(hook))

    # get names
    def _inject_names(m: nn.Module, previous_names: List[str]) -> None:
        info_list = []
        for child in m.children():
            current_names = previous_names + [type(child).__name__]
            current_name = ".".join(current_names)
            module_names[child] = current_name
            info_list.append((child, current_name, current_names))
        counts: Dict[str, int] = defaultdict(int)
        idx_mapping: Dict[nn.Module, int] = {}
        for child, current_name, _ in info_list:
            idx_mapping[child] = counts[current_name]
            counts[current_name] += 1
        for child, current_name, current_names in info_list:
            if counts[current_name] == 1:
                continue
            current_name = f"{current_name}-{idx_mapping[child]}"
            module_names[child] = current_name
            current_names[-1] = current_name.split(".")[-1]
        for child, _, current_names in info_list:
            _inject_names(child, current_names)

    module_names: Dict[nn.Module, str] = OrderedDict()
    model_name = type(m).__name__
    module_names[m] = model_name
    _inject_names(m, [model_name])

    # create properties
    raw_summary_dict: Dict[str, Any] = OrderedDict()
    hooks: List[Any] = []

    # register hook
    m.apply(register_hook)

    # make a forward pass
    with eval_context(m, use_grad=None):
        (summary_forward or m)(sample_batch)
        for param in m.parameters():
            param.grad = None

    # remove these hooks
    for h in hooks:
        h.remove()

    # get hierarchy
    hierarchy: Dict[str, Any] = OrderedDict()
    for key in raw_summary_dict:
        split = key.split(".")
        d = hierarchy
        for elem in split[:-1]:
            d = d.setdefault(elem, OrderedDict())
        d.setdefault(split[-1], None)

    # reconstruct summary_dict
    def _inject_summary(current_hierarchy: Any, previous_keys: List[str]) -> None:
        current_layer = len(previous_keys)
        current_count = hierarchy_counts.get(current_layer, 0)
        prefix = "  " * current_layer
        for k, v in current_hierarchy.items():
            current_keys = previous_keys + [k]
            concat_k = ".".join(current_keys)
            current_summary = raw_summary_dict.get(concat_k)
            summary_dict[f"{prefix}{k}-{current_count}"] = current_summary
            hierarchy_counts[current_layer] = current_count + 1
            if v is not None:
                _inject_summary(v, current_keys)

    hierarchy_counts: Dict[int, int] = {}
    summary_dict: Dict[str, Any] = OrderedDict()
    _inject_summary(hierarchy, [])

    line_length = 120
    messages = ["=" * line_length]
    line_format = "{:30}  {:>20} {:>40} {:>20}"
    headers = "Layer (type)", "Input Shape", "Output Shape", "Trainable Param #"
    messages.append(line_format.format(*headers))
    messages.append("-" * line_length)
    total_output = 0
    for layer, layer_summary in summary_dict.items():
        layer_name = "-".join(layer.split("-")[:-1])
        valid_layer_name = layer_name.strip()
        num_spaces = len(layer_name) - len(valid_layer_name)
        valid_layer_name = truncate_string_to_length(valid_layer_name, 30 - num_spaces)
        layer_name = " " * num_spaces + valid_layer_name
        if layer_summary is None:
            messages.append(line_format.format(layer_name, "", "", ""))
        else:
            output_shape_item = layer_summary["output_shape"]
            all_output_shapes: List[List[int]] = []
            only_one = len(output_shape_item) == 1
            for i, key in enumerate(sorted(output_shape_item)):
                value = output_shape_item[key]
                if only_one:
                    key = ""
                output_shape_str = f"{key} {str(value):>16s}"
                ntp_str = "{0:,}".format(layer_summary["num_trainable_params"])
                is_title = i == 0
                messages.append(
                    line_format.format(
                        layer_name if is_title else "",
                        str(layer_summary["input_shape"]) if is_title else "",
                        output_shape_str,
                        ntp_str if is_title else "",
                    )
                )
                all_output_shapes.append(value)
            for shape in all_output_shapes:
                total_output += int(round(prod(shape)))

    total_params, trainable_params = _get_param_counts(m)
    # assume 4 bytes/number (float on cuda).
    x_batch = sample_batch[INPUT_KEY]
    get_size = lambda t: abs(prod(t.shape[1:]) * 4.0 / (1024**2.0))
    if not isinstance(x_batch, list):
        x_batch = [x_batch]
    total_input_size = sum(map(get_size, x_batch))
    # x2 for gradients
    total_output_size = abs(2.0 * total_output * 4.0 / (1024**2.0))
    total_params_size = abs(total_params * 4.0 / (1024**2.0))
    total_size = total_params_size + total_output_size + total_input_size

    non_trainable_params = total_params - trainable_params
    messages.append("=" * line_length)
    messages.append("Total params: {0:,}".format(total_params))
    messages.append("Trainable params: {0:,}".format(trainable_params))
    messages.append("Non-trainable params: {0:,}".format(non_trainable_params))
    messages.append("-" * line_length)
    messages.append("Input size (MB): %0.2f" % total_input_size)
    messages.append("Forward/backward pass size (MB): %0.2f" % total_output_size)
    messages.append("Params size (MB): %0.2f" % total_params_size)
    messages.append("Estimated Total Size (MB): %0.2f" % total_size)
    messages.append("-" * line_length)
    msg = "\n".join(messages)
    if not return_only:
        console.log(msg)
    return msg


class toggle_optimizer:
    """
    A context manager for only enabling the gradients of a module for a specific optimizer,
    and disabling the gradients of other parameters.

    Parameters
    ----------
    m : nn.Module
        The module to toggle the gradients of.
    optimizer : Optimizer
        The optimizer.
    enabled : bool, optional
        Whether to enable this context manager. Default is True.

    Examples
    --------
    >>> m = nn.Linear(10, 2)
    >>> optimizer = torch.optim.SGD([m.weight], lr=0.1)
    >>> with toggle_optimizer(m, optimizer):
    ...     print(m.weight.requires_grad)  # True
    ...     print(m.bias.requires_grad)  # False
    >>> print(m.weight.requires_grad)  # True
    >>> print(m.bias.requires_grad)  # True

    """

    def __init__(self, m: nn.Module, optimizer: Optimizer, *, enabled: bool = True):
        self.m = m
        self.optimizer = optimizer
        self.enabled = enabled
        self.requires_grad: Dict[str, bool] = {}

    def __enter__(self) -> None:
        if not self.enabled:
            return
        self.requires_grad = {k: p.requires_grad for k, p in self.m.named_parameters()}
        for p in self.m.parameters():
            p.requires_grad = False
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                p.requires_grad = True

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if not self.enabled:
            return
        for k, p in self.m.named_parameters():
            requires_grad = self.requires_grad.get(k)
            if requires_grad is not None:
                p.requires_grad = requires_grad


class toggle_module:
    """
    A context manager for temporarily enabling or disabling the gradients of a module.

    Parameters
    ----------
    m : nn.Module
        The module to toggle the gradients of.
    requires_grad : bool
        Whether to enable gradient computation.
    enabled : bool, optional
        Whether to enable this context manager. Default is True.

    Examples
    --------
    >>> m = nn.Linear(10, 2)
    >>> with toggle_module(m, requires_grad=False):
    ...     print(m.weight.requires_grad)  # False
    >>> print(m.weight.requires_grad)  # True

    """

    def __init__(self, m: nn.Module, *, requires_grad: bool, enabled: bool = True):
        self.m = m
        self.enabled = enabled
        self.requires_grad = requires_grad
        self.backup: Dict[str, bool] = {}

    def __enter__(self) -> None:
        if not self.enabled:
            return
        self.backup = {k: p.requires_grad for k, p in self.m.named_parameters()}
        for p in self.m.parameters():
            p.requires_grad = self.requires_grad

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if not self.enabled:
            return
        for k, p in self.m.named_parameters():
            requires_grad = self.backup.get(k)
            if requires_grad is not None:
                p.requires_grad = requires_grad


class mode_context:
    """
    A context manager for temporarily setting the mode of a module and
    optionally enabling or disabling gradient computation.

    Parameters
    ----------
    module : nn.Module
        The module to set the mode of.
    to_train : Optional[bool]
        Whether to set the module to training mode.
        If None, the mode of the module is not changed.
    use_grad : Optional[bool]
        Whether to enable gradient computation.
        If None, the `requires_grad` attribute of the parameters of the module is not changed.
    use_inference : Optional[bool]
        Whether to enable inference mode.
        If None, inference mode is not used.

    Examples
    --------
    >>> m = nn.Linear(10, 2)
    >>> with mode_context(m, to_train=False, use_grad=False):
    ...     print(m.training, any(p.requires_grad for p in m.parameters()))
    False False
    >>> print(m.training, any(p.requires_grad for p in m.parameters()))
    True True

    """

    def __init__(
        self,
        module: nn.Module,
        *,
        to_train: Optional[bool],
        use_grad: Optional[bool],
        use_inference: Optional[bool] = None,
    ):
        self._to_train = to_train
        self._module, self._training = module, module.training
        self._grad_context: ContextManager
        self._inference_context: ContextManager
        if use_grad is None:
            self._grad_context = nullcontext()
        else:
            self._grad_context = torch.enable_grad() if use_grad else torch.no_grad()
        if use_inference is None or not use_inference:
            self._inference_context = nullcontext()
        else:
            self._inference_context = torch.inference_mode()

    def __enter__(self) -> None:
        if self._to_train is not None:
            self._module.train(mode=self._to_train)
        self._grad_context.__enter__()
        self._inference_context.__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._to_train is not None:
            self._module.train(mode=self._training)
        self._inference_context.__exit__(exc_type, exc_val, exc_tb)
        self._grad_context.__exit__(exc_type, exc_val, exc_tb)


class train_context(mode_context):
    """
    A context manager for temporarily setting a module to training mode and
    optionally enabling gradient computation.

    Parameters
    ----------
    module : nn.Module
        The module to set to training mode.
    use_grad : bool, optional
        Whether to enable gradient computation. Default is True.

    Examples
    --------
    >>> m = nn.Linear(10, 2)
    >>> m.eval()
    >>> with train_context(m):
    ...     print(m.training)
    True
    >>> print(m.training)
    False

    """

    def __init__(self, module: nn.Module, *, use_grad: bool = True):
        super().__init__(module, to_train=True, use_grad=use_grad, use_inference=False)


class eval_context(mode_context):
    """
    A context manager for temporarily setting a module to evaluation mode
    and optionally disabling gradient computation.

    Parameters
    ----------
    module : nn.Module
        The module to set to evaluation mode.
    use_grad : Optional[bool], optional
        Whether to enable gradient computation. Default is False.
    use_inference : Optional[bool], optional
        Whether to enable inference mode.
        If None and `use_grad` is not None, inference mode is set to the opposite of `use_grad`.

    Examples
    --------
    >>> m = nn.Linear(10, 2)
    >>> m.train()
    >>> with eval_context(m):
    ...     print(m.training)
    False
    >>> print(m.training)
    True

    """

    def __init__(
        self,
        module: nn.Module,
        *,
        use_grad: Optional[bool] = False,
        use_inference: Optional[bool] = None,
    ):
        if use_inference is None and use_grad is not None:
            use_inference = not use_grad
        super().__init__(
            module,
            to_train=False,
            use_grad=use_grad,
            use_inference=use_inference,
        )


class no_grad_context:
    """
    A context manager for disabling gradient computation.

    Parameters
    ----------
    enabled : bool
        Whether to enable this context manager.

    Examples
    --------
    >>> x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    >>> with no_grad_context(enabled=True):
    ...     y = x ** 2
    >>> y.requires_grad
    False

    """

    def __init__(self, *, enabled: bool):
        self.enabled = enabled
        self._context = torch.no_grad()

    def __enter__(self) -> None:
        if not self.enabled:
            return
        self._context.__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if not self.enabled:
            return
        self._context.__exit__(exc_type, exc_val, exc_tb)


class Initializer:
    """
    A class for initializing the parameters of a module.

    Parameters
    ----------
    config : Optional[Dict[str, Any]], optional
        A dictionary of configuration options.
        Default is None, which means an empty dictionary is used.

    Attributes
    ----------
    defined_initialization : set
        A set of the names of the defined initialization methods.
    custom_initializer : Dict[str, Callable]
        A dictionary mapping the names of custom initialization methods to the methods themselves.

    Examples
    --------
    >>> initializer = Initializer()
    >>> m = nn.Linear(10, 2)
    >>> initializer.initialize(m.weight, "xavier_uniform")

    """

    defined_initialization = {
        "xavier_uniform",
        "xavier_normal",
        "normal",
        "truncated_normal",
    }
    custom_initializer: Dict[str, Callable] = {}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._verbose_level = self.config.setdefault("verbose_level", 2)

    def initialize(self, param: param_type, method: str) -> Any:
        """
        Initialize a parameter using a specified method.

        Parameters
        ----------
        param : param_type
            The parameter to initialize.
        method : str
            The name of the initialization method to use.

        Returns
        -------
        Any
            The result of the initialization method.

        Examples
        --------
        >>> initializer = Initializer()
        >>> m = nn.Linear(10, 2)
        >>> initializer.initialize(m.weight, "xavier_uniform")

        """

        custom_initializer = self.custom_initializer.get(method)
        if custom_initializer is None:
            return getattr(self, method)(param)
        return custom_initializer(self, param)

    @classmethod
    def register(cls, name: str) -> Callable[[Callable], Callable]:
        """
        Register a custom initialization method.

        Parameters
        ----------
        name : str
            The name of the initialization method to register.

        Returns
        -------
        Callable[[Callable], Callable]
            A decorator for registering the initialization method.

        Examples
        --------
        >>> @Initializer.register("custom")
        ... def custom(self, param):
        ...     with torch.no_grad():
        ...         param.data.fill_(1.0)

        """

        def _register(f: Callable) -> Callable:
            if name in cls.defined_initialization:
                raise ValueError(f"'{name}' initializer is already defined")
            cls.defined_initialization.add(name)
            cls.custom_initializer[name] = f
            return f

        return _register

    def xavier_uniform(self, param: param_type) -> None:
        """
        Initialize a parameter using the Xavier uniform initialization method.

        Parameters
        ----------
        param : param_type
            The parameter to initialize.

        Examples
        --------
        >>> initializer = Initializer()
        >>> m = nn.Linear(10, 2)
        >>> initializer.xavier_uniform(m.weight)

        """

        gain = self.config.setdefault("gain", 1.0)
        nn.init.xavier_uniform_(param.data, gain)

    def xavier_normal(self, param: param_type) -> None:
        """
        Initialize a parameter using the Xavier normal initialization method.

        Parameters
        ----------
        param : param_type
            The parameter to initialize.

        Examples
        --------
        >>> initializer = Initializer()
        >>> m = nn.Linear(10, 2)
        >>> initializer.xavier_normal(m.weight)

        """

        gain = self.config.setdefault("gain", 1.0)
        nn.init.xavier_normal_(param.data, gain)

    def normal(self, param: param_type) -> None:
        """
        Initialize a parameter using the normal initialization method.

        Parameters
        ----------
        param : param_type
            The parameter to initialize.

        Examples
        --------
        >>> initializer = Initializer()
        >>> m = nn.Linear(10, 2)
        >>> initializer.normal(m.weight)

        """

        mean = self.config.setdefault("mean", 0.0)
        std = self.config.setdefault("std", 1.0)
        with torch.no_grad():
            param.data.normal_(mean, std)

    def truncated_normal(self, param: param_type) -> None:
        """
        Initialize a parameter using the truncated normal initialization method.

        Parameters
        ----------
        param : param_type
            The parameter to initialize.

        Examples
        --------
        >>> initializer = Initializer()
        >>> m = nn.Linear(10, 2)
        >>> initializer.truncated_normal(m.weight)

        """

        span = self.config.setdefault("span", 2.0)
        mean = self.config.setdefault("mean", 0.0)
        std = self.config.setdefault("std", 1.0)
        tol = self.config.setdefault("tol", 0.0)
        epoch = self.config.setdefault("epoch", 20)
        num_elem = param.numel()
        weight_base = param.new_empty(num_elem).normal_()
        get_invalid = lambda w: (w < mean - span * std) | (w > mean + span * std)
        invalid = get_invalid(weight_base)
        success = False
        for _ in range(epoch):
            num_invalid = invalid.sum().item()
            if num_invalid / num_elem <= tol:
                success = True
                break
            with torch.no_grad():
                weight_base[invalid] = param.new_empty(num_invalid).normal_()
                invalid = get_invalid(weight_base)
        if not success:
            console.warn(
                "invalid ratio for truncated normal : "
                f"{invalid.to(torch.float32).mean():8.6f}, it might cause by "
                f"too little epoch ({epoch}), too small span ({span}), "
                f"or too small tolerance ({tol})",
            )
        with torch.no_grad():
            param.data.copy_(weight_base.reshape(param.shape))
            param.data.mul_(std).add_(mean)

    def orthogonal(self, param: param_type) -> None:
        """
        Initialize a parameter using the orthogonal initialization method.

        Parameters
        ----------
        param : param_type
            The parameter to initialize.

        Examples
        --------
        >>> initializer = Initializer()
        >>> m = nn.Linear(10, 2)
        >>> initializer.orthogonal(m.weight)

        """

        gain = self.config.setdefault("gain", 1.0)
        nn.init.orthogonal_(param.data, gain)


class ONNX:
    """
    A class for making predictions with an ONNX model.

    Parameters
    ----------
    onnx_path : str
        The path to the ONNX model file.

    Attributes
    ----------
    ort_session : InferenceSession
        The ONNX Runtime inference session.
    output_names : List[str]
        The names of the output nodes of the model.

    Raises
    ------
    ValueError
        If the `onnxruntime` package is not installed or `onnx_path` is not provided.

    Examples
    --------
    >>> model = ONNX("model.onnx")
    >>> inputs = {"input": np.array([1, 2, 3])}
    >>> outputs = model.predict(inputs)
    >>> print(outputs)
    {'output': array([2., 4., 6.])}

    """

    def __init__(self, onnx_path: str):
        if InferenceSession is None:  # pragma: no cover
            msg = "`ONNX` is not available when `onnxruntime` is not installed"
            raise ValueError(msg)
        self.ort_session = InferenceSession(onnx_path)
        self.output_names = [node.name for node in self.ort_session.get_outputs()]

    def predict(self, new_inputs: np_dict_type) -> np_dict_type:
        """
        Make a prediction with the ONNX model.

        Parameters
        ----------
        new_inputs : np_dict_type
            A dictionary mapping input node names to input data.

        Returns
        -------
        np_dict_type
            A dictionary mapping output node names to output data.

        Examples
        --------
        >>> model = ONNX("model.onnx")
        >>> inputs = {"input": np.array([1, 2, 3])}
        >>> outputs = model.predict(inputs)
        >>> print(outputs)
        {'output': array([2., 4., 6.])}

        """

        ort_inputs = {
            node.name: to_standard(new_inputs[node.name])
            for node in self.ort_session.get_inputs()
        }
        return dict(zip(self.output_names, self.ort_session.run(None, ort_inputs)))
