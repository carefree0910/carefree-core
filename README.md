Meta framework for Deep Learning frameworks with [PyTorch](https://pytorch.org/).

## Getting Started

`carefree-core` requires:

- Python 3.8 or higher.
- A suitable [PyTorch](https://pytorch.org/) installation.

`carefree-core` is not meant to be installed, but rather as a 'portable' package that integrates into your own project.

To use this library, follow the instructions below:

- Copy the `core` folder to somewhere in your project.
- Copy the `install_requires` listed in `setup.py` into your own `setup.py`.
- Copy the `setup.cfg` file into your own project, if you want to follow the `mypy` style.

And that's all - the codes are yours, modify them as you wish!

## Introductions

### `toolkit` package

> - `core/toolkit`
> - Originates from [`carefree-toolkit`](https://github.com/carefree0910/carefree-toolkit).

This package is meant to hold the common classes / functions that are used across the whole library.

### `flow` package

> - `core/flow`
> - Originates from [`carefree-workflow`](https://github.com/carefree0910/carefree-workflow).

This package is a lightweight package for building arbitray workflows, here are the highlights:

- **Async**: `async` is by design.
- **Parallel**: nodes can be executed in parallel.
- **Powerful**: complex locks / logics / dependencies can be handled.
  - You can even perform a loop with loop backs in the workflow!
- **Automated**:
  - All nodes, as well as the workflow itself, can be **automatically** turned into RESTful APIs.
  - Detailed [documentation](https://github.com/carefree0910/carefree-workflow/tree/main/docs.md) of the design / nodes / workflow / ... can be **automatically** generated, which makes this package and its extended versions agent-friendly.
    - That is to say, you can build a GPT-agent on top of this package by simply feed the auto-generated documentation to it. After which, you can interact with the agent via natural language and it will tell you how to build the workflow you want (it may even be able to give you the final workflow JSON directly)!
    - We even support auto-generating a 'RAG friendly' version of the documentation, which makes **R**etrieval-**A**ugmented **G**eneration easier. This version uses `__RAG__` as the special separator, so you can chunk the documentation into suitable parts for RAG.
- **Extensible**: you can easily extend the package with your own nodes.
- **Serializable**: the workflow can be serialized into / deserialized from a single JSON file.
- **Human Readable**: the workflow JSON file is human readable and easy to understand.
- **Lightweight**: the package is lightweight (core implementation is ~500 lines of code in a single file `core/flow/core.py`) and easy to use.

### `learn` package

> - `core/learn`
> - Originates from [`carefree-learn`](https://github.com/carefree0910/carefree-learn).

This package is the 'main' package of the PyTorch framework, here are some main design principles:

- **Data**: Will use `torch.utils.data.DataLoader` as the data loader.
- **Model**: Will be split into `module` and `model`:
  - **`module`** is the key part of the Model, and should be self-contained at inference stage.
  - **`model`** is the wrapper of `module`, and should contain the training / evaluation / ... logic.
- **Trainer**: Will use the [`accelerate`](https://github.com/huggingface/accelerate) library.
- **Training Abstraction**: Will use `TrainStep` for fine-grained control.

## License

`carefree-core` is MIT licensed, as found in the [`LICENSE`](https://github.com/carefree0910/carefree-core/blob/main/LICENSE) file.
