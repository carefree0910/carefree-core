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

## License

`carefree-core` is MIT licensed, as found in the [`LICENSE`](https://github.com/carefree0910/carefree-core/blob/main/LICENSE) file.
