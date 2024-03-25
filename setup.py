from setuptools import setup
from setuptools import find_packages


VERSION = "0.1.0"
DESCRIPTION = "Meta framework for Deep Learning frameworks with PyTorch"
with open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="carefree-core",
    version=VERSION,
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        "dill",
        "rich",
        "onnx",
        "tqdm",
        "wandb",
        "future",
        "pathos",
        "pillow",
        "psutil",
        "aiohttp",
        "fastapi",
        "uvicorn",
        "networkx",
        "requests",
        "matplotlib",
        "websockets",
        "onnxruntime",
        "safetensors",
        "numpy>=1.22.3",
        "pydantic>=2.0.0",
        "accelerate>=0.28.0",
        "onnx-simplifier>=0.4.1",
    ],
    author="carefree0910",
    author_email="syameimaru.saki@gmail.com",
    url="https://github.com/carefree0910/carefree-core",
    download_url=f"https://github.com/carefree0910/carefree-core/archive/v{VERSION}.tar.gz",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="python framework deep-learning pytorch",
)
