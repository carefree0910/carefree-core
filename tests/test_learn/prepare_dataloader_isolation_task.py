import torch
import argparse

import core.learn as cflearn
import core.learn.testing as testing

from torch.utils.data import DataLoader
from accelerate import Accelerator
from torch.utils.data import TensorDataset
from accelerate import DataLoaderConfiguration


def make_sibling() -> DataLoader:
    dataset = TensorDataset(torch.tensor([1, 2]))
    return DataLoader(dataset, batch_size=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dispatch-batches", action="store_true")
    args = parser.parse_args()
    accelerator = Accelerator(
        cpu=True,
        dataloader_config=DataLoaderConfiguration(
            dispatch_batches=args.dispatch_batches,
        ),
    )
    before = accelerator.prepare(make_sibling())
    baseline_before_values = [batch[0].item() for batch in before]
    assert baseline_before_values == [1, 2]

    data = testing.arange_data(n=2, batch_size=1)[0]
    core_loader = data.build_loaders()[0]
    prepared = cflearn.prepare_dataloaders(accelerator, core_loader)[0]
    after = accelerator.prepare(make_sibling())

    assert len(list(prepared)) == 2
    try:
        repeated_before_values = [batch[0].item() for batch in before]
        after_values = [batch[0].item() for batch in after]
    except AttributeError as err:
        if "has no attribute 'data'" not in str(err):
            raise
        raise SystemExit(42)
    assert repeated_before_values == after_values == baseline_before_values


if __name__ == "__main__":
    main()
