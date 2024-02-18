import torch
import unittest

import core.learn as cflearn
import torch.nn as nn

from torch import Tensor
from typing import Any


def get_loss(name: str, x: Tensor, y: Tensor, **kwargs: Any) -> Tensor:
    loss = cflearn.build_loss(name, **kwargs)
    return loss({cflearn.PREDICTIONS_KEY: x}, {cflearn.LABEL_KEY: y})


class TestLosses(unittest.TestCase):
    def test_bce_loss(self) -> None:
        x = torch.randn(7, 1)
        y = torch.randint(0, 2, (7, 1)).float()
        bce = get_loss("bce", x, y)
        gt_bce = nn.BCEWithLogitsLoss()(x, y)
        torch.testing.assert_close(bce, gt_bce)

    def test_corr_loss(self) -> None:
        x = torch.randn(11, 1)
        y = torch.randn(11, 1)
        corr = get_loss("corr", x, y)
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        std = torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2))
        gt_corr = -torch.sum(vx * vy) / std
        torch.testing.assert_close(corr, gt_corr)

    def test_multi_loss(self) -> None:
        x = torch.randn(13, 1)
        y = torch.randn(13, 1)
        mse = get_loss("mse", x, y)
        corr = get_loss("corr", x, y)
        multi = get_loss(
            "multi_loss",
            x,
            y,
            losses=[
                {"name": "mse", "weight": 0.17},
                {"name": "corr", "weight": 0.19},
            ],
        )[cflearn.LOSS_KEY]
        torch.testing.assert_close(multi, 0.17 * mse + 0.19 * corr)


if __name__ == "__main__":
    unittest.main()
