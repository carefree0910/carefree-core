import torch
import pytest
import unittest

import core.learn as cflearn
import torch.nn as nn

from torch import Tensor
from typing import Any
from unittest.mock import Mock


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
        @cflearn.register_loss("foo")
        class _(cflearn.ILoss):
            def forward(self, forward_results, batch, state=None) -> None:
                return None

        x = torch.randn(13, 1)
        y = torch.randn(13, 1)
        mse = get_loss("mae", x, y)
        mse = get_loss("mse", x, y)
        corr = get_loss("corr", x, y)
        multi = get_loss(
            "multi_loss",
            x,
            y,
            losses=[
                {"name": "mse", "weight": 0.17},
                {"name": "corr", "weight": 0.19},
                {"name": "foo", "weight": 123.456},
            ],
        )[cflearn.LOSS_KEY]
        multi = get_loss(
            "multi_loss",
            x,
            y,
            losses=[
                {"name": "mse", "weight": 0.17},
                {"name": "corr", "weight": 0.19},
                {
                    "name": "multi_loss",
                    "config": {
                        "losses": [
                            {"name": "mse", "weight": 0.17},
                            {"name": "corr", "weight": 0.19},
                        ]
                    },
                    "weight": 0.64,
                },
            ],
        )[cflearn.LOSS_KEY]
        gt = 0.17 * mse + 0.19 * corr
        torch.testing.assert_close(multi, gt + 0.64 * gt)

    def test_multi_loss_legacy_result_contracts(self) -> None:
        tensor_loss = torch.tensor(2.0)
        dict_loss = torch.tensor(3.0)
        auxiliary_loss = torch.tensor(5.0)
        dict_result = {
            cflearn.LOSS_KEY: dict_loss,
            "auxiliary": auxiliary_loss,
        }

        @cflearn.register_loss("contract_tensor", allow_duplicate=True)
        class TensorLoss(cflearn.ILoss):
            def forward(self, forward_results, batch, state=None):
                return tensor_loss

        @cflearn.register_loss("contract_dict", allow_duplicate=True)
        class DictLoss(cflearn.ILoss):
            def forward(self, forward_results, batch, state=None):
                return dict_result

        @cflearn.register_loss("contract_none", allow_duplicate=True)
        class NoneLoss(cflearn.ILoss):
            def forward(self, forward_results, batch, state=None) -> None:
                return None

        loss = cflearn.MultiLoss(
            [
                {"name": "contract_tensor", "tag": "tensor", "weight": 2.0},
                {"name": "contract_dict", "tag": "dict", "weight": 3.0},
                {"name": "contract_none", "tag": "none", "weight": 4.0},
            ]
        )
        result = loss({}, {})

        self.assertSetEqual(set(result), {"tensor", "dict_auxiliary", "loss"})
        torch.testing.assert_close(result["tensor"], tensor_loss)
        torch.testing.assert_close(result["dict_auxiliary"], auxiliary_loss)
        torch.testing.assert_close(
            result[cflearn.LOSS_KEY],
            2.0 * tensor_loss + 3.0 * dict_loss,
        )

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="MultiLoss currently pops the primary loss from child dictionaries",
    )
    def test_multi_loss_preserves_child_dict_results(self) -> None:
        primary = torch.tensor(2.0)
        auxiliary = torch.tensor(3.0)
        result = {cflearn.LOSS_KEY: primary, "auxiliary": auxiliary}
        original = result.copy()

        @cflearn.register_loss("contract_preserved_dict", allow_duplicate=True)
        class DictLoss(cflearn.ILoss):
            def forward(self, forward_results, batch, state=None):
                return result

        cflearn.MultiLoss([{"name": "contract_preserved_dict"}])({}, {})
        self.assertDictEqual(result, original)

    def test_train_step_loss_legacy_shape(self) -> None:
        loss = torch.tensor(2.0)
        loss_tensors = {cflearn.LOSS_KEY: loss[None]}
        result = cflearn.TrainStepLoss(loss, loss_tensors)

        self.assertTupleEqual(result._fields, ("loss", "loss_tensors"))
        self.assertEqual(len(result), 2)
        self.assertIs(result[0], loss)
        self.assertIs(result[1], loss_tensors)
        self.assertIs(result.loss, loss)
        self.assertIs(result.loss_tensors, loss_tensors)
        positional_loss, positional_loss_tensors = result
        self.assertIs(positional_loss, loss)
        self.assertIs(positional_loss_tensors, loss_tensors)

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="MultiLoss currently accepts an empty loss configuration",
    )
    def test_multi_loss_rejects_empty_configuration(self) -> None:
        with self.assertRaises(ValueError):
            cflearn.MultiLoss([])

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="MultiLoss currently overwrites duplicate effective tags",
    )
    def test_multi_loss_rejects_duplicate_effective_tags(self) -> None:
        with self.assertRaises(ValueError):
            cflearn.MultiLoss(
                [
                    {"name": "mse", "tag": "duplicate"},
                    {"name": "mae", "tag": "duplicate"},
                ]
            )

    @pytest.mark.xfail(
        strict=True,
        raises=ZeroDivisionError,
        reason="IModel.evaluate currently divides by zero without losses or metrics",
    )
    def test_evaluate_defines_empty_score_boundary(self) -> None:
        outputs = cflearn.InferenceOutputs({}, {}, None, None)
        inference = Mock()
        inference.get_outputs.return_value = outputs
        try:
            result = cflearn.IModel.evaluate(
                Mock(),
                cflearn.TrainerConfig(),
                None,
                inference,
                Mock(),
            )
        except ValueError:
            return
        self.assertIs(result, outputs)
        self.assertIsNotNone(result.metric_outputs)


if __name__ == "__main__":
    unittest.main()
