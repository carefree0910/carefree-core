import torch
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

    def test_multi_loss_rejects_empty_configuration(self) -> None:
        with self.assertRaises(ValueError):
            cflearn.MultiLoss([])

    def test_multi_loss_rejects_duplicate_effective_tags(self) -> None:
        with self.assertRaises(ValueError):
            cflearn.MultiLoss(
                [
                    {"name": "mse", "tag": "duplicate"},
                    {"name": "mae", "tag": "duplicate"},
                ]
            )

    def test_evaluate_defines_empty_score_boundary(self) -> None:
        outputs = cflearn.InferenceOutputs({}, {}, None, None)
        inference = Mock()
        inference.get_outputs.return_value = outputs
        with self.assertRaisesRegex(ValueError, "losses or metrics"):
            cflearn.IModel.evaluate(
                Mock(),
                cflearn.TrainerConfig(),
                None,
                inference,
                Mock(),
            )

        outputs.loss_items = {}
        result = cflearn.IModel.evaluate(
            Mock(),
            cflearn.TrainerConfig(),
            None,
            inference,
            Mock(),
        )
        self.assertIs(result, outputs)
        self.assertIsNotNone(result.metric_outputs)
        self.assertEqual(result.metric_outputs.final_score, 0.0)

    def test_train_step_loss_normalization(self) -> None:
        primary = torch.tensor(2.0)
        auxiliary = torch.tensor(3.0)
        tensor_result = cflearn.normalize_loss_result(primary)
        self.assertIsNotNone(tensor_result)
        self.assertIs(tensor_result.loss, primary)
        self.assertDictEqual(
            dict(tensor_result.loss_tensors),
            {cflearn.LOSS_KEY: primary},
        )

        raw = {cflearn.LOSS_KEY: primary, "auxiliary": auxiliary}
        result = cflearn.normalize_loss_result(raw)

        self.assertIsNotNone(result)
        self.assertIs(result.loss, primary)
        self.assertDictEqual(dict(result.loss_tensors), raw)
        self.assertIsNot(result.loss_tensors, raw)
        loss_tensors: Any = result.loss_tensors
        with self.assertRaises(TypeError):
            loss_tensors["new"] = torch.tensor(4.0)
        self.assertDictEqual(raw, {cflearn.LOSS_KEY: primary, "auxiliary": auxiliary})
        self.assertIsNone(cflearn.normalize_loss_result(None))
        with self.assertRaisesRegex(ValueError, cflearn.LOSS_KEY):
            cflearn.normalize_loss_result({"auxiliary": auxiliary})
        invalid: Any = {cflearn.LOSS_KEY: primary, "auxiliary": 1.0}
        with self.assertRaisesRegex(TypeError, "auxiliary"):
            cflearn.normalize_loss_result(invalid)

    def test_multi_loss_inactive_and_collision_boundaries(self) -> None:
        @cflearn.register_loss("contract_inactive", allow_duplicate=True)
        class InactiveLoss(cflearn.ILoss):
            def forward(self, forward_results, batch, state=None) -> None:
                return None

        inactive = cflearn.MultiLoss([{"name": "contract_inactive"}])
        self.assertIsNone(inactive({}, {}))
        with self.assertRaisesRegex(ValueError, "should not be None"):
            cflearn.CommonTrainStep(inactive).loss_fn(Mock(), None, {}, {})

        @cflearn.register_loss("contract_collision_a", allow_duplicate=True)
        class CollisionA(cflearn.ILoss):
            def forward(self, forward_results, batch, state=None):
                return {cflearn.LOSS_KEY: torch.tensor(1.0), "b_c": torch.tensor(2.0)}

        @cflearn.register_loss("contract_collision_b", allow_duplicate=True)
        class CollisionB(cflearn.ILoss):
            def forward(self, forward_results, batch, state=None):
                return {cflearn.LOSS_KEY: torch.tensor(3.0), "c": torch.tensor(4.0)}

        collision = cflearn.MultiLoss(
            [
                {"name": "contract_collision_a", "tag": "a"},
                {"name": "contract_collision_b", "tag": "a_b"},
            ]
        )
        with self.assertRaisesRegex(ValueError, "a_b_c"):
            collision({}, {})
        with self.assertRaisesRegex(ValueError, "reserved"):
            cflearn.MultiLoss([{"name": "mse", "tag": cflearn.LOSS_KEY}])

    def test_evaluate_rejects_loss_metric_key_collisions(self) -> None:
        outputs = cflearn.InferenceOutputs(
            {},
            {},
            cflearn.MetricsOutputs(1.0, {"shared": 1.0}, {"shared": True}),
            {"shared": 2.0},
        )
        inference = Mock()
        inference.get_outputs.return_value = outputs
        with self.assertRaisesRegex(ValueError, "shared"):
            cflearn.IModel.evaluate(
                Mock(),
                cflearn.TrainerConfig(),
                None,
                inference,
                Mock(),
            )


if __name__ == "__main__":
    unittest.main()
