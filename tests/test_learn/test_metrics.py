import torch
import unittest

import core.learn as cflearn
import core.learn.schema as learn_schema


class TestMetrics(unittest.TestCase):
    def test_public_metric_contracts(self) -> None:
        self.assertIs(cflearn.IMetric, learn_schema.IMetric)
        self.assertIs(cflearn.IStreamMetric, learn_schema.IStreamMetric)
        self.assertIs(cflearn.MetricValues, learn_schema.MetricValues)
        self.assertIs(cflearn.MetricsOutputs, learn_schema.MetricsOutputs)
        self.assertIs(cflearn.MultipleMetrics, learn_schema.MultipleMetrics)

        metric_values = cflearn.MetricValues(
            {cflearn.SCORE_KEY: 1.25},
            {cflearn.SCORE_KEY: True},
        )
        self.assertTupleEqual(
            cflearn.MetricValues._fields,
            ("values", "is_positive"),
        )
        self.assertTupleEqual(
            tuple(metric_values),
            (
                {cflearn.SCORE_KEY: 1.25},
                {cflearn.SCORE_KEY: True},
            ),
        )

        metric_outputs = cflearn.MetricsOutputs(
            1.25,
            {"metric": 1.25},
            {"metric": True},
        )
        self.assertTupleEqual(
            cflearn.MetricsOutputs._fields,
            ("final_score", "metric_values", "is_positive"),
        )
        self.assertTupleEqual(
            tuple(metric_outputs),
            (
                1.25,
                {"metric": 1.25},
                {"metric": True},
            ),
        )

    def test_legacy_metric_returns(self) -> None:
        class LegacyFloatMetric(cflearn.IMetric):
            __identifier__ = "legacy_float"

            @property
            def is_positive(self) -> bool:
                return False

            def forward(
                self,
                tensor_batch: cflearn.tensor_dict_type,
                tensor_outputs: cflearn.tensor_dict_type,
                loader: None = None,
            ) -> float:
                return 2.5

        class LegacyValuesMetric(cflearn.IMetric):
            __identifier__ = "legacy_values"

            @property
            def is_positive(self) -> bool:
                return True

            def forward(
                self,
                tensor_batch: cflearn.tensor_dict_type,
                tensor_outputs: cflearn.tensor_dict_type,
                loader: None = None,
            ) -> cflearn.MetricValues:
                return cflearn.MetricValues(
                    {
                        cflearn.SCORE_KEY: 1.25,
                        "detail": 3.5,
                    },
                    {
                        cflearn.SCORE_KEY: True,
                        "detail": False,
                    },
                )

        class LegacyDiagnosticMetric(cflearn.IMetric):
            __identifier__ = "legacy_diagnostic"

            @property
            def is_positive(self) -> bool:
                return True

            @property
            def not_include_in_score(self) -> bool:
                return True

            def forward(
                self,
                tensor_batch: cflearn.tensor_dict_type,
                tensor_outputs: cflearn.tensor_dict_type,
                loader: None = None,
            ) -> float:
                return 9.0

        float_outputs = LegacyFloatMetric().evaluate({}, {})
        self.assertEqual(
            float_outputs,
            cflearn.MetricsOutputs(
                -2.5,
                {"legacy_float": 2.5},
                {"legacy_float": False},
            ),
        )

        values_outputs = LegacyValuesMetric().evaluate({}, {})
        self.assertEqual(
            values_outputs,
            cflearn.MetricsOutputs(
                1.25,
                {
                    "legacy_values": 1.25,
                    "legacy_values_detail": 3.5,
                },
                {
                    "legacy_values": True,
                    "legacy_values_detail": False,
                },
            ),
        )

        multiple_outputs = cflearn.MultipleMetrics(
            [
                LegacyFloatMetric(),
                LegacyDiagnosticMetric(),
            ]
        ).evaluate({}, {})
        self.assertAlmostEqual(multiple_outputs.final_score, -2.5)
        self.assertDictEqual(
            multiple_outputs.metric_values,
            {
                "legacy_float": 2.5,
                "legacy_diagnostic": 9.0,
            },
        )
        self.assertDictEqual(
            multiple_outputs.is_positive,
            {
                "legacy_float": False,
                "legacy_diagnostic": True,
            },
        )

    def test_legacy_stream_metric_lifecycle(self) -> None:
        class LegacyStreamMetric(cflearn.IStreamMetric):
            __identifier__ = "legacy_stream"

            def __init__(self) -> None:
                self.num_reset = 0
                self.num_update = 0
                self.total = 0.0
                self.count = 0

            @property
            def is_positive(self) -> bool:
                return True

            def reset(self) -> None:
                self.num_reset += 1
                self.num_update = 0
                self.total = 0.0
                self.count = 0

            def update(
                self,
                tensor_batch: cflearn.tensor_dict_type,
                tensor_outputs: cflearn.tensor_dict_type,
                loader: None = None,
            ) -> None:
                labels = tensor_batch[cflearn.LABEL_KEY]
                self.num_update += 1
                self.total += labels.sum().item()
                self.count += labels.numel()

            def finalize(self) -> float:
                return self.total / self.count

        metric = LegacyStreamMetric()
        metric.reset()
        metric.update(
            {cflearn.LABEL_KEY: torch.tensor([1.0, 3.0])},
            {},
        )
        metric.update(
            {cflearn.LABEL_KEY: torch.tensor([5.0])},
            {},
        )
        self.assertEqual(metric.num_reset, 1)
        self.assertEqual(metric.num_update, 2)
        self.assertEqual(metric.finalize(), 3.0)
        self.assertEqual(
            metric.report(metric.finalize()),
            cflearn.MetricsOutputs(
                3.0,
                {"legacy_stream": 3.0},
                {"legacy_stream": True},
            ),
        )

        metric.reset()
        self.assertEqual(metric.num_reset, 2)
        self.assertEqual(metric.num_update, 0)

    def test_metrics(self) -> None:
        def to_tensor(key: str) -> torch.Tensor:
            return torch.tensor(outputs.metric_values[key])

        x = torch.randn(11, 1)
        y = torch.randn(11, 1)
        metric = cflearn.IMetric.fuse(["mae", "mse", "corr", "stream_mse"])
        with self.assertRaises(RuntimeError):
            metric.is_positive
        with self.assertRaises(NotImplementedError):
            metric.forward(None, None)
        batch = {cflearn.LABEL_KEY: y}
        predictions = {cflearn.PREDICTIONS_KEY: x}
        outputs = metric.evaluate(batch, predictions)
        # mae
        gt_mae = torch.mean(torch.abs(x - y))
        torch.testing.assert_close(to_tensor("mae"), gt_mae)
        # mse
        gt_mse = torch.mean((x - y) ** 2)
        torch.testing.assert_close(to_tensor("mse"), gt_mse)
        # corr
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        std = torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2))
        gt_corr = torch.sum(vx * vy) / std
        torch.testing.assert_close(to_tensor("corr"), gt_corr)
        # score
        torch.testing.assert_close(
            torch.tensor(outputs.final_score),
            (-gt_mae - gt_mse + gt_corr) / 3.0,
        )
        # stream mse
        metric.reset()
        metric.update(batch, predictions)
        outputs = metric.finalize()
        torch.testing.assert_close(to_tensor("stream_mse"), gt_mse)
        # weighted score
        metric = cflearn.IMetric.fuse("mae", metric_weights=dict(mae=0.123))
        outputs = metric.evaluate(
            {cflearn.LABEL_KEY: y},
            {cflearn.PREDICTIONS_KEY: x},
        )
        torch.testing.assert_close(torch.tensor(outputs.final_score), -gt_mae)
        metric = cflearn.IMetric.fuse(
            ["mae", "mse", "corr"],
            metric_weights=dict(mae=0.1, mse=0.2, corr=0.7),
        )
        outputs = metric.evaluate(
            {cflearn.LABEL_KEY: y},
            {cflearn.PREDICTIONS_KEY: x},
        )
        torch.testing.assert_close(
            torch.tensor(outputs.final_score),
            -gt_mae * 0.1 - gt_mse * 0.2 + gt_corr * 0.7,
        )
        # empty
        metric = cflearn.IMetric.fuse([])
        outputs = metric.evaluate(None, None)
        self.assertIsNone(outputs)

        # custom

        @cflearn.IMetric.register("moment", allow_duplicate=True)
        class Moment(cflearn.IMetric):
            @property
            def is_positive(self) -> bool:
                return True

            def forward(
                self,
                tensor_batch: cflearn.tensor_dict_type,
                tensor_outputs: cflearn.tensor_dict_type,
                loader: None = None,
            ) -> cflearn.MetricValues:
                predictions = tensor_outputs[cflearn.PREDICTIONS_KEY]
                mean = torch.mean(predictions).item()
                std = torch.std(predictions).item()
                return cflearn.MetricValues(
                    values={"mean": mean, "std": std, cflearn.SCORE_KEY: mean - std},
                    is_positive={"mean": True, "std": False, cflearn.SCORE_KEY: True},
                )

        metric = Moment()
        outputs = metric.evaluate(
            {cflearn.LABEL_KEY: y},
            {cflearn.PREDICTIONS_KEY: x},
        )
        torch.testing.assert_close(to_tensor("moment_mean"), torch.mean(x))
        torch.testing.assert_close(to_tensor("moment_std"), torch.std(x))
        torch.testing.assert_close(to_tensor("moment"), torch.mean(x) - torch.std(x))

        metric = cflearn.IMetric.fuse(["moment", "mae"])
        outputs = metric.evaluate(
            {cflearn.LABEL_KEY: y},
            {cflearn.PREDICTIONS_KEY: x},
        )
        torch.testing.assert_close(to_tensor("moment_mean"), torch.mean(x))
        torch.testing.assert_close(to_tensor("moment_std"), torch.std(x))
        torch.testing.assert_close(to_tensor("moment"), torch.mean(x) - torch.std(x))
        torch.testing.assert_close(to_tensor("mae"), gt_mae)


if __name__ == "__main__":
    unittest.main()
