import torch
import unittest

import core.learn as cflearn
import core.learn.schema as learn_schema
from core.toolkit.array import to_labels


class TestMetrics(unittest.TestCase):
    def test_accuracy_tensor_boundary(self) -> None:
        cases = [
            (
                "binary",
                torch.tensor([[-2.0], [0.25], [1.0]], requires_grad=True),
                torch.tensor([[0], [0], [0]]),
            ),
            (
                "multiclass",
                torch.tensor(
                    [
                        [3.0, 1.0, 0.0],
                        [0.0, 2.0, 1.0],
                        [0.0, 1.0, 4.0],
                        [1.0, 3.0, 2.0],
                    ],
                    requires_grad=True,
                ),
                torch.tensor([[0], [2], [2], [0]]),
            ),
        ]
        if torch.cuda.is_available():
            _, logits, labels = cases[0]
            cases.append(
                (
                    "binary_cuda",
                    logits.detach().cuda().requires_grad_(),
                    labels.cuda(),
                )
            )
        metric = cflearn.Accuracy()
        for name, logits, labels in cases:
            numpy_predictions = to_labels(
                logits.detach().cpu().numpy(),
                metric.threshold,
            )
            expected = (numpy_predictions == labels.cpu().numpy()).mean().item()
            with self.subTest(name=name):
                actual = metric.forward(
                    {cflearn.LABEL_KEY: labels},
                    {cflearn.PREDICTIONS_KEY: logits},
                )
                self.assertIsInstance(actual, float)
                self.assertEqual(actual, expected)

    def test_public_metric_contracts(self) -> None:
        self.assertIs(cflearn.IMetric, learn_schema.IMetric)
        self.assertIs(cflearn.IStreamMetric, learn_schema.IStreamMetric)
        self.assertIs(cflearn.MetricValues, learn_schema.MetricValues)
        self.assertIs(cflearn.MetricResult, learn_schema.MetricResult)
        self.assertIs(cflearn.MetricsOutputs, learn_schema.MetricsOutputs)
        self.assertIs(cflearn.MetricAccumulator, learn_schema.MetricAccumulator)
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

        metric_result = cflearn.MetricResult(metric_outputs)
        self.assertIs(metric_result.value, metric_outputs)
        self.assertEqual(metric_result.sample_count, 1.0)

    def test_metric_accumulator(self) -> None:
        accumulator = cflearn.MetricAccumulator()
        self.assertIsNone(accumulator.finalize())

        first = cflearn.MetricsOutputs(
            1.0,
            {
                "metric": 1.0,
                "detail": 2.0,
            },
            {
                "metric": True,
                "detail": False,
            },
        )
        second = cflearn.MetricsOutputs(
            9.0,
            {
                "metric": 9.0,
                "detail": 6.0,
            },
            {
                "metric": True,
                "detail": False,
            },
        )
        accumulator.add(cflearn.MetricResult(first, sample_count=2.0))
        other = cflearn.MetricAccumulator()
        other.add(cflearn.MetricResult(second))
        accumulator.merge(other)
        reduced = accumulator.finalize()
        self.assertIsNotNone(reduced)
        self.assertAlmostEqual(reduced.final_score, 11.0 / 3.0)
        self.assertDictEqual(
            reduced.metric_values,
            {
                "metric": 11.0 / 3.0,
                "detail": 10.0 / 3.0,
            },
        )
        self.assertDictEqual(
            reduced.is_positive,
            {
                "metric": True,
                "detail": False,
            },
        )

        conflicting = cflearn.MetricsOutputs(
            1.0,
            {"metric": 1.0},
            {"metric": False},
        )
        conflicting_accumulator = cflearn.MetricAccumulator()
        conflicting_accumulator.add(cflearn.MetricResult(first))
        with self.assertRaisesRegex(ValueError, "metric"):
            conflicting_accumulator.add(cflearn.MetricResult(conflicting))
        conflicting_other = cflearn.MetricAccumulator()
        conflicting_other.add(cflearn.MetricResult(conflicting))
        with self.assertRaisesRegex(ValueError, "metric"):
            conflicting_accumulator.merge(conflicting_other)

        invalid = cflearn.MetricsOutputs(
            1.0,
            {"metric": 1.0},
            {},
        )
        with self.assertRaisesRegex(ValueError, "identical keys"):
            cflearn.MetricAccumulator().add(cflearn.MetricResult(invalid))

    def test_metrics_outputs_reduce(self) -> None:
        outputs = [
            cflearn.MetricsOutputs(
                1.0,
                {"metric": 1.0},
                {"metric": True},
            ),
            cflearn.MetricsOutputs(
                9.0,
                {"metric": 9.0},
                {"metric": True},
            ),
        ]

        equally_reduced = cflearn.MetricsOutputs.reduce(outputs)
        self.assertEqual(equally_reduced.final_score, 5.0)
        self.assertDictEqual(equally_reduced.metric_values, {"metric": 5.0})

        sample_reduced = cflearn.MetricsOutputs.reduce(
            outputs,
            sample_counts=[2.0, 1.0],
        )
        self.assertAlmostEqual(sample_reduced.final_score, 11.0 / 3.0)
        self.assertAlmostEqual(
            sample_reduced.metric_values["metric"],
            11.0 / 3.0,
        )

        with self.assertRaises(ValueError):
            cflearn.MetricsOutputs.reduce([])
        with self.assertRaises(ValueError):
            cflearn.MetricsOutputs.reduce(outputs, sample_counts=[1.0])
        for sample_count in [0.0, -1.0, float("nan"), float("inf")]:
            with self.subTest(sample_count=sample_count):
                with self.assertRaises(ValueError):
                    cflearn.MetricsOutputs.reduce(
                        outputs,
                        sample_counts=[1.0, sample_count],
                    )

        conflicting_outputs = [
            outputs[0],
            cflearn.MetricsOutputs(
                9.0,
                {"metric": 9.0},
                {"metric": False},
            ),
        ]
        with self.assertRaisesRegex(ValueError, "metric"):
            cflearn.MetricsOutputs.reduce(conflicting_outputs)

    def test_multiple_metric_weights_and_diagnostics(self) -> None:
        class WeightedMetric(cflearn.IMetric):
            __identifier__ = "weighted"

            def __init__(self, value: float) -> None:
                self.value = value

            @property
            def is_positive(self) -> bool:
                return True

            def forward(
                self,
                tensor_batch: cflearn.tensor_dict_type,
                tensor_outputs: cflearn.tensor_dict_type,
                loader: None = None,
            ) -> float:
                return self.value

        class DiagnosticMetric(cflearn.IMetric):
            __identifier__ = "diagnostic"

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
                return 7.0

        weighted = cflearn.MultipleMetrics(
            [
                WeightedMetric(2.0),
                WeightedMetric(10.0),
            ],
            weights={
                "weighted": 1.0,
                "weighted_1": 3.0,
            },
        ).evaluate({}, {})
        self.assertIsNotNone(weighted)
        self.assertEqual(weighted.final_score, 8.0)
        self.assertDictEqual(
            weighted.metric_values,
            {
                "weighted": 2.0,
                "weighted_1": 10.0,
            },
        )
        self.assertDictEqual(
            weighted.is_positive,
            {
                "weighted": True,
                "weighted_1": True,
            },
        )

        fallback = cflearn.MultipleMetrics(
            [
                WeightedMetric(2.0),
                WeightedMetric(10.0),
            ],
            weights={"weighted": 3.0},
        ).evaluate({}, {})
        self.assertEqual(fallback.final_score, 6.0)

        zero_weight = cflearn.MultipleMetrics(
            [WeightedMetric(2.0)],
            weights={"weighted": 0.0},
        ).evaluate({}, {})
        self.assertIsNotNone(zero_weight)
        self.assertEqual(zero_weight.final_score, 0.0)
        self.assertDictEqual(zero_weight.metric_values, {"weighted": 2.0})

        diagnostic = cflearn.MultipleMetrics([DiagnosticMetric()]).evaluate({}, {})
        self.assertIsNotNone(diagnostic)
        self.assertEqual(diagnostic.final_score, 0.0)
        self.assertDictEqual(
            diagnostic.metric_values,
            {"diagnostic": 7.0},
        )

        for weight in ["invalid", -1.0, float("nan"), float("inf")]:
            with self.subTest(weight=weight):
                with self.assertRaises(ValueError):
                    cflearn.MultipleMetrics(
                        [WeightedMetric(2.0)],
                        weights={"weighted": weight},
                    ).evaluate({}, {})

    def test_multiple_metric_key_collisions(self) -> None:
        class RepeatedMetric(cflearn.IMetric):
            __identifier__ = "repeated"

            def __init__(self, value: float) -> None:
                self.value = value

            @property
            def is_positive(self) -> bool:
                return True

            def forward(
                self,
                tensor_batch: cflearn.tensor_dict_type,
                tensor_outputs: cflearn.tensor_dict_type,
                loader: None = None,
            ) -> float:
                return self.value

        class SuffixedMetric(RepeatedMetric):
            __identifier__ = "repeated_1"

        collision_safe = cflearn.MultipleMetrics(
            [
                RepeatedMetric(1.0),
                RepeatedMetric(2.0),
                SuffixedMetric(3.0),
            ]
        ).evaluate({}, {})
        self.assertIsNotNone(collision_safe)
        self.assertEqual(len(collision_safe.metric_values), 3)
        self.assertEqual(
            sorted(collision_safe.metric_values.values()),
            [1.0, 2.0, 3.0],
        )
        self.assertIn("repeated", collision_safe.metric_values)
        self.assertIn("repeated_1", collision_safe.metric_values)
        self.assertIn("repeated_2", collision_safe.metric_values)

        class DetailMetric(cflearn.IMetric):
            __identifier__ = "collision"

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
                        cflearn.SCORE_KEY: 1.0,
                        "detail": 2.0,
                    },
                    {
                        cflearn.SCORE_KEY: True,
                        "detail": True,
                    },
                )

        class CollidingMetric(cflearn.IMetric):
            __identifier__ = "collision_detail"

            @property
            def is_positive(self) -> bool:
                return True

            def forward(
                self,
                tensor_batch: cflearn.tensor_dict_type,
                tensor_outputs: cflearn.tensor_dict_type,
                loader: None = None,
            ) -> float:
                return 3.0

        with self.assertRaisesRegex(ValueError, "collision_detail"):
            cflearn.MultipleMetrics(
                [
                    DetailMetric(),
                    CollidingMetric(),
                ]
            ).evaluate({}, {})

        left = cflearn.MetricsOutputs(
            1.0,
            {"collision": 1.0},
            {"collision": True},
        )
        right = cflearn.MetricsOutputs(
            2.0,
            {"collision": 2.0},
            {"collision": True},
        )
        with self.assertRaisesRegex(ValueError, "collision"):
            left.union(right)

        weighted_left = cflearn.MetricsOutputs(
            2.0,
            {"left": 2.0},
            {"left": True},
        )
        weighted_right = cflearn.MetricsOutputs(
            10.0,
            {"right": 10.0},
            {"right": True},
        )
        weighted_union = weighted_left.union(
            weighted_right,
            weight=1.0,
            other_weight=3.0,
        )
        self.assertEqual(weighted_union.final_score, 8.0)
        zero_union = weighted_left.union(
            weighted_right,
            weight=0.0,
            other_weight=0.0,
        )
        self.assertEqual(zero_union.final_score, 0.0)

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
        self.assertIsNone(metric.get_distributed_state())
        with self.assertRaisesRegex(RuntimeError, "does not support"):
            metric.merge_distributed_states([None])

    def test_stream_mse_distributed_state(self) -> None:
        first = cflearn.StreamMSE()
        first.reset()
        first.update(
            {cflearn.LABEL_KEY: torch.zeros(2)},
            {cflearn.PREDICTIONS_KEY: torch.tensor([1.0, 3.0])},
        )
        second = cflearn.StreamMSE()
        second.reset()
        second.update(
            {cflearn.LABEL_KEY: torch.zeros(1)},
            {cflearn.PREDICTIONS_KEY: torch.tensor([2.0])},
        )

        metric = cflearn.StreamMSE()
        metric.merge_distributed_states(
            [
                first.get_distributed_state(),
                second.get_distributed_state(),
            ]
        )

        self.assertAlmostEqual(metric.finalize(), 14.0 / 3.0)

    def test_metrics(self) -> None:
        def to_tensor(key: str) -> torch.Tensor:
            return torch.tensor(outputs.metric_values[key])

        x = torch.randn(11, 1)
        y = torch.randn(11, 1)
        metric = cflearn.IMetric.fuse(["mae", "mse", "corr", "stream_mse"])
        self.assertTrue(metric.has_streaming)
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
