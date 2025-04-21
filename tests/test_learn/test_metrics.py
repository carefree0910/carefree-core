import torch
import unittest

import core.learn as cflearn


class TestMetrics(unittest.TestCase):
    def test_metrics(self) -> None:
        def to_tensor(key: str) -> torch.Tensor:
            return torch.tensor(outputs.metric_values[key])

        x = torch.randn(11, 1)
        y = torch.randn(11, 1)
        metric = cflearn.IMetric.fuse(["mae", "mse", "corr", "stream_mse"])
        with self.assertRaises(NotImplementedError):
            metric.is_positive
        with self.assertRaises(NotImplementedError):
            metric.forward(None, None)
        batch = {cflearn.LABEL_KEY: y.numpy()}
        predictions = {cflearn.PREDICTIONS_KEY: x.numpy()}
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
            {cflearn.LABEL_KEY: y.numpy()},
            {cflearn.PREDICTIONS_KEY: x.numpy()},
        )
        torch.testing.assert_close(torch.tensor(outputs.final_score), -gt_mae)
        metric = cflearn.IMetric.fuse(
            ["mae", "mse", "corr"],
            metric_weights=dict(mae=0.1, mse=0.2, corr=0.7),
        )
        outputs = metric.evaluate(
            {cflearn.LABEL_KEY: y.numpy()},
            {cflearn.PREDICTIONS_KEY: x.numpy()},
        )
        torch.testing.assert_close(
            torch.tensor(outputs.final_score),
            -gt_mae * 0.1 - gt_mse * 0.2 + gt_corr * 0.7,
        )
        # empty
        metric = cflearn.IMetric.fuse([])
        outputs = metric.evaluate(None, None)
        self.assertEqual(outputs.final_score, 0.0)
        self.assertDictEqual(outputs.metric_values, {})


if __name__ == "__main__":
    unittest.main()
