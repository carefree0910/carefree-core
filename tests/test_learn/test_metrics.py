import torch
import unittest

import core.learn as cflearn


class TestMetrics(unittest.TestCase):
    def test_metrics(self) -> None:
        def to_tensor(key: str) -> torch.Tensor:
            return torch.tensor(outputs.metric_values[key])

        x = torch.randn(11, 1)
        y = torch.randn(11, 1)
        metric = cflearn.IMetric.fuse(["mae", "mse", "corr"])
        outputs = metric.evaluate(
            {cflearn.LABEL_KEY: y.numpy()},
            {cflearn.PREDICTIONS_KEY: x.numpy()},
        )
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


if __name__ == "__main__":
    unittest.main()
