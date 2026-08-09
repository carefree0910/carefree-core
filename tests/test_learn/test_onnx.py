import pytest
import unittest

import numpy as np
import core.learn as cflearn

from typing import Optional
from pathlib import Path
from unittest.mock import patch
from core.learn.schema import DataLoader
from core.toolkit.types import np_dict_type


class TestONNX(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _prepare_tmp_path(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path

    def test_onnx(self) -> None:
        input_dim = 11
        output_dim = 7
        num_samples = 123
        batch_size = 17

        x = np.random.randn(num_samples, input_dim).astype(np.float32)
        y = np.random.randn(num_samples, output_dim).astype(np.float32)
        data = cflearn.ArrayData.init().fit(x, y)
        data.config.batch_size = batch_size
        loader = data.build_loader(x, y)

        config = cflearn.Config(
            module_name="fcnn",
            module_config=dict(input_dim=input_dim, output_dim=output_dim),
            loss_name="mse",
        )
        model = cflearn.IModel.from_config(config)
        model_inference = cflearn.Inference(model=model)
        model_outputs = model_inference.get_outputs(loader).forward_results

        onnx_file = self.tmp_path / "test.onnx"
        model.to_onnx(str(onnx_file), loader.get_input_sample())
        model.to_onnx(
            str(onnx_file),
            loader.get_input_sample(),
            dynamic_axes=[0],
            simplify=False,
            forward_fn=lambda d: model.onnx_forward(d)[cflearn.PREDICTIONS_KEY],
            num_samples=1,
        )

        @cflearn.IMetric.register("foo", allow_duplicate=True)
        class FooMetric(cflearn.IMetric):
            @property
            def is_positive(self) -> bool:
                return True

            @property
            def requires_all(self) -> bool:
                return True

            def forward(
                self,
                np_batch: np_dict_type,
                np_outputs: np_dict_type,
                loader: Optional[DataLoader] = None,
            ) -> float:
                return 0.12

        onnx_inference = cflearn.Inference(onnx=str(onnx_file))
        onnx_inference.get_outputs(loader, metrics=FooMetric(), return_labels=True)
        num_injections = 0

        def inject_outputs(_, __) -> None:
            nonlocal num_injections
            num_injections += 1

        onnx_outputs = onnx_inference.get_outputs(
            loader,
            inject_outputs_fn=inject_outputs,
        ).forward_results
        self.assertEqual(num_injections, len(loader))

        for k in model_outputs:
            mk_out = model_outputs[k]
            ok_out = onnx_outputs[k]
            np.testing.assert_array_almost_equal(mk_out, ok_out)

        empty_loader = data.build_loader(x[:0], y[:0])
        self.assertIsNotNone(onnx_inference.onnx)
        with patch.object(
            onnx_inference.onnx,
            "predict",
            wraps=onnx_inference.onnx.predict,
        ) as predict:
            empty_outputs = onnx_inference.get_outputs(empty_loader)
        predict.assert_not_called()
        self.assertDictEqual(empty_outputs.forward_results, {})
        self.assertDictEqual(empty_outputs.labels, {})
        self.assertIsNone(empty_outputs.metric_outputs)
        self.assertIsNone(empty_outputs.loss_items)


if __name__ == "__main__":
    unittest.main()
