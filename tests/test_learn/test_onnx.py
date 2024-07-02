import unittest

import numpy as np
import core.learn as cflearn


class TestONNX(unittest.TestCase):
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

        onnx_file = "test.onnx"
        model.to_onnx(onnx_file, loader.get_input_sample())
        model.to_onnx(
            onnx_file,
            loader.get_input_sample(),
            dynamic_axes=[0],
            simplify=False,
            forward_fn=lambda d: model.onnx_forward(d)[cflearn.PREDICTIONS_KEY],
            num_samples=1,
        )
        onnx_inference = cflearn.Inference(onnx=onnx_file)
        onnx_outputs = onnx_inference.get_outputs(loader).forward_results

        for k in model_outputs:
            mk_out = model_outputs[k]
            ok_out = onnx_outputs[k]
            np.testing.assert_array_almost_equal(mk_out, ok_out)


if __name__ == "__main__":
    unittest.main()
