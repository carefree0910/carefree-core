import torch
import unittest

from torch.nn import functional as F
from core.learn.modules.activations import *


class TestActivations(unittest.TestCase):
    def test_activations(self) -> None:
        x = torch.randn(3, 5, 7)
        torch.testing.assert_close(x, build_activation(None)(x))
        torch.testing.assert_close(
            F.leaky_relu(x, 0.123), build_activation("leaky_relu_0.123")(x)
        )
        torch.testing.assert_close(
            x * (torch.tanh(F.softplus(x))), build_activation("mish")(x)
        )
        build_activation("glu", in_dim=x.shape[-1])(x)
        build_activation("atanh")(x)
        build_activation("isoftplus")(x)
        build_activation("sign")(x)
        build_activation("sign", differentiable=False)(x)
        build_activation("sign", randomize_at_zero=True)(x)
        build_activation("one_hot")(x)
        build_activation("one_hot", differentiable=False)(x)
        build_activation("sine")(x)
        build_activation("h_swish")(x)
        build_activation("quick_gelu")(x)
        build_activation("geglu", in_dim=x.shape[-1], out_dim=11)(x)
        build_activation("diff_relu")(x)
        # utilized torch.nn.functional
        softmax = build_activation("softmax", dim=-1)
        self.assertEqual(str(softmax), "Lambda(softmax)")
        softmax(x)
        with self.assertRaises(TypeError):
            build_activation("softmax")(x)
        with self.assertRaises(NotImplementedError):
            build_activation("softmax_bla")(x)


if __name__ == "__main__":
    unittest.main()
