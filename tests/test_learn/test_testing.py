import unittest

import core.learn as cflearn


class TestTesting(unittest.TestCase):
    def test_linear_data(self) -> None:
        cflearn.testing.linear_data(x_noise_scale=0.12, y_noise_scale=0.34)


if __name__ == "__main__":
    unittest.main()
