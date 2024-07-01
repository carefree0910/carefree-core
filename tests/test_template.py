import unittest

from core.parameters import OPT


class TestTemplate(unittest.TestCase):
    def test_template(self) -> None:
        self.assertEqual(1 + 1, 2)
        self.assertEqual(OPT.env_key, "CFCORE_ENV")


if __name__ == "__main__":
    unittest.main()
