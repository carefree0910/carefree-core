import unittest


class TestTemplate(unittest.TestCase):
    def test_template(self) -> None:
        self.assertEqual(1 + 1, 2)


if __name__ == "__main__":
    unittest.main()
