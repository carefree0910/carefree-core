import asyncio
import unittest

from core.flow import *


class TestExample(unittest.TestCase):
    def test_cleanup(self):
        @Node.register("foo")
        class FooNode(Node):
            async def initialize(self, flow: Flow) -> None:
                self.shared_pool["foo"] = 123

            async def cleanup(self) -> None:
                self.shared_pool.pop("foo")

            async def execute(self) -> None:
                ut.assertIn("foo", self.shared_pool)
                ut.assertEqual(self.shared_pool["foo"], 123)
                raise ValueError("foo")

        ut = self
        flow = Flow().push(FooNode("foo"))
        with self.assertRaises(ValueError):
            asyncio.run(flow.execute("foo"))
        self.assertNotIn("foo", flow.shared_pool)


if __name__ == "__main__":
    unittest.main()
