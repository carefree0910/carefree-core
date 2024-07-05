import asyncio
import unittest

from typing import Any
from core.flow import *
from core.flow.core import WORKFLOW_ENDPOINT_NAME


class TestCore(unittest.TestCase):
    def test_injection(self):
        injection = Injection("foo", "bar", "baz")
        injeciton_model = injection.to_model()
        self.assertEqual(injeciton_model.src_key, "foo")
        self.assertEqual(injeciton_model.src_hierarchy, "bar")
        self.assertEqual(injeciton_model.dst_hierarchy, "baz")
        injection = LoopBackInjection("foo", "bar")
        injeciton_model = injection.to_model()
        self.assertEqual(injeciton_model.src_hierarchy, "foo")
        self.assertEqual(injeciton_model.dst_hierarchy, "bar")

    def test_node(self):
        @Node.register("foo", allow_duplicate=True, before_register=lambda _: None)
        class FooNode(Node):
            async def execute(self) -> Any:
                pass

        node = FooNode(injections=[Injection("foo", "bar", "baz")])
        with self.assertRaises(ValueError):
            node.to_item()
        with self.assertRaises(ValueError):
            node.to_model()
        info = node.to_info()
        with self.assertRaises(ValueError):
            node.from_info(info)
        with self.assertRaises(ValueError):
            node.fetch_injections({})
        d = {"foo": {"bar": "baz"}}
        node.fetch_injections(d)
        self.assertDictEqual(node.data, {"baz": "baz"})
        self.assertDictEqual(node.check_api_results(d), d)
        with self.assertRaises(RuntimeError):

            @Node.register(WORKFLOW_ENDPOINT_NAME)
            class _(Node):
                pass

    def test_flow(self):
        node = EchoNode("foo", lock_key="bar")
        flow = Flow().push(node).push(EchoNode("bar", lock_key="bar"))
        self.assertNotEqual(flow, "foo")
        with self.assertRaises(ValueError):
            asyncio.run(flow.run(node.to_item(), {}, {}, False, False, {}))
        with self.assertRaises(ValueError):
            gathered = flow.gather("foo", "bar")
            asyncio.run(flow.execute(gathered))
        asyncio.run(flow.run(node.to_item(), {}, {"foo": "bar"}, False, False, {}))


if __name__ == "__main__":
    unittest.main()
