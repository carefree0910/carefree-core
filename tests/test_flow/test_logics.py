import os
import asyncio
import tempfile
import unittest

from typing import Any
from typing import Dict
from typing import Optional
from pydantic import BaseModel
from core.flow import *


class TestLogics(unittest.TestCase):
    def test_cleanup(self):
        @Node.register("foo", allow_duplicate=True)
        class FooNode(Node):
            async def initialize(self, flow: Flow) -> None:
                self.shared_pool["foo"] = 123

            async def cleanup(self) -> None:
                self.shared_pool.pop("foo")

            async def execute(self) -> None:
                case.assertIn("foo", self.shared_pool)
                case.assertEqual(self.shared_pool["foo"], 123)
                raise ValueError("foo")

        case = self
        flow = Flow().push(FooNode("foo"))
        with self.assertRaises(ValueError):
            asyncio.run(flow.execute("foo"))
        self.assertNotIn("foo", flow.shared_pool)

    def test_complex_hierarchy(self):
        @Node.register("foo", allow_duplicate=True)
        class FooNode(Node):
            async def execute(self) -> dict:
                case.assertEqual(self.data["a"]["b"][0]["c"], v)
                case.assertEqual(self.data["d"]["e"]["f"]["g"], None)
                return dict(value=v)

        v = 123
        case = self
        flow = (
            Flow()
            .push(ParametersNode("p", dict(params=dict(h=dict(i=[None, dict(j=v)])))))
            .push(
                FooNode(
                    "foo",
                    dict(a=dict(b=[])),
                    injections=[
                        Injection("p", "params.h.i.1.j", "a.b.0.c"),
                        Injection("p", "params.h.i.0", "d.e.f.g"),
                    ],
                )
            )
        )
        results = asyncio.run(flow.execute("foo", verbose=True))
        self.assertEqual(results["foo"]["value"], v)

    def test_raises(self):
        p = dict(params=dict(a=0))

        # [error] use str to index src list
        flow = (
            Flow()
            .push(ParametersNode("p", dict(params=dict(a=[None], b=0))))
            .push(ParametersNode("p2", injections=[Injection("p", "params.a.b", "c")]))
        )
        with self.assertRaises(ValueError):
            asyncio.run(flow.execute("p2"))
        # [error] use str to index src number
        flow = (
            Flow()
            .push(ParametersNode("p", dict(params=dict(a=[None], b=0))))
            .push(ParametersNode("p2", injections=[Injection("p", "params.b.a", "c")]))
        )
        with self.assertRaises(ValueError):
            asyncio.run(flow.execute("p2"))
        # [normal] inject to different dst
        flow = (
            Flow()
            .push(ParametersNode("p", dict(params=dict(a=[None], b=0))))
            .push(
                ParametersNode(
                    "p2",
                    injections=[
                        Injection("p", "params.a.0", "c"),
                        Injection("p", "params.b", "d"),
                    ],
                )
            )
        )
        asyncio.run(flow.execute("p2"))
        # [error] inject to same dst
        flow = (
            Flow()
            .push(ParametersNode("p", dict(params=dict(a=[None], b=0))))
            .push(
                ParametersNode(
                    "p2",
                    injections=[
                        Injection("p", "params.a.0", "c"),
                        Injection("p", "params.b", "c"),
                    ],
                )
            )
        )
        with self.assertRaises(ValueError):
            asyncio.run(flow.execute("p2"))
        # [error] has not provided key
        with self.assertRaises(ValueError):
            flow = Flow().push(ParametersNode(data=p))
        # [error] key has '.' in it
        flow = Flow().push(ParametersNode("p.0", p))
        with self.assertRaises(ValueError):
            asyncio.run(flow.execute("p.0"))
        # [error] data is not dict
        flow = Flow().push(ParametersNode("p", 0))
        with self.assertRaises(ValueError):
            asyncio.run(flow.execute("p"))

        # [normal] params is following schema (ParametersModel)
        flow = Flow().push(ParametersNode("p", p))
        asyncio.run(flow.execute("p"))
        # [error] params is not following schema (ParametersModel)
        flow = Flow().push(ParametersNode("p", dict(params=0)))
        with self.assertRaises(ValueError):
            asyncio.run(flow.execute("p"))

        # [error] input is not following input_names

        @Node.register("foo", allow_duplicate=True)
        class FooNode(Node):
            @classmethod
            def get_schema(cls) -> Optional[Schema]:
                return Schema(input_names=["a"])

            async def execute(self) -> dict:
                return {}

        flow = Flow().push(FooNode("foo", dict(b=0)))
        with self.assertRaises(ValueError):
            asyncio.run(flow.execute("foo"))

        # [error] output is not following output_names

        @Node.register("foo", allow_duplicate=True)
        class FooNode(Node):
            @classmethod
            def get_schema(cls) -> Optional[Schema]:
                return Schema(output_names=["a"])

            async def execute(self) -> dict:
                return dict(b=0)

        flow = Flow().push(FooNode("foo"))
        with self.assertRaises(ValueError):
            asyncio.run(flow.execute("foo"))

        @Node.register("foo", allow_duplicate=True)
        class FooNode(Node):
            async def execute(self) -> dict:
                return dict(a=0)

        # [error] use str to index dst list
        flow = (
            Flow()
            .push(FooNode("p"))
            .push(FooNode("p2", dict(a=[]), [Injection("p", "a", "a.b")]))
        )
        with self.assertRaises(ValueError):
            asyncio.run(flow.execute("p2"))
        # [normal] use int to index dst list
        flow = (
            Flow()
            .push(FooNode("p"))
            .push(FooNode("p2", dict(a=[]), [Injection("p", "a", "a.0")]))
        )
        asyncio.run(flow.execute("p2"))
        # [normal] use int to index dst list, inject to 1 and 0 is provided
        flow = (
            Flow()
            .push(FooNode("p"))
            .push(FooNode("p2", dict(a=[0]), [Injection("p", "a", "a.0")]))
        )
        asyncio.run(flow.execute("p2"))
        # [error] use int to index dst list, inject to 1 but 0 is not provided
        flow = (
            Flow()
            .push(FooNode("p"))
            .push(FooNode("p2", dict(a=[]), [Injection("p", "a", "a.1")]))
        )
        with self.assertRaises(ValueError):
            asyncio.run(flow.execute("p2"))
        # [error] index non-dict
        flow = (
            Flow()
            .push(FooNode("p"))
            .push(FooNode("p2", dict(a=0), [Injection("p", "a", "a.b")]))
        )
        with self.assertRaises(ValueError):
            asyncio.run(flow.execute("p2"))
        # [error] cyclic dependency
        flow = (
            Flow()
            .push(FooNode("p", dict(a=0), [Injection("p2", "a", "a")]))
            .push(FooNode("p2", dict(a=0), [Injection("p", "a", "a")]))
        )
        with self.assertRaises(ValueError):
            asyncio.run(flow.execute("p2"))
        # [error] non-existing dependency
        flow = Flow().push(FooNode("p", dict(a=0), [Injection("p2", "a", "a")]))
        with self.assertRaises(ValueError):
            asyncio.run(flow.execute("p"))
        # [normal] existing target
        flow = Flow().push(FooNode("p", dict(a=0)))
        asyncio.run(flow.execute("p"))
        # [error] non-existing target
        with self.assertRaises(ValueError):
            asyncio.run(flow.execute("p2"))

        # [error] results is not dict
        @Node.register("foo", allow_duplicate=True)
        class FooNode(Node):
            async def execute(self) -> int:
                return 0

        flow = Flow().push(FooNode("foo"))
        with self.assertRaises(ValueError):
            asyncio.run(flow.execute("foo"))

        # [error] results not match output_model
        class IntOutput(BaseModel):
            a: int

        @Node.register("foo", allow_duplicate=True)
        class FooNode(Node):
            @classmethod
            def get_schema(cls) -> Optional[Schema]:
                return Schema(output_model=IntOutput)

            async def execute(self) -> dict:
                return dict(a="abc")

        flow = Flow().push(FooNode("foo"))
        with self.assertRaises(ValueError):
            asyncio.run(flow.execute("foo"))

        # [error] results not match output_names
        @Node.register("foo", allow_duplicate=True)
        class FooNode(Node):
            @classmethod
            def get_schema(cls) -> Optional[Schema]:
                return Schema(output_names=["b"])

            async def execute(self) -> dict:
                return dict(a=0)

        flow = Flow().push(FooNode("foo"))
        with self.assertRaises(ValueError):
            asyncio.run(flow.execute("foo"))

        # [error] results not match api_output_model
        @Node.register("foo", allow_duplicate=True)
        class FooNode(Node):
            @classmethod
            def get_schema(cls) -> Optional[Schema]:
                return Schema(api_output_model=IntOutput)

            async def execute(self) -> dict:
                return dict(a="abc")

            async def get_api_response(self, results: Dict[str, Any]) -> Any:
                return results

        flow = Flow().push(FooNode("foo"))
        with self.assertRaises(ValueError):
            asyncio.run(flow.execute("foo", return_api_response=True))

        # test mid-exception
        @Node.register("foo", allow_duplicate=True)
        class FooNode(Node):
            async def execute(self) -> dict:
                return dict(a=0)

        @Node.register("bar", allow_duplicate=True)
        class BarNode(Node):
            async def execute(self) -> None:
                raise ValueError("bar")

        flow = (
            Flow()
            .push(FooNode("foo"))
            .push(BarNode("bar", injections=[Injection("foo", "a", "a")]))
        )
        # [normal] set `return_if_exception` to True
        asyncio.run(flow.execute("bar", return_if_exception=True, verbose=True))
        results = asyncio.run(flow.execute("bar", return_if_exception=True))
        self.assertEqual(results["foo"]["a"], 0)
        self.assertIsInstance(results[EXCEPTION_MESSAGE_KEY], str)
        # [error] default settings
        with self.assertRaises(ValueError):
            asyncio.run(flow.execute("bar"))

        # [normal] exception at cleanup stage will not be raised / recorded
        @Node.register("foo", allow_duplicate=True)
        class FooNode(Node):
            async def execute(self) -> dict:
                return dict(a=0)

            async def cleanup(self) -> None:
                raise ValueError("foo")

        flow = Flow().push(FooNode("foo"))
        results = asyncio.run(flow.execute("foo"))
        self.assertEqual(results["foo"]["a"], 0)
        self.assertIsNone(results[EXCEPTION_MESSAGE_KEY])

    def test_download_images(self):
        node_0 = DownloadImageNode("0", dict(url="https://picsum.photos/240/320"))
        node_1 = DownloadImageNode("1", dict(url="https://picsum.photos/320/240"))
        flow = Flow().push(node_0).push(node_1)
        gathered = flow.gather("0", "1")
        results = asyncio.run(flow.execute(gathered))
        self.assertEqual(results[gathered]["0"]["image"].size, (240, 320))
        self.assertEqual(results[gathered]["1"]["image"].size, (320, 240))
        results = asyncio.run(flow.execute(gathered, return_api_response=True))
        self.assertIsInstance(results[gathered]["0"]["image"], str)
        self.assertIsInstance(results[gathered]["1"]["image"], str)
        flow = Flow().push(node_0)
        results = asyncio.run(flow.execute("0", return_api_response=True))
        self.assertIsInstance(results["0"]["image"], str)

    def test_str(self):
        flow_0 = Flow().push(ParametersNode("p", dict(params=dict(a=0))))
        flow_1 = Flow().push(ParametersNode("p", dict(params=dict(a=0))))
        self.assertEqual(str(flow_0), str(flow_1))
        flow_1.push(ParametersNode("p2", dict(params=dict(a=0))))
        self.assertNotEqual(str(flow_0), str(flow_1))

    def test_loop(self):
        @Node.register("add_one")
        class AddOneNode(Node):
            async def execute(self) -> dict:
                return dict(a=dict(b=self.data["a"]["b"] + 1))

        flow = Flow().push(ParametersNode("p", dict(params=dict(a=dict(b=0)))))
        num_loop = 7
        looped = flow.loop(
            AddOneNode("add_one", injections=[Injection("p", "params.a.b", "a.b")]),
            {"loop_idx": list(range(num_loop))},
            [LoopBackInjection("a.b", "a.b")],
            extract_hierarchy="a.b",
        )
        results = asyncio.run(flow.execute(looped))
        targets = list(range(1, num_loop + 1))
        self.assertSequenceEqual(results[looped]["results"], targets)

    def test_serialization(self):
        flow = Flow().push(ParametersNode("p", dict(params=dict(a=0))))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "flow.json")
            flow.dump(path)
            loaded = Flow.load(path)
            self.assertEqual(flow, loaded)

    def test_manual_depend(self):
        @Node.register("append", allow_duplicate=True)
        class _(Node):
            async def execute(self) -> dict:
                data.append(self.data["value"])
                return {}

        data = []
        append0 = Node.make("append", dict(key="append0", data=dict(value=0)))
        append1 = Node.make("append", dict(key="append1", data=dict(value=1)))
        flow = Flow().push(append1).push(append0)
        target = flow.gather("append0", "append1")
        asyncio.run(flow.execute(target))
        self.assertListEqual(data, [1, 0])
        data = []
        append1.depend_on("append0")
        asyncio.run(flow.execute("append1"))
        self.assertListEqual(data, [0, 1])


if __name__ == "__main__":
    TestLogics().test_loop()
