import json
import asyncio
import tempfile
import unittest

import numpy as np

from PIL import Image
from core import flow as cflow
from aiohttp import ClientSession
from pathlib import Path
from pydantic import BaseModel
from unittest.mock import patch
from unittest.mock import AsyncMock
from core.flow.nodes import *
from core.flow.nodes.common import pad_parent
from core.flow.nodes.schema import HTTP_SESSION_KEY


class TestNodes(unittest.TestCase):
    def test_image_field(self):
        class TestImageModel(BaseModel):
            image: TImage

        TestImageModel(image="https://example.com/image.png")
        TestImageModel(image=Image.new("RGB", (10, 10), color="red"))
        with self.assertRaises(ValueError):
            TestImageModel(image=10)

    def test_http_image_node(self):
        async def get_session():
            return ClientSession()

        class TestHttpImageNode(IWithImageNode):
            async def execute(self):
                pass

        node = TestHttpImageNode()
        with self.assertRaises(ValueError):
            asyncio.run(node.fetch_image("foo", 10))
        session = asyncio.run(get_session())
        node.shared_pool[HTTP_SESSION_KEY] = session
        with patch("core.flow.nodes.schema.download_raw_with_retry") as mock:
            asyncio.run(node.download_raw("foo"))
            mock.assert_called_once()
        node.shared_pool[HTTP_SESSION_KEY] = None
        with self.assertRaises(ValueError):
            node.http_session
        node.shared_pool[HTTP_SESSION_KEY] = "foo"
        with self.assertRaises(TypeError):
            node.http_session
        with self.assertRaises(TypeError):
            asyncio.run(node.cleanup())
        asyncio.run(session.close())

    def test_nodes(self):
        with self.assertRaises(ValueError):
            node = LoopNode(
                data=dict(
                    base_node="common.echo",
                    base_data={},
                    loop_values={"foo": [1, 2], "bar": [3, 4, 5]},
                    loop_back_injections=None,
                )
            )
            asyncio.run(node.execute())
        d = {"foo": 10, "bar": 20}
        node = GatherNode("test")
        self.assertDictEqual(asyncio.run(node.get_api_response(d)), d)
        node.flow = {}
        with self.assertRaises(ValueError):
            asyncio.run(node.get_api_response(d))
        node.injections = [cflow.Injection("foo", "a", "b")]
        info = node.to_info()
        with self.assertRaises(ValueError):
            node.from_info(info)
        node.injections[0].src_hierarchy = None
        info = node.to_info()
        with self.assertRaises(ValueError):
            node.from_info(info)
        node.injections[0].dst_hierarchy = "foo"
        info = node.to_info()
        node.from_info(info)
        flow = cflow.Flow().push(EchoNode("echo", dict(messages="hello")))
        flow_node = WorkflowNode("test")
        flow_node.data = flow.to_model(target="echo").model_dump()
        asyncio.run(flow_node.execute())
        self.assertEqual(str(pad_parent("foo", None)), "foo")
        self.assertEqual(str(pad_parent("foo", "bar")), str(Path("bar") / "foo"))


class TestIWithWebsocketNode(unittest.IsolatedAsyncioTestCase):
    async def test_connect(self):
        class TestWebsocketNode(IWithWebsocketNode):
            async def execute(self):
                pass

        class MockWebsocket:
            def __init__(self, seq):
                self.iter = iter(seq)

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return next(self.iter)
                except StopIteration:
                    raise StopAsyncIteration

            async def send(self, data):
                case.assertEqual(data, json.dumps(d))

        async def handler(raw_message):
            return True

        url = "ws://localhost:8080"
        d = {"message": "Hello, World!"}
        headers = {"Authorization": "Bearer token"}

        ms = AsyncMock()
        ms.send = AsyncMock()
        ms.__aenter__.return_value = MockWebsocket(range(3))
        ms.__aexit__.return_value = None

        with patch("core.flow.nodes.schema.websockets.connect", return_value=ms):
            case = self
            node = TestWebsocketNode()
            await node.connect(url, handler=handler, send_data=d, headers=headers)


class TestCopyNode(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.src_file_path = Path(self.temp_dir.name) / "src.txt"
        self.dst_file_path = Path(self.temp_dir.name) / "dst.txt"
        with open(self.src_file_path, "w") as f:
            f.write("Hello, World!")

    def tearDown(self):
        self.temp_dir.cleanup()

    async def test_execute(self):
        node = CopyNode()
        node.data = {
            "src": str(self.src_file_path),
            "dst": str(self.dst_file_path),
            "parent": None,
        }
        result = await node.execute()
        self.assertEqual(result, {"dst": str(self.dst_file_path)})
        self.assertTrue(self.dst_file_path.exists())
        with open(self.dst_file_path, "r") as f:
            self.assertEqual(f.read(), "Hello, World!")

    async def test_execute_with_parent(self):
        parent_dir = Path(self.temp_dir.name) / "parent"
        node = CopyNode()
        node.data = {
            "src": str(self.src_file_path),
            "dst": "dst.txt",
            "parent": str(parent_dir),
        }
        result = await node.execute()
        expected_dst_path = pad_parent("dst.txt", str(parent_dir))
        self.assertEqual(result, {"dst": str(expected_dst_path)})
        self.assertTrue(expected_dst_path.exists())
        with open(expected_dst_path, "r") as f:
            self.assertEqual(f.read(), "Hello, World!")

    async def test_execute_with_nonexistent_src(self):
        node = CopyNode()
        node.data = {
            "src": "nonexistent.txt",
            "dst": str(self.dst_file_path),
            "parent": None,
        }
        with self.assertRaises(FileNotFoundError):
            await node.execute()


class TestSaveJsonNode(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.json_file_path = Path(self.temp_dir.name) / "data.json"
        self.data = {"message": "Hello, World!"}

    def tearDown(self):
        self.temp_dir.cleanup()

    async def test_execute(self):
        node = SaveJsonNode()
        node.data = {
            "path": str(self.json_file_path),
            "data": self.data,
            "parent": None,
        }
        result = await node.execute()
        self.assertEqual(result, {"dst": str(self.json_file_path)})
        self.assertTrue(self.json_file_path.exists())
        with open(self.json_file_path, "r") as f:
            self.assertEqual(json.load(f), self.data)

    async def test_execute_with_parent(self):
        parent_dir = Path(self.temp_dir.name) / "parent"
        node = SaveJsonNode()
        node.data = {"path": "data.json", "data": self.data, "parent": str(parent_dir)}
        result = await node.execute()
        expected_path = pad_parent("data.json", str(parent_dir))
        self.assertEqual(result, {"dst": str(expected_path)})
        self.assertTrue(expected_path.exists())
        with open(expected_path, "r") as f:
            self.assertEqual(json.load(f), self.data)


class TestSaveImageNode(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.image_file_path = Path(self.temp_dir.name) / "image.png"
        self.image = Image.new("RGB", (60, 30), color="red")
        self.array = np.array(self.image)

    def tearDown(self):
        self.temp_dir.cleanup()

    async def test_execute(self):
        node = SaveImageNode()
        node.data = {
            "url": "https://example.com/image.png",
            "path": str(self.image_file_path),
            "parent": None,
        }
        with patch.object(
            node, "get_image_from", return_value=AsyncMock()
        ) as mock_get_image_from:
            mock_get_image_from.return_value = self.image
            result = await node.execute()
            self.assertEqual(result, {})
            self.assertTrue(self.image_file_path.exists())
            saved_image = Image.open(self.image_file_path)
            np.testing.assert_array_equal(np.array(saved_image), self.array)

    async def test_execute_with_parent(self):
        parent_dir = Path(self.temp_dir.name) / "parent"
        parent_dir.mkdir()
        node = SaveImageNode()
        node.data = {
            "url": "https://example.com/image.png",
            "path": "image.png",
            "parent": str(parent_dir),
        }
        with patch.object(
            node, "get_image_from", return_value=AsyncMock()
        ) as mock_get_image_from:
            mock_get_image_from.return_value = self.image
            result = await node.execute()
            expected_path = pad_parent("image.png", str(parent_dir))
            self.assertEqual(result, {})
            self.assertTrue(expected_path.exists())
            saved_image = Image.open(expected_path)
            np.testing.assert_array_equal(np.array(saved_image), self.array)


class TestSaveImagesNode(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.image_urls = [
            "https://example.com/image1.png",
            "https://example.com/image2.png",
        ]
        self.image_files = [
            Path(self.temp_dir.name) / f"image_{i}.png"
            for i in range(len(self.image_urls))
        ]
        self.images = [Image.new("RGB", (60, 30), color="red") for _ in self.image_urls]
        self.arrays = [np.array(image) for image in self.images]

    def tearDown(self):
        self.temp_dir.cleanup()

    async def test_execute(self):
        node = SaveImagesNode()
        node.data = {
            "urls": self.image_urls,
            "prefix": "image",
            "parent": self.temp_dir.name,
        }
        with patch.object(
            node, "fetch_image", new_callable=AsyncMock
        ) as mock_fetch_image:
            mock_fetch_image.side_effect = self.images
            result = await node.execute()
            self.assertEqual(result, {})
            for i, image_file in enumerate(self.image_files):
                self.assertTrue(image_file.exists())
                saved_image = Image.open(image_file)
                np.testing.assert_array_equal(np.array(saved_image), self.arrays[i])


if __name__ == "__main__":
    unittest.main()
