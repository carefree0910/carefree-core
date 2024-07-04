import unittest
import core.flow as cflow

from typing import Any
from core.parameters import OPT
from fastapi.testclient import TestClient
from core.flow.core import WORKFLOW_ENDPOINT_NAME
from core.toolkit.constants import WEB_ERR_CODE


class TestServer(unittest.TestCase):
    def setUp(self):
        @cflow.Node.register("null_node", allow_duplicate=True)
        class _(cflow.Node):
            def execute(self) -> Any:
                pass

        @cflow.Node.register("test_server_node", allow_duplicate=True)
        class _(cflow.Node):
            @classmethod
            def get_schema(cls) -> cflow.Schema:
                return cflow.Schema(input_names=["raise"], output_names=["foo"])

            async def execute(self) -> Any:
                if self.data["raise"]:
                    raise ValueError("test_server_node")
                return {"foo": "bar"}

        OPT.flow_opt.focus = "common|null_node|test_server_node"
        cflow.server.api.initialize()
        self.client = TestClient(cflow.server.api.app)

    def test_parsers(self):
        @cflow.Node.register("foo", allow_duplicate=True)
        class FooNode(cflow.Node):
            @classmethod
            def get_schema(cls) -> cflow.Schema:
                return cflow.Schema(input_names=["foo"], output_names=["bar"])

            def execute(self) -> Any:
                pass

        im = cflow.parse_input_model(FooNode)
        om = cflow.parse_output_model(FooNode)
        self.assertListEqual(list(im.model_fields), ["foo"])
        self.assertListEqual(list(om.model_fields), ["bar"])

    def test_server_status(self):
        response = self.client.get("/server_status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["num_nodes"], len(cflow.use_all_t_nodes()))

    def test_post_node(self):
        response = self.client.post("/test_server_node", json={"raise": False})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"foo": "bar"})

        response = self.client.post("/test_server_node", json={"raise": True})
        self.assertEqual(response.status_code, WEB_ERR_CODE)

    def test_post_workflow(self):
        d = dict(
            target="r",
            nodes=[{"key": "r", "type": "test_server_node", "data": {"raise": False}}],
        )
        response = self.client.post(WORKFLOW_ENDPOINT_NAME, json=d)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["r"], {"foo": "bar"})

        d["nodes"][0]["data"]["raise"] = True
        response = self.client.post(WORKFLOW_ENDPOINT_NAME, json=d)
        self.assertEqual(response.status_code, WEB_ERR_CODE)


if __name__ == "__main__":
    unittest.main()
