import json
import unittest
import core.flow as cflow

from pathlib import Path
from fastapi.testclient import TestClient


file_folder = Path(__file__).parent
workflows_folder = file_folder.parent / "examples" / "workflows"


class TestServer(unittest.TestCase):
    def setUp(self):
        cflow.server.api.initialize()
        self.client = TestClient(cflow.server.api.app)

    def test_server_status(self):
        response = self.client.get("/server_status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["num_nodes"], len(cflow.use_all_t_nodes()))

    def test_workflows(self):
        for path in workflows_folder.glob("*.json"):
            with open(path, "r") as f:
                workflow = json.load(f)
            response = self.client.post("/workflow", json=workflow)
            self.assertEqual(response.status_code, 200)
            self.assertIsInstance(response.json(), dict)


if __name__ == "__main__":
    unittest.main()
