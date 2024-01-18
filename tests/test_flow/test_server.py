import unittest
import core.flow as cflow

from fastapi.testclient import TestClient


class TestServer(unittest.TestCase):
    def setUp(self):
        cflow.server.api.initialize()
        self.client = TestClient(cflow.server.api.app)

    def test_server_status(self):
        response = self.client.get("/server_status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["num_nodes"], len(cflow.use_all_t_nodes()))


if __name__ == "__main__":
    unittest.main()
