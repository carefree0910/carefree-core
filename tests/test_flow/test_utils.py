import unittest

from core.flow import *


class TestUtils(unittest.TestCase):
    def test_toposort(self):
        flow = (
            Flow()
            .push(ParametersNode("a", dict(params=dict(a=0))))
            .push(ParametersNode("b", {}, [Injection("a", "params.a", "params.b")]))
        )
        results = toposort(flow)
        self.assertSetEqual(results.in_edges["a"], {"b"})
        self.assertEqual(results.hierarchy[0][0].key, "a")
        self.assertEqual(results.hierarchy[1][0].key, "b")
        self.assertDictEqual(results.edge_labels, {("b", "a"): "params.b"})
        self.assertSetEqual(results.reachable, {"a", "b"})

    def test_dependency_path(self):
        flow = (
            Flow()
            .push(ParametersNode("a", dict(params=dict(a=0))))
            .push(ParametersNode("b", {}, [Injection("a", "params.a", "params.b")]))
            .push(ParametersNode("c", dict(params=dict(c=0))))
        )
        results = get_dependency_path(flow, "b")
        self.assertSetEqual(results.in_edges["a"], {"b"})
        self.assertEqual(results.hierarchy[0][0].key, "a")
        self.assertEqual(results.hierarchy[1][0].key, "b")
        self.assertDictEqual(results.edge_labels, {("b", "a"): "params.b"})
        self.assertSetEqual(results.reachable, {"a", "b"})

    def test_render_workflow(self):
        flow = (
            Flow()
            .push(ParametersNode("a", dict(params=dict(a=0))))
            .push(ParametersNode("b", {}, [Injection("a", "params.a", "params.b")]))
        )
        render_workflow(flow)

    def test_to_data_model(self):
        a = ParametersNode("a", dict(params=dict(a=0)))
        b = ParametersNode("b", {}, [Injection("a", "params.a", "params.b")])
        flow = Flow().push(a).push(b)
        data_model = to_data_model(flow, target="b")
        self.assertEqual(data_model.target, "b")
        self.assertEqual(data_model.nodes[0].key, "a")
        self.assertEqual(data_model.nodes[0].type, ParametersNode.__identifier__)
        self.assertDictEqual(data_model.nodes[0].data, a.data)
        self.assertEqual(data_model.nodes[1].key, "b")
        self.assertEqual(data_model.nodes[1].type, ParametersNode.__identifier__)
        self.assertDictEqual(data_model.nodes[1].data, b.data)


if __name__ == "__main__":
    unittest.main()
