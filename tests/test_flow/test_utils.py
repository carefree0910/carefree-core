import unittest

from core.flow import *


class TestUtils(unittest.TestCase):
    def test_toposort(self):
        flow = (
            Flow()
            .push(ParametersNode("a", dict(params=dict(a=0))))
            .push(ParametersNode("b", {}, [Injection("a", "params.a", "params.b")]))
            .push(
                ParametersNode(
                    "c",
                    {},
                    [
                        Injection("a", "params.a", "c0"),
                        Injection("a", "params.a", "c1"),
                    ],
                )
            )
        )
        results = toposort(flow)
        self.assertSetEqual(results.in_edges["a"], {"b", "c"})
        self.assertEqual(results.hierarchy[0][0].key, "a")
        self.assertSetEqual({n.key for n in results.hierarchy[1]}, {"b", "c"})
        self.assertDictEqual(
            results.edge_labels,
            {("b", "a"): "params.b", ("c", "a"): "c0, c1"},
        )
        self.assertSetEqual(results.reachable, {"a", "b", "c"})

        flow = (
            Flow()
            .push(ParametersNode("a", {}, [Injection("b", "params.b", "params.a")]))
            .push(ParametersNode("b", {}, [Injection("a", "params.a", "params.b")]))
        )
        with self.assertRaises(RuntimeError):
            toposort(flow)

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

    def test_render_workflow(self):
        flow = (
            Flow()
            .push(ParametersNode("a", dict(params=dict(a=0))))
            .push(ParametersNode("b", {}, [Injection("a", "params.a", "params.b")]))
        )
        render_workflow(flow)
        with self.assertRaises(ValueError):
            render_workflow(flow, layout="foo")

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
        c = ParametersNode("c", dict(params=dict(c=0)))
        flow = Flow().push(c)
        to_data_model(flow, target="foo")
        c.key = None
        with self.assertRaises(ValueError):
            to_data_model(flow, target="foo")


if __name__ == "__main__":
    unittest.main()
