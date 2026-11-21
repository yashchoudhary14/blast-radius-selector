import unittest

from .state import Graph, changed_paths
from selector.state_refs import signature


class StateTest(unittest.TestCase):
    def test_changed_nested_keys(self):
        graph = Graph([])
        a = graph.ref({"x": [-1, {"old": False}]})
        b = graph.ref({"x": [-2, {"new": False}]})
        self.assertCountEqual(changed_paths(graph.nodes, a, b),
                              [("x", "0"), ("x", "1", "old"), ("x", "1", "new")])

    def test_shared_equal_dict(self):
        def capture(shared):
            graph = Graph([])
            value = {"answer": 42}
            graph.scope("example", {"base": {"x": value},
                                    "child": {"x": value if shared else dict(value)}})
            return graph.roots, graph.aliases()

        a, b = capture(True), capture(False)
        self.assertEqual(a[0], b[0])
        self.assertNotEqual(a[1], b[1])

    def test_compiler_equivalence(self):
        def fn(body):
            tree = compile("def f():\n" + body, "sample.py", "exec")
            return signature(tree.co_consts[0])
        self.assertEqual(fn(" return -0"), fn("\n return 0"))
        self.assertEqual(fn(" 'old'\n return 3"), fn(" 'new'\n return 3"))
        self.assertNotEqual(fn(" 'old'\n return 'old'"), fn(" 'new'\n return 'new'"))
        self.assertNotEqual(fn(" return -1"), fn(" return -2"))
        self.assertNotEqual(fn(" return lambda: 1"), fn(" return lambda: 2"))
