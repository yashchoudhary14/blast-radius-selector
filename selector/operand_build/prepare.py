"""Compile instrumented bytecode in a disposable br-lab container, keeping source intact."""
import ast
import importlib._bootstrap_external as bootstrap
import importlib.util
import json
from pathlib import Path

OPS = ("Eq", "NotEq", "Lt", "LtE", "Gt", "GtE", "Is", "IsNot", "In", "NotIn")


def call(name, node, *args):
    return ast.copy_location(ast.Call(ast.Name("_br_operand_" + name, ast.Load()), list(args), []), node)


class Rewrite(ast.NodeTransformer):
    def __init__(self, path, sites, tree):
        self.path = path
        self.sites = sites
        self.conditions = {node.test for node in ast.walk(tree)
                           if isinstance(node, (ast.If, ast.While, ast.IfExp, ast.Assert))}
        self.conditions.update(cond for n in ast.walk(tree) if isinstance(n, ast.comprehension)
                               for cond in n.ifs)

    def site(self, node):
        key = len(self.sites)
        self.sites.append((self.path, node.lineno, node.col_offset, type(node).__name__, node.end_col_offset))
        return ast.Constant(key)

    def visit_FunctionDef(self, node):
        if node.name.startswith("__") and node.name.endswith("__"):
            return node
        return self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def mark(self, key, node):
        # In a condition the compiler still branches directly on the original expression.
        return ast.copy_location(ast.BoolOp(ast.Or(), [call("mark", node, key), node]), node)

    def visit_Compare(self, node):
        key = self.site(node)
        self.generic_visit(node)
        if len(node.ops) != 1:
            return self.mark(key, node)
        left, right = node.left, node.comparators[0]
        side = 1 if isinstance(right, ast.Constant) else -1 if isinstance(left, ast.Constant) else 0
        return call("compare", node, key, ast.Constant(OPS.index(type(node.ops[0]).__name__)),
                    ast.Constant(side), left, right)

    def visit_BoolOp(self, node):
        key = self.site(node)
        first, second, *_ = node.values
        simple = isinstance(first, (ast.Name, ast.Attribute, ast.Compare, ast.Constant))
        self.generic_visit(node)
        if len(node.values) != 2 or not simple or not isinstance(second, (ast.Name, ast.Constant)):
            return self.mark(key, node)
        name = isinstance(second, ast.Name)
        arg = second.id if name else second.value
        node.values[0] = call("boolean", first, key, node.values[0], ast.Constant(name),
                              ast.Constant(arg), ast.Constant(node in self.conditions))
        return node


def prepare():
    sites = []
    root = Path("/sqlglot")
    for file in sorted((root / "sqlglot").rglob("*.py")):
        data = file.read_bytes()
        assert b"_br_operand_" not in data
        tree = ast.parse(data, str(file))
        Rewrite(file.relative_to(root).as_posix(), sites, tree).visit(tree)
        code = compile(ast.fix_missing_locations(tree), str(file), "exec", dont_inherit=True)
        cache = Path(importlib.util.cache_from_source(str(file)))
        cache.parent.mkdir(exist_ok=True)
        cache.write_bytes(bootstrap._code_to_timestamp_pyc(code, int(file.stat().st_mtime), len(data)))
    Path("/tmp/operand-sites.json").write_text(json.dumps(sites))
