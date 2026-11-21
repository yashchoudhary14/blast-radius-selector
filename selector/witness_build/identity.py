"""Observe identity comparisons without invoking user-defined rich comparisons."""
import ast
import importlib.abc
import importlib.machinery
import sys
from pathlib import Path

OUTCOMES = {}
HOOK = "_blast_radius_pristine_identity"


def observe(key, is_not, left, right):
    identical = left is right
    if OUTCOMES.get(key, True):
        a, b = type(left), type(right)
        # object identity or equal values on the same built-in scalar implementation.
        if a is b and (a.__eq__ is str.__eq__ and a.__ne__ is str.__ne__ or
                       a.__eq__ is int.__eq__ and a.__ne__ is int.__ne__):
            safe = identical == (left == right)
        else:
            safe = all(c is type(None) or (c.__eq__ is object.__eq__ and c.__ne__ is object.__ne__)
                       for c in (a, b))
        OUTCOMES[key] = safe
    return not identical if is_not else identical


class Instrument(ast.NodeTransformer):
    def __init__(self, path):
        self.path = path

    def visit_Compare(self, node):
        self.generic_visit(node)
        if len(node.ops) == 1 and isinstance(node.ops[0], (ast.Is, ast.IsNot)):
            key = f"{self.path}:{node.lineno}:{node.col_offset}"
            call = ast.Call(ast.Name(HOOK, ast.Load()),
                            [ast.Constant(key), ast.Constant(isinstance(node.ops[0], ast.IsNot)),
                             node.left, node.comparators[0]], [])
            return ast.copy_location(call, node)
        return node


class Loader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        data = Path(self.path).read_bytes()
        tree = ast.parse(data, self.path)
        if HOOK.encode() in data:
            raise ValueError("collector binding collision")
        tree = Instrument(self.path.removeprefix("/sqlglot/")).visit(tree)
        return compile(ast.fix_missing_locations(tree), self.path, "exec", dont_inherit=True)

    def exec_module(self, module):
        module.__dict__[HOOK] = observe
        super().exec_module(module)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "sqlglot" or fullname.startswith("sqlglot."):
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec and isinstance(spec.loader, importlib.machinery.SourceFileLoader):
                spec.loader = Loader(fullname, spec.origin)
                return spec
