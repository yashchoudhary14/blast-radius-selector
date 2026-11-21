"""Reject probe edits that could communicate with the omitted code via side effects."""
import ast
import importlib.abc
import importlib.util
import operator
import sys
from pathlib import Path


class Unknown(Exception):
    pass


def unchanged(a, b):
    if isinstance(a, list):
        return len(a) == len(b) and all(unchanged(x, y) for x, y in zip(a, b))
    return ast.dump(a) == ast.dump(b)


class Guard(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self):
        self.modules = {}
        self.clean = True

    def refuse(self, *args):
        self.clean = False
        raise Unknown("changed operation reached")

    def compare(self, before, after, a, b):
        # Never invoke a Python-defined rich comparison to check a second value.
        # Mixed types, reflected dispatch, custom results and overloaded operators
        # can perform work even when their eventual truth values agree.
        cls = type(a)
        if type(b) is not cls or cls.__eq__ not in (object.__eq__, str.__eq__, int.__eq__) or (
            cls.__ne__ not in (object.__ne__, str.__ne__, int.__ne__)
        ):
            self.refuse()
        result = OPS[after](a, b)
        if result is not OPS[before](a, b):
            self.refuse()
        return result

    def add(self, path, root):
        old = ast.parse((Path("/sqlglot") / path).read_bytes())
        new = ast.parse((root / path).read_bytes())
        revised = self.rewrite(old, new, False)
        module = path[:-3].replace("/", ".").removesuffix(".__init__")
        if module in sys.modules:
            raise Unknown("late import guard")
        file = str(root / path)
        self.modules[module] = (file, compile(ast.fix_missing_locations(revised), file, "exec",
                                              dont_inherit=True), path.endswith("/__init__.py"))

    def rewrite(self, old, new, body):
        if old is new:
            return new
        if type(old) is not type(new):
            raise Unknown("changed syntax")
        if body and isinstance(old, ast.Compare) and len(old.ops) == len(new.ops) == 1:
            before, after = type(old.ops[0]).__name__, type(new.ops[0]).__name__
            if before != after and unchanged(old.left, new.left) and unchanged(
                old.comparators[0], new.comparators[0]
            ):
                if before in OPS and after in OPS:
                    args = [ast.Constant(before), ast.Constant(after), new.left, new.comparators[0]]
                    return call("compare", args, new)
                return unreached(new)
        if isinstance(old, ast.Constant):
            if unchanged(old, new):
                return new
            if body:
                return call("refuse", [], new)
            raise Unknown("changed constant at definition time")
        if body and isinstance(old, (ast.BinOp, ast.BoolOp, ast.UnaryOp)) and type(old.op) != type(new.op):
            for name, value in ast.iter_fields(old):
                if name != "op" and not unchanged(value, getattr(new, name)):
                    raise Unknown("changed operator and operands")
            # Keep operand compilation/scope intact, including assignment expressions.
            return unreached(new)

        # Only identical AST structure outside an existing method's executable body.
        # Any new call, binding, class data, import, signature or global declaration
        # takes the conservative path. Do not instrument docstrings or nested headers.
        for name, a in ast.iter_fields(old):
            b = getattr(new, name)
            inside = body
            if isinstance(old, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Import-time protocols (especially metaclass construction) can
                # install class data for dialects loaded by other test workers.
                protocol = old.name.startswith("__") and old.name.endswith("__")
                inside = name == "body" and not protocol
            if a is b:
                continue
            if isinstance(a, ast.AST) and isinstance(b, ast.AST):
                setattr(new, name, self.rewrite(a, b, inside))
            elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
                for i, (left, right) in enumerate(zip(a, b)):
                    if isinstance(left, ast.AST) and isinstance(right, ast.AST):
                        if isinstance(left, ast.Expr) and isinstance(left.value, ast.Constant) and not unchanged(left, right):
                            raise Unknown("changed standalone constant")
                        b[i] = self.rewrite(left, right, inside)
                    elif left != right:
                        raise Unknown("changed binding")
            elif a != b:
                raise Unknown("changed structure")
        return new

    def find_spec(self, fullname, path, target=None):
        if fullname in self.modules:
            file, _, package = self.modules[fullname]
            return importlib.util.spec_from_file_location(fullname, file, loader=self,
                                                          submodule_search_locations=[] if package else None)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        code = self.modules[module.__name__][1]
        module.__dict__[NAME] = self
        exec(code, module.__dict__)


OPS = {"Is": operator.is_, "IsNot": operator.is_not, "Eq": operator.eq, "NotEq": operator.ne}
NAME = "_blast_radius_probe_guard"


def unreached(node):
    return ast.copy_location(ast.IfExp(call("refuse", [], node), node, ast.Constant(None)), node)


def call(method, args, source):
    function = ast.Attribute(ast.Name(NAME, ast.Load()), method, ast.Load())
    return ast.copy_location(ast.Call(function, args, []), source)
