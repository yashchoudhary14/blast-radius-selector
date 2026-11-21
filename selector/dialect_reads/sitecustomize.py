"""Record lookup keys from the pristine test and its spawned interpreters."""

import ast
import atexit
import builtins
import importlib.abc
import importlib.machinery
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from selector.dialect_reads import SITES


class Loader(importlib.machinery.SourceFileLoader):
    def __init__(self, name, path, sites):
        super().__init__(name, path)
        self.sites = sites

    def get_code(self, fullname):
        tree = ast.parse(Path(self.path).read_bytes(), self.path)
        expected = {ast.dump(ast.parse(expr, mode="eval").body): fields for expr, fields in self.sites.items()}
        seen = set()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Call, ast.Compare)):
                continue
            syntax = ast.dump(node)
            if syntax in expected:
                assert syntax not in seen
                seen.add(syntax)
                if isinstance(node, ast.Compare):
                    key = node.left
                else:
                    slot = int(isinstance(node.func, ast.Name) and node.func.id == "getattr")
                    key = node.args[slot]
                observed = ast.copy_location(
                    ast.Call(ast.Name("_br_dispatch_key", ast.Load()),
                             [ast.Constant(expected[syntax]), key], []), key
                )
                if isinstance(node, ast.Compare):
                    node.left = observed
                else:
                    node.args[slot] = observed
        if seen != expected.keys():
            raise ValueError(f"Unknown dispatch layout in {self.path}")
        return compile(ast.fix_missing_locations(tree), self.path, "exec", dont_inherit=True)


class Finder(importlib.abc.MetaPathFinder):
    def __init__(self, base):
        self.files = {str(base / path): sites for path, sites in SITES.items()}

    def find_spec(self, fullname, path, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.origin in self.files:
            spec.loader = Loader(fullname, spec.origin, self.files[spec.origin])
            return spec


def install(base, directory):
    out = directory / f"{os.getpid()}.json"
    out.touch()
    keys = defaultdict(set)
    unknown = set()

    def record(fields, key):
        if type(key) is str:
            name = key
        elif isinstance(key, type) and key.__module__.startswith("sqlglot.expressions."):
            name = key.__name__
        elif type(key).__module__ == "sqlglot.tokenizer_core" and type(key).__name__ == "TokenType":
            name = key.name
        else:
            unknown.update(fields)
            return key
        for field in fields:
            keys[field].add(name)
        return key

    def save():
        out.write_text(json.dumps({
            "keys": {name: sorted(v) for name, v in keys.items()}, "unknown": sorted(unknown),
        }), encoding="utf8")

    atexit.register(save)
    builtins._br_dispatch_key = record
    sys.meta_path.insert(0, Finder(base))


if os.environ.get("BR_DIALECT_READS"):
    install(Path(os.environ["BR_DIALECT_BASE"]), Path(os.environ["BR_DIALECT_READS"]))
