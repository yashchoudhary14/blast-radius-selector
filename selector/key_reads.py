"""Offline collector of key reads for literal string dictionaries with lookup-only uses.

Run in the pinned image with /data/tests.json and a writable /app/maps. Only a
passing method with a recorded read at the original covered site is certified.
Child processes, fixtures and incomplete observations retain coverage candidates.
"""
import ast
import builtins
import importlib
import importlib.abc
import importlib.machinery
import json
import marshal
import os
import sys
import unittest
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .collect import BASE, MAPS


def sites():
    valid, invalid = set(), set()
    lookups = defaultdict(list)
    for folder in ("sqlglot", "tests"):
        for path in (BASE / folder).rglob("*.py"):
            tree = ast.parse(path.read_bytes())
            parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
            for node, parent in parents.items():
                if isinstance(node, ast.Dict) and isinstance(parent, (ast.AnnAssign, ast.Assign)):
                    target = parent.target if isinstance(parent, ast.AnnAssign) else parent.targets[0]
                    if (isinstance(target, ast.Name) and target.id.isupper() and node.keys
                            and all(isinstance(k, ast.Constant) and type(k.value) is str for k in node.keys)):
                        valid.add(target.id)
                if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                    name = node.attr
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    name = node.id
                else:
                    continue
                if not name.isupper():
                    continue
                key = None
                if isinstance(parent, ast.Attribute) and parent.attr == "get":
                    call = parents.get(parent)
                    if isinstance(call, ast.Call) and call.func is parent and call.args:
                        key = call.args[0]
                elif isinstance(parent, ast.Subscript) and parent.value is node:
                    key = parent.slice
                elif (isinstance(parent, ast.Compare) and parent.comparators == [node]
                      and isinstance(parent.ops[0], (ast.In, ast.NotIn))):
                    key = parent.left
                if isinstance(key, ast.expr) and not isinstance(key, ast.Starred):
                    rel = path.relative_to(BASE).as_posix()
                    loc = key.lineno, key.col_offset, key.end_lineno, key.end_col_offset, type(key).__name__
                    lookups[name].append((rel, node.lineno, loc))
                else:
                    invalid.add(name)
    return {n: lookups[n] for n in valid - invalid if lookups[n]}


class Instrument(importlib.abc.MetaPathFinder):
    def __init__(self, lookup):
        self.by_file = defaultdict(dict)
        for name, sites in lookup.items():
            for path, line, loc in sites:
                self.by_file[str(BASE / path)][loc] = (name, path, line)

    def find_spec(self, fullname, path, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec and spec.origin in self.by_file:
            points = self.by_file[spec.origin]
            def get_code(name):
                tree = ast.parse(Path(spec.origin).read_bytes(), spec.origin)
                class Keys(ast.NodeTransformer):
                    def visit(self, node):
                        loc = ((node.lineno, node.col_offset, node.end_lineno, node.end_col_offset,
                                type(node).__name__) if isinstance(node, ast.expr) else None)
                        site = points.get(loc)
                        if site and isinstance(node, ast.expr):
                            return ast.copy_location(ast.Call(ast.Name("_br_key", ast.Load()),
                                                              [ast.Constant(site), node], []), node)
                        return super().visit(node)
                return compile(ast.fix_missing_locations(Keys().visit(tree)), spec.origin, "exec")
            spec.loader.get_code = get_code
            return spec


def collect_group(job):
    file, cases, lookup = job
    os.chdir(BASE)
    sys.meta_path.insert(0, Instrument(lookup))
    observed = defaultdict(set)
    spawned = False
    def key(site, value):
        observed[site].add(value if type(value) is str else None)
        return value
    def audit(event, args):
        nonlocal spawned
        if event in ("os.fork", "os.posix_spawn", "subprocess.Popen"):
            spawned = True
    builtins._br_key = key
    sys.addaudithook(audit)
    module = importlib.import_module(file[:-3].replace("/", "."))
    collection = set(observed)
    result = {}
    for number, cls, method in cases:
        observed.clear()
        spawned = False
        outcome = unittest.TestResult()
        unittest.TestSuite([getattr(module, cls)(method)]).run(outcome)
        if outcome.wasSuccessful() and outcome.testsRun == 1 and not outcome.skipped and not spawned:
            result[number] = {site: keys for site, keys in observed.items()
                              if site not in collection and None not in keys}
    return result


def build():
    lookup = sites()
    print(f"{len(lookup)} lookup-only dictionaries", flush=True)
    grouped = defaultdict(list)
    candidates = 0
    for files in lookup.values():
        for path, line, _ in files:
            cache = MAPS / "ast" / (path + ".lines")
            if cache.is_file():
                candidates |= marshal.loads(cache.read_bytes()).get(line, 0)
    for test in json.loads(Path("/data/tests.json").read_bytes())["tests"]:
        if not candidates & (1 << test["number"]):
            continue
        file, cls, method = test["id"].split("::")
        grouped[file].append((test["number"], cls, method))
    # Respect ordinary import order within each test file. Fresh workers avoid
    # cross-file caches and synthetic dialect registration.
    index = {}
    with ProcessPoolExecutor(4, max_tasks_per_child=1) as pool:
        for result in pool.map(collect_group, [(file, tests, lookup) for file, tests in grouped.items()]):
            for number, hits in result.items():
                for site, keys in hits.items():
                    masks = index.setdefault(site, {})
                    for k in (None, *keys):
                        masks[k] = masks.get(k, 0) | (1 << number)
    (MAPS / "key-reads.bin").write_bytes(marshal.dumps(index))


if __name__ == "__main__":
    build()
