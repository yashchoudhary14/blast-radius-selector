"""Generate collection, inheritance and dialect-mention maps from the pinned checkout."""

import ast
import contextlib
import importlib
import inspect
import json
import os
import subprocess
import sys
import unittest
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = Path("/sqlglot")
MAPS = Path("/app/maps")


def inspect_file(path):
    os.chdir(BASE)
    # Imports run before pytest assigns a test context. Include fixture reads
    # performed here even when no numbered method opens the file again.
    reads, directories = set(), set()

    def observe(event, args):
        if event not in {"open", "os.listdir", "os.scandir"}:
            return
        name = args[0]
        if not isinstance(name, (str, bytes)):
            return
        name = os.path.abspath(os.fsdecode(name))
        if not name.startswith("/sqlglot/") and name != "/sqlglot":
            return
        if sys._getframe(1).f_code.co_filename == "<frozen importlib._bootstrap_external>":
            return
        relative = os.path.relpath(name, BASE)
        if event == "open":
            if args[2] & os.O_ACCMODE != os.O_WRONLY:
                reads.add(relative)
        else:
            directories.add(relative)

    sys.addaudithook(observe)
    module = importlib.import_module(path[:-3].replace("/", "."))
    collected = sorted(reads), sorted(directories)
    imports = set()
    for loaded in sys.modules.copy().values():
        name = getattr(loaded, "__file__", None)
        if name and Path(name).is_relative_to(BASE):
            imports.add(Path(name).relative_to(BASE).as_posix())
    classes = {}
    for name, cls in vars(module).items():
        if not isinstance(cls, type) or not issubclass(cls, unittest.TestCase):
            continue
        methods = unittest.defaultTestLoader.getTestCaseNames(cls)
        if not methods:
            continue
        ancestors = []
        for ancestor in cls.__mro__:
            if not ancestor.__module__.startswith("tests."):
                continue
            src = Path(inspect.getfile(ancestor)).relative_to(BASE).as_posix()
            ancestors.append(f"{src}::{ancestor.__qualname__}")
        classes[f"{path}::{name}"] = (ancestors, methods)
    return sorted(imports), classes, collected


def fresh(path):
    output = subprocess.check_output([sys.executable, "-m", "selector.collect", path])
    return path, json.loads(output)


def write(name, obj):
    (MAPS / name).write_text(json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n")


def collect():
    from sqlglot.dialects import DIALECTS
    from sqlglot import Dialect

    dialects = {name.lower(): type(Dialect.get_or_raise(name.lower())) for name in DIALECTS}
    children = {name: sorted(n for n, child in dialects.items() if issubclass(child, cls))
                for name, cls in dialects.items()}
    paths = sorted(p.relative_to(BASE).as_posix() for p in (BASE / "tests").rglob("test_*.py"))
    imports, classes, reads = {}, {}, {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for path, (loaded, found, collected) in pool.map(fresh, paths):
            imports[path] = loaded
            classes.update(found)
            if any(collected):
                reads[path] = collected
    write("collect_imports.json", imports)
    write("collection_reads.json", reads)

    def mentions(lines):
        source = "\n".join(lines).lower()
        return {name for name in dialects if name in source}

    class_text, methods_text = {}, {}
    for path in sorted({key.split("::")[0] for entries, _ in classes.values() for key in entries}):
        source = (BASE / path).read_text()
        lines = source.splitlines()
        for cls in (node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)):
            key = f"{path}::{cls.name}"
            class_text[key] = set()
            for stmt in cls.body:
                start = min([stmt.lineno, *(d.lineno for d in getattr(stmt, "decorator_list", ()))])
                found = mentions(lines[start - 1:stmt.end_lineno])
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods_text[(key, stmt.name)] = found
                else:
                    class_text[key].update(found)
    found = defaultdict(list)
    mro = {}
    for key, (ancestors, methods) in sorted(classes.items()):
        mro[key] = ancestors
        shared = set().union(*(class_text.get(a, ()) for a in ancestors))
        file = key.split("::")[0]
        dialect = Path(file).stem.removeprefix("test_")
        if file.startswith("tests/dialects/") and dialect in dialects:
            shared.add(dialect)
        for method in methods:
            specific = next((methods_text[a, method] for a in ancestors if (a, method) in methods_text), ())
            for name in shared.union(specific):
                found[name].append(f"{key}::{method}")
    write("dialect_mentions.json", {"mentions": found, "descendants": children, "test_mro": mro})


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with contextlib.redirect_stdout(sys.stderr):
            result = inspect_file(sys.argv[1])
        print(json.dumps(result))
    else:
        collect()
