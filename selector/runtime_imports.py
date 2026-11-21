"""Offline baseline map of lazy imports. Each numbered method gets a fresh module cache."""

import argparse
import ast
import gzip
import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .collect import BASE, MAPS
from .audit.sitecustomize import MissingModules


def loaded_paths():
    paths = set()
    for module in sys.modules.copy().values():
        path = getattr(module, "__file__", None)
        if isinstance(path, str) and path.startswith(str(BASE) + "/"):
            paths.add(path[len(str(BASE)) + 1:])
    return paths


def measure_file(file, tests, output):
    # Run class/module fixtures afresh as well. A fork inherits only collection
    # imports, so another method cannot satisfy this method's lazy imports.
    missing = set()
    sys.meta_path.append(MissingModules(missing.add))
    module = importlib.import_module(file[:-3].replace("/", "."))
    initial = loaded_paths()
    measured = {}
    subprocess_log = output.with_suffix(".imports")
    for test in tests:
        if test["file"] != file:
            continue
        _, klass, method = test["id"].split("::")
        pid = os.fork()
        if not pid:
            try:
                os.environ["BR_CHILD_IMPORT_LOG"] = str(subprocess_log)
                os.environ["PYTHONPATH"] = str(Path(__file__).with_name("audit")) + os.pathsep + os.environ["PYTHONPATH"]
                with open(os.devnull, "w") as sink:
                    os.dup2(sink.fileno(), 1)
                    result = unittest.TestResult()
                    case = getattr(module, klass)(method)
                    unittest.TestSuite([case]).run(result)
                    if not result.wasSuccessful():
                        raise RuntimeError((result.errors + result.failures)[:1])
                    output.write_text(json.dumps(sorted((loaded_paths() - initial) | missing)))
                os._exit(0)
            except BaseException as exc:
                print(f"{test['id']}: {exc!r}", file=sys.stderr, flush=True)
                os._exit(1)
        if os.waitpid(pid, 0)[1]:
            raise RuntimeError(f"baseline method failed: {test['id']}")
        imported = set(json.loads(output.read_bytes()))
        if subprocess_log.exists():
            imported.update(subprocess_log.read_text().splitlines())
            subprocess_log.unlink()
        measured[test["number"]] = sorted(imported - initial)
    output.write_text(json.dumps(measured))


def observes_heap(file):
    """Find methods sampling process-wide allocation state."""
    observed = set()
    for cls in ast.walk(ast.parse((BASE / file).read_bytes())):
        if isinstance(cls, ast.ClassDef):
            for func in cls.body:
                if isinstance(func, ast.FunctionDef) and any(
                    isinstance(node, ast.Attribute) and node.attr in {"get_traced_memory", "take_snapshot"}
                    for node in ast.walk(func)
                ):
                    observed.add(f"{file}::{cls.name}::{func.name}")
    return observed


def build(tests, manifest, jobs):
    files = sorted({t["file"] for t in tests})
    # Start the slowest test files first.
    seconds = defaultdict(float)
    for t in tests:
        seconds[t["file"]] += t["seconds"]
    files.sort(key=seconds.get, reverse=True)
    imports = defaultdict(set)
    with tempfile.TemporaryDirectory() as tmp:
        def measure(file):
            output = Path(tmp) / file.replace("/", "_")
            subprocess.run([sys.executable, "-m", "selector.runtime_imports", "--file", file,
                            "--output", str(output), "--manifest", str(manifest)], check=True)
            return json.loads(output.read_bytes())

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            for mapping in pool.map(measure, files):
                for number, paths in mapping.items():
                    for path in paths:
                        imports[path].add(int(number))
    observers = set().union(*(observes_heap(file) for file in files))
    result = {"imports": {p: sorted(ns) for p, ns in imports.items()},
              "heap_observers": sorted(t["number"] for t in tests if t["id"] in observers)}
    missing = {p: result["imports"].pop(p) for p in imports if not (BASE / p).exists()}
    (MAPS / "missing_imports.json.gz").write_bytes(gzip.compress(
        json.dumps(missing, sort_keys=True, separators=(",", ":")).encode(), mtime=0))
    (MAPS / "runtime_imports.json").write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("/data/tests.json"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--file")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    catalog = json.loads(args.manifest.read_bytes())["tests"]
    if args.file:
        measure_file(args.file, catalog, args.output)
    else:
        build(catalog, args.manifest, args.jobs)
