"""Rebuild per-test and per-import line maps in br-lab."""

import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from importlib.machinery import SourceFileLoader
from pathlib import Path
from unittest.mock import patch

import pytest
from coverage import Coverage

from build_maps import ROOT, write_map
from collect_tests import Collection


def preload(*, lazy_first=False):
    import sqlglot

    modules = sorted((ROOT / "sqlglot").rglob("*.py"))
    if lazy_first:
        initialize_dialects()
    for path in modules:
        name = ".".join(path.relative_to(ROOT).with_suffix("").parts).removesuffix(".__init__")
        if name == "sqlglot.__main__":
            # The CLI parses argv on import. Its --version option also expects
            # packaging metadata absent from the pinned source checkout.
            with patch.object(sys, "argv", ["sqlglot", "--tokenize", ""]), patch.object(
                sqlglot, "__version__", getattr(sqlglot, "__version__", "source checkout"), create=True
            ):
                importlib.import_module(name)
        else:
            importlib.import_module(name)

    if not lazy_first:
        initialize_dialects()
    print(f"Preloaded {len(modules)} module paths", flush=True)


def initialize_dialects():
    from sqlglot.dialects import Dialect, Dialects
    for name in Dialects:
        dialect = Dialect.get_or_raise(name)
        dialect.tokenizer()
        dialect.parser()
        dialect.generator()
    print(f"Initialized {len(Dialects)} dialects", flush=True)


def cache_lines(owner, attr, initial, exercise):
    """Link cold-only initialization lines to the lookup sites executed on a hit."""
    reads = set()

    class ObservedCache(dict):
        def record(self):
            caller = sys._getframe(2)
            reads.add((Path(caller.f_code.co_filename).relative_to(ROOT).as_posix(), caller.f_lineno))

        def get(self, key, default=None):
            self.record()
            return super().get(key, default)

        def __contains__(self, key):
            self.record()
            return super().__contains__(key)

        def __len__(self):
            self.record()
            return super().__len__()

        def __getitem__(self, key):
            self.record()
            return super().__getitem__(key)

    cov = Coverage(data_file=None, config_file=False, include=[str(ROOT / "sqlglot/**/*.py")])
    cov.set_option("run:core", "ctrace")
    cov.start()
    try:
        # Only warm lookups stand for subsequent consumers. Modules stay loaded,
        # so registry misses exercise SQLGlot's registration of imported classes.
        with patch.object(owner, attr, dict(initial)):
            cov.switch_context("cold")
            exercise()
            warm = ObservedCache(getattr(owner, attr))
        with patch.object(owner, attr, warm):
            cov.switch_context("warm")
            exercise()
    finally:
        cov.stop()
    data = cov.get_data()
    files = {}
    for path in sorted(data.measured_files()):
        data.set_query_context("cold")
        cold = set(data.lines(path) or ())
        data.set_query_context("warm")
        extra = cold.difference(data.lines(path) or ())
        if extra:
            files[Path(path).relative_to(ROOT).as_posix()] = sorted(extra)
    if not reads or not files:
        raise ValueError(f"{attr} produced no lookup or fill lines")
    return {"reads": sorted(reads), "files": files}


def build_cache_lines():
    from sqlglot import generator
    from sqlglot.dialects import Dialect, Dialects

    dialects = [Dialect.get_or_raise(name) for name in Dialects]

    def generators():
        for dialect in dialects:
            dialect.generator()

    def get_by_name():
        for name in Dialects:
            Dialect.get_or_raise(name.value)

    def get_by_subscript():
        for name in Dialects:
            Dialect[name.value]

    return [
        cache_lines(generator, "_DISPATCH_CACHE", {}, generators),
        *(cache_lines(type(Dialect), "_classes", {"": Dialect}, lookup)
          for lookup in (get_by_name, get_by_subscript, lambda: Dialect.classes)),
    ]


def build_import_lines(output):
    # Separate cold interpreter: preserve numbered test and fixture contexts.
    # The loader covers both regular imports and importlib's lazy dialect loads.
    cov = Coverage(data_file=None, config_file=False,
                   include=[str(ROOT / sub / "**/*.py") for sub in ("sqlglot", "tests")])
    cov.set_option("run:core", "ctrace")
    original = SourceFileLoader.exec_module
    owner = ""

    def measure(loader, module):
        nonlocal owner
        path = Path(loader.path)
        if not path.is_relative_to(ROOT):
            return original(loader, module)
        previous, owner = owner, path.relative_to(ROOT).as_posix()
        cov.switch_context(owner)
        try:
            return original(loader, module)
        finally:
            owner = previous
            cov.switch_context(owner)

    cov.start()
    try:
        with patch.object(SourceFileLoader, "exec_module", measure):
            preload(lazy_first=True)
            for path in sorted((ROOT / "tests").rglob("*.py")):
                name = ".".join(path.relative_to(ROOT).with_suffix("").parts).removesuffix(".__init__")
                try:
                    importlib.import_module(name)
                except FileNotFoundError:
                    # This standalone script opens its author's external fixtures at import.
                    if name != "tests.gen_fixtures":
                        raise
    finally:
        cov.stop()

    data = cov.get_data()
    expected = {p.relative_to(ROOT).as_posix() for tree in ("sqlglot", "tests")
                for p in (ROOT / tree).rglob("*.py")}
    if missing := expected - data.measured_contexts():
        raise ValueError(f"Missing import contexts: {sorted(missing)}")
    sets, files = {}, {}
    for filename in sorted(data.measured_files()):
        rows = {}
        for line, labels in sorted(data.contexts_by_lineno(filename).items()):
            modules = tuple(sorted(label for label in labels if label))
            if line > 0 and modules:
                rows[line] = sets.setdefault(modules, len(sets))
        if rows:
            files[Path(filename).relative_to(ROOT).as_posix()] = rows
    with tempfile.TemporaryDirectory(dir=output.parent) as staging:
        staged = Path(staging) / output.name
        write_map(staged, {"sets": list(sets), "files": files, "caches": build_cache_lines()})
        staged.replace(output)
    print(f"Import contexts: {len(data.measured_contexts())-1} modules, "
          f"{sum(map(len, files.values()))} source lines", flush=True)


class LineCollection(Collection):
    def __init__(self, manifest, work):
        super().__init__(manifest, work, "coverage")
        self.outcomes = {}

    def pytest_runtest_logreport(self, report):
        # A passing call can follow a failed subtest, or precede a teardown error.
        node = report.nodeid
        if report.failed:
            self.outcomes[node] = "failed"
        elif self.outcomes.get(node) != "failed" and (report.skipped or report.when == "call"):
            self.outcomes[node] = report.outcome


def collect(mode, work, manifest):
    plugin = LineCollection(manifest, work)
    try:
        if mode == "preload":
            plugin.recorder.switch("<import>")
            preload()
            plugin.recorder.switch("")
        status = pytest.main(["tests", "-q", "--tb=short", "-p", "no:cacheprovider"], plugins=[plugin])
    finally:
        plugin.recorder.finish()
    if status not in (pytest.ExitCode.OK, pytest.ExitCode.TESTS_FAILED):
        raise RuntimeError(f"{mode} pytest exited {status}")
    expected = plugin.numbered | plugin.skipped
    if plugin.outcomes.keys() != expected:
        raise RuntimeError(f"Incomplete {mode} outcomes: {expected ^ plugin.outcomes.keys()}")
    (work / "outcomes.json").write_text(json.dumps(plugin.outcomes))


def selected_coverage(work, manifest, selected):
    groups = defaultdict(list, {"": [], "<import>": []})
    groups.update({node: [] for node in manifest["skipped"]})
    for test in manifest["tests"]:
        node, number = test["id"], test["number"]
        for label in (node, "scope:" + node.rsplit("::", 1)[0], "scope:" + node.split("::", 1)[0]):
            group = groups[label]
            if node in selected:
                group.append(number)
    cov = Coverage(config_file=False, data_file=str(work / "cov"))
    cov.combine(strict=True)
    data = cov.get_data()
    contexts = data.measured_contexts()
    if missing := selected - contexts:
        raise RuntimeError(f"Missing coverage contexts: {sorted(missing)}")
    if unknown := contexts - groups.keys():
        raise RuntimeError(f"Unexpected coverage contexts: {sorted(unknown)}")
    if work.name == "preload" and "<import>" not in contexts:
        raise RuntimeError("Preload coverage context missing")
    return data, groups


def build(manifest_path, output, work):
    manifest = json.loads(manifest_path.read_bytes())
    outcomes = {}
    for mode in ("normal", "preload"):
        directory = work / mode
        directory.mkdir()
        start = time.monotonic()
        subprocess.run([sys.executable, __file__, mode, str(directory), str(manifest_path)],
                       cwd=ROOT, check=True)
        outcomes[mode] = json.loads((directory / "outcomes.json").read_bytes())
        print(f"{mode} coverage: {time.monotonic()-start:.1f}s", flush=True)
    nodes = {t["id"] for t in manifest["tests"]}
    fallback = {node for node in nodes if outcomes["preload"][node] != "passed"
                and outcomes["normal"][node] == "passed"}
    for node in sorted(outcomes["normal"]):
        old, new = outcomes["normal"][node], outcomes["preload"][node]
        if old != new:
            print(f"{node}: {old} -> {new}")
    print(f"Normal coverage fallback ({len(fallback)}): {sorted(fallback)}", flush=True)
    runs = [selected_coverage(work / "preload", manifest, nodes - fallback)]
    if fallback:
        runs.append(selected_coverage(work / "normal", manifest, fallback))

    files, interned = {}, {}
    for folder in ("sqlglot", "tests"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            hits = defaultdict(set)
            for data, groups in runs:
                for line, labels in data.contexts_by_lineno(str(path)).items():
                    if line > 0:
                        hits[line].update(n for label in labels for n in groups[label])
            files[path.relative_to(ROOT).as_posix()] = {
                line: interned.setdefault(tuple(sorted(nums)), len(interned))
                for line, nums in sorted(hits.items()) if nums
            }
    if {n for row in interned for n in row} != {t["number"] for t in manifest["tests"]}:
        raise RuntimeError("Packed map is missing numbered tests")
    with tempfile.TemporaryDirectory(dir=output.parent) as staging:
        staged = Path(staging) / output.name
        write_map(staged, {"sets": list(interned), "files": files})
        staged.replace(output)
    print(f"Packed {len(nodes)} tests, {len(files)} files, {len(interned)} unique sets", flush=True)


if __name__ == "__main__":
    if os.environ.get("SKIP_INTEGRATION") != "0":
        raise SystemExit("Set SKIP_INTEGRATION=0")
    if len(sys.argv) == 4 and sys.argv[1] in {"preload", "normal"}:
        collect(sys.argv[1], Path(sys.argv[2]), json.loads(Path(sys.argv[3]).read_bytes()))
    elif sys.argv[1:] == ["--imports-only"]:
        build_import_lines(Path("/maps/import_lines.json.gz"))
    else:
        subprocess.run([sys.executable, __file__, "--imports-only"], cwd=ROOT, check=True)
        with tempfile.TemporaryDirectory() as tmp:
            build(Path("/manifest/tests.json"), Path("/maps/lines.json.gz"), Path(tmp))
