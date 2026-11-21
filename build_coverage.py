"""Rebuild lines.json.gz in br-lab using the lab/coverage pytest collector."""

import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import pytest
from coverage import Coverage

from build_maps import ROOT, write_map
from collect_tests import Collection


def preload():
    import sqlglot

    modules = sorted((ROOT / "sqlglot").rglob("*.py"))
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

    from sqlglot.dialects import Dialect, Dialects

    for name in Dialects:
        dialect = Dialect.get_or_raise(name)
        dialect.tokenizer()
        dialect.parser()
        dialect.generator()
    print(f"Preloaded {len(modules)} module paths and initialized {len(Dialects)} dialects", flush=True)


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
    else:
        with tempfile.TemporaryDirectory() as tmp:
            build(Path("/manifest/tests.json"), Path("/maps/lines.json.gz"), Path(tmp))
