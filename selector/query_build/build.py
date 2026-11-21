"""Build pinned per-query maps. See maps/README.md for the Docker command."""

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

from runner.query_cases import QUERIES, filter_queries


MAX_NODES = "tests/test_parser.py::TestParser::test_max_nodes"
CACHE = Path("/query-cache")


def run_case(node, index):
    file, cls, method = node.split("::")
    module = importlib.import_module(file[:-3].replace("/", "."))
    case = getattr(module, cls)(method)
    if index == "contexts":
        from contexts import instrument
        instrument(case, module, node)
    elif index != "full":
        filter_queries(case, module, node, [int(index)])
    result = unittest.TestResult()
    unittest.TestSuite([case]).run(result)
    if not (result.wasSuccessful() and result.testsRun == 1 and not result.skipped):
        raise RuntimeError((node, index, result.errors, result.failures, result.skipped))


def capture(job):
    node, index = job
    output = CACHE / f"{node.rsplit('::', 1)[1]}-{index}.gz"
    if output.is_file():
        return
    with tempfile.TemporaryDirectory() as tmp:
        env_key = "BR_QUERY_CONTEXT" if index == "contexts" else "BR_QUERY_TRACE"
        subprocess.run([sys.executable, __file__, node, str(index)], check=True,
                       env={**os.environ, env_key: tmp})
        files = list(Path(tmp).iterdir())
        spawned = "test_execute_" in node or "TestOptimizer" in node
        if len(files) < 1 + int(spawned):
            raise RuntimeError(f"missing child trace for {job}")
        contexts = defaultdict(lambda: defaultdict(set))
        for file in files:
            part = json.loads(file.read_bytes())
            if not part:
                raise RuntimeError(f"empty trace for {job}")
            for i, paths in (part if index == "contexts" else {"full": part}).items():
                for path, lines in paths.items():
                    contexts[i][path].update(lines)
        data = {i: {p: sorted(v) for p, v in paths.items()} for i, paths in contexts.items()}
        output.write_bytes(gzip.compress(json.dumps(data if index == "contexts" else data["full"]).encode()))


def build():
    from tests.helpers import load_sql_fixture_pairs

    nodes = {n: len(list(load_sql_fixture_pairs("optimizer/tpc-" + family + "/tpc-" + family + ".sql")))
             for n, kind in QUERIES.items() for family in [kind.removeprefix("tpc")]}
    nodes[MAX_NODES] = 1
    CACHE.mkdir(exist_ok=True)
    jobs = [(n, i) for n, count in nodes.items() for i in range(count) if n != MAX_NODES]
    jobs = [(n, "contexts") for n in QUERIES] + [(n, "full") for n in nodes] + jobs
    with ThreadPoolExecutor(max_workers=int(os.getenv("JOBS", 4))) as pool:
        for i, _ in enumerate(pool.map(capture, jobs), 1):
            if i % 20 == 0:
                print(f"{i}/{len(jobs)}", flush=True)
    maps = {}
    for node, count in nodes.items():
        def read(index):
            path = CACHE / f"{node.rsplit('::', 1)[1]}-{index}.gz"
            return json.loads(gzip.decompress(path.read_bytes()))

        full = read("full")
        mask = (1 << count) - 1
        rows = defaultdict(dict)
        if node != MAX_NODES:
            contexts = read("contexts")
            if set(contexts) != {"-1", *map(str, range(count))}:
                raise ValueError(f"incomplete full-loop contexts for {node}")
            for i, paths in contexts.items():
                bits = mask if i == "-1" else (1 << int(i))
                for path, lines in paths.items():
                    for line in lines:
                        rows[path][line] = rows[path].get(line, 0) | bits
        # A line unique to the uninstrumented loop has no known owner.
        # Isolated runs can add owners, never narrow this shared attribution.
        for path, lines in full.items():
            for line in lines:
                rows[path].setdefault(line, mask)
        if node != MAX_NODES:
            for index in range(count):
                for path, lines in read(index).items():
                    for line in lines:
                        rows[path][line] = rows[path].get(line, 0) | (1 << index)
        # Intern masks, and group consecutive lines carrying the same mask.
        masks = sorted({v for m in rows.values() for v in m.values()})
        ids = {v: i for i, v in enumerate(masks)}
        blocks = {}
        for path, lines in sorted(rows.items()):
            groups = []
            for line, value in sorted(lines.items()):
                k = ids[value]
                if groups and groups[-1][1] == line - 1 and groups[-1][2] == k:
                    groups[-1][1] = line
                else:
                    groups.append([line, line, k])
            blocks[path] = groups
        maps[node] = [count, masks, blocks]
        print(node, count, flush=True)
    data = json.dumps(maps, separators=(",", ":")).encode()
    Path("/app/maps/query_cases.json.gz").write_bytes(gzip.compress(data, mtime=0))


if __name__ == "__main__":
    if len(sys.argv) == 3:
        run_case(*sys.argv[1:])
    else:
        build()
