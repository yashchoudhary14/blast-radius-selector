"""Measure fork and collection with test bodies omitted, in a fresh Session."""

import json
import os
import statistics
import time
from functools import partial
from pathlib import Path
from unittest.mock import patch

from . import Session
from .cases import Suite
from .warm import preload


def main():
    file = "sqlglot/parser.py"
    line = (Path("/sqlglot") / file).read_text().splitlines()[0]
    code = f"--- a/{file}\n+++ b/{file}\n@@ -1 +1,2 @@\n+# bench\n {line}\n"
    fixture = "--- /dev/null\n+++ b/tests/fixtures/runner-bench.sql\n@@ -0,0 +1 @@\n+SELECT 1\n"
    ids = ["tests/test_parser.py::TestParser::test_ref_type"]
    with patch.object(Suite, "run", lambda *args: None):
        for name, diff, enabled in (("parser", code, True), ("fixture_cold", fixture, False),
                                    ("fixture_warm", fixture, True), ("fixture_eager", fixture, True)):
            modules = ["tests.test_parser"]
            if name == "fixture_eager":
                catalog = json.loads(Path("/data/tests.json").read_bytes())["tests"]
                modules = list(dict.fromkeys(t["file"][:-3].replace("/", ".") for t in catalog))
            with (patch.dict(os.environ, OPENBLAS_NUM_THREADS="1"),
                  patch("runner.preload", partial(preload, modules=modules)),
                  Session(warm_fixtures=enabled) as session):
                rows = []
                for _ in range(8):
                    cpu = time.process_time()
                    result = session.run(diff, ids)
                    assert not result.failing and not result.notes and not result.ran, result
                    rows.append((result.cpu_s, result.cpu_s + time.process_time() - cpu))
                print(json.dumps(dict(bench=name, first_total_cpu=rows[0][1],
                                      child_median=statistics.median(x for x, _ in rows[1:]),
                                      total_median=statistics.median(x for _, x in rows[1:]))))


if __name__ == "__main__":
    main()
