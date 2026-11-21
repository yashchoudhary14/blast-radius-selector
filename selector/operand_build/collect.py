"""Run each catalog method with fresh state, retaining its module and class fixtures."""
import gzip
import importlib
import json
import marshal
import os
import subprocess
import sys
import tempfile
import unittest
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import record


def join(rows, limits, more_rows, more_limits):
    for key, values in more_rows.items():
        rows[key].update(values)
    for key, value in more_limits.items():
        if key not in limits:
            limits[key] = value
        else:
            old = limits[key]
            limits[key] = ((min(old[0], value[0]), max(old[1], value[1]))
                           if value and old and type(old[0]) is type(value[0]) else None)


def test(module, row, output):
    # Fork once per test FROM the pristine module import. Its observed operand
    # values are inherited too. No earlier method has populated any caches.
    child = os.fork()
    if child:
        _, status = os.waitpid(child, 0)
        if status:
            print(f"unusable {row['number']} child status {status}", flush=True)
        return
    try:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["BR_OPERAND_TEST"] = tmp
            # Fork-only descendants bypass sitecustomize and may use os._exit.
            # Any such child makes this observation incomplete.
            os.register_at_fork(after_in_child=lambda: (Path(tmp) / f"fork-{os.getpid()}").touch())
            _, cls, method = row["id"].split("::")
            suite = unittest.TestSuite([getattr(module, cls)(method)])
            result = unittest.TestResult()
            suite.run(result)
            if result.wasSuccessful() and result.testsRun == 1 and not result.skipped:
                for file in Path(tmp).iterdir():
                    join(record.vectors, record.bounds, *marshal.loads(file.read_bytes()))
                payload = marshal.dumps((dict(record.vectors), record.bounds))
                (output / str(row["number"])).write_bytes(gzip.compress(payload, compresslevel=1, mtime=0))
            else:
                print(f"unusable {row['number']} {str(result.errors + result.failures)[:700]}", flush=True)
        os._exit(0)
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(1)


def main():
    rows = json.loads(Path("/data/tests.json").read_bytes())["tests"]
    output = Path(sys.argv[1])
    output.mkdir(exist_ok=True, parents=True)
    if len(sys.argv) > 2:
        path = sys.argv[2]
        module = importlib.import_module(path[:-3].replace("/", "."))
        for row in rows:
            if row["file"] == path:
                test(module, row, output)
        print(path, flush=True)
        return
    from prepare import prepare
    prepare()
    (output / "sites.json").write_bytes(Path("/tmp/operand-sites.json").read_bytes())
    # Start the longest modules first, balancing the remaining files by baseline cost.
    durations = defaultdict(float)
    for row in rows:
        durations[row["file"]] += row["seconds"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        def run(path):
            subprocess.run([sys.executable, __file__, str(output), path], check=True)
        list(pool.map(run, sorted(durations, key=durations.get, reverse=True)))
    print(f"completed {len(list(output.glob('[0-9]*')))}/{len(rows)} contexts", flush=True)


if __name__ == "__main__":
    main()
