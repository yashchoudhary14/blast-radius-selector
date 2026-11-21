"""Measure pinned, unmodified test costs under the selector's fork policy."""
import argparse
import json
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

from . import Session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--singles", action="store_true")
    args = parser.parse_args()
    catalog = json.loads(Path("/data/tests.json").read_bytes())["tests"]
    import duckdb
    import pandas.testing
    with Session() as session:
        startup = session.warm()
        print(json.dumps({"preload_cpu": startup}), flush=True)
        if args.singles:
            from selector.execute import SINGLE
            for test in catalog:
                if test["number"] in SINGLE:
                    r = session.run("", [test["id"]])
                    assert r.ran == [test["id"]] and not r.failing, r
                    print(json.dumps(dict(number=test["number"], cpu=r.cpu_s)), flush=True)
        else:
            node = "tests/test_parser.py::TestParser::test_ref_type"
            for file in ("sqlglot/parser.py", "sqlglot/parsers/dremio.py", "tests/fixtures/identity.sql"):
                patch = f"--- a/{file}\n+++ b/{file}\n@@ -0,0 +1 @@\n+" + ("# bench" if file.endswith(".py") else "-- bench") + "\n"
                for runner in (session, Session(warm_fixtures=False)):
                    with nullcontext() if runner is session else runner:
                        costs = []
                        for _ in range(7):
                            start = time.process_time()
                            result = runner.run(patch, [node])
                            assert result.ran == [node] and not result.failing, result
                            costs.append(result.cpu_s + time.process_time() - start)
                        print(json.dumps(dict(file=file, warm=runner is session,
                                              median=statistics.median(costs))), flush=True)


if __name__ == "__main__":
    main()
