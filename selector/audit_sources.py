"""Distinguish source-as-data reads from opening a .py to import its bytecode."""

import gzip
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import pytest

from .collect import MAPS


class Context:
    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_runtest_protocol(self, item, nextitem):
        os.environ["BR_AUDIT_TEST"] = item.nodeid
        try:
            return (yield)
        finally:
            os.environ.pop("BR_AUDIT_TEST", None)

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_fixture_setup(self, fixturedef, request):
        if fixturedef.scope not in {"class", "module"}:
            return (yield)
        previous = os.environ["BR_AUDIT_TEST"]
        os.environ["BR_AUDIT_TEST"] = request.node.nodeid
        try:
            return (yield)
        finally:
            os.environ["BR_AUDIT_TEST"] = previous


def main():
    log = Path(os.environ["BR_AUDIT_LOG"])
    tests = json.loads(Path("/data/tests.json").read_bytes())["tests"]
    with gzip.open(MAPS / "reads.json.gz") as source:
        reads = json.load(source)
    opts = ["tests", "-q", "--tb=short", "-p", "no:cacheprovider"]
    if pytest.main(opts, plugins=[Context()]):
        raise RuntimeError("baseline pytest failed")
    imports, data, listings = defaultdict(set), defaultdict(set), defaultdict(set)
    observations = {"import": imports, "read": data, "directory": listings}
    for line in log.read_text().splitlines():
        scope, mode, path = line.split("\t")
        if scope:
            observations[mode][path].add(scope)
    by_scope = defaultdict(set)
    for test in tests:
        for scope in (test["file"], test["id"].rsplit("::", 1)[0], test["id"]):
            by_scope[scope].add(test["number"])
    directories = {path: sorted(set().union(*(by_scope[key] for key in keys)))
                   for path, keys in listings.items()}
    (MAPS / "directory_reads.json").write_text(json.dumps(directories, sort_keys=True) + "\n")
    excluded = {}
    for path, numbers in reads.items():
        if not path.endswith(".py"):
            continue
        removed = []
        for n in numbers:
            test = tests[n]["id"]
            scopes = {test, test.rsplit("::", 1)[0], tests[n]["file"]}
            if imports[path] & scopes and not data[path] & scopes:
                removed.append(n)
        if removed:
            excluded[path] = removed
    (MAPS / "import_only_reads.json").write_text(json.dumps(excluded, sort_keys=True) + "\n")
    print(f"confirmed loader-only pairs: {sum(map(len, excluded.values()))}", file=sys.stderr)


if __name__ == "__main__":
    main()
