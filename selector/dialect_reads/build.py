"""Record pristine dispatch keys for expensive methods, including worker processes."""

import argparse
import gzip
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import SITES


def collect(base, identity, expected, runner):
    file, *parts = identity.split("::")
    node = ".".join((file.removesuffix(".py").replace("/", "."), *parts))
    here = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as directory:
        env = dict(os.environ, BR_DIALECT_BASE=str(base), BR_DIALECT_READS=directory,
                   PYTHONPATH=os.pathsep.join(map(str, (here, here.parents[1], base))),
                   SKIP_INTEGRATION="0", OPENBLAS_NUM_THREADS="1", PYTHONHASHSEED="0")
        cmd = [sys.executable, "-B", runner, node] if runner else [sys.executable, "-B", "-m", "unittest", node]
        process = subprocess.run(cmd, cwd=base, env=env, capture_output=True, timeout=180)
        ran = re.search(rb"Ran (\d+) tests? in ", process.stderr)
        if process.returncode or b"skipped=" in process.stderr or not ran or int(ran[1]) < expected:
            print(f"{node}: incomplete\n{process.stderr.decode(errors='replace')[-1500:]}", file=sys.stderr)
            return identity, None
        paths = list(Path(directory).glob("*.json"))
        if not paths or any(not p.stat().st_size for p in paths):
            return identity, None
        keys = defaultdict(set)
        unknown = set()
        for path in paths:
            row = json.loads(path.read_bytes())
            unknown.update(row["unknown"])
            for field, values in row["keys"].items():
                keys[field].update(values)
        fields = {field for sites in SITES.values() for spec in sites.values() for field in spec}
        record = {field: sorted(keys[field]) for field in sorted(fields - unknown)}
        print(f"{node}: {len(paths)} interpreters, {sum(map(len, record.values()))} keys", file=sys.stderr, flush=True)
        return identity, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--runner")
    parser.add_argument("--minimum", type=float, default=1.0)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--all-files", action="store_true")
    args = parser.parse_args()
    tests = json.loads(args.manifest.read_bytes())["tests"]
    methods = sorted((test for test in tests if test["seconds"] >= args.minimum), key=lambda test: -test["seconds"])
    jobs = [(test["id"], 1) for test in methods]
    if args.all_files:
        files = defaultdict(int)
        for test in tests:
            files[test["file"]] += 1
        jobs.extend(sorted(files.items()))
    with ThreadPoolExecutor(args.jobs) as pool:
        records = dict(pool.map(lambda job: collect(args.base.resolve(), *job, args.runner), jobs))
    raw = json.dumps({node: keys for node, keys in records.items() if keys is not None},
                     sort_keys=True, separators=(",", ":")).encode()
    args.output.write_bytes(gzip.compress(raw, mtime=0))


if __name__ == "__main__":
    main()
