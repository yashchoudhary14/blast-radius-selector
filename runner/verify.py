"""Pinned-label comparison and repeatable CPU measurements (Linux container)."""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from . import Session


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/data"))
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--commit", nargs="*")
    parser.add_argument("--subset-only", action="store_true")
    parser.add_argument("--observe", action="store_true", help="record a full suite before labels exist")
    args = parser.parse_args()
    tests = json.loads((args.data / "tests.json").read_bytes())["tests"]
    labels = {} if args.observe else json.loads((args.data / "answers.json").read_bytes())["commits"]
    commits = sorted((args.data / "commits").glob("*.patch"))
    index, shards = map(int, args.shard.split("/"))
    commits = [p for p in commits[index::shards] if not args.commit or p.stem in args.commit]
    subset = sorted(random.Random(42).sample(tests, 50), key=lambda t: t["number"])
    good = True
    with Session() as session:
        for p in commits:
            # read_text() would erase literal CRs from git's hunk payload.
            patch = p.read_bytes().decode("utf-8", "surrogateescape")
            groups = [tests] if args.observe else [subset] if args.subset_only else [subset, tests]
            for group in groups:
                ids = [t["id"] for t in group]
                lookup = {t["id"]: t["number"] for t in group}
                expected = set(labels.get(p.stem, {}).get("failing", [])).intersection(lookup.values())
                started = time.monotonic()
                parent = time.process_time()
                result = session.run(patch, ids, per_test_budget={t["id"]: t["seconds"] for t in group})
                found = {lookup[i] for i in result.failing}
                row = dict(commit=p.stem, selected=len(group), ran=len(result.ran),
                           expected=len(expected), actual=len(found), missing=sorted(expected - found),
                           extra=sorted(found - expected), notes=result.notes, cpu=result.cpu_s,
                           parent_cpu=time.process_time() - parent, wall=time.monotonic() - started,
                           recorded=sum(t["seconds"] for t in group))
                if args.observe:
                    for key in ("expected", "missing", "extra"):
                        del row[key]
                    row.update(failing=sorted(found), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                print(json.dumps(row), flush=True)
                complete = (len(result.ran) == len(ids) and set(result.ran) == set(ids)
                            and "incomplete child run" not in result.notes)
                good &= complete and (args.observe or expected == found)
    return int(not good)


if __name__ == "__main__":
    raise SystemExit(main())
