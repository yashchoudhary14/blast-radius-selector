"""Compare each possible generator-memory decision with the labeled catalog, offline."""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from runner import GENERATOR_MEMORY, Session
from selector.analyze import Analyzer
from selector.dialect_cases import DialectCases


def init(manifest):
    global analyzer, cases, number
    analyzer = Analyzer(manifest)
    cases = DialectCases(analyzer.maps)
    number = analyzer.num[GENERATOR_MEMORY]


def probe(path):
    patch = path.read_bytes().decode("utf-8", "surrogateescape")
    candidate = analyzer.analyze(patch)
    initial = bool(candidate.mask & (1 << number))
    potential = candidate.mask and not candidate.broken and any(
        analyzer.importers[p] & (1 << number) for p in candidate.python_paths
    )
    if not (initial or potential):
        return {"mode": "outside"}
    names = cases.subset(candidate)
    with Session() as session:
        result = session.run(patch, [GENERATOR_MEMORY], stop_on_first_fail=True,
                             case_filter=None if names is None else {GENERATOR_MEMORY: names},
                             per_test_budget={GENERATOR_MEMORY: 7})
    if result.ran != [GENERATOR_MEMORY] or "incomplete child run" in result.notes:
        raise RuntimeError((path, result))
    return {"mode": "all" if names is None else "subset", "names": names,
            "initial": initial, "failed": GENERATOR_MEMORY in result.failing,
            "cpu_s": result.cpu_s, "notes": result.notes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", type=Path, nargs="+")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    paths, labels = {}, defaultdict(list)
    for data in args.datasets:
        tests = json.loads((data / "tests.json").read_bytes())["tests"]
        num = next(t["number"] for t in tests if t["id"] == GENERATOR_MEMORY)
        answers = json.loads((data / "answers.json").read_bytes())["commits"]
        for commit, answer in answers.items():
            path = data / "commits" / (commit + ".patch")
            digest = hashlib.sha256(path.read_bytes()).digest()
            paths[digest] = path
            labels[digest].append((data.name, commit, num in answer["failing"]))
    mismatches = []
    with ProcessPoolExecutor(args.jobs, initializer=init,
                             initargs=(args.datasets[0] / "tests.json",)) as pool:
        for digest, result in zip(paths, pool.map(probe, paths.values()), strict=True):
            for dataset, commit, label in labels[digest]:
                print(json.dumps({"set": dataset, "commit": commit, "label": label, **result}), flush=True)
                if result.get("failed", False) != label:
                    mismatches.append(f"{dataset}/{commit}")
    if mismatches:
        print("Label disagreements:", *mismatches, file=sys.stderr)
    return int(bool(mismatches))


if __name__ == "__main__":
    raise SystemExit(main())
